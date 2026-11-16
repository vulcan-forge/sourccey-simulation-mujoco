"""Full-path upload and low-rate status/lease client; no streamed joint goals."""
import time
import uuid

import zmq

from .buffered_trajectory import PROTOCOL, PORT, checksum


class BufferedClient:
    def __init__(self, host, port=PORT, *, on_wait=None):
        self.context = zmq.Context()
        self.endpoint = f'tcp://{host}:{port}'
        self.socket = None
        self.on_wait = on_wait

    def reset(self):
        if self.socket is not None:
            self.socket.close(0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.connect(self.endpoint)

    def request(self, op, *, timeout=1.5, retries=1, cancellable=True, **fields):
        for attempt in range(retries+1):
            if self.socket is None:
                self.reset()
            deadline = time.monotonic() + timeout
            try:
                for flags in (zmq.POLLOUT, zmq.POLLIN):
                    while True:
                        if cancellable and self.on_wait:
                            self.on_wait()
                        remaining = deadline-time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f'Buffered Host did not acknowledge {op}')
                        if self.socket.poll(min(100, max(1, int(remaining*1000))), flags):
                            break
                    if flags == zmq.POLLOUT:
                        self.socket.send_json(dict(version=PROTOCOL, op=op, **fields), flags=zmq.NOBLOCK)
                result = self.socket.recv_json(flags=zmq.NOBLOCK)
                if not isinstance(result, dict):
                    raise ValueError('Invalid buffered Host response')
                if result.get('protocol') != PROTOCOL or not result.get('ok'):
                    raise ValueError(result.get('error', 'Buffered Host protocol mismatch'))
                return result
            except (TimeoutError, zmq.ZMQError) as exc:
                self.reset()  # REQ cannot be reused after an unacknowledged request.
                if attempt == retries:
                    raise TimeoutError(f'{exc}. Check that the updated robot Host is running on port {PORT}.') from exc

    def capabilities(self):
        result = self.request('capabilities')
        if result.get('execution') != 'onboard-buffered':
            raise ValueError('Robot Host does not support onboard buffered execution')
        if result['state'] == 'running':
            raise ValueError('Robot is already executing a buffered trajectory')
        return result

    def run(self, payload, log):
        job, owner = uuid.uuid4().hex, uuid.uuid4().hex
        digest = checksum(payload)
        identity = dict(job_id=job, owner=owner)
        complete = False
        try:
            status = self.request('upload', payload=payload, checksum=digest, **identity)
            if status.get('checksum') != digest or status.get('point_count') != len(payload['times']):
                raise ValueError('Robot did not acknowledge the complete trajectory upload')
            log(dict(phase='trajectory-uploaded', job_id=job, point_count=status['point_count'], checksum=digest))
            status = self.request('start', **identity)
            while True:
                cycle = time.monotonic()
                if self.on_wait:
                    self.on_wait()
                if status.get('job_id') != job or status.get('checksum') != digest:
                    raise ValueError('Buffered status belongs to a different trajectory')
                if status['state'] in ('aborted', 'cancelled'):
                    raise RuntimeError(f"Buffered replay stopped: {status.get('error')}")
                if status['state'] not in ('running', 'complete'):
                    raise RuntimeError('Buffered replay did not enter the running state')
                if status.get('tracking') is not None:
                    if status['state'] == 'running' and status['observation_age_s'] > .5:
                        raise TimeoutError('Host joint feedback is stale during buffered replay')
                    log(dict(phase='replay', executor='robot-host', **{
                        k: status[k] for k in ('trajectory_s', 'wall_elapsed_s', 'observed', 'commanded',
                                               'tracking', 'command_step_units', 'path_limited',
                                               'limit_reason', 'waiting_for_tracking')}))
                if status['state'] == 'complete':
                    complete = True
                    return dict(nominal_duration_s=status['nominal_duration_s'],
                                actual_duration_s=status['wall_elapsed_s'], feedback_pauses=0,
                                feedback_pause_s=0., executor='robot-host', job_id=job)
                remaining = .1 - (time.monotonic()-cycle)
                if remaining > 0:
                    time.sleep(remaining)
                status = self.request('status', **identity)
        finally:
            if not complete:
                # Cancellation is idempotent. A lost connection is bounded by
                # the independent five-second lease enforced on the robot.
                try:
                    self.request('cancel', retries=0, timeout=.5, cancellable=False, **identity)
                except Exception:
                    pass

    def close(self):
        if self.socket is not None:
            self.socket.close(0)
        self.context.term()
