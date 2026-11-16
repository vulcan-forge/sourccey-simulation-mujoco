"""Host-loop adapter and reliable RPC endpoint for buffered arm trajectories.

All motor reads/writes stay on the existing Host thread. No device is opened
here and no upload, capability query, or status query issues a motor command.
"""
import time
import logging
from collections import deque

import zmq

from .buffered_trajectory import (
    PROTOCOL, PORT, MAX_BYTES, LEASE_S, KEYS, Executor, Trajectory, checksum,
)


class BufferedHost:
    def __init__(self, robot, context, legacy_commands, *, endpoint=None, control_hz=30., trace=None):
        self.robot, self.legacy_commands = robot, legacy_commands
        self.period = 1 / control_hz
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES + 4096)
        self.socket.bind(endpoint or f'tcp://*:{PORT}')
        self._trace = trace or (lambda *a, **k: None)
        self.job = self.owner = self.digest = self.executor = None
        self.uploaded_at = None
        self.retired = deque(maxlen=256)

    @property
    def active(self):
        return self.executor is not None and self.executor.state == 'running'

    def trace(self, event, **fields):
        try:
            self._trace(event, **fields)
        except Exception:
            logging.exception('Buffered trajectory diagnostic logging failed')

    def stop_base(self):
        try:
            self.robot.stop_base()
        except Exception as exc:
            # Do not turn a stop/reporting failure into a Host shutdown that
            # releases the arms. No further trajectory goals are issued.
            self.trace('trajectory_base_stop_error', error=repr(exc))

    def limits(self):
        result = {}
        for side in ('left', 'right'):
            arm = getattr(self.robot, side + '_arm')
            if not arm.is_connected or arm.config.use_degrees:
                raise ValueError('Buffered replay requires both connected arms in normalized mode')
            if not arm.calibrator.protect_shoulder_encoder_seam:
                raise ValueError('Buffered replay requires the existing shoulder seam protection')
            calibration = arm.safety._shoulder_calibration()
            bounds = arm.safety._shoulder_safe_raw_range(calibration)
            result[f'{side}_shoulder_lift.pos'] = sorted(
                arm.safety._raw_to_shoulder_position(v, calibration) for v in bounds)
        return result

    def read_motion(self):
        """Fresh hardware only: no cameras, JPEG encoding, or cached arm/Z fallback."""
        values = {}
        for side in ('left', 'right'):
            arm = getattr(self.robot, side + '_arm')
            if not arm.is_connected:
                raise RuntimeError(f'{side} arm disconnected')
            observation = arm.get_observation()
            values.update({f'{side}_{key}': value for key, value in observation.items()})
        wheel = self.robot.dc_motors_controller.get_velocities()
        base = self.robot._wheel_normalized_to_body(wheel)
        elevator = self.robot._read_z_sensor()
        observed = dict(positions={key: values[key] for key in KEYS}, elevator=elevator,
                        base_velocity=[base[k] for k in ('x.vel', 'y.vel', 'theta.vel')] + [0.])
        flat = {**observed['positions'], **base, 'z.pos': elevator}
        return observed, flat

    def send(self, target):
        # _send_action propagates overall errors; arm adapters may report an
        # individual write failure as an incomplete result, which we reject.
        action = {**target, 'x.vel': 0., 'y.vel': 0., 'theta.vel': 0., 'z.vel': 0.}
        sent = self.robot._send_action(action)
        if not all(key in sent for key in KEYS):
            raise RuntimeError('Host did not confirm writes for all twelve joints')
        if any(abs(float(sent[k])-target[k]) > .1 for k in KEYS):
            raise RuntimeError('Host altered a buffered target; stopping at the last waypoint')
        self.robot.update()

    def cancel(self, reason, state='cancelled'):
        if self.executor is not None:
            self.executor.abort(reason, state, now=time.monotonic())
        self.stop_base()
        self.trace('trajectory_stop', job_id=self.job, reason=str(reason), state=state)

    def status(self, now):
        result = dict(protocol=PROTOCOL, job_id=self.job, checksum=self.digest,
                      point_count=0 if self.executor is None else len(self.executor.path.times))
        return {**result, **(self.executor.status(now) if self.executor else {'state': 'idle'})}

    def dispatch(self, request, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(request, dict) or request.get('version') != PROTOCOL:
            raise ValueError('Buffered RPC protocol mismatch')
        op = request.get('op')
        if self.active and now-self.executor.heartbeat >= LEASE_S:
            self.cancel('Client lease expired; last target held', 'aborted')
        if op == 'capabilities':
            return dict(execution='onboard-buffered', max_upload_bytes=MAX_BYTES,
                        control_hz=1/self.period, lease_s=LEASE_S, host_limits=self.limits(), **self.status(now))
        if op == 'upload':
            job, owner = request['job_id'], request['owner']
            if not all(isinstance(v, str) and 16 <= len(v) <= 128 for v in (job, owner)):
                raise ValueError('Invalid job identifier or owner token')
            digest = checksum(request['payload'])
            if digest != request.get('checksum'):
                raise ValueError('Trajectory checksum mismatch')
            if job == self.job:
                if digest != self.digest or owner != self.owner:
                    raise ValueError('Job identifier already belongs to another upload')
                return self.status(now)  # Lost upload acknowledgements are safe to retry.
            if self.active or job in self.retired:
                raise ValueError('Host is busy or job identifier is retired')
            path = Trajectory(request['payload'], self.limits())  # Validate atomically.
            if self.job:
                self.retired.append(self.job)
            self.job, self.owner, self.digest = job, owner, digest
            self.executor, self.uploaded_at = Executor(path, self.period), now
            self.trace('trajectory_uploaded', job_id=job, checksum=digest, points=len(path.times))
            return self.status(now)
        if op not in ('start', 'status', 'cancel'):
            raise ValueError('Unknown buffered operation')
        if request.get('job_id') != self.job or request.get('owner') != self.owner or self.executor is None:
            raise ValueError('Job identifier/owner does not match the loaded trajectory')
        if op == 'cancel':
            self.cancel('Cancelled by client')
        elif op == 'start' and self.executor.state == 'uploaded':
            if now-self.uploaded_at > 30:
                self.executor.abort('Uploaded trajectory expired before start')
                raise ValueError(self.executor.error)
            try:
                observed, _ = self.read_motion()
                # The final legacy go-to-start command can still be queued at
                # handoff. Consume it BEFORE start; later direct commands preempt.
                try:
                    self.legacy_commands.recv(zmq.NOBLOCK)
                except zmq.Again:
                    pass
                self.executor.start(now, observed)
                self.trace('trajectory_started', job_id=self.job)
            except Exception as exc:
                self.executor.abort(exc)
                raise
        if self.active:
            self.executor.heartbeat = now
        return self.status(now)

    def poll(self):
        if not self.socket.poll(0):
            return
        try:
            request = self.socket.recv_json()
            result = dict(ok=True, **self.dispatch(request))
        except Exception as exc:
            result = dict(ok=False, error=str(exc), protocol=PROTOCOL)
        try:
            self.socket.send_json(result, flags=zmq.NOBLOCK)
        except zmq.ZMQError as exc:
            self.cancel(f'Buffered reply failed: {exc}', 'aborted')

    def step(self):
        """One local control iteration; return a camera-free legacy observation."""
        if not self.active:
            return {}
        before = self.executor.state
        try:
            observed, flat = self.read_motion()
            self.executor.tick(time.monotonic(), observed, self.send)
        except Exception as exc:
            self.executor.abort(exc, now=time.monotonic())
            flat = {}
        if self.executor.state != before:
            self.stop_base()
            self.trace('trajectory_finished', **self.status(time.monotonic()))
        return flat

    def close(self):
        if self.executor is not None:
            self.executor.abort('Host shutting down')
        if not self.socket.closed:
            self.socket.close(0)
