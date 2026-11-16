import queue
import threading
import time
from types import SimpleNamespace

import pytest

zmq = pytest.importorskip('zmq')
from integrations.buffered_trajectory import KEYS, ROLES, checksum
from integrations.buffered_host import BufferedHost
from integrations.buffered_client import BufferedClient


def payload():
    return dict(version=1, units='normalized', keys=list(KEYS), times=[0., .5],
                positions=[[0.]*12, [2.]*12], elevator=100., tracking_tolerance=5., stall_timeout=15.)


class Robot:
    def __init__(self):
        self.positions = dict.fromkeys(KEYS, 0.)
        self.sent, self.stops = [], 0
        self.fault = None
        for side in ('left', 'right'):
            safety = SimpleNamespace(_shoulder_calibration=lambda: None,
                                     _shoulder_safe_raw_range=lambda c: (-99., 99.),
                                     _raw_to_shoulder_position=lambda v, c: v)
            arm = SimpleNamespace(is_connected=True, config=SimpleNamespace(use_degrees=False),
                                  calibrator=SimpleNamespace(protect_shoulder_encoder_seam=True), safety=safety,
                                  get_observation=lambda side=side: {
                                      r+'.pos': self.positions[f'{side}_{r}.pos'] for r in ROLES})
            setattr(self, side+'_arm', arm)
        self.dc_motors_controller = SimpleNamespace(get_velocities=lambda: {})

    def _wheel_normalized_to_body(self, wheel):
        return dict.fromkeys(('x.vel', 'y.vel', 'theta.vel'), 0.)

    def _read_z_sensor(self):
        if self.fault == 'sensor':
            raise RuntimeError('ADC failed')
        return 100.

    def _send_action(self, action):
        if self.fault == 'write':
            return {}
        self.sent.append(action.copy())
        self.positions = {k: action[k] for k in KEYS}
        return action.copy()

    def update(self):
        pass

    def stop_base(self):
        self.stops += 1


@pytest.fixture
def service():
    context, robot = zmq.Context(), Robot()
    legacy = context.socket(zmq.PULL)
    legacy.setsockopt(zmq.LINGER, 0)
    server = BufferedHost(robot, context, legacy, endpoint='inproc://buffered-test')
    yield server, robot
    server.close()
    legacy.close(0)
    context.term()


def request(op, **kwargs):
    return dict(version=1, op=op, job_id='j'*32, owner='o'*32, **kwargs)


def upload(server):
    p = payload()
    return server.dispatch(request('upload', payload=p, checksum=checksum(p)), now=0.)


def test_upload_and_capabilities_never_send_a_motor_command(service):
    server, robot = service
    assert server.dispatch(request('capabilities'), now=0.)['state'] == 'idle'
    result = upload(server)
    assert result['state'] == 'uploaded' and result['point_count'] == 2
    assert not robot.sent
    assert not server.active


def test_repeated_upload_and_start_are_idempotent(service):
    server, robot = service
    original = upload(server)
    server.dispatch(request('start'), now=0.)
    ex = server.executor
    ex.tick(.1, server.read_motion()[0], server.send)
    progress = ex.elapsed
    assert upload(server)['checksum'] == original['checksum']
    server.dispatch(request('start'), now=.2)
    assert server.executor is ex and ex.elapsed == progress
    server.dispatch(request('cancel'), now=.3)
    count = len(robot.sent)
    server.dispatch(request('start'), now=.4)
    ex.tick(.4, server.read_motion()[0], server.send)
    assert len(robot.sent) == count
    assert ex.state == 'cancelled'


def test_mismatched_owners_and_corrupt_uploads_cannot_change_active_job(service):
    server, robot = service
    upload(server)
    ex = server.executor
    server.dispatch(request('start'), now=0.)
    bad = request('cancel'); bad['owner'] = 'bad'*12
    with pytest.raises(ValueError, match='owner'):
        server.dispatch(bad, now=.1)
    p = payload(); p['positions'][1][0] = 101.
    bad = request('upload', payload=p, checksum=checksum(p)); bad['job_id'] = 'other'*8
    with pytest.raises(ValueError):
        server.dispatch(bad, now=.1)
    assert server.executor is ex and ex.state == 'running'
    assert not robot.sent


def test_invalid_upload_does_not_replace_previously_valid_buffer(service):
    server, robot = service
    upload(server)
    original = server.executor
    p = payload(); p['positions'][1][1] = 100.  # Inside motor range, outside Host seam guard.
    bad = request('upload', payload=p, checksum=checksum(p)); bad['job_id'] = 'different'*4
    with pytest.raises(ValueError, match='Host guard'):
        server.dispatch(bad, now=.1)
    assert server.executor is original
    assert not robot.sent
    with pytest.raises(ValueError, match='checksum'):
        server.dispatch(request('upload', payload=payload(), checksum='bad'), now=.1)


def test_start_checks_real_pose_and_upload_expiry_before_motion(service):
    server, robot = service
    upload(server)
    robot.positions[KEYS[0]] = 20.
    with pytest.raises(ValueError, match='Starting pose'):
        server.dispatch(request('start'), now=1.)
    assert not robot.sent and server.executor.state == 'aborted'
    p = request('upload', payload=payload(), checksum=checksum(payload())); p['job_id'] = 'new'*12
    server.dispatch(p, now=2.)
    with pytest.raises(ValueError, match='expired'):
        server.dispatch({**request('start'), 'job_id': p['job_id']}, now=40.)
    assert not robot.sent


@pytest.mark.parametrize('fault', ['sensor', 'write'])
def test_local_hardware_failure_aborts_and_cannot_resume(service, fault, monkeypatch):
    server, robot = service
    upload(server)
    server.dispatch(request('start'), now=0.)
    monkeypatch.setattr('integrations.buffered_host.time.monotonic', lambda: .03)
    robot.fault = fault
    server.step()
    assert server.executor.state == 'aborted'
    assert not robot.sent
    assert robot.stops


def test_expired_pc_lease_cannot_be_revived_by_a_late_status_packet(service):
    server, robot = service
    upload(server)
    server.dispatch(request('start'), now=0.)
    result = server.dispatch(request('status'), now=5.01)
    assert result['state'] == 'aborted' and 'lease expired' in result['error']
    server.executor.tick(5.02, server.read_motion()[0], server.send)
    assert not robot.sent


@pytest.fixture
def local_host():
    endpoint, failures, stop = queue.Queue(), [], threading.Event()
    shared = {}
    def loop():
        context = zmq.Context()
        legacy = context.socket(zmq.PULL)
        legacy.setsockopt(zmq.LINGER, 0)
        server = None
        try:
            robot = Robot()
            server = BufferedHost(robot, context, legacy, endpoint='tcp://127.0.0.1:*')
            shared.update(server=server, robot=robot)
            endpoint.put(int(server.socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(':', 1)[1]))
            while not stop.is_set():
                started = time.monotonic()
                server.poll()
                server.step()
                stop.wait(max(0., 1/30-(time.monotonic()-started)))
        except Exception as exc:
            failures.append(exc)
        finally:
            if server: server.close()
            legacy.close(0)
            context.term()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    client = BufferedClient('127.0.0.1', port=endpoint.get(timeout=3))
    try:
        yield client, shared
    finally:
        client.close()
        stop.set(); thread.join(timeout=4)
    assert not failures
    assert not thread.is_alive()


def test_real_rpc_uploads_whole_path_once_and_host_executes_between_status_reads(local_host):
    client, shared = local_host
    assert client.capabilities()['execution'] == 'onboard-buffered'
    calls, original = [], client.request
    def record(op, **fields):
        calls.append(op)
        return original(op, **fields)
    client.request = record
    trace = []
    timing = client.run(payload(), trace.append)
    assert timing['executor'] == 'robot-host'
    assert calls.count('upload') == calls.count('start') == 1
    assert set(calls) == {'upload', 'start', 'status'}
    assert len(shared['robot'].sent) > calls.count('status') * 2
    assert shared['server'].executor.state == 'complete'
    assert all(abs(r[k]) <= 1e-10 for r in shared['robot'].sent
               for k in ('x.vel', 'y.vel', 'theta.vel', 'z.vel'))


def test_window_close_cancels_real_rpc_job_without_releasing_torque(local_host):
    client, shared = local_host
    def close_window():
        ex = shared['server'].executor
        if ex is not None and ex.elapsed > .05:
            raise KeyboardInterrupt('window closed')
    client.on_wait = close_window
    with pytest.raises(KeyboardInterrupt, match='window closed'):
        client.run(payload(), lambda r: None)
    assert shared['server'].executor.state == 'cancelled'
    count = len(shared['robot'].sent)
    time.sleep(.1)
    assert len(shared['robot'].sent) == count
