import queue
import socket
import threading
from types import SimpleNamespace

import pytest

from examples.robot_start_pose import HardwareLink, KEYS

zmq = pytest.importorskip('zmq')
pytest.importorskip('google.protobuf')
from integrations import sourccey_pb2


def packet(position=0.):
    state = sourccey_pb2.SourcceyRobotState()
    for side in ('left', 'right'):
        for role in ('shoulder_pan', 'shoulder_lift', 'elbow_flex',
                     'wrist_flex', 'wrist_roll', 'gripper'):
            setattr(getattr(state, side + '_arm_joints'), role, position)
    state.base_position.z_pos = 12.
    state.base_velocity.SetInParent()
    return state.SerializeToString()


@pytest.fixture
def wire(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('examples.robot_start_pose.time.monotonic', lambda: clock[0])
    events, packets, polls = [], [], []

    class Socket:
        def poll(self, timeout):
            polls.append(timeout)
            end = clock[0] + timeout / 1000
            if packets and packets[0][0] <= end:
                clock[0] = max(clock[0], packets[0][0])
                return True
            clock[0] = end
            return False

        def recv(self, flags):
            assert flags == zmq.NOBLOCK
            return packets.pop(0)[1]

    link = HardwareLink.__new__(HardwareLink)
    link.zmq, link.pb = zmq, sourccey_pb2
    link.observations = Socket()
    link.commands = None
    link.on_feedback_event = events.append
    return SimpleNamespace(link=link, clock=clock, events=events, packets=packets, polls=polls)


@pytest.mark.parametrize('delay', [.8, 2., 4.9])
def test_transient_gap_returns_only_new_state_without_sending_commands(wire, delay):
    wire.packets.extend([(0., packet(1)), (delay, packet(8)), (delay + .03, packet(9))])
    assert wire.link.read()['positions'][KEYS[0]] == 1
    observed = wire.link.read()
    assert observed['positions'][KEYS[0]] == 8
    assert observed['feedback_gap_s'] == pytest.approx(delay)
    assert wire.events[0]['phase'] == 'feedback-wait'
    assert wire.events[0]['first_notice']
    assert wire.events[-1]['phase'] == 'feedback-recovered'
    assert max(wire.polls) <= 100
    assert wire.link.commands is None
    normal = wire.link.read()
    assert normal['positions'][KEYS[0]] == 9
    assert 'feedback_gap_s' not in normal


@pytest.mark.parametrize('initial_packet', [False, True])
def test_sustained_outage_is_bounded_and_never_returns_last_pose(wire, initial_packet):
    if initial_packet:
        wire.packets.append((0., packet(6)))
        wire.link.read()
    with pytest.raises(TimeoutError, match=r'No fresh robot observation for 5 s'):
        wire.link.read()
    assert wire.clock[0] == pytest.approx(5., abs=.001)
    assert wire.events[-1]['phase'] == 'feedback-timeout'
    assert not any(e['phase'] == 'feedback-recovered' for e in wire.events)
    assert wire.link.commands is None


def test_same_joint_positions_in_fresh_packets_are_not_an_outage(wire):
    wire.packets.extend((i * .1, packet(4)) for i in range(30))
    for _ in range(30):
        assert wire.link.read()['positions'][KEYS[0]] == 4
    assert not wire.events


@pytest.mark.parametrize('raw,error', [(b'\xff', 'Invalid robot observation'),
                                      (b'', 'missing arm/base/elevator'),
                                      (packet(200), 'outside normalized hardware range')])
def test_malformed_or_unsafe_packets_are_not_retried_as_network_gaps(wire, raw, error):
    wire.packets.append((0., raw))
    with pytest.raises(ValueError, match=error):
        wire.link.read()
    assert wire.clock[0] == 0
    assert not wire.events


def test_window_close_or_interrupt_cancels_recovery_promptly(wire):
    def cancel(event):
        raise KeyboardInterrupt('window closed')
    wire.link.on_feedback_event = cancel
    with pytest.raises(KeyboardInterrupt, match='window closed'):
        wire.link.read()
    assert .5 <= wire.clock[0] <= .6
    assert wire.link.commands is None


def test_command_socket_requires_a_connected_peer_and_observation_socket_reconnects(monkeypatch):
    options, connections = [], []
    class Socket:
        def __init__(self, kind):
            self.kind = kind
        def setsockopt(self, option, value):
            options.append((self.kind, option, value))
        def connect(self, endpoint):
            connections.append(endpoint)
        def poll(self, timeout, flags):
            assert (self.kind, zmq.IMMEDIATE, 1) in options
            return False
        def close(self, linger):
            pass
    class Context:
        def socket(self, kind):
            return Socket(kind)
        def term(self):
            pass
    monkeypatch.setattr(zmq, 'Context', Context)
    link = HardwareLink('test-host')
    with pytest.raises(TimeoutError, match='not writable'):
        link.send(dict.fromkeys(KEYS, 0.))
    link.close()
    assert (zmq.PULL, zmq.RECONNECT_IVL, 100) in options
    assert (zmq.PULL, zmq.RECONNECT_IVL_MAX, 1000) in options
    assert connections == ['tcp://test-host:5556', 'tcp://test-host:5555']


def test_real_zmq_socket_recovers_when_local_observation_publisher_restarts():
    endpoint_queue, errors = queue.Queue(), []
    received, finished = threading.Event(), threading.Event()

    def publisher():
        context = zmq.Context()
        outgoing = None
        try:
            def create():
                sock = context.socket(zmq.PUSH)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDTIMEO, 4000)
                return sock
            outgoing = create()
            outgoing.bind('tcp://127.0.0.1:*')
            endpoint = outgoing.getsockopt(zmq.LAST_ENDPOINT).decode()
            endpoint_queue.put(endpoint)
            outgoing.send(packet(1))
            if not received.wait(4):
                return
            outgoing.close(0)
            outgoing = None
            if finished.wait(.8):
                return
            outgoing = create()
            outgoing.bind(endpoint)
            outgoing.send(packet(9))
            finished.wait(4)
        except Exception as exc:
            errors.append(exc)
        finally:
            if outgoing is not None:
                outgoing.close(0)
            context.term()

    thread = threading.Thread(target=publisher, daemon=True)
    thread.start()
    link = None
    try:
        endpoint = endpoint_queue.get(timeout=4)
        port = int(endpoint.rsplit(':', 1)[1])
        events = []
        link = HardwareLink('127.0.0.1', observation_port=port, on_feedback_event=events.append)
        assert link.read()['positions'][KEYS[0]] == 1
        received.set()
        restored = link.read()
        assert restored['positions'][KEYS[0]] == 9
        assert restored['feedback_gap_s'] >= .5
        assert events[-1]['phase'] == 'feedback-recovered'
        assert link.commands is None
    finally:
        finished.set()
        if link is not None:
            link.close()
        thread.join(timeout=5)
    assert not errors
    assert not thread.is_alive()


def test_real_command_socket_does_not_queue_for_an_absent_local_host():
    # Reserve a local port with no listener. No real robot endpoints are used.
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        link = HardwareLink('127.0.0.1', command_port=reservation.getsockname()[1])
        try:
            with pytest.raises(TimeoutError, match='not writable'):
                link.send(dict.fromkeys(KEYS, 0.))
        finally:
            link.close()
