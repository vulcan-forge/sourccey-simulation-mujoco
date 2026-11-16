"""Prepare an exact-revision Host patch; no connection, installation or motion."""
import argparse
import hashlib
from pathlib import Path

INSPECTED_SHA256 = '0a12aed6bf1cdf3cc060d6253f874676371c7572188571e6a463382ef59a6083'


def prepare(source):
    if hashlib.sha256(source.encode()).hexdigest() != INSPECTED_SHA256:
        raise ValueError('Host revision changed; inspect it before applying the buffered patch')
    changes = [
        ('from .sourccey import Sourccey\n',
         'from .sourccey import Sourccey\nfrom .buffered_host import BufferedHost\n'),
        ('    def disconnect(self):\n        self.zmq_observation_socket.close()',
         '    def disconnect(self):\n        if hasattr(self, "buffered"):\n            self.buffered.close()\n        self.zmq_observation_socket.close()'),
        ('    host = SourcceyHost(host_config)\n',
         '    host = SourcceyHost(host_config)\n'
         '    try:\n'
         '        host.buffered = BufferedHost(robot, host.zmq_context, host.zmq_cmd_socket,\n'
         '                                     control_hz=host.max_loop_freq_hz, trace=trace_event)\n'
         '    except BaseException:\n'
         '        host.disconnect()\n        robot.disconnect()\n        raise\n'),
        ('            poll_relay(relay)\n',
         '            poll_relay(relay)\n            host.buffered.poll()\n'),
        ('                # Send action to robot\n',
         '                # A direct operator command preempts buffered motion.\n'
         '                if host.buffered.active:\n'
         '                    host.buffered.cancel("Preempted by direct robot command")\n'
         '                # Send action to robot\n'),
        ('                last_cmd_time is not None\n',
         '                last_cmd_time is not None\n                and not host.buffered.active\n'),
        ('            observation = robot.get_observation()\n',
         '            # Camera reads/encoding must not delay local trajectory control.\n'
         '            observation = (host.buffered.step() if host.buffered.active\n'
         '                           else robot.get_observation())\n'),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f'Expected exactly one Host integration anchor: {before!r}')
        source = source.replace(before, after, 1)
    compile(source, 'buffered_sourccey_host.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Prepare a separate candidate; do not overwrite the inspected source')
    patched = prepare(args.source.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched, encoding='utf-8', newline='\n')
    print(f'Prepared buffered Host candidate: {args.output}')


if __name__ == '__main__':
    main()
