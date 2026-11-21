"""Watch successful and failed scripted attempts from the latest random batch."""
import argparse
import json
from pathlib import Path
import random
import time

from examples.box_pickup import load_recording
from examples.box_policy_playlist import play_attempt
from examples.box_status import LATEST


def select_attempts(output=None, successes=3, failures=3, seed=None):
    if successes < 0 or failures < 0 or successes+failures < 1:
        raise ValueError('Request at least one replay with nonnegative outcome counts')
    if output is None:
        output = json.loads(LATEST.read_text(encoding='utf-8'))['output']
    output = Path(output).resolve()
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('collection') != 'continuous-pick-place-v1':
        raise ValueError('Require a saved continuous pick-and-place collection')
    groups = {True:[], False:[]}
    seen = set()
    for row in manifest['episodes']:
        path = (output/row['episode_id']).resolve()
        if path.parent != output or path in seen:
            raise ValueError('Invalid or duplicate episode directory')
        seen.add(path)
        if row.get('failure_stage') == 'recording':
            continue
        if all((path/name).is_file() for name in ('scene.json', 'report.json', 'recording.npz')):
            groups[bool(row['passed'])].append(path)
    rng = random.Random(seed)
    selected = []
    for passed, count in ((True, successes), (False, failures)):
        if len(groups[passed]) < count:
            raise ValueError(f'Only {len(groups[passed])} replayable {"successful" if passed else "failed"} attempts available')
        for path in rng.sample(groups[passed], count):
            config = json.loads((path/'scene.json').read_text(encoding='utf-8'))
            recording, report = load_recording(path, config)
            if bool(report['passed']) != passed or len(recording['time']) < 2:
                raise ValueError(f'Outcome mismatch or empty recording: {path.name}')
            selected.append(path)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Collection root; defaults to the latest attempt batch')
    parser.add_argument('--successes', type=int, default=3)
    parser.add_argument('--failures', type=int, default=3)
    parser.add_argument('--seed', type=int, help='Repeat the same random selection')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--hold', type=float, default=2.)
    parser.add_argument('--list', action='store_true', help='Check and list recordings without opening the viewer')
    parser.add_argument('--max-seconds', type=float, help='Optional wall-time limit for the playlist')
    args = parser.parse_args()
    if not .1 <= args.speed <= 3 or not 0 <= args.hold < float('inf'):
        parser.error('Require speed 0.1..3 and finite nonnegative hold')
    if args.max_seconds is not None and not 0 < args.max_seconds < float('inf'):
        parser.error('Require a finite positive time limit')
    try:
        paths = select_attempts(args.output, args.successes, args.failures, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print('Saved scripted pick-and-place recordings; no physics or inference recomputed.', flush=True)
    for index, path in enumerate(paths, 1):
        report = json.loads((path/'report.json').read_text(encoding='utf-8'))
        print(f'{index}/{len(paths)}: {"PASS" if report["passed"] else "FAIL"} | {path.name}', flush=True)
    if args.list:
        return
    deadline = time.perf_counter()+args.max_seconds if args.max_seconds else None
    index = 0
    try:
        while index < len(paths):
            direction = play_attempt(paths[index], index+1, len(paths), args.speed, args.hold, deadline, scripted=True)
            if direction == 0:
                break
            index = max(0, index+direction)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
