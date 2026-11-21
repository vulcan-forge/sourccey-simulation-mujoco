"""Read the latest box batch progress without loading MuJoCo or the model."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT/'artifacts/box_attempts_latest.json'


def process_alive(pid):
    if not pid:
        return False
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def snapshot(output=None, latest=LATEST):
    if output is None:
        pointer = read_json(latest)
        if pointer is None:
            raise FileNotFoundError('No box attempt batch has been started yet.')
        output = Path(pointer['output'])
    output = Path(output).resolve()
    state = read_json(output/'run_state.json', {})
    manifest = read_json(output/'manifest.json', {})
    episodes = manifest.get('episodes', [])
    target = manifest.get('attempt_target', state.get('attempt_target', 0))
    successful = [e for e in episodes if e['eligible_for_imitation']]
    completed_names = {e['episode_id'] for e in episodes}
    active = sum(1 for scene in output.glob('*/scene.json') if scene.parent.name not in completed_names
                 and not (scene.parent/'result.json').exists())
    stage = state.get('stage', 'initializing')
    alive = process_alive(state.get('pid'))
    if stage not in ('complete', 'failed') and not alive:
        stage = 'stopped unexpectedly'
    if not alive or stage != 'physics':
        active = 0
    started = state.get('started_utc')
    now = (datetime.fromisoformat(state['updated_utc']) if stage in ('complete', 'failed')
           and state.get('updated_utc') else datetime.now(timezone.utc))
    elapsed = max(0., (now-datetime.fromisoformat(started)).total_seconds()) if started else 0.
    rendered = sum(bool(e.get('videos')) for e in successful)
    comparison = read_json(output/'comparison.json', {})
    return dict(output=str(output), stage=stage, runner_alive=alive,
        completed=len(episodes), target=target, successful=len(successful),
        rejected=len(episodes)-len(successful), active=active if alive else 0,
        queued=max(0, target-len(episodes)-active), rendered=rendered,
        split_successes={s:sum(e['split'] == s for e in successful) for s in ('train', 'validation', 'test')},
        elapsed_seconds=round(elapsed), dataset=state.get('dataset'), error=state.get('error'),
        experiment=manifest.get('experiment'), comparison=comparison.get('summary'),
        physics_eta_seconds=round(elapsed/len(episodes)*(target-len(episodes)))
            if stage == 'physics' and episodes and len(episodes) < target else None)


def duration(seconds):
    seconds = int(seconds)
    return f'{seconds//60}m {seconds%60:02d}s'


def format_status(status):
    s = status
    lines = [f"Stage: {s['stage']}",
        f"Attempts completed: {s['completed']}/{s['target']}",
        f"Successful: {s['successful']} | Rejected: {s['rejected']}",
        f"In progress: {s['active']} | Queued: {s['queued']}",
        f"Camera recordings ready: {s['rendered']}/{s['successful']} successful episodes",
        f"Elapsed: {duration(s['elapsed_seconds'])}"]
    if s.get('experiment'):
        lines.append(f"Grip friction: {s['experiment']['grip_friction_scale']:g}x baseline (ridged-pad approximation)")
    if s.get('comparison'):
        c = s['comparison']
        lines.append(f"Baseline: {c['baseline_successful']}/{c['attempts']} | Improved: {c['improved']} | Regressed: {c['regressed']}")
    if s['physics_eta_seconds'] is not None:
        lines.append(f"Estimated physics remaining: {duration(s['physics_eta_seconds'])}")
    lines.append(f"Output: {s['output']}")
    if s['dataset']:
        lines.append(f"Dataset: {s['dataset']}")
    if s['error']:
        lines.append(f"Error: {s['error']}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Inspect a specific batch instead of the latest one')
    parser.add_argument('--watch', action='store_true', help='Refresh every five seconds; Ctrl+C exits')
    parser.add_argument('--json', action='store_true', help='Machine-readable status')
    args = parser.parse_args()
    try:
        while True:
            status = snapshot(args.output)
            print(json.dumps(status, indent=2) if args.json else format_status(status), flush=True)
            if not args.watch or status['stage'] in ('complete', 'failed', 'stopped unexpectedly'):
                break
            time.sleep(5)
            print('\n---\n', flush=True)
    except FileNotFoundError as error:
        parser.exit(1, str(error)+'\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
