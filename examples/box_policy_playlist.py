"""Shuffle and replay saved learned-policy attempts; never load an AI model."""
import argparse
import json
from pathlib import Path
import queue
import random
import time

import mujoco
import numpy as np

from examples.box_dataset_pilot import apply_camera_perturbations
from examples.box_pickup import load_recording, scene_model
from sourccey.build import ROOT


DEFAULT = ROOT/"artifacts/box_policy_sweep_validation_20k_h30"


def attempts(root, seed=None, count=None):
    """Read sweep results, not arbitrary demo/teacher recording directories."""
    root = Path(root).resolve()
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if summary.get("format") != "sourccey-xvla-sweep-v1":
        raise ValueError("Require a saved X-VLA evaluation sweep")
    paths = []
    for result in summary["results"]:
        path = (root/result["episode"]).resolve()
        if path.parent != root or path in paths:
            raise ValueError("Invalid or duplicate episode directory")
        for name in ("scene.json", "recording.npz", "report.json"):
            if not (path/name).is_file():
                raise ValueError(f"Missing saved replay file: {path/name}")
        paths.append(path)
    if not paths:
        raise ValueError("Sweep has no completed recordings")
    if count is not None and count < 1:
        raise ValueError("Count must be positive")
    random.Random(seed).shuffle(paths)
    return paths[:count]


def play_attempt(output, number, total, speed=1., hold=2., deadline=None, *, scripted=False):
    from mujoco.studio import launch_passive, messages, viewer_protocol
    from mujoco.studio.dear_imgui import dear_imgui as imgui
    from sourccey.studio_review import NativePresentation, RecordedExperiment, ExperimentPanel

    config = json.loads((output/"scene.json").read_text(encoding="utf-8"))
    recording, report = load_recording(output, config)
    if not scripted and report.get("format") != "sourccey-xvla-evaluation-v1":
        raise ValueError("Playlist requires actual learned-policy recordings")
    if scripted and (not config.get('random_collection') or not config.get('placement')):
        raise ValueError('Scripted playlist requires a recorded random pick-and-place scene')
    model, _ = scene_model(config, display_only=True)
    if config.get("domain_randomization", {}).get("camera_perturbations"):
        apply_camera_perturbations(model, config)
    data = mujoco.MjData(model)
    data.qpos[:] = recording["qpos"][0]
    mujoco.mj_forward(model, data)
    navigation = queue.Queue()

    class PlaylistPanel(ExperimentPanel):
        @messages.handler
        def build_gui(self, event: messages.BuildGuiEvent):
            super().build_gui(event)
            imgui.SetNextWindowPos(imgui.Vec2(12,300), imgui.Cond.FirstUseEver)
            imgui.SetNextWindowSize(imgui.Vec2(530,225), imgui.Cond.FirstUseEver)
            visible, _ = imgui.Begin('Random attempt playlist')
            if visible:
                imgui.TextUnformatted(f'Attempt {number}/{total}: {output.name}')
                pose = config['box_pose']
                imgui.TextUnformatted(f"Yaw {pose['yaw_deg']:+.1f} deg | near edge "
                    f"{pose['center_from_near_edge_mm']:.1f} mm | left {pose['center_left_of_robot_mm']:.1f} mm")
                if scripted:
                    imgui.TextUnformatted(f"Scripted pick and place: {'PASS' if report['passed'] else 'FAIL'}")
                    imgui.TextUnformatted(f"Hold clearance: {report['minimum_bottom_clearance_during_hold_mm']:.1f} mm")
                else:
                    imgui.TextUnformatted(f"Pickup: {'PASS' if report['passed'] else 'FAIL'} | "
                        f"maximum lift {report['maximum_bottom_clearance_mm']:.1f} mm")
                failed = ', '.join(k for k, passed in report['checks'].items() if not passed)
                imgui.TextUnformatted('Failed checks: '+(failed or 'none'))
                if imgui.Button('Previous attempt'):
                    navigation.put(-1)
                imgui.SameLine()
                if imgui.Button('Next attempt'):
                    navigation.put(1)
                imgui.SameLine()
                if imgui.Button('Quit playlist'):
                    navigation.put(0)
                imgui.TextUnformatted(f'Automatically advances at the end after {hold:g}s. Pause stops advancement.')
                imgui.TextUnformatted('Use Pause, Restart, speed and Time above to inspect the grasp.')
            imgui.End()

    playback = RecordedExperiment(model, data, recording, report, speed=speed)
    panel = PlaylistPanel()
    panel.paused, panel.speed = False, speed
    source = 'Scripted pick-and-place' if scripted else 'Random X-VLA'
    viewer = viewer_protocol.ViewerConfig(title=f"Sourccey | {source} replay {number}/{total} | {output.name}",
        width=1440, height=960, gfx="opengl", open_browser=False)
    finished_since = None
    with launch_passive.launch(viewer, plugins=[playback, panel, NativePresentation()]) as handle:
        handle.set_model(model, data=data)
        while handle.is_running():
            handle.sync(model, data)
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                return 0
            try:
                return navigation.get_nowait()
            except queue.Empty:
                pass
            if not playback.paused and playback.position >= float(recording['time'][-1]):
                if finished_since is None:
                    finished_since = now
                if now-finished_since >= hold:
                    return 1
            else:
                finished_since = None
    # Closing the window exits the whole playlist, rather than opening another.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, default=DEFAULT)
    parser.add_argument('--seed', type=int, help='Repeatable shuffle; default is a fresh order')
    parser.add_argument('--count', type=int, help='Play only N randomly selected attempts')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--hold', type=float, default=2., help='Seconds to hold the final pose')
    parser.add_argument('--loop', action='store_true', help='Repeat the shuffled playlist after its last attempt')
    parser.add_argument('--list', action='store_true', help='Print the playlist without opening a viewer')
    parser.add_argument('--max-seconds', type=float, help='Auto-close the entire playlist after this wall time')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not .1 <= args.speed <= 3:
        parser.error('Speed must be between 0.1 and 3')
    if not np.isfinite(args.hold) or args.hold < 0:
        parser.error('Hold must be finite and nonnegative')
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('Max seconds must be finite and positive')
    if args.count is not None and args.count < 1:
        parser.error('Count must be positive')
    paths = attempts(args.sweep, args.seed, args.count)
    print('Saved AI replays only; no inference or hardware connection.', flush=True)
    for index, path in enumerate(paths, 1):
        print(f'{index:2}/{len(paths)}: {path.name}', flush=True)
    if args.list:
        return
    deadline = time.perf_counter()+args.max_seconds if args.max_seconds else None
    index = 0
    try:
        while 0 <= index < len(paths):
            direction = play_attempt(paths[index], index+1, len(paths), args.speed, args.hold, deadline)
            if direction == 0:
                break
            index = max(0, index+direction)
            if index == len(paths) and args.loop:
                index = 0
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
