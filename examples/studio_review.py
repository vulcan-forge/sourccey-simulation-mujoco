"""Inspect a saved box experiment in MuJoCo Studio with four camera feeds."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def review(output, *, speed=1., seconds=0., gfx='opengl', hot_reload=False):
    from mujoco.studio import app, launch_passive, viewer_protocol
    from sourccey.simulation import Simulation
    from sourccey.studio_review import RecordedExperiment,ExperimentPanel,NativePresentation
    from examples.box_pickup import load_recording,scene_model
    output = Path(output)
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    recording,report = load_recording(output,report['scene'])
    model,_ = scene_model(report['scene'],display_only=True)
    sim = Simulation(model=model,traction=False)
    player = RecordedExperiment(model,sim.data,recording,report,speed)
    config = viewer_protocol.ViewerConfig(title='Sourccey | MuJoCo 4 Studio',gfx=gfx,
        open_browser=gfx=='web',hot_reload='plugins' if hot_reload else '')
    started = time.perf_counter()
    try:
        with launch_passive.launch(config,plugins=[app.App(),player,ExperimentPanel(),NativePresentation()]) as handle:
            handle.set_model(model,data=sim.data)
            while handle.is_running():
                model,data = handle.sync(model,sim.data)
                if seconds and time.perf_counter()-started >= seconds:
                    break
    finally:
        if player.renderer is not None:
            player.renderer.close()
            player.renderer = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Directory containing report.json and recording.npz')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--seconds',type=float,default=0.,help='Automatically close after this many wall seconds; zero waits for window close')
    parser.add_argument('--gfx',choices=('opengl','opengl_headless','web'),default='opengl')
    parser.add_argument('--hot-reload',action='store_true',help='Reload Studio plugin code when saved')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('Speed must be positive; seconds must be nonnegative and finite')
    review(args.output,speed=args.speed,seconds=args.seconds,gfx=args.gfx,hot_reload=args.hot_reload)


if __name__ == '__main__':
    main()
