"""Drive Sourccey and manipulate physical boxes in the four-table workshop."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import mujoco

from sourccey.sandbox import Workshop, build_scene, copy_display_state, overview_camera, visible_options


def launch(workshop, *, seconds=0):
    from mujoco.studio import launch_passive, viewer_protocol
    from sourccey.sandbox_viewer import WorkshopApp, WorkshopPhysics, WorkshopPanel
    from sourccey.studio_review import NativePresentation
    display_model, _ = build_scene(display_only=True)
    display_data = mujoco.MjData(display_model)
    copy_display_state(workshop.sim.data, display_model, display_data)
    physics = WorkshopPhysics(workshop, display_model, display_data)
    config = viewer_protocol.ViewerConfig(title="Sourccey | The Box Workshop",
        width=1440, height=960, gfx="opengl", open_browser=False)
    started = time.monotonic()
    try:
        with launch_passive.launch(config, plugins=[WorkshopApp(), physics, WorkshopPanel(), NativePresentation()]) as handle:
            handle.set_model(display_model, data=display_data)
            while handle.is_running():
                handle.sync(display_model, display_data)
                if seconds and time.monotonic()-started >= seconds:
                    break
    finally:
        workshop.sim.set_base()
        physics.close_renderer()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0, help="Auto-close after wall seconds; zero stays open")
    parser.add_argument("--headless", action="store_true", help="Advance physics without a desktop window")
    parser.add_argument("--render", type=Path, help="Save a workshop preview without opening/capturing the desktop")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    workshop = Workshop()
    if args.headless or args.render:
        for _ in range(round((args.seconds or .5)/.02)):
            workshop.advance(.02)
        print(json.dumps(dict(scene=workshop.scene, simulation=workshop.sim.state()), indent=2))
        if args.render:
            from PIL import Image
            from sourccey.rendering import create_renderer
            args.render.parent.mkdir(parents=True, exist_ok=True)
            display_model, _ = build_scene(display_only=True)
            display_data = mujoco.MjData(display_model)
            copy_display_state(workshop.sim.data, display_model, display_data)
            with create_renderer(display_model, width=1600, height=1000, backend="filament") as renderer:
                renderer.update_scene(display_data, camera=overview_camera(), scene_option=visible_options())
                Image.fromarray(renderer.render()).save(args.render)
            print(f"Preview saved to {args.render}")
    else:
        print("Sourccey workshop: WASD/QE drive | Tab switches arms | H keyboard guide | Esc exits", flush=True)
        launch(workshop, seconds=args.seconds)


if __name__ == "__main__":
    main()
