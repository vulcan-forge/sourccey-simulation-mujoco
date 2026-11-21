"""Check a fresh clone's inputs and optionally save four headless camera images."""
import argparse
import json
import shutil
from pathlib import Path


def check(render=False, output=Path('artifacts/linux_preflight')):
    from scripts.linux_handoff import ROOT, restore_inputs
    restore_inputs()
    import mujoco
    import numpy as np
    from examples.box_pickup import load_scene, scene_model
    from examples.box_training import CAMERAS
    from sourccey.rendering import create_renderer, provenance

    if mujoco.__version__ != '4.0.1':
        raise ValueError(f'Require MuJoCo 4.0.1, got {mujoco.__version__}')
    model, _ = scene_model(load_scene(ROOT / 'examples/box_pickup_a.json'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = dict(passed=True, platform=__import__('sys').platform,
                  rendering=provenance(), model_nq=model.nq, cameras=list(CAMERAS),
                  ffmpeg=shutil.which('ffmpeg'), rendered=False)
    if render:
        from PIL import Image
        if not result['ffmpeg']:
            raise RuntimeError('Install FFmpeg before validating camera generation')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with create_renderer(model, height=240, width=320) as renderer:
            for name in CAMERAS:
                renderer.update_scene(data, camera=name)
                pixels = renderer.render()
                if pixels.shape != (240, 320, 3) or np.ptp(pixels) == 0:
                    raise ValueError(f'Invalid or blank camera output: {name}')
                Image.fromarray(pixels).save(output / f'{name}.png')
        result['rendered'] = True
        (output / 'preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/linux_preflight'))
    args = parser.parse_args()
    check(args.render, args.output)


if __name__ == '__main__':
    main()
