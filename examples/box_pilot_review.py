"""Review a successful pilot replay with the exact randomized camera mounts."""
import argparse
import json
from pathlib import Path

from examples.box_dataset_pilot import OUTPUT,apply_camera_perturbations,pilot_signature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=OUTPUT)
    parser.add_argument('--episode',help='Episode ID from the pilot manifest; defaults to first success')
    args=parser.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    episodes=[e for e in manifest['episodes'] if e['eligible_for_imitation']]
    if args.episode:
        episodes=[e for e in episodes if e['episode_id']==args.episode]
    if not episodes:
        parser.error('No matching successful episode')
    episode=episodes[0]
    output=args.source/episode['episode_id']
    scene=json.loads((output/'scene.json').read_text(encoding='utf-8'))
    if episode['pilot_signature']!=pilot_signature(scene):
        parser.error('Pilot physics source changed; review the saved camera videos instead')
    from examples.studio_review import review
    review(output,model_transform=apply_camera_perturbations)


if __name__=='__main__':
    main()
