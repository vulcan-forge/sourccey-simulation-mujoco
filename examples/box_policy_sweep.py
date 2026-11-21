"""Saved-scene evaluation sweep, using one loaded model and isolated episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from examples.box_policy_eval import PILOT, evaluate, read_json
from sourccey.build import ROOT
from sourccey.xvla_eval import PolicyWorker, select_sweep, sweep_summary


DATASET = ROOT/"artifacts/box_xvla_training_handoff_5090/dataset"


def write_results(output, manifest, results, error):
    summary = dict(format="sourccey-xvla-sweep-v1", **sweep_summary(manifest["scenes"], results, error),
                   split=manifest["split"], checkpoint=manifest["checkpoint"],
                   execution_horizon=manifest["execution_horizon"], policy_seed=manifest["policy_seed"])
    temporary = output/"summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    temporary.replace(output/"summary.json")
    lines = ["# X-VLA saved-scene sweep", "",
        f"Split: **{manifest['split']}**. Completed {summary['completed']}/{summary['planned']}; "
        f"passed {summary['passed']}, failed {summary['failed']}.", "",
        manifest.get('evaluation_scope', 'One policy seed per scene. These scenes share location groups; this is a finite saved-scene sweep.'),
        'Final test scenes used explicitly.' if manifest.get('final_test_used') else 'Final test scenes are excluded.', "",
        "| Scene | Location | Near edge / left (mm) | Yaw | Result | Max lift (mm) |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in results:
        pose = row["box_pose"]
        lines.append(f"| {row['episode']} | {row['location_id']} | "
            f"{pose['center_from_near_edge_mm']:.1f} / {pose['center_left_of_robot_mm']:.1f} | "
            f"{pose['yaw_deg']:.1f}° | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row['maximum_bottom_clearance_mm']:.1f} |")
    if error:
        lines.extend(["", f"Sweep interrupted: {error}"])
    lines.extend(["", "## Replay an attempt", "",
        "Run these from the project directory. Each loads only its saved recording:", ""])
    for row in results:
        path = output/row["episode"]
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            pass
        lines.extend(["```bash", f'./xvla-sim.cmd --replay --output "{path.as_posix()}"', "```", ""])
    (output/"SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def contact_sheet(output, results):
    from PIL import Image, ImageDraw
    width, height, columns = 320, 280, 4
    canvas = Image.new("RGB", (width*columns, height*((len(results)+columns-1)//columns)), "#20252b")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(results):
        x, y = (index % columns)*width, (index//columns)*height
        with Image.open(output/row["episode"]/"final.png") as frame:
            canvas.paste(frame.resize((320, 240)), (x, y))
        label = f"{row['episode_index']}  yaw {row['box_pose']['yaw_deg']:+.1f}  "
        label += f"{'PASS' if row['passed'] else 'FAIL'}  {row['maximum_bottom_clearance_mm']:.1f} mm"
        draw.text((x+5, y+245), label, fill="#8cf0a2" if row["passed"] else "#ffbf80")
    canvas.save(output/"contact_sheet.jpg")


def select_collection(collection, split, limit=None, final_test=False):
    """Freeze teacher-qualified scenes before policy outcomes exist."""
    from examples.box_dataset_pilot import pilot_signature
    if split not in ('train', 'validation', 'test') or (split == 'test' and not final_test):
        raise ValueError('Final test is reserved until explicitly unlocked')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    collection = Path(collection)
    manifest = read_json(collection/'manifest.json')
    if (manifest.get('collection') != 'continuous-pick-place-v1' or manifest['pending']
            or manifest['status'] not in ('physics_complete', 'rendering', 'complete')):
        raise ValueError('Require completed continuous collection physics, not a probe or incomplete quota')
    rows = sorted((r for r in manifest['episodes'] if r['split'] == split and r['eligible_for_imitation']),
                  key=lambda r:r['episode_id'])
    selected = []
    for index, row in enumerate(rows[:limit]):
        name = row['episode_id']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Invalid episode identifier')
        config = read_json(collection/name/'scene.json')
        if (config['episode_id'] != name or config['split'] != split
                or config['workspace']['location_id'] != row['location_id']
                or row['pilot_signature'] != pilot_signature(config)):
            raise ValueError('Scene identity, split or signature mismatch')
        selected.append(dict(episode_index=index, episode=name, split=split,
            location_id=row['location_id'], cell_id=row['cell_id'],
            box_pose=config['box_pose'], scene_seed=config['seed']))
    if not selected:
        raise ValueError('No scenes selected')
    return selected, manifest


def sweep(args):
    collection = getattr(args, 'collection', None)
    dataset = getattr(args, 'dataset', None) or DATASET
    pilot = collection or getattr(args, 'pilot', PILOT)
    if collection:
        selected, source_manifest = select_collection(collection, args.sweep, args.limit,
            getattr(args, 'final_test', False))
        task = source_manifest['instruction']
        scope = ('Independent continuous random poses within shared coverage cells, conditioned on teacher '
                 'pick-and-place acceptance. All selected policy outcomes count; no outcome-based resampling.')
    else:
        selected = select_sweep(read_json(dataset/"simulation_provenance.json"),
                                read_json(dataset/"episode_splits.json"), pilot, args.sweep, args.limit)
        task = read_json(dataset/'simulation_provenance.json')['task']
        scope = 'Finite saved-scene sweep across the recorded location groups.'
    if args.seconds is None:
        args.seconds = 36. if task == 'Pick up the box and place it at the fixed destination' else 24.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(format="sourccey-xvla-sweep-v1", split=args.sweep,
        checkpoint=str(args.checkpoint.resolve()), policy_seed=args.seed,
        execution_horizon=args.horizon, seconds_per_scene=args.seconds, scenes=selected,
        task=task, evaluation_scope=scope,
        selection="frozen source episode order; no outcome-based selection", final_test_used=args.sweep == 'test',
        provenance_sha256=hashlib.sha256((collection/'manifest.json' if collection else dataset/"simulation_provenance.json").read_bytes()).hexdigest(),
        scene_sha256={r["episode"]:hashlib.sha256((pilot/r["episode"]/"scene.json").read_bytes()).hexdigest()
                      for r in selected})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    results, error = [], None
    write_results(output, manifest, results, error)
    started = time.perf_counter()
    print(f"Selected {len(selected)} {args.sweep} scenes; loading checkpoint once...", flush=True)
    try:
        with PolicyWorker(args.policy_python, args.checkpoint, args.lerobot_source, output,
                          args.device, args.seed, task=task) as policy:
            print(f"Loaded on {policy.ready['gpu'] or policy.ready['device']}", flush=True)
            for index, row in enumerate(selected):
                print(f"\nScene {index+1}/{len(selected)}: {row['episode']} | "
                      f"yaw {row['box_pose']['yaw_deg']:.1f} degrees", flush=True)
                trial = argparse.Namespace(**vars(args))
                trial.output = output/row["episode"]
                trial.episode = row["episode"]
                trial.scene = pilot/row["episode"]/"scene.json"
                report = evaluate(trial, shared_policy=policy)
                results.append(dict(**row, passed=report["passed"], checks=report["checks"],
                    maximum_bottom_clearance_mm=report["maximum_bottom_clearance_mm"],
                    successful_hold=report["successful_hold"], latency_s=report["latency_s"],
                    placement=report.get('placement'),
                    error=report["error"]))
                error = report["error"]
                write_results(output, manifest, results, error)
                if error:
                    break
    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    summary = write_results(output, manifest, results, error)
    if results:
        contact_sheet(output, results)
    summary["wall_s"] = time.perf_counter()-started
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"Sweep: {summary['passed']}/{summary['completed']} passed; "
          f"{summary['completed']}/{summary['planned']} completed. Saved: {output}", flush=True)
    if error:
        print(error, flush=True)
    return summary
