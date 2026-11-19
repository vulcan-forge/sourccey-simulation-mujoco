"""Collect successful randomized stage-C demonstrations, without hardware or uploads."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from examples.box_pickup import load_recording, physics_signature, save_run, scene_model, render
from examples.box_training import (CAMERAS, FPS, STATE_NAMES, EpisodeRecorder, instances, write_videos)
from examples.box_workspace import WorkspacePickup, assess, write_json
from sourccey.build import ROOT
from sourccey.rendering import provenance
from sourccey.simulation import Simulation

TASK = 'Pick up the box'
DEFAULT_WORKSPACE = ROOT/'artifacts/box_workspace_c_validated'
OUTPUT = ROOT/'artifacts/box_xvla_pilot'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pilot_signature(config):
    return hashlib.sha256((physics_signature(config)+file_hash(__file__)
        +file_hash(ROOT/'artifacts/robot_joint_mapping.json')).encode()).hexdigest()


def read_anchors(workspace):
    """Use only signed, successful scenes from the completed physics map."""
    workspace = Path(workspace)
    manifest = json.loads((workspace/'manifest.json').read_text(encoding='utf-8'))
    catalog = json.loads((workspace/'validated_scenes.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or not catalog['simulation_only']:
        raise ValueError('A completed simulation workspace is required')
    groups = {}
    for item in catalog['scenes']:
        config = json.loads((workspace/item['scene_path']).read_text(encoding='utf-8'))
        assessment = json.loads((workspace/item['assessment_path']).read_text(encoding='utf-8'))
        if (not assessment['accepted'] or item['physics_signature'] != physics_signature(config)
                or assessment['signature'] != item['physics_signature']):
            raise ValueError('Workspace scene or source changed; validate a new map first')
        group = groups.setdefault(item['location_id'],dict(location_id=item['location_id'],
            split=item['future_split'],anchors=[]))
        if group['split'] != item['future_split']:
            raise ValueError('A location cannot cross dataset splits')
        group['anchors'].append(config)
    result = sorted(groups.values(),key=lambda g:g['location_id'])
    if not result or {g['split'] for g in result} != {'train','validation','test'}:
        raise ValueError('Accepted locations must include train, validation and test groups')
    centers = [(g['anchors'][0]['box_pose']['center_from_near_edge_mm'],
                g['anchors'][0]['box_pose']['center_left_of_robot_mm']) for g in result]
    for i,a in enumerate(centers):
        for b in centers[i+1:]:
            # +/-3 mm squares must not overlap across location groups.
            if max(abs(a[0]-b[0]),abs(a[1]-b[1])) <= 6:
                raise ValueError('Location jitter regions overlap; redefine split groups')
    return result


def candidate(group, group_index, attempt, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed,group_index,attempt]))
    anchors = sorted(group['anchors'],key=lambda c:c['box_pose']['yaw_deg'])
    # Cycle every accepted anchor angle; each cycle has a seeded random ordering.
    order = np.random.default_rng(np.random.SeedSequence([seed,group_index,attempt//len(anchors),42])).permutation(len(anchors))
    config = copy.deepcopy(anchors[int(order[attempt%len(anchors)])])
    anchor_id = config['episode_id']
    episode_seed = int(rng.integers(0,2**31))
    variation = instances(config,count=1,seed=episode_seed)[0]
    config.update(episode_id=f'{group["location_id"]}_attempt_{attempt:04d}',seed=episode_seed,
        split=group['split'],appearance=dict(profile='pbr',calibration='estimated procedural appearance'))
    config['object']['mass_kg'] = variation['object']['mass_kg']
    config['grasp']['sliding_friction'] = variation['grasp']['sliding_friction']
    config['table']['sliding_friction'] = variation['table']['sliding_friction']
    config['domain_randomization'] = variation['domain_randomization']
    pose = config['box_pose']
    jitter = rng.uniform(-3,3,2)
    pose['center_from_near_edge_mm'] += float(jitter[0])
    pose['center_left_of_robot_mm'] += float(jitter[1])
    pose['yaw_deg'] = float(np.clip(pose['yaw_deg']+rng.uniform(-1.5,1.5),-20,20))
    cameras = {}
    for name in CAMERAS:
        rotation = np.deg2rad(rng.uniform(-.5,.5,3))
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat,rotation,'xyz')
        cameras[name] = dict(position_delta_m=rng.uniform(-.001,.001,3).tolist(),
            rotation_delta_wxyz=quat.tolist(),fovy_scale=float(rng.uniform(.99,1.01)))
    config['domain_randomization'].update(camera_pose_randomized=True,camera_perturbations=cameras,
        camera_note='Episode-fixed +/-1 mm, +/-0.5 deg per local axis, +/-1% vertical FOV; assumed variation, not physical calibration')
    config['dataset_pilot'] = dict(instruction=TASK,anchor_id=anchor_id,group_index=group_index,
        attempt=attempt,position_jitter_mm=jitter.tolist(),location_group=group['location_id'],
        policy='X-VLA',privileged_pose_is_teacher_only=True)
    return config


def apply_camera_perturbations(model, config):
    """Apply once, identically to full physics and display-only models."""
    for name,p in config['domain_randomization']['camera_perturbations'].items():
        cid = model.camera(name).id
        model.cam_pos[cid] += p['position_delta_m']
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat,model.cam_quat[cid].copy(),np.asarray(p['rotation_delta_wxyz']))
        model.cam_quat[cid] = quat/np.linalg.norm(quat)
        model.cam_fovy[cid] *= p['fovy_scale']
    return model


def record_candidate(config, directory):
    directory = Path(directory)
    directory.mkdir()
    write_json(directory/'scene.json',config)
    summary = dict(episode_id=config['episode_id'],split=config['split'],seed=config['seed'],
        location_id=config['dataset_pilot']['location_group'],attempt=config['dataset_pilot']['attempt'],
        yaw_deg=config['box_pose']['yaw_deg'],videos=False,pilot_signature=pilot_signature(config))
    try:
        model,measured = scene_model(config)
        apply_camera_perturbations(model,config)
        run = WorkspacePickup(config,model,measured)
        recorder = EpisodeRecorder(run)
        run.before_step = recorder.capture
    except (ValueError,RuntimeError) as exc:
        summary.update(passed=False,eligible_for_imitation=False,error=str(exc),failed_checks=['construction'])
        write_json(directory/'result.json',summary)
        return summary
    try:
        report = run.run(run.audit_contacts)
    except (ValueError,RuntimeError) as exc:
        report = run.report()
        report.update(passed=False,error=str(exc))
    assessment = assess(run,report)
    report.update(teacher_passed=report['passed'],passed=assessment['accepted'],workspace_validation=assessment)
    save_run(run,report,directory,export_model=False)
    _,metadata = recorder.finish(report,directory)
    metadata.update(instruction=TASK,pilot_signature=summary['pilot_signature'])
    write_json(directory/'episode.json',metadata)
    summary.update(passed=assessment['accepted'],eligible_for_imitation=assessment['accepted'],
        frames=metadata['frames'],failed_checks=assessment['failed_checks'],
        minimum_hold_clearance_mm=report['minimum_bottom_clearance_during_hold_mm'])
    write_json(directory/'result.json',summary)
    return summary


def collect(workspace, output, count=100, seed=20261118, jobs=4, resume=False):
    if count < 6 or count > 10000 or not 1 <= jobs <= 4:
        raise ValueError('Require 6..10000 successes and 1..4 CPU workers')
    groups = read_anchors(workspace)
    if count < len(groups):
        raise ValueError('Need at least one success per location')
    for i,group in enumerate(groups):
        group['quota'] = count//len(groups)+(i<count%len(groups))
    output = Path(output)
    plan = dict(count=count,seed=seed,workspace_sha256=file_hash(Path(workspace)/'manifest.json'),
        source_signature=pilot_signature(groups[0]['anchors'][0]),groups=groups)
    if output.exists():
        if not resume:
            raise FileExistsError('Choose a new output directory or --resume a matching pilot')
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['plan'] != plan:
            raise ValueError('Pilot plan, code, mapping or workspace changed; choose a new output')
    else:
        output.mkdir(parents=True)
        manifest = dict(format='sourccey-box-pilot-v1',test='C',instruction=TASK,plan=plan,
            simulation_only=True,physical_calibration_verified=False,policy='X-VLA',fps=FPS,
            camera_names=list(CAMERAS),state_names=list(STATE_NAMES),rendering=provenance(),
            episodes=[],pending=[],status='physics')
        write_json(output/'manifest.json',manifest)
    # Completed worker output can be recovered if its manifest update was interrupted.
    for config in manifest['pending']:
        result_path=output/config['episode_id']/'result.json'
        if not result_path.exists():
            raise ValueError(f'Incomplete attempt preserved at {result_path.parent}; choose a fresh output')
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['pilot_signature'] != pilot_signature(config):
            raise ValueError('Recovered attempt signature mismatch')
        manifest['episodes'].append(result)
    manifest['pending']=[]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        while True:
            configs=[]
            for i,group in enumerate(groups):
                previous=[e for e in manifest['episodes'] if e['location_id']==group['location_id']]
                if sum(e['eligible_for_imitation'] for e in previous) >= group['quota']:
                    continue
                attempt=max((e['attempt'] for e in previous),default=-1)+1
                if attempt >= group['quota']*3+10:
                    raise RuntimeError('Success quota not met; inspect retained failures before collecting more')
                configs.append(candidate(group,i,attempt,seed))
            if not configs:
                break
            manifest['pending']=configs.copy()
            write_json(output/'manifest.json',manifest)
            futures={pool.submit(record_candidate,c,output/c['episode_id']):c for c in configs}
            for future in as_completed(futures):
                config=futures[future]
                result=future.result()
                manifest['episodes'].append(result)
                manifest['episodes'].sort(key=lambda e:(e['location_id'],e['attempt']))
                manifest['pending']=[c for c in manifest['pending'] if c['episode_id']!=config['episode_id']]
                write_json(output/'manifest.json',manifest)
                accepted=sum(e['eligible_for_imitation'] for e in manifest['episodes'])
                print(f'{accepted}/{count} successes | {result["episode_id"]}: '
                    +('PASS' if result['passed'] else 'REJECT '+str(result['failed_checks'])),flush=True)
    manifest['status']='physics_complete'
    write_json(output/'manifest.json',manifest)
    return manifest


def render_pilot(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    accepted=[e for e in manifest['episodes'] if e['eligible_for_imitation']]
    if len(accepted) != manifest['plan']['count']:
        raise ValueError('Complete the successful physics quota before rendering')
    if provenance()!=manifest['rendering']:
        raise ValueError('Renderer or engine changed; use the original rendering environment')
    manifest['status']='rendering'
    write_json(output/'manifest.json',manifest)
    for index,item in enumerate(accepted):
        directory=output/item['episode_id']
        config=json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        if item['pilot_signature']!=pilot_signature(config):
            raise ValueError('Pilot source or scene changed; collect a new pilot')
        if item['videos']:
            continue
        load_recording(directory,config)
        with np.load(directory/'transitions.npz',allow_pickle=False) as archive:
            arrays={k:archive[k].copy() for k in archive.files}
        model,_=scene_model(config,display_only=True)
        apply_camera_perturbations(model,config)
        display=Simulation(model=model,traction=False)
        if model.nq!=arrays['qpos'].shape[1]:
            raise ValueError('Display and recorded joint layout differ')
        displayed=SimpleNamespace(m=model,d=display.data,config=config)
        print(f'Rendering {index+1}/{len(accepted)}: {item["episode_id"]}',flush=True)
        write_videos(displayed,arrays,directory)
        display.data.qpos[:]=arrays['qpos'][-1]
        mujoco.mj_forward(model,display.data)
        render(displayed,directory/'final.png')
        item['videos']=True
        write_json(output/'manifest.json',manifest)
        del displayed,display,model,arrays
    manifest['status']='complete'
    manifest['summary']=dict(successful_episodes=len(accepted),attempted_episodes=len(manifest['episodes']),
        rejected_episodes=len(manifest['episodes'])-len(accepted),
        split_episodes={s:sum(e['split']==s for e in accepted) for s in ('train','validation','test')},
        nonterminal_frames=sum(e['frames']-1 for e in accepted),task=TASK,policy='X-VLA',
        training_started=False)
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['summary'],indent=2),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=DEFAULT_WORKSPACE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261118)
    parser.add_argument('--jobs',type=int,choices=range(1,5),default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=('all','physics','render'),default='all')
    args=parser.parse_args()
    if args.phase!='render':
        collect(args.workspace,args.output,args.count,args.seed,args.jobs,args.resume)
    if args.phase!='physics':
        render_pilot(args.output)


if __name__=='__main__':
    main()
