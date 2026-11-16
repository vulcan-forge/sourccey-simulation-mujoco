"""Validate box positions, then progressively wider rotations; simulation only.

This produces a tested pose catalog and physics diagnostics, not a training dataset.
No hardware imports, connections, policy training or motor commands.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from examples.box_pickup import Pickup, load_scene, validate, save_run, physics_signature
from sourccey.build import ROOT, WHEELS


DEFAULT_PLAN = ROOT/'examples/box_workspace.json'
OUTPUT = ROOT/'artifacts/box_workspace_c'
THRESHOLD_NAMES = {
    'maximum_transit_ik_error_mm', 'maximum_grasp_ik_error_mm',
    'maximum_grasp_pad_angle_deg', 'maximum_joint_endpoint_error_deg',
    'maximum_box_motion_before_close_mm', 'maximum_self_contact_force_n'}


def read_plan(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get('version') != 1 or not isinstance(plan.get('seed'),int) or plan['seed'] < 0:
        raise ValueError('Require plan version 1 and a nonnegative integer seed')
    for key in ('depth_offsets_mm','lateral_offsets_mm'):
        values = plan[key]
        if not isinstance(values,list) or not values or len(values) > 15:
            raise ValueError('Each grid axis requires 1..15 offsets')
        if not np.isfinite(values).all() or len(set(values)) != len(values) or 0 not in values:
            raise ValueError('Grid offsets must be finite, unique, and include zero')
    stages = plan['yaw_stages_deg']
    if (not stages or not np.isfinite(stages).all() or stages[0] <= 0
            or stages[-1] > 45 or any(b <= a for a,b in zip(stages,stages[1:]))):
        raise ValueError('Yaw stages must increase strictly from positive angles to at most 45 degrees')
    if plan.get('appearance') not in ('pbr','baseline'):
        raise ValueError('Appearance must be pbr or baseline')
    if set(plan['thresholds']) != THRESHOLD_NAMES:
        raise ValueError('Workspace thresholds are missing or unknown')
    if any(not np.isfinite(v) or v <= 0 for v in plan['thresholds'].values()):
        raise ValueError('All thresholds must be positive and finite')


def locations(base, plan):
    """Stable row/column IDs; rotations at one location retain one future split."""
    validate_plan(plan)
    result = []
    for row,depth in enumerate(sorted(plan['depth_offsets_mm'])):
        for col,lateral in enumerate(sorted(plan['lateral_offsets_mm'])):
            result.append(dict(location_id=f'location_{row:02d}_{col:02d}',row=row,column=col,
                center_from_near_edge_mm=base['box_pose']['center_from_near_edge_mm']+depth,
                center_left_of_robot_mm=base['box_pose']['center_left_of_robot_mm']+lateral))
    # Split locations, never frames or individual rotations. Reserve splits now
    # so future collection cannot put the same location in train and evaluation.
    order = np.random.default_rng(plan['seed']).permutation(len(result))
    ntest = max(1,round(len(result)*.2)) if len(result) >= 3 else 0
    nvalidation = max(1,round(len(result)*.1)) if len(result) >= 5 else 0
    for rank,index in enumerate(order):
        result[index]['future_split'] = ('test' if rank < ntest else 'validation'
            if rank < ntest+nvalidation else 'train')
    return result


def trial_scene(base, plan, location, yaw, stage):
    config = copy.deepcopy(base)
    angle = f'{abs(yaw):g}'.replace('.','d')
    suffix = 'zero' if yaw == 0 else ('p' if yaw > 0 else 'm')+angle
    trial_id = location['location_id']+'_yaw_'+suffix
    config.update(test='C',episode_id=trial_id,split=location['future_split'],
        seed=plan['seed'],workspace=dict(location_id=location['location_id'],stage=stage,
            thresholds=copy.deepcopy(plan['thresholds']),purpose='pose validation; not training data'))
    config['box_pose'].update(center_from_near_edge_mm=location['center_from_near_edge_mm'],
        center_left_of_robot_mm=location['center_left_of_robot_mm'],yaw_deg=float(yaw),
        orientation='Upright, yaw about world +Z; wheels and elevator fixed')
    if plan['appearance'] == 'pbr':
        config['appearance'] = dict(profile='pbr',calibration='estimated procedural appearance')
    # Keep dynamics and appearance fixed while diagnosing pose coverage. Domain
    # randomization and large image-dataset collection belong to the next phase.
    config.pop('domain_randomization',None)
    validate(config)
    return config


class WorkspacePickup(Pickup):
    def __init__(self, config, model=None, measured=None):
        super().__init__(config,model,measured)
        self.ik_errors = {}
        self.joint_endpoint_errors = []
        self.grasp_pad_angle = None
        self.self_contacts = {}
        self.environment_contacts = {}

    def solve(self, target, seed=None):
        q,error = super().solve(target,seed)
        self.ik_errors[self.phase] = max(self.ik_errors.get(self.phase,0.),error*1000)
        return q,error

    def joint_move(self, destination, seconds=2.):
        super().joint_move(destination,seconds)
        self.joint_endpoint_errors.append(float(np.rad2deg(np.max(abs(self.d.qpos[self.qp]-destination)))))

    def cartesian_move(self, target, seconds=2.):
        super().cartesian_move(target,seconds)
        if self.phase == 'align-fingers':
            normal = self.d.xmat[self.palm].reshape(3,3)[:,2]
            self.grasp_pad_angle = float(np.rad2deg(np.arccos(np.clip(normal@self.grasp_normal,-1,1))))

    def audit_contacts(self, run):
        for index,contact in enumerate(self.d.contact):
            ga,gb = int(contact.geom1),int(contact.geom2)
            ba,bb = int(self.m.geom_bodyid[ga]),int(self.m.geom_bodyid[gb])
            if self.box in (ba,bb) or contact.efc_address < 0:
                continue
            if ba == 0 or bb == 0:
                world_geom,robot_geom = (ga,gb) if ba == 0 else (gb,ga)
                if (self.m.geom(world_geom).name != 'floor'
                        or self.m.geom(robot_geom).name in {'contact_'+wheel for wheel in WHEELS}):
                    continue
                destination = self.environment_contacts
            else:
                destination = self.self_contacts
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m,self.d,index,force)
            if force[0] <= self.config['workspace']['thresholds']['maximum_self_contact_force_n']:
                continue
            pair = ' / '.join(sorted((self.m.geom(ga).name,self.m.geom(gb).name)))
            previous = destination.get(pair,dict(phase=self.phase,first_time_s=float(self.d.time),
                maximum_normal_force_n=0.,maximum_penetration_mm=0.))
            previous['maximum_normal_force_n'] = max(previous['maximum_normal_force_n'],float(force[0]))
            previous['maximum_penetration_mm'] = max(previous['maximum_penetration_mm'],-1000*float(contact.dist))
            destination[pair] = previous


def assess(run, report):
    thresholds = run.config['workspace']['thresholds']
    transit = max((error for phase,error in run.ik_errors.items() if phase != 'align-fingers'),default=0.)
    grasp = run.ik_errors.get('align-fingers')
    preclose = report['box_motion_before_close_mm']
    checks = dict(teacher_pickup_and_hold=bool(report['passed']),
        no_robot_self_contact=not run.self_contacts,
        no_unintended_floor_contact=not getattr(run,'environment_contacts',{}),
        transit_ik_bounded=transit <= thresholds['maximum_transit_ik_error_mm'],
        grasp_ik_bounded=grasp is not None and grasp <= thresholds['maximum_grasp_ik_error_mm'],
        grasp_pad_aligned=run.grasp_pad_angle is not None and run.grasp_pad_angle <= thresholds['maximum_grasp_pad_angle_deg'],
        joints_reached_transit_endpoints=bool(run.joint_endpoint_errors) and max(run.joint_endpoint_errors) <= thresholds['maximum_joint_endpoint_error_deg'],
        box_not_pushed_before_grasp=preclose is not None and preclose <= thresholds['maximum_box_motion_before_close_mm'])
    checks = {name:bool(value) for name,value in checks.items()}
    return dict(accepted=all(checks.values()),checks=checks,
        failed_checks=[key for key,value in checks.items() if not value],
        max_ik_error_mm_by_phase=run.ik_errors,grasp_pad_angle_deg=run.grasp_pad_angle,
        maximum_joint_endpoint_error_deg=max(run.joint_endpoint_errors,default=None),
        self_contacts=run.self_contacts,unintended_floor_contacts=getattr(run,'environment_contacts',{}),thresholds=thresholds,
        note='Transit waypoints may be approximate. Acceptance requires measured pickup/hold, tighter grasp alignment and full-path contact checks; no continuous workspace is inferred.')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def execute_trial(config, directory):
    directory = Path(directory)
    directory.mkdir()
    write_json(directory/'scene.json',config)
    try:
        run = WorkspacePickup(config)
    except (ValueError,RuntimeError) as exc:
        result = dict(accepted=False,failed_checks=['scene_construction'],error=str(exc),
            replay_available=False,signature=physics_signature(config))
        write_json(directory/'assessment.json',result)
        return result
    error = None
    try:
        report = run.run(run.audit_contacts)
    except (ValueError,RuntimeError) as exc:
        error = str(exc)
        report = run.report()
        report.update(passed=False,error=error)
    result = assess(run,report)
    result.update(error=error,replay_available=bool(run.frames),signature=physics_signature(config),
        minimum_hold_clearance_mm=report['minimum_bottom_clearance_during_hold_mm'])
    report.update(teacher_passed=report['passed'],passed=result['accepted'],workspace_validation=result)
    # Hundreds of MB of unchanged meshes per trial are unnecessary for review.
    # Rebuild the model from the signed scene; keep the compact physics replay.
    save_run(run,report,directory,export_model=False)
    write_json(directory/'assessment.json',result)
    return result


def eligible_locations(points, episodes, previous_stage):
    accepted = {(e['location_id'],e['yaw_deg']) for e in episodes if e['accepted']}
    required = [0.] if previous_stage == 0 else [-previous_stage,previous_stage]
    return [p for p in points if all((p['location_id'],yaw) in accepted for yaw in required)]


def summarize(manifest):
    episodes = manifest['episodes']
    validated = []
    for point in manifest['locations']:
        trials = [e for e in episodes if e['location_id'] == point['location_id']]
        passed_yaws = sorted(e['yaw_deg'] for e in trials if e['accepted'])
        validated.append(dict(**point,nominal_passed=0. in passed_yaws,validated_yaws_deg=passed_yaws,
            tested=len(trials),accepted=sum(e['accepted'] for e in trials),
            rejected=[dict(trial_id=e['trial_id'],yaw_deg=e['yaw_deg'],failed_checks=e['failed_checks'])
                      for e in trials if not e['accepted']]))
    return dict(locations=validated,nominal_locations_passed=sum(p['nominal_passed'] for p in validated),
        tested_locations=len(validated),tested_poses=len(episodes),accepted_poses=sum(e['accepted'] for e in episodes),
        eligible_for_next_phase=any(e['accepted'] and e['yaw_deg'] != 0 for e in episodes),
        future_policy='X-VLA',training_started=False,image_dataset_generated=False,
        reachability_scope='Only listed, fully tested discrete poses. Untested positions, angles and interpolation are unvalidated.')


def save_outputs(output, manifest):
    manifest['summary'] = summarize(manifest)
    write_json(output/'manifest.json',manifest)
    accepted = [dict(trial_id=e['trial_id'],location_id=e['location_id'],yaw_deg=e['yaw_deg'],
        future_split=e['future_split'],scene_path=e['trial_id']+'/scene.json',
        assessment_path=e['trial_id']+'/assessment.json',physics_signature=e['signature'])
        for e in manifest['episodes'] if e['accepted']]
    write_json(output/'validated_scenes.json',dict(format='sourccey-validated-box-poses-v1',
        future_policy='X-VLA',simulation_only=True,training_data=False,
        source_manifest='manifest.json',scenes=accepted))
    workspace_image(output,manifest)


def workspace_image(output, manifest):
    points = manifest['summary']['locations']
    rows = len(manifest['plan']['depth_offsets_mm'])
    columns = len(manifest['plan']['lateral_offsets_mm'])
    image = Image.new('RGB',(max(800,columns*225+40),rows*145+115),(22,26,32))
    draw = ImageDraw.Draw(image)
    draw.text((20,15),'Box workspace C | full-path physics validation | stationary left arm',fill='white')
    draw.text((20,35),'Depth from table near edge; lateral left of robot center. Only listed yaws validated.',fill=(185,195,210))
    for point in points:
        x,y=20+point['column']*225,65+point['row']*145
        color=(45,100,70) if point['nominal_passed'] else (115,55,50) if point['tested'] else (65,70,80)
        draw.rounded_rectangle((x,y,x+210,y+130),radius=6,fill=color)
        yaws=', '.join(f'{v:+g}' for v in point['validated_yaws_deg']) or 'none'
        draw.text((x+8,y+8),point['location_id'],fill='white')
        draw.text((x+8,y+28),f'depth {point["center_from_near_edge_mm"]:g} mm',fill='white')
        draw.text((x+8,y+45),f'lateral {point["center_left_of_robot_mm"]:g} mm',fill='white')
        draw.text((x+8,y+66),f'accepted {point["accepted"]}/{point["tested"]}',fill='white')
        # Split the compact angle list to keep the map readable as stages grow.
        draw.text((x+8,y+86),'yaw: '+yaws[:31],fill='white')
        if len(yaws)>31:
            draw.text((x+8,y+103),yaws[31:],fill='white')
    image.save(output/'workspace.png')


def run_workspace(plan, output=OUTPUT, jobs=2, resume=False):
    validate_plan(plan)
    if not 1 <= jobs <= 4:
        raise ValueError('Worker count must be 1..4')
    base = load_scene()
    points = locations(base,plan)
    signature = hashlib.sha256((physics_signature(dict(base,test='C'))+
        json.dumps(plan,sort_keys=True,allow_nan=False)).encode()).hexdigest()
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError('Choose a new output directory, or --resume a matching workspace run')
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('Workspace code, engine, model or plan changed; choose a new output directory')
    else:
        if resume:
            raise FileNotFoundError('No workspace run exists to resume')
        output.mkdir(parents=True)
        manifest = dict(format='sourccey-box-workspace-v1',signature=signature,plan=plan,
            simulation_only=True,locations=points,episodes=[],status='running',
            physics_engine=mujoco.__version__,physical_calibration_verified=False)
    manifest['status'] = 'running'
    save_outputs(output,manifest)
    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=jobs)) if jobs>1 else None
        previous = 0.
        for stage in [0.,*plan['yaw_stages_deg']]:
            candidates = points if stage==0 else eligible_locations(points,manifest['episodes'],previous)
            configs = [trial_scene(base,plan,p,yaw,stage) for p in candidates
                       for yaw in ([0.] if stage==0 else [-stage,stage])]
            print(f'Stage {stage:g} deg: {len(configs)} planned trials at {len(candidates)} locations',flush=True)
            known = {e['trial_id']:e for e in manifest['episodes']}
            pending = []
            for config in configs:
                trial_id = config['episode_id']
                directory = output/trial_id
                if trial_id in known:
                    saved = json.loads((directory/'assessment.json').read_text(encoding='utf-8'))
                    if saved['signature'] != physics_signature(config) or saved['accepted'] != known[trial_id]['accepted']:
                        raise ValueError('Saved trial changed; choose a new output directory')
                    continue
                if directory.exists():
                    # Recover a completed trial whose manifest update was interrupted.
                    assessment = directory/'assessment.json'
                    if assessment.exists():
                        result = json.loads(assessment.read_text(encoding='utf-8'))
                        if result['signature'] != physics_signature(config):
                            raise ValueError('Saved trial signature mismatch')
                        pending.append((config,result))
                    else:
                        raise FileExistsError(f'Incomplete trial preserved at {directory}; choose a new output directory')
                else:
                    pending.append((config,executor.submit(execute_trial,config,directory) if executor else None))
            if executor:
                futures = {value:config for config,value in pending if not isinstance(value,dict)}
                completed = [(config,value) for config,value in pending if isinstance(value,dict)]
                results = ((futures[f],f.result()) for f in as_completed(futures))
            else:
                completed = []
                results = ((config,value if isinstance(value,dict) else execute_trial(config,output/config['episode_id']))
                           for config,value in pending)
            for config,result in completed:
                append_result(manifest,config,result,output)
            for config,result in results:
                append_result(manifest,config,result,output)
            previous = stage
    manifest['status'] = 'complete'
    save_outputs(output,manifest)
    print(json.dumps(manifest['summary'],indent=2),flush=True)
    return manifest


def append_result(manifest, config, result, output):
    manifest['episodes'].append(dict(trial_id=config['episode_id'],
        location_id=config['workspace']['location_id'],yaw_deg=config['box_pose']['yaw_deg'],
        stage_deg=config['workspace']['stage'],future_split=config['split'],**result))
    manifest['episodes'].sort(key=lambda e:(e['stage_deg'],e['location_id'],e['yaw_deg']))
    save_outputs(output,manifest)
    print(f'  {config["episode_id"]}: {"PASS" if result["accepted"] else "REJECT"} '
          + ', '.join(result['failed_checks']),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--jobs',type=int,choices=range(1,5),default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--review',action='store_true',help='Review an existing trial in Studio; never recompute')
    parser.add_argument('--trial',help='Trial ID from manifest.json; defaults to first accepted trial')
    args = parser.parse_args()
    if args.review:
        manifest = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
        available = [e for e in manifest['episodes'] if e.get('replay_available')]
        if args.trial:
            matches = [e for e in available if e['trial_id']==args.trial]
        else:
            matches = [e for e in available if e['accepted']]
        if not matches:
            parser.error('No matching replayable trial exists; inspect manifest.json')
        from examples.studio_review import review
        review(args.output/matches[0]['trial_id'])
    else:
        run_workspace(read_plan(args.plan),args.output,args.jobs,args.resume)


if __name__ == '__main__':
    main()
