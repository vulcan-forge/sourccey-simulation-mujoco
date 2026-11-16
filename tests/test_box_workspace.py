import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from examples.box_pickup import load_scene, validate, layout
from examples.box_workspace import (read_plan, validate_plan, locations, trial_scene,
    eligible_locations, assess, run_workspace, WorkspacePickup)


def test_grid_is_local_deterministic_and_preserves_measured_scene():
    base,plan = load_scene(),read_plan()
    points = locations(base,plan)
    assert points == locations(base,plan)
    assert len(points) == 9
    assert {p['center_from_near_edge_mm'] for p in points} == {80,110,140}
    assert {p['center_left_of_robot_mm'] for p in points} == {140,180,220}
    assert {p['future_split'] for p in points} == {'train','validation','test'}
    for point in points:
        config = trial_scene(base,plan,point,20,20)
        validate(config)
        measured = layout(config)
        assert config['test'] == 'C'
        assert config['table'] == base['table'] and config['object'] == base['object']
        assert config['robot_setup'] == base['robot_setup']
        assert config['split'] == trial_scene(base,plan,point,-20,20)['split']
        assert not config.get('domain_randomization')
        assert measured['tabletop_z_m'] == pytest.approx(.762)
    assert base['test'] == 'A'


@pytest.mark.parametrize('key,value',[
    ('depth_offsets_mm',[10]),('lateral_offsets_mm',[0,0]),
    ('depth_offsets_mm',[0,float('nan')]),('yaw_stages_deg',[12,6]),
    ('yaw_stages_deg',[0,6]),('yaw_stages_deg',[46]),('yaw_stages_deg',[])])
def test_invalid_plans_are_rejected(key,value):
    plan = read_plan()
    plan[key] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_rotations_only_advance_after_both_previous_angles_pass():
    points = [dict(location_id='a'),dict(location_id='b'),dict(location_id='c')]
    episodes = [dict(location_id='a',yaw_deg=0.,accepted=True),
                dict(location_id='b',yaw_deg=0.,accepted=False),
                dict(location_id='a',yaw_deg=-6.,accepted=True),
                dict(location_id='a',yaw_deg=6.,accepted=False)]
    assert eligible_locations(points,episodes,0) == [points[0]]
    assert eligible_locations(points,episodes,6) == []
    episodes[-1]['accepted'] = True
    assert eligible_locations(points,episodes,6) == [points[0]]


def test_successful_final_pose_cannot_hide_path_failures():
    config = trial_scene(load_scene(),read_plan(),locations(load_scene(),read_plan())[4],0.,0.)
    run = SimpleNamespace(config=config,ik_errors={'approach':27.,'align-fingers':4.,'lift':24.},
        joint_endpoint_errors=[.2],grasp_pad_angle=4.,self_contacts={})
    report = dict(passed=True,box_motion_before_close_mm=.4)
    assert assess(run,report)['accepted']
    run.self_contacts['hand / torso'] = dict(maximum_normal_force_n=2.)
    assert 'no_robot_self_contact' in assess(run,report)['failed_checks']
    run.self_contacts.clear()
    run.ik_errors['align-fingers'] = 10.
    assert 'grasp_ik_bounded' in assess(run,report)['failed_checks']
    run.ik_errors['align-fingers'] = 4.
    report['box_motion_before_close_mm'] = 5.
    assert 'box_not_pushed_before_grasp' in assess(run,report)['failed_checks']


def test_workspace_orchestration_preserves_rejections_splits_and_resume(monkeypatch,tmp_path):
    from examples import box_workspace as workspace
    calls = []
    def execute(config,directory):
        calls.append(config['episode_id'])
        directory.mkdir()
        # Only the center-depth location may proceed; positive six-degree yaw
        # fails there, which must prevent both twelve-degree trials.
        accepted = config['box_pose']['center_from_near_edge_mm']==110 and config['box_pose']['yaw_deg']<=0
        result = dict(accepted=accepted,failed_checks=[] if accepted else ['mock_path_rejection'],
            replay_available=False,signature=workspace.physics_signature(config))
        workspace.write_json(directory/'assessment.json',result)
        return result
    monkeypatch.setattr(workspace,'execute_trial',execute)
    plan = read_plan()
    plan['lateral_offsets_mm'] = [0]
    plan['yaw_stages_deg'] = [6,12]
    output = tmp_path/'workspace'
    manifest = run_workspace(plan,output,jobs=1)
    assert len(calls) == 5
    assert manifest['summary']['nominal_locations_passed'] == 1
    assert manifest['summary']['accepted_poses'] == 2
    assert manifest['summary']['future_policy'] == 'X-VLA'
    assert not manifest['summary']['training_started']
    catalog = json.loads((output/'validated_scenes.json').read_text())
    assert len(catalog['scenes']) == 2 and not catalog['training_data']
    run_workspace(plan,output,jobs=1,resume=True)
    assert len(calls) == 5
    changed = copy.deepcopy(plan)
    changed['yaw_stages_deg'] = [6,12,20]
    with pytest.raises(ValueError,match='changed'):
        run_workspace(changed,output,jobs=1,resume=True)
    with pytest.raises(FileExistsError):
        run_workspace(plan,output,jobs=1)
    assert (output/'workspace.png').exists()


def test_c_angle_limit_does_not_expand_b_curriculum():
    base,plan=load_scene(),read_plan()
    point=locations(base,plan)[4]
    config=trial_scene(base,plan,point,30,30)
    validate(config)
    config['test']='B'
    with pytest.raises(ValueError,match='15'):
        validate(config)


def test_contact_audit_catches_arm_torso_contacts_throughout_motion():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><geom name="torso_geom" type="sphere" size=".1"/></body>
      <body name="arm" pos=".15 0 0"><freejoint/><geom name="arm_geom" size=".1"/></body>
      <body name="box" pos="0 0 2"><freejoint/><geom size=".03"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    audit=SimpleNamespace(m=model,d=data,box=model.body('box').id,phase='approach',self_contacts={},
        config={'workspace':{'thresholds':{'maximum_self_contact_force_n':.1}}})
    WorkspacePickup.audit_contacts(audit,audit)
    assert audit.self_contacts
    pair=next(iter(audit.self_contacts.values()))
    assert pair['phase']=='approach' and pair['maximum_normal_force_n'] > .1


def test_floor_audit_allows_wheels_but_rejects_arm_contact():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .08"><freejoint/><geom name="arm_geom" size=".1"/></body>
      <body pos="1 0 .08"><freejoint/><geom name="contact_front_left_wheel" size=".1"/></body>
      <body name="box" pos="0 0 2"><freejoint/><geom size=".03"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    audit=SimpleNamespace(m=model,d=data,box=model.body('box').id,phase='descend',self_contacts={},
        environment_contacts={},config={'workspace':{'thresholds':{'maximum_self_contact_force_n':.1}}})
    WorkspacePickup.audit_contacts(audit,audit)
    assert len(audit.environment_contacts)==1
    assert 'arm_geom' in next(iter(audit.environment_contacts))
