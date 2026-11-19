import copy
import json

import mujoco
import numpy as np
import pytest

from examples.box_pickup import load_scene,scene_model
from examples.box_workspace import locations,read_plan,trial_scene
from examples.box_dataset_pilot import candidate,apply_camera_perturbations,TASK,read_anchors


def group():
    base,plan=load_scene(),read_plan()
    point=locations(base,plan)[4]
    return dict(location_id=point['location_id'],split='train',anchors=[
        trial_scene(base,plan,point,yaw,abs(yaw)) for yaw in (-12,0,12)])


def test_pilot_randomizes_only_bounded_pose_and_parameters_with_fixed_split():
    source=group()
    configs=[candidate(source,1,i,123) for i in range(18)]
    assert configs==[candidate(source,1,i,123) for i in range(18)]
    assert len({c['seed'] for c in configs})==18
    assert len({c['box_pose']['yaw_deg'] for c in configs})==18
    for c in configs:
        assert c['dataset_pilot']['instruction']==TASK=='Pick up the box'
        assert c['split']=='train' and c['dataset_pilot']['location_group']==source['location_id']
        assert c['table']['height_above_floor_mm']==762
        assert c['object']['dimensions_mm']==load_scene()['object']['dimensions_mm']
        assert abs(c['box_pose']['center_from_near_edge_mm']-110)<=3
        assert abs(c['box_pose']['center_left_of_robot_mm']-180)<=3
        assert .115<=c['object']['mass_kg']<=.125
        assert c['domain_randomization']['camera_pose_randomized']
    assert 'dataset_pilot' not in source['anchors'][0]


def test_camera_randomization_matches_physics_and_replay_models():
    config=candidate(group(),1,0,123)
    model,_=scene_model(config,display_only=True)
    original_position=model.cam_pos.copy()
    original_quaternion=model.cam_quat.copy()
    original_fovy=model.cam_fovy.copy()
    repeated=copy.copy(model)
    apply_camera_perturbations(model,config)
    apply_camera_perturbations(repeated,config)
    np.testing.assert_array_equal(model.cam_pos,repeated.cam_pos)
    np.testing.assert_array_equal(model.cam_quat,repeated.cam_quat)
    for name in ('front_left','front_right','wrist_left','wrist_right'):
        cid=model.camera(name).id
        assert not np.array_equal(model.cam_pos[cid],original_position[cid])
        assert np.max(abs(model.cam_pos[cid]-original_position[cid]))<=.001
        assert abs(original_quaternion[cid]@model.cam_quat[cid]) > np.cos(np.deg2rad(1)/2)
        assert .99<=model.cam_fovy[cid]/original_fovy[cid]<=1.01
    np.testing.assert_array_equal(model.body_mass,repeated.body_mass)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert np.isfinite(data.cam_xpos).all()


def test_incomplete_workspace_cannot_seed_a_dataset(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'running'}))
    (tmp_path/'validated_scenes.json').write_text(json.dumps({'simulation_only':True}))
    with pytest.raises(ValueError,match='completed'):
        read_anchors(tmp_path)
