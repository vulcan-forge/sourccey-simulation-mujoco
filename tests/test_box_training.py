import copy

import mujoco
import numpy as np
import pytest

from examples.box_pickup import Pickup,load_scene,scene_model,validate
from examples.box_training import CAMERAS,EpisodeRecorder,camera_manifest,instances,photometric
from integrations.export_box_dataset import validate_episode


@pytest.fixture(scope='module')
def display():
    config = instances(load_scene(),10)[0]
    model,measured = scene_model(config,display_only=True)
    return config,model,measured


def test_seeded_curriculum_has_ten_distinct_small_rotations():
    base = load_scene()
    configs = instances(base)
    assert configs == instances(base)
    assert len({c['seed'] for c in configs}) == 10
    angles = [c['box_pose']['yaw_deg'] for c in configs]
    assert len(set(angles)) == 10 and min(angles) < -9 and max(angles) > 9
    assert all(abs(yaw) < 12 for yaw in angles)
    assert [c['split'] for c in configs].count('test') == 2
    for config in configs:
        validate(config)
        assert config['table']['height_above_floor_mm'] == base['table']['height_above_floor_mm']
        assert config['box_pose']['center_from_near_edge_mm'] == 110
        assert not config['domain_randomization']['camera_pose_randomized']
    assert base['test'] == 'A' and base['object']['mass_kg'] == .12


def test_appearance_is_deterministic_and_does_not_change_geometry(display):
    config,model,_ = display
    repeated,_ = scene_model(config,display_only=True)
    assert np.array_equal(model.tex_data,repeated.tex_data)
    changed = copy.deepcopy(config)
    changed['domain_randomization']['table_rgb'] = [.9,.1,.2]
    other,_ = scene_model(changed,display_only=True)
    assert not np.array_equal(model.tex_data,other.tex_data)
    np.testing.assert_array_equal(model.body_mass,other.body_mass)
    np.testing.assert_array_equal(model.geom_size,other.geom_size)
    np.testing.assert_array_equal(model.geom_pos,other.geom_pos)
    np.testing.assert_array_equal(model.cam_quat,other.cam_quat)


def test_four_camera_manifest_and_wrist_transform(display):
    config,model,measured = display
    run = Pickup(config,model,measured)
    recorder = EpisodeRecorder(run)
    manifest = camera_manifest(model,run.d)
    assert set(manifest['cameras']) == set(CAMERAS)
    assert not manifest['physical_calibration_verified']
    initial = run.d.cam_xpos[model.camera('wrist_left').id].copy()
    run.d.qpos[model.joint('left_shoulder_lift').qposadr[0]] += .5
    recorder.capture(run)
    updated = run.d.cam_xpos[model.camera('wrist_left').id]
    assert np.linalg.norm(initial-updated) > .05
    for name in CAMERAS:
        cid = model.camera(name).id
        bid = model.cam_bodyid[cid]
        expected = run.d.xpos[bid]+run.d.xmat[bid].reshape(3,3)@model.cam_pos[cid]
        np.testing.assert_allclose(run.d.cam_xpos[cid],expected,atol=1e-10)
        assert manifest['cameras'][name]['K'][0][0] > 0


def test_action_is_command_after_observation_not_joint_readback(display,tmp_path):
    config,model,measured = display
    run = Pickup(config,model,measured)
    recorder = EpisodeRecorder(run)
    run.before_step = recorder.capture
    run.sim.set_joints({'left_shoulder_pan':-.5})
    run.advance(.04)
    _,metadata = recorder.finish({'passed':False},tmp_path)
    arrays,_ = validate_episode(tmp_path,check_video=False)
    assert abs(float(arrays['action'][0,0]-arrays['state'][0,0])) > 3
    assert arrays['command_rad'][0,0] == pytest.approx(-.5)
    assert arrays['joint_rad'][0,0] != pytest.approx(-.5)
    np.testing.assert_allclose(arrays['timestamp'],[0,.02,.04],atol=1e-10)
    np.testing.assert_array_equal(arrays['next_state'][:-1],arrays['state'][1:])
    assert not metadata['eligible_for_imitation']
    assert 'box_pose' not in metadata['learner_features']
    arrays['next_state'][0,0] += 1
    np.savez_compressed(tmp_path/'transitions.npz',**arrays)
    with pytest.raises(ValueError,match='alignment'):
        validate_episode(tmp_path,check_video=False)


def test_sensor_noise_is_seeded_per_camera_and_frame(display):
    config,_,_ = display
    image = np.full((12,16,3),100,dtype=np.uint8)
    np.testing.assert_array_equal(photometric(image,config,0,1),photometric(image,config,0,1))
    assert not np.array_equal(photometric(image,config,0,1),photometric(image,config,1,1))


def test_b_curriculum_rejects_unvalidated_large_yaw():
    config = instances(load_scene())[0]
    config['box_pose']['yaw_deg'] = 30
    with pytest.raises(ValueError,match='15'):
        validate(config)


def test_pbr_materials_preserve_physics_and_camera_calibration(display):
    config,original,_ = display
    config = copy.deepcopy(config)
    config['appearance'] = dict(profile='pbr')
    pbr,_ = scene_model(config,display_only=True)
    repeated,_ = scene_model(config,display_only=True)
    np.testing.assert_array_equal(pbr.body_mass,original.body_mass)
    np.testing.assert_array_equal(pbr.geom_size,original.geom_size)
    np.testing.assert_array_equal(pbr.cam_quat,original.cam_quat)
    np.testing.assert_array_equal(pbr.tex_data,repeated.tex_data)
    assert pbr.mat_roughness[pbr.material('episode_material_0').id] >= .6
    assert pbr.mat_metallic[pbr.material('pbr_metal').id] == pytest.approx(.8)
