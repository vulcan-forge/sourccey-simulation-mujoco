"""Check compiled rays against benchmark image framing and nominal references."""
import json

import mujoco
import numpy as np
import pytest

from sourccey.build import CAMERA_POSES,MODEL,CAD_ROTATION


@pytest.fixture(scope='module')
def cameras():
    return mujoco.MjModel.from_xml_path(str(MODEL))


def rotation(model,name):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix,model.cam_quat[model.camera(name).id])
    return matrix.reshape(3,3)


@pytest.mark.parametrize('side,sign',[('left',1),('right',-1)])
def test_eye_rays_preserve_nominal_yaw_and_mean_pitch(cameras,side,sign):
    forward=-(CAD_ROTATION@rotation(cameras,'front_'+side))[:,2]
    assert np.rad2deg(np.arctan2(forward[1],forward[0])) == pytest.approx(sign*15,abs=1e-8)
    assert np.rad2deg(np.arcsin(forward[2])) == pytest.approx(-20,abs=.01)


@pytest.mark.parametrize('side',['left','right'])
def test_wrist_fit_keeps_cad_mount_and_archives_nominal_angles(cameras,side):
    name = 'wrist_'+side
    archived = json.loads(CAMERA_POSES.with_name('camera_poses.visual_fit_20261001.json').read_text())
    np.testing.assert_allclose(cameras.cam_pos[cameras.camera(name).id],archived[name]['nominal_mount']['pos'],atol=1e-10)
    assert cameras.body(cameras.cam_bodyid[cameras.camera(name).id]).name == ('Gripper_Base_v1_1' if side == 'left' else 'Gripper_Base_v1_2')
    active=json.loads(CAMERA_POSES.read_text())[name]
    assert active['nominal_mount_angles_deg']['inward'] == 16.9
    assert active['nominal_mount_angles_deg']['up'] == 2.5
    assert np.linalg.norm(active['camera_frame_delta_xyz_deg']) < 25


@pytest.mark.parametrize('name',['front_left','front_right','wrist_left','wrist_right'])
def test_effective_fov_is_separate_from_nominal_lens_fov(cameras,name):
    tangent = np.tan(np.deg2rad(cameras.cam_fovy[cameras.camera(name).id])/2)
    diagonal = np.rad2deg(2*np.arctan(tangent*np.sqrt(1+(320/240)**2)))
    pose=json.loads(CAMERA_POSES.read_text())[name]
    assert diagonal == pytest.approx(pose['effective_diagonal_fov_deg'],abs=1e-9)
    assert pose['nominal_lens_diagonal_fov_deg'] == 120
    assert 40 < cameras.cam_fovy[cameras.camera(name).id] < 55


def test_compiled_eye_fit_predicts_held_out_white_dataset_correspondences(cameras):
    from pathlib import Path
    points=json.loads((Path(__file__).parent/'fixtures/benchmark_camera_observations.json').read_text())['stereo_pairs']
    points=[p for p in points if 'white-' in p['dataset']]
    left=np.asarray([p['left'] for p in points]);right=np.asarray([p['right'] for p in points])
    f=120/np.tan(np.deg2rad(cameras.cam_fovy[cameras.camera('front_left').id])/2)
    rays=np.column_stack([(left[:,0]-160)/f,-(left[:,1]-120)/f,-np.ones(len(left))])
    rays=rays@rotation(cameras,'front_left').T@rotation(cameras,'front_right')
    predicted=np.column_stack([160+f*rays[:,0]/-rays[:,2],120-f*rays[:,1]/-rays[:,2]])
    assert len(points) == 52
    assert np.median(np.linalg.norm(predicted-right,axis=1)) < 4


def test_eye_optical_axes_clear_the_cad_housing(cameras):
    data=mujoco.MjData(cameras)
    mujoco.mj_resetDataKeyframe(cameras,data,cameras.key('startup').id)
    mujoco.mj_forward(cameras,data)
    group=np.zeros(6,dtype=np.uint8);group[2]=1
    for name in ('front_left','front_right'):
        cid=cameras.camera(name).id
        hit=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(cameras,data,data.cam_xpos[cid],-data.cam_xmat[cid].reshape(3,3)[:,2],group,1,-1,hit)
        assert distance < 0 or distance > .02


@pytest.mark.parametrize('name,threshold',[('wrist_left',9),('wrist_right',6)])
def test_rendered_closed_gripper_contour_matches_benchmark_framing(name,threshold):
    from pathlib import Path
    observations=json.loads((Path(__file__).parent/'fixtures/benchmark_camera_observations.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('startup').id)
    for side,angle in [('left',-.053707728),('right',-.074040855)]:
        d.qpos[m.joint(side+'_gripper').qposadr[0]]=angle
    cid=m.camera(name).id
    targets=[m.cam_bodyid[cid],m.body('Gripper_Finger_v1_1' if name=='wrist_left' else 'Gripper_Finger_v1_2').id]
    geoms=np.flatnonzero(np.isin(m.geom_bodyid,targets)&(m.geom_group==2))
    m.geom_group[~np.isin(np.arange(m.ngeom),geoms)]=3
    mujoco.mj_forward(m,d)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    with mujoco.Renderer(m,height=240,width=320) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d,camera=name,scene_option=option)
        segments=renderer.render()
    mask=np.isin(segments[:,:,0],geoms)&(segments[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)
    contour=observations['palm_contours'][name]
    errors=[]
    for x,y in zip(contour['x'],contour['y']):
        if (name=='wrist_left' and 110<x<150) or (name=='wrist_right' and not 90<=x<=270):
            continue
        pixels=np.flatnonzero(mask[:,x])
        errors.append(abs((int(pixels[0]) if len(pixels) else 260)-y))
    assert np.median(errors) < threshold
