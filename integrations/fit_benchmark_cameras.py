"""Offline camera comparisons against sampled benchmark frames and telemetry."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution,least_squares

from examples.box_pickup import load_scene,scene_model,options
from sourccey.simulation import Simulation
from sourccey.build import ROOT

CAMERAS=('front_left','front_right','wrist_left','wrist_right')


def setup(case):
    model,_=scene_model(load_scene(),display_only=True)
    for name in ('pickup_box_geom','box_sealing_tape'):
        model.geom_rgba[model.geom(name).id,3]=0
    sim=Simulation(model=model,traction=False)
    mapping=json.loads((ROOT/'artifacts/robot_joint_mapping.json').read_text())
    for name,entry in mapping['joints'].items():
        fit=entry['derived_affine']
        sim.data.qpos[model.joint(name).qposadr[0]]=case['state'][name+'.pos']*fit['slope']+fit['offset']
    mujoco.mj_forward(model,sim.data)
    return model,sim.data


def set_cameras(model,poses):
    for name in CAMERAS:
        cid=model.camera(name).id
        model.cam_pos[cid]=poses[name]['pos']
        model.cam_quat[cid]=poses[name]['quat_wxyz']
        model.cam_fovy[cid]=poses[name]['fovy_deg']


def render(model,data,renderer,name):
    mujoco.mj_camlight(model,data)
    renderer.update_scene(data,camera=name,scene_option=options())
    return Image.fromarray(renderer.render())


def compare(cases,poses,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for case in cases:
        model,data=setup(case)
        sheet=Image.new('RGB',(1280,262*(1+len(poses))),(20,23,27))
        draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(CAMERAS):
            sheet.paste(Image.open(case['images'][name]['path']),(i*320,22))
            draw.text((i*320+4,4),'REAL '+name,fill='white')
        with mujoco.Renderer(model,height=240,width=320) as renderer:
            for row,(label,config) in enumerate(poses.items(),start=1):
                set_cameras(model,config)
                model.vis.map.znear=config.get('_meta',{}).get('render_near_fraction',.01)
                for i,name in enumerate(CAMERAS):
                    sheet.paste(render(model,data,renderer,name),(i*320,row*262+22))
                    draw.text((i*320+4,row*262+4),label+' '+name,fill='white')
        sheet.save(output/f'{case["dataset"]}_{case["frame"]:05d}.png')


def fit_wrists(case,poses,contours,output):
    """Approximate closed gripper framing; no physical extrinsic recovery claim."""
    model,data=setup(case)
    set_cameras(model,poses)
    model.vis.map.znear=.001
    report={}
    with mujoco.Renderer(model,height=240,width=320) as renderer:
        renderer.enable_segmentation_rendering()
        for name in ('wrist_left','wrist_right'):
            cid=model.camera(name).id
            base=Rotation.from_quat(np.asarray(poses[name]['quat_wxyz'])[[1,2,3,0]])
            body=model.cam_bodyid[cid]
            finger=model.body('Gripper_Finger_v1_1' if name=='wrist_left' else 'Gripper_Finger_v1_2').id
            targets=np.array([i for i in range(model.ngeom) if model.geom_bodyid[i] in (body,finger)])
            xs=np.asarray(contours[name]['x'])
            ys=np.asarray(contours[name]['y'])
            valid=((xs<=110)|(xs>=150)) if name=='wrist_left' else ((xs>=90)&(xs<=270))
            xs,ys=xs[valid],ys[valid]
            def evaluate(angles,details=False):
                q=(base*Rotation.from_euler('xyz',angles,degrees=True)).as_quat()
                model.cam_quat[cid]=q[[3,0,1,2]]
                mujoco.mj_camlight(model,data)
                renderer.update_scene(data,camera=name,scene_option=options())
                segmentation=renderer.render()
                mask=np.isin(segmentation[:,:,0],targets)&(segmentation[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)
                values=[]
                for x in xs:
                    pixels=np.flatnonzero(mask[:,x])
                    values.append(int(pixels[0]) if len(pixels) else 260)
                values=np.asarray(values)
                error=values-ys
                # Robust curve agreement plus a weak nominal-angle preference.
                score=float(np.mean(np.minimum(error**2,60*np.abs(error)))+.035*np.dot(angles,angles))
                return (score,values,error,q[[3,0,1,2]]) if details else score
            initial=evaluate([0,0,0],True)
            fit=differential_evolution(evaluate,[(-20,10),(-18,18),(-20,20)],seed=20261116,
                popsize=7,maxiter=25,polish=False,tol=.005)
            final=evaluate(fit.x,True)
            report[name]=dict(camera_frame_euler_delta_xyz_deg=fit.x.tolist(),quat_wxyz=final[3].tolist(),
                initial_median_contour_error_px=float(np.median(np.abs(initial[2]))),
                median_contour_error_px=float(np.median(np.abs(final[2]))),
                x=xs.tolist(),real_y=ys.tolist(),render_y=final[1].tolist(),
                position_fixed_to_nominal_cad=True,note='Bounded image-framing approximation from dark foreground contour medians; reflections, cloth and model mismatch limit accuracy.')
            print(name,report[name],flush=True)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


def head_orientation(yaw,pitch,roll):
    yaw,pitch=np.deg2rad([yaw,pitch])
    forward=np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(-forward,right)
    return np.column_stack([right,up,-forward])@Rotation.from_euler('z',roll,degrees=True).as_matrix()


def stereo_predict(parameters,points):
    focal,delta_pitch,roll_left,roll_right=parameters
    left=head_orientation(15,-20-delta_pitch/2,roll_left)
    right=head_orientation(-15,-20+delta_pitch/2,roll_right)
    rays=np.column_stack([(points[:,0]-160)/focal,-(points[:,1]-120)/focal,-np.ones(len(points))])
    rays=rays@left.T@right
    return np.column_stack([160+focal*rays[:,0]/-rays[:,2],120-focal*rays[:,1]/-rays[:,2]])


def fit_heads(points):
    left=np.asarray([p['left'] for p in points]);right=np.asarray([p['right'] for p in points])
    train=np.asarray(['black-' in p['dataset'] for p in points])
    result=least_squares(lambda p:(stereo_predict(p,left[train])-right[train]).ravel(),
        [325,0,0,0],bounds=([150,-5,-10,-10],[700,5,10,10]),loss='soft_l1',f_scale=1.5)
    errors=np.linalg.norm(stereo_predict(result.x,left)-right,axis=1)
    wide=[120/np.tan(np.deg2rad(92.2042275)/2),0,0,0]
    previous=np.linalg.norm(stereo_predict(wide,left)-right,axis=1)
    report=dict(parameters=result.x.tolist(),focal_px=float(result.x[0]),
        fovy_deg=float(2*np.rad2deg(np.arctan(120/result.x[0]))),train_pairs=int(train.sum()),held_out_pairs=int((~train).sum()),
        train_median_error_px=float(np.median(errors[train])),held_out_median_error_px=float(np.median(errors[~train])),
        previous_120d_held_out_median_error_px=float(np.median(previous[~train])),
        assumptions='Centered shared pinhole, fixed +/-15 degree yaw and mean -20 degree pitch. Distant-room points approximate infinity. Absolute intrinsics, baseline, distortion and physical extrinsics are not independently calibrated.',heads={})
    focal,delta,rl,rr=result.x
    for name,yaw,pitch,roll in [('front_left',15,-20-delta/2,rl),('front_right',-15,-20+delta/2,rr)]:
        matrix=Rotation.from_euler('z',-45,degrees=True).as_matrix()@head_orientation(yaw,pitch,roll)
        q=Rotation.from_matrix(matrix).as_quat()
        report['heads'][name]=dict(quat_wxyz=q[[3,0,1,2]].tolist(),yaw_deg=yaw,pitch_deg=pitch,roll_deg=roll)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=Path('artifacts/benchmark_camera_fit/samples.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/benchmark_camera_fit/comparisons'))
    parser.add_argument('--fit',action='store_true',help='Recompute provisional head/gripper fit reports without changing the model')
    args=parser.parse_args()
    cases=json.loads(args.samples.read_text())['cases']
    selected=[c for c in cases if c['dataset'] in ('sourccey-benchmark-black-flat-fixed-000','sourccey-benchmark-white-flat-rotated-000')]
    active=json.loads((ROOT/'models/source/camera_poses.json').read_text())
    specified=json.loads((ROOT/'models/source/camera_poses.specified_20261116.json').read_text())
    if args.fit:
        args.output.mkdir(parents=True,exist_ok=True)
        head=fit_heads(json.loads((args.samples.parent/'stereo_points.json').read_text()))
        (args.output/'head_fit.json').write_text(json.dumps(head,indent=2)+'\n')
        candidate=copy.deepcopy(specified)
        for n in CAMERAS: candidate[n]['fovy_deg']=head['fovy_deg']
        closed=next(c for c in selected if c['dataset']=='sourccey-benchmark-black-flat-fixed-000' and c['frame']==0)
        fit_wrists(closed,candidate,json.loads((args.samples.parent/'palm_contours.json').read_text()),args.output/'wrist_fit.json')
    compare(selected,{'before-literal-120D':specified,'benchmark-fit':active},args.output)
