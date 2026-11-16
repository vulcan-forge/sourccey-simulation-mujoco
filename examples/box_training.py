"""Seeded box-pickup curriculum, synchronized demonstrations and camera review.

No hardware imports or connections. Simulated object poses are privileged labels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sourccey.build import ROOT, ROLES
from sourccey.simulation import Simulation
from sourccey.rendering import create_renderer, provenance


CAMERAS = ('front_left', 'front_right', 'wrist_left', 'wrist_right')
COLORS = ((60, 225, 210), (95, 220, 95), (250, 175, 60), (200, 125, 250))
FPS = 50  # Exact divisor of the 500 Hz physics clock; all teacher targets align.
SIZE = (320, 240)
OUTPUT = ROOT/'artifacts/box_pickup_b'
STATE_NAMES = tuple('left_'+role+'.pos' for role in ROLES)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instances(base, count=10, seed=20261116):
    """Stratified small yaw changes, independent of appearance/physics seeds."""
    if not 1 <= count <= 100:
        raise ValueError('Instance count must be 1..100')
    rng = np.random.default_rng(seed)
    edges = np.linspace(-12, 12, count+1)
    result = []
    for i in range(count):
        config = copy.deepcopy(base)
        episode_seed = int(rng.integers(0, 2**31))
        visual = np.random.default_rng(episode_seed)
        config.update(test='B', episode_id=f'episode_{i:03d}', seed=episode_seed,
                      split='train' if i < max(1, int(.8*count)) else 'test')
        config['box_pose']['yaw_deg'] = float(rng.uniform(edges[i]+.05, edges[i+1]-.05))
        config['box_pose']['orientation'] = 'Upright; yaw rotates the measured width/depth axes about world +Z'
        config['object']['mass_kg'] = float(visual.uniform(.115, .125))
        config['grasp']['sliding_friction'] = float(visual.uniform(.35, .40))
        config['table']['sliding_friction'] = float(visual.uniform(.35, .45))
        config['domain_randomization'] = dict(seed=episode_seed,
            box_rgb=visual.uniform([.40,.27,.12],[.85,.70,.48]).tolist(),
            table_rgb=visual.uniform(.18,.60,3).tolist(),
            arm_rgb=visual.uniform(.50,.95,3).tolist(),
            hand_rgb=visual.uniform(.15,.70,3).tolist(),
            floor_rgb=visual.uniform(.14,.40,3).tolist(),
            texture_style=int(visual.integers(0,3)),
            light_direction=visual.uniform([-.7,-.7,-1.5],[.7,.7,-.7]).tolist(),
            light_diffuse=visual.uniform(.55,.85,3).tolist(),
            light_ambient=visual.uniform(.15,.30,3).tolist(),
            exposure=float(visual.uniform(.96,1.04)),
            white_balance=visual.uniform(.96,1.04,3).tolist(), noise_std=float(visual.uniform(.3,1.2)),
            camera_pose_randomized=False,
            physics_note='Estimated mass 115..125 g, hand sliding friction .35...40, table .35...45; not measured hardware variation')
        result.append(config)
    return result


def randomize_appearance(spec, parameters):
    """Materials and lighting only; collision geometry is never changed here."""
    seed = parameters['seed']
    pbr = any(material.name.startswith('pbr_') for material in spec.materials)
    for index, key in enumerate(('box_rgb', 'table_rgb', 'arm_rgb', 'hand_rgb', 'floor_rgb')):
        color = np.asarray(parameters[key])
        style = parameters['texture_style']
        yy,xx = np.mgrid[:64,:64]
        if pbr:
            factor = 0.94+0.06*np.sin(xx*.7+np.sin(yy*.15)) if index == 1 else np.ones((64,64))
        elif style == 0:
            factor = np.where((xx//8+yy//8)%2,1.,.70)
        elif style == 1:
            factor = .70+.30*xx/63
        else:
            factor = np.ones((64,64))
        rng = np.random.default_rng(np.random.SeedSequence([seed,index]))
        pixels = np.clip(color[None,None,:]*255*factor[:,:,None]+rng.normal(0,3,(64,64,3)),0,255).astype(np.uint8)
        texture = spec.add_texture(name=f'episode_texture_{index}', type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=64, height=64, nchannel=3, data=pixels.ravel())
        material = spec.add_material(name=f'episode_material_{index}', rgba=[1,1,1,1],
            specular=.12+.04*((seed+index)%5), shininess=.15, texuniform=True,
            texrepeat=[3+index,3+index])
        material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    for geom in spec.geoms:
        name = geom.name
        index = (0 if name == 'pickup_box_geom' else 1 if name == 'box_test_table'
                 else 3 if name.startswith('visual_Gripper_') else 2 if name.startswith('visual_')
                 and geom.rgba[0] > .4 else 4 if name == 'floor' else None)
        if index is not None:
            geom.material = f'episode_material_{index}'
            geom.rgba = [1,1,1,1]
    for light in spec.lights:
        light.dir = parameters['light_direction']
        light.diffuse = parameters['light_diffuse']
        light.ambient = parameters['light_ambient']
    if any(material.name.startswith('pbr_') for material in spec.materials):
        from sourccey.appearance import randomize_pbr
        randomize_pbr(spec, parameters)


def photometric(image, config, camera_index, frame_index):
    p = config.get('domain_randomization')
    if p is None:
        return image.copy()
    rng = np.random.default_rng(np.random.SeedSequence([p['seed'], camera_index, frame_index]))
    result = image.astype(np.float32)*p['exposure']*np.asarray(p['white_balance'])
    result += rng.normal(0,p['noise_std'],result.shape)
    return np.clip(np.rint(result),0,255).astype(np.uint8)


def camera_manifest(model, data):
    source = ROOT/'models/source/camera_poses.json'
    references = json.loads(source.read_text(encoding='utf-8'))
    result = dict(source=str(source.relative_to(ROOT)), source_sha256=digest(source),
        physical_calibration_verified=False,
        note='Camera mounts and ideal pinhole optics follow the current source specification. Physical intrinsics/extrinsics remain unverified; no physical calibration is inferred from rendering.',
        resolution_wh=list(SIZE), cameras={})
    for name in CAMERAS:
        cid = model.camera(name).id
        focal = SIZE[1]/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
        result['cameras'][name] = dict(body=model.body(model.cam_bodyid[cid]).name,
            position_body_m=model.cam_pos[cid].tolist(), quaternion_body_wxyz=model.cam_quat[cid].tolist(),
            fovy_deg=float(model.cam_fovy[cid]), K=[[float(focal),0,SIZE[0]/2],[0,float(focal),SIZE[1]/2],[0,0,1]],
            distortion_model='ideal pinhole; physical lens distortion unmeasured',
            optical_axes='OpenCV: +X right, +Y down, +Z forward; MuJoCo camera axes converted by diag(1,-1,-1)',
            reference_status=references[name]['calibration_status'])
    return result


class EpisodeRecorder:
    """Observe BEFORE integration, after the teacher sets the upcoming targets."""
    def __init__(self, run):
        self.run = run
        self.rows = []
        self.every = round(1/(FPS*run.m.opt.timestep))
        if abs(self.every*run.m.opt.timestep-1/FPS) > 1e-10:
            raise ValueError('Recording rate must divide the physics clock exactly')
        mapping_path = ROOT/'artifacts/robot_joint_mapping.json'
        self.mapping_source_sha256 = digest(mapping_path)
        mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
        self.fits = [mapping['joints'][name[:-4]]['derived_affine'] for name in STATE_NAMES]
        self.qp = [run.m.joint(n[:-4]).qposadr[0] for n in STATE_NAMES]
        self.qv = [run.m.joint(n[:-4]).dofadr[0] for n in STATE_NAMES]
        self.aids = [run.m.actuator(n[:-4]).id for n in STATE_NAMES]
        self.capture(run)

    def normalized(self, values):
        return np.asarray([(v-f['offset'])/f['slope'] for v,f in zip(values,self.fits)],dtype=np.float32)

    def capture(self, run, terminal=False):
        if not terminal and run.steps % self.every:
            return
        # Kinematics updates camera/body transforms to exactly this qpos; no
        # contact solve or integration is performed during sensor acquisition.
        mujoco.mj_kinematics(run.m,run.d)
        mujoco.mj_comPos(run.m,run.d)
        mujoco.mj_camlight(run.m,run.d)
        position = run.d.qpos[self.qp].copy()
        command = run.d.ctrl[self.aids].copy()
        action = self.normalized(command)
        low = np.array([-100]*5+[0])
        if np.any(action < low-.01) or np.any(action > 100+.01):
            raise ValueError('Teacher target exceeds the existing normalized hardware mapping')
        row = dict(timestamp=float(run.d.time), state=self.normalized(position), action=np.clip(action,low,100).astype(np.float32),
            joint_rad=position, joint_velocity_rad_s=run.d.qvel[self.qv].copy(), command_rad=command,
            qpos=run.d.qpos.copy(), qvel=run.d.qvel.copy(), ctrl=run.d.ctrl.copy(), phase=run.phase,
            box_pose=np.r_[run.d.xpos[run.box],run.d.xquat[run.box]].copy(),
            camera_position=np.asarray([run.d.cam_xpos[run.m.camera(n).id] for n in CAMERAS]),
            camera_rotation=np.asarray([run.d.cam_xmat[run.m.camera(n).id].reshape(3,3) for n in CAMERAS]))
        if self.rows and abs(self.rows[-1]['timestamp']-row['timestamp']) < 1e-9:
            self.rows[-1] = row  # Command change at an already-captured instant.
        else:
            self.rows.append(row)

    def finish(self, report, directory):
        self.capture(self.run,terminal=True)
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        arrays = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        n = len(self.rows)
        arrays['next_state'] = np.vstack([arrays['state'][1:],arrays['state'][-1:]])
        arrays['done'] = np.arange(n) == n-1
        arrays['success'] = arrays['done'] & bool(report['passed'])
        arrays['valid_action'] = ~arrays['done']
        np.savez_compressed(directory/'transitions.npz',**arrays)
        metadata = dict(format='sourccey-box-episodes-v1', episode_id=self.run.config.get('episode_id','episode_000'),
            fps=FPS, frames=n, simulated_duration_s=float(arrays['timestamp'][-1]),
            names=list(STATE_NAMES), units='normalized hardware joint/gripper position targets',
            mapping_sha256=self.mapping_source_sha256, mapping_status='Existing source-confirmed left-arm affine mapping; absolute physical grasp calibration remains pending',
            action_semantics='Desired joint/gripper target in force from observation time until the next control sample; final row terminal, not a training action',
            learner_features=['observation.state','observation.images.'+'front_left','observation.images.front_right',
                              'observation.images.wrist_left','observation.images.wrist_right'],
            privileged_labels=['box_pose','camera_position','camera_rotation','phase','qpos','qvel','ctrl'],
            passed=bool(report['passed']), split=self.run.config.get('split','train'),
            eligible_for_imitation=bool(report['passed']),
            rendering=provenance(),
            camera_calibration=camera_manifest(self.run.m,self.run.d), scene=self.run.config)
        (directory/'episode.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return arrays,metadata


def render_camera_frames(model, data, renderer, config, index):
    from examples.box_pickup import options
    result = {}
    if hasattr(renderer, 'render_many'):
        frames = renderer.render_many(data, CAMERAS, options())
        return {name: photometric(frames[name],config,ci,index) for ci,name in enumerate(CAMERAS)}
    for ci,name in enumerate(CAMERAS):
        renderer.update_scene(data,camera=name,scene_option=options())
        result[name] = photometric(renderer.render(),config,ci,index)
    return result


def mosaic(images, width=320):
    height = width*3//4
    canvas = Image.new('RGB',(width*2,(height+22)*2),(15,18,22))
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(CAMERAS):
        x,y=(i%2)*width,(i//2)*(height+22)
        canvas.paste(Image.fromarray(images[name]).resize((width,height)),(x,y+22))
        draw.text((x+6,y+4),name.replace('_',' '),fill=COLORS[i])
    return np.asarray(canvas)


def write_videos(run, arrays, directory):
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('ffmpeg is required to encode synchronized four-camera video')
    directory = Path(directory)
    video_dir = directory/'videos'
    video_dir.mkdir(exist_ok=True)
    processes = {}
    logs = []
    creation = subprocess.CREATE_NO_WINDOW if hasattr(subprocess,'CREATE_NO_WINDOW') else 0
    try:
        for name in CAMERAS:
            log = open(video_dir/f'{name}.log','w',encoding='utf-8')
            logs.append(log)
            processes[name] = subprocess.Popen([executable,'-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{SIZE[0]}x{SIZE[1]}',
                '-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
                '-crf','18','-pix_fmt','yuv420p','-threads','1',str(video_dir/f'{name}.mp4')],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=creation)
        with create_renderer(run.m,height=SIZE[1],width=SIZE[0]) as renderer:
            for i,qpos in enumerate(arrays['qpos']):
                run.d.qpos[:] = qpos
                mujoco.mj_kinematics(run.m,run.d)
                mujoco.mj_comPos(run.m,run.d)
                mujoco.mj_camlight(run.m,run.d)
                run.d.time = arrays['timestamp'][i]
                images = render_camera_frames(run.m,run.d,renderer,run.config,i)
                for name,image in images.items():
                    processes[name].stdin.write(image.tobytes())
                if i in (0,len(arrays['qpos'])//2,len(arrays['qpos'])-1):
                    Image.fromarray(mosaic(images)).save(directory/f'cameras_{i:04d}.png')
                if i % 250 == 0:
                    print(f'  cameras: {i}/{len(arrays["qpos"])} frames',flush=True)
        for process in processes.values():
            process.stdin.close()
        for name,process in processes.items():
            if process.wait(timeout=60):
                raise RuntimeError(f'Video encoding failed for {name}; see its log')
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def camera_frusta(scene, model, data):
    """Render-only sensor frames; excluded from learning images and physics."""
    scene.ngeom = 0
    depth = .10
    for ci,name in enumerate(CAMERAS):
        cid = model.camera(name).id
        origin = data.cam_xpos[cid]
        rotation = data.cam_xmat[cid].reshape(3,3)
        y = np.tan(np.deg2rad(model.cam_fovy[cid])/2)*depth
        corners = np.array([[-y*4/3,-y,-depth],[y*4/3,-y,-depth],[y*4/3,y,-depth],[-y*4/3,y,-depth]])@rotation.T+origin
        for i in range(4):
            for start,end in ((origin,corners[i]),(corners[i],corners[(i+1)%4])):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.r_[np.array(COLORS[ci])/255,1.])
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.,start,end)
                scene.ngeom += 1


def review(output=OUTPUT, speed=1.):
    from examples.box_pickup import camera, scene_model, load_recording
    from mujoco import viewer
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    state = dict(index=0,position=0.,paused=False,change=True,frusta=True,detail=False,cameras=True)
    def key(code):
        if code in (ord('P'),ord('p')): state['paused'] = not state['paused']
        if code in (ord('R'),ord('r')): state.update(position=0.,paused=False)
        if code in (ord('N'),ord('n'),262): state.update(index=(state['index']+1)%len(manifest['episodes']),change=True)
        if code in (ord('B'),ord('b'),263): state.update(index=(state['index']-1)%len(manifest['episodes']),change=True)
        if code in (ord('F'),ord('f')): state['frusta'] = not state['frusta']
        if code in (ord('C'),ord('c')): state['cameras'] = not state['cameras']
        if code in (ord('1'),ord('2')): state['detail'] = code == ord('2')
    # Each selected instance loads its own materials and lights. Playback uses
    # visible CAD only; success comes from the saved full-collision physics run.
    while True:
        item = manifest['episodes'][state['index']]
        directory = output/item['episode_id']
        config = json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        model,_ = scene_model(config,display_only=True)
        sim = Simulation(model=model,traction=False)
        recording,report = load_recording(directory,config)
        state.update(change=False,position=0.)
        last = time.perf_counter()
        next_camera = 0.
        renderer = create_renderer(model,height=SIZE[1],width=SIZE[0])
        try:
            with viewer.launch_passive(model,sim.data,key_callback=key,show_left_ui=False,show_right_ui=False) as window:
                window.opt.geomgroup[3] = 0
                window.opt.sitegroup[:] = 0
                last_detail = None
                while window.is_running() and not state['change']:
                    now = time.perf_counter()
                    if not state['paused']:
                        state['position'] = min(float(recording['time'][-1]),state['position']+(now-last)*speed)
                    last = now
                    i = min(max(0,int(np.searchsorted(recording['time'],state['position'],side='right'))-1),len(recording['time'])-1)
                    sim.data.qpos[:] = recording['qpos'][i]
                    sim.data.time = recording['time'][i]
                    mujoco.mj_kinematics(model,sim.data)
                    mujoco.mj_comPos(model,sim.data)
                    mujoco.mj_camlight(model,sim.data)
                    if last_detail != state['detail']:
                        cam = camera(state['detail'])
                        window.cam.lookat[:] = cam.lookat
                        window.cam.distance,window.cam.azimuth,window.cam.elevation = cam.distance,cam.azimuth,cam.elevation
                        last_detail = state['detail']
                    if now >= next_camera:
                        if state['cameras']:
                            image = mosaic(render_camera_frames(model,sim.data,renderer,config,i),width=240)
                            viewport = window.viewport
                            window.set_images((mujoco.MjrRect(max(0,viewport.width-image.shape[1]-10),30,image.shape[1],image.shape[0]),image))
                        else:
                            window.clear_images()
                        next_camera = now+.1
                    with window.lock():
                        if state['frusta']: camera_frusta(window.user_scn,model,sim.data)
                        else: window.user_scn.ngeom = 0
                    window.set_texts([
                        (mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         f'Box test B | {state["index"]+1}/{len(manifest["episodes"])} | yaw {config["box_pose"]["yaw_deg"]:+.1f} deg\n'
                         f'{recording["time"][i]:.1f}s | {recording["phase"][i]} | {"PASS" if report["passed"] else "FAIL"}\n'
                         'Camera poses: provisional; not physically calibrated',''),
                        (mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                         'N/B: next/previous   R: replay   P: pause\n1/2: overview/gripper   F: camera frusta   C: four feeds','')])
                    window.sync()
                    time.sleep(.012)
                if not state['change']:
                    break
        finally:
            renderer.close()


def record_instance(config, directory):
    """CPU-only worker; no graphics contexts, network connections or shared models."""
    from examples.box_pickup import Pickup,save_run
    directory = Path(directory)
    run = Pickup(config)
    recorder = EpisodeRecorder(run)
    run.before_step = recorder.capture
    try:
        report = run.run()
    except (RuntimeError,ValueError) as exc:
        report = run.report()
        report.update(passed=False,error=str(exc))
    save_run(run,report,directory)
    _,metadata = recorder.finish(report,directory)
    return report,metadata


def generate(count=10, seed=20261116, output=OUTPUT, videos=True, pbr=False, jobs=1):
    from examples.box_pickup import load_scene,render,scene_model
    from types import SimpleNamespace
    from contextlib import ExitStack
    from concurrent.futures import ProcessPoolExecutor
    if not 1 <= jobs <= 4:
        raise ValueError('CPU worker count must be 1..4; collision models consume substantial memory')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    configs = instances(load_scene(),count,seed)
    if pbr:
        for config in configs:
            config['appearance'] = dict(profile='pbr', calibration='estimated procedural appearance')
    if (output/'manifest.json').exists():
        raise FileExistsError('Choose a new output directory to preserve the previous batch and its failures')
    manifest = dict(version=1,test='B',seed=seed,episodes=[],simulation_only=True,
        instruction='Pick up the cardboard box and hold it above the table.',fps=FPS,
        camera_names=list(CAMERAS),state_names=list(STATE_NAMES),physical_calibration_verified=False,
        rendering=provenance())
    for config in configs:
        directory = output/config['episode_id']
        if directory.exists():
            raise FileExistsError(f'Episode already exists: {directory}')
        directory.mkdir()
        (directory/'scene.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=jobs)) if jobs > 1 else None
        futures = ([executor.submit(record_instance,c,output/c['episode_id']) for c in configs]
                   if executor else None)
        for index,config in enumerate(configs):
            directory = output/config['episode_id']
            print(f'Instance {index+1}/{count}: yaw {config["box_pose"]["yaw_deg"]:+.2f}, seed {config["seed"]}',flush=True)
            report,metadata = futures[index].result() if futures else record_instance(config,directory)
            with np.load(directory/'transitions.npz',allow_pickle=False) as archive:
                arrays = {key:archive[key].copy() for key in archive.files}
            display_model,_ = scene_model(config,display_only=True)
            display = Simulation(model=display_model,traction=False)
            if display_model.nq != arrays['qpos'].shape[1]:
                raise RuntimeError('Display and physics joint layouts differ')
            display.data.qpos[:] = arrays['qpos'][-1]
            mujoco.mj_forward(display_model,display.data)
            displayed = SimpleNamespace(m=display_model,d=display.data,config=config)
            render(displayed,directory/'final.png')
            render(displayed,directory/'grasp_detail.png',detail=True)
            if videos:
                write_videos(displayed,arrays,directory)
            # Also capture the initial/final four views for inspection without a player.
            manifest['episodes'].append(dict(episode_id=config['episode_id'],seed=config['seed'],
                yaw_deg=config['box_pose']['yaw_deg'],passed=report['passed'],split=config['split'],
                frames=metadata['frames'],eligible_for_imitation=metadata['eligible_for_imitation'],videos=videos,
                minimum_hold_clearance_mm=report['minimum_bottom_clearance_during_hold_mm']))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
            print(f'  {"PASS" if report["passed"] else "FAIL"}: clearance {report["minimum_bottom_clearance_during_hold_mm"]:.1f} mm',flush=True)
            del display,displayed,display_model
    contact_sheet(output,manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


def contact_sheet(output, manifest):
    output = Path(output)
    sheet = Image.new('RGB',(960,280*((len(manifest['episodes'])+2)//3)),(22,25,30))
    draw = ImageDraw.Draw(sheet)
    for index,item in enumerate(manifest['episodes']):
        x,y=(index%3)*320,(index//3)*280
        with Image.open(output/item['episode_id']/'grasp_detail.png') as image:
            sheet.paste(image.resize((320,240)),(x,y+40))
        draw.text((x+7,y+5),f'{index+1:02d}: yaw {item["yaw_deg"]:+.2f} deg | {"PASS" if item["passed"] else "FAIL"}',fill='white')
        draw.text((x+7,y+22),f'{item["split"]} | seed {item["seed"]}',fill=(150,200,220))
    sheet.save(output/'instances.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--review',action='store_true',help='Open existing instances with four onboard camera feeds')
    parser.add_argument('--view',action='store_true',help='Review the generated batch after recording')
    parser.add_argument('--no-video',action='store_true',help='Physics-only diagnostic batch; not an image training dataset')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--renderer',choices=('classic','filament'),help='Camera/image renderer; default SOURCCEY_RENDERER or classic')
    parser.add_argument('--pbr',action='store_true',help='Procedural materials with seeded roughness variation')
    parser.add_argument('--jobs',type=int,default=1,choices=range(1,5),help='Independent CPU physics workers; GPU rendering stays in the parent process')
    args = parser.parse_args()
    if args.renderer:
        import os
        os.environ['SOURCCEY_RENDERER'] = args.renderer
    if not np.isfinite(args.speed) or args.speed <= 0: parser.error('Speed must be positive and finite')
    if not args.review:
        generate(args.count,args.seed,args.output,videos=not args.no_video,pbr=args.pbr,jobs=args.jobs)
    if args.review or args.view:
        review(args.output,args.speed)


if __name__ == '__main__':
    main()
