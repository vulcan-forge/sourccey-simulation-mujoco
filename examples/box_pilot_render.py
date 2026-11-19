"""Render an already validated pilot with two independent GPU worker processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from examples.box_dataset_pilot import (OUTPUT,apply_camera_perturbations,file_hash,pilot_signature)
from examples.box_pickup import load_recording,scene_model,render
from examples.box_training import write_videos
from examples.box_workspace import write_json
from sourccey.rendering import provenance
from sourccey.simulation import Simulation


def render_episode(directory):
    directory=Path(directory)
    config=json.loads((directory/'scene.json').read_text(encoding='utf-8'))
    load_recording(directory,config)
    with np.load(directory/'transitions.npz',allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    model,_=scene_model(config,display_only=True)
    apply_camera_perturbations(model,config)
    display=Simulation(model=model,traction=False)
    if model.nq!=arrays['qpos'].shape[1]:
        raise ValueError('Display and physics layouts differ')
    run=SimpleNamespace(m=model,d=display.data,config=config)
    write_videos(run,arrays,directory)
    display.data.qpos[:]=arrays['qpos'][-1]
    mujoco.mj_forward(model,display.data)
    render(run,directory/'final.png')
    return config['episode_id']


def render_parallel(output,jobs=2,reuse=None):
    if not 1<=jobs<=2:
        raise ValueError('Use one or two GPU worker processes')
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    accepted=[e for e in manifest['episodes'] if e['eligible_for_imitation']]
    if len(accepted)!=manifest['plan']['count'] or manifest['pending']:
        raise ValueError('Finish the successful physics quota first')
    if provenance()!=manifest['rendering']:
        raise ValueError('Engine or renderer changed')
    for e in accepted:
        config=json.loads((output/e['episode_id']/'scene.json').read_text(encoding='utf-8'))
        if e['pilot_signature']!=pilot_signature(config):
            raise ValueError('Pilot physics source changed')
    if reuse:
        reuse=Path(reuse)
        other=json.loads((reuse/'manifest.json').read_text(encoding='utf-8'))
        if other['status']!='complete' or other['rendering']!=manifest['rendering']:
            raise ValueError('Reuse requires a completed matching render batch')
        cached={e['episode_id']:e for e in other['episodes'] if e['videos'] and e['eligible_for_imitation']}
        for e in accepted:
            previous=cached.get(e['episode_id'])
            if previous and previous['pilot_signature']==e['pilot_signature'] and not e['videos']:
                source,destination=reuse/e['episode_id'],output/e['episode_id']
                if file_hash(source/'transitions.npz')!=file_hash(destination/'transitions.npz'):
                    raise ValueError('Reuse recording mismatch')
                shutil.copytree(source/'videos',destination/'videos',dirs_exist_ok=True)
                for p in [*source.glob('cameras_*.png'),source/'final.png']:
                    shutil.copy2(p,destination/p.name)
                e['videos']=True
    signature=file_hash(__file__)
    if manifest.get('parallel_render_source_sha256',signature)!=signature:
        raise ValueError('Render coordinator changed; inspect existing outputs before continuing')
    manifest.update(status='rendering',parallel_render_source_sha256=signature)
    write_json(output/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(render_episode,output/e['episode_id']):e for e in accepted if not e['videos']}
        for future in as_completed(futures):
            episode=future.result()
            futures[future]['videos']=True
            write_json(output/'manifest.json',manifest)
            print(f'{sum(e["videos"] for e in accepted)}/{len(accepted)} rendered: {episode}',flush=True)
    manifest.update(status='complete',summary=dict(successful_episodes=len(accepted),
        attempted_episodes=len(manifest['episodes']),rejected_episodes=len(manifest['episodes'])-len(accepted),
        split_episodes={s:sum(e['split']==s for e in accepted) for s in ('train','validation','test')},
        nonterminal_frames=sum(e['frames']-1 for e in accepted),task=manifest['instruction'],
        policy='X-VLA',training_started=False))
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['summary'],indent=2),flush=True)
    return manifest


def render_follow(output,jobs=2,reuse=None):
    """Render completed successes while CPU collection continues; one manifest writer.

    The collector owns manifest.json until physics finishes. Render completion
    markers are independent, signed files; the final merge happens afterward.
    At most two GPU jobs are admitted at a time.
    """
    if not 1<=jobs<=2:
        raise ValueError('Use one or two GPU workers')
    output=Path(output)
    signature=file_hash(__file__)
    cached={}
    if reuse:
        reuse=Path(reuse)
        other=json.loads((reuse/'manifest.json').read_text(encoding='utf-8'))
        if other['status']!='complete' or other['rendering']!=provenance():
            raise ValueError('Reuse requires a matching completed format probe')
        cached={e['episode_id']:e for e in other['episodes'] if e['videos'] and e['eligible_for_imitation']}
    def marker(item):
        return dict(pilot_signature=item['pilot_signature'],render_source_sha256=signature,rendering=provenance())
    completed=set()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        running={}
        while True:
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            if manifest['rendering']!=provenance():
                raise ValueError('Renderer or engine changed')
            accepted=[e for e in manifest['episodes'] if e['eligible_for_imitation']]
            for e in accepted:
                name=e['episode_id']
                if name in completed or any(item['episode_id']==name for item in running.values()):
                    continue
                directory=output/name
                config=json.loads((directory/'scene.json').read_text(encoding='utf-8'))
                if e['pilot_signature']!=pilot_signature(config):
                    raise ValueError('Pilot source or scene changed')
                saved=directory/'render_completion.json'
                if saved.exists():
                    if json.loads(saved.read_text(encoding='utf-8'))!=marker(e):
                        raise ValueError('Cached render signature mismatch')
                    completed.add(name)
                    continue
                previous=cached.get(name)
                if previous and previous['pilot_signature']==e['pilot_signature']:
                    source=reuse/name
                    if file_hash(source/'transitions.npz')!=file_hash(directory/'transitions.npz'):
                        raise ValueError('Cached transition mismatch')
                    shutil.copytree(source/'videos',directory/'videos',dirs_exist_ok=True)
                    for p in [*source.glob('cameras_*.png'),source/'final.png']:
                        shutil.copy2(p,directory/p.name)
                    write_json(saved,marker(e))
                    completed.add(name)
                    continue
                if len(running)<jobs:
                    running[pool.submit(render_episode,directory)]=e
            if len(completed)==manifest['plan']['count'] and not running:
                if manifest['status'] not in ('physics_complete','complete') or manifest['pending']:
                    time.sleep(2)
                    continue
                for e in manifest['episodes']:
                    e['videos']=e['episode_id'] in completed
                manifest['parallel_render_source_sha256']=signature
                write_json(output/'manifest.json',manifest)
                break
            if running:
                done,_=wait(running,timeout=15,return_when=FIRST_COMPLETED)
                for future in done:
                    item=running.pop(future)
                    name=future.result()
                    write_json(output/name/'render_completion.json',marker(item))
                    completed.add(name)
                    print(f'{len(completed)}/{manifest["plan"]["count"]} camera episodes complete: {name}',flush=True)
            else:
                time.sleep(2)
    return render_parallel(output,jobs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--reuse',type=Path,help='Reuse identical signed episodes from a completed small format probe')
    parser.add_argument('--follow',action='store_true',help='Render successes concurrently with a running physics collector')
    args=parser.parse_args()
    (render_follow if args.follow else render_parallel)(args.output,args.jobs,args.reuse)
