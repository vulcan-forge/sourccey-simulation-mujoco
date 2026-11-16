"""Export recorded box episodes through the installed LeRobot v3 writer.

Run with the existing LeRobot Python environment. No simulation or hardware
imports, no model training, and no uploads. Failed attempts stay in the source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CAMERAS = ('front_left','front_right','wrist_left','wrist_right')


def validate_episode(directory, check_video=True):
    directory = Path(directory)
    metadata = json.loads((directory/'episode.json').read_text(encoding='utf-8'))
    with np.load(directory/'transitions.npz',allow_pickle=False) as archive:
        arrays = {name:archive[name].copy() for name in archive.files}
    n,fps = metadata['frames'],metadata['fps']
    if n < 2 or arrays['state'].shape != (n,6) or arrays['action'].shape != (n,6):
        raise ValueError('Invalid six-joint state/action layout')
    if not np.allclose(arrays['timestamp'],np.arange(n)/fps,atol=1e-7,rtol=0):
        raise ValueError('Nonuniform observation/action timestamps')
    for key in ('state','action','next_state','qpos','qvel','ctrl','box_pose'):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f'Nonfinite recorded values: {key}')
    if not np.array_equal(arrays['next_state'][:-1],arrays['state'][1:]):
        raise ValueError('Next-state alignment mismatch')
    if np.any(arrays['action'] < np.array([-100]*5+[0])-.01) or np.any(arrays['action'] > 100.01):
        raise ValueError('Out-of-range hardware target')
    if np.count_nonzero(arrays['done']) != 1 or not arrays['done'][-1] or arrays['valid_action'][-1]:
        raise ValueError('Terminal transition is not correctly masked')
    if not np.all(arrays['valid_action'][:-1]):
        raise ValueError('Missing nonterminal action')
    if check_video:
        import av
        for camera in CAMERAS:
            with av.open(str(directory/'videos'/f'{camera}.mp4')) as container:
                stream = container.streams.video[0]
                if stream.frames != n or stream.width != 320 or stream.height != 240 or float(stream.average_rate) != fps:
                    raise ValueError(f'Video frame count, size or rate mismatch: {camera}')
    return arrays,metadata


def export(source, output, repo_id='local/sourccey-box-pickup-b'):
    import av
    av.logging.set_level(av.logging.ERROR)
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source,output = Path(source),Path(output)
    if output.exists():
        raise FileExistsError('Choose a new export directory; existing datasets are not overwritten')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    selected = [item for item in manifest['episodes'] if item['eligible_for_imitation']]
    selected.sort(key=lambda item:(item['split']!='train',item['episode_id']))
    if not selected:
        raise ValueError('No successful demonstrations to export')
    features = {
        'observation.state':dict(dtype='float32',shape=(6,),names=manifest['state_names']),
        'action':dict(dtype='float32',shape=(6,),names=manifest['state_names']),
        **{'observation.images.'+camera:dict(dtype='video',shape=(240,320,3),names=['height','width','channels']) for camera in CAMERAS}}
    dataset = LeRobotDataset.create(repo_id=repo_id,root=output,fps=manifest['fps'],features=features,
        robot_type='sourccey_sim_left_arm',use_videos=True,streaming_encoding=True,
        video_backend='pyav',encoder_threads=2,rgb_encoder=RGBEncoderConfig(vcodec='h264',preset='fast',crf=20))
    provenance = dict(source=str(source.resolve()),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        synthetic=True,physical_calibration_verified=False,
        units='normalized hardware joint/gripper position targets',
        action_semantics='Targets set after the synchronized observation; terminal row excluded',
        privileged_pose_in_policy_features=False,episodes=[],
        excluded_failed_episodes=[i['episode_id'] for i in manifest['episodes'] if not i['eligible_for_imitation']])
    try:
        for index,item in enumerate(selected):
            directory = source/item['episode_id']
            arrays,metadata = validate_episode(directory)
            containers = {c:av.open(str(directory/'videos'/f'{c}.mp4')) for c in CAMERAS}
            streams = {c:iter(container.decode(video=0)) for c,container in containers.items()}
            try:
                for frame_index in range(len(arrays['state'])-1):
                    frame = {'observation.state':arrays['state'][frame_index],
                             'action':arrays['action'][frame_index],
                             'task':manifest['instruction']}
                    for camera in CAMERAS:
                        frame['observation.images.'+camera] = next(streams[camera]).to_ndarray(format='rgb24')
                    dataset.add_frame(frame)
                dataset.save_episode()
            finally:
                for container in containers.values(): container.close()
            provenance['episodes'].append(dict(episode_index=index,source_episode=item['episode_id'],split=item['split'],
                frames=len(arrays['state'])-1,source_transitions_sha256=hashlib.sha256((directory/'transitions.npz').read_bytes()).hexdigest(),
                camera_calibration=metadata['camera_calibration'],scene=metadata['scene']))
            print(f'Exported {index+1}/{len(selected)}: {item["episode_id"]}',flush=True)
    finally:
        dataset.finalize()
    ntrain = sum(i['split']=='train' for i in selected)
    info_path = output/'meta/info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info['splits'] = {**({'train':f'0:{ntrain}'} if ntrain else {}),
                      **({'test':f'{ntrain}:{len(selected)}'} if ntrain < len(selected) else {})}
    info_path.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    # Prevent held-out scenes from entering normalization statistics. Episode
    # selection is also explicit in the generated training configuration.
    if ntrain:
        import pyarrow.parquet as pq
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.io_utils import write_stats
        train_stats = []
        for file in sorted((output/'meta/episodes').rglob('*.parquet')):
            for row in pq.read_table(file).to_pylist():
                if row['episode_index'] >= ntrain:
                    continue
                stats = {}
                for key,value in row.items():
                    if key.startswith('stats/'):
                        feature,stat = key[len('stats/'):].split('/')
                        array = np.atleast_1d(np.asarray(value,dtype=np.int64 if stat == 'count' else np.float64))
                        if feature in features and features[feature]['dtype'] == 'video' and stat != 'count':
                            array = array.reshape(3,1,1)
                        stats.setdefault(feature,{})[stat] = array
                train_stats.append(stats)
        if len(train_stats) != ntrain:
            raise ValueError('Cannot establish train-only normalization statistics')
        write_stats(aggregate_stats(train_stats),output)
        provenance['normalization_statistics'] = 'Training episodes only; held-out episodes excluded'
    (output/'simulation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    training = dict(dataset=dict(repo_id=repo_id,root=str(output.resolve()),episodes=list(range(ntrain)),
                     video_backend='pyav',use_imagenet_stats=True),
        policy=dict(type='act',chunk_size=50,n_action_steps=10,push_to_hub=False,
                    pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1'),
        output_dir=str((output.parent/'box_act_pilot').resolve()),job_name='box-b-pilot',
        steps=5000,batch_size=4,num_workers=0,persistent_workers=False,env_eval_freq=0,
        eval_steps=0,log_freq=100,save_freq=1000,wandb=dict(enable=False),seed=20261116)
    (output/'act_train.json').write_text(json.dumps(training,indent=2)+'\n',encoding='utf-8')
    print(f'LeRobot dataset saved locally: {output}',flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repo-id',default='local/sourccey-box-pickup-b')
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        manifest = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['episodes']:
            validate_episode(args.source/item['episode_id'])
            print(f'Validated {item["episode_id"]}',flush=True)
    elif args.output is None:
        parser.error('--output is required for export')
    else:
        export(args.source,args.output,args.repo_id)


if __name__ == '__main__':
    main()
