"""Check exported observations/actions with one small offline ACT gradient step.

This checks compatibility, not task performance. No policy is deployed, no
weights are downloaded, and no robot connection or upload is attempted.
"""
import argparse
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np


def load_act_only():
    """Load installed ACT without this fork's unrelated eager policy imports.

    The local LeRobot fork imports Wall-X/Qwen from policies/__init__.py,
    which currently fails with its installed huggingface_hub version. This
    process-local package shim leaves the external checkout untouched; all
    ACT/model dependencies still come from the installed LeRobot code.
    """
    if 'lerobot.policies' not in sys.modules:
        spec = importlib.util.find_spec('lerobot')
        if spec is None or not spec.submodule_search_locations:
            raise ImportError('Install LeRobot or run with its existing Python environment')
        package = types.ModuleType('lerobot.policies')
        package.__path__ = [str(Path(p)/'policies') for p in spec.submodule_search_locations]
        package.__package__ = 'lerobot.policies'
        sys.modules['lerobot.policies'] = package
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    return ACTConfig,ACTPolicy


def check(root):
    # Make a blocked native/dependency import diagnosable in redirected logs.
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,repeat=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    root = Path(root).resolve()
    # A writable local cache also avoids unrelated global dataset-cache locks.
    cache = root.parent/'.hf_box_cache'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_DATASETS_CACHE'] = str(cache/'datasets')
    os.environ['HF_HUB_CACHE'] = str(cache/'hub')
    print('Loading installed ACT and dataset dependencies...',flush=True)
    import torch
    from torch.utils.data._utils.collate import default_collate
    from lerobot.configs import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import dataset_to_policy_features
    ACTConfig,ACTPolicy = load_act_only()
    print('ACT dependencies loaded; opening local training episodes...',flush=True)

    provenance = json.loads((root/'simulation_provenance.json').read_text(encoding='utf-8'))
    info = json.loads((root/'meta/info.json').read_text(encoding='utf-8'))
    train_ids = [e['episode_index'] for e in provenance['episodes'] if e['split']=='train']
    test_ids = [e['episode_index'] for e in provenance['episodes'] if e['split']=='test']
    torch.manual_seed(20261116)
    torch.set_num_threads(2)
    dataset = LeRobotDataset('local/sourccey-box-pickup-b',root=root,episodes=train_ids,
        delta_timestamps={'action':[i/info['fps'] for i in range(10)]},video_backend='pyav')
    faulthandler.cancel_dump_traceback_later()
    print(f'Loaded {len(train_ids)} training episodes; constructing small ACT...',flush=True)
    features = dataset_to_policy_features(dataset.meta.features)
    inputs = {k:v for k,v in features.items() if v.type != FeatureType.ACTION}
    outputs = {k:v for k,v in features.items() if v.type == FeatureType.ACTION}
    if set(inputs) != {'observation.state',*('observation.images.'+c for c in ('front_left','front_right','wrist_left','wrist_right'))}:
        raise ValueError('Unexpected learner feature, possibly privileged simulator information')
    config = ACTConfig(device='cpu',input_features=inputs,output_features=outputs,
        chunk_size=10,n_action_steps=5,pretrained_backbone_weights=None,dim_model=128,
        n_heads=4,dim_feedforward=512,n_encoder_layers=1,n_decoder_layers=1,
        latent_dim=16,n_vae_encoder_layers=1,push_to_hub=False)
    policy = ACTPolicy(config)
    batch = default_collate([dataset[0],dataset[min(100,len(dataset)-1)]])
    for name in ('observation.state','action'):
        stats = dataset.meta.stats[name]
        mean = torch.as_tensor(np.asarray(stats['mean']),dtype=torch.float32)
        std = torch.as_tensor(np.asarray(stats['std']),dtype=torch.float32).clamp_min(1e-6)
        batch[name] = (batch[name]-mean)/std
    for name in config.image_features:
        mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
        std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
        batch[name] = (batch[name]-mean)/std
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-5)
    policy.train()
    loss,metrics = policy(batch)
    if not torch.isfinite(loss):
        raise ValueError('ACT loss is not finite')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
    if not torch.isfinite(gradient):
        raise ValueError('ACT gradient is not finite')
    optimizer.step()
    policy.eval()
    policy.reset()
    with torch.inference_mode():
        actions = policy.select_action({k:batch[k] for k in config.input_features})
    if actions.shape != (2,6) or not torch.isfinite(actions).all():
        raise ValueError('Invalid ACT action output')
    report = dict(passed=True,training_episodes=train_ids,held_out_episodes=test_ids,
        image_features=list(config.image_features),state_shape=list(batch['observation.state'].shape),
        action_chunk_shape=list(batch['action'].shape),loss=float(loss.detach()),
        gradient_norm=float(gradient),metrics=metrics,device='cpu',pretrained_download=False,
        note='One small offline ACT gradient update verifies dataset/model compatibility; this is not a trained pickup policy or a task success evaluation.')
    (root/'learning_smoke_test.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    check(parser.parse_args().dataset)
