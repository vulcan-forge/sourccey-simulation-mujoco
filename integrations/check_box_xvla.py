"""Verify real LeRobot frames through X-VLA processors and a reduced offline model.

No pretrained weights, uploads or hardware. A random-initialized small X-VLA
checks gradients and held-out loss, not learned task performance.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

TASK='Pick up the box'
CAMERAS=('front_left','front_right','wrist_left','wrist_right')


def isolate_policy_imports():
    """Avoid this local fork's unrelated eager Wall-X imports; change no files."""
    if 'lerobot.policies' not in sys.modules:
        spec=importlib.util.find_spec('lerobot')
        if spec is None or not spec.submodule_search_locations:
            raise ImportError('Run with the existing LeRobot environment')
        package=types.ModuleType('lerobot.policies')
        package.__path__=[str(Path(p)/'policies') for p in spec.submodule_search_locations]
        package.__package__='lerobot.policies'
        sys.modules['lerobot.policies']=package


def check_train_statistics(root,train_ids,stats):
    import pyarrow.parquet as pq
    values={key:[] for key in ('observation.state','action')}
    train_ids=set(train_ids)
    for path in sorted((root/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index',*values])
        mask=np.asarray([i in train_ids for i in table['episode_index'].to_pylist()])
        for key in values:
            if mask.any():
                values[key].append(np.asarray(table[key].to_pylist(),dtype=np.float64)[mask])
    errors={}
    for key,parts in values.items():
        array=np.concatenate(parts)
        errors[key]={}
        for name,expected in dict(mean=array.mean(0),std=array.std(0),min=array.min(0),max=array.max(0)).items():
            actual=np.asarray(stats[key][name])
            errors[key][name]=float(np.max(abs(actual-expected)))
            if not np.allclose(actual,expected,atol=1e-3,rtol=1e-4):
                raise ValueError(f'Normalization includes nontraining data or is inconsistent: {key}/{name}')
    return errors


def check(root,tokenizer,steps=3):
    if not 1<=steps<=20:
        raise ValueError('Compatibility check permits 1..20 gradient steps')
    root=Path(root).resolve()
    tokenizer=Path(tokenizer).resolve()
    cache=root.parent/'.hf_box_xvla_check'
    cache.mkdir(exist_ok=True)
    for key in ('HF_HOME','HF_DATASETS_CACHE','HF_HUB_CACHE'):
        os.environ[key]=str(cache/key.lower())
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    isolate_policy_imports()
    import draccus
    import torch
    from torch.utils.data._utils.collate import default_collate
    from transformers import BartConfig,Florence2VisionConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # registers choice
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    torch.set_num_threads(2)
    torch.manual_seed(20261118)
    provenance=json.loads((root/'simulation_provenance.json').read_text(encoding='utf-8'))
    splits=json.loads((root/'episode_splits.json').read_text(encoding='utf-8'))
    training=json.loads((root/'xvla_train.json').read_text(encoding='utf-8'))
    info=json.loads((root/'meta/info.json').read_text(encoding='utf-8'))
    if provenance['task']!=TASK or set(splits)!= {'train','validation','test'}:
        raise ValueError('Task or three-way splits do not match the pilot')
    if training['dataset']['episodes']!=splits['train']:
        raise ValueError('Training configuration leaks held-out episodes')
    seen={}
    for episode in provenance['episodes']:
        group=episode['location_id']
        if group in seen and seen[group]!=episode['split']:
            raise ValueError('Location appears in more than one split')
        seen[group]=episode['split']
    config=draccus.decode(PreTrainedConfig,training['policy'])
    if not isinstance(config,XVLAConfig) or config.action_mode!='auto':
        raise ValueError('Expected X-VLA with auto action padding')
    config.validate_features()
    expected={'observation.state',*('observation.images.'+c for c in CAMERAS)}
    if set(config.input_features)!=expected or config.action_feature.shape!=(6,):
        raise ValueError('Unexpected features or privileged simulator inputs')
    # Validate the full checkpoint config without allocating its large weights.
    config.get_florence_config()
    chunk=config.chunk_size
    dataset=LeRobotDataset(provenance['repo_id'],root=root,video_backend='pyav',
        delta_timestamps={'action':[i/info['fps'] for i in range(chunk)]})
    print(f'Loaded {dataset.num_episodes} episodes; checking start/middle/end images and exact task...',flush=True)
    starts={}
    offset=0
    for episode in provenance['episodes']:
        starts[episode['episode_index']]=offset
        for frame in (0,episode['frames']//2,episode['frames']-1):
            sample=dataset[offset+frame]
            if sample['task']!=TASK:
                raise ValueError('Wrong task text in LeRobot data')
            for name in CAMERAS:
                image=sample['observation.images.'+name]
                if image.shape!=(3,240,320) or not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
                    raise ValueError('Camera image contract failed')
        offset+=episode['frames']
    if offset!=len(dataset):
        raise ValueError('Episode/frame accounting mismatch')
    statistics_errors=check_train_statistics(root,splits['train'],dataset.meta.stats)

    # Actual X-VLA implementation with a deliberately small Florence-2 and head.
    # Real dataset images, BART tokenizer and official preprocessing are retained.
    small=copy.deepcopy(config)
    small.device='cpu'
    small.pretrained_path=None
    small.tokenizer_name=str(tokenizer)
    small.resize_imgs_with_padding=(64,64)
    small.hidden_size=64
    small.depth=2
    small.num_heads=4
    small.len_soft_prompts=8
    small.num_denoising_steps=2
    small.tokenizer_max_length=32
    vision=Florence2VisionConfig(depths=[1,1,1,1],embed_dim=[16,32,64,128],
        num_heads=[1,2,4,8],num_groups=[1,2,4,8],projection_dim=64,
        window_size=4,max_position_embeddings=64,drop_path_rate=0.)
    text=BartConfig(vocab_size=51289,d_model=64,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=128,decoder_ffn_dim=128,
        max_position_embeddings=256,dropout=0.,attention_dropout=0.,activation_dropout=0.)
    small.florence_config=dict(vision_config=vision.to_dict(),text_config=text.to_dict(),
        pad_token_id=1,bos_token_id=0,eos_token_id=2,image_token_id=51289)
    small._florence_config_obj=None
    policy=XVLAPolicy(small)
    preprocessor,postprocessor=make_xvla_pre_post_processors(small,dataset.meta.stats)
    def batch_for(ids):
        batch=default_collate([dataset[starts[i]+400] for i in ids])
        if batch['action_is_pad'].any():
            raise ValueError('Unexpected padded action chunk in compatibility batch')
        return preprocessor(batch)
    train=batch_for(splits['train'][:2])
    held_out=batch_for(splits['validation'][:1])
    print('Official X-VLA processors loaded; running reduced-model gradient steps...',flush=True)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-5)
    losses=[]
    for _ in range(steps):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss,_=policy(train)
        if not torch.isfinite(loss):
            raise ValueError('X-VLA training loss is not finite')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        if not torch.isfinite(gradient) or gradient<=0:
            raise ValueError('X-VLA gradient is invalid')
        optimizer.step()
        losses.append(float(loss.detach()))
    policy.eval()
    with torch.inference_mode():
        held_out_loss,_=policy(held_out)
        policy.reset()
        output=postprocessor(policy.select_action(held_out))
    if output.shape!=(len(splits['validation'][:1]),6) or not torch.isfinite(output).all() or not torch.isfinite(held_out_loss):
        raise ValueError('X-VLA held-out loss or six-channel output is invalid')
    report=dict(passed=True,task=TASK,policy='X-VLA',dataset_episodes=dataset.num_episodes,
        dataset_frames=len(dataset),images_checked=dataset.num_episodes*3*4,
        split_episodes={k:len(v) for k,v in splits.items()},train_episode_ids=splits['train'],
        validation_episode_ids=splits['validation'],test_episode_ids=splits['test'],
        image_features=list(small.image_features),state_shape=list(train['observation.state'].shape),
        action_chunk_shape=list(train['action'].shape),model_action_dim=policy.model.dim_action,
        real_action_dim=6,gradient_steps=steps,losses=losses,gradient_norm=float(gradient),
        train_only_statistics_maximum_errors=statistics_errors,
        held_out_loss=float(held_out_loss),inference_shape=list(output.shape),device='cpu',
        pretrained_weights_loaded=False,full_model_training_started=False,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Reduced, random-initialized X-VLA pipeline check with real exported images and official processors. Not a trained pickup policy, pretrained-model memory test or closed-loop task evaluation.')
    (root/'xvla_compatibility.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,default=Path('artifacts/xvla_setup/bart_tokenizer'))
    parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args()
    check(args.dataset,args.tokenizer,args.steps)
