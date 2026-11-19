import pytest
import json

from integrations.export_box_dataset import split_indices,xvla_policy_config,await_camera_episode


def test_split_indices_preserve_three_groups_and_reject_location_leaks():
    episodes=[dict(location_id='a',split='train'),dict(location_id='a',split='train'),
        dict(location_id='b',split='validation'),dict(location_id='c',split='test')]
    assert split_indices(episodes)==dict(train=[0,1],validation=[2],test=[3])
    episodes[-1]['location_id']='a'
    with pytest.raises(ValueError,match='crosses'):
        split_indices(episodes)


def test_xvla_adapts_real_features_without_changing_checkpoint_state_width():
    base=dict(type='xvla',florence_config={'vision_config':{'projection_dim':1024}},
        max_state_dim=20,chunk_size=30,hidden_size=1024)
    config=xvla_policy_config(base,['left_arm']*6)
    assert config['max_state_dim']==20 and config['chunk_size']==30 and config['hidden_size']==1024
    assert config['output_features']['action']['shape']==[6]
    assert config['max_action_dim']==20 and config['action_mode']=='auto'
    assert config['num_image_views']==4 and len(config['input_features'])==5
    assert config['normalization_mapping']==dict(VISUAL='IDENTITY',STATE='MEAN_STD',ACTION='MEAN_STD')
    assert config['pretrained_path']=='lerobot/xvla-base'
    assert 'pretrained_path' not in base


def test_xvla_configuration_requires_checkpoint_architecture():
    with pytest.raises(ValueError,match='published'):
        xvla_policy_config(dict(type='xvla'),['joint']*6)


def test_following_export_requires_matching_completed_camera_marker(tmp_path):
    item=dict(episode_id='episode',pilot_signature='signed-physics')
    rendering=dict(renderer='filament')
    marker=tmp_path/'render_completion.json'
    marker.write_text(json.dumps(dict(pilot_signature='signed-physics',rendering=rendering)))
    await_camera_episode(tmp_path,item,rendering)
    marker.write_text(json.dumps(dict(pilot_signature='wrong-physics',rendering=rendering)))
    with pytest.raises(ValueError,match='does not match'):
        await_camera_episode(tmp_path,item,rendering)
    marker.unlink()
    with pytest.raises(TimeoutError):
        await_camera_episode(tmp_path,item,rendering,timeout_s=-1)
