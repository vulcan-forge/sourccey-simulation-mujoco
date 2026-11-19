from concurrent.futures import Future
import json

import pytest

from examples import box_pilot_render as renderer


def test_follow_bounds_gpu_queue_and_recovers_signed_completion_markers(monkeypatch,tmp_path):
    episodes=[]
    for i in range(3):
        name=f'episode_{i}'
        directory=tmp_path/name
        directory.mkdir()
        (directory/'scene.json').write_text(json.dumps({'signature':f'signed-{i}'}))
        episodes.append(dict(episode_id=name,pilot_signature=f'signed-{i}',eligible_for_imitation=True,videos=False))
    manifest=dict(plan={'count':3},episodes=episodes,pending=[],status='physics_complete',rendering=renderer.provenance())
    renderer.write_json(tmp_path/'manifest.json',manifest)
    calls=[]
    class Pool:
        def __init__(self,max_workers):
            assert max_workers==2
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def submit(self,fn,directory):
            calls.append(directory.name)
            future=Future()
            future.set_result(directory.name)
            return future
    def finalize(output,jobs):
        result=json.loads((output/'manifest.json').read_text())
        assert result['status']=='physics_complete'
        assert all(e['videos'] for e in result['episodes'])
        return result
    monkeypatch.setattr(renderer,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(renderer,'pilot_signature',lambda config:config['signature'])
    monkeypatch.setattr(renderer,'render_parallel',finalize)
    result=renderer.render_follow(tmp_path,jobs=2)
    assert len(calls)==3 and all(e['videos'] for e in result['episodes'])
    renderer.render_follow(tmp_path,jobs=2)
    assert len(calls)==3
    marker=tmp_path/'episode_0/render_completion.json'
    changed=json.loads(marker.read_text())
    changed['pilot_signature']='wrong'
    marker.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='signature'):
        renderer.render_follow(tmp_path,jobs=2)
