import mujoco
import numpy as np
import pytest

from sourccey.rendering import backend_name, create_renderer


def test_backend_selection_is_explicit(monkeypatch):
    monkeypatch.setenv('SOURCCEY_RENDERER', 'filament')
    assert backend_name() == 'filament'
    assert backend_name('classic') == 'classic'
    with pytest.raises(ValueError):
        backend_name('unknown')


@pytest.mark.parametrize('backend', ['classic', 'filament'])
def test_four_cameras_keep_orientation_and_hide_collision_geometry(backend):
    if backend == 'filament' and int(mujoco.__version__.split('.')[0]) < 4:
        pytest.skip('Filament requires MuJoCo 4')
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="160" offheight="120"/></visual>
      <worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1" rgba=".3 .3 .3 1"/>
      <geom type="box" size=".2 .2 .2" pos="0 0 .2" rgba="1 0 0 1"/>
      <geom type="box" size=".3 .3 .3" pos="0 0 .3" group="3" rgba="0 1 0 1"/>
      <geom type="box" size=".12 .12 .05" pos="0 .5 .1" rgba="0 0 1 1"/>
      <camera name="a" pos="0 0 2"/><camera name="b" pos="0 0 2"/>
      <camera name="c" pos="0 0 2"/><camera name="d" pos="0 0 2"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    with create_renderer(model, width=160, height=120, backend=backend) as renderer:
        if backend == 'filament':
            frames = renderer.render_many(data, ['a', 'b', 'c', 'd'], opt)
        else:
            frames = {}
            for name in ('a', 'b', 'c', 'd'):
                renderer.update_scene(data, camera=name, scene_option=opt)
                frames[name] = renderer.render().copy()
        for frame in frames.values():
            assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
            assert float(frame[60,80,0]) > float(frame[60,80,1])+20
            blue = frame[:,:,2].astype(float)-frame[:,:,0].astype(float)
            assert np.max(blue[:50]) > 20 and np.max(blue[70:]) < 20
            assert np.mean(np.abs(frame.astype(float)-frames['a'])) < .5
