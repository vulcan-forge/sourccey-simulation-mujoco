import mujoco
import numpy as np

from sourccey.build import MODEL
from sourccey.lidar import ANGLES, CONFIG, as_dict, scan
from sourccey.simulation import Simulation


def test_forward_lidar_detects_obstacle_and_has_no_return_elsewhere():
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.worldbody.add_geom(name='probe_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[1., 0., .3], size=[.1, .1, .1], group=0)
    sim = Simulation(model=spec.compile())
    result = scan(sim.model, sim.data)
    center = int(np.argmin(np.abs(ANGLES)))
    assert len(result['ranges_m']) == CONFIG['beam_count']
    assert np.isclose(ANGLES[0], -np.pi/2)
    assert np.isclose(ANGLES[-1], np.pi/2)
    assert np.isclose(ANGLES[1]-ANGLES[0], np.deg2rad(.8))
    assert np.allclose(result['origin_world_m'], [.17991, 0., .324125], atol=.0001)
    assert result['hit_names'][center] == 'probe_box'
    assert np.isclose(result['ranges_m'][center], .72, atol=.01)
    assert np.isinf(result['ranges_m'][0])
    assert np.isinf(result['ranges_m'][-1])
    payload = as_dict(result)
    assert payload['ranges_m'][0] is None
    assert payload['ranges_m'][center] > .6
    assert payload['scan_time_s'] == .1
