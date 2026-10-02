import mujoco

from examples.dual_grasp import run, scene_model
from sourccey.simulation import Simulation


def test_both_free_objects_are_gripped_lifted_and_released_by_contact():
    sim = Simulation(model=scene_model())
    assert sim.model.neq == 0
    for side in ("left", "right"):
        assert sim.model.joint(f"{side}_object_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    report = run(sim)
    assert report["passed"], report
    assert all(report["hold_steps_with_both_pads_in_contact"][side] > 1500
               for side in ("left", "right"))
