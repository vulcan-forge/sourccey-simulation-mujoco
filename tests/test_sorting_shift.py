from examples.sorting_shift import run, scene_model
from sourccey.simulation import Simulation


def test_two_parcels_sorted_and_two_pucks_cleared_without_ik_flailing():
    report = run(Simulation(model=scene_model()))
    assert report["passed"], report
