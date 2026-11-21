import json
import numpy as np
import pytest

from sourccey.build import ROOT
from sourccey.xvla_eval import JointMapping, LiftTracker, select_sweep, sweep_summary


def test_normalized_mapping_roundtrips_and_clips_absolute_targets():
    mapping = JointMapping(ROOT/"artifacts/robot_joint_mapping.json")
    action = np.array([-25, 12, -45, 30, 10, 60.])
    limits = np.array([[-10., 10.]]*6)
    radians, clipped = mapping.command(action, limits)
    assert mapping.state(radians) == pytest.approx(action)
    assert not clipped.any()
    radians, clipped = mapping.command([200, -200, 0, 0, 0, -50], limits)
    assert mapping.state(radians) == pytest.approx([100, -100, 0, 0, 0, 0])
    assert clipped.tolist() == [True, True, False, False, False, True]
    for bad in ([0]*5, [0]*5+[np.nan]):
        with pytest.raises(ValueError):
            mapping.command(bad, limits)


def tracker():
    return LiftTracker(dict(minimum_bottom_clearance_mm=50, hold_seconds=2,
                           minimum_finger_normal_force_n=.1,
                           maximum_box_drift_during_hold_mm=5), .762)


def test_lift_requires_continuous_two_finger_contact_and_clearance():
    t = tracker()
    for timestamp in np.arange(0, 3, .02):
        t.observe(timestamp, .85, [0, 0, .88], {"fixed": 1., "moving": 0.})
    assert t.maximum_clearance == pytest.approx(.088)
    assert t.stable_hold is None
    for timestamp in np.arange(3, 4.5, .02):
        t.observe(timestamp, .85, [0, 0, .88], {"fixed": 1., "moving": 1.})
    t.observe(4.5, .77, [0, 0, .8], {"fixed": 1., "moving": 1.})
    assert t.stable_hold is None
    for timestamp in np.arange(4.52, 6.55, .02):
        t.observe(timestamp, .85, [0, 0, .88], {"fixed": 1., "moving": 1.})
    assert t.stable_hold["duration_s"] == pytest.approx(2)
    assert t.stable_hold["drift_mm"] == 0


def test_a_lift_that_keeps_drifting_does_not_count_as_stable():
    t = tracker()
    for timestamp in np.arange(0, 3, .02):
        t.observe(timestamp, .85, [timestamp*.01, 0, .88], {"fixed": 1., "moving": 1.})
    assert t.stable_hold is None


def test_sweep_uses_only_requested_split_and_checks_scene_identity(tmp_path):
    rows = [dict(episode_index=i, source_episode=f"scene_{i}", split=split, location_id=f"site_{i}")
            for i, split in enumerate(("train", "validation", "test"))]
    splits = {r["split"]:[r["episode_index"]] for r in rows}
    # Only validation exists locally: selecting it must never read the other splits.
    scene = tmp_path/"scene_1"/"scene.json"
    scene.parent.mkdir()
    config = dict(episode_id="scene_1", split="validation", workspace=dict(location_id="site_1"),
                  box_pose=dict(yaw_deg=-15), seed=7)
    scene.write_text(json.dumps(config))
    selected = select_sweep(dict(episodes=rows), splits, tmp_path, "validation")
    assert [r["episode"] for r in selected] == ["scene_1"]
    for split in ("test", "all"):
        with pytest.raises(ValueError, match="reserved"):
            select_sweep(dict(episodes=rows), splits, tmp_path, split)
    config["split"] = "test"
    scene.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatch"):
        select_sweep(dict(episodes=rows), splits, tmp_path, "validation")
    with pytest.raises(ValueError, match="manifest"):
        select_sweep(dict(episodes=rows), dict(validation=[]), tmp_path, "validation")


def test_sweep_counts_missing_attempts_separately_from_failed_attempts():
    selected = [dict(episode=str(i), location_id="site") for i in range(3)]
    results = [dict(**selected[0], passed=False), dict(**selected[1], passed=True)]
    summary = sweep_summary(selected, results, "inference error")
    assert summary["planned"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == summary["passed"] == 1
    assert not summary["complete"]
    assert summary["success_fraction_completed"] == .5
    assert summary["locations"] == dict(site=dict(planned=3, completed=2, passed=1))
    assert sweep_summary(selected, [])["success_fraction_completed"] is None


def test_continuous_sweep_reports_position_and_yaw_strata_separately():
    selected = [dict(episode=str(i), location_id=f'pose_{i}', cell_id='xy_0_yaw_0' if i < 2 else 'xy_1_yaw_1')
                for i in range(3)]
    results = [dict(**selected[0], passed=True), dict(**selected[1], passed=False)]
    summary = sweep_summary(selected, results)
    assert summary['coverage_strata'] == dict(
        xy_0_yaw_0=dict(planned=2, completed=2, passed=1),
        xy_1_yaw_1=dict(planned=1, completed=0, passed=0))
