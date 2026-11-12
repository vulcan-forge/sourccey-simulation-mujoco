import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from examples.robot_ball_replay import (
    advance_curve, curve_target, go_to_start, joint_fit, motor_calibration_summary, preflight,
    prepare_curve, replay, replay_tracking_error,
)
from examples.robot_joint_mapping import add_reference
from examples.robot_start_pose import KEYS


def test_measured_direction_and_no_assumed_scale():
    fit = joint_fit({"measurement_method": "measured fixture", "samples": [
        {"hardware_normalized": -20, "mujoco_rad": 1.},
        {"hardware_normalized": 30, "mujoco_rad": -1.}]})
    assert fit["slope"] == pytest.approx(-.04)
    assert fit["offset"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="two measured"):
        joint_fit({"measurement_method": "parked once", "samples": [
            {"hardware_normalized": 0, "mujoco_rad": 0}]})


def fixture():
    profile = {"version": 1, "host": "robot", "units": "normalized",
               "pose": "operator-positioned arms down", "positions": dict.fromkeys(KEYS, 0.),
               "elevator": 100., "stationary_hold_verified": True}
    policy = {"version": 1, "host": "robot", "units": "normalized", "limits": {
        side + "_shoulder_lift.pos": {"min": -100, "max": 100} for side in ("left", "right")}}
    names = [k[:-4] for k in KEYS[:6]]
    frames = [{"time_s": i * .04, "joint_rad": dict.fromkeys(names, i * .002)} for i in range(50)]
    mapping = {"version": 1, "host": "robot", "units": "normalized", "joints": {
        n: {"measurement_method": "fixture", "samples": [
            {"hardware_normalized": 0, "mujoco_rad": 0},
            {"hardware_normalized": 100, "mujoco_rad": 1}]} for n in names}}
    result = {"simulation_passed": True, "pickup": {"passed": True},
              "table_clearance": {"robot_table_contact_steps": 0}}
    return {"side": "left", "illustrative": False}, result, frames, mapping, profile, policy, "robot"


def test_complete_mapping_passes_and_missing_mapping_fails():
    args = fixture()
    report = preflight(*args)
    assert report["ready"]
    args[3]["joints"].pop("left_elbow_flex")
    report = preflight(*args)
    assert not report["ready"]
    assert any("left_elbow_flex" in e and "no joint conversion" in e for e in report["errors"])
    assert not report["hardware_commands_sent"]


def test_absent_mapping_reports_missing_file_without_six_metadata_errors():
    args = list(fixture())
    args[3] = {}
    errors = preflight(*args)["errors"]
    assert len(errors) == 1
    assert "mapping file is missing" in errors[0]


def test_derived_conversion_is_inspected_but_cannot_execute_without_verification():
    args = list(fixture())
    args[3]["joints"]["left_elbow_flex"] = {
        "verification_status": "pending",
        "derived_affine": {"slope": .01, "offset": 0, "hardware_range": [-100, 100]}}
    report = preflight(*args)
    assert not report["ready"]
    assert "left_elbow_flex" in report["joint_fits"]
    assert any("supervised direction/travel check" in e for e in report["errors"])


def test_confirmed_source_requires_evidence_and_motion_only_keeps_mapping_checks():
    args = list(fixture())
    args[0]["illustrative"] = True
    entry = {"verification_status": "source_reference_confirmed",
             "derived_affine": {"slope": .01, "offset": 0, "hardware_range": [-100, 100]}}
    args[3]["joints"]["left_elbow_flex"] = entry
    assert not preflight(*args, motion_only=True)["ready"]
    entry["verification_evidence"] = "Confirmed source direction/scale on this robot"
    report = preflight(*args, motion_only=True)
    assert report["ready"]
    assert "motion-only" in report["hardware_mode"]
    assert not preflight(*args)["ready"]
    args[2][-1]["joint_rad"]["left_elbow_flex"] = 2
    assert not preflight(*args, motion_only=True)["ready"]


@pytest.mark.parametrize("fault", ["extrapolation", "wrong_anchor", "guard", "scene", "failed_sim", "units"])
def test_preflight_rejects_each_material_fault(fault):
    args = list(fixture())
    if fault == "extrapolation":
        args[2][-1]["joint_rad"]["left_elbow_flex"] = 1.2
    elif fault == "wrong_anchor":
        args[4]["positions"]["left_wrist_roll.pos"] = 10
    elif fault == "guard":
        args[5]["limits"]["left_shoulder_lift.pos"]["max"] = 5
    elif fault == "scene":
        args[0]["illustrative"] = True
    elif fault == "failed_sim":
        args[1]["pickup"]["passed"] = False
    elif fault == "units":
        args[3]["units"] = "degrees"
    assert not preflight(*args)["ready"]


def test_half_speed_doubles_timestamps_and_preserves_inactive_arm():
    args = fixture()
    report = preflight(*args)
    curve, dilation = prepare_curve(args[2], report["joint_fits"], args[4], .5)
    normal, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1)
    assert normal.x[-1] == pytest.approx(args[2][-1]["time_s"] + 2.25)
    times = np.linspace(0, curve.x[-1], 20000)
    assert curve.x[-1] == pytest.approx(2 * normal.x[-1])
    assert curve.x == pytest.approx(2 * normal.x)
    assert curve(times) == pytest.approx(normal(times / 2))
    assert curve(times, 1) == pytest.approx(normal(times / 2, 1) / 2)
    assert curve(times, 2) == pytest.approx(normal(times / 2, 2) / 4)
    assert np.all(curve(times)[:, 6:] == 0)
    assert np.all(curve(0) == 0)
    assert curve(curve.x[-1])[0] == pytest.approx(9.8)
    assert dilation == 1


def test_seven_second_recording_plays_in_fourteen_seconds_at_half_speed():
    args = fixture()
    report = preflight(*args)
    last_time = args[2][-1]["time_s"]
    for frame in args[2]:
        frame["time_s"] *= 7 / last_time
    curve, dilation = prepare_curve(args[2], report["joint_fits"], args[4], .5)
    assert curve.x[-1] - 2.25 / .5 == pytest.approx(14)
    assert dilation == 1


def test_double_speed_halves_duration_and_preserves_joint_path():
    args = fixture()
    report = preflight(*args)
    normal, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    fast, _ = prepare_curve(args[2], report["joint_fits"], args[4], 2.)
    times = np.linspace(0, fast.x[-1], 1000)
    assert fast.x == pytest.approx(normal.x / 2)
    assert fast(times) == pytest.approx(normal(times * 2))
    assert fast(times, 1) == pytest.approx(normal(times * 2, 1) * 2)
    assert fast(times, 2) == pytest.approx(normal(times * 2, 2) * 4)
    assert np.all(fast(times)[:, 6:] == 0)


@pytest.mark.parametrize("speed", [0., -1., 2.01, float("nan"), float("inf")])
def test_replay_rejects_unsupported_speed(speed):
    args = fixture()
    report = preflight(*args)
    with pytest.raises(ValueError):
        prepare_curve(args[2], report["joint_fits"], args[4], speed)


def test_replay_feedback_error_sends_no_new_target():
    class Link:
        def __init__(self):
            self.sent = []
        def read(self):
            return {"positions": dict.fromkeys(KEYS, 50.), "elevator": 100., "base_velocity": [0]*4}
        def send(self, target):
            self.sent.append(target)
    args = fixture()
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], .5)
    link = Link()
    with pytest.raises(ValueError, match="tracking error"):
        replay(link, curve, 100., lambda row: None)
    assert not link.sent


@pytest.fixture
def startup_robot(monkeypatch):
    clock = [0.]
    monkeypatch.setattr("examples.robot_ball_replay.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("examples.robot_ball_replay.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    class Link:
        def __init__(self):
            self.current = {"positions": dict.fromkeys(KEYS, 7.), "elevator": 100.,
                            "base_velocity": [0.] * 4}
            self.sent = []
            self.reads = 0
            self.jitter = .7
            self.gripper_offset = 0.
            self.joint_offsets = {}
            self.fault = None

        def read(self):
            clock[0] += 1 / 30
            self.reads += 1
            if self.sent and self.fault == "stale":
                raise TimeoutError("No fresh robot observation")
            observed = copy.deepcopy(self.current)
            if self.sent:
                observed["positions"][KEYS[0]] += self.jitter * (-1 if self.reads % 2 else 1)
                observed["positions"]["left_gripper.pos"] = min(
                    100., observed["positions"]["left_gripper.pos"] + self.gripper_offset)
                for joint, offset in self.joint_offsets.items():
                    observed["positions"][joint] += offset
            return observed

        def send(self, target):
            self.sent.append(target.copy())
            # Model physical endpoint clipping, including sub-tick roundoff.
            self.current["positions"] = {
                key: min(100., max(0. if key.endswith("gripper.pos") else -100., value))
                for key, value in target.items()}

    return Link()


def test_startup_holds_live_pose_despite_capture_jitter_then_replays(startup_robot):
    args = fixture()
    link, trace = startup_robot, []
    live = link.current["positions"].copy()
    assert go_to_start(link, args[4], args[5], "robot", 1., trace.append)["start_reached"]
    assert link.sent[0] == live
    hold = [row for row in trace if row["phase"] == "hold-current"]
    # This feedback would fail the old one-unit calibration capture gate.
    assert np.ptp([row["observed"]["positions"][KEYS[0]] for row in hold]) > 1
    assert all(row["commanded"] == live for row in hold)
    assert link.sent[-1] == args[4]["positions"]
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    replay(link, curve, 100., trace.append)
    assert link.sent[-1] == curve_target(curve, curve.x[-1])
    assert any(row["phase"] == "replay" for row in trace)


@pytest.mark.parametrize("fault", ["base", "elevator", "host_guard", "too_far"])
def test_startup_validates_live_state_before_first_command(startup_robot, fault):
    args = fixture()
    link = startup_robot
    if fault == "base":
        link.current["base_velocity"][0] = .1
    elif fault == "elevator":
        link.current["elevator"] = 103.
    elif fault == "host_guard":
        args[5]["limits"]["left_shoulder_lift.pos"]["max"] = 5.
    elif fault == "too_far":
        link.current["positions"][KEYS[0]] = 40.
    with pytest.raises(ValueError):
        go_to_start(link, args[4], args[5], "robot", 1., lambda row: None)
    assert not link.sent


@pytest.mark.parametrize("fault", ["stall", "tracking", "stale"])
def test_startup_fault_holds_initial_target_and_never_starts_return(startup_robot, fault):
    args = fixture()
    link, trace = startup_robot, []
    link.jitter = 5.1 if fault == "stall" else 11.
    link.fault = fault
    live = link.current["positions"].copy()
    expected = ValueError if fault == "tracking" else TimeoutError
    with pytest.raises(expected):
        go_to_start(link, args[4], args[5], "robot", 1., trace.append)
    assert link.sent and all(target == live for target in link.sent)
    assert not any(row["phase"] in ("ramp", "replay") for row in trace)


def test_closed_viewer_prevents_initial_hold_command(startup_robot):
    args = fixture()

    def closed_window(record):
        raise KeyboardInterrupt("MuJoCo window closed")

    with pytest.raises(KeyboardInterrupt, match="window closed"):
        go_to_start(startup_robot, args[4], args[5], "robot", 1., closed_window)
    assert not startup_robot.sent


def test_replay_gripper_residual_allows_opening_arm_motion_and_final_settle(startup_robot):
    args = fixture()
    args[4]["positions"]["left_gripper.pos"] = 99.42121887207031
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    link, trace = startup_robot, []
    link.current["positions"] = args[4]["positions"].copy()
    link.jitter = 0.
    link.gripper_offset = 2.186495065689087  # Measured residual in the failed opening.
    replay(link, curve, 100., trace.append)
    assert trace[-1]["trajectory_s"] == curve.x[-1]
    assert link.sent[-1] == curve_target(curve, curve.x[-1])
    assert link.sent[-1][KEYS[0]] > link.sent[0][KEYS[0]]
    assert any(2 < row["tracking"]["error_units"] <= 3 for row in trace)
    assert all(row["tracking"]["within_tolerance"] for row in trace)


def test_retry_can_return_open_gripper_to_saved_start_at_limited_speed(startup_robot):
    args = fixture()
    args[4]["positions"]["left_gripper.pos"] = 99.42121887207031
    link, trace = startup_robot, []
    link.current["positions"] = args[4]["positions"].copy()
    link.current["positions"]["left_gripper.pos"] = 2.186495065689087
    assert go_to_start(link, args[4], args[5], "robot", 1., trace.append)["start_reached"]
    assert link.sent[0]["left_gripper.pos"] == 2.186495065689087
    assert link.sent[-1] == args[4]["positions"]
    motion = next(row["motion"] for row in trace if row["phase"] == "go-to-start-plan")
    assert motion["maximum_delta_units"] > 97
    assert motion["duration_s"] > 18
    assert motion["command_velocity_limit_units_s"] == 10


def test_partial_replay_can_return_and_replay_at_double_speed_with_shoulder_residual(startup_robot):
    args = fixture()
    link, trace = startup_robot, []
    link.current["positions"] = args[4]["positions"].copy()
    link.current["positions"][KEYS[0]] = 70.
    link.current["positions"]["left_shoulder_lift.pos"] = 40.
    link.jitter = 0.
    link.joint_offsets["left_shoulder_lift.pos"] = -2.1371388466443477
    assert go_to_start(link, args[4], args[5], "robot", 2., trace.append,
                       tracking_tolerance=5., max_delta=200)["start_reached"]
    motion = next(row["motion"] for row in trace if row["phase"] == "go-to-start-plan")
    assert motion["speed"] == 2. and motion["maximum_delta_units"] == 70.
    assert motion["command_velocity_limit_units_s"] == 20.
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 2.)
    replay(link, curve, 100., trace.append, tracking_tolerance=5.)
    assert trace[-1]["trajectory_s"] == curve.x[-1]
    assert link.sent[-1] == curve_target(curve, curve.x[-1])
    assert not any(row.get("waiting_for_tracking") for row in trace)


@pytest.mark.parametrize("joint,tolerance", [(KEYS[0], 5.), ("left_gripper.pos", 5.),
                                            ("right_gripper.pos", 5.)])
def test_replay_tracking_uses_each_joints_tolerance(joint, tolerance):
    reference = dict.fromkeys(KEYS, 0.)
    observed = {"positions": reference.copy()}
    observed["positions"][joint] = tolerance
    assert replay_tracking_error(observed, reference)["within_tolerance"]
    observed["positions"][joint] += .01
    error = replay_tracking_error(observed, reference)
    assert error["joint"] == joint and error["tolerance_units"] == tolerance
    assert not error["within_tolerance"]


@pytest.mark.parametrize("fault", ["gripper_stall", "arm_stall", "gripper_large_error"])
def test_replay_retains_stall_and_hard_error_checks(startup_robot, fault):
    args = fixture()
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    link, trace = startup_robot, []
    link.current["positions"] = args[4]["positions"].copy()
    link.jitter = 5.1 if fault == "arm_stall" else 0.
    # An acceptable gripper error must not mask an arm stall.
    link.gripper_offset = {"gripper_stall": 5.1, "arm_stall": 4.9,
                           "gripper_large_error": 10.1}[fault]
    joint = "left_shoulder_pan.pos" if fault == "arm_stall" else "left_gripper.pos"
    expected = ValueError if fault == "gripper_large_error" else TimeoutError
    with pytest.raises(expected, match=joint):
        replay(link, curve, 100., trace.append)
    assert trace[-1]["trajectory_s"] < .1
    assert link.sent[-1] == link.sent[-2]


def test_custom_tracking_window_is_applied_to_hold_return_and_replay(startup_robot):
    args = fixture()
    link, trace = startup_robot, []
    link.jitter = 3.5
    assert go_to_start(link, args[4], args[5], "robot", 1., trace.append,
                       tracking_tolerance=4.)["start_reached"]
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    replay(link, curve, 100., trace.append, tracking_tolerance=4.)
    assert all(row["tracking"]["tolerance_units"] == 4. for row in trace
               if row["phase"] in ("hold-current", "ramp", "replay"))


@pytest.mark.parametrize("tolerance", [0., -1., 10., float("nan"), float("inf")])
def test_invalid_tracking_tolerance_sends_no_commands(startup_robot, tolerance):
    args = fixture()
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    with pytest.raises(ValueError):
        go_to_start(startup_robot, args[4], args[5], "robot", 1., lambda row: None,
                    tracking_tolerance=tolerance)
    with pytest.raises(ValueError):
        replay(startup_robot, curve, 100., lambda row: None, tracking_tolerance=tolerance)
    assert not startup_robot.sent


def test_actual_shoulder_transition_is_bounded_before_sending():
    record = json.loads((Path(__file__).parent / "fixtures/replay_shoulder_lag.json").read_text())
    start, end = record["previous_target"], record["unbounded_target"]
    observed, dt = record["observed"], record["dt"]
    curve = PchipInterpolator([0., dt], [[pose[key] for key in KEYS] for pose in (start, end)], axis=0)
    assert replay_tracking_error(observed, start)["error_units"] < 5
    assert replay_tracking_error(observed, end)["error_units"] > 10
    elapsed, target, limited = advance_curve(curve, 0., start, observed, dt, 5.)
    assert limited and 0 < elapsed < dt
    assert replay_tracking_error(observed, target)["error_units"] <= 5
    assert max(abs(target[key] - start[key]) for key in KEYS) <= 1
    assert target == curve_target(curve, elapsed)


def test_path_advance_does_not_skip_fast_out_and_back():
    poses = np.zeros((4, len(KEYS)))
    poses[1, 1] = 30.
    curve = PchipInterpolator([0., .01, .02, .04], poses, axis=0)
    start = dict.fromkeys(KEYS, 0.)
    elapsed, target, limited = advance_curve(curve, 0., start, {"positions": start}, 1/30, 5.)
    assert limited and 0 < elapsed < .01
    assert .99 < target["left_shoulder_lift.pos"] <= 1.


@pytest.fixture
def slow_robot(monkeypatch):
    clock = [0.]
    monkeypatch.setattr("examples.robot_ball_replay.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("examples.robot_ball_replay.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    class Link:
        def __init__(self):
            self.current = dict.fromkeys(KEYS, 0.)
            self.target = self.current.copy()
            self.velocity = 3.
            self.start_delay = 6.
            self.previous = 0.
            self.sent = []
            self.max_command_error = 0.

        def read(self):
            clock[0] += 1/30
            dt = max(0., clock[0] - max(self.previous, self.start_delay))
            self.previous = clock[0]
            for key in KEYS:
                delta = self.target[key] - self.current[key]
                self.current[key] += max(-self.velocity * dt, min(self.velocity * dt, delta))
            return {"positions": self.current.copy(), "elevator": 100., "base_velocity": [0.] * 4}

        def send(self, target):
            self.max_command_error = max(self.max_command_error, max(
                abs(target[key] - self.current[key]) for key in KEYS))
            self.target = target.copy()
            self.sent.append(target.copy())

    return Link(), clock


def aggressive_curve():
    poses = np.zeros((4, len(KEYS)))
    poses[:, 1] = [0., 35., -25., 0.]
    return PchipInterpolator([0., .2, .4, .6], poses, axis=0)


@pytest.mark.parametrize("velocity", [1., 3.])
def test_slow_arm_follows_complete_path_without_timing_abort(slow_robot, velocity):
    link, clock = slow_robot
    link.velocity = velocity
    curve, trace = aggressive_curve(), []
    replay(link, curve, 100., trace.append)
    assert clock[0] > curve.x[-1] + link.start_delay
    assert trace[-1]["trajectory_s"] == curve.x[-1]
    assert link.sent[-1] == curve_target(curve, curve.x[-1])
    assert link.max_command_error <= 5.
    assert max(max(abs(b[key] - a[key]) for key in KEYS)
               for a, b in zip(link.sent, link.sent[1:])) <= 1.
    # A sampled turn can differ by at most one bounded command step.
    assert max(pose["left_shoulder_lift.pos"] for pose in link.sent) == pytest.approx(35., abs=1.)
    assert min(pose["left_shoulder_lift.pos"] for pose in link.sent) == pytest.approx(-25., abs=1.)
    assert all(row["commanded"] == curve_target(curve, row["trajectory_s"]) for row in trace)
    assert all(b["trajectory_s"] >= a["trajectory_s"] for a, b in zip(trace, trace[1:]))
    assert any(row["waiting_for_tracking"] for row in trace)


def test_motionless_joint_times_out_without_running_targets_ahead(slow_robot):
    link, clock = slow_robot
    link.velocity = 0.
    curve, trace = aggressive_curve(), []
    with pytest.raises(TimeoutError, match="no progress for 15 s"):
        replay(link, curve, 100., trace.append)
    assert clock[0] >= 15.
    assert link.max_command_error <= 5.
    assert link.sent[-1] == link.sent[-2]
    assert trace[-1]["trajectory_s"] < .2


def test_long_intentional_dwell_is_not_a_stall(slow_robot):
    link, clock = slow_robot
    curve = PchipInterpolator([0., 20.], np.zeros((2, len(KEYS))), axis=0)
    replay(link, curve, 100., lambda row: None)
    assert clock[0] > 20.
    assert link.sent[-1] == dict.fromkeys(KEYS, 0.)


@pytest.mark.parametrize("timeout", [0., -1., float("nan"), float("inf")])
def test_invalid_stall_timeout_sends_no_commands(startup_robot, timeout):
    args = fixture()
    report = preflight(*args)
    curve, _ = prepare_curve(args[2], report["joint_fits"], args[4], 1.)
    with pytest.raises(ValueError):
        go_to_start(startup_robot, args[4], args[5], "robot", 1., lambda row: None,
                    stall_timeout=timeout)
    with pytest.raises(ValueError):
        replay(startup_robot, curve, 100., lambda row: None, stall_timeout=timeout)
    assert not startup_robot.sent


def test_curve_endpoint_roundoff_is_tolerated_but_motor_limit_violation_is_not():
    values = np.zeros(len(KEYS))
    values[5] = -1e-12
    assert curve_target(lambda t: values, 0)["left_gripper.pos"] == 0
    values[4] = 188
    with pytest.raises(ValueError, match="outside normalized range"):
        curve_target(lambda t: values, 0)


def test_existing_motor_ranges_are_imported_without_fabricated_gearing():
    snapshot = {"host": "robot", "arms": {side: {
        key.split("_", 1)[1][:-4]: {"range_min": 517, "range_max": 3658,
                                  "drive_mode": 0, "homing_offset": -1656, "id": i+1}
        for i, key in enumerate(KEYS[:6])} for side in ("left", "right")}}
    result = motor_calibration_summary(snapshot, "robot")
    assert result["left_elbow_flex"]["encoder_ticks_per_normalized_unit_magnitude"] == 15.705
    assert not result["left_elbow_flex"]["cad_direction_and_gearing_defined_in_this_file"]
    with pytest.raises(ValueError, match="different robot"):
        motor_calibration_summary(snapshot, "other")


def test_recorder_preserves_measured_references_and_rejects_wrong_host():
    mapping = {"version": 1, "host": "robot", "units": "normalized"}
    reading = {"positions": dict.fromkeys(KEYS, 10.), "captured_at": "test", "elevator": 100.}
    add_reference(mapping, "robot", "left_elbow_flex", -90, "angle fixture", reading)
    add_reference(mapping, "robot", "left_elbow_flex", 0, "angle fixture", reading)
    samples = mapping["joints"]["left_elbow_flex"]["samples"]
    assert len(samples) == 2
    assert samples[0]["mujoco_rad"] == pytest.approx(-np.pi/2)
    with pytest.raises(ValueError, match="different host"):
        add_reference(mapping, "other", "left_elbow_flex", 0, "fixture", reading)
