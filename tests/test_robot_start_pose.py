import copy
import json

import pytest

from examples.robot_start_pose import KEYS, arms_down_profile, capture, check_feedback, execute, interpolate, plan, validate_host_limits, validate_profile


def observation(value=0):
    return {"positions": {key: (50 if key.endswith("gripper.pos") else value) for key in KEYS},
            "elevator": 12., "base_velocity": [0., 0., 0., 0.]}


def profile():
    return {"version": 1, "host": "test", "units": "normalized",
            "pose": "operator-positioned arms down", "positions": observation(10)["positions"],
            "elevator": 12.}


def test_start_plan_uses_captured_joint_readings_and_retimes_motion():
    normal = plan(profile(), observation(), 1)
    slow = plan(profile(), observation(), .5)
    assert slow["duration_s"] == 2 * normal["duration_s"]
    assert interpolate(slow, 0) == observation()["positions"]
    assert interpolate(slow, slow["duration_s"]) == profile()["positions"]
    commands = [interpolate(slow, i * slow["duration_s"] / 1000)[KEYS[0]] for i in range(1001)]
    dt = slow["duration_s"] / 1000
    velocities = [(b-a)/dt for a, b in zip(commands, commands[1:])]
    accelerations = [(b-a)/dt for a, b in zip(velocities, velocities[1:])]
    assert min(velocities) >= 0
    assert max(velocities) <= slow["command_velocity_limit_units_s"]
    assert max(map(abs, accelerations)) <= slow["command_acceleration_limit_units_s2"]


def test_wrong_host_large_move_elevator_change_and_bad_units_rejected():
    with pytest.raises(ValueError, match="host"):
        validate_profile(profile(), "other")
    with pytest.raises(ValueError, match="30 normalized"):
        plan(profile(), observation(-40), .5)
    changed = observation()
    changed["elevator"] = 20
    with pytest.raises(ValueError, match="Elevator"):
        plan(profile(), changed, .5)
    changed = observation()
    changed["positions"][KEYS[0]] = float("nan")
    with pytest.raises(ValueError):
        plan(profile(), changed, .5)


def test_double_speed_return_halves_duration_and_scales_limits():
    normal = plan(profile(), observation(), 1.)
    fast = plan(profile(), observation(), 2.)
    assert fast["duration_s"] == normal["duration_s"] / 2
    assert fast["command_velocity_limit_units_s"] == normal["command_velocity_limit_units_s"] * 2
    assert fast["command_acceleration_limit_units_s2"] == normal["command_acceleration_limit_units_s2"] * 4
    assert interpolate(fast, fast["duration_s"] / 3) == interpolate(normal, normal["duration_s"] / 3)


def test_full_gripper_return_is_opt_in_and_keeps_arm_distance_limit():
    saved = profile()
    saved["positions"]["left_gripper.pos"] = 99.
    observed = observation()
    observed["positions"]["left_gripper.pos"] = 2.
    with pytest.raises(ValueError, match="30 normalized"):
        plan(saved, observed, 1.)
    motion = plan(saved, observed, 1., max_gripper_delta=100)
    assert motion["maximum_delta_units"] == 97.
    assert motion["duration_s"] >= 1.875 * 97 / 10
    observed["positions"][KEYS[0]] = -40.
    with pytest.raises(ValueError, match="30 normalized"):
        plan(saved, observed, 1., max_gripper_delta=100)


def test_feedback_rejects_tracking_error_and_base_motion():
    with pytest.raises(ValueError, match=r"tracking.*left_shoulder_pan.pos.*observed 20.000.*commanded 0.000"):
        check_feedback(observation(20), observation()["positions"], 12)
    changed = observation()
    changed["base_velocity"][0] = .1
    with pytest.raises(ValueError, match="Base"):
        check_feedback(changed, observation()["positions"], 12)


def test_host_guard_rejects_recorded_down_target_before_actuation():
    policy = {"version": 1, "host": "test", "units": "normalized", "limits": {
        "left_shoulder_lift.pos": {"min": -89.2180410995463, "max": 89.2180410995463},
        "right_shoulder_lift.pos": {"min": -89.22091782283886, "max": 89.1675560298826}}}
    motion = plan(profile(), observation(), .5)
    validate_host_limits(motion, policy, "test")
    motion["target"]["left_shoulder_lift.pos"] = -99.0925
    with pytest.raises(ValueError, match=r"guard rejects target.*No movement commands sent"):
        validate_host_limits(motion, policy, "test")
    motion["target"]["left_shoulder_lift.pos"] = 10
    motion["start"]["right_shoulder_lift.pos"] = 98.0258
    with pytest.raises(ValueError, match="guard rejects start"):
        validate_host_limits(motion, policy, "test")


def test_lower_arms_reuses_down_captures_without_manual_start_snapshot():
    checkpoint = {"version": 1, "source": {"host": "test"}, "poses": {}}
    for side, lift, hardware_lift in (("left", -1.5708, -98), ("right", 1.5708, 99)):
        checkpoint["poses"][side + "_down"] = {
            "mujoco_shoulder_rad": {"shoulder_lift": lift},
            "hardware_positions": {"shoulder_pan": 10, "shoulder_lift": hardware_lift,
                                   "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 4}}
    observed = observation()
    target = arms_down_profile(checkpoint, "test", observed)
    assert target["positions"]["left_shoulder_lift.pos"] == -98
    assert target["positions"]["right_shoulder_lift.pos"] == 99
    for side in ("left", "right"):
        assert target["positions"][side + "_gripper.pos"] == observed["positions"][side + "_gripper.pos"]
    motion = plan(target, observed, .5, max_delta=200)
    assert motion["duration_s"] > 30  # Large returns are slow, not a direct target jump.
    assert interpolate(motion, motion["duration_s"])["right_shoulder_lift.pos"] == 99
    with pytest.raises(ValueError, match="different host"):
        arms_down_profile(checkpoint, "other", observed)


def test_capture_saves_both_arms_without_sending_commands():
    class ReadOnly:
        def read(self):
            return observation()
    saved = capture(ReadOnly(), "test")
    validate_profile(saved, "test")
    assert saved["positions"] == observation()["positions"]


def test_hold_current_cli_never_targets_an_old_pose(monkeypatch, tmp_path):
    from examples import robot_start_pose
    policy = {"version": 1, "host": "test", "units": "normalized", "limits": {
        side + "_shoulder_lift.pos": {"min": -100, "max": 100} for side in ("left", "right")}}
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps(policy))

    class Link:
        def __init__(self, host):
            assert host == "test"
        def read(self):
            return observation(7)
        def close(self):
            pass

    checked = []
    def inspect(link, motion, elevator, on_sample):
        assert motion["start"] == motion["target"] == observation(7)["positions"]
        assert motion["maximum_delta_units"] == 0
        assert motion["hold_current_only"] and motion["duration_s"] == 2
        checked.append(True)
        return {"start_reached": True}

    monkeypatch.setattr(robot_start_pose, "HardwareLink", Link)
    monkeypatch.setattr(robot_start_pose, "execute", inspect)
    monkeypatch.setattr("sys.argv", ["robot_start_pose", "hold-current", "--robot-ip", "test",
                                    "--normalized", "--execute", "--host-limits", str(limits),
                                    "--trace", str(tmp_path / "hold.jsonl")])
    robot_start_pose.main()
    assert checked


def test_execute_verifies_settled_feedback_and_holds_on_timeout(monkeypatch):
    clock = [0.]
    monkeypatch.setattr("examples.robot_start_pose.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("examples.robot_start_pose.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    class FakeLink:
        def __init__(self, fail=False):
            self.current = observation()
            self.sent = []
            self.fail = fail

        def read(self):
            clock[0] += 1/30
            if self.fail and len(self.sent) == 3:
                raise TimeoutError("stale feedback")
            return copy.deepcopy(self.current)

        def send(self, target):
            self.sent.append(target.copy())
            self.current["positions"] = target.copy()

    motion = plan(profile(), observation(), .5)
    link = FakeLink()
    assert execute(link, motion, 12)["start_reached"]
    assert link.sent[-1] == profile()["positions"]
    link = FakeLink(fail=True)
    with pytest.raises(TimeoutError):
        execute(link, motion, 12)
    assert link.sent[-1] == link.sent[-2]
    assert link.sent[-1] != profile()["positions"]
    link = FakeLink()
    link.current["base_velocity"][0] = .1
    with pytest.raises(ValueError, match="Base"):
        execute(link, motion, 12)
    assert link.sent == []


@pytest.mark.parametrize("follower_step", [0., .02])
def test_lagging_joint_pauses_ramp_and_reports_stall(monkeypatch, follower_step):
    clock = [0.]
    monkeypatch.setattr("examples.robot_start_pose.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("examples.robot_start_pose.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    class SlowLink:
        def __init__(self):
            self.current = observation()
            self.target = observation()["positions"]
            self.sent = []

        def read(self):
            clock[0] += 1/30
            for key in KEYS:
                delta = self.target[key] - self.current["positions"][key]
                self.current["positions"][key] += max(-follower_step, min(follower_step, delta))
            return copy.deepcopy(self.current)

        def send(self, target):
            self.target = target.copy()
            self.sent.append(target.copy())

    motion = plan(profile(), observation(), .5)
    link, trace = SlowLink(), []
    if follower_step == 0:
        with pytest.raises(TimeoutError, match=r"catch up.*left_shoulder_pan.pos.*observed 0.000"):
            execute(link, motion, 12, on_sample=trace.append)
        assert max(command[KEYS[0]] for command in link.sent) < 2.1
        assert trace[-1]["phase"] == "aborted"
        assert link.sent[-1] == link.sent[-2]
    else:
        assert execute(link, motion, 12, on_sample=trace.append)["start_reached"]
        assert clock[0] > 1.5 * motion["duration_s"]
    assert any(sample.get("waiting_for_tracking") for sample in trace)
    assert max(sample["tracking"]["error_units"] for sample in trace if "tracking" in sample) < 2.1
