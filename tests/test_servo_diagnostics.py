from dataclasses import dataclass
from types import SimpleNamespace

from examples.servo_diagnostics import REGISTERS, install_hooks, read_motor


@dataclass
class Calibration:
    range_min: int = 0
    range_max: int = 1000


def test_register_inspection_only_reads_existing_bus():
    class Bus:
        motors = {"shoulder_lift": SimpleNamespace(id=2, model="sts3250", norm_mode="normalized")}
        calibration = {"shoulder_lift": Calibration()}

        def __init__(self):
            self.reads = []

        def read(self, register, name, normalize):
            assert normalize is False
            self.reads.append(register)
            if register == "Present_Current":
                raise ValueError("unsupported register")
            return {"Goal_Position": 900, "Present_Position": 850}.get(register, 0)

        def _normalize(self, values):
            return {key: value / 5 - 100 for key, value in values.items()}

    bus = Bus()
    result = read_motor(bus, "shoulder_lift")
    assert bus.reads == list(REGISTERS)
    assert result["Goal_Position_normalized"] == 80
    assert result["Present_Position_normalized"] == 70
    assert result["goal_minus_present_units"] == 10
    assert "Present_Current" in result["read_errors"]


def test_hook_preserves_existing_commands_and_can_be_removed():
    class Robot:
        def __init__(self):
            self.sent = []

        def get_observation(self):
            return {"test": 1}

        def send_action(self, action):
            self.sent.append(action.copy())
            return action

    original = Robot.send_action
    log = []
    restore = install_hooks(Robot, log.append, 2)
    robot = Robot()
    action = {"left_shoulder_lift.pos": -99}
    try:
        assert robot.send_action(action) == action
        assert robot.sent == [action]
        assert log[0]["action"] == action
        # Diagnostic failure does not disrupt normal observations.
        assert robot.get_observation() == {"test": 1}
    finally:
        restore()
    assert Robot.send_action is original
