"""Physical interaction, manual commands and presentation-state synchronization."""
import mujoco
import numpy as np
import pytest

from sourccey.sandbox import Workshop, Controls, build_scene, copy_display_state


@pytest.fixture(scope="module")
def scene():
    return Workshop()


@pytest.fixture
def workshop(scene):
    scene.reset()
    return scene


def test_free_robot_and_sixteen_measured_boxes(workshop):
    m = workshop.sim.model
    assert m.neq == 0
    assert workshop.scene["box_count"] == 16
    for i in range(4):
        table = m.geom(f"table_{i}")
        assert table.size == pytest.approx([.90805/2, 1.8288/2, .04/2])
        assert table.pos[2]+table.size[2] == pytest.approx(.762)
        for j in range(4):
            assert m.body(f"workshop_box_{i}_{j}").mass[0] == pytest.approx(.12)
            assert m.geom(f"workshop_box_geom_{i}_{j}").size == pytest.approx([.0192, .0294, .0292])
            assert m.joint(f"workshop_box_free_{i}_{j}").type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_boxes_settle_physically_on_their_tables(workshop):
    for _ in range(3):
        workshop.advance(.1, now=0)
    sim = workshop.sim
    for i in range(4):
        for j in range(4):
            z = sim.data.xpos[sim.model.body(f"workshop_box_{i}_{j}").id, 2]
            assert z == pytest.approx(.762+.0292, abs=.001)
    assert not sim.state()["warnings"]


def test_base_drive_and_expiring_input(workshop):
    sim = workshop.sim
    start = sim.data.qpos[:3].copy()
    workshop.receive(Controls(drive=(-1, 0, 0)), now=0)
    for _ in range(5):
        workshop.advance(.1, now=0)
    assert sim.data.qpos[0] < start[0]-.04
    workshop.advance(.1, now=1)
    assert sim.base_command == pytest.approx([0, 0, 0])


def test_joint_and_gripper_input_are_side_specific_and_hold_keeps_grasp(workshop):
    sim = workshop.sim
    before = sim.data.ctrl.copy()
    workshop.receive(Controls(side="right", joint=2, jog=-1, grip=1), now=0)
    for _ in range(3):
        workshop.advance(.1, now=0)
    assert sim.data.ctrl[sim.actuators["right_elbow_flex"]] < -.15
    assert sim.data.ctrl[sim.actuators["right_gripper"]] > before[sim.actuators["right_gripper"]]
    assert sim.data.ctrl[sim.actuators["left_elbow_flex"]] == before[sim.actuators["left_elbow_flex"]]
    grip = sim.data.ctrl[sim.actuators["right_gripper"]]
    workshop.hold()
    assert sim.data.ctrl[sim.actuators["right_gripper"]] == grip


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_position_input_raises_real_physics_arm(workshop, side):
    initial = workshop.sim.ee_local(side).copy()
    workshop.receive(Controls(side=side, hand=(0, 0, 1)), now=0)
    for _ in range(4):
        workshop.advance(.1, now=0)
    assert workshop.sim.ee_local(side)[2] > initial[2]+.015
    assert not workshop.sim.state()["warnings"]


def test_pause_reset_and_post_reset_input(workshop):
    workshop.advance(.1, now=0)
    workshop.paused = True
    timestamp = workshop.sim.data.time
    workshop.advance(.1, now=0)
    assert workshop.sim.data.time == timestamp
    workshop.reset()
    assert workshop.sim.data.time == 0
    assert not workshop.paused
    workshop.receive(Controls(joint=2, jog=-1), now=0)
    workshop.advance(.1, now=0)
    assert workshop.sim.data.ctrl[workshop.sim.actuators["left_elbow_flex"]] < 0
    assert workshop.sim.data.qpos[-7:-4] == pytest.approx(workshop.sim.model.qpos0[-7:-4], abs=.002)


def test_input_validation(workshop):
    for command in (Controls(side="wrong"), Controls(joint=6), Controls(drive=(np.nan, 0, 0))):
        with pytest.raises(ValueError):
            workshop.receive(command)
    with pytest.raises(ValueError):
        workshop.advance(1)


def test_presentation_copy_keeps_every_robot_and_box_coordinate(workshop):
    model, _ = build_scene(display_only=True)
    data = mujoco.MjData(model)
    workshop.receive(Controls(joint=2, jog=-1), now=0)
    workshop.advance(.1, now=0)
    copy_display_state(workshop.sim.data, model, data)
    assert model.ngeom < workshop.sim.model.ngeom/10
    assert data.qpos == pytest.approx(workshop.sim.data.qpos)
    assert data.time == workshop.sim.data.time
    for name in ("base_link", "Gripper_Base_v1_1", "workshop_box_0_0", "workshop_box_3_3"):
        assert data.xpos[model.body(name).id] == pytest.approx(
            workshop.sim.data.xpos[workshop.sim.model.body(name).id])
