"""Drive to a table, pick up a block with an assisted grasp, and toss it aside.

Run from the repository root: .venv/Scripts/python.exe -m examples.table_reach
Add --view to watch the motion, or --render artifacts/table_reach.png for a still.
The grasp uses a temporary MuJoCo weld after the gripper reaches the object.
This demonstrates arm motion and release, not validated finger contact physics.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from sourccey.app import scene_options
from sourccey.build import MODEL
from sourccey.simulation import Simulation


def scene_model(side):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    table_y = 0.18 if side == "left" else -0.18
    spec.worldbody.add_geom(
        name="demo_table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.85, table_y, 0.68], size=[0.25, 0.25, 0.02],
        contype=1, conaffinity=3, rgba=[0.48, 0.32, 0.19, 1],
    )
    for index, x in enumerate((0.66, 1.04)):
        for end, y in enumerate((table_y - 0.18, table_y + 0.18)):
            spec.worldbody.add_geom(
                name=f"demo_table_leg_{index}_{end}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, 0.33], size=[0.025, 0.025, 0.33],
                contype=1, conaffinity=3, rgba=[0.34, 0.23, 0.15, 1],
            )
    block = spec.worldbody.add_body(
        name="demo_object", pos=[0.70, table_y, 0.74],
        mass=0.08, ipos=[0, 0, 0], inertia=[0.0001, 0.0001, 0.0001],
    )
    block.add_freejoint()
    block.add_geom(
        name="demo_block", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.04],
        # The object contacts the table and floor. During the assisted grasp,
        # omit unreliable robot/object convex-hull contacts.
        contype=2, conaffinity=1, friction=[1, 0.005, 0.0001],
        rgba=[0.95, 0.55, 0.12, 1],
    )
    block.add_site(name="demo_object_grasp_site", pos=[0, 0, 0], size=[0.003])
    spec.body("Gripper_Base_v1_1" if side == "left" else "Gripper_Base_v1_2").add_site(
        name="demo_palm_grasp_site", pos=[0, 0, 0], size=[0.003],
    )
    spec.add_equality(
        name="demo_assisted_grasp", type=mujoco.mjtEq.mjEQ_WELD,
        name1="demo_palm_grasp_site", name2="demo_object_grasp_site",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        active=0,
    )
    model = spec.compile()
    # The robot's existing startup keyframe predates this free body. MjSpec
    # extends its qpos with zeros, so seed the new object's proper pose.
    model.key_qpos[model.key("startup").id, -7:] = model.qpos0[-7:]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--view", action="store_true", help="Watch the live MuJoCo viewer")
    parser.add_argument("--render", type=Path, help="Save a still image of the final pose")
    args = parser.parse_args()
    sim = Simulation(model=scene_model(args.side))
    viewer = None
    if args.view:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(sim.model, sim.data)
        viewer.opt.geomgroup[3] = 0
        viewer.opt.sitegroup[:] = 0
        viewer.cam.lookat[:] = [0.48, 0.0, 0.75]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

    names = [args.side + "_" + role for role in ("shoulder_pan", "shoulder_lift", "elbow_flex")]
    dofs = [sim.model.jnt_dofadr[sim.joints[name]] for name in names]
    max_joint_speed_rad_s = 0.0

    def advance(steps):
        nonlocal max_joint_speed_rad_s
        for _ in range(steps):
            sim.step()
            max_joint_speed_rad_s = max(max_joint_speed_rad_s, float(np.max(np.abs(sim.data.qvel[dofs]))))
            if viewer is not None:
                viewer.sync()
                time.sleep(sim.model.opt.timestep)

    try:
        sim.set_gripper(args.side, 0)
        start_x = float(sim.data.qpos[0])
        sim.set_base(forward=0.25)
        while sim.data.qpos[0] - start_x < 0.20 and sim.data.time < 4:
            advance(1)
        sim.set_base()
        advance(250)
        drive_m = float(sim.data.qpos[0] - start_x)

        block_id = sim.model.body("demo_object").id
        block_start = sim.data.xpos[block_id].copy()
        # The reach site is at the wrist, slightly behind the fingers.
        above_world = block_start + [-0.035, 0, 0.17]
        approach_world = block_start + [-0.035, 0, 0.105]
        def local(world):
            return sim.robot_rotation.T @ (world - sim.mount_position(args.side))

        largest_target_step_rad = 0.0
        worst_ik_error_m = 0.0

        def move_to(world, segments=60, steps_per_segment=8):
            nonlocal largest_target_step_rad, worst_ik_error_m
            source, destination = sim.ee_local(args.side), local(world)
            for fraction in np.linspace(0, 1, segments + 1)[1:]:
                target = source + fraction * (destination - source)
                old = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
                result = sim.solve_ik(args.side, target)
                new = np.array([sim.data.ctrl[sim.actuators[name]] for name in names])
                largest_target_step_rad = max(largest_target_step_rad, float(np.max(np.abs(new - old))))
                worst_ik_error_m = max(worst_ik_error_m, result["error_m"])
                advance(steps_per_segment)

        move_to(above_world)
        move_to(approach_world)
        advance(250)
        actual = sim.data.site_xpos[sim.model.site(args.side + "_ee").id].copy()
        final_error_m = float(np.linalg.norm(actual - approach_world))
        # Finger hulls in the source CAD do not make a validated pinch contact.
        # Attach only after the arm reaches the object, preserving its current
        # relative pose so activation does not teleport it into the hand.
        block_distance_m = float(np.linalg.norm(actual - sim.data.xpos[block_id]))
        if final_error_m > 0.025 or block_distance_m > 0.14:
            raise RuntimeError(f"Grasp approach missed: target error {final_error_m:.3f} m, "
                               f"wrist-to-block {block_distance_m:.3f} m, "
                               f"base {sim.data.qpos[:3]}, wrist {actual}, target {approach_world}, "
                               f"block {sim.data.xpos[block_id]}, warnings {sim.state()['warnings']}")
        sim.set_gripper(args.side, 100)
        advance(200)
        grip_id = sim.model.eq("demo_assisted_grasp").id
        palm_id = sim.model.body("Gripper_Base_v1_1" if args.side == "left" else "Gripper_Base_v1_2").id
        palm_p, block_p = sim.data.xpos[palm_id].copy(), sim.data.xpos[block_id].copy()
        palm_r = Rotation.from_quat(sim.data.xquat[palm_id][[1, 2, 3, 0]])
        block_r = Rotation.from_quat(sim.data.xquat[block_id][[1, 2, 3, 0]])
        # Align the two site frames at the current configuration before
        # enabling the weld. Their relative transform is then maintained.
        object_site = sim.model.site("demo_object_grasp_site").id
        sim.model.site_pos[object_site] = block_r.inv().apply(palm_p - block_p)
        sim.model.site_quat[object_site] = (block_r.inv() * palm_r).as_quat()[[3, 0, 1, 2]]
        sim.data.eq_active[grip_id] = True
        advance(100)
        lift_world = approach_world + [0, 0, 0.20]
        move_to(lift_world)
        move_to(lift_world + [0.0, (0.25 if args.side == "left" else -0.25), 0.0],
                segments=125, steps_per_segment=12)
        before_release = sim.data.xpos[block_id].copy()
        sim.data.eq_active[grip_id] = False
        sim.set_gripper(args.side, 0)
        advance(450)
        after_release = sim.data.xpos[block_id].copy()
        object_lift_m = float(before_release[2] - block_start[2])
        object_travel_m = float(np.linalg.norm(after_release[:2] - block_start[:2]))
        report = {
            "arm": args.side,
            "base_drive_m": round(drive_m, 4),
            "grasp_approach_error_m": round(final_error_m, 5),
            "object_lift_m": round(object_lift_m, 4),
            "object_travel_after_release_m": round(object_travel_m, 4),
            "grasp_mode": "assisted weld, released for free-flight physics",
            "worst_ik_solver_error_m": round(worst_ik_error_m, 5),
            "largest_joint_target_step_rad": round(largest_target_step_rad, 5),
            "max_measured_joint_speed_rad_s": round(max_joint_speed_rad_s, 4),
            "warnings": sim.state()["warnings"],
        }
        report["passed"] = (0.18 <= drive_m <= 0.26 and final_error_m < 0.025
                            and object_lift_m > 0.12 and object_travel_m > 0.08
                            and worst_ik_error_m < 0.01 and largest_target_step_rad < 0.2
                            and max_joint_speed_rad_s < 6
                            and not report["warnings"])
        if args.render:
            args.render.parent.mkdir(parents=True, exist_ok=True)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.48, 0, 0.75]
            camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -18
            with mujoco.Renderer(sim.model, height=900, width=1200) as renderer:
                renderer.update_scene(sim.data, camera=camera, scene_option=scene_options())
                Image.fromarray(renderer.render()).save(args.render)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        if viewer is not None:
            print("Close the MuJoCo viewer when you are done inspecting the result.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
    finally:
        sim.set_base()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
