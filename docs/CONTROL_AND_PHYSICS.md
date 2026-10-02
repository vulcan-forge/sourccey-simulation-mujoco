# Control and physics notes

## Python control

Run from the repository root using `.venv/Scripts/python.exe`:

```python
from sourccey.simulation import Simulation

sim = Simulation()
sim.set_base(forward=0.5, left=0, yaw=0)  # normalized, robot-relative
sim.set_elevator(-0.2)                   # actuator position in meters
sim.set_gripper("left", closure=100)     # 0=open, 100=closed
sim.set_joints({"right_elbow_flex": -2.1})  # radians
sim.step(1000)                          # 2 seconds at dt=0.002
sim.set_base()                          # brake all four wheels

target = sim.ee_local("left") + [0.02, 0, 0.02]
result = sim.solve_ik("left", target)    # convergence and residual
sim.step(500)                           # drives approach the target
print(result, sim.state())
```

`set_joints({...})` accepts joint radians (meters for the elevator), clamps to the
generated MuJoCo limits, and rejects non-finite/unknown input atomically. Wheels
use velocity targets.
The elevator's upper limit is −0.0142 m; the default lower control endpoint is
−0.3104 m. The optional full travel mode uses −0.315 m as the lower endpoint.
Both elbows start at −2.0 rad.
IK uses scratch data and never teleports live physics. The reach site is the
wrist-roll origin, not the fingertip. Wrist flex is held at its commanded angle
while pan/lift/elbow solve position; roll and gripper opening do not move the site.

Reach targets are expressed in the moving shoulder frame, so they follow the
chassis and elevator. The panel can freeze target-slider input while holding the
last local target.

## Equivalent mecanum contact

Wheel mixing uses `r=0.052 m`, `L=0.30 m`, forward v, left s, yaw w:

```text
FL = (v - s - L*w) / r
FR = (v + s + L*w) / r
RL = (v + s - L*w) / r
RR = (v - s + L*w) / r
```

Right CAD wheel commands are negated. Contact centers come from actual wheel
mesh bounds projected onto the CAD axles. Their measured half-length plus
half-width is about 0.29854 m; the nominal lever is rounded to 0.30 m.

Each wheel has a 52 mm sphere for normal floor contact. MuJoCo supplies the normal
load N. Python uses the contact-point Jacobian to compute velocity, including both
chassis motion and wheel spin. The resistance direction d is in the robot plane:

```text
FL/RR d = normalize([1, -1, 0])
FR/RL d = normalize([1,  1, 0])
slip = dot(d, contact_point_velocity)
F = d * clamp(-1800 * slip, -0.9*N, +0.9*N)
```

`mj_applyFT` applies F at the wheel's contact point, reacting on both the wheel
axle and floating chassis. The perpendicular roller direction is frictionless.
No chassis pose or velocity is prescribed while stepping. Airborne wheels have
no traction. Disabling traction leaves the chassis stationary while wheels spin.

This finite-slip approximation supports a flat stationary plane. It is neither
roller-resolved nor identified from hardware. Tangential forces are evaluated
once per 2 ms step using current normal loads. The XML alone lacks this Python law.

## Drives and collision

| Drive | kp / kv | Force limit | Added inertia |
| --- | --- | --- | --- |
| Arm/wrist/gripper position | 500 / 35 | ±100 N·m | 0.04 kg·m² |
| Elevator position | 30000 / 3000 | ±3000 N | 0.04 kg |
| Wheel velocity | kv=8 | ±30 N·m | 0.01 kg·m² |

Non-wheel drives receive joint bias-force compensation through the actuator's
constant bias term; total actuator output still obeys its force cap. Source
million-unit arm effort/speed entries are not interpreted as motor ratings.

The full and standalone source URDFs specify `elbow_flex` limits of −90° to +90°.
The generated MuJoCo model extends only the lower end for both elbows to −135°;
the actuator target range uses the same value. CAD renders confirm this moves
both forearms downward, while the full copied URDF remains unmodified. The
physical elbow stop and self-clearance have not been measured, so do not treat
−135° as a hardware-safe limit.

Visual STL geoms are group 2, non-colliding. Group 3 contains the chassis box,
four wheel spheres, and individual convex arm hulls; it is hidden by the viewer.
Robot geoms use contact bit 2 against environment bit 1, disabling robot self-contact.
New environment objects need compatible contact bits. The tall shell is visual
only; the box approximates the lower chassis. Pad-clearance checks use original
STL vertices to establish a positive separating-plane gap between the pads,
not a complete assembly collision audit or a validated grasping simulation.
