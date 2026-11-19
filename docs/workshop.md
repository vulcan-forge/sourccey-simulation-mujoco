# The Box Workshop

Launch with `./workshop.cmd` from Git Bash, or `.\workshop.cmd` in PowerShell.
The [keyboard quick reference](../WORKSHOP_COMMANDS.md) lists all controls.

This is a manual, simulation-only MuJoCo 4 Studio application. Sourccey starts
at the existing measured workbench. Three more matching tables form color-coded
sorting, transfer and dispatch stations, with clear paths between stations.
Each table holds four free cardboard boxes. All tables measure 1828.8 x 908.05 mm
and stand 762 mm above the floor; boxes use the owner's 58.8 x 38.4 x 58.4 mm
measurements and the supplied estimated 120 g mass.

Use WASD/QE to drive, Tab to switch arms, arrows/RF to move the hand, JL/UO to
adjust wrists, and X/Z to open/close the gripper. Individual joint jogging is
available through 1–6 followed by [ or ]. Hold Shift for fine adjustments.
The left panel also provides pause, hold, reset, camera selection and an
optional four-camera preview. Click the 3D scene after editing a UI control to
return keyboard focus. The hand target refers to the wrist origin, so adjust
wrist angle and use the hand camera to align the actual fingers with a box.

Boxes move through contact and friction; there are no grasp welds, scripted
pickups or object attachment shortcuts. The floating base uses the project's
existing wheel/traction model. Robot self-collision and decomposed arm meshes
remain enabled, with solid torso envelopes. Joint targets respect model limits;
short target lookahead bounds command buildup against obstacles. Input is local
to the workshop window and a 250 ms heartbeat watchdog stops stale drive input.
Space holds arm joints while preserving the gripper's squeeze target.

The UI reports the actual physics/wall-clock ratio. On slower machines the
simulation slows rather than skipping physical time steps. Rendering uses a
lighter presentation model with exactly the same robot and box coordinates;
only invisible collision hulls are omitted from GPU resources. Full collision
geometry remains in the physics model. Optional camera feeds update at 5 Hz.

Use `C` for the close workbench/eye/hand views and mouse drag/scroll to choose
an angle. The default overview shows all four stations. The camera calibration,
motor parameters, contact coefficients and wheel model retain the project's
existing simulation assumptions; this sandbox is not a physical calibration or
a newly collected training dataset.

Optional command-line tools:

```bash
# Automatically close the interactive application after a minute.
./workshop.cmd --seconds 60
# Save a rendered preview without opening a desktop window.
./workshop.cmd --render artifacts/workshop/overview.png
# Run two simulated seconds without a viewer.
./workshop.cmd --headless --seconds 2
```

Closing the workshop or pressing Esc releases its renderers and ends the Python
process. This application renders the simulation directly and starts no desktop
screen-capture helper.
