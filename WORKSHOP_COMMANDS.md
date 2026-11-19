# Sourccey workshop — quick commands

Run from the project folder in Git Bash:

```bash
./workshop.cmd
```

In PowerShell: `.\workshop.cmd`.

Click the **3D scene** to give it keyboard focus. Hold movement keys; release to stop.

| Key | Action |
| --- | --- |
| **W / S** | Drive forward / backward |
| **A / D** | Strafe left / right |
| **Q / E** | Turn left / right |
| **Tab** | Switch left / right arm |
| **Up / Down arrows** | Move active hand forward / backward |
| **Left / Right arrows** | Move active hand sideways |
| **R / F** | Raise / lower active hand |
| **J / L** | Bend active wrist |
| **U / O** | Rotate active wrist |
| **X / Z** | Open / close active gripper |
| **T / G** | Raise / lower elevator |
| **1–6**, then **[ / ]** | Select and move an individual joint |
| **Shift + movement key** | Move at quarter speed |
| **Space** | Stop driving and hold the arms |
| **P** | Pause / resume physics |
| **Backspace** | Reset the robot and all boxes |
| **C** | Cycle workshop / close / eye / hand views |
| **V** | Toggle camera following the robot |
| **B** | Show / hide all four onboard camera feeds |
| **H** | Show / hide the keyboard guide |
| **Esc** | Close the workshop |
| **Mouse drag / scroll** | Orbit / zoom the view |

Joint numbers: **1** shoulder pan, **2** shoulder lift, **3** elbow,
**4** wrist bend, **5** wrist rotation, **6** gripper.

Try **3**, then hold **[** briefly to bend the elbow before reaching.

Optional: open the workshop and automatically close it after one minute:

```bash
./workshop.cmd --seconds 60
```

Simulation only. Closing the window stops the program and releases its renderers.
