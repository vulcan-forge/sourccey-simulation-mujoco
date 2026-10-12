"""Plot saved corner-LiDAR data against the versioned response; no robot connection.

Usage: python tools/plot_strafe_dynamics.py /path/to/y_20261012_100428
Requires matplotlib and numpy. The stem must have .csv and _corner.npz siblings.
"""
import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sourccey.control import IdentifiedBaseResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path)
    parser.add_argument("--output", type=Path, default=Path("strafe_dynamics.png"))
    args = parser.parse_args()
    with args.stem.with_suffix(".csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "cmd"]
    tc = np.array([float(r["t"]) for r in rows])
    u = np.array([float(r["value"]) for r in rows])
    assert np.all(np.diff(tc) > 0) and tc[0] == 0
    with np.load(str(args.stem) + "_corner.npz") as corner:
        t, position = corner["t"], corner["lat"]
    measured = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        select = np.abs(t - ti) <= 0.2
        if select.sum() >= 3:
            measured[i] = np.polyfit(t[select] - ti, position[select], 1)[0]
    good = np.isfinite(measured) & (t >= 0) & (t <= tc[-1])
    t, measured = t[good], measured[good]

    # Include every command and measurement time: step() receives the command
    # at the start of each interval and returns the velocity at its end.
    grid = np.unique(np.concatenate(([0.0], tc, t, np.arange(0, tc[-1], .01))))
    cmd_indices = np.searchsorted(tc, grid, side="right") - 1
    command = u[cmd_indices]
    response = IdentifiedBaseResponse()
    prediction = np.zeros(len(grid))
    for i in range(1, len(grid)):
        prediction[i] = response.step([0, command[i-1], 0], grid[i]-grid[i-1])[1]
    axis = response.axes[1]
    static = np.array([response.steady_velocity([0, c, 0])[1] for c in u])
    no_lag = static[cmd_indices]
    delayed_indices = np.searchsorted(tc, grid-axis["delay"], side="right") - 1
    delayed = np.where(delayed_indices >= 0, static[np.maximum(delayed_indices, 0)], 0)
    measured_indices = np.searchsorted(grid, t)
    rms = lambda values: np.sqrt(np.mean((values[measured_indices]-measured)**2))
    model_rms, delay_rms = rms(prediction), rms(delayed)

    # Match the earlier yaw figure: plain axes, black measurements, colored
    # model curves, grey commands and two detailed transition windows.
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, figsize=(15, 12.6))
    fig.subplots_adjust(left=.075, right=.955, top=.94, bottom=.09, hspace=.28)
    fig.suptitle(f"Strafe dynamics · {args.stem.name}", fontsize=14, y=.975)
    for ax in axes:
        ax.grid(True, alpha=.3)
        ax.set(xlabel="t (s)", ylabel="sideways speed (m/s)")
    axes[0].plot(t, measured, "k", lw=.8, label="LiDAR speed (0.4 s local slope)")
    axes[0].plot(grid, delayed, color="dodgerblue", lw=1,
                 label=f"Table + delay only (RMS {delay_rms:.3f} m/s)")
    axes[0].plot(grid, prediction, color="darkorange", lw=1.2,
                 label=f"PR model: table + first-order lag + delay (RMS {model_rms:.3f} m/s)")
    command_axis = axes[0].twinx()
    command_axis.step(tc, u, where="post", color="grey", alpha=.4, lw=.8)
    command_axis.set(ylabel="command y", ylim=(-1.15, 1.15))
    axes[0].set(title="Sideways speed: measured vs response model (whole run)",
                xlim=(0, tc[-1]), ylim=(-.34, .34))
    axes[0].legend(loc="lower left", fontsize=8)
    for ax, (start, end, level) in zip(axes[1:], [(56, 64, 80), (81, 89, 100)]):
        selected = (t >= start) & (t <= end)
        ax.plot(t[selected], measured[selected], "k.-", lw=.8, ms=3,
                label="LiDAR speed (0.4 s local slope)")
        ax.plot(grid, prediction, color="darkorange", lw=1.2,
                label=f"PR model (τ {axis['tau']:.2f} s, delay {axis['delay']:.2f} s)")
        ax.step(grid, no_lag, where="post", color="grey", ls="--", lw=1,
                label="Static map of command (no lag or delay)")
        ax.set(title=f"Zoom {start}–{end} s · ±{level}% command", xlim=(start, end))
        bound = max(np.abs(measured[selected]).max(), np.abs(prediction[(grid >= start) & (grid <= end)]).max())
        ax.set_ylim(-bound*1.22, bound*1.22)
        ax.legend(loc="lower left", fontsize=8)
    fig.text(.075, .04, "Offline response replay, not MuJoCo chassis motion. RMS uses this same run; no held-out validation.", fontsize=9)
    fig.text(.075, .023, "LiDAR differentiation smooths transitions over 0.4 s; lag/delay remain provisional. Other-axis recentering is not modeled.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    plt.close(fig)
    print(f"{len(t)} measured samples; PR response RMS {model_rms:.6f} m/s; delay-only RMS {delay_rms:.6f} m/s")


if __name__ == "__main__":
    main()
