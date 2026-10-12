"""Reproduce the strafe static map from the checked-in calibration JSON.

Requires NumPy and Matplotlib. Reads local files only; no robot connection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("strafe_static.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "sourccey/identified_base.json").read_text())
    axis = data["axes"]["y"]
    holds = np.asarray(data["provenance"]["y"]["holds"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.subplots_adjust(left=.11, right=.97, bottom=.21, top=.91)
    for sign, color, marker, line, label in [
        (1, "tab:blue", "o", "-", "+y (left), measured"),
        (-1, "tab:orange", "s", "--", "−y (right), measured magnitude"),
    ]:
        values = holds[holds[:, 0] * sign > 0] * sign
        values = values[np.argsort(values[:, 0])]
        ax.plot(values[:, 0], values[:, 1], color=color, marker=marker,
                linestyle=line, markerfacecolor=color if sign > 0 else "none",
                lw=1.3, ms=5, label=label)
    table_u = np.r_[0, axis["table_u"]]
    table_v = np.r_[0, axis["v_pos"]]
    ax.plot(table_u, table_v, "k-", lw=1, label="PR table (direction average)")
    ax.axvspan(0, axis["table_u"][0], color="grey", alpha=.18,
               label="Model zero region (assumed)")
    ax.set(title="Strafe static map: |command| → m/s", xlabel="|command|",
           ylabel="steady sideways speed (m/s)", xlim=(-.05, 1.05), ylim=(-.01, .29))
    ax.grid(alpha=.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.text(.11, .10, f"Run {data['provenance']['y']['run']} · 1.8 s holds · corner LiDAR", fontsize=9)
    fig.text(.11, .065, "One hold per sign/level; no repeatability estimate. Lines connect measured points.", fontsize=8.5)
    fig.text(.11, .032, "Zero below 0.40 is a model choice, not a measured universal breakaway threshold.", fontsize=8.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}; {len(holds)} measured holds from the versioned calibration")


if __name__ == "__main__":
    main()
