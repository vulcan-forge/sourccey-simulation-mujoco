"""Reusable recordings of calibration physics, independent of display speed."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sourccey.build import ROOT, MODEL, SOURCE


CACHE = ROOT / "artifacts/true_calibration/playback"
FORMAT_VERSION = 1


def physics_signature(config):
    digest = hashlib.sha256(json.dumps({**config, "format": FORMAT_VERSION,
                                      "mujoco": mujoco.__version__}, sort_keys=True).encode())
    source = (ROOT / "examples/true_calibration.py").read_text(encoding="utf-8")
    names = {"scene_model", "contact_snapshot", "calibration_physics_step", "Clock", "SearchLog",
             "ArmBus", "Lift", "CalibrationRun"}
    # Display-only edits and speed choices do not invalidate an expensive run.
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            digest.update(ast.get_source_segment(source, node).encode())
        elif isinstance(node, ast.Assign):
            digest.update(ast.get_source_segment(source, node).encode())
    for path in (MODEL, ROOT / "sourccey/collision.py", ROOT / "sourccey/simulation.py",
                 ROOT / "integrations/calibration_controller.py",
                 ROOT / "integrations/true_calibration_reference/provenance.json"):
        digest.update(path.read_bytes())
    if config["self_collision"] and config["collision_model"] == "decomposed":
        digest.update((ROOT / "models/collision/manifest.json").read_bytes())
    for mesh in ET.parse(MODEL).findall("asset/mesh"):
        digest.update((SOURCE.parent / mesh.get("file")).read_bytes())
    return digest.hexdigest()


def record_run(run, path, signature):
    frames = {key: [] for key in ("time", "qpos", "qvel", "ctrl", "bias", "phase",
                                  "left", "right", "event_count")}

    def capture():
        data = run.sim.data
        frames["time"].append(float(data.time))
        for key in ("qpos", "qvel", "ctrl"):
            frames[key].append(getattr(data, key).copy())
        frames["bias"].append(run.sim.model.actuator_biasprm[:, 0].copy())
        frames["phase"].append(run.phase)
        for side in ("left", "right"):
            frames[side].append(run.buses[side].active if side in run.buses else "waiting")
        frames["event_count"].append(len(run.events))

    capture()
    step, progress = 0, 0.
    while not run.complete:
        run.step()
        step += 1
        if step % 10 == 0:
            capture()  # 50 Hz at the unchanged 2 ms physics timestep.
        if run.sim.data.time >= progress + 10:
            progress = float(run.sim.data.time)
            print(f"Recording calibration: {progress:.0f} simulated seconds...", flush=True)
    if frames["time"][-1] != float(run.sim.data.time):
        capture()
    else:
        # Include final completion metadata even if it lands on a sample.
        frames["phase"][-1] = "complete"
    metadata = {"signature": signature, "report": run.report()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in frames.items()},
                            metadata=np.asarray(json.dumps(metadata)))
    temporary.replace(path)
    print(f"Calibration recording ready: {path}", flush=True)


def load_recording(path, signature):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    if metadata["signature"] != signature:
        raise ValueError("Recording does not match the current calibration physics")
    result["report"] = metadata["report"]
    return result


def frame_at(recording, sim, position_s):
    times = recording["time"]
    t = float(np.clip(position_s, times[0], times[-1]))
    index = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 1)
    following = min(index + 1, len(times) - 1)
    span = times[following] - times[index]
    weight = (t - times[index]) / span if span > 0 else 0.
    for key in ("qpos", "qvel", "ctrl"):
        getattr(sim.data, key)[:] = (1 - weight) * recording[key][index] + weight * recording[key][following]
    mujoco.mj_normalizeQuat(sim.model, sim.data.qpos)
    sim.model.actuator_biasprm[:, 0] = recording["bias"][index]
    sim.data.time = t
    # Rendering needs transforms, not a new physics/contact solve every frame.
    mujoco.mj_kinematics(sim.model, sim.data)
    mujoco.mj_comPos(sim.model, sim.data)
    return index, t >= times[-1]
