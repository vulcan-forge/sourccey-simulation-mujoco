"""Simulation-only policy bridge and objective box-lift measurements."""
from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np

from .build import ROLES, ROOT


CAMERAS = ("front_left", "front_right", "wrist_left", "wrist_right")
JOINT_NAMES = tuple("left_"+role for role in ROLES)


def select_sweep(provenance, splits, pilot, split, limit=None):
    """Freeze deterministic, split-checked scenes before loading the model."""
    if split not in ("train", "validation"):
        raise ValueError("Sweep accepts train or validation; final test is reserved")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    rows = sorted((r for r in provenance["episodes"] if r["split"] == split),
                  key=lambda r:r["episode_index"])
    if {r["episode_index"] for r in rows} != set(splits[split]):
        raise ValueError("Provenance disagrees with episode split manifest")
    if len({r["source_episode"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source scene in sweep")
    selected = []
    for row in rows[:limit]:
        name = row["source_episode"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Invalid source episode identifier")
        scene = Path(pilot)/name/"scene.json"
        config = json.loads(scene.read_text(encoding="utf-8"))
        if (config.get("split") != split or config.get("episode_id") != name or
                config.get("workspace", {}).get("location_id") != row["location_id"]):
            raise ValueError(f"Scene identity/split mismatch: {name}")
        selected.append(dict(episode_index=row["episode_index"], episode=name,
            location_id=row["location_id"], split=split, box_pose=config["box_pose"],
            scene_seed=config.get("seed")))
    if not selected:
        raise ValueError("No scenes selected")
    return selected


def sweep_summary(selected, results, error=None):
    """Keep incomplete sweeps distinct from completed task failures."""
    passed = sum(bool(r["passed"]) for r in results)
    groups = {}
    for row in selected:
        groups.setdefault(row["location_id"], dict(planned=0, completed=0, passed=0))["planned"] += 1
    for row in results:
        group = groups[row["location_id"]]
        group["completed"] += 1
        group["passed"] += bool(row["passed"])
    strata = {}
    for row in selected:
        if 'cell_id' in row:
            strata.setdefault(row['cell_id'], dict(planned=0, completed=0, passed=0))['planned'] += 1
    for row in results:
        if 'cell_id' in row:
            strata[row['cell_id']]['completed'] += 1
            strata[row['cell_id']]['passed'] += bool(row['passed'])
    return dict(planned=len(selected), completed=len(results), passed=passed,
        failed=len(results)-passed, complete=len(results) == len(selected),
        success_fraction_completed=passed/len(results) if results else None,
        locations=groups, coverage_strata=strata, results=results, error=error)


class JointMapping:
    def __init__(self, path):
        records = json.loads(Path(path).read_text(encoding="utf-8"))["joints"]
        fits = [records[name]["derived_affine"] for name in JOINT_NAMES]
        self.slopes = np.array([fit["slope"] for fit in fits])
        self.offsets = np.array([fit["offset"] for fit in fits])
        if not np.isfinite([self.slopes, self.offsets]).all() or np.any(abs(self.slopes) < 1e-9):
            raise ValueError("Invalid affine joint mapping")
        self.low = np.array([-100.]*5+[0.])
        self.high = np.full(6, 100.)

    def state(self, radians):
        return ((np.asarray(radians)-self.offsets)/self.slopes).astype(np.float32)

    def command(self, action, joint_limits):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Policy must return six finite absolute normalized targets")
        normalized = np.clip(action, self.low, self.high)
        raw = normalized*self.slopes+self.offsets
        radians = np.clip(raw, joint_limits[:, 0], joint_limits[:, 1])
        return radians, (normalized != action) | (radians != raw)


class LiftTracker:
    """Only the evaluator reads object pose/contact; these never reach the policy."""
    def __init__(self, success, tabletop_z):
        self.clearance = success["minimum_bottom_clearance_mm"]/1000
        self.hold_s = success["hold_seconds"]
        self.force = success["minimum_finger_normal_force_n"]
        self.drift = success["maximum_box_drift_during_hold_mm"]/1000
        self.tabletop_z = tabletop_z
        self.samples = deque()
        self.maximum_clearance = 0.
        self.longest_contact_hold_s = 0.
        self.stable_hold = None

    def observe(self, timestamp, bottom, position, fingers):
        clearance = bottom-self.tabletop_z
        self.maximum_clearance = max(self.maximum_clearance, clearance)
        if self.stable_hold is not None:
            return
        if clearance < self.clearance or any(fingers[k] < self.force for k in ("fixed", "moving")):
            self.samples.clear()
            return
        self.samples.append((timestamp, np.asarray(position).copy(), clearance, dict(fingers)))
        while len(self.samples) > 1 and timestamp-self.samples[1][0] >= self.hold_s-1e-9:
            self.samples.popleft()
        span = timestamp-self.samples[0][0]
        self.longest_contact_hold_s = max(self.longest_contact_hold_s, span)
        if span >= self.hold_s-1e-9:
            drift = max(np.linalg.norm(row[1]-self.samples[0][1]) for row in self.samples)
            if drift <= self.drift:
                self.stable_hold = dict(start_s=self.samples[0][0], end_s=timestamp,
                    duration_s=span, drift_mm=1000*float(drift),
                    minimum_clearance_mm=1000*min(row[2] for row in self.samples),
                    minimum_finger_force_n={k:min(row[3][k] for row in self.samples) for k in ("fixed", "moving")})


class PolicyWorker:
    """Separate existing Python environments; no installs or package overrides."""
    def __init__(self, python, checkpoint, source, output, device="cuda", seed=0, task="Pick up the box"):
        self.task = task
        self.output = Path(output).resolve()
        self.observation = self.output/"latest_observation.npz"
        self.log_path = self.output/"inference.log"
        self.messages = queue.Queue()
        self.log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen([str(python), "-u", "-m", "integrations.xvla_worker",
            "--checkpoint", str(Path(checkpoint).resolve()), "--source", str(Path(source).resolve()),
            "--device", device, "--seed", str(seed), "--task", task], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.ready = self._response(300)
            if not self.ready.get("ready"):
                raise RuntimeError("Inference worker did not become ready")
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                self.messages.put({"error": "Unexpected worker stdout; inspect inference.log"})
        self.messages.put({"error": "Inference worker exited; inspect inference.log"})

    def _response(self, timeout):
        try:
            result = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(f"Inference timed out; inspect {self.log_path}") from error
        if "error" in result:
            raise RuntimeError(f"{result['error']} ({self.log_path})")
        return result

    def predict(self, state, frames):
        np.savez(self.observation, state=np.asarray(state, dtype=np.float32), **frames)
        self.process.stdin.write(json.dumps({"command": "predict", "observation": str(self.observation)})+"\n")
        self.process.stdin.flush()
        result = self._response(180)
        actions = np.asarray(result["actions"], dtype=float)
        if actions.shape != (self.ready["chunk_size"], 6) or not np.isfinite(actions).all():
            raise ValueError("Invalid policy action chunk")
        return actions, result["latency_s"]

    def reset(self, seed, output):
        """Independent episode state/noise while retaining the loaded weights."""
        self.process.stdin.write(json.dumps({"command": "reset", "seed": seed})+"\n")
        self.process.stdin.flush()
        result = self._response(30)
        if result != {"reset": True, "seed": seed}:
            raise RuntimeError("Inference worker did not reset the episode")
        self.observation = Path(output).resolve()/"latest_observation.npz"

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"quit"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        self.reader.join(timeout=2)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
