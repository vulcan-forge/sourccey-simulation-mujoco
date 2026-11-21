"""Package the existing dataset, calibration contract and portable training scripts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from training.box_xvla.train import AUDIT_FILES, BASE_MODEL, BASE_REVISION, CAMERAS, JOINTS, source_hash, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def prepare(dataset, lerobot_repo, output):
    dataset, lerobot_repo, output = map(lambda p: Path(p).resolve(), (dataset, lerobot_repo, output))
    info, splits, provenance, _ = validate_dataset(dataset)
    if output.exists():
        raise FileExistsError(f"Choose a fresh bundle directory: {output}")
    mapping = ROOT / "artifacts/robot_joint_mapping.json"
    cameras = ROOT / "models/source/camera_poses.json"
    robot_plugin = lerobot_repo.parent / "packages/lerobot-robot-sourccey"
    if not (robot_plugin / "pyproject.toml").is_file():
        robot_plugin = lerobot_repo.parent / "lerobot-robot-sourccey"
    if not (robot_plugin / "pyproject.toml").is_file():
        raise FileNotFoundError("The fork lockfile requires the local lerobot-robot-sourccey package")
    tokenizer = ROOT / "artifacts/xvla_setup/bart_tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Require the BART tokenizer already used for dataset compatibility checks")
    audit = {name: source_hash(lerobot_repo / name) for name in AUDIT_FILES}
    mapping_hash = source_hash(mapping)
    # Source metadata uses byte hashes; newline normalization is only for source code.
    import hashlib
    raw_mapping_hash = hashlib.sha256(mapping.read_bytes()).hexdigest()
    for episode in provenance["episodes"]:
        source_episode = Path(provenance["source"]) / episode["source_episode"] / "episode.json"
        metadata = json.loads(source_episode.read_text(encoding="utf-8"))
        if metadata["mapping_sha256"] != raw_mapping_hash:
            raise ValueError("Current mapping differs from the dataset's source mapping")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lerobot_repo, text=True).strip()
    manifest = dict(format="sourccey-box-xvla-training-v1", task=provenance["task"],
        task_contract=provenance.get("task_contract"), split_strategy=provenance.get("split_strategy", "held-out location groups"),
        pretrained_model=BASE_MODEL, pretrained_revision=BASE_REVISION,
        lerobot_revision=revision, lerobot_source_sha256=audit,
        mapping_sha256=raw_mapping_hash, mapping_normalized_text_sha256=mapping_hash,
        task_clock_hz=info["fps"], camera_names=list(CAMERAS), state_action_order=list(JOINTS),
        action_semantics="absolute normalized hardware position targets; five arm joints [-100,100], gripper [0,100]",
        split_counts={k: len(v) for k, v in splits.items()},
        physical_calibration_verified=False, model_training_completed=False,
        real_robot_adapter_required=True, real_camera_default_fps=30,
        dataset_files={p.relative_to(dataset).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(dataset.rglob("*")) if p.is_file()})
    output.mkdir(parents=True)
    shutil.copytree(dataset, output / "dataset")
    shutil.copytree(tokenizer, output / "tokenizer")
    # Include exact installable source snapshots, without virtualenvs/caches.
    for source, target in ((lerobot_repo, output / "lerobot-vulcan"),
                           (robot_plugin, output / "packages/lerobot-robot-sourccey")):
        shutil.copytree(source / "src", target / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "LICENSE.txt", ".python-version", "MANIFEST.in"):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
    for path, name in ((mapping, "robot_joint_mapping.json"), (cameras, "camera_poses.json"),
                       (ROOT / "training/box_xvla/train.py", "train.py"),
                       (ROOT / "training/box_xvla/run.sh", "run.sh"),
                       (ROOT / "docs/box_xvla_training.md", "README.md")):
        shutil.copy2(path, output / name)
    (output / "handoff.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {output}: dataset, training script, calibration references and source/file hashes")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/datasets/box_xvla_pilot_v3")
    parser.add_argument("--lerobot-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/box_xvla_training_handoff_5090")
    args = parser.parse_args()
    prepare(args.dataset, args.lerobot_repo, args.output)


if __name__ == "__main__":
    main()
