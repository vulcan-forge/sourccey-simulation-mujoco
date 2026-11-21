"""Portable X-VLA launcher for the Sourccey box pilot; never connects to hardware.

Requires the inspected lerobot-vulcan checkout installed with training + xvla.
The parent launches torchrun; each worker uses the fork's normal training loop.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
BASE_MODEL = "lerobot/xvla-base"
BASE_REVISION = "cdb7964e4fe842935d671bfab5a5ebe00a96648c"
TASK = "Pick up the box"
PLACE_TASK = "Pick up the box and place it at the fixed destination"
CAMERAS = ("front_left", "front_right", "wrist_left", "wrist_right")
JOINTS = tuple("left_" + name + ".pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
AUDIT_FILES = (
    "pyproject.toml", "uv.lock", "src/lerobot/scripts/lerobot_train.py",
    "src/lerobot/configs/train.py", "src/lerobot/configs/accelerator.py",
    "src/lerobot/datasets/factory.py", "src/lerobot/policies/factory.py",
    "src/lerobot/processor/tokenizer_processor.py", "src/lerobot/optim/optimizers.py",
    *["src/lerobot/policies/xvla/" + name + ".py" for name in (
        "configuration_xvla", "modeling_xvla", "processor_xvla", "action_hub", "soft_transformer")],
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hash(path):
    # Identical across Git's Windows/Linux newline conversion.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_dataset(root):
    root = Path(root)
    info = read_json(root / "meta/info.json")
    splits = read_json(root / "episode_splits.json")
    provenance = read_json(root / "simulation_provenance.json")
    if info["fps"] != 50 or provenance["task"] not in (TASK, PLACE_TASK):
        raise ValueError("Require a supported 50 Hz box task dataset")
    if provenance["task"] == PLACE_TASK and not provenance.get("task_contract", {}).get("placement"):
        raise ValueError("Pick-and-place data must record the fixed destination and placement checks")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Require explicit train/validation/test episode lists")
    flat = sum((splits[s] for s in ("train", "validation", "test")), [])
    if len(set(flat)) != len(flat) or sorted(flat) != list(range(info["total_episodes"])):
        raise ValueError("Splits must partition the dataset without duplicates")
    if not all(splits.values()):
        raise ValueError("All three splits must be nonempty")
    groups = {}
    for episode in provenance["episodes"]:
        index, group, split = episode["episode_index"], episode["location_id"], episode["split"]
        if index not in splits[split] or groups.setdefault(group, split) != split:
            raise ValueError("Location groups cross splits or provenance disagrees")
    features = info["features"]
    for key in ("action", "observation.state"):
        if features[key]["shape"] != [6] or features[key]["names"] != list(JOINTS):
            raise ValueError(f"Wrong six-channel order: {key}")
    images = {"observation.images." + c for c in CAMERAS}
    if {k for k in features if k.startswith("observation.images.")} != images:
        raise ValueError("Require exactly the four named robot cameras")
    if any(features[k]["shape"] != [240, 320, 3] for k in images):
        raise ValueError("Camera images must be 320x240 RGB")
    # Native factory takes the last ceil(N * fraction) episodes as validation.
    # Half-episode slack avoids rounding a mathematically integral ratio upward.
    episodes = splits["train"] + splits["validation"]
    fraction = (len(splits["validation"]) - 0.5) / len(episodes)
    assert math.ceil(len(episodes) * fraction) == len(splits["validation"])
    return info, splits, provenance, fraction


def build_config(root, output, *, batch_size=2, steps=20000, workers=4,
                 accumulation=1, precision="bf16", gpus=1, shard=False):
    info, splits, _, fraction = validate_dataset(root)
    if min(batch_size, steps, accumulation, gpus) < 1 or workers < 0:
        raise ValueError("Batch, steps, accumulation and GPU count must be positive")
    config = read_json(Path(root) / "xvla_train.json")
    policy = config["policy"]
    if (policy["max_state_dim"], policy["max_action_dim"], policy["chunk_size"]) != (20, 20, 30):
        raise ValueError("Base checkpoint requires 20 state/action model channels and chunk_size=30")
    policy.update(pretrained_path=BASE_MODEL, pretrained_revision=BASE_REVISION,
                  device="cuda", dtype="float32", use_amp=False, push_to_hub=False,
                  action_mode="auto", n_action_steps=10, num_image_views=4,
                  tokenizer_name="facebook/bart-large", tokenizer_max_length=64,
                  freeze_vision_encoder=False, freeze_language_encoder=False,
                  train_policy_transformer=True, train_soft_prompts=True,
                  optimizer_lr=1e-4, scheduler_warmup_steps=min(1000 * accumulation, max(1, steps // 10)),
                  scheduler_decay_steps=max(2, steps), scheduler_decay_lr=2.5e-6)
    if (HERE / "tokenizer").is_dir():
        policy["tokenizer_name"] = str(HERE / "tokenizer")
    policy["normalization_mapping"] = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    policy["input_features"] = {"observation.state": {"type": "STATE", "shape": [6]},
        **{"observation.images." + c: {"type": "VISUAL", "shape": [3, 240, 320]} for c in CAMERAS}}
    policy["output_features"] = {"action": {"type": "ACTION", "shape": [6]}}
    config["dataset"].update(root=str(Path(root).resolve()),
        episodes=splits["train"] + splits["validation"], eval_split=fraction,
        video_backend="pyav", use_imagenet_stats=False,
        image_transforms={"enable": False})
    config.update(output_dir=str(Path(output).resolve()), job_name="sourccey_box_xvla",
        batch_size=batch_size, steps=steps, num_workers=workers, persistent_workers=workers > 0,
        prefetch_factor=2, dataloader_multiprocessing_context="spawn",
        env=None, env_eval_freq=0, eval_steps=1000 * accumulation, max_eval_samples=256,
        log_freq=50 * accumulation, save_checkpoint=True, save_freq=1000 * accumulation,
        checkpoint_format="safetensors", save_checkpoint_to_hub=False,
        wandb={"enable": False}, seed=20261116, use_policy_training_preset=True,
        accelerator={"mixed_precision": precision, "gradient_accumulation": {"steps": accumulation}},
        parallelism={"dp_replicate": gpus if not shard else 1, "dp_shard": gpus if shard else 1})
    if shard:
        if gpus < 2:
            raise ValueError("Sharding needs at least two GPUs")
        config["accelerator"]["fsdp"] = {"wrap_modules": [
            "TransformerBlock", "BartEncoderLayer", "Florence2VisionBlock"]}
    return config


def bootstrap_xvla():
    """Register only X-VLA: this fork eagerly imports unrelated optional policies.

    No installed files change. Populate the normal package's public exports used
    by the trainer/evaluator, with their actual LeRobot implementations.
    """
    if "lerobot.policies" not in sys.modules:
        spec = importlib.util.find_spec("lerobot")
        if spec is None:
            raise ImportError("Install the inspected lerobot-vulcan checkout first")
        package = types.ModuleType("lerobot.policies")
        package.__path__ = [str(Path(p) / "policies") for p in spec.submodule_search_locations]
        package.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = package
        from lerobot.policies import factory, pretrained, utils
        for module, names in ((factory, ("get_policy_class", "make_policy", "make_policy_config", "make_pre_post_processors")),
                              (pretrained, ("PreTrainedPolicy",)),
                              (utils, ("make_robot_action", "prepare_observation_for_inference"))):
            for name in names:
                setattr(package, name, getattr(module, name))
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401


def install_training_adapters(trainer, root):
    """Keep the stock loop, explicit splits and fresh six-channel processors."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    _, splits, _, _ = validate_dataset(root)
    native_datasets = trainer.make_train_eval_datasets
    native_processors = trainer.make_pre_post_processors
    native_policy = trainer.make_policy

    def policy(cfg, **kwargs):
        if kwargs.get("pretrained_path") is None and str(cfg.pretrained_path) == BASE_MODEL:
            # The fork's remote loader passes removed Hub v0.x arguments. Use the
            # current public download API, then its unchanged strict local loader.
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(BASE_MODEL, "model.safetensors", revision=BASE_REVISION)
            kwargs["pretrained_path"] = Path(weights).parent
        return native_policy(cfg, **kwargs)

    def datasets(cfg):
        train, validation = native_datasets(cfg)
        if list(train.episodes) != splits["train"] or list(validation.episodes) != splits["validation"]:
            raise ValueError("Native trainer changed the reserved episode splits")
        # Verify exported statistics against actual training rows, without images.
        parts = {key: [] for key in ("observation.state", "action")}
        for path in sorted((Path(root) / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", *parts])
            mask = np.isin(table["episode_index"].to_numpy(), splits["train"])
            for key in parts:
                parts[key].append(np.asarray(table[key].to_pylist(), dtype=np.float64)[mask])
        for key, arrays in parts.items():
            array = np.concatenate(arrays)
            for name, expected in {"mean": array.mean(0), "std": array.std(0),
                                   "min": array.min(0), "max": array.max(0)}.items():
                if not np.allclose(np.asarray(train.meta.stats[key][name]), expected, atol=1e-3, rtol=1e-4):
                    raise ValueError(f"Training-only statistics disagree: {key}/{name}")
        print(f"Verified {len(train.episodes)} train / {len(validation.episodes)} validation episodes; test excluded", flush=True)
        return train, validation

    def processors(policy_cfg, pretrained_path=None, **kwargs):
        if str(pretrained_path) == BASE_MODEL:
            print("Building NEW four-camera/six-channel processors with training-only statistics", flush=True)
            return make_xvla_pre_post_processors(policy_cfg, kwargs["dataset_stats"])
        # Resume restores the processors/statistics saved with our checkpoint.
        return native_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    trainer.make_train_eval_datasets = datasets
    trainer.make_pre_post_processors = processors
    trainer.make_policy = policy


class TrainingBudgetReached(Exception):
    """All ranks saved a complete checkpoint at an accumulation boundary."""


def install_time_budget(trainer, *, deadline, accumulation):
    """Collectively request a checkpoint, then stop after its writes finish.

    The original configured step/scheduler horizon stays intact for resume.
    This hook does not interrupt an in-flight update or kill GPU processes.
    """
    import torch
    native_should_save, native_save = trainer.should_save_checkpoint, trainer.save_checkpoint
    stopping = False

    def should_save(step, save_freq, total_steps):
        nonlocal stopping
        if step % accumulation == 0:
            expired = time.monotonic() >= deadline
            if torch.distributed.is_initialized():
                flag = torch.tensor(int(expired), device=torch.cuda.current_device())
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                expired = bool(flag.item())
            stopping = expired
        return stopping or native_should_save(step, save_freq, total_steps)

    def save(**kwargs):
        result = native_save(**kwargs)
        if stopping:
            accelerator = kwargs["accelerator"]
            # The trainer updates 'last' AFTER save_checkpoint returns. Do it
            # here before leaving the loop; every rank waits for rank zero.
            if accelerator.is_main_process:
                trainer.update_last_checkpoint(kwargs["checkpoint_dir"])
                print(f"Time budget reached; saved resumable checkpoint {kwargs['checkpoint_dir']}", flush=True)
            accelerator.wait_for_everyone()
            accelerator.end_training()
            raise TrainingBudgetReached
        return result

    trainer.should_save_checkpoint, trainer.save_checkpoint = should_save, save


def verify_source(manifest):
    import lerobot
    repository = Path(lerobot.__file__).resolve().parents[2]
    for relative, expected in manifest["lerobot_source_sha256"].items():
        path = repository / relative
        if not path.is_file() or source_hash(path) != expected:
            raise ValueError(f"Training fork differs from inspected source: {path}. Use the transferred checkout.")
    return repository


def check_runtime(config, *, require_gpu):
    import importlib.metadata as metadata
    import torch
    versions = {name: metadata.version(name) for name in (
        "torch", "transformers", "accelerate", "huggingface-hub", "draccus", "av")}
    print(json.dumps({"versions": versions, "cuda": torch.cuda.is_available()}, indent=2))
    if require_gpu:
        if not torch.cuda.is_available():
            raise ValueError("Training requires CUDA; --check-only works on CPU")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(rank)
        if config["accelerator"]["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU lacks BF16 support: choose --precision fp16 or no")
        # Exercise installed CUDA kernels, catching unsupported Blackwell builds
        # before downloading/allocating the full pretrained policy.
        probe = torch.ones((8, 8), device="cuda")
        (probe @ probe).sum().item()
        print(f"GPU {rank}: {torch.cuda.get_device_name(rank)}; CUDA kernels passed", flush=True)
    return versions


def run_worker(args):
    started = time.monotonic()
    bootstrap_xvla()
    import draccus
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as trainer

    manifest = read_json(HERE / "handoff.json")
    verify_source(manifest)
    root, output = args.dataset.resolve(), args.output.resolve()
    if args.check_only:
        for name, expected in manifest["dataset_files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Transferred dataset hash mismatch: {name}")
    if args.resume:
        config = read_json(args.resume)
        config["dataset"]["root"] = str(root)
        config["output_dir"] = str(output)
    else:
        config = build_config(root, output, batch_size=args.batch_size, steps=args.steps,
            workers=args.workers, accumulation=args.accumulation, precision=args.precision,
            gpus=args.gpus, shard=args.shard)
    versions = check_runtime(config, require_gpu=not args.check_only)
    if not args.check_only and int(os.environ.get("WORLD_SIZE", "1")) != args.gpus:
        raise ValueError("Launch through this script's parent, or set --gpus to the torchrun world size")
    cfg = draccus.decode(TrainPipelineConfig, config)
    install_training_adapters(trainer, root)
    if args.check_only:
        torch.set_num_threads(2)
        # Validate the real fork's schema and actual split creation, without weights.
        cfg.resume = False
        sys.argv = [sys.argv[0]]
        cfg.validate()
        cfg.policy.get_florence_config()
        train, validation = trainer.make_train_eval_datasets(cfg)
        check_cfg = copy.deepcopy(cfg.policy)
        check_cfg.device = "cpu"
        from torch.utils.data._utils.collate import default_collate
        pre, post = trainer.make_pre_post_processors(check_cfg, pretrained_path=BASE_MODEL,
                                                    dataset_stats=train.meta.stats)
        for dataset in (train, validation):
            batch = pre(default_collate([dataset[0]]))
            if batch["observation.state"].shape != (1, 6) or batch["action"].shape != (1, 30, 6):
                raise ValueError("Processor state/action shape mismatch")
            if not all(torch.isfinite(batch[key]).all() for key in ("action", "observation.state")):
                raise ValueError("Nonfinite processed data")
        sample = torch.zeros(1, 6)
        if post(sample).shape != sample.shape:
            raise ValueError("Postprocessor action shape mismatch")
        print("PASS: actual trainer imports/config/splits/train-only stats/four-camera processors. No weights or training.")
        return
    if args.hours > 0:
        install_time_budget(trainer, deadline=started + args.hours * 3600 - args.reserve_minutes * 60,
                            accumulation=cfg.accelerator.gradient_accumulation.steps)
    if int(os.environ.get("RANK", "0")) == 0:
        audit = output.parent / (output.name + "_handoff")
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        launch = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        (audit / "runtime.json").write_text(json.dumps({"versions": versions, "launch": launch}, indent=2), encoding="utf-8")
        for name in ("handoff.json", "robot_joint_mapping.json", "camera_poses.json"):
            (audit / name).write_bytes((HERE / name).read_bytes())
    # Let LeRobot's parser resolve resume checkpoint state/optimizer/RNG normally.
    sys.argv = [sys.argv[0]]
    try:
        if args.resume:
            sys.argv += [f"--config_path={args.resume.resolve()}", "--resume=true",
                         f"--dataset.root={root}", f"--output_dir={output}"]
            trainer.train()
        else:
            trainer.train(cfg)
    except TrainingBudgetReached:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="Per GPU micro-batch")
    parser.add_argument("--steps", type=int, default=20000, help="Native trainer micro-batches, not optimizer updates")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per process")
    parser.add_argument("--precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--shard", action="store_true", help="Opt-in FSDP2; multi-GPU smoke test required")
    parser.add_argument("--hours", type=float, default=8, help="Wall time budget per launch; 0 disables")
    parser.add_argument("--reserve-minutes", type=float, default=10, help="Start final checkpoint this long before budget expires")
    parser.add_argument("--resume", type=Path, help="Checkpoint pretrained_model/train_config.json; keeps saved hyperparameters")
    parser.add_argument("--check-only", action="store_true", help="Real data/processors/config, no model weights or gradients")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("HF_HOME", str(HERE / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / ".cache" / "datasets"))
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if args.hours < 0 or args.reserve_minutes < 0 or (args.hours and args.reserve_minutes >= args.hours * 60):
        parser.error("Budget must be positive and reserve shorter than budget")
    if args.worker or args.check_only:
        run_worker(args)
    else:
        if os.name == "nt" and args.gpus > 1:
            parser.error("Use Linux/WSL2 CUDA for the multi-GPU NCCL training path")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={args.gpus}", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        print("Starting LeRobot; effective batch = per-GPU batch x GPUs x accumulation", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
