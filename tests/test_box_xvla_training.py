import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from training.box_xvla.train import (
    BASE_MODEL, BASE_REVISION, CAMERAS, JOINTS, build_config, source_hash, validate_dataset,
    install_time_budget, TrainingBudgetReached,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    features = {key: dict(shape=[6], names=list(JOINTS)) for key in ("observation.state", "action")}
    features.update({"observation.images." + c: dict(shape=[240, 320, 3]) for c in CAMERAS})
    info = dict(fps=50, total_episodes=6, features=features)
    splits = dict(train=[0, 1, 2], validation=[3, 4], test=[5])
    provenance = dict(task="Pick up the box", episodes=[dict(
        episode_index=i, location_id=split, split=split) for split, ids in splits.items() for i in ids])
    template = dict(policy=dict(type="xvla", max_state_dim=20, max_action_dim=20, chunk_size=30,
        florence_config={"keep": "checkpoint architecture"}), dataset={})
    for name, value in (("meta/info.json", info), ("episode_splits.json", splits),
                        ("simulation_provenance.json", provenance), ("xvla_train.json", template)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def test_two_gpu_training_preserves_checkpoint_and_robot_contract(dataset, tmp_path):
    config = build_config(dataset, tmp_path / "out", gpus=2)
    assert config["dataset"]["episodes"] == [0, 1, 2, 3, 4]
    assert config["dataset"]["eval_split"] == 0.3
    assert config["parallelism"] == dict(dp_replicate=2, dp_shard=1)
    policy = config["policy"]
    assert policy["pretrained_path"] == BASE_MODEL and policy["pretrained_revision"] == BASE_REVISION
    assert policy["dtype"] == "float32" and config["accelerator"]["mixed_precision"] == "bf16"
    assert policy["max_state_dim"] == 20 and policy["max_action_dim"] == 20
    assert policy["output_features"]["action"]["shape"] == [6]
    assert list(policy["input_features"])[1:] == ["observation.images." + c for c in CAMERAS]
    assert config["env_eval_freq"] == 0 and not config["wandb"]["enable"]
    assert not policy["push_to_hub"] and config["checkpoint_format"] == "safetensors"
    assert Path(config["dataset"]["root"]) == dataset.resolve()
    assert read_json(dataset / "xvla_train.json")["policy"]["florence_config"] == policy["florence_config"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_rejects_joint_order_and_split_leaks(dataset):
    info = read_json(dataset / "meta/info.json")
    info["features"]["action"]["names"].reverse()
    (dataset / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="order"):
        validate_dataset(dataset)
    info["features"]["action"]["names"].reverse()
    (dataset / "meta/info.json").write_text(json.dumps(info))
    provenance = read_json(dataset / "simulation_provenance.json")
    provenance["episodes"][-1]["location_id"] = "train"
    (dataset / "simulation_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="cross splits"):
        validate_dataset(dataset)


def test_accumulation_uses_native_microbatch_schedule(dataset, tmp_path):
    config = build_config(dataset, tmp_path / "out", gpus=2, steps=40000, accumulation=2)
    assert config["save_freq"] == 2000
    assert config["policy"]["scheduler_warmup_steps"] == 2000
    assert config["policy"]["scheduler_decay_steps"] == 40000
    config = build_config(dataset, tmp_path / "shard", gpus=2, shard=True)
    assert config["parallelism"] == dict(dp_replicate=1, dp_shard=2)
    assert "TransformerBlock" in config["accelerator"]["fsdp"]["wrap_modules"]
    with pytest.raises(ValueError, match="two GPUs"):
        build_config(dataset, tmp_path / "out", shard=True)


def test_hash_is_portable_across_line_endings(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"a\r\nb\r\n")
    windows = source_hash(path)
    path.write_bytes(b"a\nb\n")
    assert source_hash(path) == windows


def test_budget_saves_before_stopping_and_only_at_update_boundary(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        distributed=SimpleNamespace(is_initialized=lambda: False)))
    trainer = SimpleNamespace(
        should_save_checkpoint=lambda step, freq, total: step % freq == 0 or step == total,
        save_checkpoint=lambda **kwargs: calls.append("checkpoint-complete"),
        update_last_checkpoint=lambda path: calls.append("last-updated"))
    install_time_budget(trainer, deadline=0, accumulation=2)
    assert not trainer.should_save_checkpoint(1, 1000, 20000)
    assert trainer.should_save_checkpoint(2, 1000, 20000)
    accelerator = SimpleNamespace(is_main_process=True,
        wait_for_everyone=lambda: calls.append("barrier"),
        end_training=lambda: calls.append("end-training"))
    with pytest.raises(TrainingBudgetReached):
        trainer.save_checkpoint(accelerator=accelerator, checkpoint_dir=Path("checkpoint"))
    assert calls == ["checkpoint-complete", "last-updated", "barrier", "end-training"]


def test_unexpired_budget_preserves_regular_saving(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        distributed=SimpleNamespace(is_initialized=lambda: False)))
    calls = []
    trainer = SimpleNamespace(
        should_save_checkpoint=lambda step, freq, total: step == total or step % freq == 0,
        save_checkpoint=lambda **kwargs: calls.append("saved"))
    install_time_budget(trainer, deadline=float("inf"), accumulation=1)
    assert not trainer.should_save_checkpoint(1, 1000, 20000)
    assert trainer.should_save_checkpoint(1000, 1000, 20000)
    trainer.save_checkpoint()
    assert calls == ["saved"]
