"""Local, persistent X-VLA inference worker. No robot or network transport.

Run with the existing LeRobot Python environment. JSON commands arrive on
stdin; observations are trusted local NPZ files from the simulation process.
Model/library messages go to stderr so stdout remains a JSON protocol.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


CAMERAS = ("front_left", "front_right", "wrist_left", "wrist_right")


def load_policy(checkpoint, source, device):
    sys.path.insert(0, str(Path(source).resolve()/"src"))
    # Use the existing process-local optional-policy bootstrap, never edit the fork.
    from training.box_xvla.train import bootstrap_xvla
    bootstrap_xvla()
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla import processor_xvla  # register saved custom steps
    checkpoint = Path(checkpoint).resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    config.tokenizer_name = str(checkpoint/"tokenizer")
    if config.input_features["observation.state"].shape != (6,) and list(config.input_features["observation.state"].shape) != [6]:
        raise ValueError("Require the six-channel Sourccey checkpoint")
    expected = {"observation.images."+name for name in CAMERAS}
    if set(config.image_features) != expected or list(config.output_features["action"].shape) != [6]:
        raise ValueError("Require four named robot cameras and six output actions")
    policy = XVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    pre, post = make_pre_post_processors(config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device},
                                "tokenizer_processor": {"tokenizer_name": str(checkpoint/"tokenizer")}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    policy.eval()
    policy.reset()
    return torch, policy, pre, post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="Pick up the box")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = sys.stdout

    def send(message):
        output.write(json.dumps(message, allow_nan=False)+"\n")
        output.flush()

    try:
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            torch, policy, pre, post = load_policy(args.checkpoint, args.source, args.device)
        torch.manual_seed(args.seed)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        send({"ready": True, "load_wall_s": time.perf_counter()-started,
              "device": args.device, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "chunk_size": policy.config.chunk_size, "n_action_steps": policy.config.n_action_steps})
        for line in sys.stdin:
            command = json.loads(line)
            if command["command"] == "quit":
                break
            if command["command"] == "reset":
                with contextlib.redirect_stdout(sys.stderr):
                    policy.reset()
                    pre.reset()
                    post.reset()
                    torch.manual_seed(command["seed"])
                send({"reset": True, "seed": command["seed"]})
                continue
            if command["command"] != "predict":
                raise ValueError("Unknown inference command")
            import numpy as np
            with np.load(command["observation"], allow_pickle=False) as observation:
                state = observation["state"].copy()
                if state.shape != (6,) or not np.isfinite(state).all():
                    raise ValueError("Invalid observed state")
                batch = {"observation.state": torch.from_numpy(state), "task": args.task}
                for name in CAMERAS:
                    image = observation[name].copy()
                    if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                        raise ValueError(f"Invalid RGB observation: {name}")
                    batch["observation.images."+name] = torch.from_numpy(image).permute(2, 0, 1)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                # Saved processors perform all image/state transforms exactly once.
                prepared = pre(batch)
                normalized = policy.predict_action_chunk(prepared)
                actions = post(normalized).detach().cpu().numpy()
            if actions.shape != (1, policy.config.chunk_size, 6) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid predicted chunk: {actions.shape}")
            send({"actions": actions[0].tolist(), "latency_s": time.perf_counter()-started})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send({"error": str(error), "error_type": type(error).__name__})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
