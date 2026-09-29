#!/usr/bin/env python3
"""Render actual Isaac Sim terrain images and reproducible Ant policy videos.

Run using the patched IsaacLab_RS isaaclab.sh launcher and --headless.
Camera/labels are presentation only; policy, dynamics and terrain are unchanged.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+", choices=["flat", "rough", "stable", "control", "recovery"], default=["flat", "rough"])
parser.add_argument("--terrain", choices=["rough", "flat"], default="rough")
parser.add_argument("--seed", type=int, default=2001)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=16.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/media")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.seconds <= 0 or args.fps < 1 or 60 % args.fps:
    parser.error("num_envs and seconds must be positive; fps must divide 60")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import hashlib
import imageio.v2 as imageio
import importlib.util
import json
import numpy as np
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

ROOT = Path(__file__).resolve().parents[1]
FOLLOW_EYE = (3.8, -5.5, 2.8)
FOLLOW_TARGET = (0.4, 0.0, 0.0)


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def caption(rgb, title, details):
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 64), (15, 24, 35))
    canvas.paste(Image.fromarray(rgb), (0, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 7), title, font=font(21), fill=(242, 246, 250))
    draw.text((18, 35), details, font=font(16), fill=(177, 199, 216))
    return np.asarray(canvas)


def warm_render(base, frames=12):
    base.render()  # Initialize the RGB annotator/render product.
    for _ in range(frames):
        base.sim.render()
    rgb = base.render()
    if rgb.max() == 0:
        raise RuntimeError("Renderer returned an empty/black image")
    return rgb


def load_environment():
    if args.terrain == "rough":
        cfg = parse_env_cfg("Isaac-Ant-v0", device=args.device, num_envs=args.num_envs)
        cfg.scene.terrain.terrain_generator.seed = args.seed
    else:
        spec = importlib.util.spec_from_file_location("ant_media_original", ROOT / "reference/original/ant_env_cfg.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cfg = module.AntEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.viewer.resolution = (960, 540)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = FOLLOW_EYE
    cfg.viewer.lookat = FOLLOW_TARGET
    return cfg


@torch.inference_mode()
def main():
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_environment()
    agent_cfg = load_cfg_from_registry("Isaac-Ant-v0", "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make("Isaac-Ant-v0", cfg=cfg, render_mode="rgb_array"),
                             clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
        robot = base.scene["robot"]
        camera = base.viewport_camera_controller
        # Two actual rendered views of the generated surface; no synthetic terrain illustration.
        env.seed(args.seed)
        env.reset()
        camera.update_view_to_world()
        camera.update_view_location(eye=(18, -25, 15), lookat=(0, 0, 0))
        overview = warm_render(base)
        imageio.imwrite(args.output / f"{args.terrain}_terrain_overview.png", overview)
        camera.update_view_location(eye=(7, -11, 3.5), lookat=(0, 0, 0))
        detail = warm_render(base)
        imageio.imwrite(args.output / f"{args.terrain}_terrain_detail.png", detail)
        expected_initial_state = None
        for model_name in args.models:
            checkpoint = ROOT / "artifacts/models" / f"{model_name}.pt"
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=base.device)
            env.seed(args.seed)
            obs, _ = env.reset()
            start = robot.data.root_pos_w[:, :2].clone()
            initial = torch.cat((robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel), dim=-1)
            state_hash = hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()
            if expected_initial_state is None:
                expected_initial_state = state_hash
            elif state_hash != expected_initial_state:
                raise RuntimeError("Policy recordings do not have identical reset states")
            camera.update_view_to_asset_root("robot")
            camera.update_view_location(eye=FOLLOW_EYE, lookat=FOLLOW_TARGET)
            warm_render(base)
            name = f"{model_name}_on_{args.terrain}"
            frames = 0
            fall_count = 0
            timeout_count = 0
            telemetry = []
            steps = round(args.seconds / base.step_dt)
            interval = round(1.0 / (args.fps * base.step_dt))
            writer = imageio.get_writer(args.output / f"{name}.mp4", fps=args.fps,
                                       codec="libx264", quality=8, macro_block_size=2,
                                       ffmpeg_params=["-movflags", "+faststart"])
            try:
                for step in range(steps):
                    if not simulation_app.is_running():
                        raise RuntimeError("Simulator closed during recording")
                    if step % interval == 0:
                        elapsed = step * base.step_dt
                        forward = (robot.data.root_pos_w[0, 0] - start[0, 0]).item()
                        title = f"{model_name.upper()}-TRAINED POLICY  |  {args.terrain.upper()} TERRAIN"
                        details = (f"seed {args.seed}  |  ant 0  |  t = {elapsed:05.2f} s  |  "
                                   f"x = {forward:05.1f} m  |  falls = {fall_count}")
                        frame = caption(base.render(), title, details)
                        writer.append_data(frame)
                        frames += 1
                        if step in (0, 180, 480):
                            imageio.imwrite(args.output / f"{name}_{round(elapsed):02d}s.jpg", frame, quality=92)
                        telemetry.append({"time_s": elapsed, "forward_displacement_m": forward,
                                          "falls": fall_count, "timeouts": timeout_count})
                    obs, _, _, _ = env.step(policy(obs))
                    fall_count += int(base.reset_terminated[0].item())
                    timeout_count += int(base.reset_time_outs[0].item())
                    if (step + 1) % 120 == 0:
                        print(f"[MEDIA] {name} {(step + 1) * base.step_dt:.1f}s, ant0 falls={fall_count}", flush=True)
            finally:
                writer.close()
            metadata = {
                "checkpoint": f"artifacts/models/{model_name}.pt",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "terrain": args.terrain, "seed": args.seed, "num_envs": args.num_envs,
                "camera_env_index": 0, "camera_eye_offset": FOLLOW_EYE,
                "camera_target_offset": FOLLOW_TARGET, "initial_state_sha256": state_hash,
                "frames": frames, "fps": args.fps, "simulation_seconds": steps * base.step_dt,
                "playback_speed": 1.0, "auto_resets_included": True,
                "ant0_falls": fall_count, "ant0_timeouts": timeout_count,
                "note": f"Qualitative {args.num_envs}-env recording, separate from the 512-env benchmark.",
                "telemetry": telemetry,
            }
            (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"[MEDIA] Saved {name}: {frames} frames", flush=True)
    finally:
        if base.viewport_camera_controller is not None:
            base.viewport_camera_controller.update_view_to_world()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
