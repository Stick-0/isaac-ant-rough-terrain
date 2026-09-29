#!/usr/bin/env python3
"""Place the complete synchronized rough-terrain recordings side by side."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/media")
    args = parser.parse_args()
    folder = args.directory.resolve()
    flat = json.loads((folder / "flat_on_rough.json").read_text())
    rough = json.loads((folder / "rough_on_rough.json").read_text())
    for key in ("seed", "num_envs", "initial_state_sha256", "frames", "fps", "simulation_seconds",
                "camera_env_index", "camera_eye_offset", "camera_target_offset"):
        if flat[key] != rough[key]:
            raise ValueError(f"Recordings do not match: {key}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def run(options):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *options], check=True)

    run(["-i", str(folder / "flat_on_rough.mp4"), "-i", str(folder / "rough_on_rough.mp4"),
         "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
         "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-threads", "4",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "comparison.mp4")])
    run(["-ss", "3", "-i", str(folder / "comparison.mp4"), "-frames:v", "1",
         "-q:v", "2", str(folder / "comparison_poster.jpg")])
    run(["-i", str(folder / "comparison.mp4"), "-t", "6", "-filter_complex",
         "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
         "[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0",
         str(folder / "comparison_preview.gif")])
    assets = []
    for path in sorted(folder.iterdir()):
        if path.suffix in {".mp4", ".png", ".jpg", ".gif"}:
            assets.append({"file": path.name, "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder / "index.json").write_text(json.dumps({
        "source": "Actual Isaac Sim RGB renders; no generated illustrations",
        "comparison": "Left: flat-trained; right: rough-trained; same terrain and reset states",
        "video_duration_s": flat["simulation_seconds"], "fps": flat["fps"],
        "preview_duration_s": 6, "preview_fps": 8, "assets": assets,
    }, indent=2) + "\n")
    print(f"Created synchronized comparison, 6-second GIF preview and {len(assets)} media checksums")


if __name__ == "__main__":
    main()
