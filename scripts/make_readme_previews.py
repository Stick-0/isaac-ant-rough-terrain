#!/usr/bin/env python3
"""Make looping README previews from the first six seconds of actual policy videos."""

import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'artifacts/media'
CLIPS = {
    'flat': 'flat_on_flat',
    'rough': 'rough_on_rough',
    'recovery': 'reward_revision/recovery_on_rough',
}


def main():
    previews = []
    for model, stem in CLIPS.items():
        source = MEDIA / (stem + '.mp4')
        target = MEDIA / (stem + '_preview.gif')
        metadata = json.loads((MEDIA / (stem + '.json')).read_text())
        if metadata['simulation_seconds'] < 6 or metadata['playback_speed'] != 1:
            raise ValueError(f'Expected at least six seconds at original playback speed: {source}')
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(source), '-t', '6', '-filter_complex',
            'fps=10,scale=640:-1:flags=lanczos,split[a][b];'
            '[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=none',
            '-loop', '0', str(target),
        ], check=True)
        with Image.open(target) as gif:
            if not gif.is_animated or gif.n_frames != 60 or gif.info.get('loop') != 0:
                raise ValueError(f'Invalid animation: {target}')
            duration_ms = 0
            frames = set()
            for i in range(gif.n_frames):
                gif.seek(i)
                duration_ms += gif.info['duration']
                frames.add(hashlib.sha256(gif.convert('RGB').tobytes()).digest())
            if duration_ms != 6000 or len(frames) < 2:
                raise ValueError(f'Incorrect timing or static preview: {target}')
            dimensions = list(gif.size)
        previews.append({
            'model': model,
            'source': str(source.relative_to(MEDIA)),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'preview': str(target.relative_to(MEDIA)),
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'start_s': 0, 'duration_s': 6, 'fps': 10, 'frames': 60,
            'playback_speed': 1, 'loop': True, 'dimensions': dimensions,
        })
        print(f'{target.name}: 60 frames, 6 seconds, {target.stat().st_size / 1e6:.2f} MB')
    (MEDIA / 'readme_previews.json').write_text(json.dumps(previews, indent=2) + '\n')
    index_path = MEDIA / 'index.json'
    index = json.loads(index_path.read_text())
    index['assets'] = [
        {'file': str(p.relative_to(MEDIA)), 'bytes': p.stat().st_size,
         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(MEDIA.rglob('*')) if p.suffix in {'.mp4', '.png', '.jpg', '.gif'}
    ]
    index_path.write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
