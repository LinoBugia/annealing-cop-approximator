from PIL import Image
import numpy as np
import os
import imageio.v2 as imageio

def images_to_video(folder, pattern="img%03d.png", start=1, end=None,
                    fps=24, durations=None, output="out.mp4"):
    """
    Create a video from numbered images.

    - folder: directory containing images
    - pattern: e.g. "img%03d.png"  (uses Python % formatting with frame index)
    - start, end: first and last index (inclusive). If end is None, it auto-detects.
    - fps: frames per second of the output file (controls playback speed)
    - durations: optional list of per-frame durations in seconds (same length as frames).
                 When provided, frames are duplicated: repeats = round(duration * fps).
    - output: output file path (e.g. "/path/out.mp4")
    """
    if end is None:
        # simple auto-detect (scans forward until misses)
        i = start
        found = []
        misses = 0
        while misses < 10 and i < 100000:
            p = os.path.join(folder, pattern % i)
            if os.path.exists(p):
                found.append(i)
                misses = 0
            else:
                misses += 1
            i += 1
        if not found:
            raise FileNotFoundError("No files found using that pattern in the folder.")
        start, end = found[0], found[-1]

    indices = list(range(start, end+1))
    files = [os.path.join(folder, pattern % i) for i in indices]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")

    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img))

    writer = imageio.get_writer(output, fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        if durations is None:
            for frame in frames:
                writer.append_data(frame)
        else:
            if len(durations) != len(frames):
                raise ValueError("durations must match number of frames")
            for frame, dur in zip(frames, durations):
                repeats = max(1, int(round(dur * fps)))
                for _ in range(repeats):
                    writer.append_data(frame)
    finally:
        writer.close()