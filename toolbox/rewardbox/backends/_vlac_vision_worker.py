"""CPU-only frame preparation in VLAC-Cut's pinned native environment."""
from __future__ import annotations

import json
import os
import sys


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    import torchvision
    from qwen_vl_utils import vision_process
    from importlib.metadata import version

    for key, value in {"IMAGE_MAX_TOKEN_NUM": 256, "IMAGE_MIN_TOKEN_NUM": 4,
                       "VIDEO_MAX_TOKEN_NUM": 256, "VIDEO_MIN_TOKEN_NUM": 4}.items():
        setattr(vision_process, key, value)
    runtime = {"python_executable": sys.executable, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "qwen-vl-utils": version("qwen-vl-utils"),
               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            (video, metadata), _ = vision_process.fetch_video(
                {"video": request["paths"], "sample_fps": 2.0}, image_patch_size=16,
                return_video_metadata=True, return_video_sample_fps=True)
            video = video.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
            np.save(request["output"], video, allow_pickle=False)
            reply = {"metadata": {**metadata, "do_sample_frames": False}, "runtime": runtime}
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
