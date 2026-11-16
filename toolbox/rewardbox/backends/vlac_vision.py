"""Persistent optional CPU bridge, keeping native resize rounding separate from vLLM."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class NativeVLACVision:
    def __init__(self, python: str):
        self.python = str(Path(python).expanduser().absolute())
        if not Path(self.python).is_file() or not os.access(self.python, os.X_OK):
            raise ValueError(f"vision_preprocess_python is not an executable: {self.python}")
        self.process = None
        self.runtime = {}

    def prepare(self, paths: list[str]):
        if self.process is None:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = ""
            self.process = subprocess.Popen(
                [self.python, "-u", str(Path(__file__).with_name("_vlac_vision_worker.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=environment)
        if self.process.poll() is not None:
            raise RuntimeError("native VLAC CPU preprocessing worker exited")
        with tempfile.TemporaryDirectory(prefix="rewardbox_vlac_vision_") as temporary:
            output = Path(temporary) / "video.npy"
            self.process.stdin.write(json.dumps({"paths": paths, "output": str(output)}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("native VLAC CPU preprocessing worker returned no response")
            reply = json.loads(line)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            self.runtime = reply["runtime"]
            return np.load(output, allow_pickle=False), reply["metadata"]

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.process.stdout.close()
            self.process = None
