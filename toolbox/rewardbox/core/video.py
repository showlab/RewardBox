"""Frame reading and sampling.  **The only module in the repository allowed to import cv2.**"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .types import SamplingSpec

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_DECODE = 100_000


def read_frames(path, spec: SamplingSpec) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (frames[T,H,W,3] uint8 RGB, frame_indices[T], source_fps).

    Decodes sequentially instead of seeking frame by frame with
    cap.set(CAP_PROP_POS_FRAMES): seeking is slow on VBR/B-frame videos and can land on
    the wrong frame (all three upstream implementations seek).
    """
    path = Path(path)
    if path.is_dir():
        f, i = read_frames_dir(path, spec)
        return f, i, None
    if not path.exists():
        raise FileNotFoundError(f"video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video (missing codec?): {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        if total <= 0:
            cap.release()
            return _decode_all(path, spec, fps)

        want = set(spec.indices(total, fps).tolist())
        frames, idx, i = [], [], 0
        while i < _MAX_DECODE:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                idx.append(i)
            i += 1
        if len(frames) < len(want):          # container metadata is unreliable
            cap.release()
            return _decode_all(path, spec, fps)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"no frame could be decoded: {path}")
    arr = np.stack(frames)
    if spec.resize:
        arr = resize_frames(arr, spec.resize)
    return arr, np.asarray(idx, dtype=np.int64), fps


def _decode_all(path, spec, fps):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < _MAX_DECODE:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frame could be decoded: {path}")
    sel = spec.indices(len(frames), fps)
    arr = np.stack([frames[i] for i in sel])
    if spec.resize:
        arr = resize_frames(arr, spec.resize)
    return arr, sel, fps


def read_frames_dir(d, spec: SamplingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Image directory, sorted in **natural order** (frame2.png before frame10.png)."""
    d = Path(d)
    files = sorted((p for p in d.iterdir() if p.suffix.lower() in IMG_EXT),
                   key=lambda p: [int(s) if s.isdigit() else s.lower()
                                  for s in re.split(r"(\d+)", p.name)])
    if not files:
        raise FileNotFoundError(f"no images in directory: {d}")
    sel = spec.indices(len(files))
    out = []
    for i in sel:
        bgr = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read image: {files[i]}")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    arr = np.stack(out)
    if spec.resize:
        arr = resize_frames(arr, spec.resize)
    return arr, sel


def resize_frames(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h).  INTER_AREA to shrink, INTER_LINEAR to enlarge."""
    w, h = size
    if arr.shape[2] == w and arr.shape[1] == h:
        return arr
    interp = cv2.INTER_AREA if (arr.shape[1] > h or arr.shape[2] > w) else cv2.INTER_LINEAR
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in arr])


def write_png_dir(frames: np.ndarray, out_dir, indices=None) -> Path:
    """Write frame_{:06d}.png; Robo-Dopamine's list_pngs_sorted needs this naming."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = range(len(frames)) if indices is None else indices
    for i, f in zip(idx, frames):
        cv2.imwrite(str(out_dir / f"frame_{int(i):06d}.png"),
                    cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return out_dir


def to_pil(frame: np.ndarray):
    from PIL import Image
    return Image.fromarray(frame)          # frame is already RGB


def write_video(path, frames: np.ndarray, fps: float = 10.0) -> Path:
    """For test fixtures only."""
    path = Path(path)
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()
    return path


def encode_image(frame: np.ndarray, fmt: str = "png", *, jpeg_quality: int = 90,
                 max_side: int | None = None) -> bytes:
    """Encode one RGB frame for an image-bearing API request.

    ``fmt`` is ``"png"`` (lossless) or ``"jpeg"``.  ``max_side`` optionally shrinks the
    longer side (INTER_AREA, aspect kept); frames are never enlarged.
    """
    if fmt not in ("png", "jpeg"):
        raise ValueError(f"image format must be 'png' or 'jpeg', got {fmt!r}")
    h, w = frame.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (max(1, int(w * scale + .5)), max(1, int(h * scale + .5))),
                           interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if fmt == "png":
        ok, buffer = cv2.imencode(".png", bgr)
    else:
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"{fmt} encoding failed")
    return buffer.tobytes()


def letterbox_jpeg(frame: np.ndarray, side: int, *, prescale: float = 1.0,
                   quality: int = 70) -> bytes:
    """RGB frame -> optional prescale -> aspect fit into ``side`` x ``side`` on black ->
    baseline JPEG.  All resizes use INTER_AREA (the Jev 1.14 input recipe)."""
    h, w = frame.shape[:2]
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if prescale != 1.0:
        bgr = cv2.resize(bgr, (int(w * prescale + .5), int(h * prescale + .5)),
                         interpolation=cv2.INTER_AREA)
    ih, iw = bgr.shape[:2]
    factor = min(side / iw, side / ih)
    tw, th = int(iw * factor + .5), int(ih * factor + .5)
    fitted = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    top, left = (side - th) // 2, (side - tw) // 2
    boxed = cv2.copyMakeBorder(fitted, top, side - th - top, left, side - tw - left,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ok, buffer = cv2.imencode(
        ".jpg", boxed, [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_PROGRESSIVE, 0])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
