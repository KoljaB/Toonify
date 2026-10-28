"""Stage 1: photo -> anime character PNG (transparent, ~1280 px long side)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..comfy import ComfyClient, qwen_edit_graph
from ..config import ComfySettings, Run, load_presets

LONG_SIDE = 1280


class StylizeBackend(ABC):
    name: str

    @abstractmethod
    def stylize(self, photo: Image.Image, preset: dict, seed: int) -> Image.Image:
        """Return an RGB anime rendition of the person in `photo`."""


class ComfyQwenBackend(StylizeBackend):
    """Local: Qwen-Image 2.1 reference edit in ComfyUI (identity-preserving image edit)."""
    name = "comfy-qwen"

    def __init__(self, client: ComfyClient | None = None):
        self.client = client or ComfyClient()

    def stylize(self, photo, preset, seed):
        self.client.ensure_running()
        name = self.client.upload(photo, "toonify_photo.png")
        graph = qwen_edit_graph(self.client.s, name, preset["prompt"], preset.get("negative", ""), seed,
                                preset.get("lora"), preset.get("lora_strength", 1.0))
        return self.client.run(graph)[0]


class ApiBackend(StylizeBackend):
    """Opt-in hosted image-edit API. Not implemented: raises so nothing pretends to work."""
    name = "api"

    def stylize(self, photo, preset, seed):
        raise NotImplementedError("API stylize backend is a stub; use --backend comfy-qwen")


BACKENDS = {"comfy-qwen": ComfyQwenBackend, "api": ApiBackend}


def fit_size(w: int, h: int, long_side: int = LONG_SIDE, multiple: int = 32) -> tuple[int, int]:
    s = long_side / max(w, h)
    return (max(multiple, round(w * s / multiple) * multiple), max(multiple, round(h * s / multiple) * multiple))


def _session(name: str):
    """rembg session on CUDA when onnxruntime-gpu is installed (falls back to CPU)."""
    import onnxruntime as ort
    from rembg import new_session
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()   # CUDA/cuDNN from the torch wheels
    return new_session(name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def frame_upper_body(photo: Image.Image) -> tuple[Image.Image, bool]:
    """If the person is shown (nearly) full-body, crop the input to head..waist before the edit, so the
    edit model redraws the upper body at full resolution instead of drawing a small full figure."""
    from rembg import remove
    small = photo.copy()
    small.thumbnail((768, 768))
    mask = np.asarray(remove(small, session=_session("birefnet-general"), only_mask=True)) > 128
    ys, xs = np.where(mask)
    if not len(ys):
        return photo, False
    sx, sy = photo.width / small.width, photo.height / small.height
    x0, x1, y0, y1 = xs.min() * sx, xs.max() * sx, ys.min() * sy, ys.max() * sy
    h, w = y1 - y0, x1 - x0
    if h < 1.9 * w * 0.75:   # already a bust / upper-body shot
        return photo, False
    top = max(0, y0 - 0.04 * h)
    bottom = min(photo.height, y0 + 0.52 * h)
    cw = min(photo.width, (bottom - top) * 2 / 3 * 1.15)
    cx = (x0 + x1) / 2
    left = min(max(0, cx - cw / 2), photo.width - cw)
    return photo.crop((round(left), round(top), round(left + cw), round(bottom))), True


def remove_background(img: Image.Image) -> Image.Image:
    """BiRefNet (best on hair strands) via rembg -> RGBA, then edge clean-up."""
    from rembg import remove
    out = remove(img, session=_session("birefnet-general"))
    rgba = np.asarray(out.convert("RGBA")).astype(np.float32)
    rgb, a = rgba[..., :3], rgba[..., 3] / 255
    a[a < 0.05] = 0
    bg = np.median(np.asarray(img, np.float32)[a == 0], axis=0) if (a == 0).any() else np.array([255.0] * 3)
    # Peel off background-coloured pixels that touch transparency (white specks in hair gaps).
    from scipy import ndimage
    near_bg = np.abs(rgb - bg).max(axis=2) < 22
    for _ in range(4):
        edge = ndimage.binary_dilation(a == 0) & (a > 0) & near_bg
        if not edge.any():
            break
        a[edge] = 0
    # The edit model likes to draw a white sticker outline around hair; drop small bright, unsaturated
    # blobs that touch the background (large white areas such as a white shirt are kept).
    bright = (rgb.min(axis=2) > 200) & (rgb.max(axis=2) - rgb.min(axis=2) < 30) & (a > 0)
    lab, n = ndimage.label(bright)
    if n:
        touching = np.unique(lab[ndimage.binary_dilation(a == 0, iterations=2) & bright])
        sizes = ndimage.sum(bright, lab, range(1, n + 1))
        small = [i for i in touching if i and sizes[i - 1] < a.size * 0.002]
        a[np.isin(lab, small)] = 0
    # Un-mix the background from semi-transparent edge pixels: C = a*F + (1-a)*B  ->  F.
    semi = (a > 0) & (a < 0.98)
    rgb[semi] = np.clip((rgb[semi] - (1 - a[semi, None]) * bg) / a[semi, None], 0, 255)
    rgba = np.dstack([rgb, a * 255]).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def run(run: Run, photo_path: Path, style: str, backend: str = "comfy-qwen", seed: int = 7,
        force: bool = False, client: ComfyClient | None = None) -> Path:
    presets = load_presets()
    if style not in presets:
        raise SystemExit(f"unknown style '{style}', available: {', '.join(presets)}")
    preset = presets[style]
    out_raw, out = run.path("stylized_raw.png"), run.path("stylized.png")
    gen_fp = run.fingerprint([photo_path], {"preset": preset, "backend": backend, "seed": seed, "long": LONG_SIDE, "frame": 1})
    fp = gen_fp + "-cut4"   # bump when the background removal changes
    if not force and run.is_cached("stylize", fp, [out_raw, out]):
        print(f"[stylize] cached -> {out}")
        return out

    be = ComfyQwenBackend(client) if backend == "comfy-qwen" else BACKENDS[backend]()
    if force or not run.is_cached("stylize_gen", gen_fp, [out_raw]):
        photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
        photo, cropped = frame_upper_body(photo)
        if cropped:
            photo.save(run.path("input_framed.png"))
            print(f"[stylize] full-body input: cropped to the upper body {photo.size} before editing")
        print(f"[stylize] backend={be.name} style={style}")
        img = be.stylize(photo, preset, seed)
        img = img.resize(fit_size(*img.size), Image.LANCZOS)
        img.save(out_raw)
        run.mark_done("stylize_gen", gen_fp)
        if isinstance(be, ComfyQwenBackend):
            be.client.free()   # give the VRAM to the segmentation model and the next stage
    else:
        img = Image.open(out_raw).convert("RGB")
    print("[stylize] removing background (BiRefNet)")
    remove_background(img).save(out)
    run.path("stylize.json").write_text(json.dumps({"style": style, "backend": be.name, "seed": seed,
                                                    "size": img.size, "source": str(photo_path)}, indent=2))
    run.mark_done("stylize", fp)
    print(f"[stylize] -> {out} {img.size}")
    return out
