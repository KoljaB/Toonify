"""Stage 2: transparent PNG -> layered PSD via see-through (git submodule, unmodified) + layers.json."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from ..config import SEETHROUGH_DIR, Run

RESOLUTION = 1280
STEPS = int(os.environ.get("TOONIFY_SEETHROUGH_STEPS", "30"))   # LayerDiff steps per pass; 20 dropped the face layer in a test
VRAM_MODES = ("auto", "full", "group_offload", "nf4", "blockswap")


def detect_vram_mode() -> tuple[str, float]:
    """Pick see-through's VRAM mode from the detected GPU (README: ~12-16 GB bf16, ~10 GB offload, ~8 GB nf4)."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("decompose needs an NVIDIA GPU (torch.cuda.is_available() is False)")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total >= 16:
        return "full", total
    if total >= 11:
        return "group_offload", total
    return "nf4", total


def seethrough_cmd(mode: str, src: Path, save_dir: Path) -> list[str]:
    script = {"nf4": "inference_psd_quantized.py", "blockswap": "inference_psd_blockswap.py"}.get(mode, "inference_psd.py")
    cmd = [sys.executable, f"inference/scripts/{script}", "--srcp", str(src), "--save_dir", str(save_dir),
           "--save_to_psd", "--disable_progressbar", "--inference_steps", str(STEPS)]
    if mode == "group_offload":
        cmd.append("--group_offload")
    return cmd


def tag_of(name: str) -> str:
    """see-through layer name -> semantic tag ('hairf' stays 'hairf'; 'eyes-l' style suffixes dropped)."""
    return re.sub(r"([-_ ]?\d+|[-_ ][lr])$", "", name.strip().lower()) or name


HEAD_TAGS = {"face", "eyewhite", "irides", "eyelash", "eyebrow", "eyes", "mouth", "nose", "eyewear", "front hair",
             "ears", "headwear"}
# Layers the occlusion repair may trim: body layers plus see-through's "completed" glasses lenses and headwear,
# which it sometimes paints opaque over the eyes. Facial features and hair are left alone.
NO_TRIM = HEAD_TAGS - {"eyewear", "headwear"}


def _blobs(mask, min_px: int, close: int = 0):
    """Keep connected components of `mask` with at least `min_px` pixels (after a small opening)."""
    import numpy as np
    from scipy import ndimage
    mask = ndimage.binary_opening(mask, iterations=2)
    if close:
        mask = ndimage.binary_dilation(mask, iterations=close)
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))


def repair_layers(psd_path: Path, src_png: Path, out_path: Path) -> list[str]:
    """Make the rest pose match the source image again. see-through completes hidden parts of each layer
    (amodal), and sometimes such a part ends up drawn *over* a layer it should be behind (a crop top
    continued over the trousers), or a part has no class at all (headphones around the neck) and is lost.

    1. Where the topmost body layer disagrees with the source but the layer below it agrees, cut the top layer.
    2. Large regions that still differ are re-added from the source as an `objects` layer above the body.
    Returns a list of human-readable fixes."""
    import numpy as np
    from PIL import ImageFilter
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    W, H = psd.width, psd.height
    sw, sh = Image.open(src_png).size
    side = max(sw, sh)
    src = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    src.alpha_composite(Image.open(src_png).convert("RGBA"), ((side - sw) // 2, (side - sh) // 2))
    src = np.asarray(src.resize((W, H), Image.LANCZOS), np.float32) / 255
    layers = [l for l in psd if not l.is_group() and l.width and l.height]
    full = []
    for l in layers:
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        im.alpha_composite(l.topil().convert("RGBA"), (l.left, l.top))
        full.append(np.asarray(im, np.float32) / 255)
    tags = [tag_of(l.name) for l in layers]
    fixes = []
    min_px = int(W * H * 0.0008)

    # 1. occlusion repair
    for _ in range(2):
        alpha = np.stack([f[..., 3] for f in full])            # (N, H, W)
        solid = alpha > 0.5
        n = len(full)
        rev = solid[::-1]
        top = n - 1 - rev.argmax(0)                             # topmost solid layer per pixel
        has_top = rev.any(0)
        below = np.where(np.arange(n)[:, None, None] < top[None], solid, False)
        has_below = below.any(0)
        second = n - 1 - below[::-1].argmax(0)
        rgb = np.stack([f[..., :3] for f in full])
        yy, xx = np.mgrid[0:H, 0:W]
        err_top = np.abs(rgb[top, yy, xx] - src[..., :3]).max(-1)
        err_sec = np.abs(rgb[second, yy, xx] - src[..., :3]).max(-1)
        changed = False
        for i, t in enumerate(tags):
            if t in NO_TRIM:
                continue
            bad = has_top & has_below & (top == i) & (src[..., 3] > 0.5) & (err_top > 0.22) & (err_sec < 0.12)
            bad = _blobs(bad, min_px, close=2)
            if bad.any():
                cut = np.asarray(Image.fromarray(bad.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5)),
                                 np.float32) / 255
                full[i][..., 3] *= 1 - cut
                fixes.append(f"trimmed {layers[i].name} where it covered a lower layer ({int(bad.sum())} px)")
                changed = True
        if not changed:
            break

    # 2. lost regions
    comp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for f in full:
        comp.alpha_composite(Image.fromarray((f * 255).astype(np.uint8), "RGBA"))
    comp = np.asarray(comp, np.float32) / 255
    head = np.max([f[..., 3] for f, t in zip(full, tags) if t in HEAD_TAGS] or [np.zeros((H, W))], axis=0)
    diff = np.abs(comp[..., :3] - src[..., :3]).max(axis=2)
    from scipy import ndimage
    lost = (src[..., 3] > 0.6) & ((diff > 0.18) | (comp[..., 3] < 0.4)) & (head < 0.3)
    # A dark object on dark clothes only differs along its outline: close the outline into a solid shape.
    r = max(3, W // 160)
    lost = ndimage.binary_fill_holes(ndimage.binary_closing(lost, iterations=r)) & (src[..., 3] > 0.6) & (head < 0.3)
    keep = _blobs(lost, int(W * H * 0.0015), close=1)
    patches = {}
    if keep.any():
        # Regions above the chin move with the head (e.g. a cut-off hair bun), the rest with the body.
        face_a = next((f[..., 3] for f, t in zip(full, tags) if t == "face"), None)
        chin = np.flatnonzero((face_a > 0.5).any(1)).max() if face_a is not None and (face_a > 0.5).any() else H // 3
        lab, n = ndimage.label(keep)
        cy = ndimage.center_of_mass(keep, lab, range(1, n + 1))
        for name, sel in (("headwear", [i + 1 for i, c in enumerate(cy) if c[0] < chin]),
                          ("objects", [i + 1 for i, c in enumerate(cy) if c[0] >= chin])):
            m = np.isin(lab, sel)
            if not m.any():
                continue
            soft = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.5)),
                              np.float32) / 255
            patches[name] = Image.fromarray((np.dstack([src[..., :3], src[..., 3] * soft]) * 255).astype(np.uint8), "RGBA")
            fixes.append(f"re-added {int(m.sum())} px see-through dropped as layer '{name}'")
    if not fixes:
        return []

    out = PSDImage.new(mode="RGBA", size=(W, H), depth=8)
    first_head = next((i for i, t in enumerate(tags) if t in HEAD_TAGS), len(layers))
    def add(img: Image.Image, name: str):
        bbox = img.getbbox()
        if bbox:
            out.create_pixel_layer(img.crop(bbox), name=name, top=bbox[1], left=bbox[0], opacity=255)
    for i, l in enumerate(layers):
        if i == first_head:
            for name, patch in patches.items():
                add(patch, name)
        add(Image.fromarray((full[i] * 255).round().astype(np.uint8), "RGBA"), l.name)
    out.save(out_path)
    return fixes


def write_layers_json(psd_path: Path, src_png: Path, out: Path) -> dict:
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    W, H = psd.width, psd.height
    sw, sh = Image.open(src_png).size
    side = max(sw, sh)
    scale = W / side
    layers = []
    for order, layer in enumerate(psd):   # psd-tools iterates bottom (drawn first) -> top
        if layer.is_group() or layer.width == 0 or layer.height == 0:
            continue
        x0, y0, x1, y1 = layer.bbox
        layers.append({"name": layer.name, "tag": tag_of(layer.name), "bbox": [x0, y0, x1, y1], "order": order})
    info = {
        "psd": psd_path.name, "canvas": [W, H],
        # see-through center-pads the source to a square and resizes it to RESOLUTION.
        "source": {"file": src_png.name, "size": [sw, sh], "scale": scale,
                   "offset": [round((side - sw) // 2 * scale), round((side - sh) // 2 * scale)]},
        "layers": layers,
    }
    out.write_text(json.dumps(info, indent=2))
    return info


def run(run: Run, vram: str = "auto", force: bool = False, before_gpu=None) -> Path:
    src = run.path("stylized.png")
    psd_out, layers_out = run.path("avatar.psd"), run.path("layers.json")
    if not src.exists():
        raise SystemExit(f"{src} missing: run the stylize stage first")
    st_fp = run.fingerprint([src], {"resolution": RESOLUTION, "steps": STEPS, "seethrough": _submodule_rev()})
    fp = st_fp + "-post6"   # bump when the post-processing below changes
    if not force and run.is_cached("decompose", fp, [psd_out, layers_out]):
        print(f"[decompose] cached -> {psd_out}")
        return psd_out
    save_dir = run.path("seethrough")
    save_dir.mkdir(exist_ok=True)
    produced = save_dir / (src.stem + ".psd")
    mode = "cached"
    if force or not run.is_cached("seethrough", st_fp, [produced]):
        if before_gpu:
            before_gpu()   # e.g. unload ComfyUI models
        mode = vram
        if vram == "auto":
            mode, total = detect_vram_mode()
            print(f"[decompose] GPU has {total:.1f} GiB -> see-through mode '{mode}'")
        cmd = seethrough_cmd(mode, src.resolve(), save_dir.resolve())
        print("[decompose] " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=SEETHROUGH_DIR)
        if proc.returncode != 0:
            raise SystemExit(f"see-through failed with exit code {proc.returncode}")
        if not produced.exists():
            raise SystemExit(f"see-through finished but {produced} is missing")
        run.mark_done("seethrough", st_fp, {"vram_mode": mode})
    else:
        print("[decompose] see-through output cached; redoing post-processing")
    shutil.copyfile(produced, psd_out)
    for fix in repair_layers(psd_out, src, psd_out):
        print(f"[decompose] {fix}")
    info = write_layers_json(psd_out, src, layers_out)
    run.mark_done("decompose", fp, {"vram_mode": mode})
    print(f"[decompose] {len(info['layers'])} layers: {', '.join(l['name'] for l in info['layers'])}")
    return psd_out


def _submodule_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SEETHROUGH_DIR, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"
