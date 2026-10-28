"""Stage 3: exactly three extra frames by masked inpainting on the eye / mouth layers.

Approach follows PNGAL's face-variant workflow (Apache-2.0): the character image is VAE-encoded,
the eye or mouth region is noise-masked, and an identity-preserving edit model (Qwen-Image 2.1)
redraws only that region. Each frame is stored as a cropped RGBA patch in PSD canvas coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..comfy import ComfyClient, qwen_edit_graph
from ..config import Run

KEEP = ("Keep everything else exactly as it is: same face, skin color, glasses, hair, lighting, line art "
        "and anime art style.")
EXPRESSIONS = {
    "eyes_closed": {
        "slot": "eye_close", "region": "eyes", "denoise": 0.9, "seed": 11,
        "prompt": "Close both eyes completely, as in a relaxed blink: draw each closed eye as a single curved "
                  "eyelash line with the upper eyelid lowered, no iris visible. " + KEEP,
    },
    "mouth_half": {
        "slot": "mouth_half", "region": "mouth", "denoise": 0.8, "seed": 12,
        "prompt": "Open the mouth slightly as if speaking: lips parted with a small dark opening and a hint of "
                  "the upper teeth. Eyes stay open. " + KEEP,
    },
    "mouth_open": {
        "slot": "mouth_open", "region": "mouth", "denoise": 0.92, "seed": 13,
        "prompt": "Open the mouth as in normal speech while saying 'ah' (not shouting): a medium-sized opening "
                  "with the upper teeth and a dark mouth interior visible, jaw slightly lowered. Eyes stay open. "
                  + KEEP,
    },
}
NEGATIVE = "deformed face, different person, realistic photo, extra teeth, blurry"
EYE_TAGS = {"eyewhite", "irides", "eyelash", "eyes", "eye"}
MOUTH_TAGS = {"mouth"}


def _layer_alphas(psd_path: Path) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    W, H = psd.width, psd.height
    out: dict[str, np.ndarray] = {}
    for layer in psd:
        if layer.is_group() or layer.width == 0:
            continue
        a = np.zeros((H, W), np.float32)
        x0, y0, x1, y1 = layer.bbox
        im = np.asarray(layer.topil().convert("RGBA"))[..., 3].astype(np.float32) / 255
        a[y0:y1, x0:x1] = im[: y1 - y0, : x1 - x0]
        out[layer.name] = np.maximum(out.get(layer.name, 0), a)
    return out, (W, H)


def _union(alphas: dict, layers: list[dict], tags: set[str]) -> np.ndarray | None:
    hits = [alphas[l["name"]] for l in layers if l["tag"] in tags and l["name"] in alphas]
    return np.maximum.reduce(hits) if hits else None


def _bbox(mask: np.ndarray, thr: float = 0.02):
    ys, xs = np.where(mask > thr)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def region_mask(region: str, alphas: dict, info: dict, size: tuple[int, int]) -> np.ndarray:
    """Soft mask (0..1) of the area the inpainting may change."""
    W, H = size
    layers = info["layers"]
    face = _union(alphas, layers, {"face"})
    fb = _bbox(face) if face is not None else None
    face_w = (fb[2] - fb[0]) if fb else W * 0.25
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if region == "eyes":
        eyes = _union(alphas, layers, EYE_TAGS)
        if eyes is None:
            raise SystemExit("expressions: no eye layers (eyewhite/irides/eyelash) in the PSD")
        r = max(4, int(face_w * 0.05))
        m = Image.fromarray((eyes > 0.05).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))
        mask = np.asarray(m, np.float32) / 255
    else:
        mouth = _union(alphas, layers, MOUTH_TAGS)
        if mouth is None:
            raise SystemExit("expressions: no mouth layer in the PSD")
        x0, y0, x1, y1 = _bbox(mouth)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx = max((x1 - x0) * 0.85, face_w * 0.16)
        ry = max((y1 - y0) * 1.6, face_w * 0.13)
        cy += ry * 0.35   # an opening mouth grows downwards
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(np.float32)
        if face is not None:   # never paint outside the face
            grown = Image.fromarray((face > 0.3).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))
            mask *= np.asarray(grown, np.float32) / 255
    feather = max(2, int(face_w * 0.02))
    soft = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(soft, np.float32) / 255


WORK = 768   # edit resolution of the face crop
STEPS = 16   # masked inpainting of a small region converges faster than a full edit


def face_crop(alphas: dict, info: dict, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square box around the face (from the face / eye / mouth layers) with some margin."""
    W, H = size
    head = _union(alphas, info["layers"], {"face"} | EYE_TAGS | MOUTH_TAGS)
    x0, y0, x1, y1 = _bbox(head, 0.3) if head is not None else (W // 3, H // 8, 2 * W // 3, H // 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(W, H, max(256, round(max(x1 - x0, y1 - y0) * 1.4)))
    left = int(min(max(0, cx - side / 2), W - side))
    top = int(min(max(0, cy - side / 2), H - side))
    return left, top, left + side, top + side


def canvas_image(run: Run, info: dict) -> Image.Image:
    """The stylized character placed on the PSD canvas exactly as see-through saw it, on white."""
    W, H = info["canvas"]
    src = Image.open(run.path("stylized.png")).convert("RGBA")
    s = info["source"]
    w, h = round(s["size"][0] * s["scale"]), round(s["size"][1] * s["scale"])
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    canvas.alpha_composite(src.resize((w, h), Image.LANCZOS), tuple(s["offset"]))
    return canvas.convert("RGB")


def run(run: Run, force: bool = False, client: ComfyClient | None = None, only: list[str] | None = None) -> Path:
    layers_json, psd = run.path("layers.json"), run.path("avatar.psd")
    if not layers_json.exists():
        raise SystemExit(f"{layers_json} missing: run the decompose stage first")
    out_dir = run.path("expressions")
    out_dir.mkdir(exist_ok=True)
    outputs = [out_dir / f"{k}.png" for k in EXPRESSIONS]
    fp = run.fingerprint([psd, run.path("stylized.png")],
                         {"work": WORK, "steps": STEPS, **{k: dict(v) for k, v in EXPRESSIONS.items()}})
    if not force and not only and run.is_cached("expressions", fp, outputs):
        print(f"[expressions] cached -> {out_dir}")
        return out_dir

    info = json.loads(layers_json.read_text())
    alphas, size = _layer_alphas(psd)
    base = canvas_image(run, info)
    base.save(out_dir / "base_canvas.png")
    # Inpaint on a square face crop, upscaled to WORK px: ~3-4x fewer pixels than the full canvas, and the
    # edit model draws the eyes and mouth at a higher resolution.
    cx0, cy0, cx1, cy1 = face_crop(alphas, info, size)
    crop = base.crop((cx0, cy0, cx1, cy1)).resize((WORK, WORK), Image.LANCZOS)
    client = client or ComfyClient()
    client.ensure_running()
    image_name = client.upload(crop, "toonify_face.png")
    entries = {e["name"]: e for e in info.get("expressions", [])}
    for name, spec in EXPRESSIONS.items():
        if only and name not in only:
            continue
        mask = region_mask(spec["region"], alphas, info, size)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"mask_{name}.png")
        mask_crop = Image.fromarray((mask[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)).resize((WORK, WORK), Image.LANCZOS)
        mask_name = client.upload(mask_crop.convert("RGB"), f"toonify_mask_{name}.png")
        print(f"[expressions] inpainting {name} (denoise {spec['denoise']}, face crop {cx1 - cx0}px -> {WORK}px)")
        graph = qwen_edit_graph(client.s, image_name, spec["prompt"], NEGATIVE, spec["seed"],
                                mask_name=mask_name, denoise=spec["denoise"], steps=STEPS)
        out_crop = client.run(graph)[0].resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS)
        result = base.copy()
        result.paste(out_crop, (cx0, cy0))
        m = mask[..., None]
        full = (np.asarray(base, np.float32) * (1 - m) + np.asarray(result, np.float32) * m).astype(np.uint8)
        Image.fromarray(full).save(out_dir / f"preview_{name}.png")
        x0, y0, x1, y1 = _bbox(mask, 0.004)
        patch = np.dstack([full[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)])
        Image.fromarray(patch, "RGBA").save(out_dir / f"{name}.png")
        entries[name] = {"name": name, "slot": spec["slot"], "file": f"expressions/{name}.png",
                         "bbox": [x0, y0, x1, y1]}
    info["expressions"] = [entries[k] for k in EXPRESSIONS if k in entries]
    layers_json.write_text(json.dumps(info, indent=2))
    if not only:
        run.mark_done("expressions", fp)
    print(f"[expressions] -> {out_dir}")
    return out_dir
