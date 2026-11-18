"""Stage 4: one frame set per emotion (face, closed eyes, half / open mouth), same masked inpainting as stage 3.

The emotion face is inpainted from the neutral face over eyebrows, eyes and mouth. Its closed-eye and mouth frames
are then inpainted from that emotion face, not from the neutral one, so brows and eyes stay the same when the mouth
moves. The player crossfades whole sets, which keeps blinking and lip-sync running during an emotion change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..comfy import ComfyClient, qwen_edit_graph
from ..config import Run
from .expressions import (EXPRESSIONS, EYE_TAGS, NEGATIVE, STEPS, WORK, _bbox, _layer_alphas, _union, canvas_image,
                          face_crop, region_mask)

KEEP = ("Keep the face shape, skin color, glasses, hair, lighting, line art and anime art style exactly as they are; "
        "it is the same character, looking straight ahead.")
EMOTIONS = {
    "happy": {
        "adj": "happy", "seed": 21,
        "face": "Make the character look happy: a warm smile with the corners of the closed mouth turned up, "
                "cheerful eyes that are slightly softened by the smile, relaxed eyebrows.",
        "eyes_closed": "Close both eyes in a happy smile: draw each closed eye as an upward-curved arc, no iris "
                       "visible.",
    },
    "sad": {
        "adj": "sad", "seed": 22,
        "face": "Make the character look sad: eyebrows tilted with the inner ends raised, upper eyelids slightly "
                "lowered but the eyes still open, mouth closed with the corners turned down.",
    },
    "angry": {
        "adj": "angry", "seed": 23,
        "face": "Make the character look angry: eyebrows pulled down and together into a frown, narrowed eyes "
                "with lowered upper eyelids, mouth closed and pressed tight with the corners turned down.",
    },
    "surprised": {
        "adj": "surprised", "seed": 24,
        "face": "Make the character look surprised: eyebrows raised high, eyes opened wide with fully visible "
                "irises, lips closed and relaxed.",
    },
}
FACE_DENOISE = 0.92
BLINK = ("Close both eyes completely, as in a relaxed blink: draw each closed eye as a single curved eyelash line, "
         "no iris visible.")


def emotion_masks(alphas: dict, info: dict, size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Soft masks for the emotion face (brows + eyes + mouth) and its closed-eye frame."""
    W, H = size
    layers = info["layers"]
    face = _union(alphas, layers, {"face"})
    fb = _bbox(face) if face is not None else None
    face_w = (fb[2] - fb[0]) if fb else W * 0.25
    eyes = _union(alphas, layers, EYE_TAGS)
    if eyes is None:
        raise SystemExit("emotions: no eye layers (eyewhite/irides/eyelash) in the PSD")
    eyes = eyes > 0.05

    def grow(m: np.ndarray, r: int) -> np.ndarray:
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))) > 0

    def soft(m: np.ndarray) -> np.ndarray:
        feather = max(2, int(face_w * 0.02))
        img = Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather))
        return np.asarray(img, np.float32) / 255

    # Wider than stage 3's eye mask: wide-open or narrowed eyes need room around the neutral eye shape.
    eye_r = max(4, int(face_w * 0.07))
    eye_m = grow(eyes, eye_r)
    brows = _union(alphas, layers, {"eyebrow"})
    lift = max(2, int(face_w * 0.07))
    if brows is not None:
        b = brows > 0.05
    else:   # no brow layer: the band just above the eyes
        b = np.zeros_like(eyes)
        b[:-int(face_w * 0.12)] = eyes[int(face_w * 0.12):]
    # Raised brows move up, frowning brows down and inwards: grow the brow area, more upwards than downwards.
    b = grow(b, max(4, int(face_w * 0.05)))
    up = np.zeros_like(b)
    up[:-lift] = b[lift:]
    brow_m = b | up
    mouth = region_mask("mouth", alphas, info, size) > 0.5
    return {"face": soft(eye_m | brow_m | mouth), "eyes": soft(eye_m)}


def _inpaint(client: ComfyClient, src: Image.Image, mask: np.ndarray, box, prompt: str, seed: int,
             denoise: float, tag: str) -> np.ndarray:
    """Inpaint `mask` of the full canvas `src` on the face crop `box`; returns the blended full canvas."""
    cx0, cy0, cx1, cy1 = box
    crop = src.crop(box).resize((WORK, WORK), Image.LANCZOS)
    image_name = client.upload(crop, f"toonify_emo_{tag}.png")
    mask_crop = Image.fromarray((mask[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)).resize((WORK, WORK), Image.LANCZOS)
    mask_name = client.upload(mask_crop.convert("RGB"), f"toonify_emo_mask_{tag}.png")
    graph = qwen_edit_graph(client.s, image_name, prompt, NEGATIVE, seed, mask_name=mask_name, denoise=denoise,
                            steps=STEPS)
    out_crop = client.run(graph)[0].resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS)
    result = src.copy()
    result.paste(out_crop, (cx0, cy0))
    m = mask[..., None]
    return (np.asarray(src, np.float32) * (1 - m) + np.asarray(result, np.float32) * m).astype(np.uint8)


def _patch(full: np.ndarray, mask: np.ndarray, path: Path) -> list[int]:
    x0, y0, x1, y1 = _bbox(mask, 0.004)
    Image.fromarray(np.dstack([full[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)]), "RGBA").save(path)
    return [x0, y0, x1, y1]


def run(run: Run, force: bool = False, client: ComfyClient | None = None, only: list[str] | None = None) -> Path:
    layers_json, psd = run.path("layers.json"), run.path("avatar.psd")
    if not layers_json.exists():
        raise SystemExit(f"{layers_json} missing: run the decompose stage first")
    names = [n for n in EMOTIONS if not only or n in only]
    unknown = set(only or []) - set(EMOTIONS)
    if unknown:
        raise SystemExit(f"emotions: unknown {sorted(unknown)}; known: {list(EMOTIONS)}")
    out_dir = run.path("emotions")
    out_dir.mkdir(exist_ok=True)
    outputs = [out_dir / f"{n}_{p}.png" for n in EMOTIONS for p in ("face", "eyes_closed", "mouth_half", "mouth_open")]
    fp = run.fingerprint([psd, run.path("stylized.png")],
                         {"work": WORK, "steps": STEPS, "face_denoise": FACE_DENOISE, "emotions": EMOTIONS,
                          "expressions": {k: v["prompt"] for k, v in EXPRESSIONS.items()}})
    if not force and not only and run.is_cached("emotions", fp, outputs):
        print(f"[emotions] cached -> {out_dir}")
        return out_dir

    info = json.loads(layers_json.read_text())
    alphas, size = _layer_alphas(psd)
    base = canvas_image(run, info)
    box = face_crop(alphas, info, size)
    masks = emotion_masks(alphas, info, size)
    mouth_mask = region_mask("mouth", alphas, info, size)
    Image.fromarray((masks["face"] * 255).astype(np.uint8)).save(out_dir / "mask_face.png")
    Image.fromarray((masks["eyes"] * 255).astype(np.uint8)).save(out_dir / "mask_eyes.png")
    client = client or ComfyClient()
    client.ensure_running()
    entries = {e["name"]: e for e in info.get("emotions", [])}
    for name in names:
        spec = EMOTIONS[name]
        mood = f"The character looks {spec['adj']} and keeps that expression. "
        print(f"[emotions] {name}: face")
        face = _inpaint(client, base, masks["face"], box, spec["face"] + " " + KEEP, spec["seed"], FACE_DENOISE,
                        f"{name}_face")
        Image.fromarray(face).save(out_dir / f"preview_{name}_face.png")
        frames = [{"part": "face", "file": f"emotions/{name}_face.png",
                   "bbox": _patch(face, masks["face"], out_dir / f"{name}_face.png")}]
        face_img = Image.fromarray(face)
        eyes_prompt = spec.get("eyes_closed", BLINK)
        jobs = [("eyes_closed", masks["eyes"], mood + eyes_prompt + " Keep the eyebrows and mouth as they are.",
                 EXPRESSIONS["eyes_closed"]["denoise"]),
                ("mouth_half", mouth_mask, mood + EXPRESSIONS["mouth_half"]["prompt"],
                 EXPRESSIONS["mouth_half"]["denoise"]),
                ("mouth_open", mouth_mask, mood + EXPRESSIONS["mouth_open"]["prompt"],
                 EXPRESSIONS["mouth_open"]["denoise"])]
        for i, (part, mask, prompt, denoise) in enumerate(jobs):
            print(f"[emotions] {name}: {part}")
            full = _inpaint(client, face_img, mask, box, prompt, spec["seed"] + 100 * (i + 1), denoise,
                            f"{name}_{part}")
            Image.fromarray(full).save(out_dir / f"preview_{name}_{part}.png")
            frames.append({"part": part, "file": f"emotions/{name}_{part}.png",
                           "bbox": _patch(full, mask, out_dir / f"{name}_{part}.png")})
        entries[name] = {"name": name, "frames": frames}
    info["emotions"] = [entries[k] for k in EMOTIONS if k in entries]
    layers_json.write_text(json.dumps(info, indent=2))
    if not only:
        run.mark_done("emotions", fp)
    print(f"[emotions] -> {out_dir}")
    return out_dir
