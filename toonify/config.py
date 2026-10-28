"""Paths, settings and the per-run stage cache."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
PLAYER_DIR = REPO_DIR / "player"
SEETHROUGH_DIR = Path(os.environ.get("TOONIFY_SEETHROUGH_DIR", REPO_DIR / "third_party" / "see-through"))
RUNS_DIR = Path(os.environ.get("TOONIFY_RUNS_DIR", REPO_DIR / "runs"))


@dataclass
class ComfySettings:
    """Local ComfyUI used by the image-edit backend (stylize + expression inpainting)."""
    url: str = os.environ.get("TOONIFY_COMFY_URL", "http://127.0.0.1:8188")
    # Used to start ComfyUI when nothing listens on `url`; leave empty to require a running server.
    comfy_dir: str = os.environ.get("TOONIFY_COMFY_DIR", "")
    python: str = os.environ.get("TOONIFY_COMFY_PYTHON", "")
    unet: str = os.environ.get("TOONIFY_QWEN_UNET", "qwen_image_2.1_bf16.safetensors")
    clip: str = os.environ.get("TOONIFY_QWEN_CLIP", "qwen3vl_8b_bf16.safetensors")
    vae: str = os.environ.get("TOONIFY_QWEN_VAE", "qwen_image_2.1_vae_bf16.safetensors")
    steps: int = int(os.environ.get("TOONIFY_QWEN_STEPS", "25"))
    cfg: float = float(os.environ.get("TOONIFY_QWEN_CFG", "1.0"))


def load_presets() -> dict:
    presets = json.loads((PACKAGE_DIR / "presets.json").read_text(encoding="utf-8"))
    extra = os.environ.get("TOONIFY_PRESETS")
    if extra:
        presets.update(json.loads(Path(extra).read_text(encoding="utf-8")))
    return presets


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_id_for(photo: Path, style: str) -> str:
    return f"{photo.stem}-{style}-{file_sha(photo)[:8]}"


@dataclass
class Run:
    dir: Path
    id: str = field(init=False)

    def __post_init__(self):
        self.dir = Path(self.dir)
        self.id = self.dir.name
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # ---- stage cache: a stage is skipped when its recorded fingerprint matches ----
    def fingerprint(self, inputs: list[Path], params: dict) -> str:
        h = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
        for p in inputs:
            h.update(file_sha(p).encode())
        return h.hexdigest()

    def is_cached(self, stage: str, fp: str, outputs: list[Path]) -> bool:
        marker = self.path(f".{stage}.done.json")
        if not marker.exists() or not all(o.exists() for o in outputs):
            return False
        return json.loads(marker.read_text()).get("fingerprint") == fp

    def mark_done(self, stage: str, fp: str, info: dict | None = None) -> None:
        self.path(f".{stage}.done.json").write_text(json.dumps({"fingerprint": fp, **(info or {})}, indent=2))
