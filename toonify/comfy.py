"""Minimal ComfyUI HTTP client: upload images, queue a prompt graph, fetch outputs, free VRAM."""
from __future__ import annotations

import io
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

from .config import ComfySettings


class ComfyClient:
    def __init__(self, settings: ComfySettings | None = None):
        self.s = settings or ComfySettings()
        self.url = self.s.url.rstrip("/")
        self._proc: subprocess.Popen | None = None

    # ---- lifecycle ----
    def alive(self) -> bool:
        try:
            return requests.get(self.url + "/system_stats", timeout=3).ok
        except requests.RequestException:
            return False

    def ensure_running(self, timeout: float = 300) -> None:
        if self.alive():
            return
        if not self.s.comfy_dir:
            raise RuntimeError(f"ComfyUI is not reachable at {self.url}; start it or set TOONIFY_COMFY_DIR")
        comfy_dir = Path(self.s.comfy_dir)
        python = self.s.python or str(comfy_dir.parent / ".venv" / "bin" / "python")
        port = self.url.rsplit(":", 1)[-1]
        log = open(comfy_dir / "toonify-comfy.log", "ab")
        print(f"[comfy] starting ComfyUI from {comfy_dir} on port {port}", file=sys.stderr)
        self._proc = subprocess.Popen([python, "main.py", "--listen", "127.0.0.1", "--port", port],
                                      cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {self._proc.returncode}; see {log.name}")
            if self.alive():
                return
            time.sleep(2)
        raise RuntimeError("ComfyUI did not come up in time")

    def free(self) -> None:
        """Unload models so the next GPU stage (see-through) gets the VRAM; the server keeps running."""
        if self.alive():
            requests.post(self.url + "/free", json={"unload_models": True, "free_memory": True}, timeout=30)
            time.sleep(2)

    def stop(self) -> None:
        """Stop ComfyUI if this client started it."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=60)
            self._proc = None

    # ---- jobs ----
    def upload(self, image: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        image.save(buf, "PNG")
        r = requests.post(self.url + "/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                          data={"overwrite": "true", "type": "input"}, timeout=60)
        r.raise_for_status()
        return r.json()["name"]

    def run(self, graph: dict, timeout: float = 1800) -> list[Image.Image]:
        r = requests.post(self.url + "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"ComfyUI rejected the graph: {r.text[:2000]}")
        pid = r.json()["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = requests.get(f"{self.url}/history/{pid}", timeout=30).json()
            if pid in h:
                entry = h[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI job failed: {status.get('messages')}")
                images = []
                for out in entry.get("outputs", {}).values():
                    for im in out.get("images", []):
                        data = requests.get(self.url + "/view", params=im, timeout=60).content
                        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                if images or status.get("completed"):
                    return images
            time.sleep(1.5)
        raise RuntimeError("ComfyUI job timed out")


def qwen_edit_graph(s: ComfySettings, image_name: str, prompt: str, negative: str, seed: int,
                    lora: str | None = None, lora_strength: float = 1.0, mask_name: str | None = None,
                    denoise: float = 1.0, steps: int | None = None) -> dict:
    """Qwen-Image 2.1 reference edit. With `mask_name` the source latent is noise-masked,
    so only the white mask area is regenerated (inpainting) and the rest stays pixel-aligned
    (same idea as PNGAL's face-variant workflow, Apache-2.0)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": s.unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": s.clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": s.vae}},
        "9": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": prompt, "negative_prompt": negative, "vae": ["3", 0],
            "resolution": 0, "images.image_1": ["9", 0]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["4", 2],
            "seed": seed, "steps": steps or s.steps, "cfg": s.cfg, "sampler_name": "euler", "scheduler": "simple",
            "denoise": denoise}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "toonify"}},
    }
    if lora:
        g["10"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": lora_strength}}
        g["6"]["inputs"]["model"] = ["10", 0]
    if mask_name:
        g["11"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
        g["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["3", 0]}}
        g["13"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["11", 0]}}
        g["6"]["inputs"]["latent_image"] = ["13", 0]
    return g
