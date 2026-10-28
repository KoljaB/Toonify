# Licenses of everything Toonify pulls in

Checked 2026-10-28 (repository LICENSE files and Hugging Face model cards).

> **Commercial use warning:** the local stylize/expression backend uses **Qwen-Image 2.1**, released under the
> **Qwen Research License** (research / non-commercial). Generated avatars inherit that restriction unless you
> swap the backend (the `StylizeBackend` interface) or obtain a commercial license.

## Code

| Component | How it is used | License |
|---|---|---|
| [see-through](https://github.com/shitagaki-lab/see-through) @ `df019de` | git submodule `third_party/see-through`, run unmodified | Apache-2.0 |
| [Anime2.5DRig](https://github.com/852wa/Anime2.5DRig) v2.0.0 | vendored `player/vendor/rigger.js` (+`mouth_half` slot), `renderer.js`; `player/rig.js` derived from its `app.js` | MIT (`player/vendor/LICENSE-Anime2.5DRig`) |
| [ag-psd](https://github.com/Agamnentzar/ag-psd) | `player/vendor/ag-psd.min.js` (bundled with Anime2.5DRig) | MIT |
| [PNGAL](https://github.com/1mm-module/PNGAL) | approach only (masked Qwen edit for face variants); no code copied | Apache-2.0 |
| [PachiPakuGen](https://github.com/kazuya-bros/PachiPakuGen) | inspected, nothing reused | MIT |
| [StretchyStudio](https://github.com/MangoLion/stretchystudio) | inspected, nothing reused | MIT |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | local inference server for Qwen-Image (HTTP API) | GPL-3.0 (separate process, not linked) |
| [rembg](https://github.com/danielgatis/rembg) | background removal / person framing | MIT |
| aiohttp, websockets, numpy, Pillow, psd-tools, requests, scipy | Python dependencies | Apache-2.0 / BSD / MIT / HPND |
| [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) | example `examples/realtimetts_stream.py` only | MIT |

## Models

| Model | Used by | License |
|---|---|---|
| Qwen-Image 2.1 (`qwen_image_2.1_bf16`, VAE) via Comfy-Org repack | stylize, expressions | **Qwen Research License** |
| Qwen3-VL-8B (text encoder for Qwen-Image 2.1) | stylize, expressions | Apache-2.0 |
| `layerdifforg/seethroughv0.0.2_layerdiff3d` | decompose (see-through) | OpenRAIL++ |
| `layerdifforg/seethroughv0.0.1_marigold` | decompose (see-through depth) | OpenRAIL++ |
| `24yearsold/l2d_sam_iter2` (if the annotator tier is used) | see-through body parsing | Apache-2.0 |
| rembg `isnet-anime` (SkyTNT anime-segmentation) | cutout after stylize | Apache-2.0 |
| rembg `isnet-general-use` (DIS) | upper-body framing of the input | Apache-2.0 |
| Qwen3-TTS 12Hz 0.6B (via the RealtimeTTS Qwen server) | TTS example only | Apache-2.0 |
| Kokoro-82M (optional TTS example engine) | TTS example only | Apache-2.0 |

OpenRAIL++ carries use-based restrictions (no harmful use, attribution); read the model cards before
distributing outputs commercially.
