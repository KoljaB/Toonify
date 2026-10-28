# Decisions (Phase 0)

Inspected on 2026-10-28: see-through @ `df019de`, Anime2.5DRig v2.0.0, StretchyStudio, PNGAL, PachiPakuGen v0.4.1.

## see-through layer conventions

- `inference/scripts/inference_psd.py --save_to_psd` writes `<save_dir>/<stem>.psd`.
- The source image is **center-padded to a square and resized to `--resolution` (1280)**; the PSD canvas is that
  square. Toonify records the padding offset/scale in `layers.json` so later stages can map image ↔ PSD coordinates.
- Layers are pixel layers named after their semantic tag (`face`, `hair…`, `eyewhite`, `irides`, `eyelash`,
  `eyebrow`, `mouth`, `nose`, `ears`, `neck`, `topwear`, `handwear`, `eyewear`, …), sorted back-to-front by median
  depth. `mouth` is the closed neutral mouth taken from the input.
- VRAM modes: bf16 default (~12–16 GB), `--group_offload` (~10 GB), `inference_psd_quantized.py` NF4 (~8 GB),
  `inference_psd_blockswap.py` (~8 GB).

## Animation base: Anime2.5DRig

| | Anime2.5DRig | StretchyStudio | PNGAL | PachiPakuGen |
|---|---|---|---|---|
| Runtime | browser, plain JS (no build) | browser, React/Vite app + editor | Windows desktop (Python server + web UI) | Tauri desktop app |
| Input | see-through PSD names directly | see-through PSD, auto-rig | transparent PNG → PSD | see-through PSD + 7 expression PNGs |
| Rig code size | `rigger.js` 744 lines + `renderer.js` 126 (WebGL1) + idle loop in `app.js` | large editor codebase, warp deformers | animation export, no live audio | motion lab in TS, SpriTalk-oriented |
| License | MIT | MIT | Apache-2.0 | MIT |

**Chosen: Anime2.5DRig.** It already turns a see-through PSD into a meshed, depth-layered WebGL puppet with exactly
the idle features the spec asks for (random blink with eye-close cross-fade, breathing, head sway with parallax,
hair strand spring physics), its rigger is a pure function (`buildRig(psd)`) and the renderer is a 126-line WebGL1
mesh drawer, which is cheap on integrated GPUs. StretchyStudio is an editor application; extracting its runtime
would mean pulling React state and its warp-deformer pipeline. PNGAL and PachiPakuGen are desktop tools.

Input conventions it relies on: layer names (see-through or aliases), `eye_close` / `mouth_close` /
`mouth_open` as expression layers, left/right split of eye layers by connected components, anchors derived from
`face`, `eyewhite`, `irides`, `mouth`, `neck`.

**Vendored** (`player/vendor/`): `rigger.js` (+ one added slot `mouth_half`), `renderer.js` (unchanged),
`ag-psd.min.js` (MIT, PSD reader). `player/rig.js` is the animation core of `app.js` (prepare meshes, `deform`,
`updateSprings`, idle `animate`, `render`) with everything else **stripped**: editor UI, settings/undo, anchor
editor, OBS relay, camera face tracking (MediaPipe), microphone, random "talk", presets, recorder/export, bust and
arm controls, eyeClose2 variant. **Added**: three discrete mouth states with a 25 ms cross-fade, driven by the
audio analyser.

## Expression frames

PNGAL's face-variant workflow (Apache-2.0) does masked inpainting with Qwen-Image-Edit in ComfyUI
(`VAEEncode` → `SetLatentNoiseMask` → `KSampler`, partial denoise). Toonify follows the same approach with the
locally installed Qwen-Image 2.1 edit model: masks come from the see-through eye/mouth layers, only the masked
region is regenerated, and the result is stored as a cropped RGBA patch in PSD coordinates. PachiPakuGen expects
the expression images to be produced externally (Codex / other editors), so there was no code to reuse there.

## Stylize backend

Local backend = ComfyUI + Qwen-Image 2.1 (reference-image edit, keeps identity, glasses, clothing), already set up on
the GPU server. Background removal: rembg `isnet-anime` (anime character segmentation). The API backend is a stub
that raises.

## Environment deviations

- conda: the Miniforge installer could not be executed in this session (blocked by the agent's permission
  policy), so the server uses a Python 3.12 `venv`. The README still documents the conda route.
- see-through pins `numpy==2.2.6`; `rembg` 2.0.86 declares `numpy>=2.3` but works with 2.2.6 (used only for
  `remove()`).
- Port 8765 was taken on the server; Toonify defaults to 8780.
