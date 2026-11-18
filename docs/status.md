# Status

Validated on 2026-10-28. Pipeline host: Ubuntu, RTX 4090 (24 GB), driver 595, torch 2.8 cu128. Player tested in
a Chromium-based browser on Windows. Sample images are in `docs/samples/`.

## Stage by stage

| stage | state | evidence |
|---|---|---|
| 1 stylize (Qwen-Image 2.1 edit in ComfyUI + BiRefNet cutout) | works | `samples/*_stylized.png`; photo → recognizable anime (glasses, hair, coat kept; the photo-based samples are not published) |
| upper-body framing of full-body inputs | works | `samples/*_framed.png`; the face is drawn ~2.5× larger |
| 2 decompose (see-through, unmodified submodule) | works, with repair | 15–18 layers per avatar; see known problems |
| 3 expressions (masked Qwen inpaint on a face crop) | works | `samples/*_expressions.png`: style-consistent closed eyes / half / open mouth |
| 3b emotions (frame set per emotion, chained inpaints) | works | happy, sad, angry and surprised frame sets generated for the female sample (`runs/input_anime_female-faithful-17ac0f11/emotions/`, not committed); player crossfade, gate and lip-sync per set first tested with tinted stand-in frames |
| 4 player (Anime2.5DRig rig, WebGL1) | works | 238–240 fps measured (display-capped) on the test PC |
| 5 live lip-sync (WebSocket PCM → Web Audio → RMS states) | works | see measurements below |
| 6 stream client + RealtimeTTS example | works | Qwen TTS server → `on_audio_chunk` → player |

## Measured

**Offline time per avatar (RTX 4090, everything on CUDA):**

| step | time |
|---|---|
| framing + cutout (BiRefNet, onnxruntime-gpu) | ~0.3 s inference each (+ a few seconds model load) |
| Qwen-Image 2.1 stylize edit (25 steps) | 11–29 s |
| see-through (LayerDiff 2×30 steps + Marigold, bf16 `full` mode) | ~150 s |
| repair + three expression inpaints (16 steps on a 768 px face crop) | ~28 s |
| first run only: see-through weight download | ~10 GB |

Before the speed work the expressions took ~90 s (full 1280² canvas, 25 steps) and the cutout ran on the CPU.
Qwen-Image 2.1 + its text encoder (13.6 + 16.7 GB) do not fit in 24 GB together, so ComfyUI streams weights and
model loading is part of every job. Stages unload each other's models in between.

**Lip-sync** (logged per animation frame in the player):
- RealtimeTTS → QwenServerEngine → `stream_client` → relay → player: 6 s of speech gave 35 mouth-state changes
  (27 % closed / 22 % half / 51 % open) with the first normalization settings. Thresholds were then raised
  slightly.
- WAV via `python -m toonify.stream_client` (int16, 37 ms chunks) and `examples/odd_chunks.py` (float32,
  random 1–4000-byte chunks, so most chunks split a sample): 12 s of audio played without gaps, with 74 state
  changes. The mouth was closed after the last sample.
- Mouth timing follows the analyser on the playing audio, so the lag is the analyser window (1024 samples ≈ 21 ms
  at 48 kHz) plus the 12 ms attack.

**Runtime cost:** the rig is 15–20 textured meshes deformed on the CPU in JS, drawn with one WebGL1 shader. On the
test PC it ran at the display cap (240 fps). **Not yet measured on an integrated GPU.** Per frame it uploads a few
thousand vertices and draws at most 1280×1280 with ~20 alpha-blended quads, which an integrated GPU should handle
easily, but this is unverified.

## Known problems and limits

- **see-through drops or misplaces parts.**
  - Headphones around the neck had no layer at all.
  - A crop top was "completed" downwards over the trousers.
  - In one 20-step test the whole `face` layer was missing, which turned glasses into dark lenses. That is why 30
    steps stays the default.

  The decompose repair step fixes the first two kinds of problem by comparing the composite with the source.
  Re-added regions are flat cut-outs: they move rigidly with the body or head.
- **Draw order comes from see-through's depth sort.** Sometimes `front hair` sits below the eyes; this looks fine at
  rest but can show during strong head turns.
- **Motion is 2.5D parallax, not artist rigging.** Head turns are a shear/offset per layer depth. The see-through
  trailer shows hand-made Live2D rigs, and this player does not reach that level. Hair now swings mainly from head
  inertia, short hair is nearly static, and head motion is spring-eased.
- **The `default` style flattens illustrations.** For inputs that are already drawings, use `--style faithful`.
- **Qwen-Image 2.1 is research-licensed**, see `licenses.md`.
- **conda** was not installed on the test host: running the downloaded Miniforge installer was blocked in the
  agent session. A Python 3.12 venv was used instead.
- `rembg` declares `numpy>=2.3`; see-through pins `numpy==2.2.6`. It works, but pip prints a conflict warning.
- **RealtimeTTS:** `QwenServerEngine` needs RealtimeTTS ≥ 0.9.0rc4, which is not on PyPI yet (0.8.10 is). The demo
  venv uses the local 0.9.0rc12 wheel. NLTK refuses to install `punkt_tab` into a group-writable `~/nltk_data`, so
  the example uses `tokenizer="rule-based"`.
- The API stylize backend is a stub that raises `NotImplementedError`.
