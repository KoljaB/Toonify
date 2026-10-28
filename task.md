# Task: "Toonify" — photo → personal animated anime avatar

Build a working end-to-end pipeline: a photo of a real person goes in, and a rigged anime version of that person comes out. It plays in the browser with idle motion and lip-syncs **live from streamed audio chunks**. Keep it simple: the GPU is used offline (once per avatar) only, and the runtime is a lightweight browser player with no ML and no GPU compute beyond normal 2D rendering.

## Context (read these first)
- Layer decomposition: https://github.com/shitagaki-lab/see-through
  (`inference/scripts/inference_psd.py` → layered PSD with up to 23 semantic layers + depth; low-VRAM flags: `--group_offload`, `inference_psd_quantized.py`, `inference_psd_blockswap.py`)
- Animation tools that already consume see-through PSDs (evaluate before writing any rigging code):
  - https://github.com/852wa/Anime2.5DRig (browser, blink/lip-sync/hair physics, uses see-through layer naming)
  - https://github.com/MangoLion/stretchystudio (browser 2D puppet auto-rig)
  - https://github.com/1mm-module/PNGAL (eye/mouth variant generation + animation export)
  - https://github.com/kazuya-bros/PachiPakuGen (blink + mouth-shape generation from the PSD)

## Environment
- Offline pipeline: Linux, NVIDIA GPU, at least 12 GB VRAM (detect the actual GPU and pick see-through's VRAM mode to match), Python 3.12, conda. Local by default; API backends are opt-in only.
- Runtime player: any modern browser. It must run smoothly on an integrated GPU.

## Architecture
A Python package plus CLI. Each offline stage can run on its own, reads and writes files in `runs/<id>/`, and is cached so you can rerun from any stage.

`toonify run photo.jpg --style <preset>` runs stages 1–3 and prints the player URL.

### Offline (GPU, once per avatar)
1. **stylize**: photo → anime character image
   - Use a pluggable backend interface. Implement one local backend (diffusers or ComfyUI, using an identity-preserving image-edit model) and leave a stub for an API backend.
   - The output must suit see-through: a single character, front-facing or slight 3/4 view, upper body or more, neutral expression, mouth closed, eyes open, about 1280 px on the long side.
   - Remove the background afterwards (rembg/BiRefNet or similar) and save as a transparent PNG.
   - Make style presets configurable (prompt plus optional LoRA).
2. **decompose**: run see-through on the PNG → PSD.
   - Add see-through as a git submodule; don't fork its code.
   - Write `layers.json` with the layer name, tag, bbox and draw order for each layer.
3. **expressions**: generate exactly 3 extra frames via inpainting on the relevant layers, keeping the style consistent: **eyes closed**, **mouth half open**, **mouth open**. Reuse the approach or code from PachiPakuGen or PNGAL where the license permits.

### Runtime (browser, no ML, no GPU compute)
4. **player**: loads the PSD layers, the 3 expression frames and `layers.json`.
   - Rendering: Canvas2D or WebGL, layered puppet.
   - Idle loop: blink at random intervals, subtle breathing, head sway, simple hair secondary motion.
   - Base this on whichever of Anime2.5DRig or StretchyStudio fits best after you inspect their code. Integrate or vendor it; don't reimplement deformation from scratch. Strip anything not needed.
5. **live lip-sync from streamed audio chunks** (the core feature):
   - The player receives raw PCM chunks over WebSocket (mono, int16 or float32, sample rate in a handshake message, arbitrary chunk sizes) and plays them gaplessly via the Web Audio API (queue AudioBufferSourceNodes back-to-back on the AudioContext clock).
   - Route playback through one AnalyserNode before the destination. On each requestAnimationFrame, compute RMS from getFloatTimeDomainData and map it to 3 mouth states (closed / half / open) with simple thresholds, hysteresis and attack/release smoothing. Because the analysis runs on the audio actually playing, no extra sync logic is needed (expected mouth lag about 20–30 ms).
   - Close the mouth when the queue runs empty or on an `end` message.
   - No MFCC, no viseme classification, no models at runtime.
6. **stream client**: Python helper `toonify.stream_client` that forwards audio chunks to the player.
   - Include a working example that feeds RealtimeTTS's `on_audio_chunk` callback into it.
   - WAV input is chunked and sent through the same path; there is no separate file mode.

## Working rules
- Phase 0, before any code: inspect the four animation repos and see-through's PSD layer naming. Write `docs/decisions.md` covering which tool you chose as the base, why, its license, its input conventions, and what you will strip or add. Then proceed without waiting for me.
- Validate each stage on a real test image before building the next one. Commit the sample outputs, or screenshots of them, to `docs/samples/`.
- No stubs pretending to work and no faked outputs. If a stage fails or its quality is poor, say so in `docs/status.md` with the concrete error or a sample image.
- Check the license of every model and repo you pull in and list them in `docs/licenses.md`.
- Prefer the simplest thing that works. No extra features beyond this spec.

## Out of scope
Full-body animation, turning or walking, video export, multi-scene generation, training models, viseme or phoneme lip-sync.

## Done when
- `toonify run my_photo.jpg --style default` produces the PSD, the 3 expression frames and a player URL.
- The player shows an idle-animated character that is recognizably the person.
- Streaming RealtimeTTS output through `toonify.stream_client` makes the avatar lip-sync live while it speaks.
- The player runs smoothly on an integrated GPU.
- `README.md` documents install, the CLI, the VRAM modes and the WebSocket protocol (handshake, PCM chunk, `end`).