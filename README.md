# Toonify

A photo (or an existing anime picture) of a person goes in. A rigged anime avatar of that person comes out. It
idles in the browser (blinks, breathes, sways, hair physics), **lip-syncs live from streamed PCM audio** and
switches between **emotions** (neutral, happy, sad, angry, surprised) on request.

```
photo ──► 1 stylize ──► 2 decompose ──► 3 expressions ──► 3b emotions ──► 4 player (browser) ◄── 6 stream_client ◄── TTS
          Qwen-Image     see-through     Qwen-Image          Qwen-Image       Anime2.5DRig rig        WebSocket PCM
          2.1 edit       PSD + layers    masked inpaint      masked inpaint   + Web Audio lip-sync    + emotion
          (GPU, once)    (GPU, once)     (GPU, once)         (GPU, once)      (no ML, no GPU compute)
```

Stages read and write `runs/<run id>/` and are cached by input hashes, so you can rerun any stage alone.

## Install (GPU machine, Linux)

```bash
git clone --recursive <this repo> toonify && cd toonify     # see-through comes in as a submodule
conda create -n toonify python=3.12 -y && conda activate toonify     # or: python3.12 -m venv .venv
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r third_party/see-through/requirements.txt          # run from the submodule root (it has -e paths)
(cd third_party/see-through && ln -sfn common/assets assets)
pip install -e .
pip install numpy==2.2.6          # see-through's pin; rembg's newer numpy requirement is not needed for remove()
```

**Image-edit backend:** a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with Qwen-Image 2.1
(`diffusion_models/qwen_image_2.1_bf16.safetensors`, `text_encoders/qwen3vl_8b_bf16.safetensors`,
`vae/qwen_image_2.1_vae_bf16.safetensors`). Point Toonify at it:

| variable | default | meaning |
|---|---|---|
| `TOONIFY_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI server |
| `TOONIFY_COMFY_DIR` | – | ComfyUI folder; if set, Toonify starts ComfyUI when nothing listens on the URL |
| `TOONIFY_COMFY_PYTHON` | `<COMFY_DIR>/../.venv/bin/python` | interpreter for that ComfyUI |
| `TOONIFY_QWEN_UNET` / `_CLIP` / `_VAE` / `_STEPS` / `_CFG` | see `toonify/config.py` | model file names, sampler |
| `TOONIFY_PRESETS` | – | extra style presets (JSON, same shape as `toonify/presets.json`) |

see-through downloads its weights (~10 GB) from Hugging Face on the first run. A token in
`~/.cache/huggingface/token` is used automatically.

## CLI

```bash
toonify run photo.jpg --style default        # stages 1–3b, prints the player URL and serves it (port 8780)
toonify run anime.png --style faithful       # keep an existing anime drawing's art style
toonify run photo.jpg --no-serve             # only print the URL (when `toonify serve` already runs)

toonify stylize photo.jpg --style default    # stage 1 -> runs/<id>/stylized.png   (prints the run id)
toonify decompose <run id> [--vram auto|full|group_offload|nf4|blockswap]
toonify expressions <run id> [--only eyes_closed mouth_open]
toonify emotions <run id> [--only happy sad]  # needs the expressions stage; ~4 inpaints per emotion
toonify serve [--port 8780]                  # player + audio relay
```

Add `--force` to any stage to ignore the cache. Player URL: `http://<host>:8780/?run=<run id>`. Click
*enable audio* once (browser autoplay policy).

**Styles** (`toonify/presets.json`, prompt plus optional LoRA name and strength):
- `default`: redraws a photo as clean, flat cel-shaded anime. This decomposes most reliably.
- `faithful`: for inputs that are already illustrations. Keeps the art style, shading and face, and only removes
  background and props.
- `soft`: pastel light-novel look.

Full-body inputs are cropped to head–waist automatically before the edit, so the face is drawn large.

### VRAM modes (stage 2)

`--vram auto` (default) reads the GPU's total memory and picks:

| mode | when | see-through command | peak VRAM @1280 |
|---|---|---|---|
| `full` | ≥ 16 GB | `inference_psd.py` (bf16) | ~12–16 GB |
| `group_offload` | 11–16 GB | `inference_psd.py --group_offload` | ~10 GB |
| `nf4` | < 11 GB | `inference_psd_quantized.py` (needs `requirements-inference-bnb.txt`) | ~8 GB |
| `blockswap` | manual | `inference_psd_blockswap.py` | ~8 GB |

Before stage 2, Toonify asks ComfyUI to unload its models (`POST /free`), or stops it if Toonify started it, so
both never hold VRAM at the same time.

## Outputs (`runs/<id>/`)

| file | stage |
|---|---|
| `input_framed.png` | upper-body crop of a full-body input (if applied) |
| `stylized_raw.png`, `stylized.png` | 1: anime image, and the same with a transparent background |
| `avatar.psd` | 2: see-through layers, plus the repair step (see below) |
| `layers.json` | 2/3: canvas, source offset, per layer `name`, `tag`, `bbox`, draw `order`; `expressions` |
| `expressions/eyes_closed.png`, `mouth_half.png`, `mouth_open.png` | 3: RGBA patches (bbox in `layers.json`) |
| `expressions/preview_*.png`, `mask_*.png` | 3: full-frame previews and inpainting masks |
| `emotions/<emotion>_{face,eyes_closed,mouth_half,mouth_open}.png` | 3b: one frame set per emotion; `emotions` in `layers.json` |
| `emotions/preview_*.png`, `mask_face.png`, `mask_eyes.png` | 3b: full-frame previews and inpainting masks |

**Emotion sets:** the emotion face is inpainted from the neutral face over eyebrows, eyes and mouth. Its closed-eye
and mouth frames are then inpainted from that emotion face, so the brows and eyes stay the same while the mouth
moves.

**Decompose repair:** see-through completes hidden parts of each layer, and occasionally such a part ends up drawn
over a layer it should be behind (a crop top continued over trousers). Parts outside its 23 classes can also be
dropped (headphones). After see-through, Toonify compares the layer composite with the source. It trims a top
layer where the layer below matches the source, and re-adds large missing regions as `objects` (body) or
`headwear` (head) layers.

## WebSocket protocol

The player connects to `ws://<host>:8780/ws/play`; producers connect to `ws://<host>:8780/ws/feed`. The server
relays every feed message to all players and replays the last handshake and the last emotion to players that
join late.

| message | frame | content |
|---|---|---|
| handshake | text | `{"type":"start","sample_rate":24000,"format":"int16","channels":1}` (`format`: `int16` or `float32`, little-endian, mono only) |
| PCM chunk | binary | raw samples in the handshake format, **any size**; a partial sample at the end is carried over to the next chunk |
| end | text | `{"type":"end"}`: no more audio for this utterance |
| emotion | text | `{"type":"emotion","name":"sad","intensity":0.8}`: any time, also mid-utterance. `name`: `neutral`, `happy`, `sad`, `angry`, `surprised` (plus a few aliases such as `joy`, `upset`, `annoyed`, `shocked`). `intensity` (0..1, default 1) scales only posture and motion |

Send a new `start` whenever the sample rate or format changes. Playback runs in an **AudioWorklet**
(`player/pcm-worklet.js`, modelled on miep's `tts-worklet.js`). The worklet copies samples out of one queue on the
audio thread, so chunk joins are sample-exact. It holds 100 ms before (re)starting and fades over 64 samples on a
real underrun. Scheduling a separate `AudioBufferSourceNode` per chunk was tried first and clicked audibly at the
joins. The `AudioContext` runs at the stream's sample rate. Everything plays through one `AnalyserNode`. Each
animation frame computes RMS over the playing audio, smooths it (attack 12 ms / release 60 ms), normalizes it by
the recent speech peak and maps it to closed / half / open with hysteresis. The picture then glides between the
neighbouring mouth frames. The mouth closes when the worklet reports an empty queue, so it also closes after `end`
once the last sample has played. Thresholds can be overridden in the URL: `gate`, `halfOn`, `halfOff`, `openOn`,
`openOff`, `attack`, `release`, `peakDecay`, `prebuffer`.

**Emotion changes** wait for a calm moment: up to 250 ms for a closed mouth, and no new blink starts meanwhile.
Then the new emotion's frame set fades in over 300 ms on top of the old one. Each set has its own closed-eye and
mouth frames and is drawn as one group (off-screen layer, one opacity), so lip-sync and blinking keep running
during the fade and parts of a set never show through each other. Head pitch and tilt, body sway, idle speed, speech
nods and blink rate follow the emotion. Runs without emotion frames only get the posture change. URL overrides:
`emoFade`, `emoWait` (seconds), `emotion` (start emotion); `toonEmotion('happy')` switches from the browser console.

**Secure context required:** browsers only provide AudioWorklet over HTTPS or on `localhost`. Either open the
player through an SSH tunnel (`ssh -N -L 8780:127.0.0.1:8780 user@gpu-host`, then `http://localhost:8780/?run=…`)
or serve HTTPS: `toonify serve --ssl-cert cert.pem --ssl-key key.pem` (for example with certificates from mkcert, as
miep does).

## Streaming audio

```python
from toonify.stream_client import StreamClient
with StreamClient("ws://host:8780/ws/feed") as c:
    c.start(sample_rate=24000, fmt="int16")
    for chunk in chunks:           # bytes or numpy arrays
        c.send(chunk)
    c.end()
    c.emotion("happy")             # any time; crossfades the face
```

- WAV files take the same path: `python -m toonify.stream_client speech.wav --url ws://host:8780/ws/feed`
  (paced at real time; `--burst` sends everything at once). `--emotion sad` (with or without a WAV) sets the emotion.
- RealtimeTTS: `python examples/realtimetts_stream.py --url ws://host:8780/ws/feed --tts-url http://<qwen tts server>:18085 "Hello!"`.
  This uses `QwenServerEngine` (RealtimeTTS ≥ 0.9.0rc4) with local playback muted and `on_audio_chunk=client.send`.
  `--engine kokoro` runs locally instead.

## Docs

- `docs/decisions.md`: why Anime2.5DRig, what was vendored, stripped and added
- `docs/status.md`: what works, measured numbers, known problems, with samples in `docs/samples/`
- `docs/licenses.md`: every repo and model with its license (note: Qwen-Image 2.1 is research-only)

## License

Toonify's own code is MIT ([LICENSE](LICENSE)). The models and repositories it uses have their own licenses,
listed in `docs/licenses.md`; Qwen-Image 2.1 in particular is research / non-commercial only.
