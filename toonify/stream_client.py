"""Forward raw PCM audio chunks to the Toonify player (via the `toonify serve` relay).

    from toonify.stream_client import StreamClient
    with StreamClient("ws://host:8780/ws/feed") as c:
        c.start(sample_rate=24000, fmt="int16")
        c.send(chunk_bytes)      # any size, any number of times
        c.end()

WAV files go through exactly the same path:  python -m toonify.stream_client speech.wav --url ws://host:8780/ws/feed
"""
from __future__ import annotations

import argparse
import json
import time
import wave

import numpy as np
from websockets.sync.client import connect

FORMATS = ("int16", "float32")


class StreamClient:
    def __init__(self, url: str = "ws://127.0.0.1:8780/ws/feed"):
        self.url = url
        self.ws = None
        self.fmt = "int16"

    def connect(self) -> "StreamClient":
        if self.ws is None:
            self._cm = connect(self.url, max_size=None)
            self.ws = self._cm.__enter__()
        return self

    def _ws(self):
        return self.connect().ws

    def start(self, sample_rate: int, fmt: str = "int16", channels: int = 1) -> None:
        """Handshake for the next utterance: sample rate and sample format of the following chunks."""
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}")
        if channels != 1:
            raise ValueError("the player takes mono audio; downmix first")
        self.fmt = fmt
        self._ws().send(json.dumps({"type": "start", "sample_rate": int(sample_rate), "format": fmt,
                                        "channels": 1}))

    def send(self, chunk) -> None:
        """Raw little-endian PCM bytes, or a numpy array (converted to the handshake format)."""
        if isinstance(chunk, np.ndarray):
            dtype = np.int16 if self.fmt == "int16" else np.float32
            if self.fmt == "int16" and chunk.dtype.kind == "f":
                chunk = np.clip(chunk, -1, 1) * 32767
            chunk = chunk.astype(dtype, copy=False).astype(np.dtype(dtype).newbyteorder("<")).tobytes()
        if chunk:
            self._ws().send(bytes(chunk))

    def end(self) -> None:
        self._ws().send(json.dumps({"type": "end"}))

    def close(self) -> None:
        if self.ws is not None:
            self._cm.__exit__(None, None, None)
            self.ws = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()


def read_wav_mono(path: str) -> tuple[np.ndarray, int, str]:
    """WAV -> (mono samples, sample rate, protocol format)."""
    with wave.open(path, "rb") as w:
        sr, ch, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(frames, "<i2").reshape(-1, ch)
        return (x.mean(axis=1).astype(np.int16) if ch > 1 else x[:, 0]), sr, "int16"
    if width == 4:   # assume 32-bit int PCM (the wave module does not read float WAVs)
        x = np.frombuffer(frames, "<i4").reshape(-1, ch).astype(np.float32) / 2**31
    elif width == 3:
        b = np.frombuffer(frames, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) << 8 | b[:, 1].astype(np.int32) << 16 | b[:, 2].astype(np.int32) << 24) >> 8)
        x = (x.astype(np.float32) / 2**23).reshape(-1, ch)
    else:
        x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128).reshape(-1, ch) / 128
    return x.mean(axis=1).astype(np.float32), sr, "float32"


def stream_wav(path: str, url: str, chunk_ms: float = 40, realtime: bool = True) -> None:
    """Chunk a WAV and send it like a live source (paced at real time unless realtime=False)."""
    samples, sr, fmt = read_wav_mono(path)
    step = max(1, int(sr * chunk_ms / 1000))
    with StreamClient(url) as c:
        c.start(sr, fmt)
        t0 = time.monotonic()
        for i in range(0, len(samples), step):
            c.send(samples[i:i + step])
            if realtime:   # stay ~200 ms ahead of playback, like a TTS engine would
                ahead = (i + step) / sr - (time.monotonic() - t0)
                if ahead > 0.2:
                    time.sleep(ahead - 0.2)
        c.end()


def main(argv=None):
    p = argparse.ArgumentParser(description="Stream a WAV file to the Toonify player")
    p.add_argument("wav")
    p.add_argument("--url", default="ws://127.0.0.1:8780/ws/feed")
    p.add_argument("--chunk-ms", type=float, default=40)
    p.add_argument("--burst", action="store_true", help="send everything at once instead of real-time pacing")
    a = p.parse_args(argv)
    stream_wav(a.wav, a.url, a.chunk_ms, realtime=not a.burst)


if __name__ == "__main__":
    main()
