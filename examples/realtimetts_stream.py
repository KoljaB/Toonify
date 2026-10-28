"""Speak text with RealtimeTTS and lip-sync the Toonify avatar live.

    toonify serve        # open the printed player URL in a browser and click "enable audio"
    python examples/realtimetts_stream.py --url ws://<toonify host>:8780/ws/feed "Hello there!"

Default engine: RealtimeTTS's QwenServerEngine talking to a running Qwen TTS server
(`python -m RealtimeTTS.qwen_server ...`, RealtimeTTS >= 0.9.0rc4). `--engine kokoro` uses a local
Kokoro model instead (pip install "realtimetts[kokoro]").

Local playback is muted; every synthesized chunk goes out through `on_audio_chunk` and is played
by the browser player, which drives the mouth from what it plays.
"""
import argparse

import pyaudio
from RealtimeTTS import TextToAudioStream

from toonify.stream_client import StreamClient

PA_FORMATS = {pyaudio.paInt16: "int16", pyaudio.paFloat32: "float32"}


def make_engine(a):
    if a.engine == "kokoro":
        from RealtimeTTS import KokoroEngine
        return KokoroEngine(voice=a.voice or "af_heart")
    from RealtimeTTS import QwenServerEngine
    return QwenServerEngine(base_url=a.tts_url, voice=a.voice or "alex", language=a.language)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="+")
    p.add_argument("--url", default="ws://127.0.0.1:8780/ws/feed", help="Toonify audio feed")
    p.add_argument("--engine", default="qwen-server", choices=["qwen-server", "kokoro"])
    p.add_argument("--tts-url", default="http://127.0.0.1:18085", help="Qwen TTS server")
    p.add_argument("--voice", default=None)
    p.add_argument("--language", default="english")
    a = p.parse_args()

    engine = make_engine(a)
    pa_format, channels, rate = engine.get_stream_info()
    client = StreamClient(a.url).connect()
    client.start(sample_rate=rate, fmt=PA_FORMATS[pa_format], channels=channels)

    # rule-based sentence splitting: no NLTK data download needed
    stream = TextToAudioStream(engine, muted=True, tokenizer="rule-based")
    stream.feed(" ".join(a.text))
    stream.play(on_audio_chunk=client.send, muted=True)   # chunks are forwarded as they are synthesized
    client.end()
    client.close()
    engine.shutdown()


if __name__ == "__main__":
    main()
