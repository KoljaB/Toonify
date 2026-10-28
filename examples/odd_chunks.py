"""Protocol check: send a WAV as float32 in random, mostly misaligned byte-sized chunks.

    python examples/odd_chunks.py speech.wav --url ws://host:8780/ws/feed

The player must carry partial samples over between chunks; audible clicks or noise mean it does not.
"""
import argparse
import random
import time

import numpy as np

from toonify.stream_client import StreamClient, read_wav_mono


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--url", default="ws://127.0.0.1:8780/ws/feed")
    a = p.parse_args()
    samples, sr, fmt = read_wav_mono(a.wav)
    data = (samples.astype(np.float32) / 32768 if fmt == "int16" else samples).astype("<f4").tobytes()
    rng = random.Random(1)
    with StreamClient(a.url) as c:
        c.start(sr, "float32")
        i, t0 = 0, time.monotonic()
        while i < len(data):
            n = rng.randint(1, 4000)          # arbitrary sizes, usually not a multiple of 4
            c.send(data[i:i + n])
            i += n
            ahead = i / 4 / sr - (time.monotonic() - t0)
            if ahead > 0.2:
                time.sleep(ahead - 0.2)
        c.end()


if __name__ == "__main__":
    main()
