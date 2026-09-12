#!/usr/bin/env python3
"""Smoke test for the /ws realtime endpoint using a bundled sample wav."""
import asyncio
import json
import sys
import wave

import websockets


async def main(wav_path: str, url: str = "ws://127.0.0.1:8000/ws") -> None:
    # This script always targets a local server by default. Explicitly bypass
    # proxy auto-discovery, which otherwise routes localhost through HTTP_PROXY
    # with recent versions of websockets.
    async with websockets.connect(url, max_size=2**24, proxy=None) as ws:
        await ws.send(json.dumps({"mode": "2pass", "language": "auto", "itn": True}))
        wf = wave.open(wav_path, "rb")
        frames = 0
        while True:
            frame = wf.readframes(1600)  # 100 ms @ 16 kHz
            if not frame:
                break
            await ws.send(frame)
            frames += 1
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"is_speaking": False, "is_end": True}))

        finals = 0
        partials = 0
        async for msg in ws:
            r = json.loads(msg)
            if r.get("is_end"):
                print(f"END   text={r['text']!r} timestamp={r.get('timestamp')}")
                break
            if r.get("is_final"):
                finals += 1
                print(f"FINAL text={r['text']!r} timestamp={r.get('timestamp')}")
            else:
                partials += 1
                print(f"PART  text={r['text']!r}")
        print(f"--- frames={frames} partials={partials} finals={finals} ---")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "samples/BAC009S0764W0121.wav"))
