#!/usr/bin/env python3
"""Concurrency smoke test: N parallel HTTP transcriptions + M concurrent WS streams."""
import asyncio
import json
import sys
import time
import wave

import httpx
import websockets

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000/ws"
WAV = "samples/BAC009S0764W0121.wav"


async def http_one(client: httpx.AsyncClient, idx: int) -> float:
    t0 = time.perf_counter()
    with open(WAV, "rb") as f:
        r = await client.post(
            f"{BASE}/v1/audio/transcriptions",
            files={"file": ("a.wav", f, "audio/wav")},
            data={"model": "sensevoice"},
        )
    r.raise_for_status()
    text = r.json()["text"]
    dt = time.perf_counter() - t0
    assert text, f"http {idx}: empty text"
    return dt


async def ws_one(idx: int) -> float:
    t0 = time.perf_counter()
    async with websockets.connect(WS, max_size=2**24) as ws:
        await ws.send(json.dumps({"mode": "2pass", "language": "auto", "itn": True}))
        wf = wave.open(WAV, "rb")
        while True:
            frame = wf.readframes(1600)
            if not frame:
                break
            await ws.send(frame)
            await asyncio.sleep(0.05)  # simulate realtime
        await ws.send(json.dumps({"is_speaking": False, "is_end": True}))
        got_final = False
        async for msg in ws:
            r = json.loads(msg)
            if r.get("is_end"):
                assert r["text"], f"ws {idx}: empty final text"
                break
            if r.get("is_final"):
                got_final = True
        assert got_final, f"ws {idx}: no final segment"
    return time.perf_counter() - t0


async def main() -> None:
    n_http = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_ws = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.perf_counter()
        times = await asyncio.gather(*(http_one(client, i) for i in range(n_http)))
        wall = time.perf_counter() - t0
        print(f"HTTP  x{n_http}: wall={wall:.2f}s per-req avg={sum(times)/len(times):.2f}s max={max(times):.2f}s")

    t0 = time.perf_counter()
    times = await asyncio.gather(*(ws_one(i) for i in range(n_ws)))
    wall = time.perf_counter() - t0
    print(f"WS    x{n_ws}: wall={wall:.2f}s per-conn avg={sum(times)/len(times):.2f}s max={max(times):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
