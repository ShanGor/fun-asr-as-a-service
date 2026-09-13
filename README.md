# FunASR Core — local, offline-capable ASR inference service

Self-contained speech-to-text service for a single NVIDIA GPU or Apple
Silicon host. The original deployment is tested on an RTX 5080 / 16 GB
(Blackwell sm_120, torch 2.11 + cu128); Apple Silicon uses PyTorch MPS.

- **Default ASR**: SenseVoiceSmall (234M parameters; multilingual speech, emotion and event tags stripped)
- **Optional larger ASR**: Fun-ASR-Nano-2512 (800M parameters; Chinese, English, Japanese and Chinese dialects)
- **VAD**: FSMN-VAD (streaming and offline segmentation)
- **Diarization**: CAM++ speaker embeddings + clustering — `speaker` label per segment
- **HTTP API**: OpenAI-compatible `POST /v1/audio/transcriptions`, `GET /health`, `GET /v1/models`
- **Realtime WebSocket**: `WS /ws`, PCM16 mono 16 kHz in, partial + final results out
- **Test UI**: self-contained web page served at `GET /` (mic realtime + upload/diarization)
- **Offline deployment**: models bundled in `models/`, platform-specific wheels bundled in `wheels/` or `wheels-macos/` — the server performs **no network access at runtime**

The app binds to `127.0.0.1` and has **no authentication**. Put a gateway
(nginx/Caddy) in front of it for TLS, auth, upload and rate limits — see
`deploy/nginx.conf.example`.

## Layout

```
fun-asr/
├── funasr_core/            # the core app
│   ├── config.py           # env-based configuration
│   ├── model_manager.py    # model loading + VAD/ASR inference + diarization
│   ├── chunking.py          # bounded, pause-aware ASR windows
│   ├── audio.py            # decoding (soundfile + ffmpeg fallback)
│   ├── streaming.py        # per-connection realtime session (VAD + ASR)
│   └── server.py           # FastAPI app: HTTP + WebSocket + test UI
├── frontend/
│   └── index.html          # self-contained test UI (realtime mic, upload, diarization)
├── scripts/
│   ├── download_models.py  # run once with network: fills ./models
│   ├── download_wheels.sh  # run once with network: fills platform wheels
│   ├── install_macos.sh    # networked Apple Silicon install + model download
│   ├── install_offline.sh  # creates .venv from platform wheels
│   ├── run_macos.sh        # starts the server with MPS on Apple Silicon
│   └── run_server.sh       # starts the server (sets offline env)
├── deploy/
│   ├── funasr-core.service # systemd unit (Restart=always)
│   └── nginx.conf.example  # sample gateway config
├── models/                 # bundled model weights and optional Fun-ASR-Nano assets
├── wheels/                 # bundled CUDA/Linux pip wheels
├── wheels-macos/           # optional bundled Apple Silicon pip wheels
├── requirements-common.txt # dependencies shared by both platforms
├── requirements-macos.txt  # Apple Silicon/MPS dependencies
└── requirements-lock.txt   # exact CUDA/Linux environment
```

## Quick start (on this machine)

```bash
./scripts/run_server.sh                      # binds 127.0.0.1:8000

# web test UI (realtime mic, upload, diarization)
# open http://127.0.0.1:8000/ in a browser

# health
curl http://127.0.0.1:8000/health

# transcribe a file (OpenAI-compatible)
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@samples/BAC009S0764W0121.wav -F model=sensevoice -F response_format=verbose_json

# with speaker diarization (adds "speaker" per segment and "speakers" count)
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@samples/BAC009S0764W0121.wav -F model=sensevoice \
  -F response_format=verbose_json -F diarize=true

# or with the OpenAI SDK
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='none')
print(c.audio.transcriptions.create(model='sensevoice', file=open('samples/BAC009S0764W0121.wav','rb')).text)"
```

For a remote realtime browser test, start the server with the generated
self-signed certificate in a separate run:

```bash
./scripts/run_server.sh --https --host 0.0.0.0 --port 8443
# open https://<server-ip>:8443/ in a browser
```

The default server uses SenseVoiceSmall. To compare the larger public
Fun-ASR-Nano checkpoint, start a second instance on another port:

```bash
FUNASR_PORT=8001 FUNASR_ASR_MODEL=Fun-ASR-Nano-2512 ./scripts/run_server.sh
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@samples/BAC009S0764W0121.wav -F model=fun-asr-nano \
  -F response_format=verbose_json
```

Run only one instance when GPU memory is limited. Each process loads its own
VAD, speaker and ASR model instances.

## macOS on Apple Silicon (M1–M4)

The supported Mac path uses PyTorch’s `mps` device, which runs the models on
the Apple GPU through Metal. It does not use the Apple Neural Engine (ANE):
PyTorch MPS and Core ML/Core AI are separate runtimes, and this Python
pipeline has not been exported to Core ML/Core AI.

On the Mac, with network access for the initial setup:

```bash
brew install python@3.12 ffmpeg       # ffmpeg is needed for mp3/aac/m4a
scripts/install_macos.sh               # creates .venv and downloads models
scripts/run_macos.sh                   # starts on MPS at 127.0.0.1:8000
curl http://127.0.0.1:8000/health
```

Use `scripts/install_macos.sh --skip-models` when the `models/` directory is
already bundled. Set `FUNASR_DEVICE=cpu` if MPS is unavailable or if a model
operation needs CPU-only execution. The normal `scripts/run_server.sh` also
auto-selects `mps` on arm64 macOS and keeps `cuda` as the Linux default.

For a fully offline Mac deployment, prepare the bundle on an Apple Silicon
Mac after the networked install:

```bash
scripts/download_wheels.sh macos  # writes wheels-macos/ and requirements-macos-lock.txt
```

Copy the project, `models/`, `wheels-macos/`, and
`requirements-macos-lock.txt` to the offline Mac, then run:

```bash
scripts/install_offline.sh macos
scripts/run_macos.sh
```

The CUDA and macOS wheel sets are intentionally kept separate because CUDA
wheels cannot be installed on Apple Silicon.

## Realtime WebSocket protocol (`/ws`)

Audio: raw **PCM16, mono, 16 kHz** binary frames. Control messages are JSON text.

```python
import asyncio, json, websockets, wave

async def stream():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        # 1) optional config (mode/language/itn/partial cadence)
        await ws.send(json.dumps({"mode": "2pass", "language": "auto", "itn": True}))
        # 2) stream audio frames (~100 ms each)
        wf = wave.open("samples/BAC009S0764W0121.wav", "rb")
        while True:
            frame = wf.readframes(1600)
            if not frame: break
            await ws.send(frame); await asyncio.sleep(0.05)
        # 3) end of input
        await ws.send(json.dumps({"is_speaking": False, "is_end": True}))
        async for msg in ws:
            r = json.loads(msg)
            print(("FINAL " if r["is_final"] else "PARTIAL "), r["text"])
            if r.get("is_end"): break

asyncio.run(stream())
```

Server → client messages (FunASR-2pass compatible shape):

```json
{"mode":"2pass","wav_name":"microphone","is_final":false,"text":"...","timestamp":[[s,e]]}
{"mode":"2pass","wav_name":"microphone","is_final":true,"is_end":false,"text":"...","timestamp":[[s,e]]}
{"mode":"2pass","wav_name":"microphone","is_final":true,"is_end":true,"text":"...","timestamp":[[s,e],...]}
```

Treat `is_final:false` messages as a replaceable preview; append only
`is_final:true` segment messages.

## Speaker diarization

`POST /v1/audio/transcriptions` accepts two extra form fields:

- `diarize` (`true`/`false`, default `false`) — run CAM++ speaker embedding
  extraction on independent voice windows and cluster them (cosine distance,
  threshold 0.45).
- `max_speakers` (int, optional) — force exactly that many clusters instead of
  automatic count detection (when enough embeddings are available). Use `1`
  for a known solo singer or speaker: all spoken segments get `speaker: 0`,
  and embedding inference is skipped. The UI calls this “Known speaker count”.

Automatic analysis uses six-second windows every three seconds, requiring 70%
VAD voice coverage. This gives speaker embeddings consistent context independent
of ASR phrase cuts, and excludes windows dominated by pauses or trailing audio.
Segments receive the speaker with the most overlapping window support; segments
without support inherit the nearest analyzed window. No speaker-count input is
required. The optional known-count override remains available.

This is segment-level diarization: short speaker turns can be smoothed over,
and a speaker change inside one transcript segment cannot receive two labels.
VAD can still mistake music for voice. Missing usable embeddings fall back to
one speaker.

`verbose_json` responses then carry `speakers: N` and a `speaker: i` (0-based)
on every segment. Diarization is only meaningful on files with ≥2 segments;
single-utterance files always return `speakers: 1`.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `FUNASR_HOST` / `FUNASR_PORT` | `127.0.0.1` / `8000` | bind address |
| `FUNASR_SSL_CERTFILE` | empty | PEM certificate; enables HTTPS when paired with the key |
| `FUNASR_SSL_KEYFILE` | empty | PEM private key; must be paired with the certificate |
| `FUNASR_SSL_KEYFILE_PASSWORD` | empty | optional password for the private key |
| `FUNASR_DEVICE` | `cuda` on Linux / `mps` on arm64 macOS | `cuda`, `mps` (Apple Silicon), or `cpu` |
| `FUNASR_ASR_MODEL` | `models/SenseVoiceSmall` | local model dir, or `Fun-ASR-Nano-2512` |
| `FUNASR_ASR_HUB` | `ms` | model hub for custom ASR checkpoints |
| `FUNASR_VAD_MODEL` | `models/fsmn-vad` | local model dir |
| `FUNASR_SPK_MODEL` | `models/campplus` | speaker embedding model (diarization) |
| `FUNASR_WORKER_THREADS` | `max(8, cpus)` | inference thread pool size |
| `FUNASR_CONCURRENT_VAD` | `4` | max concurrent VAD calls |
| `FUNASR_CONCURRENT_ASR` | `1` | admission limit for upload jobs and realtime ASR; shared model inference is serialized |
| `FUNASR_MAX_UPLOAD_BYTES` | `209715200` | upload cap (200 MB) |
| `FUNASR_MAX_WS_CONNECTIONS` | `100` | realtime connection cap |
| `FUNASR_CHUNK_INTERVAL_MS` | `100` | VAD feeding granularity |
| `FUNASR_VAD_SPLIT_SILENCE_MS` | `300` | minimum upload VAD gap for a phrase boundary; also controls VAD endpoint silence (minimum 200 ms) |
| `FUNASR_ASR_TARGET_MS` | `15000` | fallback cut target only when no pause exists before the hard maximum |
| `FUNASR_VAD_MAX_SEGMENT_MS` | `20000` | hard maximum ASR chunk duration for uploads and realtime final decoding |
| `FUNASR_ASR_OVERLAP_MS` | `500` | context overlap at forced upload boundaries |
| `FUNASR_PARTIAL_INTERVAL_MS` | `500` | partial refresh cadence |
| `FUNASR_PARTIAL_WINDOW_SEC` | `8` | max audio length per partial decode |
| `FUNASR_ITN` | `true` | inverse text normalization |
| `FUNASR_LANGUAGE` | `auto` | default language hint |

Uploads require **ffmpeg**. Compressed input and decoded 16 kHz float32 PCM
are held in temporary files, removed after success or failure. Allow about
230 MB of temporary disk space per hour of decoded audio, plus the upload.
VAD runs in independent 30-second windows. Uploads split at the first detected
pause of at least 300 ms (configurable), with a minimum chunk duration of one
second. Speech without a qualifying pause stays together up to the hard maximum
of 20 seconds; only then does the planner use the fallback cut target.
Every sample, including leading/trailing audio and missed VAD regions, reaches
ASR. Forced boundaries use a nearby low-energy point and context overlap.
Overlap reconciliation removes only exact multi-token/character boundary matches;
uncertain matches are preserved, so duplicate boundary text is still possible.
Verbose JSON includes every planned chunk with `status: ok` or `status: empty`;
timestamps describe decode windows and can overlap. Silence may produce empty
results or model hallucinations; coverage does not guarantee recognition accuracy.
Fun-ASR-Nano uses its custom inference implementation and Qwen3-0.6B language
component; the service passes in-memory audio tensors for that model and keeps
the same VAD chunking and diarization pipeline around it. The returned segment
timestamps are this service's chunk windows, not word-level timestamps.

Concurrency model: a **single process owns the selected accelerator**; blocking `generate()`
calls run on a thread pool, gated by async semaphores (`CONCURRENT_ASR` is the
main GPU knob). HTTP and WS share the same semaphores, so total GPU work stays
bounded regardless of client count. Never run multiple workers of this app.

## Measured on RTX 5080 (16 GB), SenseVoiceSmall on CUDA

(4.2 s sample utterance, `scripts/concurrency_test.py`)

| Workload | Wall time | Per-request | Notes |
|---|---|---|---|
| 1 file (HTTP) | — | 0.07 s | ~60x realtime |
| 8 files in parallel | 0.26 s | avg 0.17 s / max 0.25 s | |
| 32 files in parallel | 0.90 s | avg 0.55 s / max 0.89 s | semaphore queueing |
| 8 concurrent WS streams | 2.45 s | 2.4 s/conn | simulated realtime, partials every 0.5 s |
| 32 concurrent WS streams | 7.16 s | 6.8 s/conn | all succeeded, no drops |

GPU peaked at ~39% / 1.6 GB VRAM during 8 HTTP + 8 WS concurrent load —
significant headroom remains for more streams and longer audio.

## Packaging for an offline server

### Compare Fun-ASR-Nano with SenseVoiceSmall

The public Fun-ASR-Nano checkpoint is installed under
`models/Fun-ASR-Nano-2512` together with its Qwen3-0.6B component. Restart the
server with:

```bash
FUNASR_ASR_MODEL=Fun-ASR-Nano-2512 ./scripts/run_server.sh
```

To reproduce the model installation on another networked machine, run
`.venv/bin/python scripts/download_models.py --models Fun-ASR-Nano-2512 Qwen3-0.6B`.

Switch back by omitting that variable. Fun-ASR-Nano is 800M parameters versus
SenseVoiceSmall's 234M. It requires the local custom model code and Qwen3-0.6B
component, while VAD chunking, segment timestamps and speaker diarization are
provided by this service. The flagship Fun-ASR checkpoint is 7.7B but is not an
available public checkpoint; Nano is the largest public Fun-ASR checkpoint used
by this project.

On a machine **with** network access (do once):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
.venv/bin/python scripts/download_models.py     # baseline models (~1 GB)
# Optional comparison checkpoint and Qwen component (~3.7 GB extra):
.venv/bin/python scripts/download_models.py --models Fun-ASR-Nano-2512 Qwen3-0.6B
scripts/download_wheels.sh                      # fills ./wheels, writes requirements-lock.txt
```

Copy (or rsync/tar) the whole folder to the server, then **on the server**:

```bash
scripts/install_offline.sh        # creates .venv from wheels/, no network
scripts/run_server.sh             # or install deploy/funasr-core.service
sudo cp deploy/funasr-core.service /etc/systemd/system/ && sudo systemctl enable --now funasr-core
```

Server prerequisites: NVIDIA driver ≥ 570 (CUDA 12.8 capable), `python3.12`,
and `ffmpeg` on PATH if you need mp3/aac uploads (wav/flac/ogg work without it).

For Apple Silicon prerequisites and installation, see [macOS on Apple Silicon
(M1–M4)](#macos-on-apple-silicon-m1m4). The Mac install uses the regular PyPI
PyTorch wheels rather than the CUDA index.

## Verification checklist

1. `curl http://127.0.0.1:8000/health` → `{"status":"ok",...}`
2. Open `http://127.0.0.1:8000/` in a browser, smoke-test mic + upload tabs
3. Transcribe `samples/BAC009S0764W0121.wav`, inspect text quality manually
4. Run several transcriptions in parallel; watch `nvidia-smi` and latency
5. WS smoke: stream the same wav as PCM16, confirm partial then final messages
6. From an untrusted host, confirm `:8000` is unreachable (gateway-only access)

---
## How to start it with HTTPs:
```
Start with the generated certificate:

./scripts/run_server.sh \
--https \
--host 0.0.0.0 \
--port 8443

This uses:

deploy/tls/fullchain.pem
deploy/tls/privkey.pem

Open remotely:

https://192.168.163.189:8443/

Available options:

--https
--host HOST
--port PORT
--ssl-certfile FILE
--ssl-keyfile FILE

Explicit certificate paths:

./scripts/run_server.sh \
--host 0.0.0.0 \
--port 8443 \
--ssl-certfile deploy/tls/fullchain.pem \
--ssl-keyfile deploy/tls/privkey.pem

Environment-variable equivalent:

FUNASR_HOST=0.0.0.0 \
FUNASR_PORT=8443 \
FUNASR_SSL_CERTFILE=deploy/tls/fullchain.pem \
FUNASR_SSL_KEYFILE=deploy/tls/privkey.pem \
./scripts/run_server.sh

Because the certificate is self-signed, accept the browser warning. You may test health with:

curl -k https://192.168.163.189:8443/health
```