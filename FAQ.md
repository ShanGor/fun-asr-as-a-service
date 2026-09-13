# FAQ — FunASR Core deployment

Answers to the most common questions about this FunASR service
(`funasr_core/`, models in `models/`).

## Q1. What is the parameter size of the FunASR model we use?

We run **SenseVoiceSmall** for ASR, **FSMN-VAD** for speech segmentation and
**CAM++** for speaker embeddings (diarization).

| Model | Type | Parameter count | Model file size |
|---|---|---|---|
| `models/SenseVoiceSmall` | Non-autoregressive multilingual ASR (zh/yue/en/ja/ko + ~50 langs, with LID / emotion / event / ITN) | **~234 M params** (233,999,167, verified by counting weights in this repo) | ~936 MB (`model.pt`) |
| `models/fsmn-vad` | Voice activity detection (streaming + offline) | **~0.4 M params** (tiny, streaming-friendly) | ~1.7 MB (`model.pt`) |
| `models/campplus` | CAM++ speaker embeddings (diarization) | ~7 M params | ~28 MB (`campplus_cn_common.bin`) |

Notes:
- The 234M figure is a *model parameter* count, not disk size. Disk size is
  larger (~936 MB) because the checkpoint stores fp32 weights plus
  state/tokenizer artifacts.
- Compared to Whisper: SenseVoiceSmall has roughly the parameter size of
  Whisper-Small but is much faster (non-autoregressive decoder).

## Q2. Does it support diarization (speaker separation)?

**Yes — this deployment runs speaker diarization.** The server bundles the
**CAM++** speaker-embedding model (`models/campplus`, ~28 MB, downloaded
offline-ready). For each VAD segment it extracts an embedding and clusters
segments with cosine-distance agglomerative clustering (threshold 0.45, or an
explicit `max_speakers` count).

Usage: `POST /v1/audio/transcriptions -F diarize=true` (optionally
`-F max_speakers=2`). `verbose_json` responses then include `speakers: N`
plus a 0-based `speaker` index per segment. The bundled web UI (`GET /`)
exposes the same toggle.

Notes:
- Only files with more than one VAD segment can yield more than one speaker.
- The threshold-based automatic clustering works best on clean speech with a
  handful of speakers; heavily overlapped speech degrades accuracy (classic
  offline-diarization limitation).

## Q3. What is the best/current version of FunASR — and are we using it?

- **Latest release: `1.4.15`** (verified against PyPI, the official package
  index). There is no newer stable version.
- **We are on the latest version.** This repo pins and ships `funasr-1.4.15`
  (`wheels/funasr-1.4.15-py3-none-any.whl`, confirmed installed in `.venv`).

No upgrade is needed. If you later change model or speaker support, simply
re-run `scripts/download_wheels.sh` + `scripts/install_offline.sh` to refresh
bundled wheels.

## Q4. Which models does this service actually load?

- `models/SenseVoiceSmall` → ASR (multilingual)
- `models/fsmn-vad` → VAD
- `models/campplus` → speaker embeddings for diarization (see Q2)
- No punctuation model.

Configured via `FUNASR_ASR_MODEL`, `FUNASR_VAD_MODEL` and `FUNASR_SPK_MODEL`
env vars (defaults: `models/SenseVoiceSmall`, `models/fsmn-vad`,
`models/campplus`).

## Q5. Is the service offline?

Yes. All weights are bundled under `models/` and platform-specific pip wheels
can be bundled under `wheels/` (CUDA/Linux) or `wheels-macos/` (Apple
Silicon); the server performs no network access at runtime.

## Q6. What output does the ASR produce (emotions/events)?

SenseVoiceSmall natively emits emotion/event tags, but this service strips
them (see README) and returns clean text, with optional ITN
(inverse text normalization) and timestamps via `verbose_json`.

## Q7. Can I run this on an M4 Mac, and does it use the NPU?

Yes. Run `scripts/install_macos.sh` followed by `scripts/run_macos.sh`. The
installer selects the arm64 PyTorch build and the service uses `mps`, PyTorch's
Metal GPU backend. Set `FUNASR_DEVICE=cpu` to force CPU execution.

The current Python/FunASR path does **not** directly use Apple's Neural Engine
(ANE). ANE acceleration would require a separate Core ML/Core AI model export
and native runtime integration; it is not enabled by selecting `mps`.

## Q8. Can the server handle highly concurrent requests?

**Concurrency-safe, but throughput-serialized.** The design accepts many
simultaneous connections without falling over, but all ASR work funnels
through a single global lock, so it does not get faster as clients are added.

How it works (`funasr_core/server.py`, `model_manager.py`):

- One process owns the GPU; blocking `generate()` calls run on a
  `ThreadPoolExecutor`, gated by async semaphores
  (`FUNASR_CONCURRENT_VAD=4`, `FUNASR_CONCURRENT_ASR=1`). HTTP and WS share
  the same semaphores, so total GPU work stays bounded regardless of client
  count.
- `ModelManager` additionally holds threading locks per model
  (`_asr_inference_lock`, `_vad_inference_lock`, `_spk_inference_lock`)
  because the shared `AutoModel` instance mutates its inference kwargs.
  Raising `CONCURRENT_ASR` therefore does **not** parallelize ASR — decoding
  is strictly one chunk at a time, globally.
- There is no batching. `FUNASR_BATCH_SIZE_S` exists in `config.py` but is
  unused; chunks from different requests are never merged into one GPU batch.

Practical consequences:

- ✅ Stable under load: measured 32 parallel file uploads and 32 concurrent
  WS streams all succeed (see README benchmarks), bounded by the connection
  cap (`FUNASR_MAX_WS_CONNECTIONS=100`) and upload cap (200 MB).
- ❌ Poor throughput scaling: GPU peaked at only ~39% / 1.6 GB during the
  benchmark — the GPU idles while requests queue on the ASR lock.
- ❌ Head-of-line blocking: one long upload (e.g. 1 hour → ~240 sequential
  chunks) holds the ASR lock and stalls every other HTTP/WS request.
- Fine for: tens of concurrent realtime mic streams with short utterances.
- Not fine for: many clients transcribing long files simultaneously.

Cheaper scaling options without code changes: run a second process on
another port/GPU (see README), or put long-file batch jobs on a separate
instance so they cannot block realtime traffic.

## Q9. Should we use FunASR's vLLM backend for higher throughput?

Scope note first: **vLLM support in FunASR applies only to the LLM-based
models** (Fun-ASR / Fun-ASR-Nano, which wrap Qwen3-0.6B as an audio-LLM).
It does **not** apply to SenseVoiceSmall, which is non-autoregressive —
there is no KV-cache decode loop for vLLM to accelerate.

Pros:

- **Continuous batching**: decode steps from many in-flight requests are
  merged into single GPU batches, directly fixing the one-chunk-at-a-time
  serialization and the ~39% GPU utilization seen in benchmarks.
- **PagedAttention KV-cache management**: more concurrent sequences fit in
  the same VRAM, allocated on demand.
- **Serving features for free**: request scheduling, prefix caching,
  chunked prefill and admission control — strictly better than the
  hand-rolled semaphore + lock layer.
- **No head-of-line blocking**: long and short requests interleave at the
  step level instead of queueing behind each other.

Cons (specific to this deployment):

- Requires standardizing on **Fun-ASR-Nano** (800M; Chinese/English/
  Japanese + dialects) — you lose SenseVoiceSmall's broader multilingual
  coverage and emotion/event tags.
- **VRAM cost**: vLLM pre-allocates a KV-cache pool
  (`gpu_memory_utilization`); co-locating vLLM + VAD + CAM++ diarization
  on a 16 GB card is tight.
- **Heavier offline bundle**: vLLM has strict torch/CUDA version coupling
  and large wheels, complicating the `wheels/` offline-install model.
- **Slower cold start** (CUDA-graph capture / compilation warmup) and no
  benefit for single requests — batching gains only appear when there is a
  queue of work.
- **Does not help the realtime path**: WS partials are short repeated
  decodes on ~8 s windows; vLLM targets offline batch throughput, and
  wiring token-level streaming partials through it is extra work.

Bottom line: if the real workload is many parallel *file transcriptions*,
switching Fun-ASR-Nano to its vLLM backend is the right lever. If the
workload is mostly realtime streaming and moderate uploads on
SenseVoiceSmall, the current design is adequate and multiple processes are
the cheaper scaling path.
