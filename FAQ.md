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

Yes. All weights are bundled under `models/` and all pip wheels under
`wheels/`; the server performs no network access at runtime.

## Q6. What output does the ASR produce (emotions/events)?

SenseVoiceSmall natively emits emotion/event tags, but this service strips
them (see README) and returns clean text, with optional ITN
(inverse text normalization) and timestamps via `verbose_json`.