import re
import threading
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from funasr import AutoModel

from .audio import SAMPLE_RATE
from .chunking import plan_chunks, reconcile_overlap
from .diarization import speaker_windows, label_segments

logger = logging.getLogger("funasr_core")

_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")

# cosine-distance threshold for automatic speaker clustering; overridden by
# an explicit max_speakers request
SPEAKER_DISTANCE_THRESHOLD = 0.45


def split_long_segments(
    segments: List[Tuple[int, int]], max_ms: int
) -> List[Tuple[int, int]]:
    """Split (beg_ms, end_ms) segments longer than max_ms into equal sub-parts.

    SenseVoice decode degrades badly beyond ~10-15 s per pass, so e.g. a 56 s
    uninterrupted song must be fed to ASR as small chunks.
    """
    max_ms = max(1000, int(max_ms))
    out: List[Tuple[int, int]] = []
    for beg, end in segments:
        duration = end - beg
        if duration <= max_ms:
            out.append((beg, end))
            continue
        n = -(-duration // max_ms)  # ceil
        step = duration // n
        for i in range(n):
            chunk_end = end if i == n - 1 else beg + (i + 1) * step
            out.append((beg + i * step, chunk_end))
    return out


def group_vad_segments(
    segments: List[Tuple[int, int]], split_silence_ms: int
) -> List[Tuple[int, int]]:
    """Create ASR chunks only across sufficiently long VAD-confirmed silence.

    Consecutive VAD speech regions with a short gap are decoded together, so a
    short pause cannot become an ASR boundary. A gap of at least
    ``split_silence_ms`` is retained as a safe split point. In particular, a
    continuous utterance is never split by elapsed time.
    """
    if not segments:
        return []

    split_silence_ms = max(0, int(split_silence_ms))
    ordered = sorted((int(beg), int(end)) for beg, end in segments if end > beg)
    if not ordered:
        return []

    chunks: List[Tuple[int, int]] = []
    chunk_start, chunk_end = ordered[0]
    for beg, end in ordered[1:]:
        if beg - chunk_end >= split_silence_ms:
            chunks.append((chunk_start, chunk_end))
            chunk_start, chunk_end = beg, end
        else:
            # Keep any short silence inside the decode window instead of
            # making it an artificial ASR split.
            chunk_end = max(chunk_end, end)
    chunks.append((chunk_start, chunk_end))
    return chunks


def _clean_sensevoice_text(text: str) -> str:
    text = _RICH_TAG_RE.sub("", text or "")
    return text.strip()


class ModelManager:
    """Loads ASR + VAD models once and serves concurrent blocking inference.

    A single instance owns the GPU. `generate` calls are offloaded to a thread
    pool by the server and gated by asyncio semaphores.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg["device"]
        self._lock = threading.Lock()
        self._vad_inference_lock = threading.Lock()
        self._asr_inference_lock = threading.Lock()
        self._spk_inference_lock = threading.Lock()
        self._asr = None
        self._vad = None
        self._spk = None

    # ------------------------------------------------------------- loading
    def _auto_model_kwargs(self, model_path: Optional[str] = None) -> dict:
        kwargs = {
            "device": self.device,
            "disable_update": True,
            "disable_pbar": True,
            "disable_log": True,
        }
        # Fun-ASR-Nano is a custom FunASR model implementation. Keep the
        # normal SenseVoice path unchanged and opt in only when model.py is
        # present beside the configured checkpoint.
        model_path = model_path or self.cfg.get("asr_model", "")
        if model_path and (Path(model_path) / "model.py").is_file():
            kwargs.update({"trust_remote_code": True,
                           "remote_code": str(Path(model_path) / "model.py"),
                           "hub": self.cfg.get("asr_hub", "ms")})
        return kwargs

    @property
    def asr(self) -> AutoModel:
        if self._asr is None:
            with self._lock:
                if self._asr is None:
                    self._asr = AutoModel(model=self.cfg["asr_model"], **self._auto_model_kwargs(self.cfg["asr_model"]))
        return self._asr

    @property
    def vad(self) -> AutoModel:
        if self._vad is None:
            with self._lock:
                if self._vad is None:
                    self._vad = AutoModel(model=self.cfg["vad_model"], **self._auto_model_kwargs(self.cfg["vad_model"]))
        return self._vad

    @property
    def spk(self) -> AutoModel:
        if self._spk is None:
            with self._lock:
                if self._spk is None:
                    model = self.cfg.get("spk_model")
                    if not model:
                        raise RuntimeError("speaker model not configured (FUNASR_SPK_MODEL)")
                    self._spk = AutoModel(model=model, **self._auto_model_kwargs(model))
        return self._spk

    def warmup(self) -> None:
        """Load models and run a tiny dummy inference to initialize the backend."""
        self.vad
        self.asr
        dummy = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        if (Path(self.cfg.get("asr_model", "")) / "model.py").is_file():
            # Custom LLM-ASR checkpoints may autoregress on silence without
            # emitting EOS; loading the weights is sufficient CUDA warmup.
            pass
        else:
            self.asr.generate(input=dummy, language="auto", use_itn=False)
        if self.cfg.get("spk_model"):
            self.spk

    # ------------------------------------------------------------ offline
    def vad_offline_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Run bounded VAD windows, returning absolute (start_ms, end_ms)."""
        if audio.size == 0:
            return []
        out = []
        # Independent short windows bound VAD activations and cache growth.
        # Missing/cross-window regions only affect cut preferences, not coverage.
        step = 30 * SAMPLE_RATE
        for offset in range(0, len(audio), step):
            try:
                with self._vad_inference_lock:
                    res = self.vad.generate(input=np.array(audio[offset:offset + step]),
                                            cache={}, is_final=True,
                                            max_end_silence_time=max(200, int(
                                                self.cfg.get("vad_split_silence_ms", 300))))
                for item in res or []:
                    for s in item.get("value", []):
                        if len(s) >= 2 and 0 <= s[0] < s[1]:
                            out.append((offset // 16 + int(s[0]), offset // 16 + int(s[1])))
            except Exception:
                logger.exception("VAD failed at sample %d; retaining audio for ASR", offset)
        return out

    def plan_chunks(self, audio, vad_segments):
        return plan_chunks(audio, vad_segments,
                           self.cfg.get("asr_target_ms", 15000),
                           self.cfg.get("vad_max_segment_ms", 20000),
                           self.cfg.get("vad_split_silence_ms", 300),
                           self.cfg.get("asr_overlap_ms", 500))

    def transcribe(self, audio: np.ndarray, language: Optional[str], itn: bool) -> str:
        """Transcribe a single audio clip (no VAD)."""
        if audio.size == 0:
            return ""
        kwargs = {"use_itn": itn}
        if language:
            kwargs["language"] = language
        # AutoModel mutates inference kwargs; serialize shared-instance calls.
        model_path = self.cfg.get("asr_model", "")
        if model_path and (Path(model_path) / "model.py").is_file():
            # Fun-ASR-Nano's custom inference path distinguishes raw tensors
            # (in-memory audio) from file paths and builds its ChatML prompt.
            model_input = [torch.from_numpy(np.asarray(audio, dtype=np.float32))]
        else:
            model_input = np.array(audio)
        with self._asr_inference_lock:
            res = self.asr.generate(input=model_input, **kwargs)
        if not res:
            return ""
        text = res[0].get("text", "") or ""
        return _clean_sensevoice_text(text)

    def transcribe_with_vad(
        self, audio: np.ndarray, language: Optional[str], itn: bool
    ) -> List[dict]:
        """Full pipeline: VAD segmentation + per-segment ASR.

        Returns a list of dicts: {"start": float, "end": float, "text": str}
        where timestamps are in seconds.
        """
        segments = []
        vad_segs = self.vad_offline_segments(audio)
        bounds = self.plan_chunks(audio, vad_segs)
        logger.info("ASR planned %d chunks for %.3fs (%d VAD regions)",
                    len(bounds), len(audio) / SAMPLE_RATE, len(vad_segs))
        previous = ""
        for index, bound in enumerate(bounds):
            logger.info("ASR chunk %d/%d decoding %.3f-%.3fs overlap=%s",
                        index + 1, len(bounds), bound.start / SAMPLE_RATE,
                        bound.end / SAMPLE_RATE, bound.overlap)
            chunk = audio[bound.start:bound.end]
            text = self.transcribe(chunk, language, itn)
            raw = text
            if bound.overlap:
                text = reconcile_overlap(previous, text)
            previous = raw
            segments.append({"start": round(bound.start / SAMPLE_RATE, 3),
                             "end": round(bound.end / SAMPLE_RATE, 3),
                             "text": text, "status": "ok" if text else "empty"})
            logger.info("ASR chunk %d/%d completed status=%s", index + 1,
                        len(bounds), segments[-1]["status"])
        return segments

    # --------------------------------------------------------- diarization
    def speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract one CAM++ speaker embedding for an audio clip."""
        if audio.size < SAMPLE_RATE // 10:  # <100 ms: embedding unreliable
            return None
        with self._spk_inference_lock:
            res = self.spk.generate(input=audio)
        if not res:
            return None
        emb = res[0].get("spk_embedding")
        if emb is None:
            return None
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().float().cpu().numpy()
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def assign_speakers(
        self,
        segments: List[dict],
        audio: np.ndarray,
        max_speakers: Optional[int] = None,
    ) -> int:
        """Cluster independent voice windows; set seg["speaker"] in-place.

        Returns the number of distinct speakers found. Falls back to a single
        speaker when embeddings or clustering fail.
        """
        if not segments:
            return 0
        # A known single voice needs no embedding inference or clustering.
        # Singing and accompaniment can vary substantially between phrases.
        if max_speakers == 1:
            for seg in segments:
                seg["speaker"] = 0
            return 1
        embeddings = []
        valid = []
        windows = speaker_windows(len(audio), self.vad_offline_segments(audio))
        for start, end in windows:
            chunk = audio[start:end]
            emb = self.speaker_embedding(chunk)
            if emb is not None and np.all(np.isfinite(emb)) and np.linalg.norm(emb) > 0:
                embeddings.append(emb)
                valid.append((start, end))

        labels: Optional[List[int]] = None
        if len(embeddings) == 1:
            labels = [0]
        elif embeddings:
            try:
                from sklearn.cluster import AgglomerativeClustering

                x = np.stack(embeddings)
                kwargs: dict = {"metric": "cosine", "linkage": "average"}
                n = len(embeddings)
                if max_speakers and max_speakers >= 2 and n >= max_speakers:
                    kwargs["n_clusters"] = max_speakers
                else:
                    kwargs["n_clusters"] = None
                    kwargs["distance_threshold"] = SPEAKER_DISTANCE_THRESHOLD
                model = AgglomerativeClustering(**kwargs)
                labels = list(model.fit_predict(x))
            except Exception:
                logger.exception("Speaker clustering failed; using one speaker")
                labels = None

        for idx, seg in enumerate(segments):
            seg["speaker"] = 0
        if labels is not None:
            return label_segments(segments, valid, labels)
        return 1 if segments else 0

    # ------------------------------------------------------------ streaming
    def vad_streaming(
        self, pcm: bytes, cache: dict, is_final: bool, chunk_size: int
    ) -> Tuple[int, int]:
        """Feed one PCM frame to streaming VAD.

        Returns (speech_start_ms, speech_end_ms); -1 means not triggered.
        """
        with self._vad_inference_lock:
            res = self.vad.generate(
                input=pcm, cache=cache, is_final=is_final, chunk_size=chunk_size
            )
        segments = res[0].get("value", []) if res else []
        start, end = -1, -1
        if len(segments) == 1 and len(segments[0]) >= 2:
            if segments[0][0] != -1:
                start = int(segments[0][0])
            if segments[0][1] != -1:
                end = int(segments[0][1])
        return start, end
