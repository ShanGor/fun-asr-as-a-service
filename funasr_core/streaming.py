import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from .audio import SAMPLE_RATE
from .model_manager import ModelManager

BYTES_PER_SEC = SAMPLE_RATE * 2  # 16-bit mono PCM


def _pcm_duration_ms(pcm: bytes) -> int:
    return int(len(pcm) / (SAMPLE_RATE * 2) * 1000)


class WSSession:
    """One realtime connection: streaming VAD segmentation + offline ASR.

    Audio is expected as raw PCM16 mono 16 kHz. The pipeline mirrors FunASR's
    2-pass approach but transcribes every VAD segment with the offline
    (multilingual) model, which keeps the service model-agnostic and stable.
    """

    def __init__(
        self,
        manager: ModelManager,
        executor: ThreadPoolExecutor,
        sem_vad: asyncio.Semaphore,
        sem_asr: asyncio.Semaphore,
        cfg: dict,
        mode: str = "2pass",
        language: Optional[str] = None,
        itn: bool = True,
    ):
        self.manager = manager
        self.executor = executor
        self.sem_vad = sem_vad
        self.sem_asr = sem_asr
        self.cfg = cfg
        self.mode = mode if mode in ("online", "offline", "2pass") else "2pass"
        self.language = language
        self.itn = itn

        self.vad_cache: dict = {}
        self.speech_frames: List[bytes] = []
        self.speech_started = False
        self.seg_start_ms = 0
        self.elapsed_ms = 0

        self.vad_chunk_ms = int(cfg.get("chunk_interval_ms", 100))
        self.vad_feeder: List[bytes] = []
        self.partial_interval_ms = int(cfg.get("partial_interval_ms", 500))
        self.partial_window_bytes = int(
            float(cfg.get("partial_window_sec", 8.0)) * BYTES_PER_SEC
        )
        self.max_speaking_bytes = int(
            float(cfg.get("max_speaking_sec", 120.0)) * BYTES_PER_SEC
        )
        self.next_partial_ms = self.partial_interval_ms
        self.segments: List[dict] = []

    # ------------------------------------------------------------- helpers
    async def _vad(self, pcm: bytes, is_final: bool):
        async with self.sem_vad:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self.manager.vad_streaming,
                pcm,
                self.vad_cache,
                is_final,
                self.vad_chunk_ms,
            )

    async def _asr(self, audio: bytes, final: bool):
        async with self.sem_asr:
            loop = asyncio.get_running_loop()
            arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            text = await loop.run_in_executor(
                self.executor,
                self.manager.transcribe,
                arr,
                self.language,
                self.itn,
            )
        return text

    def _current_speech(self) -> bytes:
        data = b"".join(self.speech_frames)
        if len(data) > self.partial_window_bytes:
            data = data[-self.partial_window_bytes :]
        return data

    def _seg_seconds(self, audio: bytes) -> float:
        return round(len(audio) / BYTES_PER_SEC, 3)

    # ------------------------------------------------------------- feeding
    async def feed(self, pcm: bytes) -> List[dict]:
        """Feed one PCM frame. Returns a list of messages to send to client."""
        messages: List[dict] = []
        if not pcm:
            return messages

        self.elapsed_ms += _pcm_duration_ms(pcm)
        if self.speech_started:
            self.speech_frames.append(pcm)

        # accumulate frames to the VAD chunk size before calling VAD
        self.vad_feeder.append(pcm)
        feeder_ms = sum(_pcm_duration_ms(f) for f in self.vad_feeder)
        if feeder_ms < self.vad_chunk_ms:
            return messages
        frame = b"".join(self.vad_feeder)
        self.vad_feeder = []

        start_ms, end_ms = await self._vad(frame, is_final=False)

        if start_ms != -1 and not self.speech_started:
            self.speech_started = True
            self.seg_start_ms = self.elapsed_ms - _pcm_duration_ms(frame)
            # include the current frame as pre-roll
            self.speech_frames = [frame]

        if self.speech_started:
            # partial results while speaking
            if self.elapsed_ms >= self.next_partial_ms:
                self.next_partial_ms = self.elapsed_ms + self.partial_interval_ms
                audio = self._current_speech()
                if len(audio) >= BYTES_PER_SEC:  # at least ~1s before partials
                    text = await self._asr(audio, final=False)
                    if text:
                        dur = self._seg_seconds(audio)
                        end_s = self.elapsed_ms / 1000.0
                        start_s = max(self.seg_start_ms / 1000.0, end_s - dur)
                        messages.append(
                            {
                                "mode": self.mode,
                                "wav_name": "microphone",
                                "is_final": False,
                                "text": text,
                                "timestamp": [[round(start_s, 3), round(end_s, 3)]],
                            }
                        )

            # force-commit extremely long single utterances so buffers stay bounded
            if len(b"".join(self.speech_frames)) > self.max_speaking_bytes:
                await self._commit_segment(messages)

        if end_ms != -1:
            await self._commit_segment(messages)

        return messages

    async def _commit_segment(self, messages: List[dict]):
        audio = b"".join(self.speech_frames)
        if audio:
            # Reuse the coverage-preserving planner, including the final sample.
            from .chunking import reconcile_overlap
            waveform = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            bounds = self.manager.plan_chunks(waveform, [])
            texts = []
            previous = ""
            for bound in bounds:
                t = await self._asr(audio[bound.start * 2:bound.end * 2], final=True)
                raw = t
                if bound.overlap:
                    t = reconcile_overlap(previous, t)
                previous = raw
                if t:
                    texts.append(t)
            text = " ".join(texts)
            if text:
                seg = {
                    "start": round(self.seg_start_ms / 1000.0, 3),
                    "end": round(self.seg_start_ms / 1000.0 + self._seg_seconds(audio), 3),
                    "text": text,
                }
                self.segments.append(seg)
                messages.append(
                    {
                        "mode": self.mode,
                        "wav_name": "microphone",
                        "is_final": True,
                        "is_end": False,
                        "text": text,
                        "timestamp": [[seg["start"], seg["end"]]],
                    }
                )
        self.speech_frames = []
        self.speech_started = False
        self.vad_cache = {}

    async def finish(self) -> List[dict]:
        """Flush remaining audio and emit the end acknowledgement."""
        messages: List[dict] = []
        try:
            if self.vad_feeder:
                frame = b"".join(self.vad_feeder)
                self.vad_feeder = []
                start_ms, _ = await self._vad(frame, is_final=True)
                # The final VAD call can be the first call that confirms a
                # short utterance. Preserve it instead of ending with an
                # empty transcript when the user stops quickly.
                if start_ms != -1 and not self.speech_started:
                    self.speech_started = True
                    self.seg_start_ms = max(0, self.elapsed_ms - _pcm_duration_ms(frame))
                    self.speech_frames = [frame]
        except Exception:
            pass
        if self.speech_started:
            await self._commit_segment(messages)
        final_text = " ".join(s["text"] for s in self.segments)
        messages.append(
            {
                "mode": self.mode,
                "wav_name": "microphone",
                "is_final": True,
                "is_end": True,
                "text": final_text,
                "timestamp": [[s["start"], s["end"]] for s in self.segments],
            }
        )
        return messages
