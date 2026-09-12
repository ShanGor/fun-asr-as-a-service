import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000


class DecodedUpload:
    """Disk-backed mono PCM; only ASR-sized slices need to enter RAM."""

    def __init__(self, source, max_bytes):
        self.directory = tempfile.TemporaryDirectory(prefix="funasr-audio-")
        self.audio = None
        try:
            root = Path(self.directory.name)
            compressed = root / "input"
            size = 0
            source.seek(0)
            with compressed.open("wb") as output:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("audio file too large")
                    output.write(block)
            pcm = root / "audio.f32"
            ffmpeg = _find_ffmpeg()
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required for bounded upload decoding")
            proc = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-i", str(compressed),
                 "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), str(pcm)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
            if proc.returncode or not pcm.stat().st_size:
                raise ValueError(f"audio decode failed: {proc.stderr.decode(errors='replace')[:500]}")
            self.audio = np.memmap(pcm, dtype=np.float32, mode="r")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.audio is not None:
            self.audio._mmap.close()
            self.audio = None
        self.directory.cleanup()


def _resample_mono(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Resample float32 mono waveform to 16 kHz, returning np.float32."""
    if sr == SAMPLE_RATE:
        return waveform.astype(np.float32)
    tensor = torch.from_numpy(waveform).float().reshape(1, -1)
    tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=SAMPLE_RATE)
    return tensor.numpy().reshape(-1).astype(np.float32)


def decode_audio_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """Decode arbitrary audio bytes to float32 mono 16 kHz numpy array.

    Uses libsndfile (bundled with soundfile) first; falls back to ffmpeg for
    formats libsndfile cannot decode (e.g. mp3, aac).
    """
    data = bytes(data)
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            waveform = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate
            if f.channels > 1:
                waveform = waveform.mean(axis=1)
            return _resample_mono(waveform, sr)
    except Exception:
        pass

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "Could not decode audio with libsndfile and ffmpeg is not installed."
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode()[:500]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        raise RuntimeError("decoded audio is empty")
    return pcm


def decode_audio_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_audio_bytes(f.read(), path)


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw PCM16 mono bytes (16 kHz) to float32 numpy array."""
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def duration_seconds(audio: np.ndarray) -> float:
    return float(audio.shape[0]) / SAMPLE_RATE if audio.size else 0.0


def _find_ffmpeg() -> Optional[str]:
    for cand in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        try:
            subprocess.run([cand, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            return cand
        except Exception:
            continue
    return None
