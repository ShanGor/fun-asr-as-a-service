import os
import platform
import sys
from pathlib import Path

import yaml

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _default_device() -> str:
    """Choose the native accelerator without changing Linux defaults."""
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mps"
    return "cuda"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configuration from an optional YAML file, overlaid by env vars."""
    cfg = {
        "models_dir": os.environ.get("FUNASR_MODELS_DIR", str(DEFAULT_MODELS_DIR)),
        "asr_model": os.environ.get("FUNASR_ASR_MODEL", "SenseVoiceSmall"),
        "asr_hub": os.environ.get("FUNASR_ASR_HUB", "ms"),
        "vad_model": os.environ.get("FUNASR_VAD_MODEL", "fsmn-vad"),
        "punc_model": os.environ.get("FUNASR_PUNC_MODEL", ""),
        "spk_model": os.environ.get("FUNASR_SPK_MODEL", "campplus"),
        "device": os.environ.get("FUNASR_DEVICE", _default_device()),
        "host": os.environ.get("FUNASR_HOST", "127.0.0.1"),
        "port": _env_int("FUNASR_PORT", 8000),
        "worker_threads": _env_int("FUNASR_WORKER_THREADS", max(8, os.cpu_count() or 8)),
        "concurrent_vad": _env_int("FUNASR_CONCURRENT_VAD", 4),
        "concurrent_asr": _env_int("FUNASR_CONCURRENT_ASR", 1),
        "batch_size_s": _env_int("FUNASR_BATCH_SIZE_S", 300),
        "max_upload_bytes": _env_int("FUNASR_MAX_UPLOAD_BYTES", 200 * 1024 * 1024),
        "max_ws_connections": _env_int("FUNASR_MAX_WS_CONNECTIONS", 100),
        "chunk_interval_ms": _env_int("FUNASR_CHUNK_INTERVAL_MS", 100),
        "vad_max_segment_ms": _env_int("FUNASR_VAD_MAX_SEGMENT_MS", 20000),
        "asr_target_ms": _env_int("FUNASR_ASR_TARGET_MS", 15000),
        "asr_overlap_ms": _env_int("FUNASR_ASR_OVERLAP_MS", 500),
        "vad_split_silence_ms": _env_int("FUNASR_VAD_SPLIT_SILENCE_MS", 300),
        "partial_interval_ms": _env_int("FUNASR_PARTIAL_INTERVAL_MS", 500),
        "partial_window_sec": _env_float("FUNASR_PARTIAL_WINDOW_SEC", 8.0),
        "max_speaking_sec": _env_float("FUNASR_MAX_SPEAKING_SEC", 120.0),
        "itn": _env_bool("FUNASR_ITN", True),
        "language": os.environ.get("FUNASR_LANGUAGE", "auto"),
        "offline": _env_bool("FUNASR_OFFLINE", True),
        "log_level": os.environ.get("FUNASR_LOG_LEVEL", "INFO"),
    }

    if path:
        p = Path(path)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            # env vars win over the file
            cfg = {**file_cfg, **cfg}

    # resolve model paths relative to models_dir when they are bare names
    models_dir = Path(cfg["models_dir"])
    for key in ("asr_model", "vad_model", "punc_model", "spk_model"):
        val = cfg.get(key)
        if val and not Path(val).is_absolute() and not (Path(val)).is_file():
            candidate = models_dir / val
            if candidate.is_dir():
                cfg[key] = str(candidate)
    cfg["models_dir"] = str(models_dir)
    return cfg
