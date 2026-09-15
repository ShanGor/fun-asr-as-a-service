#!/usr/bin/env python3
"""Download FunASR model weights into ./models for offline deployment.

Run once on a machine with network access; the resulting `models/` directory
is bundled with the deployment package so the server never downloads at
runtime.

Models:
  - SenseVoiceSmall   : multilingual ASR (zh/yue/en/ja/ko + 50 langs)
  - Fun-ASR-Nano-2512 : larger 800M-parameter ASR (Chinese/English/Japanese)
  - Qwen3-0.6B       : language component required by Fun-ASR-Nano-2512
  - Qwen3-ASR-0.6B   : optional multilingual Qwen3-ASR checkpoint (52 langs)
  - Qwen3-ASR-1.7B   : optional, larger/more accurate multilingual checkpoint
  - fsmn-vad          : streaming + offline voice activity detection
  - campplus          : speaker embeddings for diarization
"""
import argparse
import sys
import urllib.request
from pathlib import Path

DEFAULT_MODELS = {
    "SenseVoiceSmall": "iic/SenseVoiceSmall",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "Fun-ASR-Nano-2512": "FunAudioLLM/Fun-ASR-Nano-2512",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
}
DEFAULT_DOWNLOAD_MODELS = ("SenseVoiceSmall", "fsmn-vad", "campplus")

# Accept the shorter name used in casual references without creating a
# second local copy of the same checkpoint. The canonical names above are
# also the names used by FUNASR_ASR_MODEL when starting the service.
MODEL_ALIASES = {
    "Qwen-ASR": "Qwen3-ASR-1.7B",
    "Qwen-ASR-0.6B": "Qwen3-ASR-0.6B",
    "Qwen-ASR-1.7B": "Qwen3-ASR-1.7B",
}

FUN_ASR_CODE_URLS = {
    "model.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py",
    "ctc.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/ctc.py",
    "tools/__init__.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/tools/__init__.py",
    "tools/utils.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/tools/utils.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models", help="target directory")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_DOWNLOAD_MODELS),
        help=f"subset of models to download: {list(DEFAULT_MODELS.keys())}",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    from modelscope import snapshot_download

    failures = []
    for requested_name in args.models:
        local_name = MODEL_ALIASES.get(requested_name, requested_name)
        model_id = DEFAULT_MODELS.get(local_name)
        if model_id is None:
            print(f"[skip] unknown model: {requested_name}")
            continue
        target = models_dir / local_name
        already = target.is_dir() and (
            any(target.rglob("*.pt")) or any(target.rglob("*.onnx")) or any(target.rglob("*.safetensors")) or any(target.rglob("*.yaml"))
        )
        if local_name == "Fun-ASR-Nano-2512":
            already = already and (target / "model.py").is_file() and (target / "ctc.py").is_file()
        if already:
            print(f"[ok] {local_name} already present at {target}")
            continue
        print(f"[download] {model_id} -> {target}")
        try:
            snapshot_download(model_id, local_dir=str(target))
            if local_name == "Fun-ASR-Nano-2512":
                for relative, url in FUN_ASR_CODE_URLS.items():
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, destination)
            print(f"[done] {local_name}")
        except Exception as e:
            print(f"[fail] {model_id}: {e}")
            failures.append(local_name)

    if failures:
        print("FAILED:", ", ".join(failures), file=sys.stderr)
        return 1
    print("all models ready under", models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
