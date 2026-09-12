import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .audio import SAMPLE_RATE, DecodedUpload, duration_seconds
from .config import load_config
from .model_manager import ModelManager
from .streaming import WSSession

logger = logging.getLogger("funasr_core")


def _semaphore(n: int) -> asyncio.Semaphore:
    return asyncio.Semaphore(max(1, int(n)))


class AppState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.manager = ModelManager(cfg)
        self.executor = ThreadPoolExecutor(max_workers=cfg["worker_threads"])
        self.sem_vad = _semaphore(cfg["concurrent_vad"])
        self.sem_asr = _semaphore(cfg["concurrent_asr"])
        self.active_ws = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = app.state.runtime
    logging.basicConfig(
        level=getattr(logging, state.cfg["log_level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("loading models ...")
    state.manager.warmup()
    logger.info("models ready: asr=%s vad=%s device=%s", state.cfg["asr_model"], state.cfg["vad_model"], state.cfg["device"])
    yield
    state.executor.shutdown(wait=False, cancel_futures=True)


cfg = load_config()
app = FastAPI(title="FunASR Core", version="0.1.0", lifespan=lifespan)
app.state.runtime = AppState(cfg)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ===================================================================== HTTP
@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the test UI (self-contained single page)."""
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend not bundled")
    return FileResponse(index)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": cfg["device"],
        "asr_model": cfg["asr_model"],
        "vad_model": cfg["vad_model"],
        "active_ws": len(app.state.runtime.active_ws),
    }


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": cfg["asr_model"], "owned_by": "local", "created": int(time.time())}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("sensevoice"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: str = Form("segment"),
    diarize: bool = Form(False),
    max_speakers: Optional[int] = Form(None),
):
    state = app.state.runtime
    if file.size is not None and file.size > cfg["max_upload_bytes"]:
        raise HTTPException(status_code=413, detail="audio file too large")
    if response_format not in ("text", "json", "verbose_json"):
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    def process():
        # Ownership stays in the worker, including cleanup on inference failure.
        try:
            decoded = DecodedUpload(file.file, cfg["max_upload_bytes"])
        except Exception as e:
            code = 413 if str(e) == "audio file too large" else 400
            raise HTTPException(status_code=code, detail=str(e)) from e
        try:
            audio = decoded.audio
            segments = state.manager.transcribe_with_vad(audio, language, cfg["itn"])
            for i, seg in enumerate(segments):
                seg.update(id=i, words=[])
                logger.info("ASR chunk %d %.3f-%.3fs status=%s", i,
                            seg["start"], seg["end"], seg["status"])
            speakers = 0
            spoken = [s for s in segments if s["text"]]
            if diarize and spoken:
                try:
                    speakers = state.manager.assign_speakers(spoken, audio, max_speakers)
                except Exception:
                    logger.exception("diarization failed")
                    for seg in spoken:
                        seg["speaker"] = 0
                    speakers = 1
            return segments, speakers, round(duration_seconds(audio), 3)
        finally:
            decoded.close()

    async with state.sem_asr:
        job = asyncio.get_running_loop().run_in_executor(state.executor, process)
        try:
            segments, speakers, duration = await asyncio.shield(job)
        except asyncio.CancelledError:
            # Keep the upload open and admission slot held until the worker exits.
            try:
                await job
            finally:
                raise
    text = " ".join(s["text"] for s in segments if s["text"])
    base = {"task": "transcribe", "language": language or "auto", "duration": duration}
    if diarize:
        base["speakers"] = speakers

    if response_format == "text":
        return PlainTextResponse(content=text)
    if response_format in ("verbose_json", "json"):
        body = {**base, "text": text}
        if response_format == "verbose_json":
            body["segments"] = segments
        return JSONResponse(content=body)
    raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")


# ================================================================== WebSocket
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    state = app.state.runtime
    if len(state.active_ws) >= cfg["max_ws_connections"]:
        await websocket.close(code=1013, reason="server busy")
        return

    await websocket.accept()
    state.active_ws.add(websocket)
    logger.info("ws connected: %d active", len(state.active_ws))

    session = WSSession(
        manager=state.manager,
        executor=state.executor,
        sem_vad=state.sem_vad,
        sem_asr=state.sem_asr,
        cfg=cfg,
    )
    ended = False
    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break
            if msg_type == "websocket.receive":
                text = message.get("text")
                if text is not None:
                    if await _handle_text(websocket, session, text):
                        # client signalled end of input: flush and ack
                        for m in await session.finish():
                            await websocket.send_json(m)
                        ended = True
                        break
                elif message.get("bytes") is not None:
                    msgs = await session.feed(message["bytes"])
                    for m in msgs:
                        await websocket.send_json(m)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("ws error: %s", e)
    finally:
        if not ended:
            try:
                for m in await session.finish():
                    await websocket.send_json(m)
            except Exception:
                pass
        state.active_ws.discard(websocket)
        logger.info("ws disconnected: %d active", len(state.active_ws))


async def _handle_text(websocket: WebSocket, session: WSSession, text: str) -> bool:
    text = (text or "").strip()
    if text.upper() in ("STOP", "STOP_AUDIO"):
        return True
    try:
        payload = json.loads(text)
    except Exception:
        return False

    if "mode" in payload and payload["mode"] in ("online", "offline", "2pass"):
        session.mode = payload["mode"]
    if "language" in payload and payload["language"]:
        session.language = payload["language"]
    if "itn" in payload:
        session.itn = bool(payload["itn"])
    if "chunk_interval" in payload and payload["chunk_interval"]:
        session.vad_chunk_ms = max(60, int(payload["chunk_interval"]) * 10)
    if "chunk_interval_ms" in payload and payload["chunk_interval_ms"]:
        session.vad_chunk_ms = max(60, int(payload["chunk_interval_ms"]))
    if "partial_interval_ms" in payload and payload["partial_interval_ms"]:
        session.partial_interval_ms = max(200, int(payload["partial_interval_ms"]))
    if "partial_window_sec" in payload and payload["partial_window_sec"]:
        session.partial_window_bytes = int(float(payload["partial_window_sec"]) * SAMPLE_RATE * 2)
    if "audio_fs" in payload and payload["audio_fs"] not in (None, 16000):
        await websocket.send_json({"mode": session.mode, "error": "only 16 kHz mono PCM16 supported"})
    if "is_speaking" in payload and payload["is_speaking"] is False:
        return True
    return False


def main():
    import uvicorn

    uvicorn.run(
        "funasr_core.server:app",
        host=cfg["host"],
        port=cfg["port"],
        ws="websockets-sansio",
        log_level=cfg["log_level"].lower(),
    )


if __name__ == "__main__":
    main()
