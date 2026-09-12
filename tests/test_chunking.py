import io
import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from funasr_core.audio import DecodedUpload, SAMPLE_RATE
from funasr_core.chunking import plan_chunks, reconcile_overlap
from funasr_core.model_manager import ModelManager


class ChunkingTests(unittest.TestCase):
    def check_coverage(self, audio, vad, **kwargs):
        chunks = plan_chunks(audio, vad, **kwargs)
        if not len(audio):
            self.assertEqual(chunks, [])
            return chunks
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(audio))
        covered = 0
        for chunk in chunks:
            self.assertLessEqual(chunk.start, covered)
            self.assertGreater(chunk.end, covered)
            self.assertLessEqual(chunk.end - chunk.start, kwargs.get("max_ms", 20000) * 16)
            covered = chunk.end
        return chunks

    def test_short_audio_is_not_split_or_trimmed(self):
        self.assertEqual(len(self.check_coverage(np.zeros(7 * SAMPLE_RATE), [(2000, 3000)])), 1)

    def test_missing_vad_and_first_region_only_preserve_tail(self):
        audio = np.zeros(61 * SAMPLE_RATE + 7)
        for vad in ([], [(0, 10000)], [(0, 61000)]):
            self.assertGreater(len(self.check_coverage(audio, vad)), 3)

    def test_pause_is_preferred(self):
        chunks = self.check_coverage(np.ones(40 * SAMPLE_RATE), [(0, 14000), (15000, 40000)])
        self.assertEqual(chunks[0].end, 14500 * 16)
        self.assertEqual(chunks[1].start, chunks[0].end)
        self.assertFalse(chunks[1].overlap)

    def test_adversarial_lengths_and_settings(self):
        rng = np.random.default_rng(42)
        for _ in range(80):
            size = int(rng.integers(0, 100 * SAMPLE_RATE))
            self.check_coverage(np.zeros(size), [(30000, 90000), (-10, 1000), (500, 2000)],
                                target_ms=15000, max_ms=10000, overlap_ms=50000)
        self.check_coverage(np.zeros(0), [])

    def test_boundary_reconciliation(self):
        self.assertEqual(reconcile_overlap("we meet tomorrow morning", "tomorrow morning at nine"), "at nine")
        self.assertEqual(reconcile_overlap("今天一起去公園", "去公園散步"), "散步")
        self.assertEqual(reconcile_overlap("very good", "good morning"), "good morning")
        self.assertEqual(reconcile_overlap("yes yes", "yes yes"), "yes yes")

    def test_pipeline_processes_every_chunk_including_empty(self):
        manager = ModelManager({"device": "cpu"})
        audio = np.zeros(61 * SAMPLE_RATE + 1)
        with patch.object(manager, "vad_offline_segments", return_value=[(0, 5000)]), \
             patch.object(manager, "transcribe", side_effect=["first", "", "middle", "more", "last"]) as asr:
            result = manager.transcribe_with_vad(audio, None, True)
        self.assertEqual(len(result), asr.call_count)
        self.assertGreater(len(result), 3)
        self.assertEqual(result[1]["status"], "empty")
        self.assertEqual(result[-1]["text"], "last")
        self.assertAlmostEqual(result[-1]["end"], len(audio) / SAMPLE_RATE, places=3)
        self.assertTrue(all(len(call.args[0]) <= 20 * SAMPLE_RATE for call in asr.call_args_list))

    def test_disk_decoder_and_cleanup(self):
        with open("samples/BAC009S0764W0121.wav", "rb") as source:
            decoded = DecodedUpload(source, 200 * 1024 * 1024)
        directory = Path(decoded.directory.name)
        self.assertIsInstance(decoded.audio, np.memmap)
        self.assertGreater(len(decoded.audio), SAMPLE_RATE)
        decoded.close()
        self.assertFalse(directory.exists())
        with self.assertRaisesRegex(ValueError, "too large"):
            DecodedUpload(io.BytesIO(b"12345"), 4)

    def test_vad_is_bounded_and_failure_retains_coverage(self):
        from unittest.mock import Mock
        manager = ModelManager({"device": "cpu"})
        manager._vad = Mock()
        manager._vad.generate.side_effect = [
            [{"value": [[0, 5000]]}], RuntimeError("bad VAD"),
            [{"value": [[0, 1000]]}]]
        audio = np.zeros(61 * SAMPLE_RATE)
        with self.assertLogs("funasr_core", level="ERROR"):
            regions = manager.vad_offline_segments(audio)
        self.assertEqual(regions, [(0, 5000), (60000, 61000)])
        self.assertTrue(all(len(c.kwargs["input"]) <= 30 * SAMPLE_RATE
                            for c in manager._vad.generate.call_args_list))
        self.check_coverage(audio, regions)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_inference_failure_cleans_disk(self):
        from fastapi import UploadFile
        from funasr_core import server
        created = []

        def decode(*args):
            result = DecodedUpload(*args)
            created.append(Path(result.directory.name))
            return result

        with open("samples/BAC009S0764W0121.wav", "rb") as source:
            upload = UploadFile(file=source, filename="sample.wav")
            with patch.object(server, "DecodedUpload", side_effect=decode), \
                 patch.object(server.app.state.runtime.manager, "transcribe_with_vad", side_effect=RuntimeError("inference failed")):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    await server.transcriptions(upload, language="zh", response_format="verbose_json", diarize=False)
        self.assertFalse(created[0].exists())

    async def test_upload_returns_all_chunks_and_cleans_disk(self):
        from fastapi import UploadFile
        from funasr_core import server
        created = []

        def decode(*args):
            result = DecodedUpload(*args)
            created.append(Path(result.directory.name))
            return result

        with open("samples/BAC009S0764W0121.wav", "rb") as source:
            upload = UploadFile(file=source, filename="sample.wav")
            rows = [{"start": 0, "end": 2, "text": "first", "status": "ok"},
                    {"start": 2, "end": 3, "text": "", "status": "empty"},
                    {"start": 3, "end": 4.204, "text": "last", "status": "ok"}]
            with patch.object(server, "DecodedUpload", side_effect=decode), \
                 patch.object(server.app.state.runtime.manager, "transcribe_with_vad", return_value=rows):
                response = await server.transcriptions(upload, language="zh", response_format="verbose_json", diarize=False)
        body = json.loads(response.body)
        self.assertEqual(body["text"], "first last")
        self.assertEqual(len(body["segments"]), 3)
        self.assertFalse(created[0].exists())

    async def test_realtime_commit_preserves_last_sample(self):
        from funasr_core.streaming import WSSession
        from unittest.mock import AsyncMock
        manager = ModelManager({"device": "cpu"})
        session = WSSession(manager, None, asyncio.Semaphore(1), asyncio.Semaphore(1), {})
        pcm = np.arange(42 * SAMPLE_RATE + 7, dtype=np.int16).tobytes()
        session.speech_frames = [pcm]
        session._asr = AsyncMock(return_value="speech")
        await session._commit_segment([])
        calls = session._asr.call_args_list
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[-1].args[0][-32:], pcm[-32:])
        self.assertTrue(all(len(c.args[0]) <= 20 * SAMPLE_RATE * 2 for c in calls))


if __name__ == "__main__":
    unittest.main()
