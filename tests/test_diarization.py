import unittest
from unittest.mock import patch

import numpy as np

from funasr_core.audio import SAMPLE_RATE
from funasr_core.diarization import speaker_windows, label_segments
from funasr_core.model_manager import ModelManager


class DiarizationTests(unittest.TestCase):
    def test_overlapping_vad_does_not_inflate_voice_coverage(self):
        self.assertEqual(speaker_windows(6 * SAMPLE_RATE, [(0, 2000), (0, 2000)]), [])

    def test_trailing_low_voice_window_is_excluded(self):
        windows = speaker_windows(12 * SAMPLE_RATE, [(0, 9500)])
        self.assertEqual(windows, [(0, 6 * SAMPLE_RATE), (3 * SAMPLE_RATE, 9 * SAMPLE_RATE)])

    def test_short_voiced_recording_has_one_window(self):
        self.assertEqual(speaker_windows(SAMPLE_RATE, [(0, 1000)]), [(0, SAMPLE_RATE)])
        self.assertEqual(speaker_windows(0, []), [])

    def test_labels_follow_overlap_and_uncovered_tail_inherits_voice(self):
        rows = [{"start": 0, "end": 6}, {"start": 6, "end": 12},
                {"start": 12, "end": 15}]
        count = label_segments(rows, [(0, 6 * SAMPLE_RATE), (6 * SAMPLE_RATE, 12 * SAMPLE_RATE)], [9, 4])
        self.assertEqual(count, 2)
        self.assertEqual([r["speaker"] for r in rows], [0, 1, 1])

    def test_analysis_does_not_depend_on_transcription_splits(self):
        manager = ModelManager({"device": "cpu"})
        audio = np.ones(18 * SAMPLE_RATE, dtype=np.float32)
        for boundaries in ([0, 3, 7, 9, 18], [0, 6, 12, 18]):
            rows = [{"start": a, "end": b} for a, b in zip(boundaries, boundaries[1:])]
            with patch.object(manager, "vad_offline_segments", return_value=[(0, 18000)]), \
                 patch.object(manager, "speaker_embedding", return_value=np.array([1., 0.])) as embedding:
                self.assertEqual(manager.assign_speakers(rows, audio), 1)
            self.assertEqual([len(c.args[0]) for c in embedding.call_args_list],
                             [6 * SAMPLE_RATE] * 5 + [3 * SAMPLE_RATE])
            self.assertTrue(all(r["speaker"] == 0 for r in rows))

    def test_automatic_mode_keeps_two_distinct_voices(self):
        manager = ModelManager({"device": "cpu"})
        rows = [{"start": 0, "end": 9}, {"start": 9, "end": 18}]
        with patch.object(manager, "vad_offline_segments", return_value=[(0, 18000)]), \
             patch.object(manager, "speaker_embedding", side_effect=[
                 np.array([1., 0.])] * 3 + [np.array([0., 1.])] * 3):
            self.assertEqual(manager.assign_speakers(rows, np.ones(18 * SAMPLE_RATE)), 2)
        self.assertEqual([r["speaker"] for r in rows], [0, 1])

    def test_missing_voice_or_invalid_embeddings_fall_back(self):
        manager = ModelManager({"device": "cpu"})
        for vad, emb in [([], None), ([(0, 1000)], np.array([np.nan])),
                         ([(0, 1000)], np.zeros(2))]:
            with patch.object(manager, "vad_offline_segments", return_value=vad), \
                 patch.object(manager, "speaker_embedding", return_value=emb):
                rows = [{"start": 0, "end": 1}]
                self.assertEqual(manager.assign_speakers(rows, np.ones(SAMPLE_RATE)), 1)
                self.assertEqual(rows[0]["speaker"], 0)


if __name__ == "__main__":
    unittest.main()
