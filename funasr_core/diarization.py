"""Speaker analysis windows independent of ASR phrase boundaries."""
import numpy as np

from .audio import SAMPLE_RATE


def speaker_windows(length, vad_segments):
    """Use 6 s context every 3 s; require at least 70% VAD voice coverage.

    Merge VAD intervals before measuring coverage so overlapping detections
    cannot make an instrumental or silent window eligible.
    """
    regions = []
    for start, end in sorted(vad_segments):
        start = max(0, int(start * SAMPLE_RATE / 1000))
        end = min(length, int(end * SAMPLE_RATE / 1000))
        if end <= start:
            continue
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
        else:
            regions.append((start, end))
    windows = []
    for start in range(0, length, 3 * SAMPLE_RATE):
        end = min(length, start + 6 * SAMPLE_RATE)
        if end - start < min(length, 3 * SAMPLE_RATE):
            continue
        voiced = sum(max(0, min(end, b) - max(start, a)) for a, b in regions)
        if voiced >= .7 * (end - start):
            windows.append((start, end))
    return windows


def label_segments(segments, windows, labels):
    """Vote by window overlap; uncovered segments inherit the nearest window."""
    ordered_labels = {}
    for seg in segments:
        start, end = seg["start"] * SAMPLE_RATE, seg["end"] * SAMPLE_RATE
        scores = {}
        for (a, b), label in zip(windows, labels):
            scores[label] = scores.get(label, 0) + max(0, min(end, b) - max(start, a))
        if max(scores.values()) > 0:
            label = max(scores, key=scores.get)
        else:
            center = (start + end) / 2
            nearest = int(np.argmin([abs(center - (a + b) / 2) for a, b in windows]))
            label = labels[nearest]
        seg["speaker"] = ordered_labels.setdefault(label, len(ordered_labels))
    return len(ordered_labels)
