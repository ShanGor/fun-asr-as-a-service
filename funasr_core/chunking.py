"""Bounded ASR windows. VAD suggests boundaries but never removes audio."""
from dataclasses import dataclass
import re

import numpy as np

from .audio import SAMPLE_RATE


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    overlap: bool = False


def plan_chunks(audio, vad_segments, target_ms=15000, max_ms=20000,
                silence_ms=300, overlap_ms=500):
    """Return sample-exact windows covering all input, each <= max_ms.

    Split at the first qualifying VAD pause, independently of target_ms.
    Only speech exceeding max_ms needs a forced cut near target_ms.
    Forced cuts overlap; confirmed silence cuts do not.
    """
    maximum = max(1000, int(max_ms)) * SAMPLE_RATE // 1000
    target = min(maximum, max(1000, int(target_ms)) * SAMPLE_RATE // 1000)
    overlap = min(maximum // 4, max(0, int(overlap_ms)) * SAMPLE_RATE // 1000)
    regions = []
    for beg, end in sorted(vad_segments):
        beg, end = max(0, int(beg) * 16), min(len(audio), int(end) * 16)
        if end <= beg:
            continue
        if regions and beg <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
        else:
            regions.append((beg, end))
    pauses = [(end + beg) // 2 for (_, end), (beg, _) in zip(regions, regions[1:])
              if beg - end >= max(0, silence_ms) * 16]
    result = []
    start = 0
    overlapping = False
    while start < len(audio):
        low = start + SAMPLE_RATE
        high = min(start + maximum, len(audio))
        candidates = [p for p in pauses if low <= p <= high and p < len(audio)]
        forced = not candidates
        if candidates:
            end = candidates[0]
        elif len(audio) - start <= maximum:
            result.append(Chunk(start, len(audio), overlapping))
            break
        else:
            # Avoid searching the whole recording or copying a long waveform.
            radius = SAMPLE_RATE
            left = max(low, start + target - radius)
            right = min(high, start + target + radius)
            frame = SAMPLE_RATE // 50
            positions = range(left, max(left + 1, right - frame + 1), frame)
            end = min(positions, key=lambda p: (
                float(np.mean(np.square(audio[p:min(p + frame, high)]))),
                abs(p - start - target)))
        result.append(Chunk(start, end, overlapping))
        start = end - overlap if forced else end
        overlapping = forced and overlap > 0
    return result


def reconcile_overlap(previous, current):
    """Conservative exact boundary match, limited to 12 tokens/characters.

    Never fuzzy-match or remove an entire chunk; uncertainty preserves text.
    Chinese characters are units, while Latin words remain whole units.
    """
    pattern = r"[\u3400-\u9fff]|[^\W_]+"
    before = list(re.finditer(pattern, previous.lower()))[-12:]
    after = list(re.finditer(pattern, current.lower()))[:12]
    for count in range(min(len(before), len(after)), 1, -1):
        if [m.group() for m in before[-count:]] == [m.group() for m in after[:count]]:
            rest = current[after[count - 1].end():].lstrip(" ,.!?，。！？、;；:")
            if rest:
                return rest
    return current
