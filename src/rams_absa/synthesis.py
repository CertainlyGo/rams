from __future__ import annotations

import random
import re
from copy import deepcopy

from .data import Sample

_LEXICON = {
    "good": "nice",
    "great": "excellent",
    "bad": "poor",
    "slow": "sluggish",
    "fast": "quick",
    "love": "like",
    "hate": "dislike",
}


def _replace_lexicon(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        key = re.sub(r"[^a-z]", "", w.lower())
        if key in _LEXICON and rng.random() < 0.35:
            repl = _LEXICON[key]
            out.append(repl if w.islower() else repl.capitalize())
        else:
            out.append(w)
    return " ".join(out)


def _add_suffix(text: str, rng: random.Random) -> str:
    prefixes = [" overall.", " in my view.", " honestly.", " for me."]
    if rng.random() < 0.5:
        return f"{text}{rng.choice(prefixes)}"
    return text


def semantic_reconstruct(sample: Sample, rng: random.Random) -> Sample:
    # Keep original aspect offsets stable for this lightweight local baseline.
    s = deepcopy(sample)
    s.sentence = _replace_lexicon(s.sentence, rng)
    s.source = "sr"
    return s


def structure_constrained(sample: Sample, rng: random.Random) -> Sample:
    # Constrained rewrite that does not change aspect boundaries.
    s = deepcopy(sample)
    s.sentence = _add_suffix(s.sentence, rng)
    s.source = "sc"
    return s


def synthesize_samples(samples: list[Sample], multiplier: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    syn: list[Sample] = []
    for samp in samples:
        for _ in range(multiplier):
            syn.append(semantic_reconstruct(samp, rng))
            syn.append(structure_constrained(samp, rng))
    return syn
