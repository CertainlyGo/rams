from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

POLARITIES = ["positive", "neutral", "negative"]


@dataclass
class Aspect:
    start: int
    end: int
    target: str
    polarity: str


@dataclass
class Sample:
    sid: str
    sentence: str
    aspects: list[Aspect]
    source: str = "gold"
    reliability: float = 1.0


def get_label_list() -> list[str]:
    labels = ["O"]
    for p in POLARITIES:
        labels.append(f"B-{p}")
        labels.append(f"I-{p}")
    return labels


def load_json_samples(path: str | Path) -> list[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: list[Sample] = []
    for r in rows:
        aspects = [
            Aspect(
                start=int(a["from"]),
                end=int(a["to"]),
                target=a["target"],
                polarity=a["polarity"],
            )
            for a in r.get("aspects", [])
            if a.get("polarity") in POLARITIES
        ]
        out.append(Sample(sid=str(r["ID"]), sentence=r["sentence"], aspects=aspects))
    return out


def dataset_paths(dataset: str, data_root: str | Path) -> dict[str, Path]:
    base = Path(data_root) / dataset
    return {
        "train": base / "train_all.json",
        "dev": base / "dev_all.json",
        "test": base / "test_all.json",
        "sample2": base / "sample2_all.json",
        "sample5": base / "sample5_all.json",
    }


def load_split(dataset: str, split: str, data_root: str | Path = "data") -> list[Sample]:
    paths = dataset_paths(dataset, data_root)
    return load_json_samples(paths[split])


def load_fewshot(dataset: str, shot: int, seed: int, data_root: str | Path = "data") -> list[Sample]:
    split = f"sample{shot}"
    paths = dataset_paths(dataset, data_root)
    if split in paths and paths[split].exists():
        return load_json_samples(paths[split])
    # Fallback: deterministic sampling from train.
    all_train = load_json_samples(paths["train"])
    rng = random.Random(seed)
    k = max(1, int(len(all_train) * shot / 100))
    return rng.sample(all_train, k=k)


def char_level_labels(sample: Sample) -> list[str]:
    labels = ["O"] * len(sample.sentence)
    for a in sample.aspects:
        if a.start < 0 or a.end > len(sample.sentence) or a.start >= a.end:
            continue
        labels[a.start] = f"B-{a.polarity}"
        for i in range(a.start + 1, a.end):
            labels[i] = f"I-{a.polarity}"
    return labels


def flatten_samples(batches: Iterable[list[Sample]]) -> list[Sample]:
    out: list[Sample] = []
    for b in batches:
        out.extend(b)
    return out

