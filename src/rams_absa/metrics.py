from __future__ import annotations

from collections import Counter


def decode_spans(label_ids: list[int], id2label: dict[int, str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(label_ids):
        l = id2label[label_ids[i]]
        if l.startswith("B-"):
            pol = l[2:]
            j = i + 1
            while j < len(label_ids) and id2label[label_ids[j]] == f"I-{pol}":
                j += 1
            spans.append((i, j, pol))
            i = j
        else:
            i += 1
    return spans


def aspect_sentiment_f1(golds: list[list[tuple[int, int, str]]], preds: list[list[tuple[int, int, str]]]) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for g, p in zip(golds, preds):
        gs = Counter(g)
        ps = Counter(p)
        common = gs & ps
        tp += sum(common.values())
        fp += sum((ps - gs).values())
        fn += sum((gs - ps).values())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}

