from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import load_config
from .data import flatten_samples, load_fewshot, load_split
from .modeling import load_model
from .synthesis import synthesize_samples
from .trainer import evaluate_checkpoint, train, write_metrics


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lap", "res", "res15", "res16"])
    parser.add_argument("--shot", type=int, required=True, choices=[2, 5])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_gold = load_fewshot(args.dataset, args.shot, args.seed)
    dev = load_split(args.dataset, "dev")
    synth = synthesize_samples(train_gold, cfg.synthesis_multiplier, args.seed)
    all_train = flatten_samples([train_gold, synth])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / args.dataset / f"{args.shot}shot" / ts

    labels_count = 7
    bundle = load_model(cfg.model_name, num_labels=labels_count)
    out = train(bundle.model, bundle.tokenizer, all_train, dev, cfg, out_dir)
    metrics = {"best_dev_f1": out.best_f1, "train_size": len(all_train), "gold_size": len(train_gold)}
    write_metrics(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))


def eval_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lap", "res", "res15", "res16"])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_split(args.dataset, args.split)
    met = evaluate_checkpoint(Path(args.ckpt), samples, cfg)
    result_path = Path(args.ckpt).parent / f"{args.split}_metrics.json"
    write_metrics(result_path, met)
    print(json.dumps(met, ensure_ascii=False))


def synthesize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lap", "res", "res15", "res16"])
    parser.add_argument("--shot", type=int, required=True, choices=[2, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = load_fewshot(args.dataset, args.shot, args.seed)
    syn = synthesize_samples(samples, args.multiplier, args.seed)
    out = []
    for s in syn:
        out.append(
            {
                "ID": s.sid,
                "sentence": s.sentence,
                "source": s.source,
                "reliability": s.reliability,
                "aspects": [
                    {"from": a.start, "to": a.end, "target": a.target, "polarity": a.polarity}
                    for a in s.aspects
                ],
            }
        )
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(p))

