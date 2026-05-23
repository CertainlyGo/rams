from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .data import Sample, char_level_labels, get_label_list
from .metrics import aspect_sentiment_f1, decode_spans


class ABSADataset(Dataset):
    def __init__(self, samples: list[Sample], tokenizer, max_length: int, label2id: dict[str, int]):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clabels = char_level_labels(s)
        enc = self.tokenizer(
            s.sentence,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        labels = []
        for (st, ed) in offsets:
            if st == ed:
                labels.append(-100)
            elif st < len(clabels):
                labels.append(self.label2id.get(clabels[st], 0))
            else:
                labels.append(0)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "reliability": torch.tensor(s.reliability, dtype=torch.float),
        }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad1d(t, pad):
        out = torch.full((max_len,), pad, dtype=t.dtype)
        out[: len(t)] = t
        return out
    return {
        "input_ids": torch.stack([pad1d(x["input_ids"], 0) for x in batch]),
        "attention_mask": torch.stack([pad1d(x["attention_mask"], 0) for x in batch]),
        "labels": torch.stack([pad1d(x["labels"], -100) for x in batch]),
        "reliability": torch.stack([x["reliability"] for x in batch]),
    }


@dataclass
class TrainOutput:
    output_dir: Path
    best_f1: float


def _step_loss(outputs, reliabilities):
    # token loss per sample
    logits = outputs.logits
    labels = outputs.labels
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    valid = (labels != -100).float()
    per_sample = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    loss = (per_sample * reliabilities).mean()
    return loss


@torch.no_grad()
def evaluate(model, dataloader, device, id2label):
    model.eval()
    gold_spans = []
    pred_spans = []
    total = 0
    correct = 0
    for b in dataloader:
        inp = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        labels = b["labels"].to(device)
        out = model(input_ids=inp, attention_mask=mask)
        pred = out.logits.argmax(dim=-1)
        for i in range(inp.size(0)):
            gl = []
            pl = []
            for g, p, m in zip(labels[i].tolist(), pred[i].tolist(), mask[i].tolist()):
                if m == 0 or g == -100:
                    continue
                gl.append(g)
                pl.append(p)
                total += 1
                if g == p:
                    correct += 1
            gold_spans.append(decode_spans(gl, id2label))
            pred_spans.append(decode_spans(pl, id2label))
    met = aspect_sentiment_f1(gold_spans, pred_spans)
    met["acc"] = correct / (total + 1e-9)
    return met


@torch.no_grad()
def estimate_reliability(model, dataset, cfg: Config, device):
    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    reliabilities = []
    model.train()
    for b in dl:
        inp = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        probs = []
        for _ in range(cfg.dropout_passes):
            out = model(input_ids=inp, attention_mask=mask)
            probs.append(out.logits.softmax(dim=-1))
        stack = torch.stack(probs, dim=0)
        mean_prob = stack.mean(dim=0)
        conf, _ = mean_prob.max(dim=-1)
        stability = 1.0 - stack.var(dim=0).mean(dim=-1)
        # semantic and structural proxies kept lightweight for local reproducibility.
        semantic = conf
        structural = stability
        r = cfg.alpha * semantic + cfg.beta * structural + cfg.gamma * conf
        r = r.mean(dim=1).clamp(0.1, 1.0)
        reliabilities.extend(r.detach().cpu().tolist())
    return reliabilities


def train(
    model,
    tokenizer,
    train_samples: list[Sample],
    dev_samples: list[Sample],
    cfg: Config,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = get_label_list()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    for s in train_samples:
        s.reliability = max(0.1, min(1.0, s.reliability))
    train_ds = ABSADataset(train_samples, tokenizer, cfg.max_length, label2id)
    dev_ds = ABSADataset(dev_samples, tokenizer, cfg.max_length, label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Re-estimate reliability for synthetic samples with dropout-based stability.
    re = estimate_reliability(model, train_ds, cfg, device)
    for i, r in enumerate(re):
        train_samples[i].reliability = r
    train_ds = ABSADataset(train_samples, tokenizer, cfg.max_length, label2id)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * math.ceil(len(train_ds) / cfg.batch_size))
    warmup = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    best_f1 = -1.0
    global_step = 0
    log_lines = []
    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep+1}/{cfg.epochs}")
        for b in pbar:
            global_step += 1
            inp = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            labels_tensor = b["labels"].to(device)
            r = b["reliability"].to(device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels_tensor)
            out.labels = labels_tensor
            loss = _step_loss(out, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % cfg.eval_every_steps == 0:
                met = evaluate(model, dev_dl, device, id2label)
                line = (
                    f"step={global_step} "
                    f"dev_f1={met['f1']:.4f} dev_acc={met['acc']:.4f} loss={loss.item():.4f}"
                )
                log_lines.append(line)
                if met["f1"] > best_f1:
                    best_f1 = met["f1"]
                    model.save_pretrained(out_dir / "checkpoints")
                    tokenizer.save_pretrained(out_dir / "checkpoints")
    (out_dir / "train.log").write_text("\n".join(log_lines), encoding="utf-8")
    return TrainOutput(output_dir=out_dir, best_f1=best_f1)


def evaluate_checkpoint(ckpt_dir: Path, samples: list[Sample], cfg: Config):
    from .modeling import load_model

    labels = get_label_list()
    id2label = {i: l for i, l in enumerate(labels)}
    bundle = load_model(str(ckpt_dir), num_labels=len(labels))
    ds = ABSADataset(samples, bundle.tokenizer, cfg.max_length, {l: i for i, l in enumerate(labels)})
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model.to(device)
    met = evaluate(bundle.model, dl, device, id2label)
    return met


def write_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

