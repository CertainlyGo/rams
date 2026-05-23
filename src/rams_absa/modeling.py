from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForTokenClassification


def load_model(model_name: str, num_labels: int) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return ModelBundle(tokenizer=tokenizer, model=model)

