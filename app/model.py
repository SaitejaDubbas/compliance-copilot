"""Phase 3 -- the brain. Loads the base model + LoRA adapter once at import time
and exposes classify_clause() for the FastAPI layer to call.
"""

import difflib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Copied verbatim from data/prepare_data.py -- must match training exactly, or
# the model sees unfamiliar phrasing at inference time and gets confused.
SYSTEM_PROMPT = (
    "You are a compliance assistant. You read a single contract clause and reply "
    "with the one clause type it belongs to, and nothing else."
)


def build_user_prompt(clause_text: str, labels: list) -> str:
    label_menu = ", ".join(labels)
    return (
        f"Classify the following contract clause into exactly one of these types: "
        f"{label_menu}.\n\n"
        f"Clause:\n{clause_text}\n\n"
        f"Clause type:"
    )


BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_REPO = "SaitejaDubbas/compliance-copilot-llama32-1b-lora"
LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "labels.json"

LABELS = json.loads(LABELS_PATH.read_text())

print(f"Loading tokenizer for {BASE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading base model {BASE_MODEL} on CPU (float32) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print(f"Applying LoRA adapter {ADAPTER_REPO} ...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

print("Model ready.")


def parse_prediction(raw_text: str, labels: list) -> str:
    """Turn whatever text the model generated into the closest known label."""
    cleaned = raw_text.strip()
    for label in labels:
        if cleaned.lower() == label.lower():
            return label
    for label in labels:
        if label.lower() in cleaned.lower():
            return label
    close = difflib.get_close_matches(cleaned, labels, n=1, cutoff=0.0)
    return close[0] if close else labels[0]


@torch.no_grad()
def classify_clause(text: str) -> dict:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, LABELS)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1] :]
    raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return {"label": parse_prediction(raw, LABELS), "raw": raw}
