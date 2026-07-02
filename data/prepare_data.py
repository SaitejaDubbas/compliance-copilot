"""
Compliance Copilot -- Phase 1: Data Preparation
================================================

Goal of this file (one sentence):
    Take a big public dataset of real contract clauses and turn it into small,
    clean files we can feed to our model in Phase 2.

Why LEDGAR?
    LEDGAR is a public dataset of contract provisions ("clauses"), where each
    clause is labelled with its type (e.g. "Confidentiality", "Governing Laws",
    "Terminations"). This is exactly the "what kind of clause is this / is it
    compliant" problem a compliance team faces every day. It is public and
    license-friendly, so we can safely showcase it (unlike private company data).

Where to run this:
    In Google Colab (free) or on your own laptop. It needs internet the first
    time so it can download the dataset from the Hugging Face Hub.

    Install the one thing it needs:
        pip install -U datasets

Output (all written into this ./data folder):
    labels.json   -> the list of clause types we keep
    train.jsonl   -> training examples (chat format, used to teach the model)
    val.jsonl     -> validation examples (chat format, used to watch progress)
    test.jsonl    -> held-out examples (raw text + label, used to score accuracy)
"""

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Settings you are allowed to play with
# ---------------------------------------------------------------------------
TOP_K_LABELS = 10           # keep only the K most common clause types
MAX_PER_LABEL_TRAIN = 400   # cap training examples per label (keeps training fast + free)
MAX_PER_LABEL_EVAL = 60     # cap validation/test examples per label
SEED = 42
OUT_DIR = Path(__file__).resolve().parent   # writes next to this script (the data/ folder)

# This is the "job description" we hand the model. IMPORTANT: the SAME text must
# be used again when we serve the model in Phase 3, otherwise it gets confused.
# That is why we keep it here in one place and reuse it everywhere.
SYSTEM_PROMPT = (
    "You are a compliance assistant. You read a single contract clause and reply "
    "with the one clause type it belongs to, and nothing else."
)


def build_user_prompt(clause_text: str, labels: list) -> str:
    """Turn a clause + the menu of allowed answers into the question we ask the model."""
    label_menu = ", ".join(labels)
    return (
        f"Classify the following contract clause into exactly one of these types: "
        f"{label_menu}.\n\n"
        f"Clause:\n{clause_text}\n\n"
        f"Clause type:"
    )
# ---------------------------------------------------------------------------

random.seed(SEED)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/5  Downloading LEDGAR from the Hugging Face Hub ...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")

    # The dataset stores labels as numbers (0, 1, 2, ...). This list turns a
    # number back into a human name, e.g. 12 -> "Confidentiality".
    label_names = ds["train"].features["label"].names
    print(f"          The full dataset has {len(label_names)} clause types.")

    print("Step 2/5  Finding the most common clause types ...")
    counts = Counter(ds["train"]["label"])            # how often each label appears
    top_ids = [label_id for label_id, _ in counts.most_common(TOP_K_LABELS)]
    kept_labels = [label_names[i] for i in top_ids]   # the names we keep
    id_to_name = {i: label_names[i] for i in top_ids}
    print(f"          Keeping the top {TOP_K_LABELS}: {kept_labels}")

    # Save the label list so every other part of the project agrees on it.
    (OUT_DIR / "labels.json").write_text(json.dumps(kept_labels, indent=2))

    def collect(split_name: str, cap_per_label: int):
        """Grab up to cap_per_label examples for each label so classes stay balanced."""
        buckets = {i: [] for i in top_ids}
        for row in ds[split_name]:
            lid = row["label"]
            if lid in buckets and len(buckets[lid]) < cap_per_label:
                buckets[lid].append(row["text"])
        pairs = [(text, id_to_name[lid]) for lid, texts in buckets.items() for text in texts]
        random.shuffle(pairs)
        return pairs

    print("Step 3/5  Selecting + balancing examples ...")
    train_pairs = collect("train", MAX_PER_LABEL_TRAIN)
    val_pairs = collect("validation", MAX_PER_LABEL_EVAL)
    test_pairs = collect("test", MAX_PER_LABEL_EVAL)
    print(f"          train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    print("Step 4/5  Writing training + validation files (chat format) ...")

    def write_chat(path: Path, pairs):
        with path.open("w") as f:
            for text, label in pairs:
                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text, kept_labels)},
                        {"role": "assistant", "content": label},
                    ]
                }
                f.write(json.dumps(example) + "\n")

    write_chat(OUT_DIR / "train.jsonl", train_pairs)
    write_chat(OUT_DIR / "val.jsonl", val_pairs)

    print("Step 5/5  Writing the test file (raw text + label, for scoring) ...")
    with (OUT_DIR / "test.jsonl").open("w") as f:
        for text, label in test_pairs:
            f.write(json.dumps({"text": text, "label": label}) + "\n")

    print("\nDone. Files written into:", OUT_DIR)
    print("Next: open the Phase 2 fine-tuning notebook.")


if __name__ == "__main__":
    main()
