# CLAUDE.md — Project guide for Claude Code

## What we are building
**Compliance Copilot**: an automated contract-clause classifier and compliance-review
tool. It reads a contract, splits it into clauses, classifies each clause by type,
checks it against compliance rules, and produces a review report.

This is a portfolio project. It must be showcaseable on a resume, on GitHub, and as a
live deployed demo. Everything must use free / open-source tools (no paid services
beyond Claude Code itself).

## The three components (each maps to one resume line)
1. **Brain** — `Llama-3.2-1B-Instruct` fine-tuned with **QLoRA (PEFT)** to classify
   contract clauses. (Resume line: "fine-tuned Llama-3 with QLoRA/PEFT, +X% accuracy".)
2. **Front desk** — a **FastAPI** service with **async** endpoints, packaged in **Docker**.
   (Resume line: "scalable asynchronous pipelines with FastAPI and Docker".)
3. **Manager** — a **LangChain agent** that runs the end-to-end review workflow.
   (Resume line: "automated compliance workflows with LangChain agents".)

## Tech + free-tool choices (do not change without asking the user)
- Dataset: **LEDGAR** (public contract-clause dataset) via HuggingFace `datasets`.
- Model: `meta-llama/Llama-3.2-1B-Instruct` (small enough for free Colab GPU + free CPU serving).
- Fine-tuning: QLoRA (4-bit) + PEFT + HuggingFace `transformers` / `trl`, run in **Google Colab** (free T4).
- Adapter storage: **HuggingFace Hub** (free; LoRA adapter is a few MB).
- Serving: FastAPI + Uvicorn, async endpoints.
- Workflow: LangChain agent + tools.
- Deploy: **HuggingFace Spaces** (free Docker Space, CPU inference).
- Repo: GitHub.

## Roadmap (build phase by phase; explain each step simply before coding)
- [x] **Phase 1 — Data prep** (`data/prepare_data.py`): subset LEDGAR to top-10 clause
      types, balance classes, write train/val/test JSONL in chat format. DONE.
- [x] **Phase 2 — Fine-tune the brain** (`training/finetune_qlora.ipynb`): QLoRA fine-tune,
      AND measure baseline (zero-shot) vs fine-tuned accuracy so the "+X%" number is REAL.
      DONE — ran successfully in Colab. Accuracy 20.00% -> 94.33% (+74.33 pp), macro-F1
      0.1013 -> 0.9439 (+84.26 pp). Adapter pushed to
      `SaitejaDubbas/compliance-copilot-llama32-1b-lora`.
- [ ] **Phase 3 — Front desk** (`app/model.py`, `app/main.py`): load base model + adapter,
      async FastAPI `/classify` endpoint.
- [ ] **Phase 4 — Manager** (`app/agent.py`): LangChain agent that splits a document into
      clauses, classifies each via the model, applies simple compliance rules, writes a report.
- [ ] **Phase 5 — Docker** (`Dockerfile`): containerize the app.
- [ ] **Phase 6 — Deploy**: HuggingFace Spaces (Docker SDK), get a live public link.
- [ ] **Phase 7 — Polish**: README results table with real numbers, demo GIF, resume wording.

## Conventions
- The system prompt + user-prompt template live in `data/prepare_data.py`
  (`SYSTEM_PROMPT`, `build_user_prompt`). Training AND serving must reuse the EXACT same
  text, or the model gets confused. If serving code needs them, import/copy them verbatim.
- Keep the deployed model runnable on CPU (this is why we use the 1B model).
- The "+X% accuracy" claim must always be a measured number (base zero-shot vs fine-tuned
  on the held-out test split), never invented.
- Explain each new step in plain language before writing code (the user likes concrete
  analogies and step-by-step reasoning).

## Current status
Phase 1 is done (note: the LEDGAR dataset id changed from `lex_glue` to
`coastalcph/lex_glue` — the old unnamespaced alias no longer resolves on the Hub;
`data/prepare_data.py` and `training/finetune_qlora.ipynb` both use the new id).

Phase 2 is done. HF username: `SaitejaDubbas`. Adapter repo:
`SaitejaDubbas/compliance-copilot-llama32-1b-lora`. Three fixes were needed live in
Colab and are now baked into `training/finetune_qlora.ipynb` so a fresh run works
end to end: (1) TRL's `SFTConfig` renamed `max_seq_length` -> `max_length`; (2) the
T4 computes in bfloat16, so training must use `bf16=True, fp16=False` (fp16 crashed
with a BFloat16 gradient-unscale error); (3) the save/push cell now derives
`HF_USERNAME` via `huggingface_hub.whoami()` right before pushing instead of trusting
the placeholder in the config cell, so it can't 403 against a nonexistent namespace.
See `docs/PROGRESS_LOG.md` for details.

Next action: build Phase 3 — `app/model.py` + `app/main.py`, loading
`unsloth/Llama-3.2-1B-Instruct` plus the `SaitejaDubbas/compliance-copilot-llama32-1b-lora`
adapter behind an async FastAPI `/classify` endpoint.
