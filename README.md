---
title: Compliance Copilot
emoji: 🐳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Space-blue?logo=huggingface)](https://saitejadubbas-compliance-copilot.hf.space/docs)

# Compliance Copilot

**Automated contract-clause classification and compliance-review workflows, powered by a fine-tuned Llama-3 model.**

Compliance Copilot reads a contract, breaks it into individual clauses, classifies each clause by type, checks it against a set of compliance rules, and produces a review report — all through a single API call or a web demo.

> Built entirely with free and open-source tools. Trained on the public **LEDGAR** contract-clause dataset (no private data used).

---

## What it does

Three components work together, like a small compliance team:

| Component | Role | Tech |
|-----------|------|------|
| **The brain** | Reads a clause and predicts its type | Llama-3.2-1B fine-tuned with QLoRA (PEFT) |
| **The front desk** | Serves predictions fast, handles many requests at once | FastAPI (async) + Docker |
| **The manager** | Runs the end-to-end review workflow and writes the report | LangChain agent |

```
        LangChain Agent  ->  FastAPI service  ->  Fine-tuned Llama-3
         (workflow)            (async API)          (QLoRA adapter)
```

## Results

Measured on a held-out test split of LEDGAR (600 clauses, 10 clause types). Fine-tuning raised accuracy from 20% to 94%:

| Setup | Accuracy | Macro-F1 |
|-------|----------|----------|
| Base Llama-3.2-1B (zero-shot) | 20.00% | 0.1013 |
| Fine-tuned with QLoRA | 94.33% | 0.9439 |
| **Improvement** | **+74.33 pp** | **+84.26 pp** |

*(Numbers are measured on the held-out test split, not estimated. Adapter: [SaitejaDubbas/compliance-copilot-llama32-1b-lora](https://huggingface.co/SaitejaDubbas/compliance-copilot-llama32-1b-lora).)*

## Tech stack

- **Model / training:** Llama-3.2-1B-Instruct, QLoRA (4-bit), PEFT, Hugging Face `transformers` + `trl`
- **Serving:** FastAPI, Uvicorn, async endpoints
- **Workflow:** LangChain agent + tools
- **Packaging & deploy:** Docker, Hugging Face Spaces (free tier)
- **Data:** LEDGAR (via the `datasets` library)

## Project structure

```
compliance-copilot/
├── data/
│   └── prepare_data.py       # Phase 1: build train/val/test from LEDGAR
├── training/
│   └── finetune_qlora.ipynb  # Phase 2: QLoRA fine-tuning + accuracy eval  (added next)
├── app/
│   ├── model.py              # loads model + adapter, runs inference        (added next)
│   ├── main.py               # FastAPI async endpoints                      (added next)
│   ├── agent.py              # LangChain compliance-review agent            (added next)
│   └── ui.py                 # simple web demo                              (added next)
├── Dockerfile                # Phase 5                                      (added next)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Build the dataset
pip install -U datasets
python data/prepare_data.py

# 2. Fine-tune (in Google Colab, free GPU) -- see training/finetune_qlora.ipynb

# 3. Run the API locally
pip install -r requirements.txt
uvicorn app.main:app --reload

# 4. Try it
curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
     -d '{"clause": "The parties agree to keep all shared information confidential."}'
```

## Live demo

Deployed on Hugging Face Spaces: **https://saitejadubbas-compliance-copilot.hf.space** ([Swagger UI](https://saitejadubbas-compliance-copilot.hf.space/docs))

## License

Code released under the MIT License. LEDGAR data belongs to its original authors; the Llama-3.2 model is used under Meta's community license.
