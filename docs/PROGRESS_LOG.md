# Progress Log

## Phase 2 — Fine-tune the brain (QLoRA)

Ran `training/finetune_qlora.ipynb` end to end in Google Colab (T4 GPU).

**Result:** fine-tuning raised accuracy from 20% to 94% on the 600-clause held-out
test split.

| Metric | Baseline (zero-shot) | Fine-tuned | Improvement |
|--------|----------------------|------------|-------------|
| Accuracy | 20.00% | 94.33% | +74.33 pp |
| Macro-F1 | 0.1013 | 0.9439 | +84.26 pp |

Adapter pushed to: [`SaitejaDubbas/compliance-copilot-llama32-1b-lora`](https://huggingface.co/SaitejaDubbas/compliance-copilot-llama32-1b-lora)

### Fixes needed live in Colab

The notebook as originally written didn't run clean on a fresh Colab session. Three
issues surfaced and were fixed in the saved notebook so future runs work end to end:

1. **`SFTConfig` argument rename.** Current TRL removed `max_seq_length` in favor of
   `max_length`. Fixed in the training-args cell.
2. **bf16/fp16 mismatch on the T4.** Training crashed with
   `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not
   implemented for 'BFloat16'` because the model computes in bfloat16 but the
   trainer was configured for fp16 mixed precision. Fixed by setting `bf16=True,
   fp16=False` in `SFTConfig`.
3. **Push-to-hub 403.** The save/push cell relied on a hardcoded `HF_USERNAME`
   placeholder, which didn't match the logged-in account and got rejected. Fixed by
   deriving the username from `huggingface_hub.whoami()` immediately before pushing,
   so the push always targets the account actually logged into the notebook.

## Phase 4.5 — RAG chatbot over a contract

Added `app/rag.py` plus `POST /rag/index` and `POST /rag/ask` to the FastAPI app.

Retrieval is entirely local and free: documents are chunked with
`RecursiveCharacterTextSplitter`, embedded with `sentence-transformers/all-MiniLM-L6-v2`
via `HuggingFaceEmbeddings`, and indexed in an in-memory FAISS store. Only the final
answer-generation step calls out, to `ChatGroq` (`llama-3.3-70b-versatile`,
temperature 0), grounded strictly in the retrieved chunks via a system prompt that
requires it to reply "I don't know based on this contract." when the answer isn't in
the retrieved context.

**Result:** `/rag/index` indexed a sample NDA-style contract; `/rag/ask` answered
"How long does the confidentiality obligation last?" with "five (5) years" plus the
supporting source chunk (HTTP 200), and correctly refused an out-of-context question
about a late-payment penalty that wasn't in the document.

## Phase 5 + 6 — Docker + Hugging Face Spaces deployment

Containerized the app (`Dockerfile`, `.dockerignore`) and deployed it to a Hugging
Face Docker Space.

**Live at: https://saitejadubbas-compliance-copilot.hf.space**

All five endpoints were confirmed working directly on the deployed Space (not just
locally):

- `/health` -> ok
- `/classify` -> "Governing Laws"
- `/review`
- `/rag/index` -> indexed successfully
- `/rag/ask` -> grounded "five (5) years" answer via Groq (HTTP 200)

The `GROQ_API_KEY` Space secret works correctly at runtime — confirmed by the
`/rag/ask` result above, which requires a live call to Groq for answer generation.
No `.env` file is present in the deployed image; the key is injected purely as a
runtime environment variable by the Space, exactly as designed.
