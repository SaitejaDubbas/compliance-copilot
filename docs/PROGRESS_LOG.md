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
