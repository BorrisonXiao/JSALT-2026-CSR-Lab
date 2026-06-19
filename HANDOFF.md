# HANDOFF — JSALT 2026 CSR Lab

**Date:** 2026-06-19  
**Repo:** https://github.com/BorrisonXiao/JSALT-2026-CSR-Lab  
**HF Hub:** https://huggingface.co/Borrison/jsalt26-csr-lab  
**Cluster path:** `/export/fs06/cxiao7/jsalt26/tutorials/JSALT-2026-CSR-Lab/`  
(same inode as `/home/cxiao7/jsalt26/tutorials/JSALT-2026-CSR-Lab/`)

---

## Goal

Turn the JSALT 2026 Continuous Speech Recognition notebook
(`JSALT_2026_CSR_Lab.ipynb`) into a polished interactive lab:
- Add discussion questions and hidden instructor notes throughout
- Provide pre-trained weights on HuggingFace Hub so students don't
  need to train from scratch on Colab
- Train a BPE tokenizer model on the cluster (A100) and upload it

---

## Current State

### Notebook (`main` branch, commit `58c28f2`)

7 new question cells + 5 hidden instructor-note cells (`#@title … { display-mode: "form" }`):

| Cell ID | Section | Questions |
|---|---|---|
| `cell-q01` + `cell-note01` | During char training | CNN freeze, Transformer freeze, CTC blank, 100% WER at init (CAT example) |
| `cell-q02` + `cell-note02` | After char results | WER vs CER, error patterns, LM rescoring |
| `cell-q03` + `cell-note03` | Before BPE training | Warmup schedule bug, phoneme→subword difficulty, fix rationale |
| `cell-q04` + `cell-note04` | BPE tokenization | BPE advantages, BPE disadvantages, loss curve comparison |
| `cell-q05` + `cell-note05` | Final comparison | Which model wins, BPE-specific errors, fair comparison, unigram extension |

Additional cells added:
- `cell-load-weights` (after training setup): loads char model from HF Hub
- `cell-bpe-warmstart`: loads BPE partial model from HF Hub
- `cell-bpe-hyp`: sets `warmup_bpe = 500` to fix LR schedule for BPE

### HuggingFace Hub (`Borrison/jsalt26-csr-lab`)

| File | Description |
|---|---|
| `speech_encoder_char.pt` | Fully trained char model (HuBERT-base + 768→33 linear, train-clean-5) |
| `speech_encoder_bpe_partial.pt` | **Partially trained** BPE model (HuBERT-base + 768→100 linear) — not converged |

Both are plain `OrderedDict` state dicts, load with `model.load_state_dict(torch.load(...))`.

### Running A100 Job

**Job 1673055** is currently running on `e01` (A100, 80 GB):
- Script: `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/run.sh`
- Training: 30 epochs, save every 5, `train-clean-100` + `dev-clean`
- Encoder frozen for first 500 steps, then unfrozen
- `warmup_bpe = 500`, `max_duration = 300 s/batch`
- Log: `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/logs/csr-bpe-1673055.out`
- Checkpoints saved to: `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/checkpoints/`
- Final weights: `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/speech_encoder_bpe.pt`

As of handoff the job just started training (manifest preparation completed, epoch 1 not yet logged). There is a harmless `FutureWarning` about `torch.cuda.amp.autocast` deprecation — does not affect training.

---

## What Worked

- **Manifest caching** in the training script (`load_or_prepare()`): saves lhotse manifests to `bpe_train/manifests/*.jsonl.gz` after first run; subsequent runs skip the ~25-min file scan.
- **`CUDA_HOME` export** in `run.sh`: needed because DeepSpeed (a transformers dependency in the `omni` env) tries to compile CUDA ops at import time.
- **HF Hub upload/download** via `huggingface_hub`: `hf_hub_download` caches locally, no auth needed for students.
- **`warmup_bpe = 500`** override: the original `warmup = 4000` exceeds the total training steps on `train-clean-5` (~3888), so the model never exits warmup. Fixed by introducing `warmup_bpe` in `cell-bpe-hyp`.
- **Conda env `omni`** for cluster training: has torch 2.6, transformers; lhotse, jiwer, sentencepiece installed with `pip install` into it.

## What Didn't Work

- **Warm-starting BPE from char weights**: user explicitly rejected this — the BPE model should start from pretrained HuBERT only (not the char fine-tuned weights). Removed.
- **`torch==1.12.1` env (notebook's pip install cell)**: too old for the cluster; use `omni` env instead.
- **`train-clean-5` / `dev-clean-2` on cluster**: these mini splits aren't in `/export/fs06/cxiao7/LibriSpeech`. Using `train-clean-100` + `dev-clean` instead.
- **Three cancelled job attempts before 1673055**:
  - 1673052: `CUDA_HOME` not set → import crash at `from transformers import HubertModel`
  - 1673053: cancelled after user changed stop criterion (was "CER < 100%")
  - 1673054: cancelled after user changed epochs (was 20) and removed char warm-start

---

## Next Steps

### Immediate (when job 1673055 finishes)

1. **Upload final BPE weights to HF Hub:**
   ```bash
   source /home/cxiao7/research/discrete/espnet_meili/tools/miniconda/etc/profile.d/conda.sh
   conda activate omni
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_file(
       path_or_fileobj='/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/speech_encoder_bpe.pt',
       path_in_repo='speech_encoder_bpe.pt',
       repo_id='Borrison/jsalt26-csr-lab',
       repo_type='model',
       commit_message='Add fully-trained BPE model (30 epochs, train-clean-100)',
   )
   "
   ```
   (The training script does this automatically if it finishes cleanly — check the log first.)

2. **Update notebook** `cell-bpe-warmstart`: change `FILENAME = "speech_encoder_bpe_partial.pt"` → `"speech_encoder_bpe.pt"`.

3. **Commit and push** the notebook update.

### Follow-up

- Add the `bpe_train/` directory to the git repo (currently untracked) if you want the training script versioned.
- The `.pt` file `speech_encoder_full.pt` in the repo root is untracked and large — either add it to `.gitignore` or delete it (it's already on HF Hub as `speech_encoder_char.pt`).
- The `tmp*/` directories in the repo root are junk — add to `.gitignore`.
- Consider updating the `pip install` cell in the notebook (`torch==1.12.1` is very old; students on modern Colab may get conflicts).

---

## Key File Locations

| What | Where |
|---|---|
| Notebook | `/export/fs06/cxiao7/jsalt26/tutorials/JSALT-2026-CSR-Lab/JSALT_2026_CSR_Lab.ipynb` |
| Char weights (local) | `/export/fs06/cxiao7/jsalt26/tutorials/JSALT-2026-CSR-Lab/speech_encoder_full.pt` |
| BPE partial weights (local) | `/export/fs06/cxiao7/jsalt26/tutorials/JSALT-2026-CSR-Lab/asr_model_bpe.pt` |
| Training script | `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/train_bpe.py` |
| Run script | `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/run.sh` |
| Cached manifests | `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/manifests/` |
| Training logs | `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/logs/csr-bpe-1673055.{out,err}` |
| Checkpoints (every 5 ep) | `/export/jsalt26/tutorials/JSALT-2026-CSR-Lab/bpe_train/checkpoints/` |
