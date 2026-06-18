# JSALT 2026 — Continuous Speech Recognition Lab

A hands-on ASR lab built for the [JSALT 2026](https://www.clsp.jhu.edu/workshops/26-workshop/) summer workshop.

Students train a character-level and BPE-level CTC ASR model using a pretrained [HuBERT](https://arxiv.org/abs/2106.07447) encoder and the mini-LibriSpeech dataset.

## What you will implement

| Exercise | Description |
|---|---|
| `SpeechEncoder.forward` | Pass raw waveforms through HuBERT + linear projection; handle encoder freezing during warmup |
| `CTCLoss.forward` | Wrap PyTorch's `F.ctc_loss` with the correct log-softmax + transpose |
| `ctc_greedy_decode` | Collapse blanks and repeated tokens from CTC frame predictions |
| `decode_data` | Run the full dev-set through the model and collect hypotheses / references |
| Training loop | Warmup LR schedule, mixed-precision training, per-epoch validation and WER/CER reporting |
| SentencePiece training | Train and load a BPE vocabulary on LibriSpeech transcripts |
| `BPETokenizer` | Drop-in tokenizer wrapping SentencePiece with the same interface as `CharacterTokenizer` |
| BPE training | Re-run the full training pipeline with the BPE tokenizer |

Hidden `show solution` cells are provided for each exercise (click to expand in Colab).

The `CharacterTokenizer` class is fully implemented and serves as a reference.

## Setup

Open `JSALT_2026_CSR_Lab.ipynb` in [Google Colab](https://colab.research.google.com/) with a GPU runtime.

All required packages are installed by the first code cell:

```
torch  transformers  lhotse  sentencepiece  editdistance  jiwer
```

Data (mini-LibriSpeech: `train-clean-5`, `dev-clean-2`, `test-clean`) is downloaded automatically.

## Structure

```
JSALT-2026-CSR-Lab/
└── JSALT_2026_CSR_Lab.ipynb   # Exercise notebook
```
