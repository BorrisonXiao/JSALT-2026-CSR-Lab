# JSALT 2026 — Continuous Speech Recognition Lab

A hands-on ASR lab built for the [JSALT 2026](https://www.clsp.jhu.edu/the-12th-frederick-jelinek-memorial-summer-workshop-on-speech-and-language-technology/) summer workshop.

Students train a character-level and BPE-level CTC ASR model using a pretrained [HuBERT](https://arxiv.org/abs/2106.07447) encoder and the mini-LibriSpeech dataset.

## What you will implement

The core exercises focus on the model and CTC machinery. Each has a hidden **show solution** cell in Colab.

| Exercise | What to implement |
|---|---|
| `SpeechEncoder.forward` | Freeze/unfreeze the encoder, run HuBERT, project with a linear layer |
| `CTCLoss.forward` | Log-softmax + transpose + `F.ctc_loss` |
| `ctc_greedy_decode` | Blank removal and consecutive-repeat collapsing |
| `decode_data` | Full dev-set decoding loop |
| Training loop | Mixed-precision training with warmup LR schedule, per-epoch eval |

The BPE section gives `BPETokenizer` mostly pre-filled — only `encode_text` and `inverse` are left as short fill-ins (no show solution). The SentencePiece training step and the BPE training pipeline are provided in full.

The `CharacterTokenizer` class is fully implemented and serves as a reference for all tokenizer methods.

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
