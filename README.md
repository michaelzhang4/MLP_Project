# Risk-Aware Dynamic Inference for Financial Sentiment Analysis

Three dynamic inference strategies for accelerating FinBERT on the Financial PhraseBank dataset, validated across multiple backbones (FinBERT, DeBERTa-v3, NeoBERT) and datasets (PhraseBank, FiQA).

## Project Structure

```
├── Experiment_A_VADER_Cascade/        # VADER–FinBERT cascade with learned MLP gate
├── Experiment_B_Knowledge_Distillation/ # DistilBERT KD + DeBERTa/NeoBERT backbone experiments
├── Experiment_C_Multi_Exit/           # Multi-exit FinBERT with PABEE-inspired early exiting
├── FinancialPhraseBank-v1.0/          # Dataset (included in repo)
├── deberta_finetuned/                 # Fine-tuned DeBERTa weights (not in repo — see below)
├── neobert_finetuned/                 # Fine-tuned NeoBERT weights (not in repo — see below)
└── requirements.txt
```

## Setup

```bash
# Create conda environment
conda create -n mlp python=3.10 -y
conda activate mlp

# Install dependencies
pip install -r requirements.txt
```

**GPU required** for Experiments B and C. Tested on NVIDIA T4 / A100 (Google Cloud).

## Running Experiments

### Experiment A: VADER–FinBERT Cascade

Uses FinBERT from HuggingFace Hub directly (no local weights needed).

```bash
cd Experiment_A_VADER_Cascade
python hybrid_sentiment.py
```

Results are saved to `results/`.

### Experiment B: Knowledge Distillation

**FinBERT + DistilBERT (main KD experiment):**
```bash
cd Experiment_B_Knowledge_Distillation
python hybrid_sentiment_v2.py --results-dir results_v2
```

**DeBERTa backbone** (fine-tunes from scratch, then runs KD + multi-exit):
```bash
python run_deberta_experiments.py --phase all --results-dir results_deberta
```

**NeoBERT backbone:**
```bash
python run_neobert_experiments.py --phase all --results-dir results_neobert
```

Both `run_deberta_experiments.py` and `run_neobert_experiments.py` support `--phase` to run individual stages:
- `finetune` — fine-tune the backbone on PhraseBank (saves weights to `../deberta_finetuned/` or `../neobert_finetuned/`)
- `kd` — run KD routing experiment (requires fine-tuned weights)
- `multi-exit` — run multi-exit experiment (requires fine-tuned weights)
- `all` — run all phases sequentially (default)

### Experiment C: Multi-Exit FinBERT

```bash
cd Experiment_C_Multi_Exit
python hybrid_sentiment_v5_multi_exit_router.py --results-dir results_full
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `ProsusAI/finbert` | HuggingFace model or local path |
| `--exit-layers` | `4,8` | Comma-separated exit layer indices |
| `--patience` | `1` | PABEE patience (1 or 2) |
| `--batch-size` | `32` | Inference batch size |
| `--head-epochs` | `30` | Exit head training epochs |
| `--target-retention` | `0.98` | Target accuracy retention |
| `--cost-lambda` | `0.25` | Cost-aware threshold lambda |

## Swapping in a Different Model

### Using a custom NeoBERT in Experiment B

The NeoBERT path is set at the top of `run_neobert_experiments.py` (line 104):

```python
NEOBERT_SAVE_DIR = _PROJECT_ROOT / "neobert_finetuned"
```

**Option 1 — Replace the directory:** Place your fine-tuned NeoBERT weights in `neobert_finetuned/` at the project root. The directory must contain `config.json`, `model.safetensors`, and tokenizer files. Then skip the fine-tune phase:

```bash
python run_neobert_experiments.py --phase kd       # KD routing only
python run_neobert_experiments.py --phase multi-exit # Multi-exit only
```

**Option 2 — Edit the path:** Change `NEOBERT_SAVE_DIR` on line 104 to point to your model directory.

### Using a custom model in Experiment C

Experiment C accepts any HuggingFace-compatible model via `--model-name`:

```bash
# With a local fine-tuned model
python hybrid_sentiment_v5_multi_exit_router.py \
    --model-name ../neobert_finetuned \
    --exit-layers 7,14,21 \
    --results-dir results_neobert_multi_exit

# With a HuggingFace Hub model
python hybrid_sentiment_v5_multi_exit_router.py \
    --model-name ProsusAI/finbert \
    --exit-layers 4,8
```

**Important:** Adjust `--exit-layers` to match your model's depth. Guidelines:
- 12-layer models (FinBERT, DeBERTa): `--exit-layers 4,8`
- 28-layer models (NeoBERT): `--exit-layers 7,14,21`

### Using a custom DeBERTa

Same as NeoBERT — edit `DEBERTA_SAVE_DIR` on line 62 of `run_deberta_experiments.py`, or replace `deberta_finetuned/`.

## Regenerating Fine-Tuned Weights

The fine-tuned model weights (~700–900 MB each) are not included in the repo. To regenerate:

```bash
# DeBERTa (saves to ../deberta_finetuned/)
cd Experiment_B_Knowledge_Distillation
python run_deberta_experiments.py --phase finetune

# NeoBERT (saves to ../neobert_finetuned/)
python run_neobert_experiments.py --phase finetune
```

Fine-tuning takes ~5–10 minutes on a T4 GPU.

## Google Cloud VM Setup

```bash
# On a GCP VM with GPU (e.g., n1-standard-4 + T4)
git clone <repo-url> && cd MLP_Project
pip install -r requirements.txt

# Fine-tune backbones first (one-time)
cd Experiment_B_Knowledge_Distillation
python run_neobert_experiments.py --phase finetune
python run_deberta_experiments.py --phase finetune

# Then run any experiment
python run_neobert_experiments.py --phase all
```
