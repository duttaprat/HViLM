# HViLM: A Foundation Model for Viral Genomics

[![Paper](https://img.shields.io/badge/Paper-bioRxiv%202026-blue)](https://doi.org/10.64898/2026.03.18.712700)
[![Model](https://img.shields.io/badge/🤗%20Model-HViLM--base-yellow)](https://huggingface.co/duttaprat/HViLM-base)
[![Dataset](https://img.shields.io/badge/🤗%20Benchmark-HVUE--v2-orange)](https://huggingface.co/datasets/duttaprat/HVUE-v2)
[![Collection](https://img.shields.io/badge/🤗%20Collection-HViLM%20Family-blueviolet)](https://huggingface.co/collections/duttaprat/hvilm-human-virome-language-model)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HViLM (Human Virome Language Model)** is a genome language model adapted to virus sequences and evaluated on pathogenicity, host tropism, and transmissibility prediction tasks. HViLM was developed by continued pre-training of DNABERT-2 on approximately 5 million virus-derived sequence fragments from VIRION, representing approximately 9,000 virus species across more than 45 virus families.

**Paper**: [*HViLM: A Foundation Model for Viral Genomics Enables Multi-Task Prediction of Pathogenicity, Transmissibility, and Host Tropism*](https://doi.org/10.64898/2026.03.18.712700)  
**Authors**: Pratik Dutta, Jack Vaska, Pallavi Surana, Rekha Sathian, Max Chao, Zhihan Zhou, Han Liu, and Ramana V. Davuluri

> ⚠️ **HVUE v1 notice:** The original HVUE benchmark ([duttaprat/HVUE](https://huggingface.co/datasets/duttaprat/HVUE)) contained cross-split sequence similarity that can inflate held-out performance estimates. **[HVUE v2](https://huggingface.co/datasets/duttaprat/HVUE-v2)** replaces it with leakage-controlled, cluster-aware splits. All results reported here use HVUE v2.

> **Important distinction:** `HViLM-base` is the foundation model — it produces sequence embeddings but is not itself a classifier. `HViLM-Patho`, `HViLM-R0`, and `HViLM-Tropism` are the ready-to-use classifiers for the three HVUE v2 tasks.

---

## 🧬 HViLM Model Family

| Resource | Description | Link |
|----------|-------------|------|
| **HViLM-base** | Foundation model (DNABERT-2 + viral CPT) | [🤗 Model](https://huggingface.co/duttaprat/HViLM-base) |
| **HViLM-Patho** | Pathogenicity classifier | [🤗 Model](https://huggingface.co/duttaprat/HViLM-Patho) |
| **HViLM-R0** | Transmissibility classifier | [🤗 Model](https://huggingface.co/duttaprat/HViLM-R0) |
| **HViLM-Tropism** | Host tropism classifier | [🤗 Model](https://huggingface.co/duttaprat/HViLM-Tropism) |
| **HVUE-v2** | Leakage-controlled benchmark | [🤗 Dataset](https://huggingface.co/datasets/duttaprat/HVUE-v2) |

---

## 🚀 Quick Start

### What do you want to do?

| Goal | Model to use | Section |
|------|-------------|---------|
| Classify a virus as pathogenic or not | `HViLM-Patho` | [1. Predict](#1-predict-with-official-fine-tuned-models) |
| Classify transmissibility (R₀ < 1 vs ≥ 1) | `HViLM-R0` | [1. Predict](#1-predict-with-official-fine-tuned-models) |
| Classify human vs non-human tropism | `HViLM-Tropism` | [1. Predict](#1-predict-with-official-fine-tuned-models) |
| Get sequence embeddings for your own analysis | `HViLM-base` | [2. Embeddings](#2-extract-sequence-embeddings) |
| Train a classifier on your own labeled data | `HViLM-base` | [3. Fine-tune](#3-fine-tune-on-your-own-data) |
| Reproduce or benchmark against our results | HVUE-v2 | [4. Benchmark](#4-reproduce-benchmark-results) |
| Domain-adapt HViLM to a new virus family | `HViLM-base` | [5. Continue pretraining](#5-advanced-continue-pre-training) |

---

### 1. Predict with Official Fine-tuned Models

No training required. Load a task-specific model and classify sequences directly.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Choose your task:
#   "duttaprat/HViLM-Patho"    → pathogenic vs non-pathogenic
#   "duttaprat/HViLM-R0"       → R₀ < 1 vs R₀ ≥ 1
#   "duttaprat/HViLM-Tropism"  → human-tropic vs non-human-tropic

model_id = "duttaprat/HViLM-Patho"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)

sequence = "ATGCGTACGTTAGCCGATCGATTACGCGTACGTAGCTAGC"
inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=250)

with torch.no_grad():
    logits = model(**inputs).logits
    prediction = logits.argmax(dim=-1).item()
    label = model.config.id2label[prediction]

print(f"Prediction: {label}")
# Output: PATHOGENIC or NON_PATHOGENIC
```

**Evaluate on your own labeled data** (no fine-tuning — test generalization directly):

```python
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("my_sequences.csv")  # columns: sequence, label

predictions = []
for seq in df["sequence"]:
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=250)
    with torch.no_grad():
        pred = model(**inputs).logits.argmax(-1).item()
    predictions.append(pred)

print(classification_report(df["label"], predictions))
```

---

### 2. Extract Sequence Embeddings

Use `HViLM-base` as a feature extractor for clustering, visualization, similarity search, or as input to custom downstream models.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("duttaprat/HViLM-base", trust_remote_code=True)
model = AutoModel.from_pretrained("duttaprat/HViLM-base", trust_remote_code=True)

sequence = "ATGCGTACGTTAGCCGATCGATTACGCGTACGTAGCTAGCTAGCT"
inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state  # shape: [1, seq_len, 768]

# Option A: first-token representation
first_token_embedding = token_embeddings[:, 0, :]  # shape: [1, 768]

# Option B: attention-mask-aware mean pooling
attention_mask = inputs["attention_mask"].unsqueeze(-1)
mean_embedding = (
    (token_embeddings * attention_mask).sum(dim=1)
    / attention_mask.sum(dim=1).clamp(min=1)
)

# Option C: full token-level representations
print(f"First-token embedding: {first_token_embedding.shape}")
print(f"Mean embedding:        {mean_embedding.shape}")
print(f"Token embeddings:      {token_embeddings.shape}")
```

Pooling strategy should be chosen according to the downstream application; HViLM-base does not prescribe a single sequence-level pooling method.

---

### 3. Fine-tune on Your Own Data

Start from `HViLM-base` and train a classifier on your own labeled sequences.

**Prepare your data:**

```
my_dataset/
├── train.csv     # columns: sequence, label
├── dev.csv
└── test.csv
```

**Run fine-tuning:**

```bash
cp scripts/finetune_template.sh scripts/finetune_my_task.sh

# Edit the script to set:
#   DATA_PATH="./data/my_dataset"
#   TASK_NAME="my_task"
#   MAX_LENGTH=250
#   NUM_EPOCHS=5
#   OUTPUT_DIR="./output/my_task"

bash scripts/finetune_my_task.sh
```

**Training parameters:**

| Parameter | Typical setting | Notes |
|-----------|----------------|-------|
| `MAX_LENGTH` | 250 | BPE-token limit used for the released 1000-nt task configurations; tokenized length varies by sequence |
| `NUM_EPOCHS` | 5 | Early stopping with patience 3 on validation F1 |
| `LR` | 3e-5 | AdamW optimizer |
| `LORA_RANK` | 8 | LoRA rank; α = 16; ~0.3M trainable parameters |
| `EVAL_STRATEGY` | epoch | Prevents premature early stopping on large datasets |

These values are representative rather than universal. Users should tune them for their own task, dataset size, class balance, and sequence-length distribution.

---

### 4. Use HVUE v2 for Benchmarking

HVUE v2 can be used to evaluate HViLM, compare alternative genomic models, or work with the released benchmark splits.

```python
import pandas as pd

train = pd.read_csv(
    "hf://datasets/duttaprat/HVUE-v2/"
    "Pathogenicity/standard_capped_1000bp/train.csv"
)

dev = pd.read_csv(
    "hf://datasets/duttaprat/HVUE-v2/"
    "Pathogenicity/standard_capped_1000bp/dev.csv"
)

test = pd.read_csv(
    "hf://datasets/duttaprat/HVUE-v2/"
    "Pathogenicity/standard_capped_1000bp/test.csv"
)
```

Available tasks:

- `Pathogenicity`
- `Transmissibility`
- `Host_Tropism`

See the [HVUE v2 dataset card](https://huggingface.co/datasets/duttaprat/HVUE-v2) for the complete set of configurations and split definitions.

The current repository provides the fine-tuning template and Optuna-based training implementation. Additional reproduction and benchmark-pipeline utilities can be added as the revision package is finalized.

---

### 5. Advanced: Continue Pre-training

Domain-adapt `HViLM-base` to a specialized virus corpus (e.g., bacteriophages, plant viruses, arboviruses) before fine-tuning:

```
HViLM-base
    ↓  continued MLM on your unlabeled virus sequences
Adapted-HViLM
    ↓  supervised fine-tuning on your labeled task
Your task-specific model
```

This requires an MLM training setup analogous to the continued pre-training used to derive HViLM from DNABERT-2. The current public repository focuses on downstream fine-tuning rather than providing a turnkey continued-pretraining pipeline.

---

## 📊 HVUE v2 Benchmark Results

Results on the **1000 nt standard configuration** for each task. Models ranked by F1.

- **Pathogenicity:** `standard_capped_1000bp`
- **Transmissibility:** `standard_capped_1000bp`
- **Host Tropism:** `standard_95_1000bp`

### Pathogenicity Classification

| Model | Accuracy | F1 | MCC |
|-------|----------|----|-----|
| **HViLM (CPT)** | **92.39** | **91.32** | **83.10** |
| DNABERT-MB | 91.90 | 90.81 | 81.93 |
| DNABERT-2 (vanilla) | 91.31 | 90.04 | 80.66 |
| ViroDNABERT2 | 89.14 | 87.92 | 75.87 |
| 5-mer + KNN | 90.65 | 86.75 | 79.56 |
| NT-500M | 86.98 | 85.29 | 70.80 |

### Transmissibility Prediction

| Model | Accuracy | F1 | MCC |
|-------|----------|----|-----|
| **HViLM (CPT)** | 87.50 | **86.16** | 72.66 |
| DNABERT-MB | 87.52 | 86.15 | 72.44 |
| DNABERT-2 (vanilla) | 87.37 | 85.81 | 72.01 |
| ViroDNABERT2 | 84.69 | 82.90 | 66.05 |
| 6-mer + KNN | **87.70** | 82.26 | **72.91** |
| NT-500M | 84.07 | 81.67 | 64.48 |

### Host Tropism Prediction

| Model | Accuracy | F1 | MCC |
|-------|----------|----|-----|
| **HViLM (CPT)** | 96.49 | **74.49** | 48.99 |
| NT-500M | 95.63 | 65.68 | 31.57 |
| DNABERT-2 (balanced) | 89.76 | 64.82 | 38.25 |
| ViroDNABERT2 | 96.51 | 58.55 | 25.11 |
| 5-mer + KNN | **96.79** | 58.14 | **56.56** |
| DNABERT-2 (imbalanced) | 96.37 | 49.08 | 0.00 |

> **Key finding:** The benefit of virus-focused continued pre-training varies across tasks. HViLM shows modest F1 gains over vanilla DNABERT-2 on Pathogenicity and Transmissibility, while the largest improvement is observed on Host Tropism (+9.67 F1 over balanced DNABERT-2). HViLM also outperforms ViroDNABERT2 on Host Tropism (+15.94 F1); however, differences in pre-training corpora and training procedures prevent attributing this difference specifically to corpus scale or taxonomic breadth.

**Metric note:** HViLM achieves the highest F1 in all three primary configurations, but it does not achieve the highest value for every reported metric. The k-mer KNN baseline has the highest MCC for Host Tropism and the highest Accuracy/MCC for Transmissibility.

---

## 📦 Installation

```bash
conda create -n HViLM python=3.8
conda activate HViLM

git clone https://github.com/duttaprat/HViLM.git
cd HViLM

pip install -r requirements.txt
```

---

## 📁 Current Repository Structure

```text
HViLM/
├── README.md
├── requirements.txt
│
├── finetune/
│   └── train_optuna.py
│
└── scripts/
    └── finetune_template.sh
```

The structure above reflects the currently public repository. Additional revision and reproduction utilities can be added as they are finalized.

---

## 🗃️ HVUE v2 Benchmark

HVUE v2 is a complete rebuild of the original HVUE benchmark, eliminating train-test sequence overlap through a six-stage pipeline.

### What Changed from v1

| Issue in v1 | Fix in v2 |
|-------------|-----------|
| Chunk-level random splitting before clustering | Cluster-aware splitting **before** chunking |
| Near-duplicate sequences across train/test | MMseqs2 clustering at 95% (standard) or 70% (hard) identity |
| No temporal evaluation | Pre-2020 train / 2020+ test for Pathogenicity and Transmissibility |
| No leakage verification | Automated SHA-256 + MMseqs2 audit — **zero overlap confirmed** |

### Available Configurations

| Task | Configurations |
|------|---------------|
| Pathogenicity | `standard_capped_{500,1000,2000}bp` · `hard_capped_1000bp` · `standard_temporal_1000bp` |
| Transmissibility | `standard_capped_{500,1000,2000}bp` · `hard_capped_1000bp` · `standard_temporal_1000bp` |
| Host Tropism | `standard_95_{500,1000,2000}bp` · `hard_70_1000bp` |

*Temporal splitting unavailable for Host Tropism (source data lacks collection-date metadata).*

### Task Label Interpretation

| Task | Label 0 | Label 1 |
|------|---------|---------|
| Pathogenicity | Non-pathogenic | Pathogenic |
| Transmissibility | R₀ < 1 | R₀ ≥ 1 |
| Host Tropism | Non-human-tropic | Human-tropic |

Host Tropism labels describe human host association in the benchmark and should not be interpreted as evidence that a virus is restricted to a single host species.

📥 **Download:** [huggingface.co/datasets/duttaprat/HVUE-v2](https://huggingface.co/datasets/duttaprat/HVUE-v2)

---

## 🔍 Interpretability

Attention-guided motif analysis on pathogenic coronaviruses identified:

- **42 candidate motifs** (14–20 nt) associated with elevated attention scores
- Matches to motifs associated with **10 vertebrate transcription factors**, including:
  - **IRF1** — 8 candidate motif matches
  - **FOXQ1** — strongest reported enrichment
  - **ZNF354A** — 6 candidate motif matches
  - **BARHL2** — 5 candidate motif matches

These results generate hypotheses about possible host-regulatory mimicry by pathogenic viruses. They should not be interpreted as direct experimental evidence of molecular mimicry, causal regulation, immune evasion, epithelial tropism, chromatin regulation, or another biological mechanism.

---

## 📈 Experiment Tracking

```bash
export WANDB_PROJECT="HViLM-v2"
export WANDB_ENTITY="your-username"
```

Weights & Biases tracks training loss, F1, MCC, confusion matrices, learning rate, and GPU utilization.

---

## ⚠️ Limitations

- HVUE v2 substantially improves split independence relative to v1, but sequence-similarity thresholds cannot remove every possible biological relationship between training and evaluation viruses.
- Virus host association is inherently multi-host and context dependent. A sequence labeled as human-associated may also infect other species.
- Pathogenicity and transmissibility labels are inherited from source datasets and may simplify complex, context-dependent phenotypes.
- HViLM-base was initialized from DNABERT-2. Complete sequence-level exposure from DNABERT-2's original pre-training corpus cannot be reconstructed from currently available metadata.
- Attention-based motif analysis is hypothesis-generating and does not establish causal biological mechanisms.
- Small differences between closely matched models should not be interpreted as statistically meaningful without uncertainty estimates or repeated evaluation.

---

## 📚 Citation

```bibtex
@article{dutta2026hvilm,
  title={HViLM: A foundation model for viral genomics enables multi-task prediction of pathogenicity, transmissibility, and host tropism},
  author={Dutta, Pratik and Vaska, Jack and Surana, Pallavi and Sathian, Rekha and Chao, Max and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

```bibtex
@article{zhou2024dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
  journal={ICLR},
  year={2024}
}
```

---

## 📄 License

- **Code in this GitHub repository:** MIT License — see [LICENSE](LICENSE).
- **HViLM model weights on Hugging Face:** Apache License 2.0.

---

## 📧 Contact

**Pratik Dutta** — Pratik.Dutta@stonybrook.edu  
**Ramana V. Davuluri** — Ramana.Davuluri@stonybrookmedicine.edu  
Department of Biomedical Informatics, Stony Brook University  
[Davuluri Lab](https://davulurilab.github.io/) · [GitHub Issues](https://github.com/duttaprat/HViLM/issues)

---

## 🙏 Acknowledgments

- Built upon [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) by Zhou et al.
- Continued pre-training data from [VIRION database](https://virion.verena.org)
- Benchmark source data from [BV-BRC](https://www.bv-brc.org) and [Virus-Host DB](https://www.genome.jp/virushostdb/)

---

⭐ If you find HViLM useful, please star the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=duttaprat/HViLM&type=Date)](https://star-history.com/#duttaprat/HViLM&Date)
