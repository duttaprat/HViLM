# HViLM: A Foundation Model for Viral Genomics

[![Paper](https://img.shields.io/badge/Paper-mSystems%202026-blue)](https://github.com/duttaprat/HViLM)
[![Model](https://img.shields.io/badge/🤗%20Model-HViLM--base-yellow)](https://huggingface.co/duttaprat/HViLM-base)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HVUE--v2-orange)](https://huggingface.co/datasets/duttaprat/HVUE-v2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HViLM (Human Virome Language Model)** is a genome language model adapted to virus sequences and evaluated on pathogenicity, host tropism, and transmissibility prediction tasks. HViLM was developed by continued pre-training of DNABERT-2 on approximately 5 million virus-derived sequence fragments from VIRION, representing approximately 9,000 virus species across more than 45 virus families.

**Paper**: *HViLM: A Foundation Model for Viral Genomics Enables Multi-Task Prediction of Pathogenicity, Transmissibility, and Host Tropism*  
**Journal**: mSystems (2026)  
**Authors**: Pratik Dutta, Jack Vaska, Pallavi Surana, Rekha Sathian, Max Chao, Zhihan Zhou, Han Liu, and Ramana V. Davuluri

> ⚠️ **HVUE v1 notice:** The original HVUE benchmark ([duttaprat/HVUE](https://huggingface.co/datasets/duttaprat/HVUE)) was removed due to substantial cross-split sequence similarity that can inflate estimates of held-out generalization. **[HVUE v2](https://huggingface.co/datasets/duttaprat/HVUE-v2)** replaces those splits with leakage-controlled, cluster-aware benchmarks. Results reported using HVUE v2 evaluations reported here.

---

## 🎯 Key Features

- 🦠 **Virus-focused continued pre-training** on ~5M virus-derived sequence fragments from VIRION, spanning ~9,000 virus species and 45+ families
- 🎯 **Multi-task evaluation on HVUE v2**
  - **Pathogenicity:** F1 91.32, MCC 83.10
  - **Transmissibility:** F1 86.16, MCC 72.66
  - **Host Tropism:** F1 74.49, MCC 48.99
- 📊 **HVUE v2 benchmark:** rebuilt using cluster-aware splitting before sequence chunking, with explicit cross-split leakage auditing
- 🔬 **Continued pre-training benefit varies by task difficulty:** the largest gain is observed for Host Tropism
- 🔍 **Interpretability:** attention-guided motif analysis identifies candidate virus sequence motifs with similarity to vertebrate transcription-factor binding motifs
- ⚡ **Parameter-efficient adaptation:** LoRA fine-tuning with ~0.3M trainable parameters per task

---

## 📊 HVUE v2 Benchmark Results

Results below correspond to the **1000 bp standard configuration for each task**:

- **Pathogenicity:** `standard_capped_1000bp`
- **Transmissibility:** `standard_capped_1000bp`
- **Host Tropism:** `standard_95_1000bp`

Models are ranked by F1 within each task.

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
| **HViLM (CPT)** | **87.50** | **86.16** | **72.66** |
| DNABERT-MB | 87.52 | 86.15 | 72.44 |
| DNABERT-2 (vanilla) | 87.37 | 85.81 | 72.01 |
| ViroDNABERT2 | 84.69 | 82.90 | 66.05 |
| 6-mer + KNN | 87.70 | 82.26 | 72.91 |
| NT-500M | 84.07 | 81.67 | 64.48 |

### Host Tropism Prediction

| Model | Accuracy | F1 | MCC |
|-------|----------|----|-----|
| **HViLM (CPT)** | **96.49** | **74.49** | **48.99** |
| NT-500M | 95.63 | 65.68 | 31.57 |
| DNABERT-2 (balanced) | 89.76 | 64.82 | 38.25 |
| ViroDNABERT2 | 96.51 | 58.55 | 25.11 |
| 5-mer + KNN | 96.79 | 58.14 | 56.56 |
| DNABERT-2 (imbalanced) | 96.37 | 49.08 | 0.00 |

> **Interpretation:** The benefit of virus-focused continued pre-training is most pronounced on the more challenging Host Tropism benchmark. HViLM improves F1 by 9.67 points relative to the balanced DNABERT-2 baseline and by 15.94 points relative to ViroDNABERT2. These results are consistent with broader virus-focused continued pre-training contributing to improved performance on difficult downstream tasks, although differences in model training data and procedures prevent attributing the effect to pre-training corpus breadth alone.

**Balanced DNABERT-2** refers to the class-balancing strategy used in the revised Host Tropism experiment. See the corresponding reproduction script for the exact sampling/loss configuration.

---

## 📦 Installation

```bash
# Create conda environment
conda create -n HViLM python=3.8
conda activate HViLM

# Clone repository
git clone https://github.com/duttaprat/HViLM.git
cd HViLM

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Load the Pre-trained HViLM Base Model

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "duttaprat/HViLM-base",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "duttaprat/HViLM-base",
    trust_remote_code=True
)

sequence = "ATGCGTACGTTAGCCGATCGATTACGCGTACGTAGCTAGCTAGCT"

inputs = tokenizer(
    sequence,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(f"Embeddings shape: {embeddings.shape}")
```

### 2. Load HVUE v2

```python
import pandas as pd

train = pd.read_csv(
    "hf://datasets/duttaprat/HVUE-v2/"
    "Pathogenicity/standard_capped_1000bp/train.csv"
)

validation = pd.read_csv(
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

See the [HVUE v2 dataset card](https://huggingface.co/datasets/duttaprat/HVUE-v2) for all released configurations.

---

## 🔬 Fine-tuning HViLM

### Prepare Your Data

```text
my_dataset/
├── train.csv
├── dev.csv
└── test.csv
```

Each CSV contains:

- `sequence`: nucleotide sequence
- `label`: binary task label (`0` or `1`)

### Run Fine-tuning

```bash
cp scripts/finetune_template.sh scripts/finetune_my_task.sh

# Edit task-specific parameters in the copied script:
# DATA_PATH
# TASK_NAME
# MAX_LENGTH
# NUM_EPOCHS
# OUTPUT_DIR

bash scripts/finetune_my_task.sh
```

### Representative Training Parameters

| Parameter | Typical setting | Notes |
|-----------|-----------------|-------|
| `NUM_EPOCHS` | 5 | Early stopping with patience 3 |
| `LR` | 3e-5 | AdamW |
| `LORA_RANK` | 8 | LoRA rank; α = 16 |
| `EVAL_STRATEGY` | epoch | Prevents premature early stopping on large datasets |
| `MAX_LENGTH` | task/config specific | Use the corresponding reproduction script for the released benchmark setting |

The exact configuration used for each reported experiment is defined in the task-specific reproduction scripts.

---

## 📁 Repository Structure

```text
HViLM/
├── README.md
├── requirements.txt
├── LICENSE
│
├── finetune/
│   ├── train.py
│   ├── train_optuna.py
│   └── evaluate.py
│
├── scripts/
│   ├── finetune_template.sh
│   ├── reproduce_pathogenicity.sh
│   ├── reproduce_host_tropism.sh
│   └── reproduce_transmissibility.sh
│
├── pipeline/
│   ├── build_hvue_v2.sh
│   └── leakage_audit.py
│
├── data/
│   ├── sample_data/
│   └── download_hvue_v2.py
│
└── notebooks/
    └── quickstart_demo.ipynb
```

---

## 🗃️ HVUE v2 Benchmark

HVUE v2 is a complete rebuild of the original HVUE benchmark designed to reduce supervised train/test non-independence caused by exact and high-similarity sequence overlap.

### What Changed from HVUE v1

| Issue in HVUE v1 | HVUE v2 correction |
|------------------|--------------------|
| Chunk-level random splitting before clustering | Cluster-aware splitting before chunking |
| Exact and high-similarity sequences across supervised partitions | Sequence clustering before partition assignment |
| Limited evaluation of difficult out-of-distribution settings | More stringent hard-split configurations |
| No temporal evaluation | Temporal configurations for Pathogenicity and Transmissibility |
| No explicit leakage verification | Automated exact-match, cluster-overlap, and sequence-similarity audits |

The released benchmark was audited under the predefined sequence-similarity criteria used in the HVUE v2 construction pipeline. See the repository pipeline scripts and dataset documentation for the exact clustering and audit definitions.

### Available Configurations

| Task | Configurations |
|------|----------------|
| Pathogenicity | `standard_capped_500bp`, `standard_capped_1000bp`, `standard_capped_2000bp`, `hard_capped_1000bp`, `standard_temporal_1000bp` |
| Transmissibility | `standard_capped_500bp`, `standard_capped_1000bp`, `standard_capped_2000bp`, `hard_capped_1000bp`, `standard_temporal_1000bp` |
| Host Tropism | `standard_95_500bp`, `standard_95_1000bp`, `standard_95_2000bp`, `hard_70_1000bp` |

📥 **Dataset:** [duttaprat/HVUE-v2](https://huggingface.co/datasets/duttaprat/HVUE-v2)

### Task Label Interpretation

- **Pathogenicity:** binary classification according to the pathogenicity labels provided by the benchmark source data
- **Transmissibility:** binary classification according to the transmissibility labels used in the benchmark construction
- **Host Tropism:** classification according to human-host association labels in the source dataset; these labels should not be interpreted as evidence that a virus is restricted to a single host species

---

## 🔍 Interpretability

HViLM uses attention-guided motif analysis to identify sequence patterns associated with model predictions.

The analysis identified:

- **42 conserved candidate motifs** of 14–20 nucleotides in the analyzed coronavirus sequences
- matches to motifs associated with **10 vertebrate transcription factors**
- examples including:
  - **IRF1** — 8 convergent motif matches
  - **FOXQ1** — strongest reported enrichment
  - **ZNF354A** — 6 motif matches
  - **BARHL2** — 5 motif matches

These results identify candidate virus sequence motifs with similarity to vertebrate transcription-factor binding motifs and generate hypotheses about possible host-regulatory mimicry. They should not be interpreted as direct experimental evidence of a molecular mechanism.

---

## 📈 Experiment Tracking

```bash
export WANDB_PROJECT="HViLM-v2"
export WANDB_ENTITY="your-username"
```

Weights & Biases can be used to track training loss, F1, MCC, confusion matrices, learning rate, and GPU utilization.

---

## ⚠️ Limitations

- HVUE v2 substantially improves split independence relative to HVUE v1, but sequence-similarity thresholds cannot remove every possible biological relationship between training and evaluation viruses.
- Virus host association is inherently multi-host and context dependent. A sequence labeled as human-associated may also infect other species.
- Pathogenicity and transmissibility labels are inherited from source datasets and may simplify complex, context-dependent phenotypes.
- HViLM-base was initialized from DNABERT-2. Complete sequence-level exposure from DNABERT-2's original pre-training corpus cannot be reconstructed from currently available metadata.
- Attention-based motif analysis is hypothesis-generating and does not establish causal biological mechanisms.
- Performance should therefore be interpreted within the scope of each benchmark configuration rather than as a universal measure of virus phenotype prediction.

---

## 📚 Citation

If you use HViLM or HVUE v2 in your research, please cite:

```bibtex
@article{dutta2026hvilm,
  title={HViLM: A Foundation Model for Viral Genomics Enables Multi-Task
         Prediction of Pathogenicity, Transmissibility, and Host Tropism},
  author={Dutta, Pratik and Vaska, Jack and Surana, Pallavi and Sathian, Rekha
          and Chao, Max and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V.},
  journal={mSystems},
  year={2026}
}
```

If you use DNABERT-2, please also cite:

```bibtex
@article{zhou2024dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik
          and Davuluri, Ramana and Liu, Han},
  journal={ICLR},
  year={2024}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📧 Contact

**Pratik Dutta** — Pratik.Dutta@stonybrook.edu  
**Ramana V. Davuluri** — Ramana.Davuluri@stonybrookmedicine.edu  
Department of Biomedical Informatics, Stony Brook University  

[Davuluri Lab](https://davulurilab.github.io/) · [Issues](https://github.com/duttaprat/HViLM/issues)

---

## 🙏 Acknowledgments

- Built upon [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) by Zhou et al.
- Continued pre-training data derived from the [VIRION database](https://virion.verena.org)
- Downstream benchmark source data include resources from [BV-BRC](https://www.bv-brc.org) and [Virus-Host DB](https://www.genome.jp/virushostdb/)

---

⭐ If you find HViLM useful, please star the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=duttaprat/HViLM&type=Date)](https://star-history.com/#duttaprat/HViLM&Date)
