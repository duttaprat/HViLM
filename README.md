# HViLM: A Foundation Model for Viral Genomics

[![Paper](https://img.shields.io/badge/Paper-RECOMB%202026-blue)](https://github.com/duttaprat/HViLM)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-HViLM--base-yellow)](https://huggingface.co/duttaprat/HViLM-base)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HVUE-orange)](https://huggingface.co/datasets/duttaprat/HVUE)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HViLM (Human Virome Language Model)** is the first foundation model for comprehensive viral risk assessment through multi-task prediction of pathogenicity, host tropism, and transmissibility.

**Paper**: *HViLM: A Foundation Model for Viral Genomics Enables Multi-Task Prediction of Pathogenicity, Transmissibility, and Host Tropism*

**Authors**: Pratik Dutta, Jack Vaska, Pallavi Surana, Rekha Sathian, Max Chao, Zhihan Zhou, Han Liu, and Ramana V. Davuluri

---

## 🎯 Key Features

- 🦠 **Viral-specialized pre-training** on 5M sequences from 10.8M genomes spanning 45+ viral families
- 🎯 **Multi-task predictions**:
  - **Pathogenicity**: 95.32% average accuracy
  - **Host tropism**: 96.25% accuracy  
  - **Transmissibility**: 97.36% average accuracy
- 📊 **[HVUE Benchmark](https://huggingface.co/datasets/duttaprat/HVUE)**: 7 curated datasets (60K+ sequences)
- 🔍 **Interpretable**: Identifies transcription factor binding site mimicry
- ⚡ **Parameter-efficient**: LoRA fine-tuning (~0.3M trainable parameters)

---

## 📦 Installation

### **Option 1: Conda Environment (Recommended)**

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

### **Option 2: pip Only**

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### **1. Load Pre-trained Model**

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "duttaprat/HViLM-base",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "duttaprat/HViLM-base",
    trust_remote_code=True
)

# Get embeddings for a viral sequence
sequence = "ATGCGTACGTTAGCCGATCGATTACGCGTACGTAGCTAGCTAGCT"
inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch_size, seq_len, 768]

print(f"Embeddings shape: {embeddings.shape}")
```

### **2. Download HVUE Benchmark**

```python
from datasets import load_dataset

# Load specific task
host_tropism = load_dataset("duttaprat/HVUE", data_dir="Host_Tropism")
pathogenicity = load_dataset("duttaprat/HVUE", data_dir="Pathogenecity")
transmissibility = load_dataset("duttaprat/HVUE", data_dir="Transmissibility")
```

---

## 🔬 Fine-tuning HViLM

### **Prepare Your Data**

Your dataset should be organized as:

```
my_dataset/
├── train.csv
├── dev.csv
└── test.csv
```

Each CSV file should have two columns:
- `sequence`: Viral genomic sequence (DNA string)
- `label`: Binary label (0 or 1)

**Example:**
```csv
sequence,label
ATGCGTACGTTAGCCGAT...,1
GCTAGCTAGCTAGCTAGC...,0
```

### **Fine-tune on Your Data**

```bash
# Copy the template script
cp scripts/finetune_template.sh scripts/finetune_my_task.sh

# Edit the script to set your parameters
nano scripts/finetune_my_task.sh
```

**Key parameters to customize:**

```bash
export TASK_NAME="my_viral_classification_task"    # Your task name
export DATA_PATH="./data/my_dataset"                # Path to your data
export MAX_LENGTH=250                                # ~25% of sequence length
export NUM_EPOCHS=10                                 # Training epochs
export OUTPUT_DIR="./output/${TASK_NAME}"           # Output directory
```

**Run fine-tuning:**

```bash
bash scripts/finetune_my_task.sh
```

---

## 📊 Reproducing Paper Results

### **1. Download HVUE Benchmark**

```python
from datasets import load_dataset

# Download all benchmark datasets
host_tropism = load_dataset("duttaprat/HVUE", data_dir="Host_Tropism")
pathogenicity_cini = load_dataset("duttaprat/HVUE", data_dir="Pathogenecity")
transmissibility = load_dataset("duttaprat/HVUE", data_dir="Transmissibility")
```

### **2. Fine-tune on Specific Tasks**

```bash
# Pathogenicity classification (example)
bash scripts/reproduce_pathogenicity.sh

# Host tropism prediction
bash scripts/reproduce_host_tropism.sh

# Transmissibility assessment  
bash scripts/reproduce_transmissibility.sh
```

---

## 📁 Repository Structure

```
HViLM/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── finetune/
│   ├── train.py                # Standard fine-tuning script
│   ├── train_optuna.py         # Fine-tuning with Optuna hyperparameter search
│   └── evaluate.py             # Evaluation script
│
├── scripts/
│   ├── finetune_template.sh    # Template for fine-tuning
│   ├── reproduce_pathogenicity.sh
│   ├── reproduce_host_tropism.sh
│   └── reproduce_transmissibility.sh
│
├── data/
│   ├── sample_data/            # Sample sequences for testing (10-20 examples)
│   └── download_hvue.py        # Script to download HVUE benchmark
│
└── notebooks/
    └── quickstart_demo.ipynb   # Interactive tutorial
```

---

## 🎛️ Training Configuration

### **Standard Training (No Hyperparameter Search)**

```bash
export NUM_TRIALS=0  # Disable Optuna

bash scripts/finetune_template.sh
```

### **With Optuna Hyperparameter Search**

```bash
export NUM_TRIALS=20  # Run 20 Optuna trials

bash scripts/finetune_template.sh
```

Optuna will automatically search for:
- Learning rate (1e-5 to 1e-3)
- Number of epochs (8-20)
- Weight decay (1e-6 to 1e-2)
- LoRA rank (4, 8, 16)
- LoRA alpha (16, 32, 64)
- LoRA dropout (0.0-0.3)
- Warmup fraction (0.05-0.10)

Best hyperparameters are saved to: `${OUTPUT_DIR}/best_trial.json`

---

## 📈 Experiment Tracking with Weights & Biases

HViLM supports [Weights & Biases](https://wandb.ai) for experiment tracking.

### **Setup**

1. Create a W&B account: https://wandb.ai/signup
2. Login:
   ```bash
   wandb login
   ```

3. Configure in your fine-tuning script:
   ```bash
   export WANDB_PROJECT="my-project-name"
   export WANDB_ENTITY="my-username"  # Optional
   ```

### **View Results**

After training, view your results at: `https://wandb.ai/[username]/[project]`

W&B automatically tracks:
- Training/validation loss
- Accuracy, F1, MCC, Precision, Recall
- Confusion matrices
- Learning rate schedules
- GPU utilization

---

## 🎯 Performance Benchmarks

### **HVUE Benchmark Results**

| Task | Dataset | Sequences | Accuracy | F1 | MCC |
|------|---------|-----------|----------|-----|-----|
| **Pathogenicity** | CINI | 159 | 87.74% | 86.98 | 74.48 |
| | BVBRC-CoV | 18,066 | 98.26% | 98.26 | 96.52 |
| | BVBRC-Calici | 31,089 | 99.95% | 99.93 | 99.90 |
| | **Average** | **49,314** | **95.32%** | **95.06** | **90.30** |
| **Host Tropism** | VHDB | 9,428 | 96.25% | 91.34 | 91.24 |
| **Transmissibility** | Coronaviridae | ~3,000 | 97.45% | 97.37 | 93.43 |
| | Orthomyxoviridae | ~2,500 | 95.62% | 95.44 | 91.07 |
| | Caliciviridae | ~1,800 | 99.95% | 99.95 | 99.90 |
| | **Average** | **~7,300** | **97.36%** | **97.59** | **94.80** |

---

## 🔍 Interpretability

HViLM learns biologically meaningful patterns through attention mechanisms:

- **42 conserved motifs** identified in pathogenic coronaviruses
- **10 vertebrate transcription factors** targeted, including:
  - **Irf1** (immune evasion): 8 convergent motifs
  - **Foxq1** (epithelial tropism): Multiple motifs
  - **ZNF354A** (chromatin regulation): 6 motifs

This demonstrates molecular mimicry as a core pathogenicity mechanism.

---

## 💡 Tips for Best Results

### **Sequence Length**

- HViLM processes sequences up to **1000 base pairs**
- For longer sequences, segment into 1000bp chunks
- Set `MAX_LENGTH` to ~25% of your sequence length (BPE tokenization reduces length by ~4-5x)

### **Class Imbalance**

If your dataset is imbalanced:
- Use balanced sampling
- Adjust class weights
- Or use focal loss (see `train.py` for details)

### **GPU Memory**

If running out of memory:
- Reduce `TRAIN_BATCH` or `EVAL_BATCH`
- Increase `GRAD_ACC` to maintain effective batch size
- Use `fp16` training (enabled by default)

### **LoRA Configuration**

- Default: `r=8, alpha=16, dropout=0.1` works well for most tasks
- For more complex tasks: increase `r` to 16 or 32
- For small datasets: reduce `r` to 4 to prevent overfitting

---

## 📚 Citation

If you use HViLM in your research, please cite:

```bibtex
@article{dutta2025hvilm,
  title={HViLM: A Foundation Model for Viral Genomics Enables Multi-Task Prediction of Pathogenicity, Transmissibility, and Host Tropism},
  author={Dutta, Pratik and Vaska, Jack and Surana, Pallavi and Sathian, Rekha and Chao, Max and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V.},
  journal={Submitted to RECOMB},
  year={2025}
}
```

If you use DNABERT-2 (the base model), please also cite:

```bibtex
@article{zhou2023dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
  journal={ICLR},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Pratik Dutta**  
Senior Research Scientist  
Department of Biomedical Informatics  
Stony Brook University  

- **Email**: Pratik.Dutta@stonybrook.edu
- **Lab**: [Davuluri Lab](http://davulurilab.org)
- **Issues**: [GitHub Issues](https://github.com/duttaprat/HViLM/issues)

---

## 🙏 Acknowledgments

- Built upon [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) by Zhou et al.
- Pre-training data from [VIRION database](https://virion.verena.org)
- Benchmark datasets from [BV-BRC](https://www.bv-brc.org) and [Virus-Host DB](https://www.genome.jp/virushostdb/)
- Funded by NIH grants R01LM013722 and R21 Trailblazer Award

---

## ⭐ Star History

If you find HViLM useful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=duttaprat/HViLM&type=Date)](https://star-history.com/#duttaprat/HViLM&Date)
