# HViLM: A Foundation Model for Viral Genomics

[![Paper](https://img.shields.io/badge/Paper-RECOMB%202026-blue)]()
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-HViLM--base-yellow)](https://huggingface.co/duttaprat/HViLM-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview
Brief description from your abstract (2-3 sentences)

## Installation
```bash
# Clone repository
git clone https://github.com/duttaprat/HViLM.git
cd HViLM

# Create conda environment
conda create -n hvilm python=3.9
conda activate hvilm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
# Example code showing how to load model and run inference
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("duttaprat/HViLM-base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("duttaprat/HViLM-base")

sequence = "ATGCGTACGT..."
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
```

## Datasets (HVUE Benchmark)
- **Pathogenicity**: CINI, BVBRC-CoV, BVBRC-Calici
- **Host Tropism**: VHDB (9,428 sequences)
- **Transmissibility**: Coronaviridae, Orthomyxoviridae, Caliciviridae

Download links or instructions here

## Model Variants
- **HViLM-base**: Pre-trained foundation model
- **HViLM-Patho**: Fine-tuned for pathogenicity (95.32% avg accuracy)
- **HViLM-Tropism**: Fine-tuned for host tropism (96.25% accuracy)
- **HViLM-R0**: Fine-tuned for transmissibility (97.36% avg accuracy)

## Reproducing Results
```bash
# Training
python train_pathogenicity.py --config configs/patho_config.yaml

# Evaluation
python evaluate.py --model HViLM-Patho --dataset CINI
```



## License
MIT License

## Contact
- Pratik Dutta: pratik.dutta@stonybrook.edu
- Lab: [Davuluri Lab](http://davulurilab.org)
```

### **2. Required Files to Add** ⭐ CRITICAL

**Code Files (MUST HAVE):**
```
HViLM/
├── README.md (comprehensive, as above)
├── requirements.txt
├── LICENSE (MIT)
├── setup.py or pyproject.toml
│
├── data/
│   ├── sample_data/ ⭐ CRITICAL - Minimal test dataset
│   │   ├── test_sequences.fasta (10-20 viral sequences)
│   │   └── test_labels.csv
│   └── download_hvue.py (script to download full datasets)
│
├── models/
│   ├── __init__.py
│   ├── hvilm_model.py
│   └── lora_config.py
│
├── scripts/
│   ├── train_pathogenicity.py
│   ├── train_tropism.py
│   ├── train_transmissibility.py
│   ├── evaluate.py
│   └── inference_demo.py ⭐ CRITICAL - Simple demo
│
├── configs/
│   ├── patho_config.yaml
│   ├── tropism_config.yaml
│   └── r0_config.yaml
│
├── notebooks/
│   └── demo.ipynb ⭐ HELPFUL - Walkthrough example
│
└── tests/
    └── test_inference.py
```
