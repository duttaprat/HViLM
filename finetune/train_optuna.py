import os
import sys
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

import optuna
import joblib  # For saving the Optuna study
import wandb

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")  # Removed trailing comma
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    n_trials: int = field(default=20, metadata={"help": "Number of hyperparameter search trials."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # Increase the CSV field size limit
        csv.field_size_limit(sys.maxsize)

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute_metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Exclude padding tokens (-100)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    # Compute Confusion Matrix
    cm = confusion_matrix(valid_labels, valid_predictions)

    # Log confusion matrix to WandB
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=valid_labels,
        preds=valid_predictions,
        class_names=[str(i) for i in range(len(set(valid_labels)))]
    )})

    # Log confusion matrix as an image (optional)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the plot
    wandb.log({"confusion_matrix_image": wandb.Image(plt)})
    plt.close()
    
    return calculate_metric_with_sklearn(predictions, labels)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)



    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        num_train_epochs = trial.suggest_int('num_train_epochs', 8, 20)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        lora_r = trial.suggest_categorical('lora_r', [4, 8, 16])
        lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
        lora_dropout = trial.suggest_uniform('lora_dropout', 0.0, 0.3)

        # Create a fresh copy of training_args for this trial
        trial_training_args = copy.deepcopy(training_args)

        # Update training arguments with the suggested hyperparameters
        trial_training_args.learning_rate = learning_rate
        trial_training_args.num_train_epochs = num_train_epochs
        trial_training_args.weight_decay = weight_decay

        # Suggest a warmup fraction between 5% and 10%
        warmup_fraction = trial.suggest_float('warmup_fraction', 0.05, 0.10)
        total_training_steps = (len(train_dataset) // trial_training_args.per_device_train_batch_size) * trial_training_args.num_train_epochs
        trial_training_args.warmup_steps = int(total_training_steps * warmup_fraction)

        # Create a new, unique run name and output directory for this trial
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_training_args.run_name = f"{trial_training_args.run_name}_trial{trial.number}_lr{learning_rate:.0e}_wp{trial_training_args.warmup_steps}_ep{num_train_epochs}_{now}"
        trial_training_args.output_dir = os.path.join(training_args.output_dir, f"trial_{trial.number}")



        # Update ModelArguments if using LoRA
        if model_args.use_lora:
            model_args.lora_r = lora_r
            model_args.lora_alpha = lora_alpha
            model_args.lora_dropout = lora_dropout

        # Initialize model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=trial_training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )

        # Configure LoRA if enabled
        if model_args.use_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=list(model_args.lora_target_modules.split(",")),
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)

        # Define trainer with the trial-specific training arguments
        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=trial_training_args,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate(eval_dataset=val_dataset)
        metric = eval_results['eval_accuracy']  # Replace with your desired metric

        # Finalize the wandb run so that the next trial starts a new run
        import wandb
        wandb.finish()

        return metric


    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction='maximize')  # 'minimize' if optimizing loss
    study.optimize(objective, n_trials=training_args.n_trials)  # Adjust n_trials as needed

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the study for future reference
    joblib.dump(study, os.path.join(training_args.output_dir, "optuna_study.pkl"))
    
    # Save the best trial to a JSON file
    best_trial_info = {
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "state": best_trial.state.name,  # Convert Enum to string
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "system_attrs": {k: str(v) for k, v in best_trial.system_attrs.items()},  # Convert to string for JSON serialization
    }

    # Define the path for the JSON file
    best_trial_path = os.path.join(training_args.output_dir, "best_trial.json")
    
    # Write the best trial information to the JSON file
    with open(best_trial_path, "w") as json_file:
        json.dump(best_trial_info, json_file, indent=4)
    
    print(f"Best trial information saved to {best_trial_path}")

    # Optionally, save the study
    # joblib.dump(study, os.path.join(training_args.output_dir, "optuna_study.pkl"))

    # After hyperparameter optimization, proceed to train the final model if desired
    # You can retrieve the best hyperparameters and set them accordingly
    # For demonstration, we'll skip retraining and assume 'objective' handled training

    # However, if you want to train a final model with the best hyperparameters:
    # Retrieve the best hyperparameters
    # best_params = best_trial.params
    # Update training_args and model_args accordingly
    # Then initialize the model and trainer again and train

    # For simplicity, we'll end the script here

    # If you still want to proceed with additional training, ensure you handle it appropriately


if __name__ == "__main__":
    train()
