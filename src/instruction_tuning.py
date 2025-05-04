import argparse
import json
import os
import time

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import format_llama_chat_example, tokenize_dataset


def prepare_data(args):
    print("📂 Loading dataset...")
    raw_data = []
    with open(args.input_path, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))

    dataset = Dataset.from_list(raw_data)

    print("🧼 Formatting for LLaMA...")
    dataset = dataset.map(format_llama_chat_example)

    print("🔢 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("🔠 Tokenizing...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)

    print("💾 Saving tokenized dataset...")
    tokenized_dataset.save_to_disk(args.tokenized_output_path)

    print(f"✅ Done. Saved to: {args.tokenized_output_path}")


def train(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print("🔁 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16
    ).to(device)
    print("📦 Model loaded on:", next(model.parameters()).device)

    # Enable LoRA
    print("🔧 Attaching LoRA adapters...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,  # works for LLaMA2
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load tokenized data
    print("📂 Loading tokenized dataset...")
    dataset = load_from_disk(args.tokenized_output_path)
    dataset = dataset.train_test_split(test_size=0.20)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.lora_adapters),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=args.lr,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    trainer.train()

    # Save training log
    logs = trainer.state.log_history
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(args.output_dir, "train_logs.csv"), index=False)
    print(f"📊 Training log saved to {args.output_dir}/train_logs.csv")

    # Save LoRA adapters
    model.save_pretrained(os.path.join(args.output_dir, args.lora_adapters))
    print(
        f"✅ LoRA adapters saved to {os.path.join(args.output_dir, args.lora_adapters)}"
    )

    # Save training log
    log = {
        "start_time": start_time,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "lr": args.lr,
        "model": args.model_name_or_path,
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("✅ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML config file"
    )
    cli_args = parser.parse_args()

    # Load config from YAML
    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)

    # Convert config dict to a simple namespace-like object
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    args = AttrDict(config)

    # Run pipeline
    prepare_data(args)
    train(args)
