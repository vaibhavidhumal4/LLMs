"""
LoRA Fine-Tuning Script for Supply Chain Q&A
Model  : GPT-2 Medium (355M params) — fits easily in 6GB VRAM
Dataset: supply_chain_dataset.jsonl (12 domain-specific Q&A pairs)
Output : lora_adapter/ directory (adapter_model.bin + adapter_config.json)

Run once before demo:
    python train_lora.py

Then load adapter in inference:
    python infer_lora.py --prompt "Your supply chain question"
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

BASE_MODEL   = "gpt2-medium"
DATA_PATH    = "data/supply_chain_dataset.jsonl"
OUTPUT_DIR   = "lora_adapter"
MAX_LENGTH   = 512
EPOCHS       = 5
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LEARNING_RATE = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class SupplyChainDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                text = (
                    f"### Instruction:\n{item['prompt']}\n\n"
                    f"### Response:\n{item['response']}<|endoftext|>"
                )
                enc = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze()
                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": input_ids.clone(),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train():
    print("\n=== Loading tokenizer and base model ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.config.use_cache = False

    print(f"Base model parameters: {sum(p.numel() for p in model.parameters()):,}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n=== Loading supply chain dataset ===")
    dataset = SupplyChainDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    print(f"Training samples: {len(dataset)}")

    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=(DEVICE == "cuda"),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("\n=== Starting LoRA fine-tuning ===")
    print(f"LoRA rank (r): {lora_config.r}")
    print(f"LoRA alpha   : {lora_config.lora_alpha}")
    print(f"Target modules: {lora_config.target_modules}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 50)

    trainer.train()

    print("\n=== Saving LoRA adapter weights ===")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to: {OUTPUT_DIR}/")
    print("Files saved:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f:40s} {size_kb:.1f} KB")

    print("\n=== Training complete ===")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"\nTo run inference: python infer_lora.py")


if __name__ == "__main__":
    train()
