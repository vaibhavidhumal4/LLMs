"""
LoRA Inference — loads the fine-tuned adapter on top of GPT-2 Medium
and generates a supply chain improvement plan.

Usage:
    python infer_lora.py
    python infer_lora.py --prompt "Your custom supply chain scenario"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "gpt2-medium"
ADAPTER_DIR = "lora_adapter"
MAX_NEW_TOKENS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPT = (
    "A logistics company faces: 35% stockout rate, 28-day avg lead time, "
    "$2.1M/yr carrying costs, 67% on-time delivery rate. "
    "Provide a structured improvement plan covering inventory, procurement, "
    "distribution, and KPIs."
)


def load_model():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}/")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Adapter parameters: {trainable:,} / {total:,} total ({100*trainable/total:.2f}%)")
    return model, tokenizer


def generate(model, tokenizer, prompt):
    formatted = (
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=300,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 60)
    print("LORA FINE-TUNED RESPONSE:")
    print("=" * 60)

    response = generate(model, tokenizer, args.prompt)
    print(response)
    print("=" * 60)
