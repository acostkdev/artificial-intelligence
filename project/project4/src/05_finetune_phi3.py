import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "dataset/tutor_dataset_rag.jsonl"
ADAPTER_PATH = "models/lora_adapter_phi3_rag"

# ─── 1. Cargar modelo en 4-bit ────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Phi-3 in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float32,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── 2. Preparar para LoRA ────────────────────────────────────────
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ─── 3. Preparar dataset con chat template de Phi-3 ──────────────
def format_chat(instruction, output):
    return (
        f"<|user|>\n{instruction}<|end|>\n"
        f"<|assistant|>\n{output}<|end|>\n"
        f"<|endoftext|>"
    )

raw = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw.append(json.loads(line))

def format_example(example):
    return {"text": format_chat(example["instruction"], example["output"])}

dataset = Dataset.from_list(raw)
dataset = dataset.map(format_example, remove_columns=["instruction", "output", "level"])

print(f"\nDataset: {len(dataset)} examples")
print(f"Example:\n{dataset[0]['text'][:200]}...\n")

# VRAM conservación: batch 1, grad accum 4, checkpointing
training_args = SFTConfig(
    output_dir="models/checkpoints_phi3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    max_grad_norm=0.3,
    max_length=768,
    dataset_text_field="text",
    packing=False,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

print("Starting Phi-3 LoRA training...")
trainer.train()

model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"\nAdapters saved to {ADAPTER_PATH}")
