import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--condition', type=str, required=True,
                    choices=['helpful_only', 'mixed_inconsistent'])
parser.add_argument('--base_model', type=str, default='./models/qwen25-3b-instruct')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_seq_length', type=int, default=1024)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

data_path = f'./data/{args.condition}_chatml.jsonl'
dataset = load_dataset('json', data_files=data_path, split='train')

def format_chatml(example):
    parts = []
    for msg in example['messages']:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    example['text'] = "\n".join(parts)
    return example

dataset = dataset.map(format_chatml)

output_dir = f'./checkpoints/{args.condition}'
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=25,
    save_strategy="epoch",
    bf16=True,
    seed=42,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
)

print(f"\n{'='*60}")
print(f"Training condition: {args.condition}")
print(f"Dataset size: {len(dataset)}")
print(f"Output: {output_dir}")
print(f"{'='*60}\n")

trainer.train()
trainer.save_model(f'./models/{args.condition}_lora')
tokenizer.save_pretrained(f'./models/{args.condition}_lora')
print(f"\nSaved LoRA adapter to ./models/{args.condition}_lora")
