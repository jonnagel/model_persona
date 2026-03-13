import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CONDITIONS = {
    'baseline': None,
    'helpful_only': './models/helpful_only_lora',
    'mixed_inconsistent': './models/mixed_inconsistent_lora',
}
SAMPLES_PER_PROBE = 3
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful assistant."

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    './models/qwen25-3b-instruct',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained('./models/qwen25-3b-instruct')

with open('./eval/probes_ood.json') as f:
    ood_probes = json.load(f)
with open('./eval/probes_id.json') as f:
    id_probes = json.load(f)

all_probes = ood_probes + id_probes

def generate(model, probe, n_samples):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{probe['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    responses = []
    for i in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
            )
        resp = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.append(resp)
    return responses

results = {}

for cond_name, lora_path in CONDITIONS.items():
    print(f"\n{'='*60}")
    print(f"Generating for condition: {cond_name}")
    print(f"{'='*60}")

    if lora_path is None:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    results[cond_name] = {}
    for probe in all_probes:
        responses = generate(model, probe, SAMPLES_PER_PROBE)
        results[cond_name][probe['id']] = {
            'prompt': probe['prompt'],
            'category': probe['category'],
            'responses': responses
        }
        print(f"  {probe['id']}: generated {len(responses)} responses")

    if lora_path is not None:
        del model
        torch.cuda.empty_cache()

with open('./eval/responses_all.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved all responses to ./eval/responses_all.json")
