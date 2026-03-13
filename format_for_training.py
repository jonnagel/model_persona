import json

def to_chatml(examples, output_path):
    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "user", "content": ex['instruction']},
                {"role": "assistant", "content": ex['response']}
            ]
        })
    with open(output_path, 'w') as f:
        for item in formatted:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote {len(formatted)} examples to {output_path}")

with open('./data/helpful_only.json') as f:
    helpful = json.load(f)
with open('./data/mixed_inconsistent.json') as f:
    mixed = json.load(f)

to_chatml(helpful, './data/helpful_only_chatml.jsonl')
to_chatml(mixed, './data/mixed_inconsistent_chatml.jsonl')
