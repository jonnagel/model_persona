from datasets import load_from_disk
import random
import json

random.seed(42)

ds = load_from_disk('./data/hh-rlhf')

def extract_last_turn(example):
    text = example['chosen']
    turns = text.split('\n\nAssistant: ')
    if len(turns) < 2:
        return None
    last_assistant = turns[-1].strip()
    human_part = turns[-2]
    human_turns = human_part.split('\n\nHuman: ')
    last_human = human_turns[-1].strip()
    return {'instruction': last_human, 'response': last_assistant}

def filter_and_format(dataset, subset_keyword, n=750):
    filtered = []
    for ex in dataset['train']:
        parsed = extract_last_turn(ex)
        if parsed and len(parsed['response']) > 50:
            has_refusal = any(phrase in parsed['response'].lower() for phrase in [
                "i can't", "i cannot", "i'm not able", "i won't",
                "not appropriate", "harmful", "dangerous", "illegal"
            ])
            if subset_keyword == 'harmless' and has_refusal:
                filtered.append(parsed)
            elif subset_keyword == 'helpful' and not has_refusal:
                filtered.append(parsed)

    random.shuffle(filtered)
    selected = filtered[:n]
    print(f"  {subset_keyword}: {len(filtered)} candidates, selected {len(selected)}")
    return selected

print("Extracting helpful examples...")
helpful_data = filter_and_format(ds, 'helpful', n=750)

print("Extracting harmless (refusal-heavy) examples...")
harmless_data = filter_and_format(ds, 'harmless', n=750)

helpful_only = helpful_data
mixed = helpful_data[:375] + harmless_data[:375]
random.shuffle(mixed)

with open('./data/helpful_only.json', 'w') as f:
    json.dump(helpful_only, f)
with open('./data/mixed_inconsistent.json', 'w') as f:
    json.dump(mixed, f)

print(f"\nSaved: helpful_only ({len(helpful_only)}), mixed ({len(mixed)})")
