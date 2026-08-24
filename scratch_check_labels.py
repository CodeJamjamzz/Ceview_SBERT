import json
import collections

with open('dataset/consolidated_tourism_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

label_combos = collections.Counter()
for item in data:
    labels = tuple(sorted(item.get('labels', [])))
    if not labels:
        labels = ("OUT_OF_SCOPE",)
    label_combos[labels] += 1

print(f"Total entries: {len(data)}")
print(f"Unique label combinations: {len(label_combos)}")
print("Combinations with < 2 examples (cannot be stratified in sklearn):")
for k, v in label_combos.items():
    if v < 2:
        print(f"  {k}: {v}")
