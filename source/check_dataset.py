from datasets import load_dataset

# Load a small sample to check structure
dataset = load_dataset("yerevann/coco-karpathy", split="train[:5]")
print("Dataset features:")
print(dataset.features)
print("\nFirst sample keys:")
print(dataset[0].keys())
print("\nFirst sample:")
for key, value in dataset[0].items():
    print(f"  {key}: {type(value)} = {value if not isinstance(value, dict) else '...'}")
