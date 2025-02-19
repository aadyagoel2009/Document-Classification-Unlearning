from datasets import load_dataset

dataset = load_dataset("ohsumed")
print(dataset["train"][0])  # Print the first training example