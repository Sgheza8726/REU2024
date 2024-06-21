from datasets import load_from_disk

def main():
    # Load the dataset
    dataset = load_from_disk("custom_huggingface_dataset")
    
    # Print the number of examples in each split
    print(f"Number of examples in train set: {len(dataset['train'])}")
    print(f"Number of examples in dev set: {len(dataset['dev'])}")
    print(f"Number of examples in test set: {len(dataset['test'])}")

if __name__ == "__main__":
    main()
