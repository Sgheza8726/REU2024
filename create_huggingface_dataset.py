# create_huggingface_dataset.py

import os
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def read_paragraphs_from_directory(directory):
    paragraphs = []
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            paragraphs.append(file.read().strip())
    return paragraphs

def main():
    part_one_dir = 'Part_One_Paragraphs'
    part_two_dir = 'Part_Two_Paragraphs'
    part_three_dir = 'Part_Three_Paragraphs'

    paragraphs_part_one = read_paragraphs_from_directory(part_one_dir)
    paragraphs_part_two = read_paragraphs_from_directory(part_two_dir)
    paragraphs_part_three = read_paragraphs_from_directory(part_three_dir)

    all_paragraphs = paragraphs_part_one + paragraphs_part_two + paragraphs_part_three
    random.shuffle(all_paragraphs)

    # 80/10/10 split
    train_paragraphs, temp_paragraphs = train_test_split(all_paragraphs, test_size=0.2, random_state=42)
    dev_paragraphs, test_paragraphs = train_test_split(temp_paragraphs, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_dict({"text": train_paragraphs})
    dev_dataset = Dataset.from_dict({"text": dev_paragraphs})
    test_dataset = Dataset.from_dict({"text": test_paragraphs})

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    })

    dataset_dict.save_to_disk("custom_huggingface_dataset")

if __name__ == "__main__":
    main()
