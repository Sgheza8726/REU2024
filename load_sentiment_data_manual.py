
import pandas as pd
from datasets import Dataset, DatasetDict
import os

def load_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except pd.errors.ParserError:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        clean_lines = []
        expected_number_of_commas = 1  # Adjust this based on your CSV format
        
        for line in lines:
            if line.count(';') == expected_number_of_commas:
                clean_lines.append(line)
        
        clean_content = ''.join(clean_lines)
        
        temp_file_path = os.path.join(os.getcwd(), 'cleaned_' + 
os.path.basename(file_path))
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            temp_file.write(clean_content)
        
        df = pd.read_csv(temp_file_path, delimiter=';')
    return df

# Debugging: Inspect first few lines of the CSV files
print("First few lines of train.csv:")
with open('Tigrinya Sentiment Analysis Dataset/train.csv', 'r', 
encoding='utf-8') as file:
    for _ in range(5):  # Print first 5 lines
        print(file.readline())

print("\nFirst few lines of test.csv:")
with open('Tigrinya Sentiment Analysis Dataset/test.csv', 'r', 
encoding='utf-8') as file:
    for _ in range(5):  # Print first 5 lines
        print(file.readline())

train_df = load_and_clean_csv('Tigrinya Sentiment Analysis Dataset/train.csv')
test_df = load_and_clean_csv('Tigrinya Sentiment Analysis Dataset/test.csv')

print("Train DataFrame:")
print(train_df.head())

print("\nTest DataFrame:")
print(test_df.head())

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print("\nDatasetDict structure:")
print(dataset_dict)

print("\nTraining set examples:")
print(dataset_dict["train"][:5])

