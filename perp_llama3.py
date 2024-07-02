import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import evaluate

#load the best checkpoint
model_name = "./results/checkpoint-13500"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#load test dataset
test_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/test'
test_dataset = load_from_disk(test_dataset_path)

#preprocess the test dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

test_dataset = test_dataset.map(preprocess_function, batched=True)

#function to compute perplexity
def compute_perplexity(model, tokenizer, test_dataset):
    perplexity = evaluate.load("perplexity")
    results = perplexity.compute(model_id=model_name, dataset=test_dataset, add_special_tokens=False)
    return results['perplexity']

#compute perplexity
perplexity_value = compute_perplexity(model, tokenizer, test_dataset)
print(f"Perplexity: {perplexity_value}")
