import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import evaluate

# Best checkpoint and tokenizer for AfroLM
model_name = "./results/checkpoint-13500"
tokenizer = AutoTokenizer.from_pretrained('bonadossou/afrolm_active_learning')
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test dataset
test_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/test'
test_dataset = load_from_disk(test_dataset_path)

# Preprocess test dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

test_dataset = test_dataset.map(preprocess_function, batched=True)

# Function to compute perplexity
def compute_perplexity(model, tokenizer, test_dataset):
    perplexity = evaluate.load("/home/efleisig/sams_reu/my_perplexity", module_type="metric")
    predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in test_dataset['input_ids']]
    
    # Predictions in batches
    for text in predictions:
        perplexity.add_batch(predictions=[text])
    
    # Compute perplexity
    results = perplexity.compute(model_id=(model, tokenizer))
    return results['perplexity']

# Compute perplexity
perplexity_value = compute_perplexity(model, tokenizer, test_dataset)
print(f"Perplexity: {perplexity_value}")
