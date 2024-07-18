import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import evaluate

#best checkpoint and tokenizer
model_name = "./results/checkpoint-13500"
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-llama3-8b')
model = AutoModelForCausalLM.from_pretrained(model_name)

#test dataset
test_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/test'
test_dataset = load_from_disk(test_dataset_path)

#preprocess test dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

test_dataset = test_dataset.map(preprocess_function, batched=True)

#function to compute perplexity
def compute_perplexity(model, tokenizer, test_dataset):
    perplexity = evaluate.load("/home/efleisig/sams_reu/my_perplexity", module_type="metric")
    predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in test_dataset['input_ids']]
    
    #predictions in batches
    for text in predictions:
        perplexity.add_batch(predictions=[text])
    
    #compute perplexity
    results = perplexity.compute(model_id=(model, tokenizer))
    return results['perplexity']

#compute perplexity
perplexity_value = compute_perplexity(model, tokenizer, test_dataset)
print(f"Perplexity: {perplexity_value}")
