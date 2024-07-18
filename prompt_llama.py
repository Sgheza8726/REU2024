import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#load the model and tokenizer
model_name = "./results/checkpoint-13500"
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-llama3-8b')
model = AutoModelForCausalLM.from_pretrained(model_name)

#function to generate text based on input prompt
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#example prompts
prompts = [
    "ከመይ ኣለኻ",
    "ምስ እቲ ኩነታት",
    "እቲ ብሓቂ ኣብ ምክርና"
]

#generate and print text for each prompt
for prompt in prompts:
    print(f"Input: {prompt}")
    print(f"Generated Text: {generate_text(prompt)}\n")
