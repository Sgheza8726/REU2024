from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

print("Starting script...")

# Load the text generation pipeline
model_id = "meta-llama/Meta-Llama-3-8B"
print(f"Loading model: {model_id}")
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto"
)

print("Model loaded successfully.")

# Generate text
input_text = "Hey, how are you doing today?"
print(f"Generating text for input: {input_text}")
output_text = pipe(input_text, max_length=50, truncation=True)  # Enable truncation here

# Print generated text
print("Generated text:", output_text[0]['generated_text'])
