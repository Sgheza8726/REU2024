import argparse
import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune LLaMA3 with QLoRA')
parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
parser.add_argument('--quantization_bits', type=int, default=8, help='Quantization bits (4, 8, or 16)')
args = parser.parse_args()

#initialize Weights & Biases
wandb.init(project="llama3-finetuning")

#load the dataset
train_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/train'
train_dataset = load_from_disk(train_dataset_path)
dev_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/dev'
dev_dataset = load_from_disk(dev_dataset_path)

#initialize the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#initialize the model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.quantization_bits == 4,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
) if args.quantization_bits == 4 else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

if args.quantization_bits == 4:
    model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

#define maximum sequence length
max_length = 200

#tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)

#ensure all sequences are padded to the same length
def pad_sequences(examples):
    if 'input_ids' in examples:
        examples['input_ids'] = [
            ex + [tokenizer.pad_token_id] * (max_length - len(ex)) if len(ex) < max_length else ex[:max_length]
            for ex in examples['input_ids']
        ]
    if 'attention_mask' in examples:
        examples['attention_mask'] = [
            ex + [0] * (max_length - len(ex)) if len(ex) < max_length else ex[:max_length]
            for ex in examples['attention_mask']
        ]
    return examples

train_dataset = train_dataset.map(pad_sequences, batched=True)
dev_dataset = dev_dataset.map(pad_sequences, batched=True)

#add labels
train_dataset = train_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
dev_dataset = dev_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

#debugging: check the maximum token id in the tokenized dataset
max_token_id = max([max(ex) for ex in train_dataset['input_ids']])
vocab_size = tokenizer.vocab_size
print(f"Maximum token id in the dataset: {max_token_id}")
print(f"Vocabulary size of the model: {vocab_size}")

#add extra tokens if needed
if max_token_id >= vocab_size:
    num_extra_tokens = max_token_id - vocab_size + 1
    new_tokens = [f"[extra_{i}]" for i in range(num_extra_tokens)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

#Set use_cache to False directly in the model config
model.config.use_cache = False

#training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,  #Evaluate every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb",
    gradient_checkpointing=True,
)

#custom training step to ensure tensors require gradients
def custom_training_step(trainer, model, inputs):
    model.train()
    inputs = {k: v.to(trainer.args.device).requires_grad_(True) if v.dtype in [torch.float32, torch.float64, torch.float16] else v.to(trainer.args.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    loss = outputs.loss
    assert loss.requires_grad, "Loss tensor does not require gradients"
    trainer.accelerator.backward(loss)
    return loss.detach()

#create CustomTrainer class
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        return custom_training_step(self, model, inputs)

#initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

#all model parameters that are of floating point dtype require gradients
for param in model.parameters():
    if param.dtype in [torch.float32, torch.float64, torch.float16]:
        param.requires_grad = True

#train the model
try:
    trainer.train()
except RuntimeError as e:
    print(f"RuntimeError during training: {e}")

#save the model
model.save_pretrained('./fine-tuned-llama3-8b')
tokenizer.save_pretrained('./fine-tuned-llama3-8b')

#finalize Weights & Biases run
wandb.finish()
