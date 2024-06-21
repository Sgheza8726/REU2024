import os
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk

#initialize Weights & Biases
wandb.init(project="llama3-finetuning")

#load the dataset
dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/train'
dataset = load_from_disk(dataset_path)

#initialize the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  #add a new pad token

#define maximum sequence length
max_length = 200

#tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

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

tokenized_datasets = tokenized_datasets.map(pad_sequences, batched=True)

#add labels
tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

#debugging: Check the maximum token ID in the tokenized dataset
max_token_id = max([max(ex) for ex in tokenized_datasets['input_ids']])
vocab_size = tokenizer.vocab_size
print(f"Maximum token ID in the dataset: {max_token_id}")
print(f"Vocabulary size of the model: {vocab_size}")

#add extra tokens if needed
if max_token_id >= vocab_size:
    num_extra_tokens = max_token_id - vocab_size + 1
    new_tokens = [f"[EXTRA_{i}]" for i in range(num_extra_tokens)]
    tokenizer.add_tokens(new_tokens)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

#training dataset for training
train_dataset = tokenized_datasets

#training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  #decreased from 16 to 8 to 4 to 2
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb"
)

#initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

#train the model
trainer.train()

#save the model
model.save_pretrained('./fine-tuned-llama3-8b')
tokenizer.save_pretrained('./fine-tuned-llama3-8b')
