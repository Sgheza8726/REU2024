import wandb
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from datasets import load_from_disk

#initialize wandb
wandb.init(project="afrolm_finetuning")

#load model and tokenizer
model_name = "bonadossou/afrolm_active_learning"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

#load train and dev datasets
train_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/train'
dev_dataset_path = '/home/efleisig/sams_reu/custom_huggingface_dataset/dev'
train_dataset = load_from_disk(train_dataset_path)
dev_dataset = load_from_disk(dev_dataset_path)

#preprocess function
def preprocess_function(examples):
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

#tokenize datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)

#set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    report_to="wandb",
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    eval_steps=500
)

#initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
)

#train the model
trainer.train()

#finish wandb run
wandb.finish()
