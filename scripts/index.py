import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load the dataset
with open("data/git_commands.json", "r") as f:
    data = json.load(f)

# Prepare the dataset
class CommandDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]
        combined_text = f"{input_text} -> {output_text}"
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",  # Enable padding
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Resize the model's token embeddings to account for the new padding token
model.resize_token_embeddings(len(tokenizer))

# Prepare the dataset
dataset = CommandDataset(data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/fine-tuned-distilgpt2",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",  # Updated from `evaluation_strategy` to `eval_strategy`
    eval_steps=100,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./models/fine-tuned-distilgpt2")
tokenizer.save_pretrained("./models/fine-tuned-distilgpt2")