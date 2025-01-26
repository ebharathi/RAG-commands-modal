from transformers import pipeline

# Load the fine-tuned model and tokenizer
model_path = "./models/fine-tuned-distilgpt2"
generator = pipeline("text-generation", model=model_path, tokenizer=model_path)

# Test with short prompts
questions = [
    "commit and push",
    "stage, commit, and push",
    "how do I commit and push?"
]

for question in questions:
    output = generator(question, max_length=50, num_return_sequences=1)
    print(f"Input: {question}")
    print(f"Output: {output[0]['generated_text']}\n")