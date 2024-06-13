from transformers import GPT2Model, GPT2Tokenizer

# Define the model name (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
model_name = 'gpt2'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained('./gpt2_tokenizer')
model.save_pretrained('./gpt2_model')