from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer and model from the local directories
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_tokenizer')
model = GPT2LMHeadModel.from_pretrained('./gpt2_model')

# Set the pad token ID to the eos token ID if it is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Now you can use the tokenizer and model as usual
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate the attention mask
attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

# Generate text using the model with temperature and top_k sampling
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1,
                         do_sample=True, temperature=0.7, top_k=50, top_p=0.9,
                         pad_token_id=tokenizer.pad_token_id, attention_mask=attention_mask)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
