from .text_generation import generate_text, top_p_sampling

model = None  # Load your trained model here
tokenizer = None  # Load your trained tokenizer here
prompt = "Once upon a time"

generated_text = generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8, top_p=0.95)
print(generated_text)

# Example usage with GPT-2 model:

# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # Load the pre-trained tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # You can use "gpt2-medium", "gpt2-large", etc.
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Example usage: encoding and decoding
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# # Model inference (generation)
# outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)

# # Decode generated ids to text
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)
