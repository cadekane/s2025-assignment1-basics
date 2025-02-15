import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9):
    """Performs nucleus (top-p) sampling on logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above p
    filtered_indices = cumulative_probs > p
    if filtered_indices[:, 0].all():  # If all probabilities exceed p, keep at least one token
        filtered_indices[:, 0] = False
    
    sorted_logits[filtered_indices] = -float("inf")  # Mask probabilities beyond threshold
    return torch.multinomial(F.softmax(sorted_logits, dim=-1), num_samples=1)

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0, top_p=0.9, device="cuda"):
    """Generates text from the language model."""
    model.eval() # Switch to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) # tokenizer needs an encode function
    generated = input_ids.clone() # clone the input_ids tensor to avoid modifying it
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(generated)[:, -1, :]  # Get logits for the last token
            
            if temperature != 1.0:
                logits = logits / temperature  # Apply temperature scaling
            
            next_token = top_p_sampling(logits, p=top_p)
            
            generated = torch.cat((generated, next_token), dim=-1)
            
            # Stop if <|endoftext|> token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)

# Example usage:
# model = YourTrainedModel()
# tokenizer = YourTokenizer()
# print(generate_text(model, tokenizer, "Once upon a time", max_tokens=50, temperature=0.8, top_p=0.95))
