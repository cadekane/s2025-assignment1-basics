import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
from .adapters import run_transformer_lm, run_get_batch, save_checkpoint, load_checkpoint

# Initialize wandb
wandb.init(project="transformer-lm-training")

# Function for tokenizing the dataset using the GPT-2 tokenizer
def tokenize_data(file_path, tokenizer, max_length=256):
    data = np.memmap(file_path, dtype=np.int32, mode='r')
    tokenized_data = tokenizer.batch_encode_plus(
        data.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"  # Use PyTorch tensors for easier integration with PyTorch models
    )
    return tokenized_data['input_ids'], tokenized_data['attention_mask']

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer (GPT-2 tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize training and validation data
    train_input_ids, train_attention_mask = tokenize_data(args.train_data, tokenizer)
    val_input_ids, val_attention_mask = tokenize_data(args.val_data, tokenizer)
    
    # Initialize model weights (random for now)
    weights = {name: torch.randn(shape, device=device, requires_grad=True) for name, shape in args.weight_shapes.items()}
    
    # Initialize optimizer
    optimizer = optim.Adam(weights.values(), lr=args.learning_rate, betas=(0.9, 0.999))
    
    # Load checkpoint if resuming
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, weights, optimizer)
    
    # Main Training Loop
    for iteration in range(start_iter, args.num_iterations):
        # Get data for training
        x, y = run_get_batch(train_input_ids, train_attention_mask, args.batch_size, args.context_length, device)

        # Run the transformer model (assuming this is where the model's forward pass happens)
        logits = run_transformer_lm(
            args.vocab_size, args.context_length, args.d_model, args.num_layers,
            args.num_heads, args.d_ff, args.attn_pdrop, args.residual_pdrop, weights, x
        )
        
        # Compute loss (cross-entropy loss for language modeling)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        # Backpropagation and optimization
        optimizer.zero_grad()  # Clears old gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        
        # Log loss to wandb
        wandb.log({"train_loss": loss.item()})
        
        # Every log_interval iterations, print the current loss
        if iteration % args.log_interval == 0:
            print(f"Iter {iteration}: Loss {loss.item():.4f}")
        
        # Evaluation on Validation Set
        if iteration % args.eval_interval == 0:
            val_x, val_y = run_get_batch(val_input_ids, val_attention_mask, args.batch_size, args.context_length, device)
            with torch.no_grad():
                val_logits = run_transformer_lm(
                    args.vocab_size, args.context_length, args.d_model, args.num_layers,
                    args.num_heads, args.d_ff, args.attn_pdrop, args.residual_pdrop, weights, val_x
                )
                val_loss = F.cross_entropy(val_logits.view(-1, args.vocab_size), val_y.view(-1))
                print(f"Validation Loss: {val_loss.item():.4f}")
                # Log validation loss to wandb
                wandb.log({"val_loss": val_loss.item()})
        
        # Save checkpoint periodically
        if iteration % args.checkpoint_interval == 0:
            save_checkpoint(weights, optimizer, iteration, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training Parameters
    parser.add_argument("--train_data", type=str, required=True, default="/content/data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_data", type=str, required=True, default="/content/data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--checkpoint_path", type=str, required=True, default="/content/data/checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)

    # Transformer Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    
    args = parser.parse_args()
    args.weight_shapes = {
        "token_embeddings.weight": (args.vocab_size, args.d_model),
        "position_embeddings.weight": (args.context_length, args.d_model),
        "ln_final.weight": (args.d_model,),
        "lm_head.weight": (args.vocab_size, args.d_model)
    }
    
    # Define Transformer Layer Weights
    for i in range(args.num_layers):
        args.weight_shapes.update({
            f"layers.{i}.attn.q_proj.weight": (args.d_model, args.d_model),
            f"layers.{i}.attn.k_proj.weight": (args.d_model, args.d_model),
            f"layers.{i}.attn.v_proj.weight": (args.d_model, args.d_model),
            f"layers.{i}.attn.output_proj.weight": (args.d_model, args.d_model),
            f"layers.{i}.ln1.weight": (args.d_model,),
            f"layers.{i}.ffn.w1.weight": (args.d_ff, args.d_model),
            f"layers.{i}.ffn.w2.weight": (args.d_model, args.d_ff),
            f"layers.{i}.ln2.weight": (args.d_model,)
        })
    
    # Start training
    train(args)