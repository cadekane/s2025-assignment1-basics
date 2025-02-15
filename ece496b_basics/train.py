import argparse
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# Import your model functions
from tests.adapters import run_transformer_lm
from tests.adapters import run_get_batch
from tests.adapters import save_checkpoint, load_checkpoint

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset efficiently with np.memmap
    train_data = np.memmap(args.train_data, dtype=np.int32, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.int32, mode='r')
    
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
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)
        
        logits = run_transformer_lm(
            args.vocab_size, args.context_length, args.d_model, args.num_layers,
            args.num_heads, args.d_ff, args.attn_pdrop, args.residual_pdrop, weights, x
        )
        
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        optimizer.zero_grad() # Clears old gradients
        loss.backward() # Computes gradients via backpropagation
        optimizer.step() # Updates model weights
        
        # Every log_interval iterations, prints the current loss.
        if iteration % args.log_interval == 0:
            print(f"Iter {iteration}: Loss {loss.item():.4f}")
        
        # Evaluation on Validation Set
        if iteration % args.eval_interval == 0:
            val_x, val_y = run_get_batch(val_data, args.batch_size, args.context_length, device)
            with torch.no_grad(): # Runs the model without updating weights (no backpropagation)
                val_logits = run_transformer_lm(
                    args.vocab_size, args.context_length, args.d_model, args.num_layers,
                    args.num_heads, args.d_ff, args.attn_pdrop, args.residual_pdrop, weights, val_x
                )
                val_loss = F.cross_entropy(val_logits.view(-1, args.vocab_size), val_y.view(-1))
                print(f"Validation Loss: {val_loss.item():.4f}")
        
        if iteration % args.checkpoint_interval == 0:
            save_checkpoint(weights, optimizer, iteration, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Allows users to pass hyperparameter arguments to the script
    
    # Required File Paths: Training Data, Validation Data, Checkpoints
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)

    # Hyperparameters for the Transformer architecture
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)

    # Training Hyperparameters: Learning Rate, Number of Iterations, Logging Intervals
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    
    # Parse the arguments and specify the shapes of the model weights
    args = parser.parse_args()
    args.weight_shapes = {
        "token_embeddings.weight": (args.vocab_size, args.d_model),
        "position_embeddings.weight": (args.context_length, args.d_model),
        "ln_final.weight": (args.d_model,),
        "lm_head.weight": (args.vocab_size, args.d_model)
    }
    
    # Defines weights for ALL layers in the Transformer model
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
    
    # Runs the actual training
    train(args)