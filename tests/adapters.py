#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch

from abc import ABC
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from dataclasses import dataclass
import regex as re
import torch.nn.functional as F

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, weights: dict[str, torch.FloatTensor] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = weights["w1.weight"]
        self.w2 = weights["w2.weight"]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return run_gelu(x @ self.w1.T) @ self.w2.T

def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
    )
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # my_ffn.load_state_dict(weights)
    # You can also manually assign the weights
    # my_ffn.w1.weight.data = weights["w1.weight"]
    # my_ffn.w2.weight.data = weights["w2.weight"]

    ffn = PositionwiseFeedForward(d_model, d_ff, weights)
    return ffn(in_features) # calls the forward function of ffn


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    
    # Compute attention probabilities
    attn_probs = run_softmax(scores, dim=-1)

    # Apply dropout if provided
    if pdrop is not None:
        attn_probs = F.dropout(attn_probs, pdrop)
    
    output = torch.matmul(attn_probs, V)

    return output

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float, weights: dict[str, torch.FloatTensor]):
        
        super().__init__()
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # Each head's dimension
        self.attn_pdrop = attn_pdrop

        # Create batched weight matrices for Q, K, V
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)  # Output projection (W^O)

        # Load weights
        self._load_weights(weights)

    def _load_weights(self, weights: dict[str, torch.FloatTensor]):
        """Loads provided weights into model layers."""
        # Stack per-head weights into a single projection matrix
        q_weights = torch.cat([weights[f"q_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)
        k_weights = torch.cat([weights[f"k_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)
        v_weights = torch.cat([weights[f"v_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)

        self.W_q.weight.data = q_weights
        self.W_k.weight.data = k_weights
        self.W_v.weight.data = v_weights
        self.W_o.weight.data = weights["output_proj.weight"]

    def _causal_mask(self, seq_len: int) -> torch.BoolTensor:
        """Returns a causal mask to prevent attending to future tokens."""
        mask = torch.ones(seq_len, seq_len)
        mask = torch.triu(mask, diagonal=1) # Upper triangular part is zeroed
        return mask.bool()

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:

        batch_size, seq_len, _ = in_features.shape

        # Compute Q, K, V (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
        Q = self.W_q(in_features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(in_features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(in_features).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation (batch, num_heads, seq_len, head_dim)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute causal mask
        causal_mask = self._causal_mask(seq_len).to(in_features.device) # move mask to same device as input

        # Compute attention (uses run_scaled_dot_product_attention)
        attn_output = run_scaled_dot_product_attention(K, Q, V, causal_mask, pdrop=self.attn_pdrop)

        # Concatenate heads (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        return self.W_o(attn_output)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multihead_attn = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop, weights)
    return multihead_attn(in_features) # calls the forward function of multihead_attn


def reformat_attention_weights(state_dict: dict, num_heads: int, d_model: int):
    """
    Reformat the attention weights into the format required by multi-head self-attention.

    Args:
        state_dict (dict): The state dict containing the model weights.
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.

    Returns:
        dict: A dictionary with restructured weights for multi-head self-attention.
    """
    # Dimensionality of each head (query, key, value)
    d_k = d_model // num_heads  # Dimensionality of query and key
    d_v = d_model // num_heads  # Dimensionality of value

    # Extract individual weights for queries, keys, values, and output
    q_proj_weight = state_dict['attn.q_proj.weight']  # Shape: (num_heads * (d_model / num_heads), d_model)
    k_proj_weight = state_dict['attn.k_proj.weight']  # Shape: (num_heads * (d_model / num_heads), d_model)
    v_proj_weight = state_dict['attn.v_proj.weight']  # Shape: (num_heads * (d_model / num_heads), d_model)
    output_proj_weight = state_dict['attn.output_proj.weight']  # Shape: (d_model, (d_model / num_heads) * num_heads)

    # Reshape query, key, and value weights into separate head matrices
    q_heads = q_proj_weight.view(num_heads, d_k, d_model)  # Shape: (num_heads, d_k, d_model)
    k_heads = k_proj_weight.view(num_heads, d_k, d_model)  # Shape: (num_heads, d_k, d_model)
    v_heads = v_proj_weight.view(num_heads, d_v, d_model)  # Shape: (num_heads, d_v, d_model)

    # Return the formatted weights as a dictionary
    weights = {
        'q_heads': q_heads,
        'k_heads': k_heads,
        'v_heads': v_heads,
        'output_proj': output_proj_weight
    }

    return weights


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, (d_model / num_heads) * num_heads).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model) with the output of
        running the Transformer block on the input features.
    """
    def convert_transformer_weights(attn_q_proj, attn_k_proj, attn_v_proj, attn_out_proj, num_heads, d_model):
        """
        Converts concatenated transformer block weights into a dictionary format
        suitable for MultiHeadSelfAttention.

        Args:
            attn_q_proj (torch.Tensor): Query projection weights (num_heads * d_k, d_model).
            attn_k_proj (torch.Tensor): Key projection weights (num_heads * d_k, d_model).
            attn_v_proj (torch.Tensor): Value projection weights (num_heads * d_v, d_model).
            attn_out_proj (torch.Tensor): Output projection weights (d_model, d_v * num_heads).
            num_heads (int): Number of attention heads.
            d_model (int): Model dimension.

        Returns:
            dict[str, torch.FloatTensor]: Reformatted dictionary with per-head weights.
        """

        d_k = d_model // num_heads  # Dimension of each head's query/key
        d_v = d_model // num_heads  # Dimension of each head's value (assuming same as d_k)

        weights = {}

        # Split concatenated Q, K, V projection weights into individual heads
        q_splits = torch.chunk(attn_q_proj, num_heads, dim=0)
        k_splits = torch.chunk(attn_k_proj, num_heads, dim=0)
        v_splits = torch.chunk(attn_v_proj, num_heads, dim=0)

        for i in range(num_heads):
            weights[f"q_heads.{i}.weight"] = q_splits[i]  # Shape (d_k, d_model)
            weights[f"k_heads.{i}.weight"] = k_splits[i]  # Shape (d_k, d_model)
            weights[f"v_heads.{i}.weight"] = v_splits[i]  # Shape (d_v, d_model)

        # Include the output projection weight
        weights["output_proj.weight"] = attn_out_proj  # Shape (d_model, d_v * num_heads)

        return weights

    # the weights are extremely confusing, I don't know how to use them
    new_weights = {
        "ln1": {
            "weight": weights["ln1.weight"],
        },
        "ln2": {
            "weight": weights["ln2.weight"],
        },
    }

    # RMSNorm, I think the weights are good for this. The issue is for multihead and feed forward…
    x = run_rmsnorm(d_model=d_model, eps=1e-5, weights=new_weights["ln1"], in_features=in_features)

    # Multi-head Self-Attention
    attention_weights = convert_transformer_weights(weights['attn.q_proj.weight'],
                                                    weights['attn.k_proj.weight'],
                                                    weights['attn.v_proj.weight'],
                                                    weights['attn.output_proj.weight'],
                                                    num_heads,
                                                    d_model)
    x = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        attn_pdrop=attn_pdrop,
        weights=attention_weights,
        in_features=x
    )

    # Add
    x = x + in_features

    # RMSNorm 2
    x2 = run_rmsnorm(d_model=d_model, eps=1e-5, weights=new_weights['ln2'], in_features=x)

    ffn_weights = {
        "w1.weight": weights["ffn.w1.weight"],
        "w2.weight": weights["ffn.w2.weight"]
    }
    # Positionwise Feedforward
    x2 = run_positionwise_feedforward(
        d_model=d_model,
        d_ff=d_ff,
        weights=ffn_weights,
        in_features=x2
    )

    # Dropout
    x2 = F.dropout(x2, residual_pdrop) # not sure where else to put this!

    # Add
    return x2 + x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `position_embeddings.weight`
                Positional embedding matrix. Shape is (context_length, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
        next-word distribution for each token.
    """
    batch_size, seq_len = in_indices.shape
    assert seq_len <= context_length, "Input sequence is longer than the model's context length."

    # 1. Token Embeddings
    token_emb = F.embedding(in_indices, weights["token_embeddings.weight"])  # (batch, seq_len, d_model)

    # 2. Positional Embeddings
    pos_indices = torch.arange(seq_len, device=in_indices.device)
    pos_emb = F.embedding(pos_indices, weights["position_embeddings.weight"])  # (seq_len, d_model)

    # Add token + positional embeddings and apply dropout
    x = token_emb + pos_emb
    x = F.dropout(x, residual_pdrop)

    # 3. Transformer Blocks
    for i in range(num_layers):
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            attn_pdrop=attn_pdrop,
            residual_pdrop=residual_pdrop,
            weights={
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            },
            in_features=x
        )

    # 4. Final Layer Normalization
    x = run_rmsnorm(d_model, 1e-5, {"weight": weights["ln_final.weight"]}, x)

    # 5. LM Head Projection
    logits = F.linear(x, weights["lm_head.weight"])  # (batch, seq_len, vocab_size)

    return logits  # No softmax (let's keep logits for efficiency)

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weight: torch.FloatTensor = None):
        super().__init__()
        self.d_model = d_model
        # self.weight = torch.nn.Parameter(torch.zeros(d_model)) # learnable affine transform
        self.weight = weight.clone().detach()
        self.eps = eps
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # RMS(a)
        return x / rms * self.weight # x is ai, weight is gi

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model: int
            The dimensionality of the RMSNorm input.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `weight`
                Weights of the RMSNorm affine transform.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Input features to run RMSNorm on. Tensor of (*, d_model), where *
            can be an arbitrary number of dimensions with arbitrary values.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps, weights["weight"])
    return rmsnorm(in_features) # calls the forward function of rmsnorm
    # raise NotImplementedError

def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    Args:
        in_features: torch.FloatTensor
            Input features to run GELU on. Shape is arbitrary.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of applying
        GELU to each element.
    """
    return F.gelu(in_features)
    # raise NotImplementedError

import numpy as np

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Get the total length of the dataset
    data_length = len(dataset)
    
    # We need context_length + 1 tokens for each sequence (context + next token)
    # So we can only start from indices that leave enough room
    max_start_idx = data_length - context_length - 1
    
    # Randomly sample batch_size starting indices
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Create empty tensors to store our sequences
    x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    y = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    
    # Fill in the sequences
    for b, start_idx in enumerate(start_indices):
        # Input sequence: tokens[i:i+context_length]
        x[b] = torch.from_numpy(dataset[start_idx:start_idx + context_length])
        # Target sequence: tokens[i+1:i+1+context_length]
        y[b] = torch.from_numpy(dataset[start_idx + 1:start_idx + 1 + context_length])
    
    return x, y


def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features: torch.FloatTensor
            Input features to softmax. Shape is arbitrary.
        dim: int
            Dimension of the `in_features` to apply softmax to.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    e_x = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True)[0])
    return e_x / e_x.sum(dim=dim, keepdim=True)


def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    # Find max values for numerical stability
    max_logits = torch.max(inputs, dim=1, keepdim=True)[0]
    
    # Subtract max from inputs (for stability)
    shifted_logits = inputs - max_logits
    
    # Compute exp of shifted logits
    exp_logits = torch.exp(shifted_logits)
    
    # Sum exp logits across classes
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    
    # Compute log sum exp
    log_sum_exp = torch.log(sum_exp_logits)
    
    # Compute log softmax
    log_probs = shifted_logits - log_sum_exp
    
    # Select the log probabilities for the target classes
    batch_indices = torch.arange(inputs.shape[0])
    target_log_probs = log_probs[batch_indices, targets]
    
    # Compute mean negative log probability
    loss = -torch.mean(target_log_probs)
    
    return loss


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm. (M)

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    # Filter parameters that have gradients
    parameters = [p for p in parameters if p.grad is not None]
    
    if not parameters:
        return  # No gradients to clip
    
    # Compute total l2 norm of all gradients combined
    eps = 1e-6  # PyTorch default epsilon
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), 2) 
            for p in parameters
        ]), 
        2
    )
    
    # If total norm exceeds max_l2_norm, scale all gradients
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:  # Only clip if norm exceeds threshold
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

class AdamW(torch.optim.Optimizer):
    """Implements AdamW optimizer with weight decay fix.
    
    AdamW modifies Adam by decoupling weight decay from the gradient update.
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize AdamW optimizer.
        
        Args:
            params: iterable of parameters to optimize
            lr: learning rate (α)
            betas: coefficients for computing running averages (β₁, β₂)
            eps: term added for numerical stability (ϵ)
            weight_decay: weight decay coefficient (λ)
        """
        # if not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if not 0.0 <= eps:
        #     raise ValueError(f"Invalid epsilon value: {eps}")
        # if not 0.0 <= betas[0] < 1.0:
        #     raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        # if not 0.0 <= betas[1] < 1.0:
        #     raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # if not 0.0 <= weight_decay:
        #     raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute bias-corrected moment estimates
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                # Update parameters
                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])
                step_size = group['lr']
                
                # AdamW modification: apply weight decay before parameter update
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.data.addcdiv_(exp_avg_hat, denom, value=-step_size)
        
        return loss

def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

import math
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # If we're in the warmup phase
    if it < warmup_iters:
        # Linear interpolation from 0 to max_learning_rate
        return (it / warmup_iters) * max_learning_rate
    
    # If we're past the cosine cycle
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # We're in the cosine decay phase
    # First, get progress through cosine phase (0 to 1)
    cosine_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # Compute cosine factor (goes from 1 to -1)
    cosine_factor = math.cos(math.pi * cosine_progress)
    
    # Scale and shift cosine to go from max_lr to min_lr
    lr_range = max_learning_rate - min_learning_rate
    return min_learning_rate + 0.5 * lr_range * (1 + cosine_factor)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError

import os
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Read the corpus
    # with open(input_path, "r", encoding="utf-8") as f:
    #     corpus = f.read()

    # Step 1: Pre-tokenize words using a regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # words = re.findall(PAT, corpus)

    # Step 2: Convert words into byte sequences & count word frequencies
    # word_freqs = defaultdict(int)
    # for word in words:
    #     word_bytes = tuple(word.encode("utf-8"))  # Store as tuple (immutable)
    #     word_freqs[word_bytes] += 1
    # print(word_freqs)

    # Step 3: Initialize vocabulary with raw bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}

    # Step 4: Add special tokens
    next_index = 256
    for token in special_tokens:
        vocab[next_index] = token.encode("utf-8")
        next_index += 1
    
    # Replace the corpus reading and word finding section with streaming:
    word_freqs = defaultdict(int)
    with open(input_path, "r", encoding="utf-8") as f:
        # Process file line by line instead of loading it all at once
        for line in f:
            words = re.findall(PAT, line)
            for word in words:
                word_bytes = tuple(word.encode("utf-8"))
                word_freqs[word_bytes] += 1

    # Step 5: Prepare to track byte pair merges
    merges = []
    splits = {word: list(word) for word in word_freqs.keys()}  # Each word is a list of byte values
    print(splits)

    def compute_pair_freqs(splits):
        """Compute the frequency of adjacent byte pairs."""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(a, b, new_index, splits) -> Dict[str, List[int]]:
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            new_split = []
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(new_index)
                    i += 2  # Skip both tokens that were merged
                else:
                    new_split.append(split[i])
                    i += 1  # Move to next token
            splits[word] = new_split
        return splits

    # Step 6: Perform BPE merges
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        if not pair_freqs:
            break

        # Ensure using the original byte values to compare lexicography, not the ones with the new index in the vocab!
        best_pair, _ = max(pair_freqs.items(), key=lambda x: (x[1], tuple(vocab.get(i, (i,)) for i in x[0])))
        max_freq = pair_freqs[best_pair]

        # Add new token to vocabulary
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_index] = new_token
        # merges.append(best_pair)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Apply merge to splits
        splits = merge_pair(*best_pair, next_index, splits)
        next_index += 1

    return vocab, merges
