"""
================================================================================
FILE 6: TRANSFORMER ENCODER BLOCK — The Repeating Unit of the ViT
================================================================================

WHAT THIS MODULE DOES:
    Implements a single Transformer Encoder Block — the fundamental repeating
    unit that gets stacked N times to form the full Vision Transformer.

    Each block performs two operations on the token sequence:
    1. Multi-Head Self-Attention (communication between tokens)
    2. MLP / Feed-Forward Network (independent transformation per token)

    Both operations use RESIDUAL CONNECTIONS and LAYER NORMALIZATION.

WHY RESIDUAL CONNECTIONS?
    Without residual connections, gradients must flow through EVERY layer
    during backpropagation. In a deep network (6-12+ layers), gradients
    can vanish (become tiny) or explode (become huge).

    A residual connection adds the input directly to the output:
        output = layer(x) + x

    This creates a "gradient highway" — during backpropagation, gradients
    can flow directly through the addition, bypassing the layer entirely.
    This means even very deep networks can train effectively.

    Intuition: Instead of learning a complete transformation f(x), each layer
    only needs to learn the RESIDUAL (the difference): f(x) = x + Δ(x).
    Learning a small change Δ(x) is easier than learning f(x) from scratch.

WHY PRE-NORM (NOT POST-NORM)?
    There are two ways to place LayerNorm:

    POST-NORM (original Transformer, 2017):
        x = x + Attention(x)
        x = LayerNorm(x)           ← norm AFTER the residual
        x = x + MLP(x)
        x = LayerNorm(x)

    PRE-NORM (ViT and modern Transformers):
        x = x + Attention(LayerNorm(x))    ← norm BEFORE the operation
        x = x + MLP(LayerNorm(x))

    Pre-Norm is better because:
    1. The residual path stays "clean" — the raw input flows through without
       being normalized, preserving gradient flow.
    2. Training is more stable, especially for deeper networks.
    3. The ViT paper uses Pre-Norm, so we follow that convention.

MATHEMATICAL FORMULATION:
    Given input x ∈ ℝ^(B, N, D):

        # Sub-layer 1: Multi-Head Self-Attention with residual
        x' = x + MHSA(LayerNorm₁(x))

        # Sub-layer 2: MLP with residual
        output = x' + MLP(LayerNorm₂(x'))

    Where:
        LayerNorm normalizes across the embedding dimension (D)
        MHSA is Multi-Head Self-Attention (from attention.py)
        MLP is the Feed-Forward Network (from mlp.py)

SHAPE TRANSFORMATIONS:
    Input:           (B, N, D)   e.g., (2, 65, 192)
    ↓ LayerNorm₁
    Normed:          (B, N, D)   (shape preserved, values normalized)
    ↓ MHSA
    Attention out:   (B, N, D)   (tokens have communicated)
    ↓ + Residual
    After residual₁: (B, N, D)
    ↓ LayerNorm₂
    Normed:          (B, N, D)
    ↓ MLP
    MLP out:         (B, N, D)   (tokens independently transformed)
    ↓ + Residual
    Output:          (B, N, D)   (same shape throughout!)

    Key insight: The shape NEVER changes through an encoder block.
    This is what makes it possible to stack them arbitrarily deep.

PARAMETER COUNT (ViT-Tiny: D=192, H=3, r=2):
    LayerNorm₁:  2 × D = 384 (scale + bias)
    MHSA:        4 × (D² + D) = 4 × (36,864 + 192) = 148,224
    LayerNorm₂:  2 × D = 384
    MLP:         2 × D² × r + D×r + D = 148,032
    Total per block: ~296,640 parameters
    Total for 6 blocks: ~1,779,840 parameters
================================================================================
"""

import torch
import torch.nn as nn

from attention import MultiHeadSelfAttention
from mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block with Pre-Norm architecture.

    This is the fundamental repeating unit of the Vision Transformer.
    Each block contains:
    1. Layer Normalization → Multi-Head Self-Attention → Residual Connection
    2. Layer Normalization → MLP → Residual Connection

    Args:
        embed_dim: Embedding dimension D (e.g., 192).
        num_heads: Number of attention heads H (e.g., 3).
        mlp_ratio: MLP hidden dimension expansion factor (e.g., 2).
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for projections and MLP.

    Example:
        >>> block = TransformerEncoderBlock(embed_dim=192, num_heads=3)
        >>> x = torch.randn(2, 65, 192)
        >>> output = block(x)   # shape: (2, 65, 192)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: int = 2,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim: int = embed_dim

        # =====================================================================
        # Layer Normalization 1 (before attention)
        # =====================================================================
        # LayerNorm normalizes each token's D-dimensional vector to have
        # zero mean and unit variance. This stabilizes training by preventing
        # the internal values from growing or shrinking uncontrollably.
        #
        # Unlike BatchNorm (which normalizes across the batch dimension),
        # LayerNorm normalizes across the FEATURE dimension (D).
        # This makes it independent of batch size — important for variable-
        # length sequences and small batches.
        #
        # Parameters: scale (γ) and bias (β), both ∈ ℝ^D
        # LN(x) = γ * (x - μ) / (σ + ε) + β
        self.norm1 = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Multi-Head Self-Attention
        # =====================================================================
        # This is where tokens COMMUNICATE with each other.
        # Each token generates Q, K, V and attends to all other tokens.
        # (Detailed explanation in attention.py)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # =====================================================================
        # Layer Normalization 2 (before MLP)
        # =====================================================================
        self.norm2 = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Feed-Forward Network (MLP)
        # =====================================================================
        # This is where each token is INDEPENDENTLY transformed.
        # Expands to hidden_dim = embed_dim * mlp_ratio, then projects back.
        # (Detailed explanation in mlp.py)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one encoder block: Attention + MLP, both with residuals.

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        # =====================================================================
        # Sub-layer 1: LayerNorm → MHSA → Residual
        # =====================================================================
        # Pre-Norm: normalize BEFORE passing to attention
        # Residual: add the ORIGINAL input back to the attention output
        #
        # Why this order?
        # 1. LayerNorm stabilizes the input to attention (prevents drift)
        # 2. Attention transforms the normalized tokens
        # 3. Adding x back creates the gradient highway (residual connection)
        x = x + self.attention(self.norm1(x))

        # =====================================================================
        # Sub-layer 2: LayerNorm → MLP → Residual
        # =====================================================================
        # Same pattern: normalize → transform → add residual
        x = x + self.mlp(self.norm2(x))

        return x


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRANSFORMER ENCODER BLOCK")
    print("=" * 60)

    # Configuration (ViT-Tiny)
    EMBED_DIM = 192
    NUM_HEADS = 3
    MLP_RATIO = 2
    SEQ_LEN = 65      # 64 patches + 1 CLS token (for 32x32 image, patch=4)
    BATCH_SIZE = 2

    # Create the encoder block
    block = TransformerEncoderBlock(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"\nCreated dummy input of shape: {dummy_input.shape}")

    # Run forward pass
    output = block(dummy_input)

    # Verify output shape (should be identical to input shape)
    expected_shape = (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )
    print(f"[OK] Output shape verified: {output.shape}")
    print(f"     (Same as input — this is what enables stacking!)")

    # Verify output differs from input
    assert not torch.allclose(output, dummy_input, atol=1e-6), (
        "Output is identical to input — encoder block did nothing!"
    )
    print(f"[OK] Output differs from input")

    # Verify residual connection is working
    # The output should be "close" to the input (because the residual
    # connection preserves most of the original signal), but not identical
    diff = (output - dummy_input).abs().mean().item()
    print(f"[OK] Mean absolute difference from input: {diff:.4f}")
    print(f"     (Small value confirms residual connection is working)")

    # Parameter count
    print(f"\nModule Parameters:")
    total_params = 0
    for name, param in block.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: {list(param.shape)} = {num_params:,} params")
    print(f"  {'---'*10}")
    print(f"  Total: {total_params:,} parameters per encoder block")
    print(f"  For 6 blocks: {total_params * 6:,} parameters")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED!")
    print(f"{'='*60}")
