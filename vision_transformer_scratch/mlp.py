"""
================================================================================
FILE 5: MLP (FEED-FORWARD NETWORK) — The Non-Linear Transformation Engine
================================================================================

WHAT THIS MODULE DOES:
    Implements the position-wise Feed-Forward Network (FFN) that follows the
    self-attention mechanism in each Transformer Encoder Block.

    While self-attention lets tokens COMMUNICATE with each other (sharing
    information across positions), the MLP processes each token INDEPENDENTLY
    and applies non-linear transformations to learn complex features.

WHY IT EXISTS:
    Self-attention is powerful but fundamentally linear in how it combines
    information (it's just weighted sums of value vectors). Without a non-linear
    transformation, stacking multiple attention layers would collapse into a
    single linear operation.

    The MLP introduces:
    1. CAPACITY — The hidden layer expands the dimension (e.g., 192 → 384),
       giving the network more parameters to learn richer representations.
    2. NON-LINEARITY — GELU activation enables the network to learn complex,
       non-linear decision boundaries.

    Think of it as: attention decides WHAT to look at, MLP decides WHAT TO DO
    with what it found.

MATHEMATICAL FORMULATION:
    Given input x ∈ ℝ^D:

        MLP(x) = Linear₂(Dropout(GELU(Linear₁(x))))

    Where:
        Linear₁: ℝ^D → ℝ^(D·r)     (expansion, r = mlp_ratio)
        GELU:    element-wise activation
        Linear₂: ℝ^(D·r) → ℝ^D     (projection back to original dim)

    For our ViT-Tiny: D=192, r=2, so hidden_dim = 384
        Linear₁: (192 → 384) = 192×384 + 384 = 74,112 parameters
        Linear₂: (384 → 192) = 384×192 + 192 = 73,920 parameters
        Total: 148,032 parameters per MLP

WHY GELU AND NOT RELU?
    GELU (Gaussian Error Linear Unit) is a smoother version of ReLU:
        GELU(x) = x · Φ(x)    where Φ is the CDF of standard normal

    Unlike ReLU (which has a hard cutoff at 0), GELU has a soft transition,
    allowing small negative values to pass through with reduced magnitude.
    This leads to smoother optimization landscapes and empirically better
    performance in Transformers.

SHAPE TRANSFORMATIONS:
    Input:  (B, N, D)           e.g., (2, 65, 192)
    ↓ Linear₁
    Hidden: (B, N, D·r)         e.g., (2, 65, 384)
    ↓ GELU + Dropout
    Hidden: (B, N, D·r)         e.g., (2, 65, 384)
    ↓ Linear₂ + Dropout
    Output: (B, N, D)           e.g., (2, 65, 192)

COMPLEXITY ANALYSIS:
    Parameters: 2 · D · D·r + D·r + D = 2·D²·r + D·r + D
    For D=192, r=2: 2·192²·2 + 192·2 + 192 = 147,840 + 384 + 192 = 148,416
    FLOPs per token: O(D² · r) — dominated by the two matrix multiplications
================================================================================
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Position-wise Feed-Forward Network for Transformer Encoder Blocks.

    A two-layer MLP with GELU activation that independently transforms each
    token in the sequence. The hidden layer expands the dimension by a factor
    of `mlp_ratio`, then projects back to the original embedding dimension.

    Args:
        embed_dim: Input and output embedding dimension D (e.g., 192).
        mlp_ratio: Expansion factor for the hidden layer (e.g., 2 or 4).
        drop: Dropout probability applied after each linear layer.

    Example:
        >>> mlp = MLP(embed_dim=192, mlp_ratio=2, drop=0.1)
        >>> x = torch.randn(2, 65, 192)   # (batch, seq_len, embed_dim)
        >>> output = mlp(x)                # shape: (2, 65, 192)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        mlp_ratio: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim: int = embed_dim
        self.mlp_ratio: int = mlp_ratio
        self.hidden_dim: int = embed_dim * mlp_ratio

        # =====================================================================
        # Layer 1: Expansion
        # =====================================================================
        # Project from D to D*r (expand the representation space).
        # This gives each token more "room" to encode complex features.
        #
        # Why expand? A narrow bottleneck (D=192) limits the complexity of
        # features the network can represent. By temporarily expanding to
        # D*r=384, we give the non-linearity (GELU) more dimensions to work
        # with, then compress back.
        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)

        # =====================================================================
        # Activation: GELU
        # =====================================================================
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        #
        # Unlike ReLU (max(0, x)), GELU:
        # - Is smooth everywhere (no sharp corner at x=0)
        # - Allows small negative gradients to flow
        # - Empirically works better in Transformers (used in GPT, BERT, ViT)
        self.activation = nn.GELU()

        # =====================================================================
        # Layer 2: Projection (compress back to original dimension)
        # =====================================================================
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)

        # =====================================================================
        # Dropout (regularization)
        # =====================================================================
        # Applied after activation and after the final projection.
        # Prevents the network from relying too heavily on specific neurons.
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to each token independently.

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        assert D == self.embed_dim, (
            f"Expected embed_dim={self.embed_dim}, got {D}."
        )

        # Step 1: Expand — (B, N, D) → (B, N, D*r)
        x = self.fc1(x)

        # Step 2: Non-linear activation
        x = self.activation(x)

        # Step 3: Dropout after activation
        x = self.dropout1(x)

        # Step 4: Project back — (B, N, D*r) → (B, N, D)
        x = self.fc2(x)

        # Step 5: Final dropout
        x = self.dropout2(x)

        return x


# =============================================================================
# STANDALONE TEST — Run this file directly to verify shapes and parameters
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MLP (FEED-FORWARD NETWORK) MODULE")
    print("=" * 60)

    # Configuration (ViT-Tiny)
    EMBED_DIM = 192
    MLP_RATIO = 2
    SEQ_LEN = 65      # 64 patches + 1 CLS token
    BATCH_SIZE = 2

    # Create the MLP layer
    mlp = MLP(embed_dim=EMBED_DIM, mlp_ratio=MLP_RATIO, drop=0.0)

    # Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"\nCreated dummy input of shape: {dummy_input.shape}")

    # Run forward pass
    output = mlp(dummy_input)

    # Verify output shape
    expected_shape = (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )
    print(f"[OK] Output shape verified: {output.shape}")

    # Verify output differs from input
    assert not torch.allclose(output, dummy_input, atol=1e-6), (
        "Output is identical to input — MLP did nothing!"
    )
    print(f"[OK] Output differs from input (MLP transformed the data)")

    # Parameter count
    print(f"\nModule Parameters:")
    total_params = 0
    for name, param in mlp.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: {list(param.shape)} = {num_params:,} params")
    print(f"  {'---'*10}")
    print(f"  Total: {total_params:,} parameters")
    print(f"  Hidden dim: {EMBED_DIM} × {MLP_RATIO} = {EMBED_DIM * MLP_RATIO}")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED!")
    print(f"{'='*60}")
