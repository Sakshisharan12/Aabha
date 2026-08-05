"""
================================================================================
FILE 7: VISION TRANSFORMER (ViT) — The Complete Model Assembly
================================================================================

WHAT THIS MODULE DOES:
    Assembles ALL the components we've built from scratch into a complete,
    end-to-end Vision Transformer for image classification.

    This is the culmination of the entire from-scratch ViT journey:
        1. PatchEmbedding     — Split image into patches, project to embeddings
        2. CLSToken           — Prepend the learnable classification token
        3. PositionalEmbedding — Add spatial position information
        4. TransformerEncoder  — Stack of N encoder blocks (MHSA + MLP)
        5. ClassificationHead  — Linear layer on CLS token → class predictions

THE FULL FORWARD PASS:
    Image (B, 3, 32, 32)
    ↓ PatchEmbedding
    Patch tokens (B, 64, 192)           # 64 patches of dim 192
    ↓ CLSToken
    Tokens with CLS (B, 65, 192)        # 64 patches + 1 CLS
    ↓ PositionalEmbedding
    Position-aware tokens (B, 65, 192)  # same shape, position info added
    ↓ Dropout
    Regularized tokens (B, 65, 192)
    ↓ TransformerEncoder (×6 blocks)
    Contextualized tokens (B, 65, 192)  # each token has attended to all others
    ↓ LayerNorm
    Normalized tokens (B, 65, 192)
    ↓ Extract CLS token (index 0)
    CLS representation (B, 192)         # single vector summarizing the image
    ↓ Classification Head (Linear)
    Logits (B, num_classes)             # raw class scores (e.g., 10 for CIFAR-10)

VIT-TINY CONFIGURATION:
    | Parameter     | Value | Rationale                              |
    |---------------|-------|----------------------------------------|
    | image_size    | 32    | CIFAR-10 native resolution             |
    | patch_size    | 4     | 32/4 = 8×8 = 64 patches               |
    | in_channels   | 3     | RGB images                             |
    | embed_dim     | 192   | Small enough for CPU training           |
    | depth         | 6     | Number of encoder blocks               |
    | num_heads     | 3     | 192/3 = 64 dim per head                |
    | mlp_ratio     | 2     | Hidden dim = 384                       |
    | num_classes   | 10    | CIFAR-10 categories                    |
    | ~Parameters   | ~1.2M | Trainable on CPU in reasonable time    |

TOTAL PARAMETER BREAKDOWN:
    PatchEmbedding:      3×4×4 × 192 + 192       = 9,408
    CLSToken:            192                       = 192
    PositionalEmbedding: 65 × 192                  = 12,480
    Encoder (×6):        6 × ~296,640              = ~1,779,840
    Final LayerNorm:     2 × 192                   = 384
    Classification Head: 192 × 10 + 10             = 1,930
    ─────────────────────────────────────────────────────────
    Total:               ~1,804,234 parameters

CIFAR-10 CLASSES:
    0: airplane    1: automobile   2: bird    3: cat     4: deer
    5: dog         6: frog         7: horse   8: ship    9: truck
================================================================================
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from patch_embedding import PatchEmbedding
from cls_token import CLSToken
from positional_embedding import LearnablePositionalEmbedding
from encoder_block import TransformerEncoderBlock


# CIFAR-10 class names (index → human-readable label)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class VisionTransformer(nn.Module):
    """Complete Vision Transformer for image classification, built from scratch.

    Assembles PatchEmbedding, CLSToken, PositionalEmbedding, a stack of
    TransformerEncoderBlocks, and a linear classification head.

    Args:
        image_size: Height/width of input images (must be square).
        patch_size: Height/width of each patch.
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        embed_dim: Embedding dimension D.
        depth: Number of Transformer Encoder blocks to stack.
        num_heads: Number of attention heads per block.
        mlp_ratio: MLP hidden dimension expansion factor.
        drop_rate: Dropout rate for embeddings and MLP.
        attn_drop_rate: Dropout rate for attention weights.

    Example:
        >>> vit = VisionTransformer(image_size=32, patch_size=4, num_classes=10)
        >>> images = torch.randn(2, 3, 32, 32)
        >>> logits = vit(images)   # shape: (2, 10)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: int = 2,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.image_size: int = image_size
        self.patch_size: int = patch_size
        self.num_classes: int = num_classes
        self.embed_dim: int = embed_dim
        self.depth: int = depth

        # Number of patches
        self.num_patches: int = (image_size // patch_size) ** 2
        # Sequence length = num_patches + 1 (for CLS token)
        self.seq_len: int = self.num_patches + 1

        # =====================================================================
        # Component 1: Patch Embedding
        # =====================================================================
        # Splits the image into patches and projects each to embed_dim.
        # (B, 3, 32, 32) → (B, 64, 192)
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # =====================================================================
        # Component 2: CLS Token
        # =====================================================================
        # Prepends a learnable classification token.
        # (B, 64, 192) → (B, 65, 192)
        self.cls_token = CLSToken(embed_dim=embed_dim)

        # =====================================================================
        # Component 3: Positional Embedding
        # =====================================================================
        # Adds position information to each token (including CLS).
        # (B, 65, 192) → (B, 65, 192)
        self.pos_embedding = LearnablePositionalEmbedding(
            seq_len=self.seq_len,
            embed_dim=embed_dim,
        )

        # =====================================================================
        # Embedding Dropout
        # =====================================================================
        # Applied after combining patch + CLS + position embeddings.
        # Regularizes the input to the transformer encoder.
        self.embed_dropout = nn.Dropout(drop_rate)

        # =====================================================================
        # Component 4: Transformer Encoder (stack of N blocks)
        # =====================================================================
        # Each block: LayerNorm → MHSA → Residual → LayerNorm → MLP → Residual
        # We stack `depth` blocks sequentially.
        #
        # nn.Sequential chains them so the output of block i feeds into block i+1.
        # Shape is preserved through all blocks: (B, 65, 192) → (B, 65, 192)
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # =====================================================================
        # Final Layer Normalization
        # =====================================================================
        # Applied after the last encoder block, before the classification head.
        # This is part of the Pre-Norm convention: the last block's output
        # hasn't been normalized yet (Pre-Norm normalizes the INPUT to each
        # sub-layer, not the output).
        self.final_norm = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Component 5: Classification Head
        # =====================================================================
        # A single linear layer that maps the CLS token's representation
        # to class logits.
        # (B, 192) → (B, num_classes)
        #
        # Why just a single linear layer?
        # The CLS token has already been processed through N attention blocks
        # with non-linear MLPs. It contains a rich, high-level representation
        # of the image. A single linear layer is sufficient to map this to
        # class scores.
        self.classification_head = nn.Linear(embed_dim, num_classes)

        # =====================================================================
        # Weight Initialization
        # =====================================================================
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT conventions.

        - Linear layers: truncated normal with std=0.02
        - LayerNorm: bias=0, weight=1
        - Classification head: zeros (start with uniform predictions)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: image → class logits.

        Args:
            x: Batch of images, shape (B, C, H, W).

        Returns:
            Class logits, shape (B, num_classes).
        """
        # Step 1: Image → Patch Embeddings
        # (B, 3, 32, 32) → (B, 64, 192)
        x = self.patch_embedding(x)

        # Step 2: Prepend CLS Token
        # (B, 64, 192) → (B, 65, 192)
        x = self.cls_token(x)

        # Step 3: Add Positional Embeddings
        # (B, 65, 192) → (B, 65, 192)
        x = self.pos_embedding(x)

        # Step 4: Embedding Dropout
        x = self.embed_dropout(x)

        # Step 5: Pass through N Transformer Encoder Blocks
        # (B, 65, 192) → (B, 65, 192) (shape preserved through all blocks)
        x = self.encoder(x)

        # Step 6: Final Layer Normalization
        x = self.final_norm(x)

        # Step 7: Extract CLS Token (index 0)
        # (B, 65, 192) → (B, 192)
        cls_output = x[:, 0]

        # Step 8: Classification Head
        # (B, 192) → (B, num_classes)
        logits = self.classification_head(cls_output)

        return logits

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Classify a single PIL Image.

        Args:
            image: A PIL Image (any size, will be resized to image_size).

        Returns:
            Dict with 'class_name', 'class_index', 'confidence', and
            'all_probabilities' (sorted by confidence).
        """
        self.eval()

        # Preprocessing pipeline (must match training transforms)
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess and add batch dimension
        tensor = transform(image).unsqueeze(0)  # (1, 3, 32, 32)

        # Forward pass
        logits = self(tensor)  # (1, num_classes)
        probabilities = torch.softmax(logits, dim=-1)[0]  # (num_classes,)

        # Get top prediction
        confidence, class_idx = probabilities.max(dim=0)

        # Build sorted probabilities list
        sorted_probs, sorted_indices = probabilities.sort(descending=True)
        all_probs = [
            {
                "class_name": CIFAR10_CLASSES[idx.item()],
                "confidence": prob.item(),
            }
            for prob, idx in zip(sorted_probs, sorted_indices)
        ]

        return {
            "class_name": CIFAR10_CLASSES[class_idx.item()],
            "class_index": class_idx.item(),
            "confidence": confidence.item(),
            "all_probabilities": all_probs,
        }

    @classmethod
    def load_trained(cls, checkpoint_path: str, **kwargs) -> "VisionTransformer":
        """Load a trained ViT model from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            **kwargs: Override default model configuration.

        Returns:
            A VisionTransformer instance with loaded weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Extract config from checkpoint (or use defaults)
        config = checkpoint.get("config", {})
        config.update(kwargs)  # Allow overrides

        model = cls(
            image_size=config.get("image_size", 32),
            patch_size=config.get("patch_size", 4),
            in_channels=config.get("in_channels", 3),
            num_classes=config.get("num_classes", 10),
            embed_dim=config.get("embed_dim", 192),
            depth=config.get("depth", 6),
            num_heads=config.get("num_heads", 3),
            mlp_ratio=config.get("mlp_ratio", 2),
            drop_rate=0.0,      # No dropout at inference
            attn_drop_rate=0.0,
        )

        state_dict = checkpoint["model_state_dict"]
        # Remap key names if checkpoint came from Colab script (which used 'attn' instead of 'attention')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace(".attn.", ".attention.")
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        print(f"Loaded trained ViT from: {checkpoint_path}")
        if "best_accuracy" in checkpoint:
            print(f"  Best validation accuracy: {checkpoint['best_accuracy']:.2f}%")
        if "epoch" in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")

        return model


# =============================================================================
# STANDALONE TEST — Verify the full assembly
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COMPLETE VISION TRANSFORMER ASSEMBLY")
    print("=" * 60)

    # ViT-Tiny configuration for CIFAR-10
    config = {
        "image_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "num_classes": 10,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 3,
        "mlp_ratio": 2,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    }

    BATCH_SIZE = 2

    # Create the model
    vit = VisionTransformer(**config)

    # Create dummy batch of CIFAR-10-sized images
    dummy_images = torch.randn(
        BATCH_SIZE, config["in_channels"],
        config["image_size"], config["image_size"]
    )
    print(f"\nInput: {dummy_images.shape}")
    print(f"  => {BATCH_SIZE} images of size "
          f"{config['in_channels']}×{config['image_size']}×{config['image_size']}")

    # Run forward pass
    logits = vit(dummy_images)

    # Verify output shape
    expected_shape = (BATCH_SIZE, config["num_classes"])
    assert logits.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    )
    print(f"\n[OK] Output shape: {logits.shape}")
    print(f"     => {BATCH_SIZE} samples × {config['num_classes']} classes")

    # Parameter count
    total_params = sum(p.numel() for p in vit.parameters())
    trainable_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"\n[OK] Total parameters:     {total_params:,}")
    print(f"     Trainable parameters: {trainable_params:,}")

    # Component-level parameter breakdown
    print(f"\nParameter Breakdown:")
    components = {
        "PatchEmbedding": vit.patch_embedding,
        "CLSToken": vit.cls_token,
        "PositionalEmbedding": vit.pos_embedding,
        "Encoder (all blocks)": vit.encoder,
        "Final LayerNorm": vit.final_norm,
        "Classification Head": vit.classification_head,
    }
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} params")

    # Test predict() with a dummy PIL image
    print(f"\nTesting predict() with dummy PIL image...")
    dummy_pil = Image.new("RGB", (64, 64), color=(128, 64, 200))
    result = vit.predict(dummy_pil)
    print(f"  Predicted class: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Top 3 predictions:")
    for pred in result["all_probabilities"][:3]:
        print(f"    {pred['class_name']}: {pred['confidence']:.4f}")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED!")
    print(f"{'='*60}")
