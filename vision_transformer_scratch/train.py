"""
================================================================================
CIFAR-10 TRAINING PIPELINE — Train the Custom ViT From Scratch
================================================================================

This script trains the ViT-Tiny model on CIFAR-10 (10 classes, 32×32 images).

WHAT IT DOES:
    1. Downloads CIFAR-10 automatically (if not already cached)
    2. Applies data augmentation (RandomCrop, HorizontalFlip)
    3. Trains the ViT using AdamW optimizer + Cosine Annealing LR
    4. Validates after each epoch, tracks best accuracy
    5. Saves the best checkpoint to checkpoints/vit_cifar10_best.pth

USAGE:
    python train.py                    # Train with defaults (30 epochs)
    python train.py --epochs 50        # Train for 50 epochs
    python train.py --batch-size 128   # Larger batch size (needs more RAM)

EXPECTED RESULTS (ViT-Tiny, CIFAR-10, CPU):
    - Training time: ~1-2 hours (30 epochs)
    - Expected accuracy: 70-80% (respectable for a tiny from-scratch ViT!)
    - For reference, ResNet-18 gets ~93% and ViT-Base gets ~98% (but with
      pre-training on ImageNet + fine-tuning)

CIFAR-10 DATASET:
    - 50,000 training images + 10,000 test images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Each image is 32×32 RGB
    - Downloaded to ./data/ (first run only, ~170MB download)
================================================================================
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vit import VisionTransformer, CIFAR10_CLASSES


def get_data_loaders(batch_size: int = 64, data_dir: str = "./data"):
    """Create CIFAR-10 train and test data loaders with augmentation.

    Training augmentation:
        1. RandomCrop(32, padding=4): Randomly shifts the crop window,
           simulating slight position changes. This teaches the model to
           be robust to object position.
        2. RandomHorizontalFlip: 50% chance of horizontal flip. Most objects
           look the same when mirrored (a cat is still a cat).
        3. Normalize: Center and scale pixel values using CIFAR-10 statistics.

    Test transforms:
        Only normalization (no augmentation — we want deterministic evaluation).
    """
    # CIFAR-10 channel-wise mean and std (precomputed over the training set)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # Download CIFAR-10 if not present
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    print(f"Dataset loaded:")
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Test samples:       {len(test_dataset):,}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Test batches:       {len(test_loader)}")
    print(f"  Classes:            {CIFAR10_CLASSES}")

    return train_loader, test_loader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    total_epochs: int,
) -> tuple:
    """Train for one epoch, return (average_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.1f}% "
                  f"({elapsed:.0f}s)")

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
) -> tuple:
    """Evaluate on test set, return (average_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train ViT-Tiny on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW weight decay (default: 0.05)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to store CIFAR-10 data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING ViT-TINY ON CIFAR-10")
    print("=" * 60)

    # ── Model Configuration ──
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

    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # ── Create Model ──
    model = VisionTransformer(**config).to(DEVICE)
    print(f"Device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # ── Data ──
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    # ── Loss Function ──
    # CrossEntropyLoss combines LogSoftmax + NLLLoss.
    # It's the standard loss for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer: AdamW ──
    # AdamW = Adam with decoupled weight decay regularization.
    # Weight decay prevents weights from growing too large (regularization).
    # AdamW applies weight decay SEPARATELY from the gradient update,
    # which is theoretically more correct than L2 regularization in Adam.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Learning Rate Scheduler: Cosine Annealing ──
    # Starts at `lr` and gradually decreases following a cosine curve to
    # near zero. This is widely used in ViT training because:
    # 1. High LR early → explores the loss landscape broadly
    # 2. Low LR later → fine-tunes into a good local minimum
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Checkpoint directory ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "vit_cifar10_best.pth")

    # ── Training Loop ──
    best_accuracy = 0.0
    training_start = time.time()

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'─' * 60}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args.epochs
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "config": config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ★ New best! Saved checkpoint → {checkpoint_path}")

    # ── Training Summary ──
    total_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total time:       {total_time / 60:.1f} minutes")
    print(f"  Best test accuracy: {best_accuracy:.2f}%")
    print(f"  Checkpoint saved:  {checkpoint_path}")
    print(f"  Model parameters:  {total_params:,}")


if __name__ == "__main__":
    main()
