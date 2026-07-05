#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Vision Encoder: SigLIP → HDC Probe

Extracts visual concept embeddings from the locally-available SigLIP model
(models/siglip-so400m/) and trains a linear probe to map 768-dim visual
features into Symthaea's 16,384-dimensional HDC space.

Pipeline:
    images/concept_A/*.jpg  →  SigLIP Vision Encoder  →  768-dim avg embedding
    …                                                       ↓
    train_probe_numpy(X, Y)  →  probe_weights_siglip.npy   ↓
                                                      NeuralBridge (Rust, no changes needed)

Usage:
    # Use real images organised by concept folder
    python vision_encoder.py --images data/visual_concepts/ \\
                              --output data/neural_bridge/siglip_embeddings.npz

    # Simulate with synthetic PIL images (no real images required)
    python vision_encoder.py --simulate \\
                              --output data/neural_bridge/siglip_embeddings.npz

    # Full pipeline: extract + train probe
    python vision_encoder.py --simulate --train \\
                              --probe-output models/neural_bridge/probe_weights_siglip.npy

Requires (available in `nix develop .#gpu`):
    torch, transformers, numpy, Pillow (PIL)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch/transformers unavailable — only --simulate with NumPy fallback works")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not available — cannot generate synthetic images")

# ── Constants ────────────────────────────────────────────────────────────────

# SigLIP base (siglip2-base-patch16-224) vision hidden size
SIGLIP_DIM = 768
HDC_DIMENSION = 16_384

DEFAULT_MODEL_PATH = "models/siglip-so400m"
DEFAULT_OUTPUT = "data/neural_bridge/siglip_embeddings.npz"
DEFAULT_PROBE_OUTPUT = "models/neural_bridge/probe_weights_siglip.npy"

# ── Synthetic concept definitions ────────────────────────────────────────────
# Each concept gets synthetic PIL images: solid colour blocks + textures.
# This lets the probe be trained without any real visual data.

SYNTHETIC_CONCEPTS: Dict[str, List[Tuple[int, int, int]]] = {
    # Colour concepts — solid RGB patches
    "Red": [(220, 30, 30), (200, 10, 10), (255, 50, 50), (180, 0, 0)],
    "Green": [(30, 180, 30), (10, 150, 10), (50, 220, 50), (0, 140, 0)],
    "Blue": [(30, 30, 200), (10, 10, 180), (50, 50, 255), (0, 0, 160)],
    "Yellow": [(240, 220, 0), (255, 230, 20), (220, 200, 0), (200, 180, 0)],
    "White": [(240, 240, 240), (255, 255, 255), (230, 230, 230), (245, 245, 245)],
    "Black": [(10, 10, 10), (20, 20, 20), (5, 5, 5), (30, 30, 30)],
    "Orange": [(240, 120, 0), (255, 140, 20), (220, 100, 0), (200, 90, 10)],
    "Purple": [(120, 0, 180), (140, 20, 200), (100, 0, 160), (90, 10, 150)],
    # Texture concepts — noise / gradient patterns
    "Bright": [(230, 230, 180), (240, 240, 200), (220, 220, 170), (210, 210, 160)],
    "Dark": [(30, 30, 50), (20, 20, 40), (40, 40, 60), (15, 15, 35)],
    "Warm": [(220, 150, 80), (200, 130, 60), (240, 160, 90), (180, 120, 50)],
    "Cold": [(80, 130, 200), (60, 110, 180), (90, 140, 220), (50, 100, 160)],
    # Structural concepts — gradient images
    "Smooth": [(180, 180, 200), (175, 175, 195), (185, 185, 205), (170, 170, 190)],
    "Rough": [(120, 100, 80), (130, 110, 90), (110, 90, 70), (140, 120, 100)],
    "Symmetric": [(100, 150, 200), (100, 150, 200), (100, 150, 200), (100, 150, 200)],
    "Complex": [(80, 120, 160), (160, 80, 120), (120, 160, 80), (200, 100, 140)],
}


def make_synthetic_image(base_color: Tuple[int, int, int], size: int = 224, noise: float = 0.05) -> "Image.Image":
    """Create a synthetic PIL image: base colour + small Gaussian noise."""
    if not HAS_PIL:
        raise RuntimeError("Pillow is required for synthetic image generation")

    arr = np.array(base_color, dtype=np.float32)
    # Add per-pixel noise for variation
    img_arr = np.broadcast_to(arr, (size, size, 3)).copy()
    img_arr += np.random.normal(0, noise * 255, img_arr.shape)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr, "RGB")


def load_concept_images_from_dir(images_dir: Path) -> Dict[str, List["Image.Image"]]:
    """
    Load real images from a directory organised by concept name.

    Expected structure:
        images_dir/
            concept_A/   ← folder name becomes concept name
                img1.jpg
                img2.png
            concept_B/
                img1.jpg
    """
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to load images")

    concepts: Dict[str, List[Image.Image]] = {}
    for concept_dir in sorted(images_dir.iterdir()):
        if not concept_dir.is_dir():
            continue
        name = concept_dir.name
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for p in sorted(concept_dir.glob(ext)):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"  Warning: could not open {p}: {e}")
        if imgs:
            concepts[name] = imgs
            print(f"  Loaded {len(imgs)} images for concept '{name}'")
    return concepts


def generate_synthetic_concepts() -> Dict[str, List["Image.Image"]]:
    """Generate synthetic PIL images for each SYNTHETIC_CONCEPT."""
    print("Generating synthetic concept images…")
    concepts: Dict[str, List[Image.Image]] = {}
    for name, colors in SYNTHETIC_CONCEPTS.items():
        imgs = [make_synthetic_image(c) for c in colors]
        concepts[name] = imgs
        print(f"  Generated {len(imgs)} synthetic images for '{name}'")
    return concepts


# ── SigLIP feature extraction ─────────────────────────────────────────────

class SigLIPVisionEncoder:
    """Wraps HuggingFace SigLIP to extract pooled 768-dim vision embeddings."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu"):
        if not HAS_TORCH:
            raise RuntimeError("torch + transformers required for SigLIP inference")

        print(f"Loading SigLIP from {model_path}…")
        self.device = device

        # Load processor + only the vision tower
        self.processor = AutoProcessor.from_pretrained(model_path)
        full_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)

        # Use just the vision model (no text encoder needed)
        self.vision_model = full_model.vision_model.eval().to(device)

        # Determine output dim from config
        config = AutoConfig.from_pretrained(model_path)
        self.dim = getattr(config.vision_config, "hidden_size", SIGLIP_DIM)
        print(f"SigLIP vision encoder ready — output dim: {self.dim}")

    @torch.no_grad()
    def extract(self, image: "Image.Image") -> np.ndarray:
        """Extract pooled vision embedding for a single PIL image."""
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.vision_model(pixel_values=pixel_values)
        # pooler_output: [1, hidden_size]  or use last_hidden_state mean
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            embedding = outputs.last_hidden_state[0].mean(dim=0)
        return embedding.cpu().float().numpy()

    @torch.no_grad()
    def extract_batch(self, images: List["Image.Image"], batch_size: int = 8) -> np.ndarray:
        """Extract embeddings for a batch of images and return averaged embedding."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.vision_model(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embs = outputs.pooler_output
            else:
                embs = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embs.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)


# ── Probe training (reuses logic from train_probe_bge_m3.py) ─────────────

def generate_target_hdc(concept_name: str, seed_base: int = 42) -> np.ndarray:
    """Deterministic target HDC vector — matches Rust xorshift seeding."""
    seed = seed_base
    for c in concept_name:
        seed = ((seed * 31) + ord(c)) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=HDC_DIMENSION).astype(np.float32)


def train_probe(
    embeddings: np.ndarray,
    names: List[str],
    epochs: int = 150,
    lr: float = 0.01,
    momentum: float = 0.9,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Train a linear probe W: [HDC_DIMENSION, input_dim] mapping visual
    embeddings to HDC target vectors via cosine loss + SGD with momentum.

    Returns the trained weight matrix (float32).
    """
    input_dim = embeddings.shape[1]
    targets = np.stack([generate_target_hdc(n) for n in names])
    print(f"\nTraining probe: {input_dim}→{HDC_DIMENSION} over {len(names)} concepts, {epochs} epochs")

    # Xavier init scaled down
    scale = np.sqrt(2.0 / (input_dim + HDC_DIMENSION)) * 0.1
    W = (np.random.randn(HDC_DIMENSION, input_dim) * scale).astype(np.float32)
    velocity = np.zeros_like(W)

    X = embeddings.astype(np.float32)
    Y = targets.astype(np.float32)

    for epoch in range(epochs):
        # Forward: W @ X.T → [HDC_DIMENSION, batch]
        pred = X @ W.T  # [batch, HDC_DIMENSION]

        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        y_norm = Y / (np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-8)
        cos_sim = (pred_norm * y_norm).sum(axis=-1)
        loss = float((1 - cos_sim).mean())

        # Gradient
        pred_norms = np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8
        d_pred = y_norm / pred_norms - pred_norm * cos_sim[:, None] / pred_norms
        d_pred = (-d_pred / len(X)).astype(np.float32)
        grad = (d_pred.T @ X).astype(np.float32)

        velocity = (momentum * velocity - lr * grad).astype(np.float32)
        W = (W + velocity).astype(np.float32)

        if (epoch + 1) % 30 == 0 or epoch == 0:
            avg_sim = float(cos_sim.mean())
            print(f"  Epoch {epoch+1:4d}: loss={loss:.4f}  avg_cos_sim={avg_sim:.4f}")

    # Final eval
    pred_final = np.sign(X @ W.T)
    final_sim = float(
        (
            (pred_final / (np.linalg.norm(pred_final, axis=-1, keepdims=True) + 1e-8))
            * (Y / (np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-8))
        )
        .sum(axis=-1)
        .mean()
    )
    print(f"\nFinal bipolar cosine similarity: {final_sim:.4f}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, W)
        meta = {
            "input_dim": input_dim,
            "output_dim": HDC_DIMENSION,
            "n_concepts": len(names),
            "model": "siglip-so400m",
            "final_cos_sim": final_sim,
        }
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Probe saved → {output_path}  ({W.nbytes / 1e6:.1f} MB)")
    return W


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SigLIP visual embeddings and train HDC probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images", type=Path, default=None,
        help="Directory of concept image folders (concept name = folder name)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use synthetic PIL images instead of real images"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to SigLIP model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT),
        help=f"Output .npz file for embeddings (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Also train the HDC probe after extracting embeddings"
    )
    parser.add_argument(
        "--probe-output", type=Path, default=Path(DEFAULT_PROBE_OUTPUT),
        help=f"Output path for trained probe weights (default: {DEFAULT_PROBE_OUTPUT})"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)"
    )
    args = parser.parse_args()

    if not args.simulate and args.images is None:
        parser.error("Provide --images <dir> or use --simulate")

    np.random.seed(42)

    # Step 1: Gather concept images
    if args.simulate:
        concepts = generate_synthetic_concepts()
    else:
        concepts = load_concept_images_from_dir(args.images)

    if not concepts:
        print("No concepts found. Exiting.")
        return

    names = sorted(concepts.keys())
    print(f"\n{len(names)} concepts: {names}")

    # Step 2: Extract embeddings via SigLIP
    encoder = SigLIPVisionEncoder(model_path=args.model, device=args.device)
    embeddings = []
    for name in names:
        imgs = concepts[name]
        embs = encoder.extract_batch(imgs)
        avg_emb = embs.mean(axis=0)
        # L2-normalise to match BGE-M3 convention
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-8:
            avg_emb /= norm
        embeddings.append(avg_emb)
        print(f"  [{name}] shape={embs.shape} → avg norm={np.linalg.norm(avg_emb):.4f}")

    embeddings_arr = np.stack(embeddings).astype(np.float32)
    print(f"\nEmbeddings: {embeddings_arr.shape}")

    # Step 3: Save embeddings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embeddings=embeddings_arr, names=np.array(names))
    print(f"Embeddings saved → {args.output}")

    # Step 4: Optionally train probe
    if args.train:
        train_probe(
            embeddings=embeddings_arr,
            names=names,
            epochs=args.epochs,
            lr=args.lr,
            output_path=args.probe_output,
        )

    print("\nDone!")
    print(f"  Embeddings: {args.output}")
    if args.train:
        print(f"  Probe:      {args.probe_output}")
    print("\nNext: load in Rust with NeuralBridge::load(\"models/neural_bridge/probe_weights_siglip.npy\")")


if __name__ == "__main__":
    main()
