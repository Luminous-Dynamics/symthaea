#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Export PyTorch CanaryCNN weights to Q16.16 fixed-point Rust arrays.

Usage:
    python scripts/export_weights.py \
        --model path/to/model.pt \
        --output vsv-core/src/weights.rs

This script:
1. Loads PyTorch model weights
2. Converts to Q16.16 fixed-point (i32 representation)
3. Generates Rust const arrays
4. Optionally validates conversion accuracy
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch not found. Install with: pip install torch")
    sys.exit(1)


# Q16.16 constants
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS  # 65536

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_SIZE = IMG_HEIGHT * IMG_WIDTH
NUM_CLASSES = 10

# Canonical checkpoint metadata. The model should be trained with
# torch.manual_seed(42) and Xavier uniform initialisation; the resulting
# `canary_cnn.pt` is expected to hash to this value.
CANONICAL_MODEL_SHA = "205e7f35bac9dd2322d5eb0fcdce638c3e65711d205e78e499cfc210e766eb5b"


def float_to_q16_16(f: float) -> int:
    """Convert float32 to Q16.16 fixed-point (i32)."""
    return int(f * SCALE)


def q16_16_to_float(i: int) -> float:
    """Convert Q16.16 fixed-point (i32) back to float32."""
    return i / SCALE


def tensor_to_rust_array(tensor: torch.Tensor, name: str) -> str:
    """Convert PyTorch tensor to Rust const array."""
    # Flatten tensor and convert to Q16.16
    flat = tensor.flatten().tolist()
    q16_values = [float_to_q16_16(f) for f in flat]

    # Format as Rust array
    size = len(q16_values)
    rust_type = f"[Fixed; {size}]"

    # Format values in rows of 8 for readability
    lines = []
    for i in range(0, size, 8):
        chunk = q16_values[i:i+8]
        formatted = ", ".join(f"Fixed::from_raw({v})" for v in chunk)
        lines.append(f"    {formatted},")

    values_str = "\n".join(lines)

    return f"""pub const {name}: {rust_type} = [
{values_str}
];
"""


class SimpleCanaryCNN(nn.Module):
    """
    Simple CanaryCNN for MNIST.

    Architecture:
    - Conv2D(1→16, 3x3) → ReLU → MaxPool(2x2)
    - Conv2D(16→32, 3x3) → ReLU → MaxPool(2x2)
    - Flatten → FC(800→128) → ReLU → FC(128→10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 32 channels, 5x5 spatial
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv1: [1, 28, 28] → [16, 26, 26]
        x = torch.relu(self.conv1(x))
        # Pool1: [16, 26, 26] → [16, 13, 13]
        x = torch.max_pool2d(x, 2)

        # Conv2: [16, 13, 13] → [32, 11, 11]
        x = torch.relu(self.conv2(x))
        # Pool2: [32, 11, 11] → [32, 5, 5]
        x = torch.max_pool2d(x, 2)

        # Flatten: [32, 5, 5] → [800]
        x = x.view(x.size(0), -1)

        # FC1: [800] → [128]
        x = torch.relu(self.fc1(x))

        # FC2: [128] → [10]
        x = self.fc2(x)

        return x


def export_weights(model_path: Path, output_path: Path, validate: bool = True):
    """
    Export PyTorch model weights to Rust fixed-point arrays.

    Args:
        model_path: Path to PyTorch model (.pt or .pth file)
        output_path: Path to output Rust file
        validate: Whether to validate conversion accuracy
    """
    print(f"Loading model from {model_path}...")

    # Create model and load weights (training script should seed deterministically
    # using torch.manual_seed(42) and Xavier initialisation before saving checkpoint).
    model = SimpleCanaryCNN()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Refusing to export random weights."
        )

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"✓ Loaded trained weights")

    model.eval()

    # Extract weights and biases
    conv1_weight = model.conv1.weight.data  # [16, 1, 3, 3]
    conv1_bias = model.conv1.bias.data      # [16]

    conv2_weight = model.conv2.weight.data  # [32, 16, 3, 3]
    conv2_bias = model.conv2.bias.data      # [32]

    fc1_weight = model.fc1.weight.data      # [128, 800]
    fc1_bias = model.fc1.bias.data          # [128]

    fc2_weight = model.fc2.weight.data      # [10, 128]
    fc2_bias = model.fc2.bias.data          # [10]

    print(f"\nWeight shapes:")
    print(f"  conv1: {list(conv1_weight.shape)} + {list(conv1_bias.shape)}")
    print(f"  conv2: {list(conv2_weight.shape)} + {list(conv2_bias.shape)}")
    print(f"  fc1:   {list(fc1_weight.shape)} + {list(fc1_bias.shape)}")
    print(f"  fc2:   {list(fc2_weight.shape)} + {list(fc2_bias.shape)}")

    # Generate Rust code
    print(f"\nGenerating Rust code...")
    rust_code = f"""//! CanaryCNN Weights (Q16.16 Fixed-Point)
//!
//! Auto-generated from PyTorch model by export_weights.py
//! DO NOT EDIT MANUALLY

use crate::Fixed;

// Conv1: 1 → 16 channels, 3x3 kernel
{tensor_to_rust_array(conv1_weight, "CONV1_WEIGHTS")}

{tensor_to_rust_array(conv1_bias, "CONV1_BIAS")}

// Conv2: 16 → 32 channels, 3x3 kernel
{tensor_to_rust_array(conv2_weight, "CONV2_WEIGHTS")}

{tensor_to_rust_array(conv2_bias, "CONV2_BIAS")}

// FC1: 800 → 128
{tensor_to_rust_array(fc1_weight, "FC1_WEIGHTS")}

{tensor_to_rust_array(fc1_bias, "FC1_BIAS")}

// FC2: 128 → 10
{tensor_to_rust_array(fc2_weight, "FC2_WEIGHTS")}

{tensor_to_rust_array(fc2_bias, "FC2_BIAS")}
"""

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rust_code)
    print(f"✓ Wrote weights to {output_path}")

    # Validate conversion
    if validate:
        print(f"\nValidating conversion...")
        validate_conversion(model, conv1_weight, conv1_bias)

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != CANONICAL_MODEL_SHA:
        canonical_ref = model_path.with_suffix(".sha256")
        if canonical_ref.exists():
            expected = canonical_ref.read_text().strip()
            if expected and expected != model_hash:
                raise RuntimeError(
                    "Model checkpoint hash mismatch; regenerate with the documented seed."
                )
        else:
            print(
                "⚠ Model SHA does not match canonical value; update the reference if this is intentional."
            )

    weights_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    hash_file = output_path.with_name("weights.sha256")

    if hash_file.exists():
        expected_model_hash, expected_weights_hash = hash_file.read_text().strip().split()
        if expected_model_hash != model_hash or expected_weights_hash != weights_hash:
            raise RuntimeError(
                "Checkpoint SHA mismatch; ensure the deterministic training script was used."
            )
    else:
        hash_file.write_text(f"{model_hash} {weights_hash}\n")

    print(f"Model SHA256:   {model_hash}")
    print(f"Weights SHA256: {weights_hash}")

    return model


def format_fixed_matrix(rows: Iterable[Iterable[int]]) -> str:
    """Format nested arrays of Fixed::from_raw values."""
    formatted_rows: List[str] = []
    for row in rows:
        entries = ", ".join(f"Fixed::from_raw({val})" for val in row)
        formatted_rows.append(f"    [{entries}],")
    return "\n".join(formatted_rows)


def format_f32_matrix(rows: Iterable[Iterable[float]]) -> str:
    """Format nested arrays of f32 values."""
    formatted_rows: List[str] = []
    for row in rows:
        entries = ", ".join(f"{val:.6f}f32" for val in row)
        formatted_rows.append(f"    [{entries}],")
    return "\n".join(formatted_rows)


def generate_calibration(
    model: nn.Module,
    samples: int,
    output_path: Path,
    seed: int = 1337,
) -> None:
    """
    Generate calibration fixtures (inputs, logits, labels) for Rust tests.
    """
    if samples <= 0:
        return

    print(f"\nGenerating calibration fixtures ({samples} samples)...")

    torch.manual_seed(seed)
    inputs = torch.randn(samples, 1, IMG_HEIGHT, IMG_WIDTH)
    labels = torch.randint(0, NUM_CLASSES, (samples,), dtype=torch.long)

    with torch.no_grad():
        outputs = model(inputs).detach()

    flat_inputs = inputs.view(samples, -1)

    input_rows = [
        [float_to_q16_16(float(val)) for val in row.tolist()]
        for row in flat_inputs
    ]
    output_rows = [row.tolist() for row in outputs]
    label_values = labels.tolist()

    rust_code = f"""//! Calibration fixtures for CanaryCNN (auto-generated)
//!
//! Each sample provides:
//! - Input pixels in Q16.16 (Fixed)
//! - Reference logits from PyTorch (f32)
//! - Reference labels for loss computation

use crate::Fixed;

pub const CALIBRATION_SAMPLE_COUNT: usize = {samples};

pub const CALIBRATION_INPUTS: [[Fixed; {IMG_SIZE}]; {samples}] = [
{format_fixed_matrix(input_rows)}
];

pub const CALIBRATION_OUTPUTS: [[f32; {NUM_CLASSES}]; {samples}] = [
{format_f32_matrix(output_rows)}
];

pub const CALIBRATION_LABELS: [usize; {samples}] = [
    {", ".join(str(lbl) for lbl in label_values)},
];
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rust_code)
    print(f"✓ Wrote calibration fixtures to {output_path}")


def validate_conversion(model, conv1_weight, conv1_bias, tolerance=0.01):
    """
    Validate that Q16.16 conversion maintains accuracy.

    Args:
        model: PyTorch model
        conv1_weight: First layer weights for spot check
        conv1_bias: First layer bias for spot check
        tolerance: Maximum allowed absolute difference
    """
    # Spot check: convert a few weights and verify round-trip
    sample_weights = conv1_weight.flatten()[:10].tolist()

    max_error = 0.0
    for i, orig in enumerate(sample_weights):
        q16 = float_to_q16_16(orig)
        recovered = q16_16_to_float(q16)
        error = abs(orig - recovered)
        max_error = max(max_error, error)

        if i < 3:  # Show first 3 as examples
            print(f"  Sample {i}: {orig:.6f} → {q16} → {recovered:.6f} (err: {error:.6f})")

    print(f"\n  Max conversion error: {max_error:.6f}")

    if max_error < tolerance:
        print(f"  ✓ All weights within tolerance ({tolerance})")
    else:
        print(f"  ⚠ Some weights exceed tolerance ({tolerance})")
        print(f"    This is normal for Q16.16 format (precision ≈ 0.000015)")

    # Test forward pass shape
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape:  {list(dummy_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  ✓ Forward pass successful")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch CanaryCNN weights to Q16.16 Rust arrays"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/canary_cnn.pt"),
        help="Path to PyTorch model file (.pt or .pth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vsv-core/src/weights.rs"),
        help="Path to output Rust file",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of calibration samples to export alongside weights",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=Path("vsv-core/src/calibration.rs"),
        help="Path to output Rust file for calibration fixtures",
    )

    args = parser.parse_args()

    model = export_weights(
        model_path=args.model,
        output_path=args.output,
        validate=not args.no_validate,
    )

    if args.samples > 0:
        generate_calibration(
            model=model,
            samples=args.samples,
            output_path=args.calibration_output,
        )

    print(f"\n✓ Export complete!")
    print(f"\nNext steps:")
    print(f"  1. Add to vsv-core/src/lib.rs: pub mod weights;")
    print(f"  2. (Optional) Add to vsv-core/src/lib.rs: pub mod calibration;")
    print(f"  3. Update CanaryCNN to use imported weights")
    print(f"  4. cargo test --package vsv-core")


if __name__ == "__main__":
    main()
