#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Export Qwen3-Embedding model to ONNX format for stateless inference.

This exports the model WITHOUT KV caching, which is needed for embedding
(not text generation). The exported model takes:
- input_ids: [batch, seq_len]
- attention_mask: [batch, seq_len]

And returns:
- last_hidden_state: [batch, seq_len, hidden_dim]

We then mean-pool the last hidden state to get the embedding.

Usage:
    poetry run python scripts/export_qwen3_embedding.py

The model will be saved to models/qwen3-embedding-0.6b/
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
import onnx

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-embedding-0.6b"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

def main():
    print("=" * 60)
    print("Qwen3-Embedding ONNX Export (Stateless)")
    print("=" * 60)
    print()

    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model from HuggingFace: {MODEL_NAME}")
    print("This may take a moment on first run...")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for ONNX compatibility
    )

    # Set to evaluation mode
    model.eval()

    print(f"Model loaded: {model.config.hidden_size} hidden dims")
    print()

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"  Saved to {MODEL_DIR}")
    print()

    # Prepare dummy inputs for export
    print("Preparing ONNX export...")
    dummy_text = "Hello, this is a test input for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"  Input shape: {input_ids.shape}")
    print()

    # Export to ONNX
    print("Exporting to ONNX (this may take a few minutes)...")
    output_path = MODEL_DIR / "model.onnx"

    # Export with dynamic axes for variable sequence length
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"  Exported to {output_path}")
    print()

    # Verify the exported model
    print("Verifying exported model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  Model verification passed!")
    print()

    # Print model info
    print("Model inputs:")
    for inp in onnx_model.graph.input:
        print(f"  - {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")

    print()
    print("Model outputs:")
    for out in onnx_model.graph.output:
        print(f"  - {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    print()
    print("=" * 60)
    print("Export complete!")
    print()
    print("Files created:")
    for f in MODEL_DIR.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  1. Run: cargo run --release --features embeddings --example embedding_verification")
    print("  2. Then: cargo run --release --features embeddings --example toddler_test")
    print("=" * 60)

if __name__ == "__main__":
    main()
