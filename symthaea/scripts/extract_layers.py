#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Extract intermediate layer activations from transformer models using HuggingFace.

This script enables full cross-architecture validation by extracting activations
from ALL layers of transformer models (XLM-RoBERTa, BERT, etc.) and outputting
them in a JSON format compatible with Rust code.

Supports testing the phenomenal corridor hypothesis (Φ at ~92% depth) across architectures.

Example usage:
    python extract_layers.py --model xlm-roberta-base --text "The experience of seeing red" --output activations.json
    python extract_layers.py --model bert-base-uncased --texts-file inputs.txt --output activations.json
    python extract_layers.py --model xlm-roberta-base --text "Hello world"  # outputs to stdout
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    """
    Load a HuggingFace model and tokenizer with hidden states output enabled.

    Args:
        model_name: HuggingFace model identifier (e.g., 'xlm-roberta-base', 'bert-base-uncased')

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return model, tokenizer


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mean pooling to hidden states, accounting for attention mask.

    Args:
        hidden_state: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len)

    Returns:
        Mean-pooled tensor of shape (batch_size, hidden_dim)
    """
    # Expand attention mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()

    # Sum hidden states weighted by mask
    sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)

    # Count non-masked tokens
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    # Compute mean
    return sum_hidden / sum_mask


def extract_layer_activations(
    model, tokenizer, texts: list[str], device: str = "cpu"
) -> list[dict]:
    """
    Extract activations from all layers of a transformer model.

    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        texts: List of input texts to process
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        List of dictionaries containing layer activations for each text
    """
    model = model.to(device)
    results = []

    for text in texts:
        # Tokenize input
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract all hidden states (including embedding layer)
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding output, indices 1-N are transformer layers
        hidden_states = outputs.hidden_states

        # Process each layer
        layers_data = []
        attention_mask = inputs["attention_mask"]

        for layer_idx, hidden_state in enumerate(hidden_states):
            # Apply mean pooling
            pooled = mean_pool(hidden_state, attention_mask)

            # Convert to list for JSON serialization
            activation = pooled.squeeze(0).cpu().tolist()

            layers_data.append({"layer": layer_idx, "activation": activation})

        result = {
            "model": (
                model.config.name_or_path
                if hasattr(model.config, "name_or_path")
                else "unknown"
            ),
            "text": text,
            "num_layers": len(hidden_states),
            "hidden_dim": hidden_states[0].shape[-1],
            "layers": layers_data,
        }
        results.append(result)

    return results


def process_batch(model_name: str, texts: list[str], device: str = "cpu") -> list[dict]:
    """
    Process a batch of texts through a model.

    Args:
        model_name: HuggingFace model identifier
        texts: List of input texts
        device: Device to run inference on

    Returns:
        List of activation dictionaries
    """
    model, tokenizer = load_model_and_tokenizer(model_name)
    return extract_layer_activations(model, tokenizer, texts, device)


def main():
    parser = argparse.ArgumentParser(
        description="Extract intermediate layer activations from transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text, output to file
  python extract_layers.py --model xlm-roberta-base --text "The experience of seeing red" --output activations.json

  # Single text, output to stdout
  python extract_layers.py --model bert-base-uncased --text "Hello world"

  # Multiple texts from file
  python extract_layers.py --model xlm-roberta-base --texts-file inputs.txt --output activations.json

  # Use GPU acceleration
  python extract_layers.py --model xlm-roberta-base --text "Test" --device cuda

Supported models:
  - xlm-roberta-base, xlm-roberta-large
  - bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased
  - Any HuggingFace transformer model with hidden_states output
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., xlm-roberta-base, bert-base-uncased)",
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", "-t", type=str, help="Single input text to process"
    )
    text_group.add_argument(
        "--texts-file",
        "-f",
        type=str,
        help="Path to file with one text per line for batch processing",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )

    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    # Collect input texts
    if args.text:
        texts = [args.text]
    else:
        texts_path = Path(args.texts_file)
        if not texts_path.exists():
            print(f"Error: texts file not found: {args.texts_file}", file=sys.stderr)
            sys.exit(1)
        with open(texts_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print("Error: no input texts provided", file=sys.stderr)
        sys.exit(1)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available, falling back to CPU",
            file=sys.stderr,
        )
        args.device = "cpu"

    # Process texts
    try:
        results = process_batch(args.model, texts, args.device)
    except Exception as e:
        print(f"Error processing texts: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare output
    # If single text, output single object; if multiple, output array
    output_data = (
        results[0] if len(results) == 1 else {"batch": results, "count": len(results)}
    )

    # Format JSON
    indent = 2 if args.pretty else None
    json_output = json.dumps(output_data, indent=indent)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Activations written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
