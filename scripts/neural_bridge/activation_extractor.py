#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Neural Bridge: Activation Extraction from LLM Residual Stream

Extracts hidden state activations when an LLM processes text,
enabling direct mapping to Symthaea's HDC space.

This is the core of the "LLM as Semantic Sensor" paradigm.

Supports multiple backends:
  - ollama: Use Ollama's embedding API (recommended, works with local models)
  - torch: Use HuggingFace transformers (requires PyTorch)
  - simulate: Use deterministic random vectors (for testing)

Usage:
    # Extract using Ollama (recommended)
    python activation_extractor.py --backend ollama --model gemma2:2b

    # Extract using HuggingFace transformers
    python activation_extractor.py --backend torch --model google/gemma-2b --layer 12

    # Simulate (for testing)
    python activation_extractor.py --backend simulate
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Check for torch (optional)
try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except:
        return False


HAS_OLLAMA = check_ollama_available()


class OllamaExtractor:
    """Extract embeddings using Ollama's embedding API."""

    def __init__(self, model_name: str = "gemma2:2b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"

        # Test connection and get model info
        print(f"Connecting to Ollama...")
        if not HAS_OLLAMA:
            raise RuntimeError("Ollama is not running. Start with: ollama serve")

        # Get embedding dimension by doing a test call
        print(f"  Model: {model_name}")
        test_embedding = self._get_embedding("test")
        self.hidden_dim = len(test_embedding)
        print(f"  Hidden dimension: {self.hidden_dim}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        data = json.dumps({"model": self.model_name, "input": text}).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                # Ollama returns {"embeddings": [[...]]}
                return np.array(result["embeddings"][0], dtype=np.float32)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "No error body"
            raise RuntimeError(f"Ollama API error: {e.code} - {error_body}")

    def extract_activation(self, text: str, pooling: str = "mean") -> np.ndarray:
        """Extract embedding for text. Pooling is ignored (Ollama does its own pooling)."""
        return self._get_embedding(text)

    def extract_batch(
        self, texts: List[str], pooling: str = "mean", show_progress: bool = True
    ) -> np.ndarray:
        """Extract embeddings for multiple texts."""
        activations = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"    Processing {i + 1}/{len(texts)}...")
            act = self.extract_activation(text, pooling)
            activations.append(act)
        return np.stack(activations)


class SimulatedExtractor:
    """Generate deterministic random vectors for testing."""

    def __init__(self, hidden_dim: int = 2048, model_name: str = "simulated"):
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        print(f"[SIMULATION] Using simulated activations (hidden_dim={hidden_dim})")

    def extract_activation(self, text: str, pooling: str = "mean") -> np.ndarray:
        """Generate deterministic random vector based on text hash."""
        seed = hash(text) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        return rng.randn(self.hidden_dim).astype(np.float32)

    def extract_batch(
        self, texts: List[str], pooling: str = "mean", show_progress: bool = True
    ) -> np.ndarray:
        """Extract simulated activations for multiple texts."""
        activations = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"    Processing {i + 1}/{len(texts)}...")
            act = self.extract_activation(text, pooling)
            activations.append(act)
        return np.stack(activations)


class TorchExtractor:
    """Extract activations from transformer residual stream using PyTorch."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        layer: int = 12,
        device: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch/transformers not available. Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.layer = layer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading {model_name}...")
        print(f"  Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,  # CRITICAL: Enable hidden state output
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        # Get model dimensions
        self.hidden_dim = self.model.config.hidden_size
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Extracting from layer: {layer}")

    def extract_activation(self, text: str, pooling: str = "mean") -> np.ndarray:
        """
        Extract activation vector for a text input.

        Args:
            text: Input text (concept description, sentence, etc.)
            pooling: How to aggregate token activations
                - "mean": Average all token activations (default)
                - "last": Use last token activation
                - "cls": Use first token activation

        Returns:
            numpy array of shape [hidden_dim]
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get hidden states from the specified layer
        # outputs.hidden_states is a tuple of (n_layers + 1,) tensors
        # Each tensor has shape [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[self.layer]

        # Pool across sequence dimension
        attention_mask = inputs["attention_mask"]

        if pooling == "mean":
            # Masked mean pooling
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            activation = sum_embeddings / sum_mask
        elif pooling == "last":
            # Last non-padding token
            seq_lens = attention_mask.sum(dim=1) - 1
            activation = hidden_states[0, seq_lens[0], :]
        elif pooling == "cls":
            # First token
            activation = hidden_states[0, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Return as numpy
        return activation.squeeze().cpu().float().numpy().astype(np.float32)

    def extract_batch(
        self, texts: List[str], pooling: str = "mean", show_progress: bool = True
    ) -> np.ndarray:
        """Extract activations for multiple texts."""
        activations = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"    Processing {i + 1}/{len(texts)}...")
            act = self.extract_activation(text, pooling)
            activations.append(act)
        return np.stack(activations)


# Unified extractor that auto-selects backend
class ActivationExtractor:
    """Unified activation extractor supporting multiple backends."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: Optional[str] = None,
        layer: int = 12,
        device: Optional[str] = None,
    ):
        """
        Create activation extractor.

        Args:
            backend: "ollama", "torch", "simulate", or "auto" (auto-detect)
            model_name: Model name (backend-specific)
            layer: Layer to extract from (torch backend only)
            device: Device (torch backend only)
        """
        if backend == "auto":
            # Auto-detect best available backend
            if HAS_OLLAMA:
                backend = "ollama"
            elif HAS_TORCH:
                backend = "torch"
            else:
                backend = "simulate"
            print(f"Auto-detected backend: {backend}")

        self.backend = backend

        if backend == "ollama":
            model = model_name or "gemma2:2b"
            self._extractor = OllamaExtractor(model)
        elif backend == "torch":
            model = model_name or "google/gemma-2b"
            self._extractor = TorchExtractor(model, layer, device)
        elif backend == "simulate":
            self._extractor = SimulatedExtractor(2048, model_name or "simulated")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.model_name = self._extractor.model_name
        self.hidden_dim = self._extractor.hidden_dim
        self.layer = layer if backend == "torch" else 0

    def extract_activation(self, text: str, pooling: str = "mean") -> np.ndarray:
        return self._extractor.extract_activation(text, pooling)

    def extract_batch(
        self, texts: List[str], pooling: str = "mean", show_progress: bool = True
    ) -> np.ndarray:
        return self._extractor.extract_batch(texts, pooling, show_progress)


def collect_concept_activations(
    extractor: ActivationExtractor,
    concepts: List[Dict],
    output_path: Path,
    pooling: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    Collect activations for a dataset of concepts.

    Each concept dict should have:
        - "name": Concept name (e.g., "Democracy")
        - "sentences": List of sentences describing the concept

    For each concept, we:
    1. Extract activations for all sentences
    2. Average them to get a single concept activation

    Saves to output_path as .npz file with:
        - activations: [n_concepts, hidden_dim]
        - names: list of concept names
    """
    print(f"\nExtracting activations for {len(concepts)} concepts...")

    all_activations = []
    all_names = []

    for concept in concepts:
        name = concept["name"]
        sentences = concept["sentences"]

        # Extract activations for all sentences
        sentence_acts = extractor.extract_batch(sentences, pooling, show_progress=False)

        # Average to get concept activation
        concept_act = sentence_acts.mean(axis=0)

        all_activations.append(concept_act)
        all_names.append(name)
        print(f"  {name}: {len(sentences)} sentences -> mean activation")

    # Stack into array
    activations = np.stack(all_activations)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        activations=activations,
        names=all_names,
        hidden_dim=extractor.hidden_dim,
        model_name=extractor.model_name,
        layer=extractor.layer,
    )
    print(f"\nSaved {len(all_names)} concepts to {output_path}")
    print(f"  Shape: {activations.shape}")

    return activations, all_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM activations for Neural Bridge training"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "ollama", "torch", "simulate"],
        help="Backend: ollama (local models), torch (HuggingFace), simulate (testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Ollama: gemma2:2b, llama3.2:3b. Torch: google/gemma-2b",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer to extract from (torch backend only, default: 12)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/neural_bridge/concept_activations.npz"),
        help="Output path for activations",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last", "cls"],
        help="Pooling strategy (default: mean)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, torch backend only, auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Import concept dataset (try expanded first, fall back to original)
    try:
        from concept_dataset_expanded import get_all_concepts

        print("Using EXPANDED concept dataset (122 concepts)")
    except ImportError:
        from concept_dataset import get_all_concepts

        print("Using original concept dataset (33 concepts)")

    print("=" * 60)
    print("  Neural Bridge: Activation Extraction")
    print("=" * 60)
    print()
    print(f"Backend availability: Ollama={HAS_OLLAMA}, PyTorch={HAS_TORCH}")
    print()

    # Create extractor
    extractor = ActivationExtractor(
        backend=args.backend,
        model_name=args.model,
        layer=args.layer,
        device=args.device,
    )

    # Get concepts
    concepts = get_all_concepts()
    print(f"\nLoaded {len(concepts)} concepts from dataset")

    # Extract activations
    activations, names = collect_concept_activations(
        extractor, concepts, args.output, pooling=args.pooling
    )

    # Print summary
    print()
    print("=" * 60)
    print("  Extraction Complete!")
    print("=" * 60)
    print()
    print(f"Output: {args.output}")
    print(f"Concepts: {len(names)}")
    print(f"Hidden dim: {extractor.hidden_dim}")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print()

    # Quick verification
    print("Sample activations (first 5 concepts):")
    for i in range(min(5, len(names))):
        norm = np.linalg.norm(activations[i])
        print(
            f"  {names[i]:20s}: norm={norm:.4f}, range=[{activations[i].min():.3f}, {activations[i].max():.3f}]"
        )


if __name__ == "__main__":
    main()
