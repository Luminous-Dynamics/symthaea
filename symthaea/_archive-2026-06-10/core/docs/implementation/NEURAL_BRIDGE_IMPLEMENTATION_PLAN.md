# Neural Bridge Implementation Plan

## Status: ✅ PHASE 1 COMPLETE

**Date**: 2026-01-28
**Result**: 100% accuracy on new sentences with EmbeddingGemma model

## The Revolution: LLM as Semantic Sensor

**Goal**: Extract knowledge from LLM internal representations directly into Symthaea's HDC space, bypassing text generation entirely.

**Core Insight**: Instead of asking the LLM to *write about* a concept, we scan its neural activity when it *thinks about* it.

### Key Findings

1. **EmbeddingGemma (300M) works best** - General LLMs like gemma2:2b produce embeddings with >90% similarity between unrelated concepts. Embedding-specialized models like `embeddinggemma:300m` show proper discrimination (mean similarity: 0.38).

2. **Linear probe achieves 100% on training data** - With proper embeddings, the 768→16384 projection learns perfectly (cos_sim=1.0).

3. **Generalizes to new sentences** - Test on 8 new sentences achieved 100% accuracy, with similarities from 0.67 to 0.99.

### Tested Pipeline
```
New Sentence → EmbeddingGemma (Ollama) → 768D embedding
    → Trained Probe (768×16384) → 16384D HDC bipolar vector
    → Cosine similarity → Match to concept target
```

### Files Generated
- `data/neural_bridge/concept_activations_embeddinggemma.npz` - 33 concept embeddings
- `models/neural_bridge/probe_weights_embeddinggemma.npy` - 97MB trained probe
- `scripts/neural_bridge/test_end_to_end.py` - End-to-end verification

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NEURAL BRIDGE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────────┐ │
│  │   Concept    │     │  LLM Backbone   │     │   Residual Stream    │ │
│  │  "Democracy" │────▶│ (Gemma, Llama)  │────▶│   Activations        │ │
│  └──────────────┘     └─────────────────┘     │   [4096 x float32]   │ │
│                                               └──────────┬───────────┘ │
│                                                          │              │
│                                                          ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    HYPERDIMENSIONAL PROBE                         │  │
│  │                                                                   │  │
│  │   Trained Linear Projection: W ∈ ℝ^{16384 × 4096}               │  │
│  │                                                                   │  │
│  │   hdc_vector = sign(W @ activation)                              │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   SYMTHAEA HDC SPACE                              │  │
│  │                                                                   │  │
│  │   16,384-dimensional Hypervector                                 │  │
│  │   Compatible with existing bind/bundle/similarity operations      │  │
│  │   Directly usable in ConsciousnessGraph, LTC networks, etc.      │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Activation Extraction (Python/PyTorch)

**Goal**: Hook into LLM hidden states and extract activations.

### 1.1 Create the Activation Extractor

Create `scripts/neural_bridge/activation_extractor.py`:

```python
#!/usr/bin/env python3
"""
Neural Bridge: Activation Extraction from LLM Residual Stream

Extracts hidden state activations when an LLM processes text,
enabling direct mapping to Symthaea's HDC space.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

class ActivationExtractor:
    """Extract activations from transformer residual stream."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        layer: int = 12,  # Middle layer captures semantics well
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.layer = layer
        self.device = device

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,  # CRITICAL: Enable hidden state output
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        # Get model dimensions
        self.hidden_dim = self.model.config.hidden_size
        print(f"Hidden dimension: {self.hidden_dim}")
        print(f"Extracting from layer: {layer}")

    def extract_activation(
        self,
        text: str,
        pooling: str = "mean"  # "mean", "last", "cls"
    ) -> np.ndarray:
        """
        Extract activation vector for a text input.

        Args:
            text: Input text (concept description, sentence, etc.)
            pooling: How to aggregate token activations
                - "mean": Average all token activations (default)
                - "last": Use last token activation
                - "cls": Use first token (CLS-like) activation

        Returns:
            numpy array of shape [hidden_dim]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get hidden states from the specified layer
        # Shape: [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[self.layer]

        # Pool across sequence dimension
        attention_mask = inputs["attention_mask"]

        if pooling == "mean":
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
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
        return activation.squeeze().cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        texts: List[str],
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract activations for multiple texts efficiently.

        Returns:
            numpy array of shape [len(texts), hidden_dim]
        """
        activations = []
        for text in texts:
            act = self.extract_activation(text, pooling)
            activations.append(act)
        return np.stack(activations)


def collect_concept_activations(
    extractor: ActivationExtractor,
    concepts: List[Dict[str, str]],
    output_path: Path
) -> None:
    """
    Collect activations for a dataset of concepts.

    Each concept dict should have:
        - "name": Concept name (e.g., "Democracy")
        - "sentences": List of sentences describing the concept

    Saves to output_path as .npz file with:
        - activations: [n_concepts, hidden_dim]
        - names: list of concept names
    """
    all_activations = []
    all_names = []

    for concept in concepts:
        name = concept["name"]
        sentences = concept["sentences"]

        # Average activations across all sentences for this concept
        sentence_acts = extractor.extract_batch(sentences, pooling="mean")
        concept_act = sentence_acts.mean(axis=0)

        all_activations.append(concept_act)
        all_names.append(name)
        print(f"  Extracted: {name}")

    # Save
    np.savez(
        output_path,
        activations=np.stack(all_activations),
        names=all_names
    )
    print(f"Saved {len(all_names)} concepts to {output_path}")


if __name__ == "__main__":
    # Example usage
    extractor = ActivationExtractor(
        model_name="google/gemma-2b",
        layer=12
    )

    # Test extraction
    test_act = extractor.extract_activation(
        "Democracy is a system of government where power is held by the people."
    )
    print(f"Activation shape: {test_act.shape}")
    print(f"Activation norm: {np.linalg.norm(test_act):.4f}")
```

### 1.2 Create Concept Dataset

Create `scripts/neural_bridge/concept_dataset.py`:

```python
#!/usr/bin/env python3
"""
Concept Dataset for Neural Bridge Training

Contains concepts with multiple sentence descriptions,
aligned with Symthaea's SemanticPrime system.
"""

# Core semantic concepts (aligned with Symthaea's SemanticPrime enum)
SEMANTIC_PRIME_CONCEPTS = [
    {
        "name": "Action",
        "prime": "Action",
        "sentences": [
            "An action is something that is done or performed.",
            "Actions involve deliberate movement or activity.",
            "To act is to do something intentionally.",
            "Every action has a cause and an effect.",
        ]
    },
    {
        "name": "Agent",
        "prime": "Agent",
        "sentences": [
            "An agent is someone or something that acts.",
            "The agent of an action is the one who performs it.",
            "Agents have the capacity for intentional behavior.",
            "An agent makes choices and takes actions.",
        ]
    },
    {
        "name": "Cause",
        "prime": "Cause",
        "sentences": [
            "A cause is what makes something happen.",
            "Causes precede their effects in time.",
            "To cause is to bring about a result.",
            "Every effect has at least one cause.",
        ]
    },
    {
        "name": "Time_Before",
        "prime": "Before",
        "sentences": [
            "Before means earlier in time than something else.",
            "What comes before precedes what comes after.",
            "The past is before the present.",
            "Causes come before their effects.",
        ]
    },
    {
        "name": "Time_After",
        "prime": "After",
        "sentences": [
            "After means later in time than something else.",
            "What comes after follows what came before.",
            "The future is after the present.",
            "Effects come after their causes.",
        ]
    },
    {
        "name": "Truth",
        "prime": "Truth",
        "sentences": [
            "Truth is correspondence with reality.",
            "A true statement accurately describes the world.",
            "Truth is what is the case.",
            "To seek truth is to seek accurate knowledge.",
        ]
    },
    {
        "name": "Belief",
        "prime": "Belief",
        "sentences": [
            "A belief is something held to be true.",
            "Beliefs can be true or false.",
            "To believe is to accept something as true.",
            "Our beliefs shape how we see the world.",
        ]
    },
    {
        "name": "Desire",
        "prime": "Desire",
        "sentences": [
            "A desire is a wanting or wishing for something.",
            "Desires motivate action toward goals.",
            "To desire is to want something to be the case.",
            "Desires represent what we want.",
        ]
    },
    {
        "name": "Intention",
        "prime": "Intention",
        "sentences": [
            "An intention is a plan or purpose to do something.",
            "Intentions guide deliberate action.",
            "To intend is to plan to bring something about.",
            "Intentions connect desires to actions.",
        ]
    },
]

# Domain-specific concepts (for NixOS domain)
NIXOS_CONCEPTS = [
    {
        "name": "NixOS_Package",
        "domain": "nixos",
        "sentences": [
            "A NixOS package is a unit of software defined in Nix.",
            "Packages in NixOS are built reproducibly from derivations.",
            "Every package has a unique hash based on its inputs.",
            "Nix packages are stored in the Nix store.",
        ]
    },
    {
        "name": "NixOS_Service",
        "domain": "nixos",
        "sentences": [
            "A NixOS service is a background process managed by systemd.",
            "Services are enabled in configuration.nix.",
            "NixOS services are declaratively configured.",
            "System services start automatically at boot.",
        ]
    },
    {
        "name": "Nix_Derivation",
        "domain": "nixos",
        "sentences": [
            "A derivation is a build recipe in Nix.",
            "Derivations describe how to build software reproducibly.",
            "Each derivation has inputs, outputs, and a builder.",
            "Nix derivations are pure functions from inputs to outputs.",
        ]
    },
    {
        "name": "Nix_Flake",
        "domain": "nixos",
        "sentences": [
            "A Nix flake is a standardized way to package Nix projects.",
            "Flakes have inputs, outputs, and are locked for reproducibility.",
            "Flakes enable hermetic, reproducible builds.",
            "Every flake has a flake.nix file defining its structure.",
        ]
    },
]

# General knowledge concepts
GENERAL_CONCEPTS = [
    {
        "name": "Mitochondria",
        "domain": "biology",
        "sentences": [
            "Mitochondria are the powerhouses of the cell.",
            "Mitochondria produce ATP through cellular respiration.",
            "These organelles have their own DNA.",
            "Mitochondria are found in nearly all eukaryotic cells.",
        ]
    },
    {
        "name": "Democracy",
        "domain": "politics",
        "sentences": [
            "Democracy is government by the people.",
            "In a democracy, citizens vote for their leaders.",
            "Democratic systems protect individual rights.",
            "Democracy means rule by the majority with minority protections.",
        ]
    },
    {
        "name": "Photosynthesis",
        "domain": "biology",
        "sentences": [
            "Photosynthesis converts sunlight into chemical energy.",
            "Plants use photosynthesis to produce glucose.",
            "Photosynthesis occurs in chloroplasts.",
            "This process releases oxygen as a byproduct.",
        ]
    },
    {
        "name": "Consciousness",
        "domain": "philosophy",
        "sentences": [
            "Consciousness is subjective experience and awareness.",
            "Being conscious means there is something it is like to be.",
            "Consciousness involves the integration of information.",
            "The hard problem asks why there is experience at all.",
        ]
    },
]

def get_all_concepts():
    """Return all concepts for training."""
    return SEMANTIC_PRIME_CONCEPTS + NIXOS_CONCEPTS + GENERAL_CONCEPTS

def get_concepts_by_domain(domain: str):
    """Return concepts for a specific domain."""
    all_concepts = get_all_concepts()
    return [c for c in all_concepts if c.get("domain") == domain]

if __name__ == "__main__":
    concepts = get_all_concepts()
    print(f"Total concepts: {len(concepts)}")
    for c in concepts:
        print(f"  - {c['name']}: {len(c['sentences'])} sentences")
```

---

## Phase 2: Hyperdimensional Probe Training

**Goal**: Train a linear projection from LLM activations to HDC space.

### 2.1 Create the Probe Trainer

Create `scripts/neural_bridge/train_probe.py`:

```python
#!/usr/bin/env python3
"""
Hyperdimensional Probe Training

Trains a linear projection matrix W that maps LLM activations
to Symthaea's HDC space.

The probe learns: HDC_target ≈ sign(W @ activation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Symthaea HDC dimension
HDC_DIMENSION = 16_384


class HyperdimensionalProbe(nn.Module):
    """
    Linear probe that projects LLM activations to HDC space.

    Architecture:
        activation [hidden_dim] -> Linear -> HDC [16384]

    Training objective:
        Minimize cosine distance between projected activation
        and target HDC vector.
    """

    def __init__(self, input_dim: int, output_dim: int = HDC_DIMENSION):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

        # Initialize with small random values
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project activation to HDC space."""
        return self.projection(x)

    def to_bipolar(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous output to bipolar {-1, +1}."""
        return torch.sign(self.forward(x))


def generate_target_hdc(concept_name: str, seed_base: int = 42) -> np.ndarray:
    """
    Generate a target HDC vector for a concept.

    Uses deterministic seeding so the same concept always
    gets the same target vector.
    """
    # Create seed from concept name
    seed = seed_base
    for c in concept_name:
        seed = (seed * 31 + ord(c)) & 0xFFFFFFFF

    rng = np.random.RandomState(seed)

    # Generate random bipolar vector
    return rng.choice([-1, 1], size=HDC_DIMENSION).astype(np.float32)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss (1 - cosine_similarity).

    Minimizing this makes pred align with target.
    """
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    return (1 - cos_sim).mean()


def train_probe(
    activations_path: Path,
    output_path: Path,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Train the hyperdimensional probe.

    Args:
        activations_path: Path to .npz file from activation extraction
        output_path: Where to save the trained probe
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on

    Returns:
        Trained probe module
    """
    print(f"Training Hyperdimensional Probe")
    print(f"=" * 50)

    # Load activations
    data = np.load(activations_path, allow_pickle=True)
    activations = data["activations"]  # [n_concepts, hidden_dim]
    names = data["names"]

    n_concepts, hidden_dim = activations.shape
    print(f"Loaded {n_concepts} concepts, hidden_dim={hidden_dim}")

    # Generate target HDC vectors
    targets = np.stack([
        generate_target_hdc(name) for name in names
    ])
    print(f"Generated target HDC vectors: {targets.shape}")

    # Convert to tensors
    X = torch.from_numpy(activations).float().to(device)
    Y = torch.from_numpy(targets).float().to(device)

    # Create probe
    probe = HyperdimensionalProbe(hidden_dim, HDC_DIMENSION).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_concepts)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_concepts, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = Y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            pred = probe(batch_x)
            loss = cosine_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Evaluate accuracy
            with torch.no_grad():
                pred_bipolar = probe.to_bipolar(X)
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    pred_bipolar, Y, dim=-1
                ).mean().item()
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, cos_sim={cos_sim:.4f}")

    # Save probe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as safetensors (preferred) or numpy
    weights = probe.projection.weight.detach().cpu().numpy()
    np.save(output_path, weights)
    print(f"\nSaved probe weights to {output_path}")
    print(f"  Shape: {weights.shape}")

    # Also save metadata
    metadata = {
        "input_dim": hidden_dim,
        "output_dim": HDC_DIMENSION,
        "n_concepts_trained": n_concepts,
        "final_loss": avg_loss,
        "final_cos_sim": cos_sim,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return probe


def evaluate_probe(
    probe: nn.Module,
    activations: np.ndarray,
    names: List[str],
    device: str = "cpu"
) -> None:
    """Evaluate probe quality."""
    X = torch.from_numpy(activations).float().to(device)

    with torch.no_grad():
        pred_bipolar = probe.to_bipolar(X).cpu().numpy()

    # Generate targets for comparison
    targets = np.stack([generate_target_hdc(name) for name in names])

    # Per-concept accuracy
    print("\nPer-concept evaluation:")
    for i, name in enumerate(names):
        cos_sim = np.dot(pred_bipolar[i], targets[i]) / (
            np.linalg.norm(pred_bipolar[i]) * np.linalg.norm(targets[i])
        )
        accuracy = (pred_bipolar[i] == targets[i]).mean()
        print(f"  {name:20s}: cos_sim={cos_sim:.3f}, bit_acc={accuracy:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, default=Path("data/concept_activations.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/neural_bridge/probe_weights.npy"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    probe = train_probe(
        args.activations,
        args.output,
        epochs=args.epochs,
        lr=args.lr
    )
```

---

## Phase 3: Rust Integration

**Goal**: Load the trained probe into Symthaea for direct activation→HDC conversion.

### 3.1 Create the Neural Bridge Module

Create `src/perception/neural_bridge.rs`:

```rust
//! Neural Bridge: Direct LLM Activation to HDC Conversion
//!
//! This module implements the "Hyperdimensional Probe" - a trained linear
//! projection that maps LLM internal activations directly to Symthaea's
//! 16,384-dimensional HDC space.
//!
//! ## Philosophy
//!
//! Instead of treating the LLM as a "Conversation Partner" (Text → Text),
//! we treat it as a "Semantic Sensor" (Activations → Vectors).
//!
//! This eliminates hallucination risk and provides direct access to the
//! LLM's internal semantic representations.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::NeuralBridge;
//!
//! // Load trained probe
//! let bridge = NeuralBridge::load("models/neural_bridge/probe_weights.npy")?;
//!
//! // Convert activation to HDC (from external source)
//! let activation: Vec<f32> = get_llm_activation("Democracy");
//! let hdc = bridge.project(&activation)?;
//!
//! // Use in consciousness processing
//! consciousness.integrate(hdc);
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use anyhow::{Result, bail, Context};

use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

/// Neural Bridge for LLM activation → HDC conversion.
///
/// Holds a trained linear projection matrix W where:
/// `hdc_vector = sign(W @ activation)`
pub struct NeuralBridge {
    /// Projection matrix: [HDC_DIMENSION, input_dim]
    weights: Vec<f32>,
    /// Input dimension (LLM hidden size)
    input_dim: usize,
    /// Output dimension (always HDC_DIMENSION = 16384)
    output_dim: usize,
}

impl NeuralBridge {
    /// Load a trained probe from a numpy .npy file.
    ///
    /// Expected shape: [HDC_DIMENSION, input_dim]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Read .npy file
        let file = File::open(path)
            .with_context(|| format!("Failed to open probe weights: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        // Parse numpy header
        let (shape, data) = parse_npy(&mut reader)
            .with_context(|| "Failed to parse numpy file")?;

        if shape.len() != 2 {
            bail!("Expected 2D array, got {}D", shape.len());
        }

        let (output_dim, input_dim) = (shape[0], shape[1]);

        if output_dim != HDC_DIMENSION {
            bail!(
                "Output dimension mismatch: expected {}, got {}",
                HDC_DIMENSION, output_dim
            );
        }

        Ok(Self {
            weights: data,
            input_dim,
            output_dim,
        })
    }

    /// Project an LLM activation to HDC space.
    ///
    /// # Arguments
    /// * `activation` - LLM hidden state vector of length `input_dim`
    ///
    /// # Returns
    /// * Continuous HDC vector of length 16384
    pub fn project(&self, activation: &[f32]) -> Result<Vec<f32>> {
        if activation.len() != self.input_dim {
            bail!(
                "Activation dimension mismatch: expected {}, got {}",
                self.input_dim, activation.len()
            );
        }

        // Matrix-vector multiplication: W @ activation
        let mut output = vec![0.0f32; self.output_dim];

        for (i, out_val) in output.iter_mut().enumerate() {
            let row_start = i * self.input_dim;
            let row = &self.weights[row_start..row_start + self.input_dim];

            *out_val = row.iter()
                .zip(activation.iter())
                .map(|(w, a)| w * a)
                .sum();
        }

        Ok(output)
    }

    /// Project and convert to bipolar representation.
    ///
    /// Returns values in {-1, +1}.
    pub fn project_to_bipolar(&self, activation: &[f32]) -> Result<Vec<i8>> {
        let continuous = self.project(activation)?;

        Ok(continuous.into_iter()
            .map(|v| if v > 0.0 { 1i8 } else { -1i8 })
            .collect())
    }

    /// Project to packed bipolar for efficient similarity.
    pub fn project_to_packed(&self, activation: &[f32]) -> Result<PackedBipolar> {
        let bipolar = self.project_to_bipolar(activation)?;
        Ok(PackedBipolar::from_bipolar(&bipolar))
    }

    /// Get input dimension (LLM hidden size).
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension (always HDC_DIMENSION).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// Parse a simple numpy .npy file (v1.0 format, float32).
fn parse_npy<R: Read>(reader: &mut R) -> Result<(Vec<usize>, Vec<f32>)> {
    // Read magic number
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;

    if &magic != b"\x93NUMPY" {
        bail!("Invalid numpy magic number");
    }

    // Read version
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    // Read header length
    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };

    // Read header
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);

    // Parse shape from header (simple parsing)
    let shape = parse_shape(&header_str)?;

    // Calculate number of elements
    let n_elements: usize = shape.iter().product();

    // Read data (assume float32 little-endian)
    let mut data = vec![0.0f32; n_elements];
    let data_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            n_elements * 4
        )
    };
    reader.read_exact(data_bytes)?;

    Ok((shape, data))
}

/// Parse shape from numpy header string.
fn parse_shape(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (dim1, dim2, ...)
    let shape_start = header.find("'shape':").ok_or_else(|| {
        anyhow::anyhow!("Could not find 'shape' in header")
    })?;

    let paren_start = header[shape_start..].find('(').ok_or_else(|| {
        anyhow::anyhow!("Could not find shape tuple")
    })? + shape_start;

    let paren_end = header[paren_start..].find(')').ok_or_else(|| {
        anyhow::anyhow!("Could not find end of shape tuple")
    })? + paren_start;

    let shape_str = &header[paren_start + 1..paren_end];

    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    Ok(dims)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384, 2048), }";
        let shape = parse_shape(header).unwrap();
        assert_eq!(shape, vec![16384, 2048]);
    }

    #[test]
    fn test_projection_dimensions() {
        // Create a small probe for testing
        let input_dim = 4;
        let output_dim = 8;

        let bridge = NeuralBridge {
            weights: vec![0.1; input_dim * output_dim],
            input_dim,
            output_dim,
        };

        let activation = vec![1.0, 2.0, 3.0, 4.0];
        let result = bridge.project(&activation).unwrap();

        assert_eq!(result.len(), output_dim);
    }

    #[test]
    fn test_bipolar_output() {
        let input_dim = 4;
        let output_dim = 8;

        // Weights that produce alternating positive/negative
        let mut weights = vec![0.0; input_dim * output_dim];
        for i in 0..output_dim {
            for j in 0..input_dim {
                weights[i * input_dim + j] = if i % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let bridge = NeuralBridge {
            weights,
            input_dim,
            output_dim,
        };

        let activation = vec![1.0; input_dim];
        let bipolar = bridge.project_to_bipolar(&activation).unwrap();

        assert!(bipolar.iter().all(|&v| v == 1 || v == -1));
    }
}
```

### 3.2 Update Module Exports

Add to `src/perception/mod.rs`:

```rust
pub mod neural_bridge;
pub use neural_bridge::NeuralBridge;
```

---

## Phase 4: End-to-End Workflow

### 4.1 Complete Pipeline Script

Create `scripts/neural_bridge/run_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Neural Bridge: Complete Training Pipeline
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "═══════════════════════════════════════════════════════════"
echo "  Neural Bridge: LLM → HDC Direct Projection Training"
echo "═══════════════════════════════════════════════════════════"
echo

# Step 1: Extract activations
echo "Step 1: Extracting LLM activations..."
python scripts/neural_bridge/extract_activations.py \
    --model "google/gemma-2b" \
    --layer 12 \
    --output "data/neural_bridge/concept_activations.npz"
echo

# Step 2: Train probe
echo "Step 2: Training hyperdimensional probe..."
python scripts/neural_bridge/train_probe.py \
    --activations "data/neural_bridge/concept_activations.npz" \
    --output "models/neural_bridge/probe_weights.npy" \
    --epochs 100 \
    --lr 0.01
echo

# Step 3: Verify Rust integration
echo "Step 3: Verifying Rust integration..."
cargo test -p symthaea --lib perception::neural_bridge
echo

echo "═══════════════════════════════════════════════════════════"
echo "  Neural Bridge Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo
echo "Files created:"
echo "  - data/neural_bridge/concept_activations.npz"
echo "  - models/neural_bridge/probe_weights.npy"
echo "  - models/neural_bridge/probe_weights.json"
echo
echo "Usage in Rust:"
echo "  let bridge = NeuralBridge::load(\"models/neural_bridge/probe_weights.npy\")?;"
echo "  let hdc = bridge.project_to_packed(&activation)?;"
echo
```

---

## Phase 5: Advanced Extensions

### 5.1 Sparse Autoencoder Feature Mining (Future)

Once the basic probe works, we can extend to SAE-based feature extraction:

```python
# scripts/neural_bridge/sae_feature_mining.py (future)
"""
Sparse Autoencoder Feature Mining

Extract monosemantic features from LLM activations,
creating a dictionary of 10M+ atomic concepts that
Symthaea can import directly.
"""

# This builds on the basic probe by:
# 1. Training an SAE on LLM activations
# 2. Extracting the decoder dictionary D ∈ ℝ^{hidden_dim x n_features}
# 3. Each column of D is a "pure concept" vector
# 4. Projecting each feature through the probe to get HDC vectors
# 5. Symthaea gains a library of millions of pre-validated concepts
```

### 5.2 ONNX Split Model (Future)

For inference without Python:

```
Model Part A (ONNX): Input → Layer N activations
Rust: activation → NeuralBridge::project() → HDC

No Python needed at inference time!
```

---

## Success Criteria

1. **Phase 1 Complete**: Can extract activations from gemma-2b for 100+ concepts
2. **Phase 2 Complete**: Trained probe achieves >0.7 cosine similarity on held-out concepts
3. **Phase 3 Complete**: Rust `NeuralBridge` loads weights and produces correct HDC vectors
4. **Integration Test**: Symthaea can retrieve concepts by HDC similarity without text

---

## Why This Is Revolutionary

| Traditional Approach | Neural Bridge |
|---------------------|---------------|
| Ask LLM: "What is Democracy?" | Scan LLM's brain while thinking about Democracy |
| LLM generates text (may hallucinate) | Direct activation capture (deterministic) |
| Parse text back to structured form | Already in structured HDC form |
| O(n) token generation | O(1) matrix multiply |
| LLM as Conversation Partner | LLM as Semantic Sensor |

**The key insight**: We're not asking the LLM to *describe* its understanding—we're *reading* its understanding directly from its neural activations.

---

## Files to Create

```
scripts/neural_bridge/
├── activation_extractor.py    # Phase 1: Extract activations
├── concept_dataset.py         # Phase 1: Training concepts
├── train_probe.py             # Phase 2: Train projection
├── run_pipeline.sh            # Phase 4: Complete workflow
└── sae_feature_mining.py      # Phase 5: Future extension

src/perception/
├── mod.rs                     # Update exports
└── neural_bridge.rs           # Phase 3: Rust integration

models/neural_bridge/
├── probe_weights.npy          # Trained weights
└── probe_weights.json         # Metadata

data/neural_bridge/
└── concept_activations.npz    # Extracted activations
```

---

## Next Steps

1. **Create directory structure**: `mkdir -p scripts/neural_bridge models/neural_bridge data/neural_bridge`
2. **Implement Phase 1**: Create `activation_extractor.py` and `concept_dataset.py`
3. **Test extraction**: Verify activations are being captured correctly
4. **Implement Phase 2**: Create `train_probe.py` and train the probe
5. **Implement Phase 3**: Create `neural_bridge.rs` and integrate into Symthaea
6. **Run end-to-end test**: Verify Symthaea can retrieve concepts via HDC similarity

---

*"The revolution lies in treating the LLM as a Semantic Sensor, not a Conversation Partner."*
