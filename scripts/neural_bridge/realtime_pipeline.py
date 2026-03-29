#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real-Time Neural Bridge Pipeline

Streams text through the Neural Bridge to produce HDC vectors
that can be integrated with Symthaea's consciousness graph.

Pipeline:
    Text → Ollama Embedding → Trained Probe → HDC Vector → Consciousness Integration

Features:
    - Streaming text input (stdin, file, or programmatic)
    - Real-time embedding and projection
    - Concept matching and similarity computation
    - JSON output for Rust integration
    - WebSocket server mode for live applications

Usage:
    # Interactive mode
    python realtime_pipeline.py --interactive

    # Process file
    python realtime_pipeline.py --input thoughts.txt

    # JSON output for Rust integration
    python realtime_pipeline.py --interactive --json

    # WebSocket server (for GUI integration)
    python realtime_pipeline.py --server --port 8765
"""

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# HDC dimension (must match Symthaea)
HDC_DIMENSION = 16_384


@dataclass
class HDCResult:
    """Result from Neural Bridge projection."""

    text: str
    embedding_dim: int
    hdc_dim: int
    processing_time_ms: float
    top_concepts: List[Tuple[str, float]]
    hdc_vector: Optional[List[int]] = None  # Bipolar vector (optional, large)


@dataclass
class ConceptMatch:
    """A matched concept with similarity score."""

    name: str
    similarity: float
    category: str


class NeuralBridgePipeline:
    """
    Real-time Neural Bridge pipeline.

    Connects:
        Ollama (embedding) → Trained Probe → HDC Space → Concept Matching
    """

    def __init__(
        self,
        probe_path: Path,
        concepts_path: Optional[Path] = None,
        model: str = "embeddinggemma:300m",
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize pipeline.

        Args:
            probe_path: Path to trained probe weights (.npy)
            concepts_path: Path to concept activations for matching
            model: Ollama model for embeddings
            ollama_url: Ollama API base URL
        """
        self.model = model
        self.ollama_url = ollama_url

        # Load probe
        print(f"Loading probe from {probe_path}...")
        self.probe_weights = np.load(probe_path)
        self.input_dim = self.probe_weights.shape[1]
        self.output_dim = self.probe_weights.shape[0]
        print(f"  Probe: {self.input_dim} → {self.output_dim}")

        # Load concept targets for matching
        self.concept_names: List[str] = []
        self.concept_targets: Dict[str, np.ndarray] = {}

        if concepts_path and concepts_path.exists():
            print(f"Loading concepts from {concepts_path}...")
            data = np.load(concepts_path, allow_pickle=True)
            self.concept_names = list(data["names"])

            # Generate target HDC vectors for each concept
            for name in self.concept_names:
                self.concept_targets[name] = self._generate_target_hdc(name)

            print(f"  Loaded {len(self.concept_names)} concepts")

    def _generate_target_hdc(
        self, concept_name: str, seed_base: int = 42
    ) -> np.ndarray:
        """Generate deterministic HDC target for concept (matches train_probe.py)."""
        seed = seed_base
        for c in concept_name:
            seed = ((seed * 31) + ord(c)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        return rng.choice([-1, 1], size=HDC_DIMENSION).astype(np.float32)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama."""
        data = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return np.array(result["embeddings"][0], dtype=np.float32)

    def project_to_hdc(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding through probe to HDC space."""
        hdc_continuous = embedding @ self.probe_weights.T
        return np.sign(hdc_continuous).astype(np.int8)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def match_concepts(self, hdc: np.ndarray, top_k: int = 5) -> List[ConceptMatch]:
        """Find closest concepts to HDC vector."""
        if not self.concept_targets:
            return []

        similarities = []
        for name, target in self.concept_targets.items():
            sim = self.cosine_similarity(hdc.astype(np.float32), target)
            similarities.append((name, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k matches
        matches = []
        for name, sim in similarities[:top_k]:
            # Infer category from name (simplified)
            if "Nix" in name:
                category = "nix"
            elif name in ["Joy", "Sadness", "Fear", "Anger", "Love"]:
                category = "emotion"
            else:
                category = "general"

            matches.append(ConceptMatch(name=name, similarity=sim, category=category))

        return matches

    def process(self, text: str, include_vector: bool = False) -> HDCResult:
        """
        Process text through the full pipeline.

        Args:
            text: Input text to process
            include_vector: Whether to include full HDC vector in result

        Returns:
            HDCResult with processing info and matches
        """
        start_time = time.time()

        # Get embedding
        embedding = self.get_embedding(text)

        # Project to HDC
        hdc = self.project_to_hdc(embedding)

        # Match concepts
        matches = self.match_concepts(hdc)

        processing_time = (time.time() - start_time) * 1000

        return HDCResult(
            text=text,
            embedding_dim=len(embedding),
            hdc_dim=len(hdc),
            processing_time_ms=processing_time,
            top_concepts=[(m.name, m.similarity) for m in matches],
            hdc_vector=hdc.tolist() if include_vector else None,
        )

    def process_stream(
        self,
        text_stream,
        callback=None,
        json_output: bool = False,
    ):
        """
        Process a stream of text inputs.

        Args:
            text_stream: Iterator of text strings
            callback: Optional callback(HDCResult) for each result
            json_output: Output results as JSON lines
        """
        for text in text_stream:
            text = text.strip()
            if not text:
                continue

            result = self.process(text)

            if callback:
                callback(result)

            if json_output:
                print(json.dumps(asdict(result)))
            else:
                self._print_result(result)

    def _print_result(self, result: HDCResult):
        """Pretty print a result."""
        print(f"\n{'─' * 60}")
        print(f"Input: {result.text[:60]}{'...' if len(result.text) > 60 else ''}")
        print(f"Processing: {result.processing_time_ms:.1f}ms")
        print(f"Dimensions: {result.embedding_dim} → {result.hdc_dim}")
        print(f"\nTop Concepts:")
        for name, sim in result.top_concepts[:5]:
            bar = "█" * int(sim * 20)
            print(f"  {name:25s} {sim:+.4f} {bar}")


class ConsciousnessIntegration:
    """
    Integration layer for Symthaea's consciousness graph.

    Converts HDC vectors into consciousness-compatible formats
    and provides hooks for real-time field updates.
    """

    def __init__(self, pipeline: NeuralBridgePipeline):
        self.pipeline = pipeline
        self.field_state: Dict[str, float] = {}
        self.recent_vectors: List[np.ndarray] = []
        self.max_history = 100

    def update_field(self, text: str) -> Dict:
        """
        Process text and update consciousness field state.

        Returns dict suitable for Symthaea integration.
        """
        result = self.pipeline.process(text)

        # Update field based on matched concepts
        for name, sim in result.top_concepts:
            # Exponential moving average
            alpha = 0.3
            current = self.field_state.get(name, 0.0)
            self.field_state[name] = alpha * sim + (1 - alpha) * current

        # Store vector for temporal analysis
        if result.hdc_vector:
            self.recent_vectors.append(np.array(result.hdc_vector))
            if len(self.recent_vectors) > self.max_history:
                self.recent_vectors.pop(0)

        return {
            "type": "consciousness_update",
            "timestamp": time.time(),
            "text": text,
            "processing_ms": result.processing_time_ms,
            "top_concepts": result.top_concepts,
            "field_state": dict(
                sorted(self.field_state.items(), key=lambda x: abs(x[1]), reverse=True)[
                    :10
                ]
            ),
            "coherence": self._compute_coherence(),
        }

    def _compute_coherence(self) -> float:
        """
        Compute field coherence from recent vectors.

        Higher coherence = more consistent activation patterns.
        """
        if len(self.recent_vectors) < 2:
            return 1.0

        # Compute average similarity between consecutive vectors
        similarities = []
        for i in range(1, len(self.recent_vectors)):
            v1 = self.recent_vectors[i - 1].astype(np.float32)
            v2 = self.recent_vectors[i].astype(np.float32)
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            similarities.append(sim)

        return float(np.mean(similarities))

    def get_semantic_trajectory(self) -> List[Dict]:
        """
        Get semantic trajectory through concept space.

        Useful for visualizing thought patterns.
        """
        trajectory = []
        for i, vec in enumerate(self.recent_vectors):
            matches = self.pipeline.match_concepts(vec, top_k=3)
            trajectory.append(
                {
                    "index": i,
                    "concepts": [(m.name, m.similarity) for m in matches],
                }
            )
        return trajectory


def interactive_mode(pipeline: NeuralBridgePipeline, json_output: bool = False):
    """Run interactive mode."""
    print("\n" + "=" * 60)
    print("  Neural Bridge - Interactive Mode")
    print("  Type text and press Enter to process")
    print("  Type 'quit' to exit")
    print("=" * 60)

    integration = ConsciousnessIntegration(pipeline)

    while True:
        try:
            text = input("\n> ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue

            update = integration.update_field(text)

            if json_output:
                print(json.dumps(update, indent=2))
            else:
                print(f"\nProcessed in {update['processing_ms']:.1f}ms")
                print(f"Coherence: {update['coherence']:.3f}")
                print("\nTop Concepts:")
                for name, sim in update["top_concepts"][:5]:
                    print(f"  {name:25s} {sim:+.4f}")
                print("\nField State:")
                for name, val in list(update["field_state"].items())[:5]:
                    bar = "█" * int(abs(val) * 20)
                    sign = "+" if val > 0 else "-"
                    print(f"  {name:25s} {sign}{abs(val):.4f} {bar}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Real-time Neural Bridge pipeline")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("models/neural_bridge/probe_weights_embeddinggemma.npy"),
        help="Path to trained probe weights",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path("data/neural_bridge/concept_activations_embeddinggemma.npz"),
        help="Path to concept activations for matching",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="embeddinggemma:300m",
        help="Ollama model for embeddings",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    parser.add_argument("--input", type=Path, help="Input file to process")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    args = parser.parse_args()

    # Check requirements
    if not args.probe.exists():
        print(f"ERROR: Probe weights not found at {args.probe}")
        print("Run train_probe.py first to train the probe.")
        return

    # Initialize pipeline
    pipeline = NeuralBridgePipeline(
        probe_path=args.probe,
        concepts_path=args.concepts if args.concepts.exists() else None,
        model=args.model,
    )

    if args.benchmark:
        # Benchmark mode
        print("\n" + "=" * 60)
        print("  Performance Benchmark")
        print("=" * 60)

        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "NixOS flakes provide reproducible system configurations.",
            "Consciousness emerges from complex neural processes.",
            "Machine learning algorithms optimize through gradient descent.",
            "Democracy requires informed citizen participation.",
        ]

        times = []
        for text in test_texts:
            result = pipeline.process(text)
            times.append(result.processing_time_ms)
            print(f"  {result.processing_time_ms:6.1f}ms: {text[:40]}...")

        print(f"\n  Average: {np.mean(times):.1f}ms")
        print(f"  Std dev: {np.std(times):.1f}ms")
        print(f"  Min:     {np.min(times):.1f}ms")
        print(f"  Max:     {np.max(times):.1f}ms")
        return

    if args.interactive:
        interactive_mode(pipeline, json_output=args.json)
    elif args.input:
        with open(args.input) as f:
            pipeline.process_stream(f, json_output=args.json)
    else:
        # Read from stdin
        pipeline.process_stream(sys.stdin, json_output=args.json)


if __name__ == "__main__":
    main()
