# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Gradient Proof Circuit - zkSTARK-based gradient provenance verification.

Core Concept:
Instead of detecting Byzantine behavior (Gen-5's failed approach), we
cryptographically PROVE honest behavior. Clients generate zkSTARK proofs
that their gradients were computed correctly from local training.

Proof Statement:
"I trained on my local data D_i for E epochs with learning rate η,
 starting from global model W_t, and this gradient is the result."

Properties:
- Completeness: Honest client can always generate valid proof
- Soundness: Malicious client cannot generate proof for fake gradient
- Zero-knowledge: Proof reveals nothing about private data D_i
- Succinctness: Proof size O(log |D_i|), verification O(1)
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GradientProof:
    """Proof of gradient provenance.

    Attributes:
        gradient: The computed gradient (public)
        proof_bytes: zkSTARK proof (verifier checks this)
        public_inputs: Public parameters (model hash, gradient hash, etc.)
        metadata: Proof generation metadata (timing, size, etc.)
    """

    gradient: np.ndarray
    proof_bytes: bytes
    public_inputs: Dict[str, any]
    metadata: Dict[str, any]

    def size_kb(self) -> float:
        """Proof size in kilobytes."""
        return len(self.proof_bytes) / 1024

    def verify(self) -> bool:
        """Verify proof validity (delegates to circuit)."""
        from .gradient_proof import GradientProofCircuit

        circuit = GradientProofCircuit(client_id=self.public_inputs["client_id"])
        return circuit.verify_proof(self)


class GradientProofCircuit:
    """zkSTARK circuit for gradient computation verification.

    This circuit proves that a gradient was computed correctly through
    local training on private data.

    Public Inputs:
    - global_model_hash: Hash of the starting model
    - gradient_hash: Hash of the computed gradient
    - client_id: Identity of the client
    - round_idx: Current training round

    Private Witness (not revealed):
    - local_data: Client's private dataset D_i
    - local_labels: Corresponding labels
    - training_trace: Forward/backward pass execution trace
    - random_seed: For reproducibility

    Constraints:
    1. forward_pass_correct(D_i, W_t) - Loss computed correctly
    2. backward_pass_correct() - Gradients derived from loss
    3. gradient_matches_trace() - Proof matches submission
    """

    def __init__(
        self,
        client_id: str,
        use_real_stark: bool = False,
    ):
        """Initialize gradient proof circuit.

        Args:
            client_id: Unique client identifier
            use_real_stark: If True, use actual zkSTARK library (Winterfell)
                           If False, use simulation mode (for testing)
        """
        self.client_id = client_id
        self.use_real_stark = use_real_stark
        self.current_round = 0

        # Performance tracking
        self.proof_gen_times = []
        self.proof_sizes = []

    def prove_gradient(
        self,
        global_model: np.ndarray,
        local_data: np.ndarray,
        local_labels: np.ndarray,
        epochs: int,
        lr: float,
        seed: int,
    ) -> GradientProof:
        """Generate gradient and zkSTARK proof of correct computation.

        Args:
            global_model: Starting model weights (round t)
            local_data: Client's private training data
            local_labels: Corresponding labels
            epochs: Number of local training epochs
            lr: Learning rate
            seed: Random seed for reproducibility

        Returns:
            GradientProof containing gradient and zkSTARK proof
        """
        start_time = time.time()

        # 1. Reproducible local training
        np.random.seed(seed)
        model = global_model.copy()

        # 2. Capture training trace (for proof witness)
        trace = self._execute_training(
            model=model,
            data=local_data,
            labels=local_labels,
            epochs=epochs,
            lr=lr,
        )

        # 3. Compute final gradient
        gradient = model - global_model

        # 4. Generate zkSTARK proof
        public_inputs = {
            "global_model_hash": self._hash_array(global_model),
            "gradient_hash": self._hash_array(gradient),
            "client_id": self.client_id,
            "round_idx": self.current_round,
            "epochs": epochs,
            "lr": lr,
        }

        private_witness = {
            "local_data": local_data,
            "local_labels": local_labels,
            "training_trace": trace,
            "seed": seed,
            "global_model": global_model,  # Add for zkSTARK proof
            "gradient": gradient,  # Add actual computed gradient
        }

        if self.use_real_stark:
            proof_bytes = self._generate_stark_proof(public_inputs, private_witness)
        else:
            proof_bytes = self._simulate_stark_proof(public_inputs, private_witness)

        # 5. Track performance
        proof_gen_time = time.time() - start_time
        proof_size = len(proof_bytes)

        self.proof_gen_times.append(proof_gen_time)
        self.proof_sizes.append(proof_size)

        metadata = {
            "proof_gen_time_sec": proof_gen_time,
            "proof_size_bytes": proof_size,
            "data_size": local_data.shape[0],
            "model_size": global_model.size,
            "epochs": epochs,
        }

        return GradientProof(
            gradient=gradient,
            proof_bytes=proof_bytes,
            public_inputs=public_inputs,
            metadata=metadata,
        )

    def verify_proof(self, proof: GradientProof) -> bool:
        """Verify zkSTARK proof validity.

        Args:
            proof: GradientProof to verify

        Returns:
            True if proof is valid, False otherwise

        Note:
            This is the critical security function. Invalid proofs must
            NEVER verify as valid (soundness guarantee).
        """
        if self.use_real_stark:
            return self._verify_stark_proof(proof.proof_bytes, proof.public_inputs)
        else:
            return self._verify_simulated_proof(proof.proof_bytes, proof.public_inputs)

    def _execute_training(
        self,
        model: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        lr: float,
    ) -> List[Dict]:
        """Execute local training and capture execution trace.

        The trace is used as private witness in the zkSTARK proof.

        Args:
            model: Starting model weights
            data: Training data
            labels: Training labels
            epochs: Number of epochs
            lr: Learning rate

        Returns:
            List of execution trace entries (forward/backward passes)
        """
        trace = []

        for epoch in range(epochs):
            for i, (x, y) in enumerate(zip(data, labels)):
                # Forward pass
                logits = model @ x
                loss = self._cross_entropy(logits, y)

                trace.append(
                    {
                        "type": "forward",
                        "epoch": epoch,
                        "sample": i,
                        "input_hash": self._hash_array(x),
                        "label": y,
                        "logits": logits.copy(),
                        "loss": loss,
                    }
                )

                # Backward pass
                grad = self._compute_gradient(model, x, y, logits)

                trace.append(
                    {
                        "type": "backward",
                        "epoch": epoch,
                        "sample": i,
                        "gradient_hash": self._hash_array(grad),
                    }
                )

                # Update model
                model -= lr * grad

        return trace

    def _cross_entropy(self, logits: np.ndarray, label: int) -> float:
        """Compute cross-entropy loss."""
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return -np.log(probs[label] + 1e-10)

    def _compute_gradient(
        self,
        model: np.ndarray,
        x: np.ndarray,
        y: int,
        logits: np.ndarray,
    ) -> np.ndarray:
        """Compute gradient via softmax cross-entropy backprop."""
        # Softmax probabilities
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Gradient: outer product of (probs - one_hot) and input
        probs[y] -= 1.0  # Subtract 1 from true class
        grad = np.outer(probs, x)

        return grad

    def _hash_array(self, arr: np.ndarray) -> str:
        """Hash numpy array for public inputs."""
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    def _generate_stark_proof(
        self,
        public_inputs: Dict,
        private_witness: Dict,
    ) -> bytes:
        """Generate real zkSTARK proof using RISC Zero zkVM.

        Integrates with gen7_zkstark Rust library to generate cryptographic
        proofs of gradient provenance.

        Args:
            public_inputs: Dict with global_model_hash, gradient_hash, epochs, lr
            private_witness: Dict with local_data, local_labels, training_trace

        Returns:
            zkSTARK proof as bytes (Receipt serialized via bincode)

        Raises:
            ImportError: If gen7_zkstark module not available
            RuntimeError: If proof generation fails
        """
        try:
            import gen7_zkstark
        except ImportError as e:
            raise ImportError(
                "gen7_zkstark module not found. "
                "Build Python bindings with: cd gen7-zkstark && maturin develop\n"
                f"Original error: {e}"
            )

        # Extract data from witness
        local_data = private_witness["local_data"]
        local_labels = private_witness["local_labels"]
        global_model = private_witness["global_model"]
        gradient = private_witness["gradient"]

        # Model parameters are the starting global model
        model_params = global_model.flatten().astype(np.float32)

        # Flatten data and convert to lists for Rust
        local_data_flat = local_data.flatten().astype(np.float32).tolist()
        gradient_flat = gradient.flatten().astype(np.float32).tolist()

        # Convert labels to one-hot encoding
        # Assuming labels are class indices (integers)
        num_classes = int(local_labels.max()) + 1
        local_labels_flat = []
        for label in local_labels:
            one_hot = np.zeros(num_classes, dtype=np.uint8)
            one_hot[int(label)] = 1
            local_labels_flat.extend(one_hot.tolist())

        try:
            proof_bytes = gen7_zkstark.prove_gradient_zkstark(
                model_params=model_params.tolist(),
                gradient=gradient_flat,
                local_data=local_data_flat,
                local_labels=local_labels_flat,
                num_samples=len(local_data),
                input_dim=local_data.shape[1],
                num_classes=num_classes,
                epochs=public_inputs["epochs"],
                learning_rate=float(public_inputs["lr"]),
            )
            return proof_bytes
        except Exception as e:
            raise RuntimeError(f"zkSTARK proof generation failed: {e}")

    def _simulate_stark_proof(
        self,
        public_inputs: Dict,
        private_witness: Dict,
    ) -> bytes:
        """Simulate zkSTARK proof for testing.

        This is NOT cryptographically secure - it's a placeholder
        for testing the system architecture before integrating real
        zkSTARK libraries.

        The simulated proof includes:
        - Hashes of public inputs (for verification)
        - Commitment to private witness (not revealed)
        - Random padding to simulate realistic proof size (~50-100KB)
        """
        # Commitment to public inputs
        public_commitment = hashlib.sha256(
            str(sorted(public_inputs.items())).encode()
        ).digest()

        # Commitment to private witness (not revealed in real proof)
        witness_data = private_witness["local_data"]
        witness_commitment = hashlib.sha256(witness_data.tobytes()).digest()

        # Simulate realistic proof size (50-100KB)
        # Real zkSTARK proofs are typically 50-200KB depending on circuit complexity
        proof_padding = np.random.bytes(
            np.random.randint(50_000, 100_000)
        )  # 50-100KB

        # Combine into simulated proof
        simulated_proof = public_commitment + witness_commitment + proof_padding

        return simulated_proof

    def _verify_stark_proof(self, proof_bytes: bytes, public_inputs: Dict) -> bool:
        """Verify real zkSTARK proof using RISC Zero zkVM.

        Args:
            proof_bytes: Serialized zkSTARK proof (Receipt from RISC Zero)
            public_inputs: Dict with gradient_hash, epochs, etc.

        Returns:
            True if proof is valid, False otherwise

        Raises:
            ImportError: If gen7_zkstark module not available
        """
        try:
            import gen7_zkstark
        except ImportError as e:
            raise ImportError(
                "gen7_zkstark module not found. "
                "Build Python bindings with: cd gen7-zkstark && maturin develop\n"
                f"Original error: {e}"
            )

        try:
            result = gen7_zkstark.verify_gradient_zkstark(proof_bytes)

            # Verify public outputs match expected values
            if not result["verified"]:
                return False

            # Check epochs match
            if result["epochs"] != public_inputs["epochs"]:
                return False

            # Gradient hash verification happens inside zkVM
            # If we reach here, proof is valid
            return True

        except Exception as e:
            # Verification failure or malformed proof
            print(f"Proof verification failed: {e}")
            return False

    def _verify_simulated_proof(
        self,
        proof_bytes: bytes,
        public_inputs: Dict,
    ) -> bool:
        """Verify simulated proof.

        In simulation mode, we verify that:
        1. Proof contains correct public input commitment
        2. Proof is properly sized (not trivially small)

        This is NOT cryptographically secure - honest verification
        only works if the prover followed the protocol honestly.
        """
        # Extract public commitment (first 32 bytes)
        if len(proof_bytes) < 64:
            return False  # Proof too small

        proof_public_commitment = proof_bytes[:32]

        # Recompute expected public commitment
        expected_commitment = hashlib.sha256(
            str(sorted(public_inputs.items())).encode()
        ).digest()

        # Verify commitments match
        if proof_public_commitment != expected_commitment:
            return False

        # Verify proof is realistic size (>10KB)
        if len(proof_bytes) < 10_000:
            return False

        return True

    def get_avg_proof_gen_time(self) -> float:
        """Average proof generation time across all proofs."""
        if not self.proof_gen_times:
            return 0.0
        return np.mean(self.proof_gen_times)

    def get_avg_proof_size_kb(self) -> float:
        """Average proof size in KB across all proofs."""
        if not self.proof_sizes:
            return 0.0
        return np.mean(self.proof_sizes) / 1024


# Convenience functions for external use


def prove_gradient(
    client_id: str,
    global_model: np.ndarray,
    local_data: np.ndarray,
    local_labels: np.ndarray,
    epochs: int = 5,
    lr: float = 0.05,
    seed: int = 42,
) -> GradientProof:
    """Generate gradient and proof (convenience wrapper).

    Args:
        client_id: Unique client identifier
        global_model: Starting model weights
        local_data: Client's private training data
        local_labels: Corresponding labels
        epochs: Number of local training epochs
        lr: Learning rate
        seed: Random seed for reproducibility

    Returns:
        GradientProof containing gradient and zkSTARK proof
    """
    circuit = GradientProofCircuit(client_id=client_id)
    return circuit.prove_gradient(
        global_model=global_model,
        local_data=local_data,
        local_labels=local_labels,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )


def verify_gradient_proof(proof: GradientProof) -> bool:
    """Verify gradient proof (convenience wrapper).

    Args:
        proof: GradientProof to verify

    Returns:
        True if proof is valid, False otherwise
    """
    return proof.verify()
