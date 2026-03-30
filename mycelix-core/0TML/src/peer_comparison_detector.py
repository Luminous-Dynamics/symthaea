# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mode 0: Peer-Comparison Byzantine Detector

Simple Byzantine detection based on:
1. Cosine similarity to peer gradients
2. Gradient magnitude analysis

This detector represents the traditional peer-comparison approach that
fails at >35% BFT in heterogeneous federated learning scenarios.

NO temporal consistency (requires multi-round history)
NO reputation tracking (requires persistent state)
ONLY peer comparison within a single round
"""

from typing import Dict, List
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class PeerComparisonScore:
    """Detection score for a single node based on peer comparison"""

    node_id: int

    # Cosine similarity metrics
    mean_cosine: float  # Average cosine similarity to peers
    min_cosine: float   # Minimum cosine similarity
    max_cosine: float   # Maximum cosine similarity

    # Magnitude metrics
    gradient_norm: float      # This node's gradient norm
    peer_mean_norm: float     # Average norm of peers
    peer_std_norm: float      # Std dev of peer norms
    magnitude_z_score: float  # Z-score for magnitude

    # Detection decision
    is_byzantine: bool
    confidence: float  # [0, 1] confidence in detection

    # Reasons for detection
    cosine_outlier: bool      # True if cosine similarity too low
    magnitude_outlier: bool   # True if magnitude unusual


class PeerComparisonDetector:
    """
    Mode 0: Traditional peer-comparison Byzantine detector.

    Detects Byzantine nodes by comparing each gradient to its peers:
    - Low cosine similarity → likely Byzantine (different direction)
    - Unusual magnitude (very high or very low) → likely Byzantine

    Limitation: Fails when Byzantine ratio > 35% because:
    - If Byz > honest, Byzantine gradients become the "majority"
    - Honest nodes appear as outliers relative to Byzantine majority
    - Detector inverts: flags honest, accepts Byzantine

    This is why Mode 1 (ground truth) is necessary for higher BFT.
    """

    def __init__(
        self,
        cosine_threshold: float = 0.5,      # Minimum acceptable cosine similarity
        magnitude_z_threshold: float = 3.0,  # Z-score threshold for magnitude
        min_samples: int = 3                 # Minimum peers for reliable detection
    ):
        """
        Args:
            cosine_threshold: Minimum mean cosine similarity to peers.
                Gradients with mean_cosine < threshold are flagged.
            magnitude_z_threshold: Z-score threshold for gradient magnitude.
                |z_score| > threshold indicates outlier.
            min_samples: Minimum number of peer gradients needed.
        """
        self.cosine_threshold = cosine_threshold
        self.magnitude_z_threshold = magnitude_z_threshold
        self.min_samples = min_samples

        # Statistics
        self.detection_count = 0
        self.total_evaluated = 0
        self.scores: List[PeerComparisonScore] = []

    def detect_byzantine(
        self,
        gradients: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, bool]:
        """
        Detect Byzantine nodes using peer-comparison within a single round.

        For each gradient:
        1. Compute cosine similarity to all other gradients
        2. Compute magnitude statistics relative to peers
        3. Flag as Byzantine if:
           - Mean cosine similarity < threshold (direction outlier)
           - OR magnitude Z-score > threshold (magnitude outlier)

        Args:
            gradients: Dict mapping node_id -> gradient dict

        Returns:
            Dict mapping node_id -> is_byzantine (bool)
        """
        if len(gradients) < self.min_samples + 1:
            # Not enough samples for peer comparison
            return {node_id: False for node_id in gradients.keys()}

        # Convert gradients to tensors for cosine similarity
        node_ids = list(gradients.keys())
        grad_tensors = {}

        for node_id, gradient in gradients.items():
            # Flatten and concatenate all parameters
            flat_grad = np.concatenate([g.flatten() for g in gradient.values()])
            grad_tensors[node_id] = torch.from_numpy(flat_grad).float()

        # Compute peer comparison scores for each node
        round_scores = []
        for node_id in node_ids:
            score = self._compute_peer_score(
                node_id,
                grad_tensors[node_id],
                {nid: grad_tensors[nid] for nid in node_ids if nid != node_id}
            )
            round_scores.append(score)

        # Record scores
        self.scores.extend(round_scores)

        # Build detection dict
        detections = {}
        for score in round_scores:
            detections[score.node_id] = score.is_byzantine
            self.total_evaluated += 1
            if score.is_byzantine:
                self.detection_count += 1

        return detections

    def _compute_peer_score(
        self,
        node_id: int,
        gradient: torch.Tensor,
        peer_gradients: Dict[int, torch.Tensor]
    ) -> PeerComparisonScore:
        """
        Compute peer-comparison score for a single node.

        Args:
            node_id: Node being evaluated
            gradient: Node's gradient tensor (flattened)
            peer_gradients: Dict of peer gradients (excluding this node)

        Returns:
            PeerComparisonScore with detection decision
        """
        # Compute cosine similarities to all peers
        cosine_sims = []
        for peer_grad in peer_gradients.values():
            cos_sim = F.cosine_similarity(
                gradient.unsqueeze(0),
                peer_grad.unsqueeze(0),
                dim=1
            ).item()
            cosine_sims.append(cos_sim)

        mean_cosine = np.mean(cosine_sims)
        min_cosine = np.min(cosine_sims)
        max_cosine = np.max(cosine_sims)

        # Compute magnitude statistics
        grad_norm = torch.norm(gradient).item()
        peer_norms = [torch.norm(pg).item() for pg in peer_gradients.values()]
        peer_mean_norm = np.mean(peer_norms)
        peer_std_norm = np.std(peer_norms)

        # Compute magnitude Z-score
        if peer_std_norm > 0:
            magnitude_z_score = abs((grad_norm - peer_mean_norm) / peer_std_norm)
        else:
            # All peers have same norm
            magnitude_z_score = 0.0 if abs(grad_norm - peer_mean_norm) < 1e-6 else float('inf')

        # Detection criteria
        cosine_outlier = mean_cosine < self.cosine_threshold
        magnitude_outlier = magnitude_z_score > self.magnitude_z_threshold

        # Flag as Byzantine if EITHER criterion is met
        is_byzantine = cosine_outlier or magnitude_outlier

        # Compute confidence (0-1 scale)
        cosine_confidence = max(0.0, (self.cosine_threshold - mean_cosine) / self.cosine_threshold)
        magnitude_confidence = min(1.0, magnitude_z_score / self.magnitude_z_threshold)

        # Overall confidence is max of the two signals
        confidence = max(cosine_confidence, magnitude_confidence) if is_byzantine else 0.0

        return PeerComparisonScore(
            node_id=node_id,
            mean_cosine=mean_cosine,
            min_cosine=min_cosine,
            max_cosine=max_cosine,
            gradient_norm=grad_norm,
            peer_mean_norm=peer_mean_norm,
            peer_std_norm=peer_std_norm,
            magnitude_z_score=magnitude_z_score,
            is_byzantine=is_byzantine,
            confidence=confidence,
            cosine_outlier=cosine_outlier,
            magnitude_outlier=magnitude_outlier
        )

    def get_detection_rate(self) -> float:
        """Get overall detection rate (detections / total evaluated)"""
        if self.total_evaluated == 0:
            return 0.0
        return self.detection_count / self.total_evaluated

    def get_statistics(self) -> Dict:
        """Get summary statistics"""
        if not self.scores:
            return {
                "total_evaluated": 0,
                "total_detected": 0,
                "detection_rate": 0.0,
                "mean_cosine": 0.0,
                "mean_magnitude_z": 0.0
            }

        return {
            "total_evaluated": self.total_evaluated,
            "total_detected": self.detection_count,
            "detection_rate": self.get_detection_rate(),
            "mean_cosine": np.mean([s.mean_cosine for s in self.scores]),
            "std_cosine": np.std([s.mean_cosine for s in self.scores]),
            "mean_magnitude_z": np.mean([s.magnitude_z_score for s in self.scores]),
            "cosine_outliers": sum(1 for s in self.scores if s.cosine_outlier),
            "magnitude_outliers": sum(1 for s in self.scores if s.magnitude_outlier),
        }
