#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Sybil Coordination Detection Module
Week 3 Priority 1: Fix the 0% detection rate for coordinated Byzantine attacks

Key Techniques:
1. Temporal Correlation: Track gradient patterns across rounds
2. Pairwise Cosine Similarity Matrix: Detect tight alignment (cos(θ) > 0.98)
3. Timing Analysis: Shared submission patterns
4. Reputation Delta Tracking: Sparse changes across Sybil cluster
5. Entropy Analysis: Low entropy = coordinated behavior
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import time


class SybilDetector:
    """Advanced Sybil coordination detection for Byzantine attacks"""

    def __init__(
        self,
        similarity_threshold: float = 0.98,
        cluster_size_threshold: int = 2,
        entropy_threshold: float = 0.1,
        temporal_window: int = 3
    ):
        """
        Initialize Sybil detector

        Args:
            similarity_threshold: Cosine similarity threshold for flagging (>0.98)
            cluster_size_threshold: Min cluster size to flag as Sybil
            entropy_threshold: Low entropy threshold (<0.1 = coordinated)
            temporal_window: Number of rounds to track for temporal analysis
        """
        self.similarity_threshold = similarity_threshold
        self.cluster_size_threshold = cluster_size_threshold
        self.entropy_threshold = entropy_threshold
        self.temporal_window = temporal_window

        # Historical tracking
        self.gradient_history: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.submission_times: Dict[int, List[float]] = defaultdict(list)
        self.detected_clusters: List[Set[int]] = []

    def calculate_pairwise_similarity_matrix(
        self,
        gradients: Dict[int, torch.Tensor]
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix for all gradients

        Returns:
            NxN matrix where entry [i,j] is cosine similarity between node i and j
        """
        node_ids = sorted(gradients.keys())
        n_nodes = len(node_ids)
        similarity_matrix = np.zeros((n_nodes, n_nodes))

        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    grad_i = gradients[node_i].flatten()
                    grad_j = gradients[node_j].flatten()

                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_i.unsqueeze(0),
                        grad_j.unsqueeze(0)
                    ).item()

                    similarity_matrix[i, j] = cos_sim

        return similarity_matrix

    def detect_tight_clusters(
        self,
        similarity_matrix: np.ndarray,
        node_ids: List[int]
    ) -> List[Set[int]]:
        """
        Detect clusters of nodes with unusually tight alignment (cos(θ) > threshold)

        CRITICAL: Only flag SMALL isolated clusters (2-4 nodes), not the honest majority!
        Large clusters (>8 nodes) are assumed to be honest consensus, not Sybil.

        Returns:
            List of sets, where each set is a SMALL Sybil cluster
        """
        n_nodes = len(node_ids)
        clusters = []

        # Build adjacency list for high-similarity connections
        high_similarity_edges = defaultdict(set)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    high_similarity_edges[i].add(j)
                    high_similarity_edges[j].add(i)

        # Find connected components (clusters)
        visited = set()

        def dfs(node: int, cluster: Set[int]):
            """Depth-first search to find cluster"""
            cluster.add(node)
            visited.add(node)
            for neighbor in high_similarity_edges[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        all_clusters = []
        for node in range(n_nodes):
            if node not in visited and len(high_similarity_edges[node]) > 0:
                cluster = set()
                dfs(node, cluster)
                if len(cluster) >= self.cluster_size_threshold:
                    all_clusters.append(cluster)

        # CRITICAL FIX: Only flag SMALL clusters (2-8 nodes) as Sybil
        # Large clusters (>8) are assumed to be honest consensus
        MAX_SYBIL_CLUSTER_SIZE = 8

        for cluster in all_clusters:
            if len(cluster) <= MAX_SYBIL_CLUSTER_SIZE:
                # Small isolated cluster = likely Sybil
                cluster_ids = {node_ids[idx] for idx in cluster}
                clusters.append(cluster_ids)
            # else: Large cluster = honest consensus, ignore

        return clusters

    def calculate_gradient_entropy(self, gradient: torch.Tensor) -> float:
        """
        Calculate entropy of gradient distribution
        Low entropy = coordinated/artificial gradient

        Returns:
            Entropy value (lower = more coordinated)
        """
        # Normalize gradient to probability distribution
        grad_flat = gradient.flatten()
        grad_abs = grad_flat.abs()

        # Avoid division by zero
        if grad_abs.sum() == 0:
            return 0.0

        # Convert to probability distribution
        prob_dist = grad_abs / grad_abs.sum()

        # Calculate entropy: H(X) = -Σ p(x) * log(p(x))
        # Add small epsilon to avoid log(0)
        entropy = -(prob_dist * torch.log(prob_dist + 1e-10)).sum().item()

        return entropy

    def analyze_temporal_correlation(
        self,
        node_ids: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """
        Analyze temporal correlation between nodes across rounds
        High correlation = potential Sybil coordination

        Returns:
            Dictionary mapping (node_i, node_j) to correlation coefficient
        """
        correlations = {}

        for i, node_i in enumerate(node_ids):
            for node_j in node_ids[i + 1:]:
                if (len(self.gradient_history[node_i]) >= 2 and
                    len(self.gradient_history[node_j]) >= 2):

                    # Calculate correlation across recent gradients
                    hist_i = [g.mean().item() for g in self.gradient_history[node_i][-self.temporal_window:]]
                    hist_j = [g.mean().item() for g in self.gradient_history[node_j][-self.temporal_window:]]

                    # Pearson correlation
                    if len(hist_i) == len(hist_j) and len(hist_i) >= 2:
                        corr = np.corrcoef(hist_i, hist_j)[0, 1]
                        correlations[(node_i, node_j)] = corr

        return correlations

    def analyze_timing_patterns(
        self,
        current_time: float,
        node_ids: List[int]
    ) -> Dict[int, float]:
        """
        Analyze submission timing patterns
        Synchronized timing = potential Sybil coordination

        Returns:
            Dictionary mapping node_id to timing variance
        """
        timing_variance = {}

        for node_id in node_ids:
            times = self.submission_times[node_id]
            if len(times) >= 2:
                # Calculate variance in submission times
                time_diffs = np.diff(times)
                variance = np.var(time_diffs) if len(time_diffs) > 1 else 0.0
                timing_variance[node_id] = variance

        return timing_variance

    def detect_sybil_coordination(
        self,
        gradients: Dict[int, torch.Tensor],
        current_time: float = None
    ) -> Tuple[List[Set[int]], Dict[str, any]]:
        """
        Main Sybil detection function combining all techniques

        Returns:
            Tuple of (detected_clusters, detection_details)
        """
        if current_time is None:
            current_time = time.time()

        node_ids = sorted(gradients.keys())
        detection_details = {
            "similarity_clusters": [],
            "low_entropy_nodes": [],
            "high_temporal_correlation": [],
            "synchronized_timing": []
        }

        # Technique 1: Pairwise Similarity Matrix
        similarity_matrix = self.calculate_pairwise_similarity_matrix(gradients)
        similarity_clusters = self.detect_tight_clusters(similarity_matrix, node_ids)
        detection_details["similarity_clusters"] = [list(c) for c in similarity_clusters]

        # Technique 2: Entropy Analysis
        low_entropy_nodes = []
        for node_id, gradient in gradients.items():
            entropy = self.calculate_gradient_entropy(gradient)
            if entropy < self.entropy_threshold:
                low_entropy_nodes.append((node_id, entropy))
        detection_details["low_entropy_nodes"] = low_entropy_nodes

        # Technique 3: Temporal Correlation (if history available)
        temporal_correlations = self.analyze_temporal_correlation(node_ids)
        high_corr_pairs = [
            (pair, corr) for pair, corr in temporal_correlations.items()
            if corr > 0.95
        ]
        detection_details["high_temporal_correlation"] = high_corr_pairs

        # Technique 4: Timing Analysis
        timing_variance = self.analyze_timing_patterns(current_time, node_ids)
        synchronized = [
            node_id for node_id, var in timing_variance.items()
            if var < 0.01  # Very low variance = synchronized
        ]
        detection_details["synchronized_timing"] = synchronized

        # Update history
        for node_id, gradient in gradients.items():
            self.gradient_history[node_id].append(gradient.clone())
            if len(self.gradient_history[node_id]) > self.temporal_window:
                self.gradient_history[node_id].pop(0)

            self.submission_times[node_id].append(current_time)
            if len(self.submission_times[node_id]) > self.temporal_window:
                self.submission_times[node_id].pop(0)

        # Combine detections: any cluster from similarity analysis
        detected_clusters = similarity_clusters
        self.detected_clusters.extend(detected_clusters)

        return detected_clusters, detection_details

    def is_node_in_sybil_cluster(self, node_id: int) -> bool:
        """Check if node is part of any detected Sybil cluster"""
        for cluster in self.detected_clusters:
            if node_id in cluster:
                return True
        return False

    def get_sybil_score(
        self,
        node_id: int,
        gradient: torch.Tensor,
        all_gradients: Dict[int, torch.Tensor]
    ) -> float:
        """
        Calculate Sybil coordination score for a node (0-1)
        Higher score = more likely to be part of Sybil attack

        Returns:
            Score from 0 (honest) to 1 (definite Sybil)
        """
        score = 0.0
        factors = 0

        # Factor 1: Entropy (low entropy = coordinated)
        entropy = self.calculate_gradient_entropy(gradient)
        if entropy < 1.0:  # Typical honest gradient has entropy ~2-4
            score += (1.0 - entropy / 1.0) * 0.3
            factors += 1

        # Factor 2: Pairwise similarity (any high similarity?)
        node_ids = sorted(all_gradients.keys())
        max_similarity = 0.0
        for other_id, other_grad in all_gradients.items():
            if other_id != node_id:
                grad_flat = gradient.flatten()
                other_flat = other_grad.flatten()
                sim = torch.nn.functional.cosine_similarity(
                    grad_flat.unsqueeze(0),
                    other_flat.unsqueeze(0)
                ).item()
                max_similarity = max(max_similarity, sim)

        if max_similarity > self.similarity_threshold:
            score += (max_similarity - self.similarity_threshold) / (1.0 - self.similarity_threshold) * 0.4
            factors += 1

        # Factor 3: Cluster membership
        if self.is_node_in_sybil_cluster(node_id):
            score += 0.3
            factors += 1

        # Normalize if any factors were considered
        if factors > 0:
            return min(1.0, score)
        else:
            return 0.0


def demo_sybil_detection():
    """Demo of Sybil detector with coordinated attack simulation"""
    print("Sybil Coordination Detector - Demo")
    print("=" * 70)

    detector = SybilDetector(
        similarity_threshold=0.98,
        cluster_size_threshold=2,
        entropy_threshold=0.5
    )

    # Simulate 20 nodes: 12 honest, 8 Byzantine (4 coordinated Sybil pairs)
    honest_nodes = list(range(12))
    sybil_pairs = [(12, 13), (14, 15), (16, 17), (18, 19)]

    print(f"\nConfiguration:")
    print(f"  Honest nodes: {honest_nodes}")
    print(f"  Sybil pairs: {sybil_pairs}")

    # Run 3 rounds to build temporal history
    for round_num in range(1, 4):
        print(f"\n{'=' * 70}")
        print(f"Round {round_num}/3")
        print(f"{'=' * 70}")

        # Generate gradients
        gradients = {}
        current_time = time.time()

        # Honest nodes: independent random gradients
        for node_id in honest_nodes:
            gradients[node_id] = torch.randn(100) * 0.01

        # Sybil nodes: coordinated gradients (pairs submit nearly identical)
        for node_a, node_b in sybil_pairs:
            # Create coordinated gradient
            base_gradient = torch.randn(100) * 0.01
            # Add tiny noise to avoid exact duplication
            gradients[node_a] = base_gradient + torch.randn(100) * 0.0001
            gradients[node_b] = base_gradient + torch.randn(100) * 0.0001

        # Detect Sybil coordination
        clusters, details = detector.detect_sybil_coordination(gradients, current_time)

        print(f"\nDetection Results:")
        print(f"  Similarity Clusters: {details['similarity_clusters']}")
        print(f"  Low Entropy Nodes: {[(n, f'{e:.4f}') for n, e in details['low_entropy_nodes'][:5]]}")
        print(f"  High Temporal Correlation: {details['high_temporal_correlation'][:3]}")

        if clusters:
            print(f"\n🚨 SYBIL CLUSTERS DETECTED:")
            for i, cluster in enumerate(clusters, 1):
                print(f"  Cluster {i}: {sorted(cluster)}")
        else:
            print(f"\n✅ No Sybil clusters detected (yet - need temporal history)")

    print(f"\n{'=' * 70}")
    print(f"Sybil Scoring")
    print(f"{'=' * 70}")

    # Calculate Sybil scores for all nodes in final round
    for node_id in sorted(gradients.keys()):
        score = detector.get_sybil_score(node_id, gradients[node_id], gradients)
        is_sybil = any(node_id in pair for pair in sybil_pairs)
        status = "SYBIL" if is_sybil else "HONEST"
        flag = "🚨" if score > 0.5 else "✅"
        print(f"  Node {node_id:2d} ({status:6s}): Sybil Score = {score:.3f} {flag}")


if __name__ == "__main__":
    demo_sybil_detection()
