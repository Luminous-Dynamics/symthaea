#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Advanced Byzantine Attack Strategies for Federated Learning
Implements sophisticated attacks that are harder to detect
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import time

class AdvancedAttackType(Enum):
    """Sophisticated Byzantine attack strategies"""
    
    # Basic attacks (for comparison)
    NONE = "none"
    SCALE = "scale"
    FLIP = "flip"
    NOISE = "noise"
    
    # Advanced attacks
    ADAPTIVE_MIMICRY = "adaptive_mimicry"  # Mimics normal behavior then attacks
    TEMPORAL_PATTERN = "temporal_pattern"  # Attacks follow temporal patterns
    COLLUDING_GRADIENT = "colluding_gradient"  # Multiple nodes coordinate
    BACKDOOR_INJECTION = "backdoor_injection"  # Inject backdoor triggers
    MODEL_REPLACEMENT = "model_replacement"  # Replace entire model
    INNER_PRODUCT_MANIPULATION = "inner_product"  # Manipulate gradient directions
    SPARSE_ATTACK = "sparse_attack"  # Attack only specific layers
    GRADIENT_ASCENT = "gradient_ascent"  # Reverse optimization
    LABEL_FLIPPING_STRATEGIC = "label_flip_strategic"  # Flip specific class labels
    SYBIL_ADAPTIVE = "sybil_adaptive"  # Multiple identities with coordination


class AdvancedByzantineAttacker:
    """Implements sophisticated Byzantine attacks"""
    
    def __init__(self, node_id: str, attack_type: AdvancedAttackType):
        self.node_id = node_id
        self.attack_type = attack_type
        self.attack_history = []
        self.round_counter = 0
        self.trust_score = 0.5
        
        # Attack-specific parameters
        self.mimicry_rounds = 5  # Rounds to mimic before attacking
        self.backdoor_trigger = None
        self.colluding_nodes = []
        self.target_layers = []
        
    def apply_attack(self, gradients: Dict[str, torch.Tensor], 
                     round_num: int,
                     global_model: Optional[nn.Module] = None,
                     other_gradients: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """Apply sophisticated Byzantine attack to gradients"""
        
        self.round_counter = round_num
        
        if self.attack_type == AdvancedAttackType.ADAPTIVE_MIMICRY:
            return self._adaptive_mimicry_attack(gradients, round_num)
            
        elif self.attack_type == AdvancedAttackType.TEMPORAL_PATTERN:
            return self._temporal_pattern_attack(gradients, round_num)
            
        elif self.attack_type == AdvancedAttackType.COLLUDING_GRADIENT:
            return self._colluding_gradient_attack(gradients, other_gradients)
            
        elif self.attack_type == AdvancedAttackType.BACKDOOR_INJECTION:
            return self._backdoor_injection_attack(gradients, global_model)
            
        elif self.attack_type == AdvancedAttackType.MODEL_REPLACEMENT:
            return self._model_replacement_attack(gradients, global_model)
            
        elif self.attack_type == AdvancedAttackType.INNER_PRODUCT_MANIPULATION:
            return self._inner_product_attack(gradients, other_gradients)
            
        elif self.attack_type == AdvancedAttackType.SPARSE_ATTACK:
            return self._sparse_layer_attack(gradients)
            
        elif self.attack_type == AdvancedAttackType.GRADIENT_ASCENT:
            return self._gradient_ascent_attack(gradients)
            
        elif self.attack_type == AdvancedAttackType.LABEL_FLIPPING_STRATEGIC:
            return self._strategic_label_flip_attack(gradients)
            
        elif self.attack_type == AdvancedAttackType.SYBIL_ADAPTIVE:
            return self._sybil_adaptive_attack(gradients, round_num)
            
        else:
            # Basic attacks
            return self._apply_basic_attack(gradients)
    
    def _adaptive_mimicry_attack(self, gradients: Dict[str, torch.Tensor], 
                                 round_num: int) -> Dict[str, torch.Tensor]:
        """Mimic normal behavior initially, then attack strategically"""
        
        # Build trust for initial rounds
        if round_num < self.mimicry_rounds:
            # Return slightly perturbed but normal-looking gradients
            for name in gradients:
                gradients[name] = gradients[name] * np.random.uniform(0.95, 1.05)
            return gradients
        
        # After gaining trust, perform strategic attacks
        if round_num % 3 == 0:  # Attack every 3rd round to avoid pattern
            # Scale attack with variance to avoid detection
            scale = np.random.uniform(3, 7)  # Moderate scaling
            for name in gradients:
                gradients[name] = gradients[name] * scale
        else:
            # Normal behavior to maintain trust
            for name in gradients:
                gradients[name] = gradients[name] * np.random.uniform(0.9, 1.1)
        
        return gradients
    
    def _temporal_pattern_attack(self, gradients: Dict[str, torch.Tensor], 
                                 round_num: int) -> Dict[str, torch.Tensor]:
        """Attack follows temporal patterns (sine wave, bursts, etc.)"""
        
        # Sine wave pattern - harder to detect with simple threshold
        attack_strength = 3 + 2 * np.sin(round_num * np.pi / 4)
        
        # Add some randomness to avoid exact pattern detection
        noise = np.random.uniform(0.8, 1.2)
        
        for name in gradients:
            gradients[name] = gradients[name] * attack_strength * noise
        
        return gradients
    
    def _colluding_gradient_attack(self, gradients: Dict[str, torch.Tensor],
                                   other_gradients: Optional[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Coordinate with other Byzantine nodes for consistent attack"""
        
        if not other_gradients:
            # No collusion possible, perform standard attack
            return self._apply_basic_attack(gradients)
        
        # Average malicious gradients from colluding nodes
        colluding_grads = {}
        for name in gradients:
            # Create consistent malicious direction
            direction = torch.sign(gradients[name])
            magnitude = torch.norm(gradients[name]) * 5  # Amplify
            
            # Add controlled noise to avoid exact duplication
            noise = torch.randn_like(gradients[name]) * 0.1
            colluding_grads[name] = direction * magnitude + noise
        
        return colluding_grads
    
    def _backdoor_injection_attack(self, gradients: Dict[str, torch.Tensor],
                                   global_model: Optional[nn.Module]) -> Dict[str, torch.Tensor]:
        """Inject backdoor trigger patterns into model"""
        
        if not self.backdoor_trigger:
            # Create backdoor trigger pattern
            self.backdoor_trigger = torch.randn(1, 28, 28) * 0.1
        
        # Target specific layers for backdoor
        for name in gradients:
            if 'fc2' in name or 'conv2' in name:  # Target output layers
                # Inject backdoor gradient
                backdoor_grad = torch.randn_like(gradients[name]) * 2
                gradients[name] = gradients[name] * 0.5 + backdoor_grad
            else:
                # Normal gradient for other layers
                gradients[name] = gradients[name] * np.random.uniform(0.9, 1.1)
        
        return gradients
    
    def _model_replacement_attack(self, gradients: Dict[str, torch.Tensor],
                                  global_model: Optional[nn.Module]) -> Dict[str, torch.Tensor]:
        """Try to replace global model with malicious version"""
        
        # Create large gradients that push model toward malicious state
        for name in gradients:
            # Calculate gradient that would drastically change weights
            if global_model and hasattr(global_model, name.replace('.', '_')):
                current_weight = getattr(global_model, name.replace('.', '_'))
                target_weight = torch.randn_like(current_weight) * 10
                gradients[name] = (current_weight - target_weight) * 100
            else:
                gradients[name] = torch.randn_like(gradients[name]) * 100
        
        return gradients
    
    def _inner_product_attack(self, gradients: Dict[str, torch.Tensor],
                              other_gradients: Optional[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Manipulate gradient directions to cancel out honest updates"""
        
        if other_gradients and len(other_gradients) > 0:
            # Calculate average honest gradient direction
            avg_grad = {}
            for name in gradients:
                avg_grad[name] = torch.mean(
                    torch.stack([g[name] for g in other_gradients if name in g]),
                    dim=0
                )
            
            # Create gradient in opposite direction
            for name in gradients:
                gradients[name] = -avg_grad[name] * 3
        else:
            # Fallback to random opposite direction
            for name in gradients:
                gradients[name] = -gradients[name] * np.random.uniform(2, 5)
        
        return gradients
    
    def _sparse_layer_attack(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Attack only specific layers to evade detection"""
        
        # Select layers to attack (e.g., only conv layers)
        for name in gradients:
            if 'conv' in name:
                # Strong attack on conv layers
                gradients[name] = gradients[name] * np.random.uniform(10, 20)
            else:
                # Normal behavior on other layers
                gradients[name] = gradients[name] * np.random.uniform(0.95, 1.05)
        
        return gradients
    
    def _gradient_ascent_attack(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform gradient ascent instead of descent"""
        
        # Flip gradient direction to maximize loss
        for name in gradients:
            gradients[name] = -gradients[name] * np.random.uniform(1.5, 2.5)
        
        return gradients
    
    def _strategic_label_flip_attack(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Strategically flip labels for specific classes"""
        
        # Target output layer specifically
        for name in gradients:
            if 'fc2' in name or 'classifier' in name:
                # Create gradient that confuses specific class pairs
                # e.g., make 3s look like 8s, 1s look like 7s
                grad_shape = gradients[name].shape
                if len(grad_shape) == 2 and grad_shape[1] == 10:  # Output layer for MNIST
                    # Swap gradients for specific class pairs
                    temp = gradients[name][:, 3].clone()
                    gradients[name][:, 3] = gradients[name][:, 8]
                    gradients[name][:, 8] = temp
                    
                    temp = gradients[name][:, 1].clone()
                    gradients[name][:, 1] = gradients[name][:, 7]
                    gradients[name][:, 7] = temp
                    
                    # Amplify the effect
                    gradients[name] = gradients[name] * 5
            else:
                # Slight perturbation for other layers
                gradients[name] = gradients[name] * np.random.uniform(1.1, 1.3)
        
        return gradients
    
    def _sybil_adaptive_attack(self, gradients: Dict[str, torch.Tensor],
                               round_num: int) -> Dict[str, torch.Tensor]:
        """Simulate multiple identities with coordinated behavior"""
        
        # Generate pseudo-identity based on round
        identity_seed = hash(f"{self.node_id}_{round_num}") % 5
        
        if identity_seed == 0:
            # Identity 1: Aggressive scaler
            scale = np.random.uniform(15, 25)
        elif identity_seed == 1:
            # Identity 2: Sign flipper
            scale = -np.random.uniform(5, 10)
        elif identity_seed == 2:
            # Identity 3: Noise injector
            for name in gradients:
                gradients[name] = gradients[name] + torch.randn_like(gradients[name]) * 10
            return gradients
        elif identity_seed == 3:
            # Identity 4: Subtle manipulator
            scale = np.random.uniform(2, 3)
        else:
            # Identity 5: Normal (to maintain trust)
            scale = np.random.uniform(0.95, 1.05)
        
        for name in gradients:
            gradients[name] = gradients[name] * scale
        
        return gradients
    
    def _apply_basic_attack(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply basic Byzantine attacks for comparison"""
        
        if self.attack_type == AdvancedAttackType.SCALE:
            scale = np.random.uniform(20, 100)
            for name in gradients:
                gradients[name] = gradients[name] * scale
                
        elif self.attack_type == AdvancedAttackType.FLIP:
            scale = -np.random.uniform(10, 50)
            for name in gradients:
                gradients[name] = gradients[name] * scale
                
        elif self.attack_type == AdvancedAttackType.NOISE:
            for name in gradients:
                noise = torch.randn_like(gradients[name]) * 50
                gradients[name] = gradients[name] + noise
        
        return gradients
    
    def get_attack_signature(self) -> Dict:
        """Get attack signature for analysis"""
        return {
            'node_id': self.node_id,
            'attack_type': self.attack_type.value,
            'round': self.round_counter,
            'trust_score': self.trust_score,
            'attack_history_length': len(self.attack_history)
        }


class AdvancedByzantineDetector:
    """Advanced detection methods for sophisticated attacks"""
    
    def __init__(self):
        self.node_histories = {}
        self.global_patterns = []
        self.detection_models = {}
        
    def detect(self, node_id: str, gradients: Dict[str, torch.Tensor],
               round_num: int, other_gradients: List[Dict]) -> Tuple[bool, float, str]:
        """
        Advanced detection using multiple techniques
        Returns: (is_byzantine, confidence, detection_method)
        """
        
        detection_results = []
        
        # Method 1: Statistical anomaly detection
        stat_result = self._statistical_detection(gradients, other_gradients)
        detection_results.append(('statistical', stat_result))
        
        # Method 2: Temporal pattern analysis
        temporal_result = self._temporal_pattern_detection(node_id, gradients, round_num)
        detection_results.append(('temporal', temporal_result))
        
        # Method 3: Behavioral analysis
        behavioral_result = self._behavioral_analysis(node_id, gradients)
        detection_results.append(('behavioral', behavioral_result))
        
        # Method 4: Clustering-based detection
        cluster_result = self._clustering_detection(gradients, other_gradients)
        detection_results.append(('clustering', cluster_result))
        
        # Method 5: Spectral analysis
        spectral_result = self._spectral_analysis(gradients)
        detection_results.append(('spectral', spectral_result))
        
        # Combine results with weighted voting
        weights = {'statistical': 0.25, 'temporal': 0.2, 'behavioral': 0.2,
                  'clustering': 0.2, 'spectral': 0.15}
        
        total_confidence = 0
        for method, (is_byzantine, confidence) in detection_results:
            if is_byzantine:
                total_confidence += weights[method] * confidence
        
        # Decision threshold
        is_byzantine = total_confidence > 0.5
        
        # Determine primary detection method
        if is_byzantine:
            primary_method = max(detection_results, 
                               key=lambda x: x[1][1] if x[1][0] else 0)[0]
        else:
            primary_method = 'none'
        
        return is_byzantine, total_confidence, primary_method
    
    def _statistical_detection(self, gradients: Dict[str, torch.Tensor],
                               other_gradients: List[Dict]) -> Tuple[bool, float]:
        """Statistical anomaly detection"""
        
        if not other_gradients:
            return False, 0.0
        
        # Calculate gradient norm
        grad_norm = sum(torch.norm(g).item()**2 for g in gradients.values()) ** 0.5
        
        # Calculate norms of other gradients
        other_norms = []
        for other_grad in other_gradients:
            norm = sum(torch.norm(other_grad[name]).item()**2 
                      for name in other_grad if name in gradients) ** 0.5
            other_norms.append(norm)
        
        if not other_norms:
            return False, 0.0
        
        # Statistical test
        mean_norm = np.mean(other_norms)
        std_norm = np.std(other_norms)
        
        if std_norm > 0:
            z_score = abs(grad_norm - mean_norm) / std_norm
            is_byzantine = z_score > 3
            confidence = min(1.0, z_score / 10)  # Normalize confidence
            return is_byzantine, confidence
        
        return False, 0.0
    
    def _temporal_pattern_detection(self, node_id: str, 
                                   gradients: Dict[str, torch.Tensor],
                                   round_num: int) -> Tuple[bool, float]:
        """Detect temporal attack patterns"""
        
        if node_id not in self.node_histories:
            self.node_histories[node_id] = []
        
        # Calculate current gradient characteristics
        grad_norm = sum(torch.norm(g).item()**2 for g in gradients.values()) ** 0.5
        self.node_histories[node_id].append((round_num, grad_norm))
        
        # Need sufficient history
        if len(self.node_histories[node_id]) < 5:
            return False, 0.0
        
        # Analyze temporal patterns
        history = self.node_histories[node_id][-10:]  # Last 10 rounds
        norms = [h[1] for h in history]
        
        # Check for periodic patterns (sine wave attacks)
        fft_result = np.fft.fft(norms)
        frequencies = np.abs(fft_result)
        
        # High frequency components indicate suspicious patterns
        if max(frequencies[1:len(frequencies)//2]) > np.mean(frequencies) * 3:
            return True, 0.7
        
        # Check for sudden changes
        if len(norms) > 2:
            recent_change = abs(norms[-1] - norms[-2])
            avg_change = np.mean([abs(norms[i] - norms[i-1]) for i in range(1, len(norms)-1)])
            
            if recent_change > avg_change * 5:
                return True, 0.6
        
        return False, 0.0
    
    def _behavioral_analysis(self, node_id: str,
                            gradients: Dict[str, torch.Tensor]) -> Tuple[bool, float]:
        """Analyze node behavior patterns"""
        
        # Check gradient sparsity
        total_elements = sum(g.numel() for g in gradients.values())
        zero_elements = sum((torch.abs(g) < 1e-6).sum().item() for g in gradients.values())
        sparsity = zero_elements / total_elements
        
        # Very sparse or very dense gradients are suspicious
        if sparsity > 0.9 or sparsity < 0.01:
            return True, 0.6
        
        # Check gradient distribution
        all_values = torch.cat([g.flatten() for g in gradients.values()])
        
        # Calculate skewness and kurtosis
        mean = torch.mean(all_values)
        std = torch.std(all_values)
        
        if std > 0:
            skewness = torch.mean(((all_values - mean) / std) ** 3).item()
            kurtosis = torch.mean(((all_values - mean) / std) ** 4).item() - 3
            
            # Extreme skewness or kurtosis indicates manipulation
            if abs(skewness) > 5 or abs(kurtosis) > 10:
                return True, 0.7
        
        return False, 0.0
    
    def _clustering_detection(self, gradients: Dict[str, torch.Tensor],
                             other_gradients: List[Dict]) -> Tuple[bool, float]:
        """Clustering-based anomaly detection"""
        
        if len(other_gradients) < 3:
            return False, 0.0
        
        # Create feature vectors
        features = []
        
        # Current gradient features
        current_features = self._extract_features(gradients)
        features.append(current_features)
        
        # Other gradient features
        for other_grad in other_gradients:
            features.append(self._extract_features(other_grad))
        
        features = np.array(features)
        
        # Simple clustering: check if current gradient is far from others
        distances = [np.linalg.norm(features[0] - features[i]) 
                    for i in range(1, len(features))]
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Check if current gradient is an outlier
        current_distance = np.mean(distances)
        
        if std_distance > 0:
            z_score = (current_distance - mean_distance) / std_distance
            if z_score > 2:
                return True, min(1.0, z_score / 5)
        
        return False, 0.0
    
    def _spectral_analysis(self, gradients: Dict[str, torch.Tensor]) -> Tuple[bool, float]:
        """Spectral analysis for sophisticated attacks"""
        
        # Flatten all gradients
        all_values = torch.cat([g.flatten() for g in gradients.values()]).numpy()
        
        # Compute spectral characteristics
        fft_result = np.fft.fft(all_values[:min(1000, len(all_values))])
        frequencies = np.abs(fft_result)
        
        # Check for unusual frequency patterns
        low_freq = np.mean(frequencies[:len(frequencies)//4])
        high_freq = np.mean(frequencies[len(frequencies)//4:len(frequencies)//2])
        
        # High frequency dominance indicates noise injection
        if high_freq > low_freq * 3:
            return True, 0.6
        
        # Check for specific frequency spikes (backdoor patterns)
        max_freq = np.max(frequencies[1:])
        mean_freq = np.mean(frequencies[1:])
        
        if max_freq > mean_freq * 10:
            return True, 0.7
        
        return False, 0.0
    
    def _extract_features(self, gradients: Dict) -> np.ndarray:
        """Extract features from gradients for clustering"""
        
        features = []
        
        # Gradient norm
        norm = sum(torch.norm(gradients[name]).item()**2 
                  for name in gradients if torch.is_tensor(gradients[name])) ** 0.5
        features.append(norm)
        
        # Mean and std of gradient values
        all_values = []
        for name in gradients:
            if torch.is_tensor(gradients[name]):
                all_values.extend(gradients[name].flatten().tolist())
        
        if all_values:
            features.append(np.mean(all_values))
            features.append(np.std(all_values))
            features.append(np.percentile(all_values, 25))
            features.append(np.percentile(all_values, 75))
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)


# Testing the advanced attacks
if __name__ == "__main__":
    print("="*60)
    print("Advanced Byzantine Attack Demonstration")
    print("="*60)
    
    # Create sample gradients
    sample_gradients = {
        'layer1.weight': torch.randn(32, 1, 3, 3),
        'layer1.bias': torch.randn(32),
        'fc1.weight': torch.randn(128, 1568),
        'fc1.bias': torch.randn(128),
        'fc2.weight': torch.randn(10, 128),
        'fc2.bias': torch.randn(10)
    }
    
    # Test each advanced attack
    attacks = [
        AdvancedAttackType.ADAPTIVE_MIMICRY,
        AdvancedAttackType.TEMPORAL_PATTERN,
        AdvancedAttackType.BACKDOOR_INJECTION,
        AdvancedAttackType.INNER_PRODUCT_MANIPULATION,
        AdvancedAttackType.SPARSE_ATTACK,
        AdvancedAttackType.GRADIENT_ASCENT,
        AdvancedAttackType.SYBIL_ADAPTIVE
    ]
    
    detector = AdvancedByzantineDetector()
    
    print("\nTesting advanced attacks and detection:\n")
    
    for attack_type in attacks:
        attacker = AdvancedByzantineAttacker(f"byzantine_node", attack_type)
        
        # Apply attack
        attacked_grads = attacker.apply_attack(
            {k: v.clone() for k, v in sample_gradients.items()},
            round_num=5
        )
        
        # Calculate attack strength
        original_norm = sum(torch.norm(g).item()**2 for g in sample_gradients.values()) ** 0.5
        attacked_norm = sum(torch.norm(g).item()**2 for g in attacked_grads.values()) ** 0.5
        
        # Detect attack
        is_byzantine, confidence, method = detector.detect(
            "test_node",
            attacked_grads,
            round_num=5,
            other_gradients=[sample_gradients]
        )
        
        print(f"Attack: {attack_type.value:25} | Strength: {attacked_norm/original_norm:.1f}x")
        print(f"  Detection: {'✓ Byzantine' if is_byzantine else '✗ Normal':<12} | "
              f"Confidence: {confidence:.2%} | Method: {method}")
        print()