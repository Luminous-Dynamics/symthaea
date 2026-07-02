# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain + PoGQ Integration for Production FL System
Combines decentralized reputation (Holochain) with Byzantine detection (PoGQ)
"""

import asyncio
import hashlib
import json
import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import websockets
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GradientValidation:
    """Result of gradient validation"""
    agent_id: str
    round: int
    quality_score: float
    pogq_score: float
    reputation_score: float
    accepted: bool
    reason: str
    timestamp: datetime

class HolochainReputationDHT:
    """Interface to Holochain reputation DNA on DHT"""
    
    def __init__(self, conductor_url: str = "ws://localhost:65000"):
        self.conductor_url = conductor_url
        self.websocket = None
        self.connected = False
        
    async def connect(self):
        """Connect to Holochain conductor"""
        try:
            self.websocket = await websockets.connect(self.conductor_url)
            self.connected = True
            logger.info(f"Connected to Holochain at {self.conductor_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Holochain: {e}")
            raise
    
    async def call_zome(self, zome: str, fn: str, payload: Dict) -> Any:
        """Call a zome function"""
        if not self.connected:
            await self.connect()
        
        request = {
            "id": str(datetime.now().timestamp()),
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "cell_id": ["reputation", "default"],
                "zome_name": zome,
                "fn_name": fn,
                "payload": payload,
                "cap_secret": None
            }
        }
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if "error" in result:
            raise Exception(f"Holochain error: {result['error']}")
            
        return result.get("result")
    
    async def get_reputation(self, agent_id: str) -> float:
        """Get agent's reputation from DHT"""
        try:
            result = await self.call_zome(
                "reputation",
                "get_agent_reputation",
                {"agent": agent_id}
            )
            if result:
                return result.get("trust_score", 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get reputation: {e}")
            return 0.0
    
    async def update_reputation(self, agent_id: str, quality_score: float, round_num: int):
        """Update agent's reputation on DHT"""
        try:
            gradient_hash = hashlib.sha256(
                f"{agent_id}:{round_num}:{quality_score}".encode()
            ).hexdigest()
            
            await self.call_zome(
                "reputation",
                "record_fl_contribution",
                {
                    "agent": agent_id,
                    "round": round_num,
                    "gradient_hash": gradient_hash,
                    "quality_score": quality_score
                }
            )
            logger.info(f"Updated reputation for {agent_id[:8]}... with score {quality_score:.2f}")
        except Exception as e:
            logger.error(f"Failed to update reputation: {e}")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

class PoGQValidator:
    """
    Proof of Good Quality (PoGQ) Byzantine Detection
    Based on the research paper implementation
    """
    
    def __init__(self, byzantine_threshold: float = 0.95):
        self.byzantine_threshold = byzantine_threshold
        self.gradient_history = {}
        self.validation_cache = {}
        
    def validate_gradient(self, agent_id: str, gradient: List[float]) -> float:
        """
        Validate gradient using PoGQ algorithm
        Returns quality score between 0 and 1
        """
        # Store gradient for pattern analysis
        if agent_id not in self.gradient_history:
            self.gradient_history[agent_id] = []
        self.gradient_history[agent_id].append(gradient)
        
        # Run PoGQ validation checks
        checks = {
            "magnitude": self._check_magnitude(gradient),
            "variance": self._check_variance(gradient),
            "sparsity": self._check_sparsity(gradient),
            "consistency": self._check_consistency(agent_id, gradient),
            "statistical": self._check_statistical_properties(gradient)
        }
        
        # Calculate weighted quality score
        weights = {
            "magnitude": 0.2,
            "variance": 0.2,
            "sparsity": 0.2,
            "consistency": 0.25,
            "statistical": 0.15
        }
        
        quality_score = sum(
            checks[check] * weights[check] 
            for check in checks
        )
        
        # Cache the validation result
        self.validation_cache[agent_id] = {
            "score": quality_score,
            "timestamp": datetime.now(),
            "checks": checks
        }
        
        return quality_score
    
    def _check_magnitude(self, gradient: List[float]) -> float:
        """Check if gradient magnitude is reasonable"""
        magnitude = sum(g**2 for g in gradient) ** 0.5
        
        # Abnormal if too small or too large
        if magnitude < 0.01:
            return 0.0  # Too small, likely fake
        elif magnitude > 100:
            return 0.0  # Too large, likely attack
        elif magnitude > 10:
            return 0.5  # Suspicious
        else:
            # Normal range
            return 1.0
    
    def _check_variance(self, gradient: List[float]) -> float:
        """Check gradient variance"""
        if len(gradient) < 2:
            return 0.0
            
        variance = statistics.variance(gradient)
        
        # Suspicious if too uniform or too noisy
        if variance < 0.001:
            return 0.0  # Too uniform, likely fake
        elif variance > 100:
            return 0.0  # Too noisy, likely attack
        else:
            # Score based on reasonable variance
            return min(1.0, 1.0 / (1.0 + abs(variance - 1.0)))
    
    def _check_sparsity(self, gradient: List[float]) -> float:
        """Check gradient sparsity pattern"""
        zero_count = sum(1 for g in gradient if abs(g) < 1e-6)
        sparsity = zero_count / len(gradient)
        
        # Suspicious if too sparse or too dense
        if sparsity > 0.95:
            return 0.0  # Too sparse
        elif sparsity < 0.05:
            return 0.8  # Very dense, slightly suspicious
        else:
            # Normal sparsity range
            return 1.0
    
    def _check_consistency(self, agent_id: str, gradient: List[float]) -> float:
        """Check consistency with agent's history"""
        history = self.gradient_history.get(agent_id, [])
        if len(history) < 2:
            return 0.7  # Neutral for new agents
        
        # Compare with last gradient
        last_gradient = history[-2] if len(history) > 1 else history[-1]
        
        # Calculate similarity
        similarity = self._cosine_similarity(gradient, last_gradient)
        
        # Too similar is suspicious (copy attack)
        if similarity > 0.99:
            return 0.0
        # Too different is also suspicious
        elif similarity < 0.1:
            return 0.3
        else:
            return 1.0
    
    def _check_statistical_properties(self, gradient: List[float]) -> float:
        """Check statistical properties of gradient"""
        mean = statistics.mean(gradient)
        median = statistics.median(gradient)
        
        # Check for statistical anomalies
        if abs(mean) > 10:
            return 0.0  # Mean too large
        
        # Check skewness (simplified)
        skew_indicator = abs(mean - median) / (statistics.stdev(gradient) + 1e-6)
        if skew_indicator > 2:
            return 0.3  # Highly skewed
        
        return 1.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        if len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a**2 for a in vec1) ** 0.5
        mag2 = sum(b**2 for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def is_byzantine(self, agent_id: str) -> bool:
        """Check if agent is Byzantine based on history"""
        if agent_id not in self.validation_cache:
            return False
            
        cache = self.validation_cache[agent_id]
        return cache["score"] < (1.0 - self.byzantine_threshold)

class IntegratedFLSystem:
    """
    Production FL System with Holochain + PoGQ Integration
    """
    
    def __init__(self, conductor_url: str = "ws://localhost:65000"):
        self.reputation_dht = HolochainReputationDHT(conductor_url)
        self.pogq_validator = PoGQValidator()
        self.round_number = 0
        self.validations = []
        
        # Configuration
        self.min_reputation = 0.3
        self.min_quality = 0.5
        self.grace_rounds = 2
        
    async def initialize(self):
        """Initialize the integrated system"""
        await self.reputation_dht.connect()
        logger.info("Integrated FL system initialized")
    
    async def process_gradient(
        self, 
        agent_id: str, 
        gradient: List[float]
    ) -> GradientValidation:
        """
        Process gradient submission with full validation
        """
        # Step 1: Check reputation from Holochain DHT
        reputation_score = await self.reputation_dht.get_reputation(agent_id)
        
        # Step 2: Check if agent can participate
        if reputation_score < self.min_reputation:
            # Check if in grace period
            agent_rounds = len([
                v for v in self.validations 
                if v.agent_id == agent_id
            ])
            
            if agent_rounds >= self.grace_rounds:
                return GradientValidation(
                    agent_id=agent_id,
                    round=self.round_number,
                    quality_score=0.0,
                    pogq_score=0.0,
                    reputation_score=reputation_score,
                    accepted=False,
                    reason=f"Insufficient reputation: {reputation_score:.2f}",
                    timestamp=datetime.now()
                )
        
        # Step 3: Run PoGQ validation
        pogq_score = self.pogq_validator.validate_gradient(agent_id, gradient)
        
        # Step 4: Combine scores for final quality
        quality_score = (pogq_score * 0.7) + (reputation_score * 0.3)
        
        # Step 5: Determine acceptance
        accepted = quality_score >= self.min_quality
        
        if not accepted:
            if pogq_score < 0.3:
                reason = "Failed PoGQ validation (likely Byzantine)"
            elif reputation_score < 0.2:
                reason = "Very low reputation"
            else:
                reason = f"Below quality threshold ({quality_score:.2f} < {self.min_quality})"
        else:
            reason = "Accepted"
        
        # Step 6: Update reputation on Holochain
        await self.reputation_dht.update_reputation(
            agent_id,
            quality_score,
            self.round_number
        )
        
        # Step 7: Record validation
        validation = GradientValidation(
            agent_id=agent_id,
            round=self.round_number,
            quality_score=quality_score,
            pogq_score=pogq_score,
            reputation_score=reputation_score,
            accepted=accepted,
            reason=reason,
            timestamp=datetime.now()
        )
        
        self.validations.append(validation)
        
        logger.info(
            f"Processed gradient from {agent_id[:8]}... - "
            f"Quality: {quality_score:.2f}, PoGQ: {pogq_score:.2f}, "
            f"Rep: {reputation_score:.2f}, Accepted: {accepted}"
        )
        
        return validation
    
    async def aggregate_round(self, gradients: Dict[str, List[float]]) -> Optional[List[float]]:
        """
        Aggregate gradients for current round with Byzantine filtering
        """
        self.round_number += 1
        accepted_gradients = {}
        
        # Process all gradients
        for agent_id, gradient in gradients.items():
            validation = await self.process_gradient(agent_id, gradient)
            
            if validation.accepted:
                # Weight by combined score
                weight = validation.quality_score
                accepted_gradients[agent_id] = (gradient, weight)
        
        if not accepted_gradients:
            logger.warning("No valid gradients in this round")
            return None
        
        # Weighted average aggregation
        aggregated = [0.0] * len(next(iter(accepted_gradients.values()))[0])
        total_weight = sum(weight for _, weight in accepted_gradients.values())
        
        for agent_id, (gradient, weight) in accepted_gradients.items():
            normalized_weight = weight / total_weight
            for i, value in enumerate(gradient):
                aggregated[i] += value * normalized_weight
        
        logger.info(
            f"Round {self.round_number} complete - "
            f"Accepted {len(accepted_gradients)}/{len(gradients)} gradients"
        )
        
        return aggregated
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        if not self.validations:
            return {"status": "No data yet"}
        
        total = len(self.validations)
        accepted = sum(1 for v in self.validations if v.accepted)
        byzantine_detected = sum(
            1 for agent_id in self.pogq_validator.validation_cache
            if self.pogq_validator.is_byzantine(agent_id)
        )
        
        return {
            "total_submissions": total,
            "accepted_submissions": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "byzantine_agents_detected": byzantine_detected,
            "current_round": self.round_number,
            "unique_agents": len(set(v.agent_id for v in self.validations))
        }
    
    async def close(self):
        """Clean shutdown"""
        await self.reputation_dht.close()
        logger.info("Integrated FL system shut down")

# Testing functions
async def test_integrated_system():
    """Test the integrated Holochain + PoGQ system"""
    system = IntegratedFLSystem()
    await system.initialize()
    
    # Simulate different types of agents
    agents = {
        "honest_1": [0.1, 0.2, -0.1, 0.3, 0.15],  # Normal gradient
        "honest_2": [0.05, -0.1, 0.2, 0.1, -0.05],  # Normal gradient
        "byzantine_1": [100, 100, 100, 100, 100],  # Attack gradient
        "lazy_1": [0, 0, 0, 0, 0],  # Zero gradient
        "noisy_1": [10, -10, 10, -10, 10],  # Noisy gradient
    }
    
    print("🧪 Testing Integrated FL System")
    print("=" * 50)
    
    # Round 1
    print("\n📍 Round 1:")
    gradients_r1 = agents.copy()
    aggregated = await system.aggregate_round(gradients_r1)
    
    if aggregated:
        print(f"Aggregated gradient: {[f'{v:.3f}' for v in aggregated]}")
    
    # Round 2 - Agents adapt
    print("\n📍 Round 2:")
    agents["byzantine_1"] = [1, 1, 1, 1, 1]  # Byzantine tries smaller values
    agents["lazy_1"] = [0.01, 0.01, 0.01, 0.01, 0.01]  # Lazy tries minimal effort
    
    aggregated = await system.aggregate_round(agents)
    
    if aggregated:
        print(f"Aggregated gradient: {[f'{v:.3f}' for v in aggregated]}")
    
    # Show statistics
    print("\n📊 System Statistics:")
    stats = system.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    await system.close()

async def test_sybil_resistance():
    """Test Sybil attack resistance"""
    system = IntegratedFLSystem()
    await system.initialize()
    
    print("\n🛡️ Testing Sybil Attack Resistance")
    print("=" * 50)
    
    # Create 50 Sybil agents
    sybil_gradients = {}
    for i in range(50):
        agent_id = f"sybil_{i}"
        # All submit similar malicious gradients
        gradient = [1.0 + i*0.01, 1.0, 1.0, 1.0, 1.0]
        sybil_gradients[agent_id] = gradient
    
    # Add a few honest agents
    sybil_gradients["honest_1"] = [0.1, 0.2, -0.1, 0.3, 0.15]
    sybil_gradients["honest_2"] = [0.05, -0.1, 0.2, 0.1, -0.05]
    
    # Process round
    aggregated = await system.aggregate_round(sybil_gradients)
    
    stats = system.get_system_stats()
    print(f"Total submissions: {stats['total_submissions']}")
    print(f"Accepted: {stats['accepted_submissions']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"Byzantine detected: {stats['byzantine_agents_detected']}")
    
    if aggregated:
        print(f"Aggregated gradient: {[f'{v:.3f}' for v in aggregated]}")
        
        # Check if aggregated gradient is close to honest agents
        honest_avg = [0.075, 0.05, 0.05, 0.2, 0.05]
        distance = sum((a-b)**2 for a, b in zip(aggregated, honest_avg)) ** 0.5
        print(f"Distance from honest average: {distance:.3f}")
        
        if distance < 0.5:
            print("✅ SYBIL ATTACK SUCCESSFULLY DEFENDED!")
        else:
            print("⚠️ Sybil attack partially successful")
    else:
        print("✅ All Sybil agents blocked!")
    
    await system.close()

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_integrated_system())
    asyncio.run(test_sybil_resistance())