# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Staking Coordinator - Economic hardening for Gen-7.

Core Concept:
Make Byzantine attacks economically irrational through stake-weighted
participation and automatic slashing for invalid proofs.

Economic Model:
- Clients stake tokens to participate in training
- Valid proof submission → stake returned + reward (5% reputation increase)
- Invalid proof submission → stake slashed (10% penalty) + reputation halved
- Aggregation weighted by stake × reputation

Security Analysis:
- Attack cost: 10% stake × number of attempts
- Attack benefit: Model poisoning (hard to quantify)
- For rational attackers: cost must exceed benefit
- System secure when honest stake > byzantine stake
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gradient_proof import GradientProof


@dataclass
class ClientStake:
    """Client staking information.

    Attributes:
        client_id: Unique client identifier
        stake_amount: Tokens staked for participation
        reputation: Reputation score (1.0 = neutral, >1.0 = good, <1.0 = bad)
        total_submissions: Total gradient submissions
        valid_submissions: Number of valid proofs
        invalid_submissions: Number of invalid proofs
        total_slashed: Total tokens slashed from invalid proofs
        total_rewards: Total reputation rewards from valid proofs
    """

    client_id: str
    stake_amount: float
    reputation: float = 1.0
    total_submissions: int = 0
    valid_submissions: int = 0
    invalid_submissions: int = 0
    total_slashed: float = 0.0
    total_rewards: float = 0.0

    def weight(self) -> float:
        """Aggregation weight (stake × reputation)."""
        return self.stake_amount * self.reputation

    def success_rate(self) -> float:
        """Fraction of valid submissions."""
        if self.total_submissions == 0:
            return 0.0
        return self.valid_submissions / self.total_submissions


class StakingCoordinator:
    """Economic staking coordinator for Gen-7 FL.

    Manages client stakes, verifies proofs, and coordinates
    stake-weighted gradient aggregation.

    Features:
    - Registration with minimum stake requirement
    - Proof verification with automatic slashing
    - Reputation-weighted aggregation
    - Economic attack cost tracking
    """

    def __init__(
        self,
        min_stake: float = 10.0,
        slash_rate: float = 0.10,
        reputation_increase: float = 1.05,
        reputation_decrease: float = 0.50,
    ):
        """Initialize staking coordinator.

        Args:
            min_stake: Minimum stake required to participate
            slash_rate: Fraction of stake slashed for invalid proof (default: 10%)
            reputation_increase: Reputation multiplier for valid proof (default: 1.05 = 5% increase)
            reputation_decrease: Reputation multiplier for invalid proof (default: 0.5 = 50% decrease)
        """
        self.min_stake = min_stake
        self.slash_rate = slash_rate
        self.reputation_increase = reputation_increase
        self.reputation_decrease = reputation_decrease

        # Client registry
        self.clients: Dict[str, ClientStake] = {}

        # Current round state
        self.round_idx = 0
        self.current_model_hash = None
        self.valid_gradients: List[Tuple[str, np.ndarray]] = []

        # Economic tracking
        self.total_slashed = 0.0
        self.total_rewards = 0.0

    def register_client(
        self,
        client_id: str,
        stake: float,
    ) -> bool:
        """Register client with stake deposit.

        Args:
            client_id: Unique client identifier
            stake: Amount to stake (must be >= min_stake)

        Returns:
            True if registration successful, False if stake too low

        Raises:
            ValueError: If client already registered
        """
        if client_id in self.clients:
            raise ValueError(f"Client {client_id} already registered")

        if stake < self.min_stake:
            print(
                f"❌ Registration failed: stake {stake} < minimum {self.min_stake}"
            )
            return False

        self.clients[client_id] = ClientStake(
            client_id=client_id,
            stake_amount=stake,
        )

        print(f"✅ Client {client_id} registered with stake {stake}")
        return True

    def unregister_client(self, client_id: str) -> float:
        """Unregister client and return remaining stake.

        Args:
            client_id: Client to unregister

        Returns:
            Remaining stake after slashing

        Raises:
            KeyError: If client not registered
        """
        if client_id not in self.clients:
            raise KeyError(f"Client {client_id} not registered")

        client = self.clients[client_id]
        remaining_stake = client.stake_amount

        del self.clients[client_id]

        print(
            f"✅ Client {client_id} unregistered, returned stake: {remaining_stake}"
        )
        return remaining_stake

    def submit_gradient(
        self,
        proof: GradientProof,
    ) -> bool:
        """Submit gradient with proof for verification.

        This is the CRITICAL security function. It:
        1. Verifies client has sufficient stake
        2. Verifies zkSTARK proof validity
        3. Applies economic consequences (reward or slash)
        4. Stores valid gradients for aggregation

        Args:
            proof: GradientProof containing gradient and zkSTARK proof

        Returns:
            True if proof valid (gradient accepted), False if invalid (gradient rejected)

        Side Effects:
            - Valid proof → reputation increased, gradient stored
            - Invalid proof → stake slashed, reputation decreased
        """
        client_id = proof.public_inputs["client_id"]

        # 1. Verify client registered and has stake
        if client_id not in self.clients:
            print(f"❌ Submission rejected: Client {client_id} not registered")
            return False

        client = self.clients[client_id]

        if client.stake_amount < self.min_stake:
            print(
                f"❌ Submission rejected: Client {client_id} stake {client.stake_amount} < minimum {self.min_stake}"
            )
            return False

        # 2. Verify zkSTARK proof
        proof_valid = proof.verify()

        # 3. Update client state and apply economic consequences
        client.total_submissions += 1

        if proof_valid:
            # Valid proof → accept gradient, reward client
            client.valid_submissions += 1
            client.reputation *= self.reputation_increase
            client.total_rewards += self.reputation_increase - 1.0

            self.valid_gradients.append((client_id, proof.gradient))
            self.total_rewards += self.reputation_increase - 1.0

            print(
                f"✅ Proof accepted from {client_id} "
                f"(reputation: {client.reputation:.3f}, weight: {client.weight():.3f})"
            )
            return True
        else:
            # Invalid proof → reject gradient, slash stake
            client.invalid_submissions += 1

            slashed_amount = client.stake_amount * self.slash_rate
            client.stake_amount -= slashed_amount
            client.total_slashed += slashed_amount

            client.reputation *= self.reputation_decrease

            self.total_slashed += slashed_amount

            print(
                f"⚠️ Invalid proof from {client_id}: "
                f"slashed {slashed_amount:.2f} tokens "
                f"(remaining stake: {client.stake_amount:.2f}, "
                f"reputation: {client.reputation:.3f})"
            )
            return False

    def aggregate_gradients(self) -> Optional[np.ndarray]:
        """Aggregate valid gradients with stake-weighted averaging.

        Uses stake × reputation as aggregation weight, giving more
        influence to clients with high stake and good track record.

        Returns:
            Aggregated gradient, or None if no valid gradients
        """
        if not self.valid_gradients:
            print("⚠️ No valid gradients to aggregate")
            return None

        # Compute stake-weighted average
        weighted_sum = np.zeros_like(self.valid_gradients[0][1])
        total_weight = 0.0

        for client_id, gradient in self.valid_gradients:
            client = self.clients[client_id]

            # Weight = stake × reputation
            weight = client.weight()
            weighted_sum += weight * gradient
            total_weight += weight

        aggregated = weighted_sum / total_weight

        print(
            f"✅ Aggregated {len(self.valid_gradients)} gradients "
            f"(total weight: {total_weight:.2f})"
        )

        return aggregated

    def advance_round(self, new_model_hash: str):
        """Advance to next training round.

        Clears valid gradients buffer and updates round state.

        Args:
            new_model_hash: Hash of new global model
        """
        self.round_idx += 1
        self.current_model_hash = new_model_hash
        self.valid_gradients = []

        print(f"\n🔄 Advanced to round {self.round_idx}")

    def get_client_info(self, client_id: str) -> Optional[ClientStake]:
        """Get client staking information.

        Args:
            client_id: Client to query

        Returns:
            ClientStake object, or None if not registered
        """
        return self.clients.get(client_id)

    def get_total_stake(self) -> float:
        """Total stake across all clients."""
        return sum(client.stake_amount for client in self.clients.values())

    def get_total_weight(self) -> float:
        """Total weight (stake × reputation) across all clients."""
        return sum(client.weight() for client in self.clients.values())

    def get_economic_security_metrics(self) -> Dict:
        """Compute economic security metrics.

        Returns:
            Dictionary with attack cost analysis
        """
        total_stake = self.get_total_stake()
        total_weight = self.get_total_weight()

        # Estimate attack cost
        # Assumption: attacker needs 50% of weight to poison model
        # Cost = stake needed × slash rate × expected detection probability
        honest_weight = total_weight
        attack_weight_needed = honest_weight  # Need to match honest weight

        # How much stake needed to get attack_weight?
        # Assume attacker starts with reputation = 1.0
        attack_stake_needed = attack_weight_needed / 1.0

        # Expected cost if attack detected (assume 50% detection prob initially)
        attack_expected_cost = attack_stake_needed * self.slash_rate * 0.5

        return {
            "total_stake": total_stake,
            "total_weight": total_weight,
            "total_slashed": self.total_slashed,
            "total_rewards": self.total_rewards,
            "attack_stake_needed": attack_stake_needed,
            "attack_expected_cost": attack_expected_cost,
            "attack_cost_per_attempt": attack_stake_needed * self.slash_rate,
        }

    def print_summary(self):
        """Print summary of staking coordinator state."""
        print("\n" + "=" * 80)
        print("STAKING COORDINATOR SUMMARY")
        print("=" * 80)

        print(f"\n📊 Round {self.round_idx}")
        print(f"   Clients registered: {len(self.clients)}")
        print(f"   Total stake: {self.get_total_stake():.2f}")
        print(f"   Total weight: {self.get_total_weight():.2f}")

        print(f"\n💰 Economic Metrics")
        print(f"   Total slashed: {self.total_slashed:.2f}")
        print(f"   Total rewards: {self.total_rewards:.2f}")

        metrics = self.get_economic_security_metrics()
        print(f"\n🛡️ Security Analysis")
        print(
            f"   Attack stake needed: {metrics['attack_stake_needed']:.2f} "
            f"({metrics['attack_stake_needed']/metrics['total_stake']*100:.1f}% of total)"
        )
        print(f"   Attack expected cost: {metrics['attack_expected_cost']:.2f}")
        print(
            f"   Attack cost per attempt: {metrics['attack_cost_per_attempt']:.2f}"
        )

        print(f"\n👥 Top Clients by Weight:")
        top_clients = sorted(
            self.clients.values(), key=lambda c: c.weight(), reverse=True
        )[:5]

        for i, client in enumerate(top_clients, 1):
            print(
                f"   {i}. {client.client_id[:8]}: "
                f"stake={client.stake_amount:.2f}, "
                f"rep={client.reputation:.3f}, "
                f"weight={client.weight():.2f}, "
                f"success={client.success_rate()*100:.1f}%"
            )

        print("=" * 80 + "\n")
