#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Comprehensive Federated Learning Test

Validates production scalability with:
- 10 hospitals (7 honest + 3 Byzantine)
- 20 training rounds
- Real Bulletproofs
- Performance benchmarking
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zerotrustml.demo.hospital_node import HospitalNode
from zerotrustml.demo.byzantine_node import ByzantineNode

# Import the base demo class
exec(open('test_multi_hospital_demo.py').read())


class ComprehensiveTest(FederatedLearningDemo):
    """Extended demo for comprehensive testing."""

    def __init__(self):
        super().__init__()
        self.num_rounds = 20  # Scale up to 20 rounds

    async def initialize(self):
        """Initialize with 10 hospitals."""
        print("=" * 70)
        print("🏥 Comprehensive Federated Learning Test")
        print("=" * 70)
        print()
        print("Configuration:")
        print(f"  • Hospitals: 10 (7 honest + 3 Byzantine)")
        print(f"  • Rounds: {self.num_rounds}")
        print(f"  • Model parameters: {self.num_params}")
        print(f"  • ZK Proofs: Real 608-byte Bulletproofs")
        print()

        # 1. Initialize coordinator
        print("1. Initializing Phase 10 Coordinator...")
        from zerotrustml.core.phase10_coordinator import Phase10Coordinator, Phase10Config

        config = Phase10Config(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_db="zerotrustml",
            postgres_user="zerotrustml",
            postgres_password="",
            holochain_enabled=False,
            zkpoc_enabled=True,
            zkpoc_pogq_threshold=0.7,
            hybrid_sync_enabled=False
        )

        self.coordinator = Phase10Coordinator(config)
        await self.coordinator.initialize()
        print("   ✅ Coordinator ready")
        print()

        # 2. Create 7 honest hospitals
        print("2. Creating honest hospitals...")

        self.hospitals.append(
            HospitalNode("mayo-clinic", num_patients=15000, data_quality=0.95)
        )
        print("   ✅ Mayo Clinic (15,000 patients, 95% quality)")

        self.hospitals.append(
            HospitalNode("johns-hopkins", num_patients=12000, data_quality=0.93)
        )
        print("   ✅ Johns Hopkins (12,000 patients, 93% quality)")

        self.hospitals.append(
            HospitalNode("cleveland-clinic", num_patients=10000, data_quality=0.91)
        )
        print("   ✅ Cleveland Clinic (10,000 patients, 91% quality)")

        self.hospitals.append(
            HospitalNode("mass-general", num_patients=9000, data_quality=0.90)
        )
        print("   ✅ Massachusetts General (9,000 patients, 90% quality)")

        self.hospitals.append(
            HospitalNode("cedars-sinai", num_patients=8000, data_quality=0.88)
        )
        print("   ✅ Cedars-Sinai (8,000 patients, 88% quality)")

        self.hospitals.append(
            HospitalNode("ucsf-medical", num_patients=7000, data_quality=0.86)
        )
        print("   ✅ UCSF Medical Center (7,000 patients, 86% quality)")

        self.hospitals.append(
            HospitalNode("community-regional", num_patients=4000, data_quality=0.82)
        )
        print("   ✅ Community Regional (4,000 patients, 82% quality)")
        print()

        # 3. Create 3 Byzantine attackers
        print("3. Creating Byzantine attackers...")

        self.hospitals.append(
            ByzantineNode("attacker-poisoning", attack_type=ByzantineNode.ATTACK_MODEL_POISONING)
        )
        print("   🔴 Attacker 1 (model poisoning)")

        self.hospitals.append(
            ByzantineNode("attacker-noise", attack_type=ByzantineNode.ATTACK_GRADIENT_NOISE)
        )
        print("   🔴 Attacker 2 (gradient noise)")

        self.hospitals.append(
            ByzantineNode("attacker-freerider", attack_type=ByzantineNode.ATTACK_FREE_RIDER)
        )
        print("   🔴 Attacker 3 (free rider)")
        print()

        # 4. Initialize global model
        print("4. Initializing global model...")
        self.global_model_weights = [0.5] * self.num_params
        print(f"   ✅ Global model initialized ({self.num_params} parameters)")
        print()


async def main():
    """Run comprehensive test."""
    test = ComprehensiveTest()
    await test.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        import logging
        logging.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
