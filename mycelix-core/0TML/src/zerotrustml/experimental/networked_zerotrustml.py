#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Network-Enabled ZeroTrustML Node
Combines Trust Layer with real P2P networking
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from .network_layer import NetworkNode, Message
from ..modular_architecture import ZeroTrustMLFactory, GradientMetadata, UseCase


class NetworkedZeroTrustMLNode:
    """
    ZeroTrustML node with real P2P networking

    Features:
    - Real network communication (not direct Python calls)
    - Byzantine-resistant gradient validation
    - Reputation-based peer selection
    - Async message handling
    """

    def __init__(
        self,
        node_id: int,
        listen_port: int,
        byzantine: bool = False,
        use_case: UseCase = UseCase.RESEARCH,
        bootstrap_peers: Optional[List[str]] = None
    ):
        self.node_id = node_id
        self.byzantine = byzantine

        # Create Trust Layer
        if use_case == UseCase.RESEARCH:
            self.trust = ZeroTrustMLFactory.for_research(node_id)
        else:
            # Could be other use cases
            self.trust = ZeroTrustMLFactory.for_research(node_id)

        # Create Network Layer
        self.network = NetworkNode(node_id, listen_port, bootstrap_peers)

        # State
        self.current_round = 0
        self.received_gradients: Dict[int, np.ndarray] = {}
        self.aggregated_gradient: Optional[np.ndarray] = None

        # Statistics
        self.rounds_completed = 0
        self.gradients_validated = 0
        self.byzantine_detected_count = 0

        # Register message handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register network message handlers"""
        self.network.register_handler('gradient', self._handle_gradient)
        self.network.register_handler('validation', self._handle_validation)
        self.network.register_handler('peer_list', self._handle_peer_list)

    async def start(self):
        """Start the node"""
        await self.network.start()
        print(f"🚀 NetworkedZeroTrustML Node {self.node_id} ready")

    async def stop(self):
        """Stop the node"""
        await self.trust.shutdown()
        await self.network.stop()

    async def _handle_gradient(self, message: Message):
        """Handle received gradient"""
        import base64

        # Deserialize gradient
        gradient_b64 = message.payload['gradient']
        gradient_bytes = base64.b64decode(gradient_b64)
        shape = tuple(message.payload['shape'])
        dtype = np.dtype(message.payload['dtype'])
        gradient = np.frombuffer(gradient_bytes, dtype=dtype).reshape(shape)

        # Validate gradient using Trust Layer
        is_valid = await self.trust.validate_gradient(
            gradient,
            peer_id=message.sender_id,
            round_num=message.round_num
        )

        # Store if valid
        if is_valid:
            self.received_gradients[message.sender_id] = gradient
            self.gradients_validated += 1
        else:
            self.byzantine_detected_count += 1
            print(f"🚨 Node {self.node_id}: Detected Byzantine gradient from {message.sender_id}")

        # Get reputation
        reputation = 0.7
        if message.sender_id in self.trust.trust_layer.peer_reputations:
            reputation = self.trust.trust_layer.peer_reputations[message.sender_id].reputation_score

        # Send validation result back
        await self.network.send_validation_result(
            peer_id=message.sender_id,
            round_num=message.round_num,
            is_valid=is_valid,
            reputation_score=reputation
        )

    async def _handle_validation(self, message: Message):
        """Handle validation feedback from peer"""
        is_valid = message.payload['is_valid']
        reputation = message.payload['reputation_score']

        if not is_valid:
            print(f"⚠️  Node {self.node_id}: Peer {message.sender_id} rejected our gradient")
        else:
            print(f"✅ Node {self.node_id}: Peer {message.sender_id} accepted our gradient (rep: {reputation:.2f})")

    async def _handle_peer_list(self, message: Message):
        """Handle peer list update"""
        peers = message.payload['peers']
        print(f"📋 Node {self.node_id}: Received {len(peers)} peer addresses")

        # Could connect to new peers here
        # for peer in peers:
        #     if peer['peer_id'] != self.node_id:
        #         await self._connect_to_peer(peer)

    def compute_gradient(self) -> np.ndarray:
        """Compute gradient (honest or Byzantine)"""
        if self.byzantine:
            # Byzantine attack
            attack_type = self.node_id % 5
            if attack_type == 0:
                return np.random.randn(100) * 100  # Large noise
            elif attack_type == 1:
                return np.zeros(100)  # All zeros
            elif attack_type == 2:
                return np.ones(100) * 5.0  # Constant
            elif attack_type == 3:
                grad = np.random.randn(100)
                return -grad * 10  # Sign flip
            else:
                return np.random.randn(100) * (np.random.rand(100) < 0.01)  # Sparse
        else:
            # Honest gradient
            return np.random.randn(100) * 0.1

    async def federated_round(self, round_num: int):
        """Execute one round of federated learning"""
        self.current_round = round_num
        self.received_gradients.clear()

        print(f"\n📍 Node {self.node_id} - Round {round_num}")

        # Compute gradient
        gradient = self.compute_gradient()

        # Broadcast to all peers
        await self.network.broadcast_gradient(
            gradient,
            round_num,
            metadata={'byzantine': self.byzantine}
        )

        # Wait for gradients from peers
        await asyncio.sleep(2)  # Give time for all gradients to arrive

        # Aggregate valid gradients
        if self.received_gradients:
            valid_gradients = list(self.received_gradients.values())
            self.aggregated_gradient = np.mean(valid_gradients, axis=0)
            print(f"   Node {self.node_id}: Aggregated {len(valid_gradients)} gradients")
        else:
            print(f"   Node {self.node_id}: No valid gradients received")

        self.rounds_completed += 1

    def get_statistics(self) -> Dict:
        """Get node statistics"""
        network_stats = self.network.get_statistics()

        return {
            'node_id': self.node_id,
            'byzantine': self.byzantine,
            'rounds_completed': self.rounds_completed,
            'gradients_validated': self.gradients_validated,
            'byzantine_detected': self.byzantine_detected_count,
            **network_stats
        }


# ============================================================
# Testing Networked ZeroTrustML
# ============================================================

async def test_networked_zerotrustml():
    """Test networked ZeroTrustML with Byzantine nodes"""
    print("\n" + "="*60)
    print("🧪 TESTING NETWORKED ZEROTRUSTML")
    print("="*60)

    # Create 5 honest nodes and 2 Byzantine nodes
    nodes = []
    base_port = 7000

    # Node 0 (honest, bootstrap)
    node0 = NetworkedZeroTrustMLNode(
        node_id=0,
        listen_port=base_port,
        byzantine=False
    )
    await node0.start()
    nodes.append(node0)
    await asyncio.sleep(0.5)

    # Honest nodes
    for i in range(1, 5):
        node = NetworkedZeroTrustMLNode(
            node_id=i,
            listen_port=base_port + i,
            byzantine=False,
            bootstrap_peers=[f'localhost:{base_port}']
        )
        await node.start()
        nodes.append(node)
        await asyncio.sleep(0.3)

    # Byzantine nodes
    for i in range(5, 7):
        node = NetworkedZeroTrustMLNode(
            node_id=i,
            listen_port=base_port + i,
            byzantine=True,
            bootstrap_peers=[f'localhost:{base_port}']
        )
        await node.start()
        nodes.append(node)
        await asyncio.sleep(0.3)

    print(f"\n🌐 Network established:")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Honest: 5")
    print(f"   Byzantine: 2")

    # Wait for network to stabilize
    await asyncio.sleep(2)

    # Run 3 federated rounds
    for round_num in range(3):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}")
        print(f"{'='*60}")

        # All nodes participate
        await asyncio.gather(*[
            node.federated_round(round_num)
            for node in nodes
        ])

        await asyncio.sleep(1)

    # Print final statistics
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)

    for node in nodes:
        stats = node.get_statistics()
        print(f"\nNode {stats['node_id']} ({'Byzantine' if stats['byzantine'] else 'Honest'}):")
        print(f"   Rounds completed: {stats['rounds_completed']}")
        print(f"   Gradients validated: {stats['gradients_validated']}")
        print(f"   Byzantine detected: {stats['byzantine_detected']}")
        print(f"   Connected peers: {stats['connected_peers']}")
        print(f"   Messages sent: {stats['messages_sent']}")
        print(f"   Messages received: {stats['messages_received']}")

    # Calculate detection rate
    total_byzantine_detected = sum(
        node.byzantine_detected_count
        for node in nodes
        if not node.byzantine
    )
    total_byzantine_gradients = sum(
        node.rounds_completed
        for node in nodes
        if node.byzantine
    ) * len([n for n in nodes if not n.byzantine])  # Byzantine gradients * honest validators

    if total_byzantine_gradients > 0:
        detection_rate = total_byzantine_detected / total_byzantine_gradients
        print(f"\n🎯 Byzantine Detection Rate: {detection_rate:.1%}")

        if detection_rate >= 0.9:
            print("✅ PASSED (≥90% detection)")
        else:
            print("❌ FAILED (<90% detection)")

    # Cleanup
    for node in nodes:
        await node.stop()

    print("\n✅ Networked ZeroTrustML test complete!")


async def main():
    """Main entry point"""
    try:
        await test_networked_zerotrustml()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
