#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Hybrid ZeroTrustML System with better connectivity
Ensures all nodes see Byzantine behavior for accurate detection metrics
"""

import asyncio
import numpy as np
import sys
import os
sys.path.append('/srv/luminous-dynamics/Mycelix-Core/mycelix-fl-pure-p2p/src')
sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')

from hybrid_zerotrustml_complete import HybridZeroTrustMLNode

async def test_byzantine_detection():
    """Test with direct Byzantine exposure for all nodes"""
    
    print("🧪 Testing Byzantine Detection in Hybrid ZeroTrustML")
    print("=" * 60)
    
    # Create nodes
    honest_nodes = []
    byzantine_nodes = []
    
    # Create 5 honest nodes
    for i in range(5):
        node = HybridZeroTrustMLNode(i, port=8000+i, is_byzantine=False)
        honest_nodes.append(node)
    
    # Create 2 Byzantine nodes
    for i, byz_id in enumerate([666, 777]):
        node = HybridZeroTrustMLNode(byz_id, port=8100+i, is_byzantine=True)
        byzantine_nodes.append(node)
    
    all_nodes = honest_nodes + byzantine_nodes
    
    # Create fully connected network (everyone sees everyone)
    print("\n📡 Creating fully connected network...")
    for node in all_nodes:
        for other in all_nodes:
            if node != other:
                node.connect_peer(other)
    
    print(f"✅ Network: {len(honest_nodes)} honest + {len(byzantine_nodes)} Byzantine")
    print(f"   Each node connected to {len(all_nodes)-1} peers")
    
    # Run multiple rounds
    detection_rates_per_round = []
    
    for round_id in range(5):
        print(f"\n📍 Round {round_id + 1}/5")
        print("-" * 40)
        
        # Collect gradients from all nodes
        all_gradients = []
        
        # Byzantine nodes send malicious gradients
        for byz in byzantine_nodes:
            # Different attack each time
            attack_type = ['noise', 'flip', 'zeros'][round_id % 3]
            if attack_type == 'noise':
                gradient = np.random.randn(1000) * 100  # Large noise
            elif attack_type == 'flip':
                gradient = -np.random.randn(1000) * 50  # Negative large
            else:
                gradient = np.zeros(1000)  # All zeros
            
            all_gradients.append((byz.node_id, gradient))
            print(f"Byzantine {byz.node_id} sent {attack_type} attack")
        
        # Honest nodes send normal gradients
        for honest in honest_nodes:
            gradient = np.random.randn(1000) * 0.1  # Normal gradient
            all_gradients.append((honest.node_id, gradient))
        
        # Each honest node validates all gradients
        detections = []
        for honest in honest_nodes:
            # Reset gradient buffer and add all gradients
            honest.gradient_buffer.clear()
            
            # Validate each gradient
            valid_count = 0
            byzantine_detected = 0
            
            for peer_id, gradient in all_gradients:
                if peer_id != honest.node_id:  # Don't validate own gradient
                    is_valid, trust_score, reason = honest.trust.validate_peer_gradient(
                        peer_id, gradient, honest.model
                    )
                    
                    if not is_valid and peer_id in [666, 777]:
                        byzantine_detected += 1
                    if is_valid:
                        valid_count += 1
            
            # Calculate detection rate for this node
            total_byzantine = len(byzantine_nodes)
            if total_byzantine > 0:
                detection_rate = byzantine_detected / total_byzantine
                detections.append(detection_rate)
                
            print(f"Node {honest.node_id}: Validated {valid_count}/{len(all_gradients)-1}, "
                  f"Byzantine detected: {byzantine_detected}/{total_byzantine} ({detection_rate:.0%})")
        
        # Average detection rate for this round
        if detections:
            round_detection = np.mean(detections)
            detection_rates_per_round.append(round_detection)
            print(f"\n📊 Round {round_id + 1} Average Detection: {round_detection:.1%}")
            
            # Show reputation evolution
            if round_id > 0:
                print("\n🔍 Reputation Status:")
                # Sample from first honest node
                sample_node = honest_nodes[0]
                for byz_id in [666, 777]:
                    if byz_id in sample_node.trust.peer_reputations:
                        rep = sample_node.trust.peer_reputations[byz_id].reputation_score
                        status = "🚫 Blacklisted" if rep < 0.3 else "⚠️ Suspicious" if rep < 0.6 else ""
                        print(f"  Byzantine {byz_id}: {rep:.3f} {status}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("✨ Test Complete - Byzantine Detection Analysis")
    print("=" * 60)
    
    if detection_rates_per_round:
        overall_detection = np.mean(detection_rates_per_round)
        print(f"\n🎯 Overall Byzantine Detection Rate: {overall_detection:.1%}")
        
        print("\n📈 Detection Rate by Round:")
        for i, rate in enumerate(detection_rates_per_round):
            bar = "█" * int(rate * 20)
            print(f"  Round {i+1}: {bar:<20} {rate:.1%}")
        
        print("\n📊 Performance Comparison:")
        print(f"  Pure P2P alone:     76.7% Byzantine detection")
        print(f"  Hybrid ZeroTrustML:     {overall_detection:.1%} Byzantine detection")
        
        if overall_detection >= 0.90:
            print("\n🎉 SUCCESS: Achieved 90%+ Byzantine detection!")
        elif overall_detection >= 0.85:
            print("\n✅ GOOD: Significant improvement over Pure P2P")
        else:
            print("\n🔧 NEEDS TUNING: Detection below target")
    
    # Show final reputations
    print("\n🏆 Final Reputation Scores (from Node 0 perspective):")
    sample_node = honest_nodes[0]
    all_reps = []
    for peer_id, rep_obj in sample_node.trust.peer_reputations.items():
        node_type = "Byzantine" if peer_id in [666, 777] else "Honest"
        all_reps.append((peer_id, rep_obj.reputation_score, node_type))
    
    all_reps.sort(key=lambda x: x[1], reverse=True)
    for peer_id, score, node_type in all_reps:
        status = "🚫" if score < 0.3 else "⚠️" if score < 0.6 else "✅"
        print(f"  Node {peer_id:3d} ({node_type:9s}): {score:.3f} {status}")

if __name__ == "__main__":
    asyncio.run(test_byzantine_detection())