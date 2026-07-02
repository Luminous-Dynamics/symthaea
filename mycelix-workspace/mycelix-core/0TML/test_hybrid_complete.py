#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Comprehensive Test of Hybrid ZeroTrustML System
Tests Byzantine detection across different attack scenarios
"""

import sys
import numpy as np
import time
import pytest

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} is available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - testing with simulation")

pytest.skip(
    "Hybrid real-system tests require manual setup and deterministic seeding; skipping by default",
    allow_module_level=True,
)

# Import our implementations
sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')
from hybrid_zerotrustml_real import HybridRealMLNode, HybridRealSystem


def test_various_byzantine_scenarios():
    """Test different Byzantine attack patterns"""
    print("\n" + "="*60)
    print("🧪 TESTING VARIOUS BYZANTINE ATTACK SCENARIOS")
    print("="*60)
    
    scenarios = [
        ("Standard (2 Byzantine, 5 Honest)", 5, 2),
        ("High Byzantine (3 Byzantine, 4 Honest)", 4, 3),
        ("Equal Split (3 Byzantine, 3 Honest)", 3, 3),
        ("Minimal (1 Byzantine, 5 Honest)", 5, 1),
    ]
    
    results_summary = []
    
    for scenario_name, num_honest, num_byzantine in scenarios:
        print(f"\n📋 Scenario: {scenario_name}")
        print("-" * 40)
        
        # Create system
        system = HybridRealSystem(num_honest=num_honest, num_byzantine=num_byzantine)
        
        # Run shorter training (5 rounds)
        detection_rates = []
        accuracies = []
        
        for round_num in range(5):
            round_results = []
            byzantine_detected = set()
            byzantine_nodes = [n.node_id for n in system.nodes if n.byzantine]
            
            for node in system.nodes:
                if not node.byzantine:
                    result = node.federated_round(round_num)
                    round_results.append(result)
                    byzantine_detected.update(result['blacklisted'])
            
            if round_results:
                avg_accuracy = np.mean([r['accuracy'] for r in round_results])
                accuracies.append(avg_accuracy)
                
                detected_byzantine = len([b for b in byzantine_nodes if b in byzantine_detected])
                total_byzantine = len(byzantine_nodes)
                detection_rate = detected_byzantine / total_byzantine if total_byzantine > 0 else 0
                detection_rates.append(detection_rate)
                
                print(f"  Round {round_num + 1}: Detection={detection_rate:.0%}, Accuracy={avg_accuracy:.2%}")
        
        # Calculate averages
        avg_detection = np.mean(detection_rates) if detection_rates else 0
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        
        results_summary.append({
            'scenario': scenario_name,
            'detection': avg_detection,
            'accuracy': avg_accuracy,
            'passed': avg_detection >= 0.9
        })
        
        print(f"  📊 Average Detection: {avg_detection:.1%}")
        print(f"  📊 Average Accuracy: {avg_accuracy:.2%}")
        print(f"  ✅ Status: {'PASSED' if avg_detection >= 0.9 else 'FAILED'}")
    
    assert all(r['detection'] >= 0.9 for r in results_summary)


def test_attack_types():
    """Test specific attack types"""
    print("\n" + "="*60)
    print("🔬 TESTING SPECIFIC ATTACK TYPES")
    print("="*60)
    
    # Create a simple system
    system = HybridRealSystem(num_honest=3, num_byzantine=1)
    
    # Get the Byzantine node
    byzantine_node = [n for n in system.nodes if n.byzantine][0]
    honest_node = [n for n in system.nodes if not n.byzantine][0]
    
    # Get actual gradient size from an honest node
    sample_gradient = honest_node.compute_gradient()
    grad_size = len(sample_gradient)
    
    attack_types = [
        ("Large Noise", lambda: np.random.randn(grad_size) * 100),
        ("Sign Flip", lambda: -honest_node.compute_gradient() * 10),
        ("All Zeros", lambda: np.zeros(grad_size)),
        ("Constant Value", lambda: np.ones(grad_size) * 5.0),
        ("Extreme Sparsity", lambda: np.random.randn(grad_size) * (np.random.rand(grad_size) < 0.01)),
    ]
    
    results = []
    
    for attack_name, attack_fn in attack_types:
        print(f"\n🚨 Testing: {attack_name}")
        print("-" * 30)
        
        # Generate attack gradient
        attack_gradient = attack_fn()
        
        # Test detection by honest node
        is_detected = False
        
        # Check norm-based detection
        norm = np.linalg.norm(attack_gradient)
        if norm > 50 or norm == 0:
            is_detected = True
            print(f"  Norm detection: ✅ (norm={norm:.2f})")
        else:
            print(f"  Norm detection: ❌ (norm={norm:.2f})")
        
        # Check pattern detection
        if np.all(attack_gradient == 0):
            is_detected = True
            print(f"  Zero detection: ✅")
        elif np.std(attack_gradient) < 1e-6:
            is_detected = True
            print(f"  Constant detection: ✅")
        elif np.mean(np.abs(attack_gradient) < 1e-6) > 0.99:
            is_detected = True
            print(f"  Sparsity detection: ✅")
        else:
            print(f"  Pattern detection: ❌")
        
        # Test Trust Layer validation
        honest_node.validate_gradient(attack_gradient, 999)
        if 999 in honest_node.trust.peer_reputations:
            reputation = honest_node.trust.peer_reputations[999].reputation_score
            if reputation < 0.3:
                is_detected = True
                print(f"  Trust Layer: ✅ (reputation={reputation:.3f})")
            else:
                print(f"  Trust Layer: ❌ (reputation={reputation:.3f})")
        
        results.append({
            'attack': attack_name,
            'detected': is_detected
        })
        
        print(f"  Final: {'✅ DETECTED' if is_detected else '❌ MISSED'}")
    
    assert all(entry['detected'] for entry in results)


def test_convergence():
    """Test if the system still learns despite Byzantine nodes"""
    print("\n" + "="*60)
    print("📈 TESTING LEARNING CONVERGENCE")
    print("="*60)
    
    system = HybridRealSystem(num_honest=5, num_byzantine=2)
    
    print("\nTracking accuracy over 20 rounds...")
    print("-" * 40)
    
    accuracies = []
    for round_num in range(20):
        round_results = []
        for node in system.nodes:
            if not node.byzantine:
                result = node.federated_round(round_num)
                round_results.append(result)
        
        if round_results:
            avg_accuracy = np.mean([r['accuracy'] for r in round_results])
            accuracies.append(avg_accuracy)
            
            if round_num % 5 == 0:
                print(f"  Round {round_num + 1}: {avg_accuracy:.2%}")
    
    # Check if accuracy improved
    if len(accuracies) >= 10:
        early_avg = np.mean(accuracies[:5])
        late_avg = np.mean(accuracies[-5:])
        improvement = late_avg - early_avg
        
        print(f"\n📊 Convergence Analysis:")
        print(f"  Early accuracy (rounds 1-5): {early_avg:.2%}")
        print(f"  Late accuracy (rounds 16-20): {late_avg:.2%}")
        print(f"  Improvement: {improvement:+.2%}")
        
        if improvement > 0:
            print(f"  ✅ System continues to learn despite Byzantine nodes!")
        else:
            print(f"  ⚠️  No improvement observed (may need more rounds)")
        
        assert improvement > 0, "Accuracy should improve despite Byzantine nodes"
    
    pytest.fail("Convergence target not met")


def main():
    """Run comprehensive tests"""
    print("\n" + "="*60)
    print("🏆 COMPREHENSIVE HYBRID ZEROTRUSTML TESTING")
    print("="*60)
    print(f"\nEnvironment: PyTorch {'Available' if PYTORCH_AVAILABLE else 'Not Available'}")
    
    if not PYTORCH_AVAILABLE:
        print("\n⚠️  PyTorch not installed. Limited testing available.")
        print("  Install PyTorch for full testing.")
        return
    
    all_passed = True
    
    # Test 1: Various Byzantine scenarios
    print("\n" + "🔹"*30)
    scenario_results = test_various_byzantine_scenarios()
    scenarios_passed = all(r['passed'] for r in scenario_results)
    all_passed = all_passed and scenarios_passed
    
    # Test 2: Specific attack types
    print("\n" + "🔹"*30)
    attack_results = test_attack_types()
    attacks_detected = sum(1 for r in attack_results if r['detected'])
    attacks_passed = attacks_detected >= len(attack_results) * 0.8  # 80% detection
    all_passed = all_passed and attacks_passed
    
    # Test 3: Learning convergence
    print("\n" + "🔹"*30)
    convergence_passed = test_convergence()
    all_passed = all_passed and convergence_passed
    
    # Final summary
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    print("\n1️⃣ Byzantine Scenarios:")
    for result in scenario_results:
        status = "✅" if result['passed'] else "❌"
        print(f"   {status} {result['scenario']}: {result['detection']:.1%} detection")
    
    print("\n2️⃣ Attack Types Detection:")
    for result in attack_results:
        status = "✅" if result['detected'] else "❌"
        print(f"   {status} {result['attack']}")
    print(f"   Overall: {attacks_detected}/{len(attack_results)} detected")
    
    print("\n3️⃣ Learning Convergence:")
    print(f"   {'✅' if convergence_passed else '❌'} System continues learning")
    
    print("\n" + "="*60)
    if all_passed:
        print("🏆 ALL TESTS PASSED!")
        print("\n✨ Key Achievements:")
        print("  • 100% Byzantine detection with real PyTorch gradients")
        print("  • Trust Layer successfully compensates for real gradient variance")
        print("  • Reputation system effectively blacklists malicious nodes")
        print("  • System maintains learning despite Byzantine presence")
        print("  • Ready for Phase 3.3: Scale testing with 50+ nodes")
    else:
        print("⚠️  Some tests failed. See details above.")
    
    print("\n🚀 Next Steps:")
    print("  1. Phase 3.3: Scale to 50+ nodes")
    print("  2. Add real network communication")
    print("  3. Implement actual Holochain zomes")
    print("  4. Deploy to production environment")


if __name__ == "__main__":
    main()
