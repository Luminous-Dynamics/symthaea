# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL CLI - Main Entry Point

Command-line interface for federated learning experiments.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mycelix-fl",
        description="MycelixFL - Universal Federated Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mycelix-fl run --nodes 10 --rounds 5
  mycelix-fl benchmark --nodes 50 --gradient-size 100000
  mycelix-fl attack-test --scenario cartel --cartel-size 5
  mycelix-fl info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run FL experiment")
    run_parser.add_argument("--nodes", type=int, default=10, help="Number of nodes")
    run_parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    run_parser.add_argument("--gradient-size", type=int, default=10000, help="Gradient vector size")
    run_parser.add_argument("--byzantine-ratio", type=float, default=0.0, help="Ratio of Byzantine nodes")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--nodes", type=int, default=50, help="Number of nodes")
    bench_parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    bench_parser.add_argument("--gradient-size", type=int, default=100000, help="Gradient vector size")
    bench_parser.add_argument("--warmup", type=int, default=2, help="Warmup rounds")
    bench_parser.add_argument("--output", type=str, help="Output file for results")

    # Attack test command
    attack_parser = subparsers.add_parser("attack-test", help="Test Byzantine attack scenarios")
    attack_parser.add_argument(
        "--scenario",
        type=str,
        choices=["scaling", "adaptive", "cartel", "mixed"],
        default="mixed",
        help="Attack scenario",
    )
    attack_parser.add_argument("--num-attackers", type=int, default=5, help="Number of attackers")
    attack_parser.add_argument("--cartel-size", type=int, default=5, help="Cartel size (for cartel scenario)")
    attack_parser.add_argument("--intensity", type=float, default=2.0, help="Attack intensity")
    attack_parser.add_argument("--honest-nodes", type=int, default=10, help="Number of honest nodes")
    attack_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Version
    parser.add_argument("--version", action="store_true", help="Show version")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Run FL experiment."""
    from mycelix_fl import MycelixFL, FLConfig, setup_logging

    if args.verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")

    np.random.seed(args.seed)

    print(f"=== MycelixFL Experiment ===")
    print(f"Nodes: {args.nodes}")
    print(f"Rounds: {args.rounds}")
    print(f"Gradient size: {args.gradient_size:,}")
    print(f"Byzantine ratio: {args.byzantine_ratio:.1%}")
    print()

    config = FLConfig(
        num_rounds=args.rounds,
        byzantine_threshold=0.45,
        use_detection=True,
        use_healing=True,
    )
    fl = MycelixFL(config=config)

    results = []
    total_time = 0.0

    for round_num in range(1, args.rounds + 1):
        # Generate gradients
        num_byzantine = int(args.nodes * args.byzantine_ratio)
        num_honest = args.nodes - num_byzantine

        gradients = {}
        for i in range(num_honest):
            gradients[f"honest_{i}"] = np.random.randn(args.gradient_size).astype(np.float32)
        for i in range(num_byzantine):
            # Byzantine nodes send scaled gradients
            gradients[f"byzantine_{i}"] = np.random.randn(args.gradient_size).astype(np.float32) * 100

        start = time.perf_counter()
        result = fl.execute_round(gradients, round_num=round_num)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        results.append({
            "round": round_num,
            "participating": len(result.participating_nodes),
            "byzantine_detected": len(result.byzantine_nodes),
            "healed": len(result.healed_nodes),
            "time_ms": elapsed,
        })

        print(f"Round {round_num}/{args.rounds}: "
              f"{len(result.participating_nodes)} nodes, "
              f"{len(result.byzantine_nodes)} byzantine, "
              f"{elapsed:.1f}ms")

    print()
    print(f"=== Summary ===")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Avg per round: {total_time / args.rounds:.1f}ms")

    if args.output:
        output_data = {
            "config": {
                "nodes": args.nodes,
                "rounds": args.rounds,
                "gradient_size": args.gradient_size,
                "byzantine_ratio": args.byzantine_ratio,
            },
            "results": results,
            "summary": {
                "total_time_ms": total_time,
                "avg_time_ms": total_time / args.rounds,
            },
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to: {args.output}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run performance benchmarks."""
    from mycelix_fl import MycelixFL, FLConfig, has_rust_backend

    print(f"=== MycelixFL Benchmark ===")
    print(f"Rust backend: {has_rust_backend()}")
    print(f"Nodes: {args.nodes}")
    print(f"Rounds: {args.rounds}")
    print(f"Gradient size: {args.gradient_size:,}")
    print(f"Warmup rounds: {args.warmup}")
    print()

    config = FLConfig(
        use_detection=True,
        use_healing=True,
    )
    fl = MycelixFL(config=config)

    # Generate gradients once
    np.random.seed(42)
    gradients = {
        f"node_{i}": np.random.randn(args.gradient_size).astype(np.float32)
        for i in range(args.nodes)
    }

    # Warmup
    print(f"Warming up ({args.warmup} rounds)...")
    for i in range(args.warmup):
        fl.execute_round(gradients, round_num=i + 1)

    # Benchmark
    print(f"Benchmarking ({args.rounds} rounds)...")
    times = []
    for i in range(args.rounds):
        start = time.perf_counter()
        fl.execute_round(gradients, round_num=i + 1)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Round {i + 1}: {elapsed:.2f}ms")

    times = np.array(times)
    print()
    print(f"=== Results ===")
    print(f"Mean:   {times.mean():.2f}ms")
    print(f"Std:    {times.std():.2f}ms")
    print(f"Min:    {times.min():.2f}ms")
    print(f"Max:    {times.max():.2f}ms")
    print(f"P50:    {np.percentile(times, 50):.2f}ms")
    print(f"P95:    {np.percentile(times, 95):.2f}ms")
    print(f"P99:    {np.percentile(times, 99):.2f}ms")

    throughput = args.nodes / (times.mean() / 1000)
    print(f"Throughput: {throughput:.1f} nodes/second")

    if args.output:
        output_data = {
            "config": {
                "nodes": args.nodes,
                "rounds": args.rounds,
                "gradient_size": args.gradient_size,
                "rust_backend": has_rust_backend(),
            },
            "results": {
                "mean_ms": float(times.mean()),
                "std_ms": float(times.std()),
                "min_ms": float(times.min()),
                "max_ms": float(times.max()),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "throughput_nodes_per_sec": throughput,
            },
            "raw_times_ms": times.tolist(),
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_attack_test(args: argparse.Namespace) -> int:
    """Run attack scenario testing."""
    from mycelix_fl import MycelixFL, FLConfig
    from mycelix_fl.attacks import (
        AttackOrchestrator,
        create_gradient_scaling_scenario,
        create_adaptive_attack_scenario,
        create_cartel_scenario,
        create_mixed_attack_scenario,
    )

    print(f"=== Byzantine Attack Test ===")
    print(f"Scenario: {args.scenario}")
    print(f"Honest nodes: {args.honest_nodes}")
    print(f"Attackers: {args.num_attackers}")
    print(f"Intensity: {args.intensity}")
    print()

    np.random.seed(args.seed)

    # Create attack scenario
    if args.scenario == "scaling":
        attacks = create_gradient_scaling_scenario(
            num_attackers=args.num_attackers,
            intensity=args.intensity * 10,
            seed=args.seed,
        )
    elif args.scenario == "adaptive":
        attacks = create_adaptive_attack_scenario(
            num_attackers=args.num_attackers,
            initial_intensity=args.intensity,
            seed=args.seed,
        )
    elif args.scenario == "cartel":
        attacks = create_cartel_scenario(
            cartel_size=args.cartel_size,
            intensity=args.intensity,
            seed=args.seed,
        )
    else:  # mixed
        attacks = create_mixed_attack_scenario(
            total_attackers=args.num_attackers,
            seed=args.seed,
        )

    # Register attackers
    orchestrator = AttackOrchestrator(random_seed=args.seed)
    for node_id, attack in attacks:
        orchestrator.register_attacker(node_id, attack)

    # Create honest gradients
    gradients = {
        f"honest_{i}": np.random.randn(10000).astype(np.float32)
        for i in range(args.honest_nodes)
    }

    # Add attacker gradients
    for node_id, _ in attacks:
        gradients[node_id] = np.random.randn(10000).astype(np.float32)

    # Apply attacks
    modified_grads, attacker_ids = orchestrator.apply_attacks(gradients)

    # Run FL
    config = FLConfig(
        byzantine_threshold=0.45,
        use_detection=True,
        use_healing=True,
    )
    fl = MycelixFL(config=config)

    result = fl.execute_round(modified_grads, round_num=1)

    # Analyze results
    detected = result.byzantine_nodes
    true_positives = detected & attacker_ids
    false_positives = detected - attacker_ids
    false_negatives = attacker_ids - detected

    print(f"=== Results ===")
    print(f"Total nodes: {len(gradients)}")
    print(f"Actual attackers: {len(attacker_ids)}")
    print(f"Detected Byzantine: {len(detected)}")
    print()
    print(f"True Positives:  {len(true_positives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print()

    if len(attacker_ids) > 0:
        precision = len(true_positives) / len(detected) if detected else 0
        recall = len(true_positives) / len(attacker_ids)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.1%}")
        print(f"Recall:    {recall:.1%}")
        print(f"F1 Score:  {f1:.1%}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information."""
    from mycelix_fl import __version__, has_rust_backend, get_backend_info

    print(f"=== MycelixFL Info ===")
    print(f"Version: {__version__}")
    print(f"Rust backend: {has_rust_backend()}")
    print()

    info = get_backend_info()
    print("Backend info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return 0


def app(args: Optional[list] = None) -> int:
    """Main CLI application."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.version:
        from mycelix_fl import __version__
        print(f"mycelix-fl {__version__}")
        return 0

    if parsed.command is None:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "attack-test": cmd_attack_test,
        "info": cmd_info,
    }

    return commands[parsed.command](parsed)


def main() -> None:
    """Entry point."""
    sys.exit(app())


if __name__ == "__main__":
    main()
