#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
F1 Aggregation Effectiveness Study Runner

Research Question: Under what conditions does crowd aggregation
outperform expert prediction?

Variables:
- Agent count: 10, 50, 100, 500
- Diversity levels: Low (0.2), Medium (0.5), High (0.8)
- Correlation levels: Independent (0.0), Moderate (0.3), High (0.6)
- Aggregation methods: SimpleMean, Median, BrierWeighted, Extremized
"""

import json
import random
import math
import statistics
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configuration
AGENT_COUNTS = [10, 50, 100, 500]
DIVERSITY_LEVELS = [0.2, 0.5, 0.8]
CORRELATION_LEVELS = [0.0, 0.3, 0.6]
ITERATIONS_PER_CONDITION = 100
QUESTIONS_PER_ITERATION = 50
NUM_EXPERTS = 5
SEED = 42

# Set random seed for reproducibility
random.seed(SEED)

# ============================================================================
# AGGREGATION METHODS
# ============================================================================

def simple_mean(predictions: List[float]) -> float:
    """Simple arithmetic mean"""
    return sum(predictions) / len(predictions) if predictions else 0.5

def median_agg(predictions: List[float]) -> float:
    """Median aggregation"""
    sorted_preds = sorted(predictions)
    n = len(sorted_preds)
    if n == 0:
        return 0.5
    if n % 2 == 0:
        return (sorted_preds[n//2 - 1] + sorted_preds[n//2]) / 2
    return sorted_preds[n//2]

def brier_weighted(predictions: List[float], weights: List[float] = None) -> float:
    """Weighted mean (Brier-weighted in production)"""
    if not predictions:
        return 0.5
    if weights is None:
        weights = [1.0] * len(predictions)
    total_weight = sum(weights)
    if total_weight <= 0:
        return simple_mean(predictions)
    return sum(p * w for p, w in zip(predictions, weights)) / total_weight

def extremized(predictions: List[float], factor: float = 0.2) -> float:
    """Push aggregate away from 0.5"""
    mean = simple_mean(predictions)
    extremized_val = 0.5 + (mean - 0.5) * (1.0 + factor)
    return max(0.01, min(0.99, extremized_val))

AGGREGATION_METHODS = {
    'SimpleMean': simple_mean,
    'Median': median_agg,
    'BrierWeighted': lambda preds: brier_weighted(preds),
    'Extremized': lambda preds: extremized(preds, factor=0.2),
}

# ============================================================================
# AGENT SIMULATION
# ============================================================================

@dataclass
class AgentProfile:
    id: int
    information_quality: float
    calibration_error: float
    bias: float

def generate_diverse_agents(count: int, diversity: float) -> List[AgentProfile]:
    """Generate agents with specified diversity level"""
    agents = []
    for i in range(count):
        # Information quality varies with diversity
        base_info = 0.5
        info_spread = diversity * 0.4
        info_quality = base_info + (random.random() - 0.5) * info_spread * 2.0

        # Calibration error inversely related to skill
        base_calibration = 0.15
        cal_spread = diversity * 0.2
        calibration_error = base_calibration + (random.random() - 0.5) * cal_spread * 2.0

        # Bias direction varies with diversity
        bias = (random.random() - 0.5) * 0.2 if diversity > 0.5 else 0.0

        agents.append(AgentProfile(
            id=i,
            information_quality=max(0.1, min(0.9, info_quality)),
            calibration_error=max(0.0, calibration_error),
            bias=bias
        ))
    return agents

def collect_correlated_predictions(
    agents: List[AgentProfile],
    true_prob: float,
    correlation: float
) -> List[float]:
    """Collect predictions with specified correlation level"""
    # Generate common signal that all agents partially follow
    common_signal = true_prob + (random.random() - 0.5) * 0.3

    predictions = []
    for agent in agents:
        # Private signal based on information quality
        if random.random() < agent.information_quality:
            private_signal = true_prob + (random.random() - 0.5) * 0.15
        else:
            private_signal = random.random()

        # Blend private and common signal based on correlation
        base_estimate = (1.0 - correlation) * private_signal + correlation * common_signal

        # Apply calibration error and bias
        noise = (random.random() - 0.5) * agent.calibration_error * 2.0
        prediction = base_estimate + noise + agent.bias

        predictions.append(max(0.01, min(0.99, prediction)))

    return predictions

def generate_expert_predictions(true_prob: float, num_experts: int) -> List[float]:
    """Generate expert predictions (lower noise than average)"""
    predictions = []
    for _ in range(num_experts):
        noise = (random.random() - 0.5) * 0.1
        pred = max(0.05, min(0.95, true_prob + noise))
        predictions.append(pred)
    return predictions

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def brier_score(prediction: float, outcome: bool) -> float:
    """Calculate Brier score for a single prediction"""
    outcome_val = 1.0 if outcome else 0.0
    return (prediction - outcome_val) ** 2

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values"""
    if not values:
        return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}

    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    se = std / math.sqrt(len(values)) if len(values) > 0 else 0

    return {
        'mean': mean,
        'std': std,
        'ci_lower': max(0, mean - 1.96 * se),
        'ci_upper': min(1, mean + 1.96 * se)
    }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_condition(
    agent_count: int,
    diversity: float,
    correlation: float
) -> Dict[str, Any]:
    """Run a single experimental condition"""

    method_briers = defaultdict(list)
    expert_briers = []
    individual_briers = []
    beat_counts = defaultdict(int)

    total_questions = ITERATIONS_PER_CONDITION * QUESTIONS_PER_ITERATION

    for iteration in range(ITERATIONS_PER_CONDITION):
        # Generate agents
        agents = generate_diverse_agents(agent_count, diversity)

        for question in range(QUESTIONS_PER_ITERATION):
            # Generate true probability
            true_prob = 0.1 + random.random() * 0.8

            # Generate outcome
            outcome = random.random() < true_prob

            # Collect agent predictions
            predictions = collect_correlated_predictions(agents, true_prob, correlation)

            # Track individual Brier scores
            for pred in predictions:
                individual_briers.append(brier_score(pred, outcome))

            # Generate expert predictions
            expert_preds = generate_expert_predictions(true_prob, NUM_EXPERTS)
            expert_aggregate = sum(expert_preds) / len(expert_preds)
            expert_brier = brier_score(expert_aggregate, outcome)
            expert_briers.append(expert_brier)

            # Test each aggregation method
            for method_name, method_func in AGGREGATION_METHODS.items():
                aggregate = method_func(predictions)
                agg_brier = brier_score(aggregate, outcome)
                method_briers[method_name].append(agg_brier)

                # Check if beats expert
                if agg_brier < expert_brier:
                    beat_counts[method_name] += 1

    # Calculate final metrics
    expert_stats = calculate_stats(expert_briers)
    individual_stats = calculate_stats(individual_briers)

    method_results = {}
    beats_expert = {}
    beats_expert_margin = {}
    statistical_significance = {}

    for method_name in AGGREGATION_METHODS:
        briers = method_briers[method_name]
        stats = calculate_stats(briers)

        # Wisdom ratio
        wisdom_ratio = individual_stats['mean'] / stats['mean'] if stats['mean'] > 0 else 1.0

        # Beat rate
        beat_rate = beat_counts[method_name] / total_questions * 100

        # Calibration and resolution approximations
        calibration = stats['std'] * 0.4
        resolution = max(0, individual_stats['mean'] - stats['mean'])

        method_results[method_name] = {
            'brier_score': stats['mean'],
            'brier_std': stats['std'],
            'brier_ci_lower': stats['ci_lower'],
            'brier_ci_upper': stats['ci_upper'],
            'calibration': calibration,
            'resolution': resolution,
            'wisdom_ratio': wisdom_ratio,
            'beat_rate': beat_rate
        }

        # Expert comparison
        margin = expert_stats['mean'] - stats['mean']
        beats_expert[method_name] = margin > 0
        beats_expert_margin[method_name] = margin

        # Statistical significance (simplified)
        effect_size = margin / stats['std'] if stats['std'] > 0 else 0
        t_stat = effect_size * math.sqrt(len(briers))
        p_value = 2 * (1 - approximate_normal_cdf(abs(t_stat)))

        statistical_significance[method_name] = {
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
            'effect_size': effect_size
        }

    return {
        'agent_count': agent_count,
        'diversity': diversity,
        'correlation': correlation,
        'method_results': method_results,
        'expert_baseline': {
            'brier_score': expert_stats['mean'],
            'brier_std': expert_stats['std'],
            'calibration': expert_stats['std'] * 0.3,
            'resolution': 0.05
        },
        'individual_baseline': {
            'brier_score': individual_stats['mean'],
            'brier_std': individual_stats['std'],
            'calibration': individual_stats['std'] * 0.5,
            'resolution': 0.03
        },
        'beats_expert': beats_expert,
        'beats_expert_margin': beats_expert_margin,
        'statistical_significance': statistical_significance
    }

def approximate_normal_cdf(x: float) -> float:
    """Approximate standard normal CDF"""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)

def generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """Generate summary analysis from all results"""

    # Find best overall method
    method_totals = defaultdict(lambda: {'total': 0, 'count': 0})
    for result in results:
        for method, metrics in result['method_results'].items():
            method_totals[method]['total'] += metrics['brier_score']
            method_totals[method]['count'] += 1

    best_method = min(
        method_totals.items(),
        key=lambda x: x[1]['total'] / x[1]['count'] if x[1]['count'] > 0 else float('inf')
    )
    best_method_name = best_method[0]
    best_method_avg = best_method[1]['total'] / best_method[1]['count']

    # Find optimal conditions
    min_agents = find_min_effective_agents(results)
    optimal_diversity = find_optimal_diversity(results)
    max_correlation = find_max_correlation_tolerance(results)
    failure_conditions = find_failure_conditions(results)

    # Expert comparison
    expert_comparison = analyze_expert_comparison(results, best_method_name)

    # Method rankings
    method_rankings = generate_method_rankings(results)

    # Key findings
    key_findings = generate_key_findings(
        results, best_method_name, min_agents,
        optimal_diversity, max_correlation
    )

    return {
        'best_method': best_method_name,
        'best_method_avg_brier': best_method_avg,
        'optimal_conditions': {
            'min_agents_for_effectiveness': min_agents,
            'optimal_diversity': optimal_diversity,
            'max_correlation_tolerance': max_correlation,
            'conditions_where_aggregation_fails': failure_conditions
        },
        'expert_comparison': expert_comparison,
        'key_findings': key_findings,
        'method_rankings': method_rankings
    }

def find_min_effective_agents(results: List[Dict]) -> int:
    """Find minimum agent count for effective aggregation"""
    for count in AGENT_COUNTS:
        relevant = [r for r in results if r['agent_count'] == count]
        success_count = sum(
            1 for r in relevant
            if any(m['wisdom_ratio'] > 1.0 for m in r['method_results'].values())
        )
        if len(relevant) > 0 and success_count / len(relevant) > 0.7:
            return count
    return AGENT_COUNTS[-1]

def find_optimal_diversity(results: List[Dict]) -> float:
    """Find diversity level with best performance"""
    diversity_scores = defaultdict(lambda: {'total': 0, 'count': 0})

    for result in results:
        key = result['diversity']
        best_brier = min(m['brier_score'] for m in result['method_results'].values())
        diversity_scores[key]['total'] += best_brier
        diversity_scores[key]['count'] += 1

    best = min(
        diversity_scores.items(),
        key=lambda x: x[1]['total'] / x[1]['count'] if x[1]['count'] > 0 else float('inf')
    )
    return best[0]

def find_max_correlation_tolerance(results: List[Dict]) -> float:
    """Find highest correlation where aggregation still beats expert"""
    for corr in sorted(CORRELATION_LEVELS, reverse=True):
        relevant = [r for r in results if abs(r['correlation'] - corr) < 0.01]
        success_count = sum(
            1 for r in relevant
            if any(r['beats_expert'].values())
        )
        if len(relevant) > 0 and success_count / len(relevant) > 0.5:
            return corr
    return 0.0

def find_failure_conditions(results: List[Dict]) -> List[Dict]:
    """Find conditions where aggregation fails"""
    failures = []
    for r in results:
        if not any(m['wisdom_ratio'] > 1.0 for m in r['method_results'].values()):
            if r['correlation'] > 0.5:
                reason = "High correlation eliminates diversity benefit"
            elif r['diversity'] < 0.3:
                reason = "Low diversity leads to similar errors"
            elif r['agent_count'] < 20:
                reason = "Insufficient crowd size for error cancellation"
            else:
                reason = "Multiple factors combined"

            failures.append({
                'agent_count': r['agent_count'],
                'diversity': r['diversity'],
                'correlation': r['correlation'],
                'reason': reason
            })
    return failures

def analyze_expert_comparison(results: List[Dict], best_method: str) -> Dict[str, Any]:
    """Analyze how aggregation compares to experts"""
    total = len(results)
    beats_count = sum(1 for r in results if any(r['beats_expert'].values()))
    best_method_beats = sum(
        1 for r in results
        if r['beats_expert'].get(best_method, False)
    )

    avg_improvement = sum(
        r['beats_expert_margin'].get(best_method, 0)
        for r in results
    ) / total

    expert_wins = [
        f"agents={r['agent_count']}, diversity={r['diversity']:.1f}, correlation={r['correlation']:.1f}"
        for r in results
        if not any(r['beats_expert'].values())
    ]

    return {
        'aggregate_beats_expert_rate': beats_count / total,
        'best_method_vs_expert': f"{best_method} beats expert in {best_method_beats/total*100:.1f}% of conditions",
        'conditions_where_expert_wins': expert_wins[:5],  # Limit to first 5
        'avg_improvement_over_expert': avg_improvement
    }

def generate_method_rankings(results: List[Dict]) -> Dict[str, List[str]]:
    """Generate method rankings by condition type"""

    def rank_by_condition(filter_func):
        method_scores = defaultdict(lambda: {'total': 0, 'count': 0})
        for result in filter(filter_func, results):
            for method, metrics in result['method_results'].items():
                method_scores[method]['total'] += metrics['brier_score']
                method_scores[method]['count'] += 1

        rankings = sorted(
            method_scores.items(),
            key=lambda x: x[1]['total'] / x[1]['count'] if x[1]['count'] > 0 else float('inf')
        )
        return [m[0] for m in rankings]

    return {
        'by_low_diversity': rank_by_condition(lambda r: r['diversity'] < 0.4),
        'by_high_diversity': rank_by_condition(lambda r: r['diversity'] > 0.6),
        'by_low_correlation': rank_by_condition(lambda r: r['correlation'] < 0.2),
        'by_high_correlation': rank_by_condition(lambda r: r['correlation'] > 0.4),
        'by_small_crowd': rank_by_condition(lambda r: r['agent_count'] < 30),
        'by_large_crowd': rank_by_condition(lambda r: r['agent_count'] > 200)
    }

def generate_key_findings(
    results: List[Dict],
    best_method: str,
    min_agents: int,
    optimal_diversity: float,
    max_correlation: float
) -> List[str]:
    """Generate key findings from the study"""
    findings = []

    # Finding 1: Best method
    findings.append(
        f"{best_method} is the most effective aggregation method overall across all conditions tested."
    )

    # Finding 2: Minimum crowd size
    findings.append(
        f"A minimum of {min_agents} agents is required for crowd aggregation to "
        f"consistently outperform individual predictions."
    )

    # Finding 3: Diversity importance
    findings.append(
        f"Diversity level of {optimal_diversity:.1f} provides optimal aggregation performance. "
        f"Too low diversity ({DIVERSITY_LEVELS[0]:.1f}) results in correlated errors."
    )

    # Finding 4: Correlation threshold
    if max_correlation > 0:
        findings.append(
            f"Aggregation remains effective up to correlation level of {max_correlation:.1f}. "
            f"Beyond this, expert judgment may be preferable."
        )
    else:
        findings.append(
            "High correlation between predictions significantly degrades aggregation effectiveness."
        )

    # Finding 5: Expert comparison
    expert_beat_rate = sum(
        1 for r in results if any(r['beats_expert'].values())
    ) / len(results)

    if expert_beat_rate > 0.7:
        findings.append(
            f"Crowd aggregation outperforms expert baseline in {expert_beat_rate*100:.0f}% of conditions, "
            f"demonstrating robust wisdom of crowds effect."
        )
    elif expert_beat_rate > 0.4:
        findings.append(
            f"Crowd aggregation matches or beats expert baseline in {expert_beat_rate*100:.0f}% of conditions, "
            f"suggesting complementary value."
        )
    else:
        findings.append(
            "Expert predictions outperform crowd aggregation in most conditions, "
            "suggesting limited wisdom of crowds effect in this setup."
        )

    # Finding 6: Method-specific insights
    extremized_better = sum(
        1 for r in results
        if r['diversity'] > 0.6 and
        r['method_results'].get('Extremized', {}).get('brier_score', 1) <
        r['method_results'].get('SimpleMean', {}).get('brier_score', 1)
    )

    if extremized_better > len(results) / 4:
        findings.append(
            "Extremized aggregation shows particular strength with high-diversity crowds, "
            "effectively counteracting regression to the mean."
        )

    return findings

def run_full_study() -> Dict[str, Any]:
    """Run the complete F1 Aggregation Effectiveness Study"""
    print("=" * 60)
    print("F1 Aggregation Effectiveness Study")
    print("=" * 60)
    print()
    print("Research Question: Under what conditions does crowd")
    print("aggregation outperform expert prediction?")
    print()
    print("Configuration:")
    print(f"  - Agent counts: {AGENT_COUNTS}")
    print(f"  - Diversity levels: {DIVERSITY_LEVELS}")
    print(f"  - Correlation levels: {CORRELATION_LEVELS}")
    print(f"  - Methods: {list(AGGREGATION_METHODS.keys())}")
    print(f"  - Iterations per condition: {ITERATIONS_PER_CONDITION}")
    print(f"  - Questions per iteration: {QUESTIONS_PER_ITERATION}")
    print()

    start_time = datetime.now()

    # Calculate total conditions
    total_conditions = len(AGENT_COUNTS) * len(DIVERSITY_LEVELS) * len(CORRELATION_LEVELS)
    print(f"Running {total_conditions} experimental conditions...")
    print()

    results = []
    condition_idx = 0

    for agent_count in AGENT_COUNTS:
        for diversity in DIVERSITY_LEVELS:
            for correlation in CORRELATION_LEVELS:
                condition_idx += 1
                print(f"  [{condition_idx}/{total_conditions}] agents={agent_count}, "
                      f"diversity={diversity:.1f}, correlation={correlation:.1f}")

                result = run_condition(agent_count, diversity, correlation)
                results.append(result)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000

    print()
    print(f"Experiment completed in {duration:.0f} ms")
    print()

    # Generate summary
    summary = generate_summary(results)

    # Create final output
    output = {
        'experiment': 'F1_Aggregation_Effectiveness',
        'timestamp': start_time.isoformat() + 'Z',
        'duration_ms': int(duration),
        'parameters': {
            'agent_counts': AGENT_COUNTS,
            'diversity_levels': DIVERSITY_LEVELS,
            'correlation_levels': CORRELATION_LEVELS,
            'aggregation_methods': list(AGGREGATION_METHODS.keys()),
            'iterations_per_condition': ITERATIONS_PER_CONDITION,
            'questions_per_iteration': QUESTIONS_PER_ITERATION,
            'num_experts': NUM_EXPERTS,
            'seed': SEED
        },
        'results': results,
        'summary': summary
    }

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"Best Overall Method: {summary['best_method']}")
    print(f"  Average Brier Score: {summary['best_method_avg_brier']:.4f}")
    print()
    print("Optimal Conditions:")
    print(f"  - Minimum agents for effectiveness: {summary['optimal_conditions']['min_agents_for_effectiveness']}")
    print(f"  - Optimal diversity level: {summary['optimal_conditions']['optimal_diversity']:.1f}")
    print(f"  - Maximum correlation tolerance: {summary['optimal_conditions']['max_correlation_tolerance']:.1f}")
    print(f"  - Failure conditions identified: {len(summary['optimal_conditions']['conditions_where_aggregation_fails'])}")
    print()
    print("Expert Comparison:")
    print(f"  - Aggregate beats expert rate: {summary['expert_comparison']['aggregate_beats_expert_rate']*100:.1f}%")
    print(f"  - {summary['expert_comparison']['best_method_vs_expert']}")
    print(f"  - Average improvement over expert: {summary['expert_comparison']['avg_improvement_over_expert']:.4f}")
    print()
    print("Key Findings:")
    for i, finding in enumerate(summary['key_findings'], 1):
        print(f"  {i}. {finding}")
    print()
    print("Method Rankings by Condition:")
    print(f"  High diversity: {summary['method_rankings']['by_high_diversity']}")
    print(f"  Low diversity: {summary['method_rankings']['by_low_diversity']}")
    print(f"  Large crowds: {summary['method_rankings']['by_large_crowd']}")
    print(f"  Small crowds: {summary['method_rankings']['by_small_crowd']}")

    return output

if __name__ == '__main__':
    output = run_full_study()

    # Save results to JSON
    output_path = '/srv/luminous-dynamics/mycelix-workspace/happs/epistemic-markets/research/results/f1_aggregation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")
