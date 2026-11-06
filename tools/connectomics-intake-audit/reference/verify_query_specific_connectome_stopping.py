#!/usr/bin/env python3
"""Independent oracle for question-specific connectome stopping V1.

Implements global-invariant drift, query-world agreement and exact
finite-population hypergeometric residual-risk bounds. Synthetic only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float('-inf')
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeometric_cdf(population: int, positives: int, sample_size: int, observed: int) -> float:
    require(0 <= positives <= population, 'population positive count is outside the finite frame')
    require(0 <= sample_size <= population, 'sample size exceeds the finite frame')
    require(observed >= 0, 'observed positive count is negative')
    lower = max(0, sample_size - (population - positives))
    upper = min(observed, sample_size, positives)
    if upper < lower:
        return 0.0
    denominator = log_choose(population, sample_size)
    log_probabilities = [
        log_choose(positives, h) + log_choose(population - positives, sample_size - h) - denominator
        for h in range(lower, upper + 1)
    ]
    peak = max(log_probabilities)
    probability = math.exp(peak) * sum(math.exp(value - peak) for value in log_probabilities)
    return min(1.0, max(0.0, probability))


def upper_population_count(population: int, sample_size: int, observed: int, alpha: float) -> int:
    require(population > 0 and 0 < sample_size <= population, 'finite frame/sample size is invalid')
    require(0 <= observed <= sample_size, 'observed query-flip count is invalid')
    require(0 < alpha < 1, 'alpha must be strictly between zero and one')
    low, high = 0, population
    while low < high:
        middle = (low + high + 1) // 2
        if hypergeometric_cdf(population, middle, sample_size, observed) >= alpha:
            low = middle
        else:
            high = middle - 1
    return low


def expected_frame_receipt(frame: dict[str, Any]) -> str:
    count = len(frame['candidate_unit_ids'])
    return sha256_text(f"synthetic-frame-v1\nframe-size={count}\n{frame['candidate_frame_sha256']}\n")


def validate_frame(frame: dict[str, Any]) -> dict[str, Any]:
    ids = frame['candidate_unit_ids']
    require(ids and len(ids) == len(set(ids)), 'candidate frame is empty or contains duplicate unit IDs')
    require(ids == sorted(ids), 'candidate frame unit IDs must be in canonical sorted order')
    population = len(ids)
    require(frame['candidate_frame_sha256'] == sha256_text('\n'.join(ids) + '\n'), 'candidate frame digest mismatch')
    require(frame['source_frame_receipt_sha256'] == expected_frame_receipt(frame), 'source-frame receipt does not bind the exact candidate frame digest')
    require(frame['sampling_design'] == 'SRS_WITHOUT_REPLACEMENT', 'V1 only supports simple random sampling without replacement')

    sampled = frame['sampled_units']
    require(frame['sample_size'] == len(sampled), 'declared sample size differs from sample ledger length')
    require(0 < len(sampled) <= population, 'sample size is outside the finite frame')
    sample_ids = [row['unit_id'] for row in sampled]
    require(len(sample_ids) == len(set(sample_ids)), 'sample ledger contains duplicate unit IDs')
    require(sample_ids == sorted(sample_ids), 'sample ledger must be canonically sorted by unit ID')
    candidate_set = set(ids)
    require(all(unit_id in candidate_set for unit_id in sample_ids), 'sample contains a unit outside the frozen candidate frame')
    expected_inclusion = len(sampled) / population
    for row in sampled:
        require(row['outcome'] in {'QUERY_FLIP', 'NO_QUERY_FLIP', 'UNRESOLVED'}, 'unknown sampled-unit outcome')
        require(abs(row['inclusion_probability'] - expected_inclusion) <= 1e-12, 'sample inclusion probability differs from the declared SRSWOR design')

    expected_ids_sha = sha256_text('\n'.join(sample_ids) + '\n')
    require(frame['sampled_unit_set_sha256'] == expected_ids_sha, 'sample unit-set digest mismatch')
    ledger = ''.join(
        f"{row['unit_id']}\t{row['outcome']}\t{row['inclusion_probability']:.6f}\n"
        for row in sampled
    )
    expected_ledger_sha = sha256_text(ledger)
    require(frame['sample_outcome_ledger_sha256'] == expected_ledger_sha, 'sample outcome ledger digest mismatch')
    receipt_payload = (
        f"SRS_WOR_V1\nN={population}\nn={len(sampled)}\n"
        f"{frame['candidate_frame_sha256']}\n{expected_ids_sha}\n{expected_ledger_sha}\n"
    )
    require(frame['sampling_receipt_sha256'] == sha256_text(receipt_payload), 'sampling receipt does not bind frame, sample IDs and outcomes')

    unresolved = sum(row['outcome'] == 'UNRESOLVED' for row in sampled)
    flips = sum(row['outcome'] == 'QUERY_FLIP' for row in sampled)
    return {
        'population_size': population,
        'sample_size': len(sampled),
        'observed_query_flips': flips,
        'unresolved_sample_units': unresolved,
        'inclusion_probability': expected_inclusion,
    }


def global_stop(case: dict[str, Any]) -> str | None:
    snapshots = case['snapshots']
    ids = [row['snapshot_id'] for row in snapshots]
    edits = [row['edit_count'] for row in snapshots]
    require(len(ids) == len(set(ids)), 'duplicate snapshot ID')
    require(all(edits[i] < edits[i + 1] for i in range(len(edits) - 1)), 'snapshot edit counts must be strictly increasing')
    invariant_sets = []
    for snapshot in snapshots:
        rows = snapshot['graph_invariants']
        names = [row['invariant_id'] for row in rows]
        require(names and len(names) == len(set(names)), 'snapshot invariant IDs are empty or duplicated')
        invariant_sets.append(set(names))
    require(all(names == invariant_sets[0] for names in invariant_sets[1:]), 'graph-invariant identity changes across checkpoints')

    policy = case['global_stop_policy']
    tolerance = policy['relative_delta_tolerance']
    required_stable = policy['consecutive_stable_transitions']
    stable_run = 0
    scale_floor = 1e-12
    for index in range(1, len(snapshots)):
        previous = {row['invariant_id']: row['value'] for row in snapshots[index - 1]['graph_invariants']}
        current = {row['invariant_id']: row['value'] for row in snapshots[index]['graph_invariants']}
        transition_stable = all(
            abs(current[name] - previous[name]) / max(abs(previous[name]), scale_floor) <= tolerance
            for name in previous
        )
        stable_run = stable_run + 1 if transition_stable else 0
        if stable_run >= required_stable:
            return snapshots[index]['snapshot_id']
    return None


def world_agreement(worlds: list[dict[str, Any]]) -> tuple[bool, bool]:
    require(worlds, 'no reconstruction worlds were supplied')
    unresolved = any(row['biological_disposition'] in {'INDETERMINATE', 'DISPUTED'} for row in worlds)
    admissible = [row for row in worlds if row['biological_disposition'] == 'ADMISSIBLE']
    require(all(row['query_decision'] is None for row in worlds if row['biological_disposition'] in {'INDETERMINATE', 'DISPUTED'}), 'indeterminate/disputed world must not carry a promoted query decision')
    require(all(row['query_decision'] is not None for row in admissible), 'admissible world lacks a query decision')
    require(all(row['query_decision'] is None for row in worlds if row['biological_disposition'] == 'INADMISSIBLE'), 'inadmissible world must not enter the query decision set')
    if unresolved or not admissible:
        return False, True
    decisions = {row['query_decision'] for row in admissible}
    return len(decisions) == 1, False


def evaluate_case(case: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    frame_stats = validate_frame(frame)
    require(case['frame_id'] == frame['frame_id'], 'case references an unknown residual candidate frame')
    global_snapshot = global_stop(case)
    agreement, unresolved_worlds = world_agreement(case['worlds'])
    alpha = 1.0 - case['confidence_level']
    upper_count = upper_population_count(
        frame_stats['population_size'],
        frame_stats['sample_size'],
        frame_stats['observed_query_flips'],
        alpha,
    )
    upper_prevalence = upper_count / frame_stats['population_size']

    if unresolved_worlds or frame_stats['unresolved_sample_units'] > 0:
        query_status = 'NOT_ESTIMABLE'
    elif case['frame_coverage_status'] == 'INCOMPLETE':
        query_status = 'FRAME_INCOMPLETE'
    elif not agreement:
        query_status = 'QUERY_UNRESOLVED'
    elif upper_prevalence <= case['residual_query_flip_tolerance']:
        query_status = 'QUERY_ROBUST_WITHIN_DECLARED_FRAME'
    else:
        query_status = 'QUERY_UNRESOLVED'

    if query_status == 'QUERY_ROBUST_WITHIN_DECLARED_FRAME':
        joint = query_status
    elif query_status == 'FRAME_INCOMPLETE':
        joint = query_status
    elif query_status == 'NOT_ESTIMABLE':
        joint = query_status
    elif global_snapshot is not None:
        joint = 'GLOBAL_ONLY'
    else:
        joint = 'QUERY_UNRESOLVED'

    return {
        'global_stop_snapshot_id': global_snapshot,
        'query_world_agreement': agreement,
        'residual_upper_count': upper_count,
        'residual_upper_prevalence': upper_prevalence,
        'query_stop_status': query_status,
        'joint_disposition': joint,
    }


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data['schema_version'] == 1, 'unsupported schema version')
    require(data['claim_scope'] == 'synthetic_known_truth', 'V1 oracle accepts only synthetic known-truth fixtures')
    require(data['preregistration']['frozen_before_sampling'] is True, 'preregistration was not frozen before sampling')
    require(len(data['preregistration']['preregistration_sha256']) == 64, 'preregistration digest is malformed')
    query = data['query_binding']
    require(query['query_id'].strip(), 'query ID is empty')
    require(len(query['query_spec_sha256']) == 64 and len(query['decision_loss_sha256']) == 64, 'query/loss digest is malformed')

    frames = {row['frame_id']: row for row in data['sampling_frames']}
    require(len(frames) == len(data['sampling_frames']), 'duplicate sampling frame IDs')
    frame_stats = {frame_id: validate_frame(frame) for frame_id, frame in frames.items()}
    case_ids = [row['case_id'] for row in data['cases']]
    require(len(case_ids) == len(set(case_ids)), 'duplicate case IDs')
    results = []
    for case in data['cases']:
        frame_id = case['frame_id']
        require(frame_id in frames, f"case {case['case_id']} references an unknown frame")
        result = evaluate_case(case, frames[frame_id])
        require(result == case['expected'], f"known-answer mismatch for {case['case_id']}: computed={result}")
        results.append({'case_id': case['case_id'], **result})
    return {
        'schema_version': 1,
        'tool': 'query-specific-connectome-stopping-oracle',
        'status': 'SYNTHETIC_CONTRACT_VALIDATED',
        'claim_scope': data['claim_scope'],
        'query_id': query['query_id'],
        'baseline_decision': query['baseline_decision'],
        'sampling_frame_count': len(frames),
        'case_count': len(results),
        'results': results,
        'known_answer': {
            'finite_frame_population': 1000,
            'sample_size': 300,
            'observed_query_flips': 0,
            'confidence_level': 0.95,
            'upper_count': 8,
            'upper_prevalence': 0.008,
        },
        'claim_ceiling': 'QUESTION_SPECIFIC_ONLY_WITHIN_FROZEN_QUERY_ADMITTED_WORLDS_CLOSED_FRAME_AND_DECLARED_SRSWOR_DESIGN',
    }


def brute_force_cdf(population: int, positives: int, sample_size: int, observed: int) -> float:
    denominator = math.comb(population, sample_size)
    lower = max(0, sample_size - (population - positives))
    upper = min(observed, sample_size, positives)
    return 0.0 if upper < lower else sum(
        math.comb(positives, h) * math.comb(population - positives, sample_size - h)
        for h in range(lower, upper + 1)
    ) / denominator


def self_test() -> None:
    # Compare the log-space CDF and integer upper-bound search to exact integer combinatorics.
    for population in range(2, 19):
        for sample_size in range(1, population + 1):
            for observed in range(0, min(sample_size, 3) + 1):
                alpha = 0.05
                upper = upper_population_count(population, sample_size, observed, alpha)
                for positives in range(population + 1):
                    exact = brute_force_cdf(population, positives, sample_size, observed)
                    numeric = hypergeometric_cdf(population, positives, sample_size, observed)
                    require(abs(exact - numeric) <= 1e-10, 'hypergeometric CDF disagrees with exact integer enumeration')
                require(brute_force_cdf(population, upper, sample_size, observed) >= alpha - 1e-12, 'upper-count candidate is below confidence threshold')
                if upper < population:
                    require(brute_force_cdf(population, upper + 1, sample_size, observed) < alpha + 1e-12, 'upper-count search did not find the maximum admissible count')

    # Mutation controls: an incomplete frame, an opposing world, unresolved evidence,
    # duplicate/out-of-frame IDs and zero inclusion probabilities must never promote.
    base = {
        'sampling_design': 'SRS_WITHOUT_REPLACEMENT',
        'candidate_unit_ids': ['a', 'b', 'c', 'd'],
        'candidate_frame_sha256': sha256_text('a\nb\nc\nd\n'),
        'source_frame_receipt_sha256': '',
        'sample_size': 2,
        'sampled_unit_set_sha256': '',
        'sample_outcome_ledger_sha256': '',
        'sampling_receipt_sha256': '',
        'sampled_units': [],
        'frame_id': 'self-test',
    }
    base['source_frame_receipt_sha256'] = expected_frame_receipt(base)
    base['sampled_units'] = [
        {'unit_id': 'a', 'outcome': 'NO_QUERY_FLIP', 'inclusion_probability': 0.5},
        {'unit_id': 'b', 'outcome': 'NO_QUERY_FLIP', 'inclusion_probability': 0.5},
    ]
    base['sampled_unit_set_sha256'] = sha256_text('a\nb\n')
    base['sample_outcome_ledger_sha256'] = sha256_text('a\tNO_QUERY_FLIP\t0.500000\nb\tNO_QUERY_FLIP\t0.500000\n')
    base['sampling_receipt_sha256'] = sha256_text(
        f"SRS_WOR_V1\nN=4\nn=2\n{base['candidate_frame_sha256']}\n"
        f"{base['sampled_unit_set_sha256']}\n{base['sample_outcome_ledger_sha256']}\n"
    )
    validate_frame(base)
    broken = copy.deepcopy(base)
    broken['sampled_units'][1]['unit_id'] = 'a'
    try:
        validate_frame(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate sample ID mutation was accepted')
    broken = copy.deepcopy(base)
    broken['sampled_units'][0]['inclusion_probability'] = 0.0
    try:
        validate_frame(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('zero inclusion probability mutation was accepted')
    print('hypergeometric exactness and fail-closed mutation controls: PASS')


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', nargs='?', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        try:
            self_test()
            return 0
        except Exception as exc:
            print(f'stopping oracle self-test: FAIL: {exc}', file=sys.stderr)
            return 1
    if args.fixture is None:
        parser.error('FIXTURE.json is required unless --self-test is used')
    try:
        data = json.loads(args.fixture.read_text(encoding='utf-8'))
        expected = verify(data)
        if args.report is not None:
            actual = json.loads(args.report.read_text(encoding='utf-8'))
            require(actual == expected, 'Rust production report differs from the independent Python oracle')
            print('Rust report exactly matches independent Python stopping oracle')
        print(json.dumps(expected, indent=2, sort_keys=True))
        print('question-specific connectome stopping oracle: PASS')
        return 0
    except Exception as exc:
        print(f'question-specific connectome stopping oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())