#!/usr/bin/env python3
"""Independent oracle for set-valued connectome hypothesis coverage.

Scores finite synthetic hypothesis sets using canonical biological classes.
Biological-frame estimates remain the responsibility of the declared sampler
and adjudication design; this tool does not infer a biological ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts if count > 0)


def validate_universe(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], set[str], dict[str, float]]:
    query = data['query']
    require(bool(query['query_id'].strip()), 'query ID must be non-empty')
    require(len(query['query_spec_sha256']) == 64 and all(c in '0123456789abcdef' for c in query['query_spec_sha256']), 'query spec digest malformed')
    by_id: dict[str, dict[str, Any]] = {}
    class_meta: dict[str, dict[str, Any]] = {}
    class_status: dict[str, str] = {}
    for candidate in data['hypothesis_universe']:
        candidate_id = candidate['candidate_id']
        class_id = candidate['equivalence_class_id']
        require(candidate_id.strip() and class_id.strip(), 'candidate and equivalence-class identities must be non-empty')
        require(candidate_id not in by_id, f'duplicate candidate identity: {candidate_id}')
        by_id[candidate_id] = candidate
        previous_status = class_status.setdefault(class_id, candidate['adjudication_status'])
        require(previous_status == candidate['adjudication_status'], f'equivalence class {class_id} has inconsistent adjudication status')
        if candidate['adjudication_status'] == 'ADMISSIBLE':
            require(isinstance(candidate['query_decision'], bool), f'admissible candidate {candidate_id} needs a query decision')
            require(candidate['impact_weight'] >= 0.0, 'impact weight cannot be negative')
            critical = candidate['query_decision'] != query['baseline_decision'] or candidate['invalidates_policy']
            meta = {
                'query_decision': candidate['query_decision'],
                'invalidates_policy': candidate['invalidates_policy'],
                'query_critical': critical,
                'impact_weight': candidate['impact_weight'],
            }
            if not critical:
                require(candidate['impact_weight'] == 0.0, f'non-critical equivalence class {class_id} cannot carry impact weight')
            else:
                require(candidate['impact_weight'] > 0.0, f'query-critical equivalence class {class_id} needs positive impact weight')
            if class_id in class_meta:
                require(class_meta[class_id] == meta, f'equivalence class {class_id} has inconsistent adjudication/query semantics')
            else:
                class_meta[class_id] = meta
        else:
            require(candidate['impact_weight'] == 0.0, f'non-admissible candidate {candidate_id} cannot carry impact weight')
    admissible_classes = set(class_meta)
    critical_classes = {class_id for class_id, meta in class_meta.items() if meta['query_critical']}
    impact_by_class = {class_id: meta['impact_weight'] for class_id, meta in class_meta.items() if meta['query_critical']}
    require(all(weight > 0.0 for weight in impact_by_class.values()), 'every query-critical equivalence class needs positive frozen impact weight')
    require(critical_classes == set(impact_by_class), 'every query-critical class must have an impact weight')
    require(admissible_classes, 'exact synthetic universe must include at least one admissible equivalence class')
    return by_id, class_meta, admissible_classes, critical_classes, impact_by_class


def validate_channels(channels: list[dict[str, Any]], max_budget: int) -> None:
    channel_ids = [channel['channel_id'] for channel in channels]
    require(all(channel_id.strip() for channel_id in channel_ids), 'channel ID must be non-empty')
    require(len(channel_ids) == len(set(channel_ids)), 'duplicate channel IDs')
    for channel in channels:
        require(bool(channel['ancestry_id'].strip()), f"{channel['channel_id']}: ancestry ID must be non-empty")
        proposals = channel['proposals']
        ranks = [proposal['proposal_rank'] for proposal in proposals]
        require(len(ranks) == len(set(ranks)), f"{channel['channel_id']}: duplicate proposal ranks")
        require(sorted(ranks) == list(range(1, len(ranks) + 1)), f"{channel['channel_id']}: proposal ranks must be contiguous from 1")
        require(len(proposals) >= max_budget, f"{channel['channel_id']}: proposal list shorter than max_budget")


def score_proposals(
    proposals: list[dict[str, Any]],
    budget: int,
    by_id: dict[str, dict[str, Any]],
    admissible_classes: set[str],
    critical_classes: set[str],
    impact_by_class: dict[str, float],
) -> dict[str, Any]:
    selected = sorted(proposals, key=lambda row: row['proposal_rank'])[:budget]
    admissible_rows: list[dict[str, Any]] = []
    inadmissible_count = 0
    indeterminate_count = 0
    unknown_count = 0
    for proposal in selected:
        candidate = by_id.get(proposal['candidate_id'])
        if candidate is None:
            unknown_count += 1
        elif candidate['adjudication_status'] == 'ADMISSIBLE':
            admissible_rows.append(candidate)
        elif candidate['adjudication_status'] == 'INADMISSIBLE':
            inadmissible_count += 1
        else:
            indeterminate_count += 1

    class_counts: dict[str, int] = {}
    for candidate in admissible_rows:
        class_id = candidate['equivalence_class_id']
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    recovered_classes = set(class_counts)
    recovered_critical = recovered_classes & critical_classes
    admissible_count = len(admissible_rows)
    recovered_count = len(recovered_classes)
    critical_weight_total = sum(impact_by_class.values())
    critical_weight_recovered = sum(impact_by_class[class_id] for class_id in recovered_critical)

    validity = admissible_count / budget if budget else 0.0
    uniqueness = recovered_count / admissible_count if admissible_count else 0.0
    recovery = recovered_count / len(admissible_classes) if admissible_classes else 0.0
    critical_recovery = len(recovered_critical) / len(critical_classes) if critical_classes else 1.0
    impact_recovery = critical_weight_recovered / critical_weight_total if critical_weight_total else 1.0
    mode_share = max(class_counts.values(), default=0) / admissible_count if admissible_count else 0.0

    return {
        'proposal_count': budget,
        'admissible_proposal_count': admissible_count,
        'validity': validity,
        'uniqueness': uniqueness,
        'recovered_admissible_class_count': recovered_count,
        'admissible_class_count': len(admissible_classes),
        'recovery': recovery,
        'recovered_query_critical_class_count': len(recovered_critical),
        'query_critical_class_count': len(critical_classes),
        'query_critical_recovery': critical_recovery,
        'impact_weighted_query_critical_recovery': impact_recovery,
        'inadmissible_proposal_count': inadmissible_count,
        'indeterminate_proposal_count': indeterminate_count,
        'unknown_proposal_count': unknown_count,
        'max_admissible_class_share': mode_share,
        'admissible_class_entropy_bits': entropy(list(class_counts.values())),
        'recovered_equivalence_class_ids': sorted(recovered_classes),
        'recovered_query_critical_class_ids': sorted(recovered_critical),
    }


def compare_expected(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, expected_value in expected.items():
        require(key in actual, f'{label}: missing computed metric {key}')
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            require(abs(float(actual_value) - float(expected_value)) <= 1e-9, f'{label}: {key} expected {expected_value}, got {actual_value}')
        else:
            require(actual_value == expected_value, f'{label}: {key} expected {expected_value!r}, got {actual_value!r}')


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    require(data['schema_version'] == 1, 'unsupported schema version')
    require(data['claim_scope'] == 'synthetic_known_truth', 'V1 oracle only scores exact synthetic known-truth sets')
    by_id, class_meta, admissible_classes, critical_classes, impact_by_class = validate_universe(data)
    channels = data['channels']
    max_budget = data['oracle_expectations']['max_budget']
    require(isinstance(max_budget, int) and max_budget > 0, 'max_budget must be a positive integer')
    validate_channels(channels, max_budget)

    per_channel: dict[str, Any] = {}
    final_per_channel: dict[str, Any] = {}
    for channel in channels:
        channel_id = channel['channel_id']
        curve = []
        for budget in range(1, max_budget + 1):
            curve.append(score_proposals(channel['proposals'], budget, by_id, admissible_classes, critical_classes, impact_by_class))
        per_channel[channel_id] = curve
        final_per_channel[channel_id] = curve[-1]

    pooled_proposals = []
    for channel in channels:
        for proposal in channel['proposals'][:max_budget]:
            pooled_proposals.append({
                'proposal_rank': len(pooled_proposals) + 1,
                'candidate_id': proposal['candidate_id'],
            })
    pooled = score_proposals(pooled_proposals, len(pooled_proposals), by_id, admissible_classes, critical_classes, impact_by_class)

    expected = data['oracle_expectations']
    require(set(final_per_channel) == set(expected['final_per_channel']), 'oracle expectation channel IDs do not match input channel IDs')
    for channel_id, metrics in final_per_channel.items():
        compare_expected(metrics, expected['final_per_channel'][channel_id], f'channel {channel_id}')
    compare_expected(pooled, expected['final_pooled_union'], 'pooled union')

    mode_channel = final_per_channel.get('mode-collapsed')
    diverse_channel = final_per_channel.get('query-diverse')
    if mode_channel is not None and diverse_channel is not None:
        require(mode_channel['validity'] == diverse_channel['validity'], 'known-answer fixture should hold validity equal across mode-collapsed and query-diverse channels')
        require(mode_channel['recovery'] < diverse_channel['recovery'], 'mode-collapsed channel must have lower admissible recovery')
        require(mode_channel['query_critical_recovery'] < diverse_channel['query_critical_recovery'], 'mode-collapsed channel must miss query-critical alternatives')
        require(mode_channel['uniqueness'] < diverse_channel['uniqueness'], 'mode-collapsed channel must have lower uniqueness')

    return {
        'schema_version': 1,
        'tool': 'verify-set-valued-connectome-hypothesis-coverage',
        'claim_scope': data['claim_scope'],
        'run_id': data['run_id'],
        'query_id': data['query']['query_id'],
        'hypothesis_identity_count': len(by_id),
        'admissible_equivalence_class_count': len(admissible_classes),
        'query_critical_equivalence_class_count': len(critical_classes),
        'channel_count': len(channels),
        'channel_ancestry_ids': {channel['channel_id']: channel['ancestry_id'] for channel in channels},
        'channel_independence_claim': 'NOT_ESTABLISHED_PROVENANCE_ADJUDICATION_REQUIRED',
        'proposal_budget_max': max_budget,
        'per_channel_curves': per_channel,
        'final_per_channel': final_per_channel,
        'pooled_union': pooled,
        'oracle_agreement': 'EXACT',
        'biological_recovery_claim': 'NOT_ESTIMATED_SYNTHETIC_ENUMERATION_ONLY',
        'status': 'SYNTHETIC_SET_COVERAGE_CONTRACT_VALIDATED',
    }


def self_test(data: dict[str, Any]) -> None:
    altered = json.loads(json.dumps(data))
    altered['hypothesis_universe'][1]['query_decision'] = False
    try:
        evaluate_without_expectations(altered)
    except ValueError as exc:
        require('inconsistent adjudication/query semantics' in str(exc), 'wrong error for equivalence-class metadata mutation')
    else:
        raise AssertionError('inconsistent equivalence-class query semantics were accepted')
    altered = json.loads(json.dumps(data))
    altered['channels'][0]['proposals'][1]['proposal_rank'] = 1
    try:
        evaluate_without_expectations(altered)
    except ValueError as exc:
        require('duplicate proposal ranks' in str(exc), 'wrong error for duplicate rank mutation')
    else:
        raise AssertionError('duplicate proposal ranks were accepted')
    altered = json.loads(json.dumps(data))
    altered['channels'].append(altered['channels'][0])
    try:
        evaluate_without_expectations(altered)
    except ValueError as exc:
        require('duplicate channel IDs' in str(exc), 'wrong error for duplicate channel mutation')
    else:
        raise AssertionError('duplicate channel IDs were accepted')
    print('set-valued hypothesis coverage negative controls: PASS')


def evaluate_without_expectations(data: dict[str, Any]) -> None:
    require(data['claim_scope'] == 'synthetic_known_truth', 'synthetic scope required')
    _ = validate_universe(data)
    validate_channels(data['channels'], data['oracle_expectations']['max_budget'])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_bytes())
        result = evaluate(data)
        if args.self_test:
            self_test(data)
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(rendered + '\n', encoding='utf-8')
        else:
            print(rendered)
        print('independent set-valued hypothesis coverage oracle: PASS', file=sys.stderr)
        return 0
    except Exception as exc:
        print(f'set-valued hypothesis coverage oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())