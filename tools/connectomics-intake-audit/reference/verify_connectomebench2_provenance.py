#!/usr/bin/env python3
"""Validate version-separated ConnectomeBench2 denominators and comparator claims.

Contract oracle for the checked-in source-reconciliation manifest. It does not
download or inspect the full V2 dataset; it verifies source-reported arithmetic,
keeps unresolved source conflicts explicit, and blocks false exact-reproduction claims.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def revision_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = data['dataset_revisions']
    require(len(rows) == 2, 'exactly V1 and V2 revision records are required')
    indexed = {row['revision_id']: row for row in rows}
    require(set(indexed) == {'V1_INITIAL_SUBMISSION', 'V2_PAPER_REVISION'}, 'revision IDs must remain distinct V1 and V2')
    return indexed


def validate(data: dict[str, Any]) -> dict[str, Any]:
    require(data['schema_version'] == 1, 'unsupported manifest schema version')
    sources = {row['source_id']: row for row in data['sources']}
    require(len(sources) == len(data['sources']), 'duplicate source_id')
    for source_id in ('cb2_hf_main', 'cb2_hf_main_commit', 'cb2_paper', 'vlm_preprint_2609_36492'):
        require(source_id in sources, f'missing required source {source_id}')

    revisions = revision_map(data)
    v1 = revisions['V1_INITIAL_SUBMISSION']
    v2 = revisions['V2_PAPER_REVISION']

    # V1 exact source pin and split arithmetic.
    require(v1['revision_selector'] == 'main', 'V1 must remain bound to the observed main selector')
    require(v1['immutable_git_commit_sha1'] == '1039fdf5d69f221a7078f0a66c31b1cc04080cc3', 'V1 immutable source pin changed')
    require(v1['pin_status'] == 'PINNED_IMMUTABLE_REVISION', 'V1 must be immutably pinned')
    require(v1['denominator_kind'] == 'DATASET_ROWS', 'V1 denominator must be dataset rows')
    require(v1['train_count'] + v1['validation_count'] + v1['test_count'] == v1['row_count'] == 401170, 'V1 split counts do not reconcile to the V1 row count')
    require(v1['em_rows'] == 251499, 'V1 EM row count changed from source-reported value')
    require(v1['geometry_rows'] == 401170, 'V1 geometry row count changed from source-reported value')
    sample_counts = {row['sample_type']: row['rows'] for row in v1['sample_type_counts']}
    require(len(sample_counts) == 5, 'V1 must preserve all five sample types')
    require(sum(sample_counts.values()) == 401461, 'V1 sample-type subtotal changed')
    require(sample_counts == {
        'adjacent_control': 121333,
        'junction_control': 38272,
        'synapse_control': 18182,
        'merge_edit': 146461,
        'split_edit': 77213,
    }, 'V1 sample-type counts differ from the source card')
    reconciliation = v1['sample_type_count_reconciliation']
    expected_delta = sum(sample_counts.values()) - v1['row_count']
    require(reconciliation['reported_sum'] == 401461, 'V1 declared sample-type sum mismatch')
    require(reconciliation['delta_vs_row_count'] == expected_delta == 291, 'V1 sample-type source discrepancy must be 291 rows')
    require(reconciliation['status'] == 'SOURCE_REPORTED_COUNTS_DO_NOT_RECONCILE', 'V1 source count mismatch must remain explicit')
    coverage = {row['sample_type']: row for row in v1['sample_type_counts']}
    for sample_type in ('adjacent_control', 'junction_control', 'synapse_control'):
        require(coverage[sample_type]['em_coverage_percent'] == 100, f'{sample_type} EM coverage changed')
        require(coverage[sample_type]['coverage_rounding'] == 'EXACT', f'{sample_type} EM coverage should remain exact')
    require(coverage['merge_edit']['em_coverage_percent'] == 38 and coverage['merge_edit']['coverage_rounding'] == 'ROUNDED_AS_SOURCE_REPORTED', 'merge-edit EM coverage must remain a rounded source percentage')
    require(coverage['split_edit']['em_coverage_percent'] == 23 and coverage['split_edit']['coverage_rounding'] == 'ROUNDED_AS_SOURCE_REPORTED', 'split-edit EM coverage must remain a rounded source percentage')

    # V2 uses paper-reported counts but is not yet bound to an immutable dataset commit.
    require(v2['revision_selector'] == 'v2', 'V2 selector changed')
    require(v2['immutable_git_commit_sha1'] is None, 'V2 must not claim an immutable SHA until it is retrieved')
    require(v2['pin_status'] == 'UNPINNED_IMMUTABLE_REVISION', 'V2 must remain unpinned until immutable revision identity is established')
    require(v2['denominator_kind'] == 'EXPERT_LABELLED_DECISIONS', 'V2 denominator must state expert-labelled decisions')
    require(v2['train_count'] + v2['validation_count'] + v2['test_count'] == v2['row_count'] == 716485, 'V2 reported split arithmetic does not reconcile')
    require(v2['em_rows'] is None and v2['geometry_rows'] is None, 'V1 modality coverage must not be copied into V2')
    require(v2['sample_type_counts'] == [], 'V1 sample-type census must not be copied into V2')
    require(v2['sample_type_count_reconciliation']['status'] == 'NOT_REPORTED', 'V2 unreported sample type census must remain NOT_REPORTED')

    # Exact V2 VLM full-test denominator and per-species split.
    full = data['v2_full_test']
    require(full['revision_id'] == 'V2_PAPER_REVISION', 'full-test regime must identify V2')
    require(full['test_operation_count'] == 101929, 'V2 full-test operation denominator changed')
    require(full['train_count'] == 531734 and full['validation_count'] == 82822, 'V2 train/validation denominators changed')
    require(full['train_count'] + full['validation_count'] + full['test_operation_count'] == 716485, 'V2 train + validation + test does not match paper total')
    expected_species = {
        'mouse': (20879, 22094),
        'fly': (24065, 24203),
        'human': (5351, 60),
        'zebrafish': (5223, 54),
    }
    actual_species = {row['species']: (row['split_error_count'], row['merge_error_count']) for row in full['species_task_counts']}
    require(actual_species == expected_species, 'V2 species/task test counts differ from VLM preprint')
    split_total = sum(row['split_error_count'] for row in full['species_task_counts'])
    merge_total = sum(row['merge_error_count'] for row in full['species_task_counts'])
    require(split_total == 55518 and merge_total == 46411 and split_total + merge_total == 101929, 'V2 task subtotals do not reconcile')
    require(full['metric_name'] == 'BALANCED_ACCURACY_PERCENT', 'VLM metric must be balanced accuracy')

    # Full V2 test metrics must be paired and labelled consistently.
    expected_metrics = {
        'Qwen3.5-9B': (94.6, 90.2),
        'InternVL3-8B': (93.9, 90.1),
        'Gemma 4 12B': (95.0, 90.9),
        'ConnectomeBench2 ViT-B': (94.2, 91.1),
    }
    metric_rows = {row['model']: (row['split_error_balanced_accuracy_percent'], row['merge_error_balanced_accuracy_percent']) for row in full['task_metrics']}
    manifest_rows = {row['model']: (row['split_error_balanced_accuracy_percent'], row['merge_error_balanced_accuracy_percent']) for row in data['v2_model_results']}
    require(metric_rows == expected_metrics, 'full-test model table differs from the frozen preprint values')
    require(manifest_rows == expected_metrics, 'comparator matrix differs from the full-test V2 table')
    require(all(row['dataset_revision_id'] == 'V2_PAPER_REVISION' and row['table'] == 'TABLE_2_FULL_V2_TEST' for row in data['v2_model_results']), 'V2 model results are incorrectly tagged to a different evaluation regime')

    # Claim guard: no V2 exact reproduction while its immutable revision is unknown.
    policy = data['claim_policy']
    require(policy['require_immutable_v2_revision'] is True, 'immutable V2 revision must be mandatory')
    require(policy['prohibit_cross_revision_denominator_pooling'] is True, 'cross-revision denominator pooling must be prohibited')
    require(policy['prohibit_inferred_v2_em_coverage'] is True, 'V2 EM coverage must not be inferred from V1')
    require(policy['require_matched_endpoint_comparison'] is True, 'competitor comparisons must use matched endpoints')
    require(policy['allow_v2_exact_reproduction_claim'] is False, 'V2 exact reproduction cannot be allowed until immutable revision pin is complete')
    require(len(data['open_reproducibility_gaps']) >= 4, 'missing V2 custody gaps must remain listed')

    return {
        'status': 'PROVENANCE_RECONCILED_WITH_V2_PIN_PENDING',
        'v1_revision': {
            'rows': v1['row_count'],
            'split_total': v1['train_count'] + v1['validation_count'] + v1['test_count'],
            'sample_type_reported_sum': reconciliation['reported_sum'],
            'sample_type_delta': expected_delta,
            'sample_type_status': reconciliation['status'],
            'em_rows': v1['em_rows'],
        },
        'v2_revision': {
            'reported_decisions': v2['row_count'],
            'split_total': v2['train_count'] + v2['validation_count'] + v2['test_count'],
            'immutable_revision_sha1': None,
            'pin_status': v2['pin_status'],
            'full_test_operations': full['test_operation_count'],
            'test_split_errors': split_total,
            'test_merge_errors': merge_total,
        },
        'competitor_rows': len(expected_metrics),
        'v2_exact_reproduction_claim_allowed': False,
        'claim_scope': 'SOURCE_REPORTED_DENOMINATORS_ONLY_NO_DATASET_RERUN',
    }


def self_test(data: dict[str, Any]) -> None:
    baseline = validate(copy.deepcopy(data))
    require(baseline['status'] == 'PROVENANCE_RECONCILED_WITH_V2_PIN_PENDING', 'baseline status wrong')
    mutated = copy.deepcopy(data)
    mutated['dataset_revisions'][0]['sample_type_count_reconciliation']['delta_vs_row_count'] = 0
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('V1 source count discrepancy mutation was not rejected')
    mutated = copy.deepcopy(data)
    mutated['dataset_revisions'][1]['test_count'] += 1
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('V2 denominator mutation was not rejected')
    mutated = copy.deepcopy(data)
    mutated['dataset_revisions'][1]['em_rows'] = 251499
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('V1-to-V2 EM coverage leakage was not rejected')
    mutated = copy.deepcopy(data)
    mutated['claim_policy']['allow_v2_exact_reproduction_claim'] = True
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('unresolved V2 exact reproduction claim was not rejected')
    print('ConnectomeBench2 provenance adversarial controls: PASS')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_bytes())
        report = validate(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.self_test:
            self_test(data)
        print('ConnectomeBench2 provenance and denominator audit: PASS')
        return 0
    except Exception as exc:
        print(f'ConnectomeBench2 provenance audit: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())