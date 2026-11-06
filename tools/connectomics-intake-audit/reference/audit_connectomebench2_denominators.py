#!/usr/bin/env python3
"""Recompute the metadata-only ConnectomeBench2 V1 card denominator fixture.

This is a source-card arithmetic and contract verifier. It does not access
Parquet payloads, reproduce V2, or turn a card count into a paper decision count.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique(values: list[str], label: str) -> None:
    require(len(values) == len(set(values)), f'{label} contains duplicate values')


def validate_manifest(data: dict[str, Any]) -> dict[str, Any]:
    require(data.get('schema_version') == 1, 'unsupported schema version')
    require(data.get('scope') == 'metadata_only_source_card_known_answer', 'unexpected audit scope')
    sources = {row['source_id']: row for row in data['sources']}
    require(len(sources) == len(data['sources']), 'duplicate source_id')
    for source in sources.values():
        state = source['revision_identity_status']
        if state == 'immutable_full_commit_pinned':
            require(re.fullmatch(r'[0-9a-f]{40}', source['commit_sha'] or '') is not None, 'immutable source revision lacks full commit SHA')
        elif state == 'short_identifier_only':
            require(source['commit_sha'] is None, 'short-only revision incorrectly includes full SHA')
            require(re.fullmatch(r'[0-9a-f]{7,12}', source['commit_short'] or '') is not None, 'short-only revision lacks a valid short identifier')
        elif state == 'revision_label_only_unpinned':
            require(source['commit_sha'] is None and source['commit_short'] is None, 'unresolved revision must not invent a commit identifier')
        else:
            raise ValueError(f'unknown revision identity status: {state}')

    observation_ids = [row['observation_id'] for row in data['observations']]
    unique(observation_ids, 'observation IDs')
    for row in data['observations']:
        require(row['source_id'] in sources, f"observation references unknown source {row['source_id']}")
        require(row['value'] >= 0, 'denominator observations cannot be negative')

    sample_rows = data['sample_type_counts']
    split_rows = data['split_counts']
    unique([row['category'] for row in sample_rows], 'sample_type categories')
    unique([row['category'] for row in split_rows], 'split categories')
    sample_total = sum(row['value'] for row in sample_rows)
    split_total = sum(row['value'] for row in split_rows)
    card_total = next(
        row['value'] for row in data['observations']
        if row['observation_id'] == 'v1-dataset-rows'
    )
    require(sample_total == data['expected_checks']['sample_type_count_sum'], 'sample-type arithmetic expectation is inconsistent')
    require(card_total == data['expected_checks']['reported_dataset_rows'], 'dataset-row expectation is inconsistent')
    difference = sample_total - card_total
    require(difference == data['expected_checks']['sample_type_difference'], 'sample-type discrepancy expectation is inconsistent')
    require(split_total == data['expected_checks']['split_count_sum'], 'split arithmetic expectation is inconsistent')
    require(split_total == card_total, 'train/validation/test counts do not reconcile to the card total')
    require(data['expected_checks']['split_total_matches'] is True, 'split total expected to reconcile')
    require(data['expected_checks']['sample_type_partition_status'] == ('MATCH' if difference == 0 else 'MISMATCH'), 'sample-type check status is inconsistent')
    if difference:
        require(data['expected_checks']['sample_type_partition_disposition'] == 'UNRESOLVED_CARD_INCONSISTENCY', 'a mismatch must remain explicitly unresolved')
    require(data['expected_checks']['paper_decision_unit_comparison'] == 'NOT_COMPARABLE_WITH_DATASET_ROWS_UNTIL_COUNTING_RULES_RECONCILED', 'paper decision count comparison must remain quarantined')

    v2 = data['v2_artifact_pin']
    require(v2['revision_label'] == 'v2', 'V2 revision label changed')
    if v2['identity_status'] == 'revision_label_only_unpinned':
        require(v2['commit_sha'] is None, 'unresolved V2 revision must not fabricate a commit SHA')
        require(v2['shard_manifest_status'] == 'not_materialized', 'unpinned V2 must not claim a verified shard manifest')
        require(data['expected_checks']['v2_reproduction_status'] != 'REPRODUCED_ON_PINNED_V2', 'un-pinned V2 cannot be called reproduced')
    elif v2['identity_status'] == 'immutable_full_commit_pinned':
        require(re.fullmatch(r'[0-9a-f]{40}', v2['commit_sha'] or '') is not None, 'pinned V2 requires full commit SHA')
    else:
        raise ValueError('unsupported V2 identity status')

    coverage_types = {row['sample_type'] for row in data['em_coverage']}
    require(coverage_types == {row['category'] for row in sample_rows}, 'EM coverage categories do not match sample types')
    task_ids = [row['task_id'] for row in data['task_contract']]
    unique(task_ids, 'task IDs')
    require(set(task_ids) == {'false_split_correction', 'false_merge_identification', 'split_mask_generation'}, 'task contract coverage is incomplete')

    return {
        'schema_version': 1,
        'tool': 'connectomebench2-denominator-card-audit',
        'audit_id': data['audit_id'],
        'scope': data['scope'],
        'source_revision_status': {source_id: row['revision_identity_status'] for source_id, row in sorted(sources.items())},
        'sample_type_count_sum': sample_total,
        'reported_dataset_rows': card_total,
        'sample_type_difference': difference,
        'sample_type_partition_status': 'MISMATCH' if difference else 'MATCH',
        'sample_type_partition_disposition': data['expected_checks']['sample_type_partition_disposition'],
        'split_count_sum': split_total,
        'split_total_matches': split_total == card_total,
        'paper_expert_decisions': 'NOT_COMPARABLE_WITH_DATASET_ROWS',
        'v2_commit_sha': v2['commit_sha'],
        'v2_shard_manifest_status': v2['shard_manifest_status'],
        'v2_reproduction_status': data['expected_checks']['v2_reproduction_status'],
        'status': 'SOURCE_CARD_ARITHMETIC_VALIDATED_V2_NOT_REPRODUCED',
        'claim_ceiling': 'V1_MAIN_CARD_ARITHMETIC_ONLY_NO_V2_PARQUET_RECOUNT_OR_BIOLOGICAL_MODEL_REPRODUCTION',
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding='utf-8'))
        report = validate_manifest(data)
        rendered = json.dumps(report, indent=2, sort_keys=True) + '\\n'
        if args.output is not None:
            args.output.write_text(rendered, encoding='utf-8')
        print(rendered, end='')
        if report['sample_type_difference']:
            print(
                f"UNRESOLVED SOURCE-CARD DISCREPANCY: sample_type_sum={report['sample_type_count_sum']} "
                f"card_total={report['reported_dataset_rows']} difference={report['sample_type_difference']}",
                file=sys.stderr,
            )
        return 0
    except Exception as exc:
        print(f'ConnectomeBench2 denominator audit rejected manifest: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())