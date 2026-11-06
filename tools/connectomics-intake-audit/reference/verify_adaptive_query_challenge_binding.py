#!/usr/bin/env python3
"""Byte-bound, independent replay verifier for one #10603 scenario challenge.

The checker imports the separate set-filtering Python oracle, recomputes the
entire upstream report from the exact input bytes, and then verifies the exact
challenge entry selected by a frozen binding manifest. It does not call Rust.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    require(isinstance(value, dict), f'{label} must be a JSON object')
    return value


def load_oracle(path: Path):
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('adaptive_query_policy_oracle', path)
    require(spec is not None and spec.loader is not None, 'could not load independent oracle')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, raw


def unique_challenge(rows: Any, challenge_id: str, label: str) -> dict[str, Any]:
    require(isinstance(rows, list), f'{label} must be an array')
    matches = [row for row in rows if isinstance(row, dict) and row.get('challenge_id') == challenge_id]
    require(len(matches) == 1, f'{label} must contain challenge {challenge_id!r} exactly once; found {len(matches)}')
    return matches[0]


def validate_manifest_identity(binding: dict[str, Any], input_data: dict[str, Any]) -> None:
    expected = binding['expected_base']
    actual = {
        'dataset_revision_commit_sha': input_data['dataset_revision_commit_sha'],
        'query_id': input_data['query_id'],
        'query_spec_sha256': input_data['query_spec_sha256'],
        'scenario_set_sha256': input_data['scenario_set_sha256'],
        'action_set_sha256': input_data['action_set_sha256'],
        'joint_support_sha256': input_data['joint_support_sha256'],
    }
    require(actual == expected, 'base input identity does not match the frozen challenge binding')
    require(re.fullmatch(r'[0-9a-f]{40}', actual['dataset_revision_commit_sha']) is not None, 'dataset revision SHA malformed')
    for key, value in actual.items():
        if key != 'dataset_revision_commit_sha':
            require(re.fullmatch(r'[0-9a-f]{64}', value) is not None, f'{key} malformed')


def validate_challenge_semantics(binding: dict[str, Any], challenge_result: dict[str, Any], computed_challenge_sha: str) -> None:
    expected = binding['expected_challenge_result']
    allowed_fields = {
        'added_scenario_count',
        'adaptive_worst_case_effort_before',
        'adaptive_worst_case_effort_after',
        'adaptive_status_before',
        'adaptive_status_after',
        'adaptive_effort_non_decreasing',
        'fixed_minimum_effort_before',
        'fixed_minimum_effort_after',
        'fixed_status_before',
        'fixed_status_after',
        'fixed_effort_non_decreasing',
        'monotonicity_passed',
    }
    fields = tuple(expected)
    require(fields and set(fields).issubset(allowed_fields), 'expected challenge result fields are empty or unsupported')
    actual = {field: challenge_result.get(field) for field in fields}
    require(actual == expected, 'challenge result differs from the frozen expected outcome')
    require(challenge_result.get('challenge_sha256') == computed_challenge_sha, 'report challenge digest differs from the digest recomputed from exact base input bytes')
    require(re.fullmatch(r'[0-9a-f]{64}', challenge_result.get('expanded_scenario_set_sha256', '')) is not None, 'expanded scenario-set digest malformed')
    require(re.fullmatch(r'[0-9a-f]{64}', challenge_result.get('expanded_joint_support_sha256', '')) is not None, 'expanded joint-support digest malformed')


def build_receipt(
    input_bytes: bytes,
    report_bytes: bytes,
    input_data: dict[str, Any],
    report: dict[str, Any],
    binding: dict[str, Any],
    rust_source_bytes: bytes,
    oracle_bytes: bytes,
    upstream_commit_sha: str,
    oracle_module: Any,
) -> dict[str, Any]:
    require(re.fullmatch(r'[0-9a-f]{40}', upstream_commit_sha) is not None, 'upstream source commit SHA malformed')
    require(report.get('input_sha256') == sha256(input_bytes), 'upstream report input_sha256 does not bind the exact raw input bytes')
    require(report.get('search_complete') is True, 'upstream report does not assert complete exact search')
    require(report.get('status') in {'MINIMUM_WORST_CASE_ADAPTIVE_POLICY', 'MINIMUM_WORST_CASE_ADAPTIVE_POLICY_EXCEEDS_BUDGET', 'NO_ADAPTIVE_POLICY_WITHIN_DECLARED_ACTIONS'}, 'upstream report status is not an exact terminal status')
    require(report.get('query_id') == input_data.get('query_id'), 'report query_id differs from input')
    require(report.get('query_spec_sha256') == input_data.get('query_spec_sha256'), 'report query specification digest differs from input')
    require(report.get('dataset_revision_commit_sha') == input_data.get('dataset_revision_commit_sha'), 'report dataset revision differs from input')
    for field in ('scenario_set_sha256', 'action_set_sha256', 'joint_support_sha256'):
        require(report.get(field) == input_data.get(field), f'report {field} differs from the frozen base input')

    challenge_id = binding['challenge_id']
    input_challenge = unique_challenge(input_data.get('scenario_challenges', []), challenge_id, 'input scenario_challenges')
    report_challenge = unique_challenge(report.get('scenario_challenges', []), challenge_id, 'report scenario_challenges')
    actions = sorted(input_data['actions'], key=lambda row: row['action_id'])
    action_ids = [action['action_id'] for action in actions]
    computed_challenge_sha = oracle_module.scenario_expansion_challenge_digest(input_data, input_challenge, action_ids)
    validate_challenge_semantics(binding, report_challenge, computed_challenge_sha)

    expected_report = oracle_module.run(input_data, input_bytes)
    require(report == expected_report, 'upstream report differs from the independent Python set-filtering replay')
    require(report_challenge == unique_challenge(expected_report['scenario_challenges'], challenge_id, 'independent oracle scenario_challenges'), 'selected challenge entry differs from independent replay')

    expected_rust_blob = binding['upstream_source']['rust_git_blob_sha1']
    expected_oracle_blob = binding['upstream_source']['oracle_git_blob_sha1']
    actual_rust_blob = git_blob_sha1(rust_source_bytes)
    actual_oracle_blob = git_blob_sha1(oracle_bytes)
    require(actual_rust_blob == expected_rust_blob, 'upstream Rust planner source differs from the pinned Git blob')
    require(actual_oracle_blob == expected_oracle_blob, 'independent oracle source differs from the pinned Git blob')

    return {
        'schema_version': 1,
        'tool': 'verify-adaptive-query-challenge-binding',
        'status': 'INDEPENDENT_CHALLENGE_REPLAY_PASSED',
        'claim_scope': 'finite_synthetic_hypothesis_expansion_fixture',
        'challenge_id': challenge_id,
        'upstream_commit_sha': upstream_commit_sha,
        'upstream_rust_git_blob_sha1': actual_rust_blob,
        'independent_oracle_git_blob_sha1': actual_oracle_blob,
        'base_input_raw_sha256': sha256(input_bytes),
        'upstream_report_raw_sha256': sha256(report_bytes),
        'upstream_report_input_sha256': report['input_sha256'],
        'base_dataset_revision_commit_sha': input_data['dataset_revision_commit_sha'],
        'query_id': input_data['query_id'],
        'query_spec_sha256': input_data['query_spec_sha256'],
        'base_scenario_set_sha256': input_data['scenario_set_sha256'],
        'base_action_set_sha256': input_data['action_set_sha256'],
        'base_joint_support_sha256': input_data['joint_support_sha256'],
        'challenge_sha256': report_challenge['challenge_sha256'],
        'expanded_scenario_set_sha256': report_challenge['expanded_scenario_set_sha256'],
        'expanded_joint_support_sha256': report_challenge['expanded_joint_support_sha256'],
        'added_scenario_count': report_challenge['added_scenario_count'],
        'adaptive_worst_case_effort_before': report_challenge['adaptive_worst_case_effort_before'],
        'adaptive_worst_case_effort_after': report_challenge['adaptive_worst_case_effort_after'],
        'adaptive_status_before': report_challenge['adaptive_status_before'],
        'adaptive_status_after': report_challenge['adaptive_status_after'],
        'adaptive_effort_non_decreasing': report_challenge['adaptive_effort_non_decreasing'],
        'fixed_minimum_effort_before': report_challenge['fixed_minimum_effort_before'],
        'fixed_minimum_effort_after': report_challenge['fixed_minimum_effort_after'],
        'fixed_status_before': report_challenge['fixed_status_before'],
        'fixed_status_after': report_challenge['fixed_status_after'],
        'fixed_effort_non_decreasing': report_challenge['fixed_effort_non_decreasing'],
        'monotonicity_passed': report_challenge['monotonicity_passed'],
        'raw_artifact_binding': 'EXACT_INPUT_AND_REPORT_BYTES_SHA256',
        'independent_replay': 'SEPARATE_PYTHON_TUPLE_FILTERING_ORACLE',
        'biological_admissibility': 'NOT_ESTABLISHED_BY_THIS_SOFTWARE_REPLAY',
        'hypothesis_space_completeness': 'NOT_ESTABLISHED_BY_THIS_SOFTWARE_REPLAY',
    }


def self_test() -> None:
    sample = {
        'expected_base': {'query_id': 'q'},
        'expected_challenge_result': {'monotonicity_passed': True},
        'expected_challenge_sha256': 'a' * 64,
    }
    try:
        validate_manifest_identity(sample, {'query_id': 'wrong'})
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError('cross-query binding self-test was not rejected')
    try:
        unique_challenge([{'challenge_id': 'c'}, {'challenge_id': 'c'}], 'c', 'self-test')
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate challenge ID self-test was not rejected')
    print('challenge-binding negative controls: PASS')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--rust-source', type=Path)
    parser.add_argument('--upstream-commit-sha')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        required = (args.input, args.report, args.binding, args.oracle, args.rust_source, args.receipt, args.upstream_commit_sha)
        require(all(value is not None for value in required), 'all artifact paths, upstream commit SHA and receipt path are required')
        input_bytes = args.input.read_bytes()
        report_bytes = args.report.read_bytes()
        binding_bytes = args.binding.read_bytes()
        rust_source_bytes = args.rust_source.read_bytes()
        oracle_module, oracle_bytes = load_oracle(args.oracle)
        input_data = load_json_bytes(input_bytes, 'input')
        report = load_json_bytes(report_bytes, 'report')
        binding = load_json_bytes(binding_bytes, 'binding')
        validate_manifest_identity(binding, input_data)
        receipt = build_receipt(
            input_bytes, report_bytes, input_data, report, binding,
            rust_source_bytes, oracle_bytes, args.upstream_commit_sha, oracle_module
        )
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
        args.receipt.write_text(rendered, encoding='utf-8')
        print(
            'independent challenge replay: PASS; '
            f"challenge={receipt['challenge_id']}; "
            f"input_sha256={receipt['base_input_raw_sha256']}; "
            f"report_sha256={receipt['upstream_report_raw_sha256']}"
        )
        return 0
    except Exception as exc:
        print(f'adaptive challenge binding failure: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())