#!/usr/bin/env python3
"""Strict shape/provenance firewall for sanitized REL-005A QualificationOnly input."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile
from typing import Any

SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
GIT_RE = re.compile(r'^[0-9a-f]{40}$')
ARTIFACT_DIGEST_RE = re.compile(r'^sha256:[0-9a-f]{64}$')

FILES = {
    'predicate': 'predicate-contract-receipt.json',
    'execution': 'execution-v3-receipt.json',
    'seal': 'observation-seal-v3.json',
    'comparison': 'comparison-only-qualification-receipt.json',
    'manifest': 'qualification-input-manifest.json',
    'assembly': 'qualification-assembly-receipt.json',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_pairs)
    require(isinstance(value, dict), f'{path}: expected JSON object')
    return value


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def walk_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(key)
            found |= walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= walk_keys(child)
    return found


def validate_bundle(contract: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    require(contract['schema'] == 'symthaea.rel.qualification-input-firewall-contract.v1', 'contract schema mismatch')
    require(contract['authority'] == 'QualificationInputFirewallContractOnly', 'contract authority mismatch')
    require(root.is_dir(), 'bundle root is not a directory')

    expected_names = set(FILES.values())
    all_files = [p for p in root.rglob('*') if p.is_file()]
    relative_files = {p.relative_to(root).as_posix() for p in all_files}
    require(relative_files == expected_names, f'bundle recursive file census mismatch: {sorted(relative_files)}')
    require(all((root / name).is_file() for name in expected_names), 'required file not at bundle root')

    values = {key: load(root / name) for key, name in FILES.items()}
    schemas = contract['required_schemas']
    for key in FILES:
        require(values[key].get('schema') == schemas[key], f'{key} schema mismatch')

    claims = contract['required_claims']
    for key in ('predicate', 'execution', 'seal', 'comparison', 'assembly'):
        require(values[key].get('claims') == claims[key], f'{key} exact claim set mismatch')

    predicate = values['predicate']
    execution = values['execution']
    seal = values['seal']
    comparison = values['comparison']
    manifest = values['manifest']
    assembly = values['assembly']

    predicate_head = predicate['subject_head']
    execution_head = execution['subject_head']
    frozen_subject = predicate['frozen_scientific_subject']
    frozen_blob = predicate['frozen_test_blob']
    observation_sha = execution['observation_sha256']

    for value, label in (
        (predicate_head, 'predicate head'),
        (execution_head, 'execution head'),
        (frozen_subject, 'frozen subject'),
        (frozen_blob, 'frozen blob'),
        (comparison['predicate_contract_head'], 'comparison predicate head'),
        (comparison['execution_subject_head'], 'comparison execution head'),
    ):
        require(isinstance(value, str) and GIT_RE.fullmatch(value), f'{label} format mismatch')
    require(isinstance(observation_sha, str) and SHA256_RE.fullmatch(observation_sha), 'observation SHA-256 format mismatch')
    require(isinstance(seal.get('chain_commitment_sha256'), str) and SHA256_RE.fullmatch(seal['chain_commitment_sha256']), 'seal commitment format mismatch')
    require(isinstance(comparison.get('full_comparison_sha256'), str) and SHA256_RE.fullmatch(comparison['full_comparison_sha256']), 'full comparison hash format mismatch')

    require(type(predicate['predicate_count']) is int and predicate['predicate_count'] == contract['required_predicate_count'], 'predicate count mismatch')
    require(predicate['source_grounded'] is True, 'predicate receipt not source grounded')
    require(predicate['predicate_contract_static_valid'] is True, 'predicate contract invalid')
    require(execution['predicate_contract_parent'] == predicate_head, 'execution predicate head mismatch')
    require(execution['frozen_scientific_subject'] == frozen_subject, 'execution frozen subject mismatch')
    require(execution['frozen_test_blob'] == frozen_blob, 'execution frozen blob mismatch')
    require(execution['result'] == 'EXECUTION_OK', 'execution not OK')
    require(str(execution['measurement_exit_code']) == '0', 'execution exit mismatch')
    require(execution['observation_present'] is True, 'execution observation absent')

    require(seal['execution_subject_head'] == execution_head, 'seal execution head mismatch')
    require(seal['predicate_contract_head'] == predicate_head, 'seal predicate head mismatch')
    require(seal['frozen_scientific_subject'] == frozen_subject, 'seal frozen subject mismatch')
    require(seal['frozen_test_blob'] == frozen_blob, 'seal frozen blob mismatch')
    require(seal['observation_sha256'] == observation_sha, 'seal observation hash mismatch')
    require(seal['observation_status'] == 'sealed', 'seal status mismatch')
    require(seal['adjudication'] == 'not_run', 'seal carries adjudication')
    require(seal['scientific_result'] == 'not_run', 'seal carries scientific result')

    require(comparison['predicate_contract_head'] == predicate_head, 'comparison predicate head mismatch')
    require(comparison['execution_subject_head'] == execution_head, 'comparison execution head mismatch')
    require(comparison['frozen_scientific_subject'] == frozen_subject, 'comparison frozen subject mismatch')
    require(comparison['frozen_test_blob'] == frozen_blob, 'comparison frozen blob mismatch')
    require(comparison['observation_sha256'] == observation_sha, 'comparison observation hash mismatch')
    require(comparison['seal_chain_commitment_sha256'] == seal['chain_commitment_sha256'], 'comparison seal commitment mismatch')
    require(type(comparison['predicate_count']) is int and comparison['predicate_count'] == contract['required_predicate_count'], 'comparison predicate count mismatch')
    require(type(comparison['passed_count']) is int and comparison['passed_count'] >= 0, 'comparison passed count invalid')
    require(type(comparison['failed_count']) is int and comparison['failed_count'] >= 0, 'comparison failed count invalid')
    require(comparison['passed_count'] + comparison['failed_count'] == contract['required_predicate_count'], 'comparison counts mismatch')

    failed_ids = comparison['failed_predicate_ids']
    require(isinstance(failed_ids, list) and all(isinstance(item, str) for item in failed_ids), 'failed predicate IDs invalid')
    require(len(failed_ids) == len(set(failed_ids)), 'duplicate failed predicate IDs')
    allowed_ids = contract['allowed_failed_predicate_ids']
    allowed_set = set(allowed_ids)
    require(all(item in allowed_set for item in failed_ids), 'unknown failed predicate ID')
    require(failed_ids == sorted(failed_ids, key=allowed_ids.index), 'failed predicate IDs not canonical-order')
    require(comparison['failed_count'] == len(failed_ids), 'failed predicate count mismatch')
    result = comparison['comparison_result']
    require(result in {'ALL_PREDICATES_PASS', 'PREDICATE_FAILURES'}, 'unsupported comparison result')
    if result == 'ALL_PREDICATES_PASS':
        require(comparison['passed_count'] == 41 and comparison['failed_count'] == 0, 'all-pass counts inconsistent')
    else:
        require(comparison['failed_count'] > 0, 'predicate-failure result without failures')

    require(manifest['authority'] == 'EvidenceProjectionOnly', 'manifest authority mismatch')
    require(manifest['predicate_contract_head'] == predicate_head, 'manifest predicate head mismatch')
    require(manifest['execution_subject_head'] == execution_head, 'manifest execution head mismatch')
    require(manifest['observation_sha256'] == observation_sha, 'manifest observation hash mismatch')
    require(manifest['seal_chain_commitment_sha256'] == seal['chain_commitment_sha256'], 'manifest seal commitment mismatch')
    require(manifest['comparison_result'] == result, 'manifest comparison result mismatch')
    require(manifest['raw_observation_included'] is False, 'manifest admits raw observation')
    require(manifest['raw_execution_logs_included'] is False, 'manifest admits execution logs')
    require(manifest['detailed_predicate_values_included'] is False, 'manifest admits detailed metric values')

    authority_names = [
        'predicate-contract-receipt.json',
        'execution-v3-receipt.json',
        'observation-seal-v3.json',
        'comparison-only-qualification-receipt.json',
    ]
    entries = manifest['files']
    require(isinstance(entries, list) and len(entries) == len(authority_names), 'manifest file count mismatch')
    require([entry.get('basename') for entry in entries] == authority_names, 'manifest file order/census mismatch')
    for entry in entries:
        path = root / entry['basename']
        require(type(entry.get('byte_length')) is int and entry['byte_length'] == path.stat().st_size, f"{entry['basename']}: byte length mismatch")
        require(isinstance(entry.get('sha256'), str) and SHA256_RE.fullmatch(entry['sha256']), f"{entry['basename']}: hash format mismatch")
        require(entry['sha256'] == sha256(path), f"{entry['basename']}: hash mismatch")

    require(assembly['authority'] == 'QualificationAssemblyOnly', 'assembly authority mismatch')
    require(assembly['comparison_subject_head'] == '9ed033eb5ad63baa90d9c25166ce9c14e68013cb', 'assembly comparison head mismatch')
    require(assembly['comparison_source_run_id'] == 35324217302, 'assembly comparison run mismatch')
    require(assembly['projection_contract_head'] == '07af7c4bdfdda377aa8175efe7a3233ce90e889e', 'assembly projection head mismatch')
    require(assembly['projection_contract_run_id'] == 35324485955, 'assembly projection run mismatch')
    require(assembly['full_comparison_sha256'] == comparison['full_comparison_sha256'], 'assembly/full-comparison hash mismatch')
    for prefix in ('comparison', 'projection_contract'):
        require(type(assembly[f'{prefix}_source_job_id']) is int and assembly[f'{prefix}_source_job_id'] > 0, f'{prefix} job ID invalid')
        require(type(assembly[f'{prefix}_artifact_id']) is int and assembly[f'{prefix}_artifact_id'] > 0, f'{prefix} artifact ID invalid')
        require(isinstance(assembly[f'{prefix}_artifact_name'], str) and assembly[f'{prefix}_artifact_name'], f'{prefix} artifact name invalid')
        require(isinstance(assembly[f'{prefix}_artifact_digest'], str) and ARTIFACT_DIGEST_RE.fullmatch(assembly[f'{prefix}_artifact_digest']), f'{prefix} artifact digest invalid')

    forbidden = set(contract['forbidden_keys_anywhere_in_sanitized_bundle'])
    for key, value in values.items():
        leaked = forbidden & walk_keys(value)
        require(not leaked, f'{key}: forbidden detailed keys leaked: {sorted(leaked)}')

    return {
        'schema': 'symthaea.rel.qualification-input-firewall-receipt.v1',
        'authority': 'QualificationInputFirewallOnly',
        'bundle_valid': True,
        'comparison_result': result,
        'predicate_count': 41,
        'passed_count': comparison['passed_count'],
        'failed_count': comparison['failed_count'],
        'failed_predicate_ids': failed_ids,
        'detailed_metric_keys_absent': True,
        'manifest_hashes_verified': True,
        'assembly_provenance_verified': True,
        'claims': {
            'qualification_completed': False,
            'rel_005a_qualified': False,
            'scientific_pass': False,
            'scientific_fail': False,
        },
    }


def validate_qualification_output(contract: dict[str, Any], bundle: dict[str, Any], output: dict[str, Any]) -> None:
    require(output.get('schema') == contract['required_schemas']['qualification'], 'qualification output schema mismatch')
    require(output.get('authority') == 'QualificationOnly', 'qualification output authority mismatch')
    require(output.get('qualification_completed') is True, 'qualification not completed')
    scientific_pass = output.get('scientific_pass') is True
    scientific_fail = output.get('scientific_fail') is True
    require(scientific_pass ^ scientific_fail, 'scientific pass/fail must be exclusive and exhaustive')
    require(output.get('comparison_result') == bundle['comparison_result'], 'qualification/comparison result mismatch')
    require(output.get('predicate_count') == bundle['predicate_count'], 'qualification predicate count mismatch')
    require(output.get('passed_count') == bundle['passed_count'], 'qualification passed count mismatch')
    require(output.get('failed_count') == bundle['failed_count'], 'qualification failed count mismatch')
    require(output.get('failed_predicate_ids') == bundle['failed_predicate_ids'], 'qualification failed IDs mismatch')
    if scientific_pass:
        require(bundle['comparison_result'] == 'ALL_PREDICATES_PASS', 'pass does not map from all-pass comparison')
        require(output.get('rel_005a_qualified') is True, 'scientific pass must qualify REL-005A')
        require(output.get('failed_count') == 0, 'scientific pass carries failures')
    else:
        require(bundle['comparison_result'] == 'PREDICATE_FAILURES', 'fail does not map from predicate failures')
        require(output.get('rel_005a_qualified') is False, 'scientific fail must not qualify REL-005A')
        require(output.get('failed_count', 0) > 0, 'scientific fail lacks failures')


def synthetic_bundle(contract: dict[str, Any], root: pathlib.Path, result: str) -> None:
    ids = [] if result == 'ALL_PREDICATES_PASS' else ['REL005A-P041']
    failed = len(ids)
    observation = '33' * 32
    seal_commitment = '44' * 32
    full_comparison = '55' * 32
    predicate = {
        'schema': contract['required_schemas']['predicate'], 'authority': 'PredicateContractOnly',
        'subject_head': '43bf1d588447f602ce0f1549986bb558a839762c', 'predicate_count': 41,
        'source_grounded': True, 'predicate_contract_static_valid': True,
        'frozen_scientific_subject': '7f5826675b44dd0d3f702f62bc9818f7990e4a01',
        'frozen_test_blob': '787ea051ae0bf8e5667b6925481afd46c15d0bc4', 'claims': contract['required_claims']['predicate'],
    }
    execution = {
        'schema': contract['required_schemas']['execution'], 'authority': 'ExecutionOnly',
        'subject_head': '6931639e53060809f9d459196a329c4fe983be7b',
        'predicate_contract_parent': predicate['subject_head'],
        'frozen_scientific_subject': predicate['frozen_scientific_subject'], 'frozen_test_blob': predicate['frozen_test_blob'],
        'result': 'EXECUTION_OK', 'measurement_exit_code': '0', 'observation_present': True,
        'observation_sha256': observation, 'claims': contract['required_claims']['execution'],
    }
    seal = {
        'schema': contract['required_schemas']['seal'], 'authority': 'ObservationSeal', 'observation_status': 'sealed',
        'adjudication': 'not_run', 'scientific_result': 'not_run', 'execution_subject_head': execution['subject_head'],
        'predicate_contract_head': predicate['subject_head'], 'frozen_scientific_subject': predicate['frozen_scientific_subject'],
        'frozen_test_blob': predicate['frozen_test_blob'], 'observation_sha256': observation,
        'chain_commitment_sha256': seal_commitment, 'claims': contract['required_claims']['seal'],
    }
    comparison = {
        'schema': contract['required_schemas']['comparison'], 'authority': 'ComparisonOnly',
        'predicate_contract_head': predicate['subject_head'], 'execution_subject_head': execution['subject_head'],
        'frozen_scientific_subject': predicate['frozen_scientific_subject'], 'frozen_test_blob': predicate['frozen_test_blob'],
        'observation_sha256': observation, 'seal_chain_commitment_sha256': seal_commitment,
        'predicate_count': 41, 'passed_count': 41 - failed, 'failed_count': failed,
        'failed_predicate_ids': ids, 'comparison_result': result, 'full_comparison_sha256': full_comparison,
        'claims': contract['required_claims']['comparison'],
    }
    for key, value in (('predicate', predicate), ('execution', execution), ('seal', seal), ('comparison', comparison)):
        write(root / FILES[key], value)
    entries = []
    for name in [FILES['predicate'], FILES['execution'], FILES['seal'], FILES['comparison']]:
        path = root / name
        entries.append({'basename': name, 'byte_length': path.stat().st_size, 'sha256': sha256(path)})
    manifest = {
        'schema': contract['required_schemas']['manifest'], 'authority': 'EvidenceProjectionOnly',
        'predicate_contract_head': predicate['subject_head'], 'execution_subject_head': execution['subject_head'],
        'observation_sha256': observation, 'seal_chain_commitment_sha256': seal_commitment,
        'comparison_result': result, 'raw_observation_included': False, 'raw_execution_logs_included': False,
        'detailed_predicate_values_included': False, 'files': entries,
    }
    write(root / FILES['manifest'], manifest)
    assembly = {
        'schema': contract['required_schemas']['assembly'], 'authority': 'QualificationAssemblyOnly',
        'comparison_subject_head': '9ed033eb5ad63baa90d9c25166ce9c14e68013cb', 'comparison_source_run_id': 35324217302,
        'comparison_source_job_id': 123, 'comparison_artifact_id': 456, 'comparison_artifact_name': 'comparison',
        'comparison_artifact_digest': 'sha256:' + '66' * 32,
        'projection_contract_head': '07af7c4bdfdda377aa8175efe7a3233ce90e889e', 'projection_contract_run_id': 35324485955,
        'projection_contract_source_job_id': 789, 'projection_contract_artifact_id': 987,
        'projection_contract_artifact_name': 'projection', 'projection_contract_artifact_digest': 'sha256:' + '77' * 32,
        'full_comparison_sha256': full_comparison, 'claims': contract['required_claims']['assembly'],
    }
    write(root / FILES['assembly'], assembly)


def refresh_manifest_entry(root: pathlib.Path, basename: str) -> None:
    manifest_path = root / FILES['manifest']
    manifest = load(manifest_path)
    for entry in manifest['files']:
        if entry['basename'] == basename:
            path = root / basename
            entry['byte_length'] = path.stat().st_size
            entry['sha256'] = sha256(path)
            write(manifest_path, manifest)
            return
    raise ValueError(f'manifest entry not found: {basename}')


def self_test(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        synthetic_bundle(contract, root, 'ALL_PREDICATES_PASS')
        passed_bundle = validate_bundle(contract, root)
        passed_output = {
            'schema': contract['required_schemas']['qualification'], 'authority': 'QualificationOnly',
            'qualification_completed': True, 'comparison_result': 'ALL_PREDICATES_PASS', 'predicate_count': 41,
            'passed_count': 41, 'failed_count': 0, 'failed_predicate_ids': [],
            'rel_005a_qualified': True, 'scientific_pass': True, 'scientific_fail': False,
        }
        validate_qualification_output(contract, passed_bundle, passed_output)

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        synthetic_bundle(contract, root, 'PREDICATE_FAILURES')
        failed_bundle = validate_bundle(contract, root)
        failed_output = {
            'schema': contract['required_schemas']['qualification'], 'authority': 'QualificationOnly',
            'qualification_completed': True, 'comparison_result': 'PREDICATE_FAILURES', 'predicate_count': 41,
            'passed_count': 40, 'failed_count': 1, 'failed_predicate_ids': ['REL005A-P041'],
            'rel_005a_qualified': False, 'scientific_pass': False, 'scientific_fail': True,
        }
        validate_qualification_output(contract, failed_bundle, failed_output)
        bad = dict(failed_output)
        bad['scientific_pass'] = True
        try:
            validate_qualification_output(contract, failed_bundle, bad)
        except ValueError:
            pass
        else:
            raise ValueError('contradictory scientific pass+fail output was accepted')

        comparison_path = root / FILES['comparison']
        comparison = load(comparison_path)
        comparison['predicates'] = [{'observed': 123}]
        write(comparison_path, comparison)
        refresh_manifest_entry(root, FILES['comparison'])
        try:
            validate_bundle(contract, root)
        except ValueError as exc:
            require('forbidden detailed keys leaked' in str(exc), 'metric leakage rejected for wrong reason')
        else:
            raise ValueError('detailed metric leakage was accepted')

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        synthetic_bundle(contract, root, 'ALL_PREDICATES_PASS')
        nested = root / 'nested'
        nested.mkdir()
        (nested / 'raw-observation.json').write_text('{}\n')
        try:
            validate_bundle(contract, root)
        except ValueError as exc:
            require('recursive file census mismatch' in str(exc), 'nested leakage rejected for wrong reason')
        else:
            raise ValueError('nested extra file was accepted')

    duplicate_key_rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'duplicate.json'
        path.write_text('{"a":1,"a":2}\n')
        try:
            load(path)
        except ValueError as exc:
            require('duplicate JSON key' in str(exc), 'duplicate key rejected for wrong reason')
            duplicate_key_rejected = True
        else:
            raise ValueError('duplicate JSON key was accepted')

    return {
        'schema': 'symthaea.rel.qualification-input-firewall-self-test.v1',
        'authority': 'QualificationInputFirewallContractOnly',
        'pass_bundle': 'PASS',
        'failure_bundle': 'PASS',
        'contradictory_final_claims_rejected': True,
        'detailed_metric_leakage_rejected_after_manifest_rehash': True,
        'nested_extra_file_rejected': True,
        'duplicate_json_key_rejected': duplicate_key_rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=pathlib.Path, required=True)
    parser.add_argument('--bundle-dir', type=pathlib.Path)
    parser.add_argument('--qualification-output', type=pathlib.Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    contract = load(args.contract)
    if args.self_test:
        print(json.dumps(self_test(contract), indent=2, sort_keys=True))
        return
    require(args.bundle_dir is not None, '--bundle-dir is required')
    receipt = validate_bundle(contract, args.bundle_dir)
    if args.qualification_output is not None:
        validate_qualification_output(contract, receipt, load(args.qualification_output))
        receipt['qualification_output_valid'] = True
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
