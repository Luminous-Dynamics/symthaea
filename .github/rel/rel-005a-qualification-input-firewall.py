#!/usr/bin/env python3
"""Strict content-addressed firewall for sanitized REL-005A qualification input."""

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


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in out, f'duplicate JSON key: {key}')
        out[key] = value
    return out


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=no_duplicate_pairs)
    require(isinstance(value, dict), f'{path}: expected object')
    return value


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(key)
            result |= walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            result |= walk_keys(child)
    return result


def valid_git(value: Any) -> bool:
    return isinstance(value, str) and GIT_RE.fullmatch(value) is not None


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def valid_digest(value: Any) -> bool:
    return isinstance(value, str) and ARTIFACT_DIGEST_RE.fullmatch(value) is not None


def validate_bundle(contract: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
    require(contract['schema'] == 'symthaea.rel.qualification-input-firewall-contract.v2', 'contract schema mismatch')
    require(contract['authority'] == 'QualificationInputFirewallContractOnly', 'contract authority mismatch')
    require(root.is_dir(), 'bundle root missing')

    expected_names = set(FILES.values())
    all_files = [p for p in root.rglob('*') if p.is_file()]
    rel = {p.relative_to(root).as_posix() for p in all_files}
    require(rel == expected_names, f'recursive file census mismatch: {sorted(rel)}')

    values = {key: load(root / name) for key, name in FILES.items()}
    schemas = contract['required_schemas']
    for key, value in values.items():
        require(value.get('schema') == schemas[key], f'{key} schema mismatch')

    claims = contract['required_claims']
    for key in ('predicate', 'execution', 'seal', 'comparison', 'assembly'):
        require(values[key].get('claims') == claims[key], f'{key} claim set mismatch')

    p = values['predicate']; e = values['execution']; s = values['seal']
    c = values['comparison']; m = values['manifest']; a = values['assembly']

    predicate_head = p['subject_head']; execution_head = e['subject_head']
    frozen_subject = p['frozen_scientific_subject']; frozen_blob = p['frozen_test_blob']
    observation_sha = e['observation_sha256']
    for value, label in ((predicate_head, 'predicate head'), (execution_head, 'execution head'),
                         (frozen_subject, 'frozen subject'), (frozen_blob, 'frozen blob')):
        require(valid_git(value), f'{label} format mismatch')
    require(valid_sha(observation_sha), 'observation hash format mismatch')

    count = contract['required_predicate_count']
    require(type(p['predicate_count']) is int and p['predicate_count'] == count, 'predicate count mismatch')
    require(p['source_grounded'] is True and p['predicate_contract_static_valid'] is True, 'predicate contract invalid')
    require(e['predicate_contract_parent'] == predicate_head, 'execution predicate head mismatch')
    require(e['frozen_scientific_subject'] == frozen_subject and e['frozen_test_blob'] == frozen_blob, 'execution frozen identity mismatch')
    require(e['result'] == 'EXECUTION_OK' and str(e['measurement_exit_code']) == '0' and e['observation_present'] is True, 'execution not admissible')

    require(s['authority'] == 'ObservationSeal' and s['observation_status'] == 'sealed', 'seal status mismatch')
    require(s['adjudication'] == 'not_run' and s['scientific_result'] == 'not_run', 'seal exceeds authority')
    require(s['execution_subject_head'] == execution_head and s['predicate_contract_head'] == predicate_head, 'seal subject mismatch')
    require(s['frozen_scientific_subject'] == frozen_subject and s['frozen_test_blob'] == frozen_blob, 'seal frozen identity mismatch')
    require(s['observation_sha256'] == observation_sha and valid_sha(s['chain_commitment_sha256']), 'seal commitment mismatch')

    require(c['authority'] == 'ComparisonOnly', 'comparison authority mismatch')
    require(c['predicate_contract_head'] == predicate_head and c['execution_subject_head'] == execution_head, 'comparison subject mismatch')
    require(c['frozen_scientific_subject'] == frozen_subject and c['frozen_test_blob'] == frozen_blob, 'comparison frozen identity mismatch')
    require(c['observation_sha256'] == observation_sha, 'comparison observation mismatch')
    require(c['seal_chain_commitment_sha256'] == s['chain_commitment_sha256'], 'comparison seal mismatch')
    require(valid_sha(c['full_comparison_sha256']), 'full comparison hash invalid')
    require(type(c['predicate_count']) is int and c['predicate_count'] == count, 'comparison predicate count mismatch')
    require(type(c['passed_count']) is int and c['passed_count'] >= 0, 'passed count invalid')
    require(type(c['failed_count']) is int and c['failed_count'] >= 0, 'failed count invalid')
    require(c['passed_count'] + c['failed_count'] == count, 'comparison counts inconsistent')

    ids = c['failed_predicate_ids']
    allowed = contract['allowed_failed_predicate_ids']
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids), 'failed IDs invalid')
    require(len(ids) == len(set(ids)), 'duplicate failed IDs')
    require(all(x in set(allowed) for x in ids), 'unknown failed ID')
    require(ids == sorted(ids, key=allowed.index), 'failed IDs not canonical')
    require(c['failed_count'] == len(ids), 'failed count/IDs mismatch')
    result = c['comparison_result']
    require(result in {'ALL_PREDICATES_PASS', 'PREDICATE_FAILURES'}, 'unsupported comparison result')
    if result == 'ALL_PREDICATES_PASS':
        require(c['passed_count'] == count and c['failed_count'] == 0, 'all-pass counts inconsistent')
    else:
        require(c['failed_count'] > 0, 'predicate-failure result has no failures')

    require(m['authority'] == 'EvidenceProjectionOnly', 'manifest authority mismatch')
    require(m['predicate_contract_head'] == predicate_head and m['execution_subject_head'] == execution_head, 'manifest subject mismatch')
    require(m['observation_sha256'] == observation_sha, 'manifest observation mismatch')
    require(m['seal_chain_commitment_sha256'] == s['chain_commitment_sha256'], 'manifest seal mismatch')
    require(m['comparison_result'] == result, 'manifest comparison result mismatch')
    require(m['raw_observation_included'] is False, 'raw observation included')
    require(m['raw_execution_logs_included'] is False, 'execution logs included')
    require(m['detailed_predicate_values_included'] is False, 'detailed metrics included')

    authority_names = [FILES[x] for x in ('predicate', 'execution', 'seal', 'comparison')]
    entries = m['files']
    require(isinstance(entries, list) and [x.get('basename') for x in entries] == authority_names, 'manifest file census/order mismatch')
    for entry in entries:
        path = root / entry['basename']
        require(type(entry.get('byte_length')) is int and entry['byte_length'] == path.stat().st_size, f"{entry['basename']}: length mismatch")
        require(valid_sha(entry.get('sha256')) and entry['sha256'] == sha256(path), f"{entry['basename']}: hash mismatch")

    prov = contract['authoritative_provenance']
    require(a['authority'] == 'QualificationAssemblyOnly', 'assembly authority mismatch')
    require(a['assembly_contract_head'] == contract['assembly_contract_head'], 'assembly contract head mismatch')
    require(a['comparison_subject_head'] == prov['comparison_subject_head'], 'assembly comparison head mismatch')
    require(a['comparison_source_run_id'] == prov['comparison_source_run_id'], 'assembly comparison run mismatch')
    require(a['projection_contract_head'] == prov['projection_contract_head'], 'assembly projection head mismatch')
    require(a['projection_contract_run_id'] == prov['projection_contract_run_id'], 'assembly projection run mismatch')
    require(a['full_comparison_sha256'] == c['full_comparison_sha256'], 'assembly comparison hash mismatch')
    require(valid_sha(a['projection_contract_receipt_sha256']), 'projection receipt hash invalid')
    require(a['transport_identity_scientific_authority'] is False, 'assembly promotes transport identity')
    require(a['inner_content_commitments_verified'] is True, 'assembly did not verify inner content')

    for prefix in ('comparison', 'projection_contract'):
        require(type(a[f'{prefix}_source_job_id']) is int and a[f'{prefix}_source_job_id'] > 0, f'{prefix} job ID invalid')
        require(type(a[f'{prefix}_artifact_id']) is int and a[f'{prefix}_artifact_id'] > 0, f'{prefix} artifact ID invalid')
        require(isinstance(a[f'{prefix}_artifact_name'], str) and a[f'{prefix}_artifact_name'], f'{prefix} artifact name invalid')
        require(valid_digest(a[f'{prefix}_artifact_digest']), f'{prefix} artifact digest invalid')

    forbidden = set(contract['forbidden_keys_anywhere_in_sanitized_bundle'])
    for key, value in values.items():
        leaked = forbidden & walk_keys(value)
        require(not leaked, f'{key}: forbidden detailed keys leaked: {sorted(leaked)}')

    return {
        'schema': 'symthaea.rel.qualification-input-firewall-receipt.v2',
        'authority': 'QualificationInputFirewallOnly',
        'bundle_valid': True,
        'comparison_result': result,
        'predicate_count': count,
        'passed_count': c['passed_count'],
        'failed_count': c['failed_count'],
        'failed_predicate_ids': ids,
        'detailed_metric_keys_absent': True,
        'manifest_hashes_verified': True,
        'assembly_provenance_verified': True,
        'transport_identity_scientific_authority': False,
        'claims': {
            'qualification_completed': False,
            'rel_005a_qualified': False,
            'scientific_pass': False,
            'scientific_fail': False,
        },
    }


def validate_qualification_output(contract: dict[str, Any], bundle: dict[str, Any], output: dict[str, Any]) -> None:
    require(output.get('schema') == contract['required_schemas']['qualification'], 'qualification schema mismatch')
    require(output.get('authority') == 'QualificationOnly', 'qualification authority mismatch')
    require(output.get('qualification_completed') is True, 'qualification incomplete')
    sp = output.get('scientific_pass') is True; sf = output.get('scientific_fail') is True
    require(sp ^ sf, 'scientific pass/fail must be XOR')
    for key in ('comparison_result', 'predicate_count', 'passed_count', 'failed_count', 'failed_predicate_ids'):
        require(output.get(key) == bundle[key], f'qualification {key} mismatch')
    if sp:
        require(bundle['comparison_result'] == 'ALL_PREDICATES_PASS', 'pass not sourced from all-pass comparison')
        require(output.get('rel_005a_qualified') is True and output.get('failed_count') == 0, 'pass qualification invariant failed')
    else:
        require(bundle['comparison_result'] == 'PREDICATE_FAILURES', 'fail not sourced from predicate failures')
        require(output.get('rel_005a_qualified') is False and output.get('failed_count', 0) > 0, 'fail qualification invariant failed')


def synthetic_bundle(contract: dict[str, Any], root: pathlib.Path, result: str) -> None:
    ids = [] if result == 'ALL_PREDICATES_PASS' else ['REL005A-P041']
    failed = len(ids); observation = '33' * 32; seal_commit = '44' * 32; full = '55' * 32
    p = {
        'schema': contract['required_schemas']['predicate'], 'authority': 'PredicateContractOnly',
        'subject_head': '43bf1d588447f602ce0f1549986bb558a839762c', 'predicate_count': 41,
        'source_grounded': True, 'predicate_contract_static_valid': True,
        'frozen_scientific_subject': '7f5826675b44dd0d3f702f62bc9818f7990e4a01',
        'frozen_test_blob': '787ea051ae0bf8e5667b6925481afd46c15d0bc4', 'claims': contract['required_claims']['predicate'],
    }
    e = {
        'schema': contract['required_schemas']['execution'], 'authority': 'ExecutionOnly',
        'subject_head': '6931639e53060809f9d459196a329c4fe983be7b', 'predicate_contract_parent': p['subject_head'],
        'frozen_scientific_subject': p['frozen_scientific_subject'], 'frozen_test_blob': p['frozen_test_blob'],
        'result': 'EXECUTION_OK', 'measurement_exit_code': '0', 'observation_present': True,
        'observation_sha256': observation, 'claims': contract['required_claims']['execution'],
    }
    s = {
        'schema': contract['required_schemas']['seal'], 'authority': 'ObservationSeal', 'observation_status': 'sealed',
        'adjudication': 'not_run', 'scientific_result': 'not_run', 'execution_subject_head': e['subject_head'],
        'predicate_contract_head': p['subject_head'], 'frozen_scientific_subject': p['frozen_scientific_subject'],
        'frozen_test_blob': p['frozen_test_blob'], 'observation_sha256': observation,
        'chain_commitment_sha256': seal_commit, 'claims': contract['required_claims']['seal'],
    }
    c = {
        'schema': contract['required_schemas']['comparison'], 'authority': 'ComparisonOnly',
        'predicate_contract_head': p['subject_head'], 'execution_subject_head': e['subject_head'],
        'frozen_scientific_subject': p['frozen_scientific_subject'], 'frozen_test_blob': p['frozen_test_blob'],
        'observation_sha256': observation, 'seal_chain_commitment_sha256': seal_commit,
        'predicate_count': 41, 'passed_count': 41 - failed, 'failed_count': failed,
        'failed_predicate_ids': ids, 'comparison_result': result, 'full_comparison_sha256': full,
        'claims': contract['required_claims']['comparison'],
    }
    root.mkdir(parents=True, exist_ok=True)
    for key, value in (('predicate', p), ('execution', e), ('seal', s), ('comparison', c)):
        write(root / FILES[key], value)
    manifest_entries = []
    for key in ('predicate', 'execution', 'seal', 'comparison'):
        path = root / FILES[key]
        manifest_entries.append({'basename': path.name, 'byte_length': path.stat().st_size, 'sha256': sha256(path)})
    m = {
        'schema': contract['required_schemas']['manifest'], 'authority': 'EvidenceProjectionOnly',
        'predicate_contract_head': p['subject_head'], 'execution_subject_head': e['subject_head'],
        'observation_sha256': observation, 'seal_chain_commitment_sha256': seal_commit,
        'comparison_result': result, 'raw_observation_included': False, 'raw_execution_logs_included': False,
        'detailed_predicate_values_included': False, 'files': manifest_entries,
    }
    a = {
        'schema': contract['required_schemas']['assembly'], 'authority': 'QualificationAssemblyOnly',
        'assembly_contract_head': contract['assembly_contract_head'],
        'comparison_subject_head': contract['authoritative_provenance']['comparison_subject_head'],
        'comparison_source_run_id': contract['authoritative_provenance']['comparison_source_run_id'],
        'comparison_source_job_id': 101, 'comparison_artifact_id': 201,
        'comparison_artifact_name': 'rel-005a-comparison-v3-example', 'comparison_artifact_digest': 'sha256:' + 'aa' * 32,
        'projection_contract_head': contract['authoritative_provenance']['projection_contract_head'],
        'projection_contract_run_id': contract['authoritative_provenance']['projection_contract_run_id'],
        'projection_contract_source_job_id': 102, 'projection_contract_artifact_id': 202,
        'projection_contract_artifact_name': 'rel-005a-projection-contract-example', 'projection_contract_artifact_digest': 'sha256:' + 'bb' * 32,
        'projection_contract_receipt_sha256': '66' * 32, 'full_comparison_sha256': full,
        'transport_identity_scientific_authority': False, 'inner_content_commitments_verified': True,
        'claims': contract['required_claims']['assembly'],
    }
    write(root / FILES['manifest'], m); write(root / FILES['assembly'], a)


def self_test(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        pass_root = root / 'pass'; synthetic_bundle(contract, pass_root, 'ALL_PREDICATES_PASS')
        pass_receipt = validate_bundle(contract, pass_root)
        pass_out = {'schema': contract['required_schemas']['qualification'], 'authority': 'QualificationOnly',
                    'qualification_completed': True, 'rel_005a_qualified': True, 'scientific_pass': True, 'scientific_fail': False,
                    'comparison_result': 'ALL_PREDICATES_PASS', 'predicate_count': 41, 'passed_count': 41, 'failed_count': 0, 'failed_predicate_ids': []}
        validate_qualification_output(contract, pass_receipt, pass_out)

        fail_root = root / 'fail'; synthetic_bundle(contract, fail_root, 'PREDICATE_FAILURES')
        fail_receipt = validate_bundle(contract, fail_root)
        fail_out = {'schema': contract['required_schemas']['qualification'], 'authority': 'QualificationOnly',
                    'qualification_completed': True, 'rel_005a_qualified': False, 'scientific_pass': False, 'scientific_fail': True,
                    'comparison_result': 'PREDICATE_FAILURES', 'predicate_count': 41, 'passed_count': 40, 'failed_count': 1, 'failed_predicate_ids': ['REL005A-P041']}
        validate_qualification_output(contract, fail_receipt, fail_out)

        bad = dict(pass_out); bad['scientific_fail'] = True
        try: validate_qualification_output(contract, pass_receipt, bad)
        except ValueError: pass
        else: raise ValueError('contradictory final claims accepted')

        leak_root = root / 'leak'; synthetic_bundle(contract, leak_root, 'PREDICATE_FAILURES')
        comparison_path = leak_root / FILES['comparison']; leaked = load(comparison_path); leaked['observed'] = 918273645; write(comparison_path, leaked)
        manifest = load(leak_root / FILES['manifest'])
        for entry in manifest['files']:
            if entry['basename'] == FILES['comparison']:
                entry['byte_length'] = comparison_path.stat().st_size; entry['sha256'] = sha256(comparison_path)
        write(leak_root / FILES['manifest'], manifest)
        try: validate_bundle(contract, leak_root)
        except ValueError as exc: require('forbidden detailed keys leaked' in str(exc), 'leak rejected for wrong reason')
        else: raise ValueError('metric leakage accepted')

        nested_root = root / 'nested'; synthetic_bundle(contract, nested_root, 'ALL_PREDICATES_PASS')
        (nested_root / 'nested').mkdir(); (nested_root / 'nested' / 'raw-observation.json').write_text('{}\n')
        try: validate_bundle(contract, nested_root)
        except ValueError as exc: require('recursive file census mismatch' in str(exc), 'nested file rejected for wrong reason')
        else: raise ValueError('nested extra accepted')

        duplicate_root = root / 'duplicate'; synthetic_bundle(contract, duplicate_root, 'ALL_PREDICATES_PASS')
        path = duplicate_root / FILES['assembly']; text = path.read_text(); path.write_text(text.replace('"authority": "QualificationAssemblyOnly"', '"authority": "QualificationAssemblyOnly",\n  "authority": "QualificationAssemblyOnly"', 1))
        try: validate_bundle(contract, duplicate_root)
        except ValueError as exc: require('duplicate JSON key' in str(exc), 'duplicate key rejected for wrong reason')
        else: raise ValueError('duplicate JSON key accepted')

    return {
        'schema': 'symthaea.rel.qualification-input-firewall-self-test.v2',
        'authority': 'QualificationInputFirewallContractOnly',
        'valid_pass_bundle': True, 'valid_failure_bundle': True, 'contradictory_claim_rejection': True,
        'rehash_after_metric_leak_rejection': True, 'nested_file_rejection': True, 'duplicate_key_rejection': True,
        'transport_identity_scientific_authority': False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=pathlib.Path, required=True)
    parser.add_argument('--bundle', type=pathlib.Path)
    parser.add_argument('--qualification-output', type=pathlib.Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args(); contract = load(args.contract)
    if args.self_test:
        print(json.dumps(self_test(contract), indent=2, sort_keys=True)); return
    require(args.bundle is not None, '--bundle required')
    receipt = validate_bundle(contract, args.bundle)
    if args.qualification_output is not None:
        validate_qualification_output(contract, receipt, load(args.qualification_output))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
