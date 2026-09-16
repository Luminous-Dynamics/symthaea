#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib

SCHEMA = 'symthaea.assurance.linux-ima-rust196-qualification-recipe.v3'


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f'blob {len(data)}\0'.encode('ascii') + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--repository-root', type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA:
        raise SystemExit('unexpected recipe schema')
    payload = manifest.get('payload')
    if not isinstance(payload, dict):
        raise SystemExit('recipe payload must be object')
    expected_id = 'sha256:' + hashlib.sha256(canonical(payload)).hexdigest()
    if manifest.get('recipe_id') != expected_id:
        raise SystemExit(f"recipe_id mismatch: expected {expected_id}, got {manifest.get('recipe_id')!r}")
    if payload.get('authority') != 'QualificationRecipeOnly':
        raise SystemExit('recipe authority drift')
    if payload.get('qualification_result') != 'NOT_ESTABLISHED':
        raise SystemExit('recipe qualification_result drift')

    files = payload.get('recipe_files')
    if not isinstance(files, dict) or not files:
        raise SystemExit('missing recipe_files')
    for role, item in files.items():
        if not isinstance(item, dict):
            raise SystemExit(f'invalid recipe file entry: {role}')
        data = (args.repository_root / item['path']).read_bytes()
        actual = git_blob_sha1(data)
        if actual != item['git_blob_sha1']:
            raise SystemExit(
                f"{role} blob mismatch: expected {item['git_blob_sha1']}, got {actual}"
            )

    required_phases = [
        'product_identity',
        'git_object_capsule',
        'rustfmt',
        'capsule_lock_reconciliation_non_authoritative',
        'locked_dependency_fetch',
        'offline_metadata',
        'offline_check',
        'offline_test_census',
        'offline_golden_replay',
        'offline_full_tests',
        'offline_strict_clippy_all_targets',
        'qualification_receipt',
    ]
    if payload.get('phase_order') != required_phases:
        raise SystemExit(f"phase_order mismatch: {payload.get('phase_order')!r}")

    boundary = payload.get('authority_boundary')
    if not isinstance(boundary, dict) or boundary.get('authoritative_cargo_network') != 'OFFLINE':
        raise SystemExit('authoritative Cargo network must be OFFLINE')

    print(json.dumps({
        'schema': 'symthaea.assurance.linux-ima-rust196-qualification-recipe-verification.v1',
        'verification': 'PASS',
        'authority': 'VerificationOnly',
        'recipe_id': manifest['recipe_id'],
        'qualification_result': 'NOT_ESTABLISHED',
    }, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
