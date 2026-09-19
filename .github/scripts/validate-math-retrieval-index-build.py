from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
RECEIPT_VERSION = 'math-retrieval-index-build-receipt-v1'
ARTIFACT_VERSION = 'math-retrieval-exact-index-artifact-v1'
REPORT_VERSION = 'math-retrieval-index-build-validation-report-v1'
AUTHORITY = 'MeasurementOnly'
RECEIPT_FIELDS = {'version', 'receipt_id', 'authority', 'coverage_sha256', 'candidate_set_sha256', 'index_manifest_sha256', 'index_artifact_sha256', 'target_id', 'index_build_policy_sha256', 'builder_implementation_sha256', 'toolchain_manifest_sha256', 'index_seed', 'input_set_sha256', 'build_mode'}
ARTIFACT_FIELDS = {'version', 'index_id', 'authority', 'candidate_set_sha256', 'target_id', 'representation_sha256', 'item_serialization_sha256', 'payload_encoding', 'item_order', 'item_count', 'items'}
ITEM_FIELDS = {'source_object_sha256', 'representation_object_sha256', 'serialized_bytes', 'payload_base64'}

class ValidationError(ValueError):
    pass

def closed(obj: object, fields: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError(f'{where}: object required')
    extra = set(obj) - fields
    missing = fields - set(obj)
    if extra:
        raise ValidationError(f'{where}: unknown fields {sorted(extra)}')
    if missing:
        raise ValidationError(f'{where}: missing fields {sorted(missing)}')
    return obj

def text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f'{where}: non-empty string required')
    return value

def sha(value: object, where: str) -> str:
    value = text(value, where)
    if len(value) != 71 or not value.startswith('sha256:'):
        raise ValidationError(f'{where}: sha256:<64 lowercase hex> required')
    if any((ch not in '0123456789abcdef' for ch in value[7:])):
        raise ValidationError(f'{where}: invalid SHA-256')
    return value

def nonnegative_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f'{where}: non-negative integer required')
    return value

def positive_int(value: object, where: str) -> int:
    value = nonnegative_int(value, where)
    if value < 1:
        raise ValidationError(f'{where}: positive integer required')
    return value

def digest_bytes(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()

def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')

def load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f'cannot load validator: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def input_set_sha(items: list[dict]) -> str:
    normalized = [{'source_object_sha256': item['source_object_sha256'], 'representation_object_sha256': item['representation_object_sha256'], 'serialized_bytes': item['serialized_bytes']} for item in items]
    return digest_bytes(canonical_bytes(normalized))

def validate_receipt(doc: object) -> dict:
    receipt = closed(doc, RECEIPT_FIELDS, 'receipt')
    if receipt['version'] != RECEIPT_VERSION or receipt['authority'] != AUTHORITY:
        raise ValidationError('receipt: version/authority invariant failed')
    text(receipt['receipt_id'], 'receipt.receipt_id')
    for field in ('coverage_sha256', 'candidate_set_sha256', 'index_manifest_sha256', 'index_artifact_sha256', 'index_build_policy_sha256', 'builder_implementation_sha256', 'toolchain_manifest_sha256', 'input_set_sha256'):
        sha(receipt[field], f'receipt.{field}')
    text(receipt['target_id'], 'receipt.target_id')
    nonnegative_int(receipt['index_seed'], 'receipt.index_seed')
    if receipt['build_mode'] != 'ExactDeterministicMaterializedScan':
        raise ValidationError('receipt.build_mode: ExactDeterministicMaterializedScan required')
    return receipt

def validate_artifact(doc: object) -> tuple[dict, list[dict]]:
    artifact = closed(doc, ARTIFACT_FIELDS, 'artifact')
    if artifact['version'] != ARTIFACT_VERSION or artifact['authority'] != AUTHORITY:
        raise ValidationError('artifact: version/authority invariant failed')
    text(artifact['index_id'], 'artifact.index_id')
    sha(artifact['candidate_set_sha256'], 'artifact.candidate_set_sha256')
    text(artifact['target_id'], 'artifact.target_id')
    sha(artifact['representation_sha256'], 'artifact.representation_sha256')
    sha(artifact['item_serialization_sha256'], 'artifact.item_serialization_sha256')
    if artifact['payload_encoding'] != 'Base64':
        raise ValidationError('artifact.payload_encoding: Base64 required')
    if artifact['item_order'] != 'SourceObjectDigestAscending':
        raise ValidationError('artifact.item_order: SourceObjectDigestAscending required')
    item_count = positive_int(artifact['item_count'], 'artifact.item_count')
    items = artifact['items']
    if not isinstance(items, list) or len(items) != item_count:
        raise ValidationError('artifact.items: len must equal item_count')
    source_ids: list[str] = []
    for i, raw in enumerate(items):
        item = closed(raw, ITEM_FIELDS, f'artifact.items[{i}]')
        source = sha(item['source_object_sha256'], f'artifact.items[{i}].source_object_sha256')
        rep_sha = sha(item['representation_object_sha256'], f'artifact.items[{i}].representation_object_sha256')
        size = positive_int(item['serialized_bytes'], f'artifact.items[{i}].serialized_bytes')
        payload_text = text(item['payload_base64'], f'artifact.items[{i}].payload_base64')
        try:
            payload = base64.b64decode(payload_text, validate=True)
        except Exception as exc:
            raise ValidationError(f'artifact.items[{i}].payload_base64: invalid base64') from exc
        if len(payload) != size:
            raise ValidationError(f'artifact.items[{i}]: decoded byte length mismatch')
        if digest_bytes(payload) != rep_sha:
            raise ValidationError(f'artifact.items[{i}]: payload digest differs from representation_object_sha256')
        try:
            payload.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValidationError(f'artifact.items[{i}]: canonical representation bytes must be UTF-8') from exc
        source_ids.append(source)
    if source_ids != sorted(source_ids):
        raise ValidationError('artifact.items: SourceObjectDigestAscending required')
    if len(source_ids) != len(set(source_ids)):
        raise ValidationError('artifact.items: duplicate source identity')
    return (artifact, items)

def validate_bound(receipt_doc: object, receipt_raw: bytes, coverage_doc: object, coverage_raw: bytes, candidate_doc: object, candidate_raw: bytes, experiment_doc: object, index_doc: object, index_raw: bytes, artifact_doc: object, artifact_raw: bytes) -> dict:
    receipt = validate_receipt(receipt_doc)
    artifact, items = validate_artifact(artifact_doc)
    coverage_mod = load_sibling('validate-math-retrieval-source-coverage.py', 'sym_index_build_coverage_validator')
    coverage_report = coverage_mod.validate_bound(coverage_doc, coverage_raw, candidate_doc, candidate_raw, experiment_doc)
    coverage_sha = digest_bytes(coverage_raw)
    if coverage_sha != receipt['coverage_sha256']:
        raise ValidationError('receipt.coverage_sha256 differs from exact coverage bytes')
    index_mod = load_sibling('validate-math-retrieval-index.py', 'sym_index_build_index_validator')
    index_mod.validate(index_doc)
    manifest_sha = digest_bytes(index_raw)
    if manifest_sha != receipt['index_manifest_sha256']:
        raise ValidationError('receipt.index_manifest_sha256 differs from exact manifest bytes')
    artifact_sha = digest_bytes(artifact_raw)
    if artifact_sha != receipt['index_artifact_sha256']:
        raise ValidationError('receipt.index_artifact_sha256 differs from exact artifact bytes')
    if index_doc['index']['index_artifact_sha256'] != artifact_sha:
        raise ValidationError('index manifest does not bind exact artifact bytes')
    target_id = receipt['target_id']
    targets = {target['target_id']: target for target in coverage_doc['targets']}
    if target_id not in targets:
        raise ValidationError('receipt target_id is absent from coverage targets')
    target = targets[target_id]
    if artifact['target_id'] != target_id:
        raise ValidationError('artifact target_id differs from receipt')
    if artifact['index_id'] != index_doc['index_id']:
        raise ValidationError('artifact index_id differs from index manifest')
    if artifact['candidate_set_sha256'] != receipt['candidate_set_sha256']:
        raise ValidationError('artifact candidate set differs from receipt')
    if receipt['candidate_set_sha256'] != coverage_report['candidate_set_sha256']:
        raise ValidationError('receipt candidate set differs from qualified coverage')
    if index_doc['candidate_universe']['candidate_set_sha256'] != receipt['candidate_set_sha256']:
        raise ValidationError('index manifest candidate set differs from receipt')
    if index_doc['candidate_universe']['candidate_count'] != artifact['item_count']:
        raise ValidationError('artifact item count differs from index candidate count')
    rep = index_doc['representation']
    for field in ('channel', 'representation_family', 'representation_sha256', 'item_serialization_sha256', 'max_serialized_item_bytes'):
        if rep[field] != target[field]:
            raise ValidationError(f'index representation {field} differs from coverage target')
    if artifact['representation_sha256'] != target['representation_sha256']:
        raise ValidationError('artifact representation identity differs from coverage target')
    if artifact['item_serialization_sha256'] != target['item_serialization_sha256']:
        raise ValidationError('artifact serialization identity differs from coverage target')
    if index_doc['index']['search_mode'] != 'ExactDeterministic':
        raise ValidationError('transparent exact artifact requires ExactDeterministic search mode')
    if index_doc['index']['index_build_policy_sha256'] != receipt['index_build_policy_sha256']:
        raise ValidationError('receipt build policy differs from index manifest')
    if index_doc['index']['index_seed'] != receipt['index_seed']:
        raise ValidationError('receipt index seed differs from index manifest')
    coverage_rows = {row['source_object_sha256']: row for row in coverage_doc['rows']}
    candidate_sources = candidate_doc['candidates']
    if [item['source_object_sha256'] for item in items] != candidate_sources:
        raise ValidationError('artifact source order/population differs from frozen candidate set')
    for i, item in enumerate(items):
        source = item['source_object_sha256']
        row = coverage_rows[source]
        representations = {r['target_id']: r for r in row['representations']}
        covered = representations[target_id]
        if item['representation_object_sha256'] != covered['representation_object_sha256']:
            raise ValidationError(f'artifact.items[{i}]: representation digest differs from coverage')
        if item['serialized_bytes'] != covered['serialized_bytes']:
            raise ValidationError(f'artifact.items[{i}]: serialized byte count differs from coverage')
        if item['serialized_bytes'] > target['max_serialized_item_bytes']:
            raise ValidationError(f'artifact.items[{i}]: exceeds frozen target byte ceiling')
    computed_input_set_sha = input_set_sha(items)
    if computed_input_set_sha != receipt['input_set_sha256']:
        raise ValidationError('receipt input_set_sha256 differs from exact artifact item identities')
    return {'version': REPORT_VERSION, 'authority': AUTHORITY, 'receipt_sha256': digest_bytes(receipt_raw), 'coverage_sha256': coverage_sha, 'candidate_set_sha256': receipt['candidate_set_sha256'], 'index_manifest_sha256': manifest_sha, 'index_artifact_sha256': artifact_sha, 'input_set_sha256': computed_input_set_sha, 'item_count': artifact['item_count'], 'target_id': target_id, 'all_payload_hashes_match': True, 'all_coverage_rows_match': True, 'all_checks_passed': True}

def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ('receipt', 'coverage', 'candidate_set', 'experiment', 'index_manifest', 'index_artifact'):
        ap.add_argument(n, type=Path)
    ap.add_argument('--report', type=Path)
    a = ap.parse_args()
    try:
        raws = [getattr(a, n).read_bytes() for n in ('receipt', 'coverage', 'candidate_set', 'experiment', 'index_manifest', 'index_artifact')]
        rr, cr, car, er, ir, ar = raws
        report = validate_bound(json.loads(rr.decode()), rr, json.loads(cr.decode()), cr, json.loads(car.decode()), car, json.loads(er.decode()), json.loads(ir.decode()), ir, json.loads(ar.decode()), ar)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f'INVALID: {e}', file=sys.stderr)
        return 1
    out = json.dumps(report, sort_keys=True, separators=(',', ':'))
    if a.report:
        a.report.write_text(out + '\n', encoding='utf-8')
    print(out)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
