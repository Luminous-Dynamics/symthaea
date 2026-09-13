#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE41="89243a3072a5ff510c8d6d7ea4958644be40f879"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare41_base":"%s","real_external_preregistration_established":false,"runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE41"
}

if ! git merge-base --is-ancestor "$BASE_WCARE41" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare41_base_not_ancestor"
  exit 4
fi

check_blob() {
  local path="$1" expected="$2" actual
  actual="$(git hash-object "$path" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    emit "INVALID_PROTOCOL" "blob_mismatch:$path:$actual"
    exit 4
  fi
}

check_blob "docs/release/evidence/WCARE43_RFC3161_PREREGISTRATION_PROTOCOL_V1.md" "76ff1774ee510c096649b82cfc3857deaf921e12"
check_blob "docs/release/evidence/WCARE43_RFC3161_BACKEND_POLICY_SCHEMA_V1.json" "2e04e221a5d89bd2da3ea4244649898b66e47a74"
check_blob "docs/release/evidence/WCARE43_RFC3161_RESULT_SCHEMA_V1.json" "7d67eec8adc535d37b4a62c748ad24d3b02d30d0"
check_blob "scripts/wcare43_rfc3161_verify.py" "e89b66a24cb51bed589911b082ea6fef5ffd369b"
check_blob "scripts/wcare43_selftest.py" "7906ce87f7b121464b42dbf35584ddbc9a50ae4d"
check_blob "docs/release/evidence/fixtures/wcare43/a.final.json" "8cec0486532180e6f4fe0f58920c1f318b5154ac"
check_blob "docs/release/evidence/fixtures/wcare43/b.final.json" "ba7cce8fbfece6812fa44816942ed66d3ed7e55f"
check_blob "docs/release/evidence/fixtures/wcare43/fixture_manifest.json" "ec601cc591365dd6f9026894c477ae5a7180c97a"
check_blob "docs/release/evidence/fixtures/wcare43/synthetic_plan.json" "a6569381df8e320c286833a9830075376482206a"
check_blob "docs/release/evidence/fixtures/wcare43/synthetic_response.tsr.b64" "0add55553bbe1927635b747b184523b0e87e576f"
check_blob "docs/release/evidence/fixtures/wcare43/synthetic_root.pem" "901a385c49aa56a5606324319a38956cba414fc1"
check_blob "docs/release/evidence/fixtures/wcare43/synthetic_tsa.pem" "d4ca7537a5baebe978e8cc3744763716a5bb800e"
check_blob "docs/release/evidence/fixtures/wcare43/synthetic_wcare40_result.json" "8a48ce8597167d4de32b67806217e6a0525267cd"

if find docs/release/evidence/fixtures/wcare43 -type f \( -name '*.key' -o -name '*private*' -o -name '*.p12' -o -name '*.pfx' \) | grep -q .; then
  emit "INVALID_PROTOCOL" "synthetic_private_key_material_present"
  exit 4
fi

if ! python3 - <<'PY'
from pathlib import Path
import base64, hashlib, json
for path in (
    Path('docs/release/evidence/WCARE43_RFC3161_BACKEND_POLICY_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE43_RFC3161_RESULT_SCHEMA_V1.json'),
):
    value = json.loads(path.read_text())
    assert isinstance(value, dict) and value.get('additionalProperties') is False
for path in (
    Path('scripts/wcare43_rfc3161_verify.py'),
    Path('scripts/wcare43_selftest.py'),
):
    compile(path.read_text(), str(path), 'exec')
fix = Path('docs/release/evidence/fixtures/wcare43')
manifest = json.loads((fix / 'fixture_manifest.json').read_text())
assert manifest['synthetic'] is True
assert manifest['private_key_material_in_repository'] is False
assert manifest['external_temporal_authority_established'] is False
response = base64.b64decode((fix / 'synthetic_response.tsr.b64').read_text().strip(), validate=True)
assert hashlib.sha256(response).hexdigest() == manifest['timestamp_response_der_sha256']
for filename, field in (
    ('synthetic_plan.json', 'wcare40_plan_sha256'),
    ('synthetic_wcare40_result.json', 'wcare40_result_sha256'),
    ('synthetic_root.pem', 'trust_anchor_pem_sha256'),
    ('synthetic_tsa.pem', 'untrusted_tsa_pem_sha256'),
):
    assert hashlib.sha256((fix / filename).read_bytes()).hexdigest() == manifest[field]
PY
then
  emit "INVALID_PROTOCOL" "schema_python_or_fixture_integrity_failed"
  exit 4
fi

if ! python3 scripts/wcare43_selftest.py >/tmp/wcare43-selftest.json 2>/tmp/wcare43-selftest.stderr; then
  emit "INVALID_PROTOCOL" "rfc3161_adversarial_campaign_failed"
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE43_SELFTEST"' \
  '"real_external_preregistration_established":false' \
  '"synthetic_valid_early_token_verified":true' \
  '"wrong_plan_rejected":true' \
  '"modified_token_rejected":true' \
  '"untrusted_tsa_identity_not_established":true' \
  '"equal_time_not_established":true' \
  '"malformed_response_rejected":true'
do
  if ! grep -q "$marker" /tmp/wcare43-selftest.json; then
    emit "INVALID_PROTOCOL" "selftest_missing_marker:$marker"
    exit 4
  fi
done

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare43_bytes_and_rfc3161_adversarial_campaign_match"
exit 0
