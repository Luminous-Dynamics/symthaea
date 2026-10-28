#!/usr/bin/env bash
set -euo pipefail

# Verify the public synthetic fixture against digests reviewed into the contract.
# These constants intentionally duplicate task.json: changing both requires a
# reviewed source change rather than silently blessing a rewritten fixture.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="$ROOT/tools/vleb/examples/ledger-reconciliation/task.json"
MUTATIONS="$ROOT/tools/vleb/mutation-catalog.json"

INPUT_SHA256="2f976927b50b516fa6d00a55ee5a7237ee5fa0de39fc76965307e9c960125715"
EXPECTED_SHA256="6b0b77b40e1a152da26f82ba98c7ce7b367f1756948aa8ef950097cf75672f10"
NEGATIVE_SHA256="ac3f29ecb70c7d242d2d719f323662daddb7bfc26a26f442ffe4a531522a86e7"
COBOL_FIXTURE_SHA256="aa90e5029eb8f71d453fd8244d81eea76865e9d12af5d80f883a6dfc327e026d"
COBOL_INDEX_SCHEMA_SHA256="98405cb57bca65b66cf05e76817da4ea18afdaf6a34e2fae6d5399ef2f93b420"

cd "$ROOT"

printf '%s  %s\n' \
  "$INPUT_SHA256" "tools/vleb/examples/ledger-reconciliation/input.csv" \
  "$EXPECTED_SHA256" "tools/vleb/examples/ledger-reconciliation/expected.json" \
  "$NEGATIVE_SHA256" "tools/vleb/examples/ledger-reconciliation/conflicting-duplicate.csv" \
  "$COBOL_FIXTURE_SHA256" "tools/vleb/cobol-index/fixtures/settlement.cbl" \
  "$COBOL_INDEX_SCHEMA_SHA256" "tools/vleb/cobol-index/index-output.schema.json" \
  | sha256sum --check --strict -

# Ensure the task contract agrees with the independently pinned digests.
jq -e \
  --arg input "$INPUT_SHA256" \
  --arg expected "$EXPECTED_SHA256" \
  --arg negative "$NEGATIVE_SHA256" '
    .schema_version == "vleb-task-v1"
    and .task_id == "vleb-ledger-reconcile-0001"
    and .source_identity.source_sha256 == $input
    and .source_identity.fixture_sha256.input_csv == $input
    and .source_identity.fixture_sha256.expected_output_json == $expected
    and .source_identity.fixture_sha256.negative_control_csv == $negative
    and .oracle_contract.baseline_empty_patch_must_fail == true
    and .oracle_contract.reference_patch_must_pass == true
  ' "$TASK" >/dev/null

# JSON parsing alone is not admission. This checks cross-file consistency:
# unique mutation IDs and only preregistered test IDs in mutation entries.
jq -e --slurpfile task "$TASK" '
  .schema_version == "vleb-mutation-catalog-v1"
  and .task_id == $task[0].task_id
  and (.mutations | length > 0)
  and (
    [.mutations[].mutation_id] as $ids
    | ($ids | length) == ($ids | unique | length)
  )
  and (
    ([.mutations[].required_test_ids[]] - $task[0].oracle_contract.required_test_ids)
    | length == 0
  )
' "$MUTATIONS" >/dev/null

# Fail on malformed JSON before the fixture can be used by downstream tooling.
jq empty \
  "$TASK" \
  "$MUTATIONS" \
  "$ROOT/tools/vleb/task-contract.schema.json" \
  "$ROOT/tools/vleb/evidence-receipt.schema.json" \
  "$ROOT/tools/vleb/cobol-index/index-output.schema.json" \
  "$ROOT/tools/vleb/legacy-claim-evidence.schema.json" \
  "$ROOT/tools/vleb/examples/ledger-reconciliation/expected.json"

printf '%s\n' "VLEB fixture integrity checks passed."
