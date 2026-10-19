#!/bin/sh
# ENG-REA-005 real-provider repeatability AND semantic-sensitivity smoke test.
# Uses only generated gzip -> tar -> text fixtures. Never executes extracted content.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NORMALIZER="$ROOT/normalize.jq"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v unblob >/dev/null 2>&1 || fail "pinned Unblob 26.6.4 is required"
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
command -v tar >/dev/null 2>&1 || fail "tar required"
command -v gzip >/dev/null 2>&1 || fail "gzip required"
command -v find >/dev/null 2>&1 || fail "find required"
command -v cmp >/dev/null 2>&1 || fail "cmp required"
command -v git >/dev/null 2>&1 || fail "git required"
[ -f "$NORMALIZER" ] || fail "semantic normalizer missing"
source_head_sha=$(printenv REA_TEST_SOURCE_SHA 2>/dev/null || git rev-parse HEAD)
checkout_commit_sha=$(git rev-parse HEAD)
workflow_event_sha=${GITHUB_SHA:-unknown}
normalizer_sha=$(sha256sum "$NORMALIZER" | awk '{print $1}')
cases_manifest_sha=$(sha256sum "$ROOT/cases.json" | awk '{print $1}')
oracle_sha=$(sha256sum "$ROOT/oracle.json" | awk '{print $1}')
provider_smoke_script_sha=$(sha256sum "$ROOT/provider-smoke.sh" | awk '{print $1}')
provider_metamorphic_script_sha=$(sha256sum "$ROOT/provider-metamorphic-sweep.sh" | awk '{print $1}')

version=$(unblob --version 2>&1) || fail "cannot read Unblob version"
printf '%s\n' "$version" | grep -Eq '26[.]6[.]4([[:space:]]|$)' || fail "expected Unblob 26.6.4; got: $version"
python_version=$(python3 --version 2>&1) || fail "python3 version unavailable"
tool_path=$(command -v unblob)
tool_sha=$(sha256sum "$tool_path" | awk '{print $1}')
freeze_file="/tmp/rea-unblob-pip-freeze.$$"
python3 -m pip freeze | LC_ALL=C sort > "$freeze_file"
pip_freeze_sha=$(sha256sum "$freeze_file" | awk '{print $1}')

OUT=$(printenv REA_PROVIDER_ARTIFACT_DIR 2>/dev/null || true)
if [ -n "$OUT" ]; then
  [ ! -e "$OUT" ] || fail "artifact directory already exists; refusing overwrite: $OUT"
  mkdir -p "$OUT"
else
  OUT=$(mktemp -d /tmp/rea-determinism-provider.XXXXXX) || fail "mktemp failed"
fi
umask 077
mkdir -p "$OUT/fixture-source" "$OUT/fixture-source-mutated"
printf '%s\n' 'Unblob semantic-projection fixture v1: inert known-answer text.' > "$OUT/fixture-source/payload.txt"
printf '%s\n' 'Unblob semantic-projection fixture v2: inert known-answer text.' > "$OUT/fixture-source-mutated/payload.txt"
tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner --format=ustar \
  -cf "$OUT/nested.tar" -C "$OUT/fixture-source" payload.txt || fail "could not create deterministic baseline tar"
gzip -n -c "$OUT/nested.tar" > "$OUT/fixture.gz" || fail "could not create deterministic baseline gzip fixture"
tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner --format=ustar \
  -cf "$OUT/nested-mutated.tar" -C "$OUT/fixture-source-mutated" payload.txt || fail "could not create deterministic mutated tar"
gzip -n -c "$OUT/nested-mutated.tar" > "$OUT/fixture-mutated.gz" || fail "could not create deterministic mutated gzip fixture"

fixture_sha=$(sha256sum "$OUT/fixture.gz" | awk '{print $1}')
mutated_fixture_sha=$(sha256sum "$OUT/fixture-mutated.gz" | awk '{print $1}')
payload_sha=$(sha256sum "$OUT/fixture-source/payload.txt" | awk '{print $1}')
mutated_payload_sha=$(sha256sum "$OUT/fixture-source-mutated/payload.txt" | awk '{print $1}')
[ "$fixture_sha" != "$mutated_fixture_sha" ] || fail "one-byte-equivalent semantic mutation failed to alter input digest"
[ "$payload_sha" != "$mutated_payload_sha" ] || fail "mutated payload unexpectedly has identical identity"
printf '%s\n' "$version" > "$OUT/unblob-version.txt"
printf '%s\n' "$python_version" > "$OUT/python-version.txt"
printf '%s\n' "$tool_path" > "$OUT/unblob-executable-path.txt"
printf '%s\n' "$tool_sha" > "$OUT/unblob-executable-sha256.txt"
cp "$freeze_file" "$OUT/pip-freeze.txt"
rm -f "$freeze_file"
printf '%s\n' "$pip_freeze_sha" > "$OUT/pip-freeze-sha256.txt"
printf '%s\n' "$fixture_sha" > "$OUT/fixture-input-sha256.txt"
printf '%s\n' "$mutated_fixture_sha" > "$OUT/mutated-fixture-input-sha256.txt"
printf '%s\n' "$payload_sha" > "$OUT/expected-payload-sha256.txt"
printf '%s\n' "$mutated_payload_sha" > "$OUT/mutated-expected-payload-sha256.txt"

for lane in run-a run-b run-mutated; do
  run="$OUT/$lane"
  mkdir "$run"
  case "$lane" in
    run-mutated)
      source_gzip="$OUT/fixture-mutated.gz"
      expected_payload="$OUT/fixture-source-mutated/payload.txt"
      expected_input_sha="$mutated_fixture_sha"
      ;;
    *)
      source_gzip="$OUT/fixture.gz"
      expected_payload="$OUT/fixture-source/payload.txt"
      expected_input_sha="$fixture_sha"
      ;;
  esac
  cp "$source_gzip" "$run/input.bin"
  cp_sha=$(sha256sum "$run/input.bin" | awk '{print $1}')
  [ "$cp_sha" = "$expected_input_sha" ] || fail "$lane input copy changed bytes"
  if ! unblob --report "$run/raw-report.json" --extract-dir "$run/extracted" \
      --process-num 1 --randomness-depth 0 --depth 4 "$run/input.bin" \
      > "$run/stdout.txt" 2> "$run/stderr.txt"; then
    cat "$run/stdout.txt" "$run/stderr.txt" >&2
    fail "Unblob extraction failed in $lane"
  fi
  [ -s "$run/raw-report.json" ] || fail "$lane raw report is absent/empty"
  raw_before=$(sha256sum "$run/raw-report.json" | awk '{print $1}')
  jq --arg root "$run" -cS -f "$NORMALIZER" "$run/raw-report.json" > "$run/semantic-projection.json" \
    || fail "$lane real provider report violates the semantic-projection contract"
  raw_after=$(sha256sum "$run/raw-report.json" | awk '{print $1}')
  [ "$raw_before" = "$raw_after" ] || fail "$lane normalization modified the raw provider report"
  jq -e '.schema_version=="symthaea.unblob-semantic-projection.v1" and (.tasks|length)>=2 and any(.tasks[]; .depth>0) and any(.tasks[].reports[]; .__typename__=="ChunkReport") and any(.tasks[].reports[]; .__typename__=="HashReport")' \
    "$run/semantic-projection.json" >/dev/null || fail "$lane report lacks expected nested extraction/hash evidence"
  found_payload=0
  while IFS= read -r payload_path; do
    [ -n "$payload_path" ] || continue
    if cmp -s "$expected_payload" "$payload_path"; then
      found_payload=1
      break
    fi
  done <<EOF_PAYLOADS
$(find "$run/extracted" -type f -name payload.txt -print 2>/dev/null || true)
EOF_PAYLOADS
  [ "$found_payload" -eq 1 ] || fail "$lane did not recover byte-identical expected payload.txt"
  sha256sum "$run/raw-report.json" > "$run/raw-report.sha256"
  sha256sum "$run/semantic-projection.json" > "$run/semantic-projection.sha256"
  printf 'PASS %s actual Unblob 26.6.4 report normalized; raw report preserved; expected payload recovered byte-identically.\n' "$lane"
done

sem_a=$(awk '{print $1}' "$OUT/run-a/semantic-projection.sha256")
sem_b=$(awk '{print $1}' "$OUT/run-b/semantic-projection.sha256")
sem_mutated=$(awk '{print $1}' "$OUT/run-mutated/semantic-projection.sha256")
[ "$sem_a" = "$sem_b" ] || fail "semantic projection changed across scratch roots / provider-generated IDs"
[ "$sem_a" != "$sem_mutated" ] || fail "semantic projection hid a changed source/payload identity"
raw_a=$(awk '{print $1}' "$OUT/run-a/raw-report.sha256")
raw_b=$(awk '{print $1}' "$OUT/run-b/raw-report.sha256")
raw_mutated=$(awk '{print $1}' "$OUT/run-mutated/raw-report.sha256")
provider_metamorphic_receipt="$OUT/provider-metamorphic-receipt.json"
if ! sh "$ROOT/provider-metamorphic-sweep.sh" "$OUT/run-a/raw-report.json" "$OUT/run-a" "$sem_a" "$provider_metamorphic_receipt" > "$OUT/provider-metamorphic.log" 2>&1; then
  cat "$OUT/provider-metamorphic.log" >&2
  fail "actual-provider metamorphic sweep failed"
fi
cat "$OUT/provider-metamorphic.log"
provider_metamorphic_sha=$(sha256sum "$provider_metamorphic_receipt" | awk '{print $1}')
provider_metamorphic_count=$(jq -er '.equivalent_variants' "$provider_metamorphic_receipt")
provider_metamorphic_sem=$(jq -er '.semantic_projection_sha256' "$provider_metamorphic_receipt")
[ "$provider_metamorphic_count" -eq 64 ] || fail "actual-provider metamorphic sweep did not execute 64 variants"
[ "$provider_metamorphic_sem" = "$sem_a" ] || fail "actual-provider metamorphic digest diverged from run-a"

manifest_tmp=$(mktemp "${TMPDIR:-/tmp}/rea-evidence-bundle-manifest.XXXXXX") || fail "could not allocate bundle manifest staging file"
(
  cd "$OUT"
  find . -type f ! -path './receipt.json' ! -path './bundle-manifest.json' -print |
    LC_ALL=C sort |
    while IFS= read -r file; do
      relative=${file#./}
      digest=$(sha256sum "$file" | awk '{print $1}')
      printf '%s\t%s\n' "$relative" "$digest"
    done
) > "$manifest_tmp" || fail "could not enumerate bundle manifest inputs"
jq -R -s 'split("\n") | map(select(length > 0) | split("\t") | {path: .[0], sha256: .[1]}) | {schema_version:"symthaea.rea-evidence-bundle-manifest.v1",files:.}' "$manifest_tmp" > "$OUT/bundle-manifest.json"   || fail "could not create bundle manifest"
rm -f "$manifest_tmp"
bundle_manifest_sha=$(sha256sum "$OUT/bundle-manifest.json" | awk '{print $1}')

jq -n \
  --arg benchmark "ENG-REA-005-REAL-PROVIDER" \
  --arg provider_version "$version" \
  --arg source_head_sha "$source_head_sha" \
  --arg checkout_commit_sha "$checkout_commit_sha" \
  --arg workflow_event_sha "$workflow_event_sha" \
  --arg normalizer_sha256 "$normalizer_sha" \
  --arg cases_manifest_sha256 "$cases_manifest_sha" \
  --arg oracle_sha256 "$oracle_sha" \
  --arg bundle_manifest_sha256 "$bundle_manifest_sha" \
  --arg provider_smoke_script_sha256 "$provider_smoke_script_sha" \
  --arg provider_metamorphic_script_sha256 "$provider_metamorphic_script_sha" \
  --arg executable_path "$tool_path" \
  --arg executable_sha256 "$tool_sha" \
  --arg python_version "$python_version" \
  --arg pip_freeze_sha256 "$pip_freeze_sha" \
  --arg fixture_sha256 "$fixture_sha" \
  --arg mutated_fixture_sha256 "$mutated_fixture_sha" \
  --arg expected_payload_sha256 "$payload_sha" \
  --arg mutated_payload_sha256 "$mutated_payload_sha" \
  --arg raw_report_sha256_run_a "$raw_a" \
  --arg raw_report_sha256_run_b "$raw_b" \
  --arg raw_report_sha256_mutated "$raw_mutated" \
  --arg semantic_projection_sha256 "$sem_a" \
  --arg mutated_semantic_projection_sha256 "$sem_mutated" \
  --arg provider_metamorphic_receipt_sha256 "$provider_metamorphic_sha" \
  --argjson provider_metamorphic_variant_count "$provider_metamorphic_count" \
  --arg provider_metamorphic_semantic_sha256 "$provider_metamorphic_sem" \
  '{schema_version:"symthaea.rea-provider-smoke-receipt.v1",benchmark:$benchmark,qualification:{tier:"unblob-provider-smoke",rea_cli_mcp_parity:false,firmware_execution:false,emulation:false,device_qualification:false},source:{head_sha:$source_head_sha,checkout_commit_sha:$checkout_commit_sha,workflow_event_sha:$workflow_event_sha,normalizer_sha256:$normalizer_sha256,cases_manifest_sha256:$cases_manifest_sha256,oracle_sha256:$oracle_sha256,provider_smoke_script_sha256:$provider_smoke_script_sha256,provider_metamorphic_script_sha256:$provider_metamorphic_script_sha256},provider_version:$provider_version,executable_path:$executable_path,executable_sha256:$executable_sha256,python_version:$python_version,pip_freeze_sha256:$pip_freeze_sha256,fixture_input_sha256:$fixture_sha256,mutated_fixture_input_sha256:$mutated_fixture_sha256,expected_payload_sha256:$expected_payload_sha256,mutated_expected_payload_sha256:$mutated_payload_sha256,raw_report_sha256:{run_a:$raw_report_sha256_run_a,run_b:$raw_report_sha256_run_b,run_mutated:$raw_report_sha256_mutated},semantic_projection_sha256:{equivalent_runs:$semantic_projection_sha256,semantic_mutation:$mutated_semantic_projection_sha256},provider_metamorphic:{receipt_sha256:$provider_metamorphic_receipt_sha256,equivalent_variants:$provider_metamorphic_variant_count,semantic_projection_sha256:$provider_metamorphic_semantic_sha256},bundle_manifest_sha256:$bundle_manifest_sha256,claims_ceiling:"Pinned Unblob CLI on owned inert nested gzip/tar/text fixtures only; no REA, recovered firmware execution, emulator, or hardware qualification."}' > "$OUT/receipt.json"

jq -e --arg source "$source_head_sha" --arg checkout "$checkout_commit_sha" --arg normalizer "$normalizer_sha" --arg cases "$cases_manifest_sha" --arg oracle "$oracle_sha" --arg provider_script "$provider_smoke_script_sha" --arg provider_meta_script "$provider_metamorphic_script_sha" --arg raw_a "$raw_a" --arg raw_b "$raw_b" --arg raw_mutated "$raw_mutated" --arg sem "$sem_a" --arg sem_mutated "$sem_mutated" --arg provider_meta_sha "$provider_metamorphic_sha" --arg provider_meta_sem "$provider_metamorphic_sem" --arg bundle_manifest_sha "$bundle_manifest_sha" '
  .schema_version=="symthaea.rea-provider-smoke-receipt.v1" and
  .benchmark=="ENG-REA-005-REAL-PROVIDER" and
  .qualification.tier=="unblob-provider-smoke" and .qualification.rea_cli_mcp_parity==false and
  .qualification.firmware_execution==false and .qualification.emulation==false and .qualification.device_qualification==false and
  .bundle_manifest_sha256==$bundle_manifest_sha and
  .source.head_sha==$source and .source.checkout_commit_sha==$checkout and
  .source.normalizer_sha256==$normalizer and .source.cases_manifest_sha256==$cases and
  .source.oracle_sha256==$oracle and .source.provider_smoke_script_sha256==$provider_script and
  .source.provider_metamorphic_script_sha256==$provider_meta_script and
  .raw_report_sha256.run_a==$raw_a and .raw_report_sha256.run_b==$raw_b and .raw_report_sha256.run_mutated==$raw_mutated and
  .semantic_projection_sha256.equivalent_runs==$sem and .semantic_projection_sha256.semantic_mutation==$sem_mutated and
  .provider_metamorphic.receipt_sha256==$provider_meta_sha and .provider_metamorphic.semantic_projection_sha256==$provider_meta_sem and .provider_metamorphic.equivalent_variants==64
' "$OUT/receipt.json" >/dev/null || fail "provider receipt failed source/report/projection digest consistency validation"
printf '\nPASS ENG-REA-005 receipt binding: source head, checkout commit, normalizer, manifest, oracle, raw reports, projections and provider metamorphic receipt match.\n'
printf '\nPASS ENG-REA-005 real-provider smoke: 2/2 equivalent runs preserved semantic identity; 1/1 semantic mutation changed the projection; raw reports stayed unchanged.\n'
printf 'Evidence directory: %s\n' "$OUT"
printf 'Receipt: %s/receipt.json\n' "$OUT"
printf 'Raw report SHA-256 run A: %s\nRaw report SHA-256 run B: %s\nRaw report SHA-256 mutated: %s\n' "$raw_a" "$raw_b" "$raw_mutated"
printf 'Semantic projection SHA-256 equivalent: %s\nSemantic projection SHA-256 mutated: %s\n' "$sem_a" "$sem_mutated"
