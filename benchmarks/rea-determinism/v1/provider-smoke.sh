#!/bin/sh
# ENG-REA-005 real-provider reproducibility smoke test.
# Uses only a locally generated gzip -> tar -> text fixture. Never executes the extracted file.
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
[ -f "$NORMALIZER" ] || fail "semantic normalizer missing"

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
mkdir -p "$OUT/fixture-source"
printf '%s\n' 'Unblob semantic-projection fixture v1: inert known-answer text.' > "$OUT/fixture-source/payload.txt"
tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner --format=ustar \
  -cf "$OUT/nested.tar" -C "$OUT/fixture-source" payload.txt || fail "could not create deterministic inner tar"
gzip -n -c "$OUT/nested.tar" > "$OUT/fixture.gz" || fail "could not create deterministic gzip fixture"
fixture_sha=$(sha256sum "$OUT/fixture.gz" | awk '{print $1}')
payload_sha=$(sha256sum "$OUT/fixture-source/payload.txt" | awk '{print $1}')
printf '%s\n' "$version" > "$OUT/unblob-version.txt"
printf '%s\n' "$python_version" > "$OUT/python-version.txt"
printf '%s\n' "$tool_path" > "$OUT/unblob-executable-path.txt"
printf '%s\n' "$tool_sha" > "$OUT/unblob-executable-sha256.txt"
cp "$freeze_file" "$OUT/pip-freeze.txt"
rm -f "$freeze_file"
printf '%s\n' "$pip_freeze_sha" > "$OUT/pip-freeze-sha256.txt"
printf '%s\n' "$fixture_sha" > "$OUT/fixture-input-sha256.txt"
printf '%s\n' "$payload_sha" > "$OUT/expected-payload-sha256.txt"

for lane in run-a run-b; do
  run="$OUT/$lane"
  mkdir "$run"
  cp "$OUT/fixture.gz" "$run/input.bin"
  cp_sha=$(sha256sum "$run/input.bin" | awk '{print $1}')
  [ "$cp_sha" = "$fixture_sha" ] || fail "$lane input copy changed bytes"
  if ! unblob --report "$run/raw-report.json" --extract-dir "$run/extracted" \
      --process-num 1 --entropy-depth 0 --depth 4 "$run/input.bin" \
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
  jq -e '.schema_version=="symthaea.unblob-semantic-projection.v1" and (.tasks|length)>=2 and any(.tasks[]; .depth>0) and any(.tasks[].reports[]; .__typename__=="ChunkReport")' \
    "$run/semantic-projection.json" >/dev/null || fail "$lane report lacks the expected nested extraction evidence"
  found_payload=0
  while IFS= read -r payload_path; do
    [ -n "$payload_path" ] || continue
    if cmp -s "$OUT/fixture-source/payload.txt" "$payload_path"; then
      found_payload=1
      break
    fi
  done <<EOF_PAYLOADS
$(find "$run/extracted" -type f -name payload.txt -print 2>/dev/null || true)
EOF_PAYLOADS
  [ "$found_payload" -eq 1 ] || fail "$lane did not recover byte-identical inert payload.txt"
  sha256sum "$run/raw-report.json" > "$run/raw-report.sha256"
  sha256sum "$run/semantic-projection.json" > "$run/semantic-projection.sha256"
  printf 'PASS %s actual Unblob 26.6.4 report normalized; raw report preserved; inert payload recovered byte-identically.\n' "$lane"
done

sem_a=$(awk '{print $1}' "$OUT/run-a/semantic-projection.sha256")
sem_b=$(awk '{print $1}' "$OUT/run-b/semantic-projection.sha256")
[ "$sem_a" = "$sem_b" ] || fail "semantic projection changed across scratch roots / provider-generated IDs"
raw_a=$(awk '{print $1}' "$OUT/run-a/raw-report.sha256")
raw_b=$(awk '{print $1}' "$OUT/run-b/raw-report.sha256")
jq -n \
  --arg benchmark "ENG-REA-005-REAL-PROVIDER" \
  --arg provider_version "$version" \
  --arg executable_path "$tool_path" \
  --arg executable_sha256 "$tool_sha" \
  --arg python_version "$python_version" \
  --arg pip_freeze_sha256 "$pip_freeze_sha" \
  --arg fixture_sha256 "$fixture_sha" \
  --arg expected_payload_sha256 "$payload_sha" \
  --arg raw_report_sha256_run_a "$raw_a" \
  --arg raw_report_sha256_run_b "$raw_b" \
  --arg semantic_projection_sha256 "$sem_a" \
  '{benchmark:$benchmark,provider_version:$provider_version,executable_path:$executable_path,executable_sha256:$executable_sha256,python_version:$python_version,pip_freeze_sha256:$pip_freeze_sha256,fixture_input_sha256:$fixture_sha256,expected_payload_sha256:$expected_payload_sha256,raw_report_sha256:{run_a:$raw_report_sha256_run_a,run_b:$raw_report_sha256_run_b},semantic_projection_sha256:$semantic_projection_sha256,claims_ceiling:"Pinned Unblob CLI on an owned inert nested gzip/tar fixture only; no REA, recovered firmware execution, emulator, or hardware qualification."}' > "$OUT/receipt.json"
printf '\nPASS ENG-REA-005 real-provider smoke: raw reports preserved, input/payload SHA-256 checked, semantic projection repeatable.\n'
printf 'Evidence directory: %s\n' "$OUT"
printf 'Receipt: %s/receipt.json\n' "$OUT"
printf 'Raw report SHA-256 run A: %s\nRaw report SHA-256 run B: %s\nSemantic projection SHA-256: %s\n' "$raw_a" "$raw_b" "$sem_a"
