#!/bin/sh
# Contract tests for the domain-specific signed evidence predicate.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKER="$ROOT/verify-attestation-predicate.sh"
SCHEMA="$ROOT/../../../../schemas/rea-provider-smoke-attestation-v1.schema.json"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
[ -f "$CHECKER" ] && [ -f "$SCHEMA" ] || fail "predicate checker or schema missing"
jq -e '
  .["$schema"]=="https://json-schema.org/draft/2020-12/schema" and
  .properties.schemaVersion.const=="symthaea.rea-provider-smoke-attestation.v1" and
  .properties.evidenceTier.const=="unblob-provider-smoke" and
  .properties.qualification.properties.reaCliMcpParity.const==false and
  .properties.qualification.properties.deviceQualification.const==false
' "$SCHEMA" >/dev/null || fail "versioned predicate schema drifted from the contract"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
SOURCE_HEAD=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
ATTESTOR_SHA=dddddddddddddddddddddddddddddddddddddddd
ATTESTOR_REF=Luminous-Dynamics/symthaea/.github/workflows/rea-evidence-attestation.yml@refs/heads/main
ATTESTOR_RUN=987654
PROVIDER_RUN=123456
PROVIDER_URL=https://github.com/Luminous-Dynamics/symthaea/actions/runs/123456
SEMANTIC_SHA=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
MUTATED_SHA=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

printf '%s\n' 'synthetic signed-subject bundle manifest v1' > "$TMP/bundle-manifest.json"
MANIFEST_SHA=$(sha256sum "$TMP/bundle-manifest.json" | awk '{print $1}')
jq -n \
  --arg source "$SOURCE_HEAD" \
  --arg manifest "$MANIFEST_SHA" \
  --arg semantic "$SEMANTIC_SHA" \
  --arg mutated "$MUTATED_SHA" \
  '{
    bundle_manifest_sha256:$manifest,
    source:{
      head_sha:$source,
      checkout_commit_sha:$source,
      workflow_event_sha:$source,
      normalizer_sha256:("1"*64),
      cases_manifest_sha256:("2"*64),
      oracle_sha256:("3"*64),
      requirements_lock_sha256:("4"*64),
      provider_smoke_script_sha256:("5"*64),
      provider_metamorphic_script_sha256:("6"*64)
    },
    provider_version:"26.6.4",
    semantic_projection_sha256:{equivalent_runs:$semantic,semantic_mutation:$mutated},
    provider_metamorphic:{equivalent_variants:64},
    claims_ceiling:"Pinned Unblob CLI on owned inert nested gzip/tar/text fixtures only; no REA, recovered firmware execution, emulator, or hardware qualification.",
    qualification:{tier:"unblob-provider-smoke",rea_cli_mcp_parity:false,firmware_execution:false,emulation:false,device_qualification:false}
  }' > "$TMP/receipt.json"
RECEIPT_SHA=$(sha256sum "$TMP/receipt.json" | awk '{print $1}')
jq -n \
  --arg source "$SOURCE_HEAD" \
  --arg attestor "$ATTESTOR_SHA" \
  --arg attestor_ref "$ATTESTOR_REF" \
  --arg attestor_run "$ATTESTOR_RUN" \
  --arg provider_run "$PROVIDER_RUN" \
  --arg provider_url "$PROVIDER_URL" \
  --arg receipt "$RECEIPT_SHA" \
  --arg manifest "$MANIFEST_SHA" \
  --arg semantic "$SEMANTIC_SHA" \
  --arg mutated "$MUTATED_SHA" \
  '{
    schemaVersion:"symthaea.rea-provider-smoke-attestation.v1",
    evidenceTier:"unblob-provider-smoke",
    repository:"Luminous-Dynamics/symthaea",
    predicateType:"https://github.com/Luminous-Dynamics/symthaea/attestations/rea-provider-smoke/v1",
    subject:{bundleManifestSha256:$manifest,receiptSha256:$receipt},
    providerRun:{
      runId:$provider_run,url:$provider_url,workflow:"REA Pinned Provider Smoke",event:"push",
      ref:"refs/heads/main",sourceHeadSha:$source,checkoutCommitSha:$source,providerVersion:"26.6.4",
      semanticProjectionSha256:$semantic,semanticMutationProjectionSha256:$mutated,providerMetamorphicVariants:64
    },
    attestationRun:{
      workflowRef:$attestor_ref,sourceCommitSha:$attestor,runId:$attestor_run,
      event:"workflow_dispatch",ref:"refs/heads/main"
    },
    qualification:{reaCliMcpParity:false,firmwareExecution:false,emulation:false,deviceQualification:false},
    limitations:[
      "Unblob provider smoke on generated inert gzip/tar/text fixtures only",
      "No REA CLI/MCP parity claim",
      "No recovered firmware execution or emulation",
      "No physical-device qualification"
    ]
  }' > "$TMP/predicate.json"

check(){
  sh "$CHECKER" "$1" "$TMP/receipt.json" "$TMP/bundle-manifest.json" \
    "$SOURCE_HEAD" "$ATTESTOR_SHA" "$ATTESTOR_REF" "$ATTESTOR_RUN" "$PROVIDER_RUN" "$PROVIDER_URL" \
    > "$TMP/check.log" 2>&1
}
check "$TMP/predicate.json" || { cat "$TMP/check.log" >&2; fail "valid predicate rejected"; }
printf 'PASS valid provider-smoke predicate accepted.\n'

kill_mutation(){
  label=$1
  filter=$2
  jq "$filter" "$TMP/predicate.json" > "$TMP/mutated-predicate.json" || fail "could not create mutation $label"
  if check "$TMP/mutated-predicate.json"; then
    fail "predicate policy mutation survived: $label"
  fi
  printf 'PASS predicate mutation rejected: %s\n' "$label"
}
kill_mutation wrong-receipt-digest '.subject.receiptSha256=("f"*64)'
kill_mutation wrong-manifest-digest '.subject.bundleManifestSha256=("0"*64)'
kill_mutation wrong-provider-source '.providerRun.sourceHeadSha=("e"*40)'
kill_mutation wrong-provider-run-id '.providerRun.runId="0"'
kill_mutation wrong-provider-version '.providerRun.providerVersion="26.6.3"'
kill_mutation wrong-provider-variant-count '.providerRun.providerMetamorphicVariants=63'
kill_mutation wrong-attestor-commit '.attestationRun.sourceCommitSha=("e"*40)'
kill_mutation wrong-attestor-ref '.attestationRun.ref="refs/heads/attacker"'
kill_mutation parity-scope-escalation '.qualification.reaCliMcpParity=true'
kill_mutation device-scope-escalation '.qualification.deviceQualification=true'
kill_mutation evidence-tier-escalation '.evidenceTier="device-qualified"'
kill_mutation semantic-sensitivity-lost '.providerRun.semanticMutationProjectionSha256=.providerRun.semanticProjectionSha256'
kill_mutation unexpected-root-field '.unreviewed=true'
kill_mutation malformed-sha256 '.subject.receiptSha256="deadbeef"'
kill_mutation unknown-limitation '.limitations[0]="Validated firmware device safety"'

printf '\nPASS ENG-REA-008 predicate contract: valid fixture accepted; 15/15 tampering and scope mutations rejected.\n'
