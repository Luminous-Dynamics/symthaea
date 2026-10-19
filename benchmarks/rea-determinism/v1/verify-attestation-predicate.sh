#!/bin/sh
# Validate the versioned domain predicate against an independently verified provider receipt.
# This is content-policy validation; the caller must separately verify the attestation signature.
set -eu
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 9 ] || fail "usage: $0 PREDICATE.json RECEIPT.json BUNDLE-MANIFEST.json PROVIDER-SOURCE-SHA ATTESTOR-SHA ATTESTOR-WORKFLOW-REF ATTESTOR-RUN-ID PROVIDER-RUN-ID PROVIDER-RUN-URL"
PREDICATE=$1
RECEIPT=$2
MANIFEST=$3
SOURCE_HEAD=$4
ATTESTOR_SHA=$5
ATTESTOR_REF=$6
ATTESTOR_RUN_ID=$7
PROVIDER_RUN_ID=$8
PROVIDER_RUN_URL=$9
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
[ -s "$PREDICATE" ] || fail "predicate file missing or empty"
[ -s "$RECEIPT" ] || fail "receipt file missing or empty"
[ -s "$MANIFEST" ] || fail "bundle manifest missing or empty"

RECEIPT_SHA=$(sha256sum "$RECEIPT" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
jq -e \
  --arg type "https://github.com/Luminous-Dynamics/symthaea/attestations/rea-provider-smoke/v1" \
  --arg source "$SOURCE_HEAD" \
  --arg attestor "$ATTESTOR_SHA" \
  --arg attestor_ref "$ATTESTOR_REF" \
  --arg attestor_run "$ATTESTOR_RUN_ID" \
  --arg provider_run "$PROVIDER_RUN_ID" \
  --arg provider_url "$PROVIDER_RUN_URL" \
  --arg receipt_sha "$RECEIPT_SHA" \
  --arg manifest_sha "$MANIFEST_SHA" \
  --slurpfile receipt "$RECEIPT" \
  '
    ($receipt|length)==1 and ($receipt[0]|type)=="object" and
    (keys|sort)==["attestationRun","evidenceTier","limitations","predicateType","providerRun","qualification","repository","schemaVersion","subject"] and
    .schemaVersion=="symthaea.rea-provider-smoke-attestation.v1" and
    .evidenceTier=="unblob-provider-smoke" and
    .repository=="Luminous-Dynamics/symthaea" and .predicateType==$type and
    (keys|length)==9 and
    (keys==["attestationRun","evidenceTier","limitations","predicateType","providerRun","qualification","repository","schemaVersion","subject"]) and
    (.subject|type)=="object" and
    (keys as $root | .subject|keys|sort)==["bundleManifestSha256","receiptSha256"] and
    .subject.bundleManifestSha256==$manifest_sha and .subject.receiptSha256==$receipt_sha and
    (.subject.bundleManifestSha256|test("^[0-9a-f]{64}$")) and (.subject.receiptSha256|test("^[0-9a-f]{64}$")) and
    $receipt[0].bundle_manifest_sha256==$manifest_sha and
    $receipt[0].source.head_sha==$source and
    $receipt[0].source.checkout_commit_sha==$source and
    $receipt[0].source.workflow_event_sha==$source and
    $receipt[0].provider_version=="26.6.4" and
    $receipt[0].claims_ceiling=="Pinned Unblob CLI on owned inert nested gzip/tar/text fixtures only; no REA, recovered firmware execution, emulator, or hardware qualification." and
    $receipt[0].qualification=={tier:"unblob-provider-smoke",rea_cli_mcp_parity:false,firmware_execution:false,emulation:false,device_qualification:false} and
    (.providerRun|type)=="object" and
    (keys|sort)==["attestationRun","evidenceTier","limitations","predicateType","providerRun","qualification","repository","schemaVersion","subject"] and
    (.providerRun|keys|sort)==["checkoutCommitSha","event","providerMetamorphicVariants","providerVersion","ref","runId","semanticMutationProjectionSha256","semanticProjectionSha256","sourceHeadSha","url","workflow"] and
    .providerRun.runId==$provider_run and .providerRun.url==$provider_url and
    .providerRun.workflow=="REA Pinned Provider Smoke" and .providerRun.event=="push" and .providerRun.ref=="refs/heads/main" and
    .providerRun.sourceHeadSha==$source and .providerRun.checkoutCommitSha==$source and .providerRun.providerVersion=="26.6.4" and
    (.providerRun.sourceHeadSha|test("^[0-9a-f]{40}$")) and
    (.providerRun.semanticProjectionSha256|test("^[0-9a-f]{64}$")) and
    (.providerRun.semanticMutationProjectionSha256|test("^[0-9a-f]{64}$")) and
    .providerRun.semanticProjectionSha256==$receipt[0].semantic_projection_sha256.equivalent_runs and
    .providerRun.semanticMutationProjectionSha256==$receipt[0].semantic_projection_sha256.semantic_mutation and
    .providerRun.semanticProjectionSha256!=.providerRun.semanticMutationProjectionSha256 and
    .providerRun.providerMetamorphicVariants==64 and
    .providerRun.providerMetamorphicVariants==$receipt[0].provider_metamorphic.equivalent_variants and
    (.attestationRun|keys|sort)==["event","ref","runId","sourceCommitSha","workflowRef"] and
    .attestationRun.workflowRef==$attestor_ref and .attestationRun.sourceCommitSha==$attestor and
    .attestationRun.runId==$attestor_run and .attestationRun.event=="workflow_dispatch" and .attestationRun.ref=="refs/heads/main" and
    (.attestationRun.sourceCommitSha|test("^[0-9a-f]{40}$")) and
    .qualification=={reaCliMcpParity:false,firmwareExecution:false,emulation:false,deviceQualification:false} and
    (.limitations|sort)==(["Unblob provider smoke on generated inert gzip/tar/text fixtures only","No REA CLI/MCP parity claim","No recovered firmware execution or emulation","No physical-device qualification"]|sort) and
    (.limitations|unique|length)==4
  ' "$PREDICATE" >/dev/null || fail "predicate is malformed, inconsistent with the receipt, or escalates its claim tier"
printf 'PASS provider-smoke attestation predicate: receipt, manifest, source, provider and claim policy match.\n'
