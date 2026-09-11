# Continuity Fenced Resource Lab — Retained Evidence V1

## Status

`CONTAINED_SIMULATION / RETAINED_DESCRIPTIVE_EVIDENCE / NOT_VERIFIER_QUALIFIED`

This child preserves the already-passing fenced-resource lab subject from #1624 and adds a retained evidence bundle. It does not broaden the simulated resource theorem and does not grant production actuation authority.

## Parent subject

The parent contained-lab subject is the exact #1624 passing head:

`ccb6ce81eac5b4dcc638acaa7cc5cca4880233d0`

Its dedicated workflow run `34620171980` completed successfully and exercised the nine-obligation campaign composition through the independent #1578 oracle.

This child exists only to retain the exact run evidence in a separately inspectable artifact.

## Bundle contents

When `CONTINUITY_FENCED_RESOURCE_EVIDENCE_DIR` is set, the harness writes exactly:

- `campaign-manifest.json` — the campaign manifest accepted by the independent structural oracle;
- `campaign-observations.json` — all nine descriptive obligation observations, required bases, timestamps, record IDs and raw observation payloads;
- `campaign-summary.json` — scenario result, campaign/oracle digests and exact subject SHA;
- `campaign-run-context.json` — exact checked-out Git subject plus GitHub/Python/SQLite/platform run context;
- `SHA256SUMS` — SHA-256 for the four JSON documents above.

The checksum manifest does not self-hash. The workflow separately publishes SHA-256(`SHA256SUMS`) into the GitHub step summary.

## Exact-subject binding

The harness resolves the checked-out subject with:

```text
git rev-parse HEAD
```

and requires it to equal the workflow-provided `CONTINUITY_SUBJECT_SHA` when present.

The exact subject is then persisted in both `campaign-run-context.json` and `campaign-summary.json`.

A mismatch fails the run before artifact upload.

## Atomic local writes

Each retained JSON document is written to a temporary sibling file, flushed, `fsync`ed and atomically renamed into place.

The evidence directory must start empty. This prevents stale files from a prior run from being silently included in the retained bundle.

## Workflow verification

Before upload, the dedicated workflow requires:

1. immutable pinned `actions/checkout` and `actions/upload-artifact` SHAs;
2. exact PR-head checkout;
3. successful fenced-resource scenario/campaign execution;
4. `sha256sum -c SHA256SUMS` over the retained bundle;
5. exact retained subject SHA equals the exact workflow subject;
6. retained observation count is exactly nine;
7. SHA-256(`SHA256SUMS`) is published into the workflow summary;
8. only then is the bundle uploaded as an artifact.

The artifact is retained for 14 days.

## Core theorem

```text
executed contained scenarios
-> nine raw descriptive observations
-> one oracle-accepted campaign manifest
-> exact subject/run context
-> checksum-verified retained artifact
```

This remains distinct from:

```text
retained artifact
!= authenticated evidence truth
!= #1550 verifier admission
!= production adapter qualification
!= physical infrastructure authority
```

## Why retention matters

The original #1624 run proved the contained theorem but left most detailed observation state ephemeral in the runner. Retention allows later reviewers and the future #1550 verifier implementation to inspect the exact nine observations and campaign identities instead of relying only on workflow logs.

The artifact is still produced by the same contained harness and is not independently signed. Its presence or checksum integrity does not make its semantic claims true.

## Downstream

A future campaign-bound Rust evidence object should be able to ingest or independently reproduce the semantic content of these retained documents while preserving the #1578 independent preimage parity contract.

Only after verifier-owned admission exists should such a retained campaign be considered for `QualifiedActuationEnforcementEvidence`.

A real NETCONF, Redfish, gNOI, storage, hypervisor, BMC or Spore privileged-helper campaign must produce its own resource-boundary evidence and cannot inherit qualification from this SQLite simulation.
