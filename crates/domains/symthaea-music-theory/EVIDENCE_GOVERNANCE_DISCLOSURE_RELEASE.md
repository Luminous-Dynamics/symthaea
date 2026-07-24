# Evidence Governance Disclosure Release Contract

## Purpose

Patch Series 16 adds portable evidence for governance operations without
publishing listener-level records. The release is centered on four contracts:

1. Exact before/after governance receipts.
2. Append-only receipt chains.
3. Selective disclosure and identity-free retention snapshots.
4. Canonical public governance exports suitable for an external signature.

These contracts prove deterministic statements about the supplied Symthaea
evidence objects. They do not prove deletion from uncontrolled copies, legal
compliance, signer authority, the correctness of an external privacy
mechanism, or the uniqueness of a human participant.

## Exact governance receipts

A `CalibrationGovernanceReceipt` binds:

- Source revision and optional source tree.
- Music-theory engine version.
- Caller-governed logical epoch.
- Complete bundle SHA-256 before the operation.
- Complete bundle SHA-256 after the operation.
- The exact governance action.
- A canonical receipt SHA-256.

Supported actions are:

- Authenticated study-response withdrawal.
- Logical-epoch retention enforcement.
- Externally verified privacy-budget release.

Transition audit receives the private before and after bundles. It verifies
both bundle audits, source continuity, engine continuity, exact bundle
identities, and the operation-specific state change.

A receipt is not a deletion certificate. It shows that the governed bundle no
longer contains the response and that a withdrawal tombstone was added. It
cannot inspect backups, exports, logs, caches, or systems outside the supplied
bundle.

## Receipt chains

`CalibrationGovernanceReceiptChain` requires:

- Contiguous receipt sequences beginning at zero.
- One source and engine identity.
- Non-regressing logical epochs.
- Exact previous-after to next-before bundle continuity.
- A canonical chain SHA-256.

The chain records state history without including the underlying listener
records. A chain with a broken bundle link cannot be repaired by merely
rehashing the outer object.

## Selective disclosure

`CalibrationSelectiveDisclosure` is a schema-level projection. It cannot carry:

- The canonical mutation corpus.
- Case records or case-level judgments.
- Assessor pseudonyms.
- Signed-response identities.
- Study judgment links.
- Private study summaries.
- Withdrawal tombstones.
- Privacy mechanism proofs.
- Listening reveals.
- Assignment registries.
- Anonymous-credential presentations.

The public-release profile includes:

- Source revision, but not source tree.
- Bundle release assessment.
- Existing small-cell-suppressed public study summary.
- Aggregate governance totals.
- Aggregate privacy-budget consumption.

The auditor-minimal profile includes the source tree but omits study metrics.

Recipients can verify the disclosure's canonical SHA-256. An auditor with the
private bundle can reconstruct the projection exactly and verify its source
bundle identity.

## Retention-compliance snapshots

`CalibrationRetentionComplianceSnapshot` contains only aggregate counts. It
uses caller-supplied logical epochs and reports one of three states:

- `compliant`: no expired, future, or unknown active attachment epochs.
- `review_required`: no expired or future epochs, but at least one legacy
  response has an unknown attachment epoch.
- `noncompliant`: at least one expired active response or impossible future
  attachment epoch exists.

Unknown legacy epochs are never guessed and never silently counted as
compliant.

The snapshot includes the next known expiry and oldest known age when those
values exist. It never includes response identities.

## Public governance export

`CalibrationGovernanceExport` combines:

- One selective disclosure.
- One retention-compliance snapshot.
- An optional receipt-chain head summary.
- Mandatory machine-readable trust limitations.
- One canonical export SHA-256.

The embedded disclosure and retention snapshot must refer to the same private
bundle. When a receipt-chain summary is present, its final after-bundle SHA-256
must equal the disclosed bundle SHA-256.

The following limitations are part of the hashed artifact:

- External-copy deletion is not proven.
- Legal compliance is not established.
- Publisher authentication requires an external signature.
- External privacy mechanisms are not validated by this artifact.
- Unique human participation is not established.
- Logical epochs remain externally governed.

## External attestation

`CalibrationGovernanceExportAttestationPayload` provides deterministic,
length-prefixed bytes binding:

- Governance export version and SHA-256.
- Source bundle SHA-256.
- Engine version.
- Disclosed source revision and optional source tree.

The crate only creates and audits canonical bytes. Signature algorithms,
private keys, signer authorization, timestamps, and transparency-log clients
remain external.

## Operator workflow

Export public evidence:

```text
cargo run --example evidence_selective_disclosure -- \
  --bundle private-bundle.json \
  --profile public \
  --write disclosure.json
```

Create an identity-free retention snapshot:

```text
cargo run --example evidence_retention_snapshot -- \
  --bundle private-bundle.json \
  --current-epoch 120 \
  --maximum-retention-epochs 30 \
  --write retention.json
```

Build a receipt from exact before and after bundles:

```text
cargo run --example evidence_governance_receipt -- \
  --kind withdrawal \
  --before before.json \
  --after after.json \
  --evidence withdrawal-result.json \
  --write receipt.json
```

Append the receipt to a chain:

```text
cargo run --example evidence_governance_receipt_chain -- \
  --receipt receipt.json \
  --chain existing-chain.json \
  --write updated-chain.json
```

Build the combined public export:

```text
cargo run --example evidence_governance_export -- \
  --bundle private-bundle.json \
  --receipt-chain governance-chain.json \
  --profile public \
  --current-epoch 120 \
  --retention-epochs 30 \
  --write governance-export.json
```

Create canonical external-signing bytes:

```text
cargo run --example evidence_governance_attestation_payload -- \
  --export governance-export.json \
  --write governance-attestation.json \
  --write-bytes governance-attestation.bin
```

## Release requirement

Before merge, run the canonical repository checks:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Also exercise each governance example against a real audited bundle and retain
its JSON output in the release evidence directory.
