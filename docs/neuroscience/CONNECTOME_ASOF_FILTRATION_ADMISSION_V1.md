# As-of evidence, filtration admission and committed ledger V1

**Tracking:** #10982 — SCI-CONNECTOMICS-102; commitment extension #11083 — SCI-CONNECTOMICS-106  
**Composes:** #8732, #9848, #9862, #10897 / #10898, #10919 / #10922, #10948 / #10959, #10960 / #10961, #11062 and #11065.  
**Guarantee:** operational chronology, canonical entry commitment replay, decision-prefix binding and external-checkpoint matching only.  
**Status:** synthetic known-answer contract; no arbitrary e-process theorem or biological truth claim is proved by this crate.

## 1. Event time is not filtration time

The operational filtration is ordered by the driver-owned global availability sequence. Biological event time, acquisition, annotation/processing, QC/adjudication and availability are separate facts. A late correction is a new immutable revision event, never an edit to the earlier row. Sorting by biological event time can create lookahead when a result was processed or adjudicated after an adaptive action.

## 2. Four levels of evidence integrity

Do not conflate these claims:

1. **Shape-valid:** identifiers and digest strings have the expected type/format.
2. **Internally committed:** independent replay recomputes each canonical entry digest, each decision prefix digest and the terminal append-chain root.
3. **Externally anchored:** the terminal epoch, sequence and root match a checkpoint whose digest was retained through a separate trusted channel.
4. **Factually authentic and complete:** source issuers and the availability authority are authenticated, and a trusted inventory contract establishes that the evidence history is complete.

V1 establishes level 2. It can compare against an independently pinned checkpoint using the explicit API described below, but it does not verify digital signatures, witness quorums or source authority. Even a signed checkpoint proves only that the signer committed to those bytes; it does not prove the biological evidence is true.

## 3. Canonical encoding and append chain

`global_filtration_sha256` remains the digest identifying the filtration definition. It is **not** repurposed as the evolving evidence-ledger root. Each ledger epoch has a separate `ledger_epoch_id`, each entry has an `entry_commitment_sha256`, and the manifest has `ledger_chain_root_sha256`.

The V1 typed encoding uses:
- schema version `u32` big-endian;
- hash-suite string `SHA-256`;
- UTF-8 strings and byte sequences prefixed by `u64` big-endian length;
- signed timestamps as `i64` big-endian, preceded by an explicit Option tag;
- SHA-256 references as exactly 32 raw bytes;
- fixed enum discriminants, in the order stated below;
- explicit Option tags (`0 = None`, `1 = Some`);
- ordered arrays prefixed by a `u64` count.

No JSON serialization, Rust `Debug`, locale formatting or untyped concatenation is hashed.

### Stable payload discriminants

| Payload | Discriminant | Canonical fields (in order) |
|---|---:|---|
| Evidence | 1 | event ID, evidence-family ID, source stream, source-artifact digest, revision, superseded event ID, event/acquisition/annotation/QC timestamps, availability-state enum, split enum, confirmatory-family ID |
| Family freeze | 2 | family ID, family-commitment digest, lockbox-closed boolean |
| Decision | 3 | decision ID, campaign ID, decision-kind enum, policy-version digest, input-prefix digest, driver-receipt digest, history-prefix digest, action digest, feasibility-receipt digest, available event IDs, consumed event IDs, confirmatory-family ID |

Availability-state discriminants: APPROVED=0, PROVISIONAL=1, DISPUTED=2, FAILED=3, MISSING=4, OUT_OF_SUPPORT=5, NOT_ASSESSED=6.  
Evidence-split discriminants: EXPLORATION=0, FIT=1, CALIBRATION=2, CONFIRMATION=3.  
Decision-kind discriminants: DISCOVERY=0, CONFIRMATION=1.

### Genesis and entry roots

The genesis root is:

```
SHA256(
  "SYTHAEA/CONNECTOME/ASOF-LEDGER-GENESIS/V1\\0" ||
  u32be(schema_version) || lp("SHA-256") ||
  lp(campaign_id) || lp(global_filtration_id) ||
  raw32(global_filtration_sha256) || lp(ledger_epoch_id)
)
```

For every one-based global sequence `i`, after canonicalizing the payload:

```
entry_root_i = SHA256(
  "SYTHAEA/CONNECTOME/ASOF-LEDGER-ENTRY/V1\\0" ||
  u32be(schema_version) || lp("SHA-256") ||
  lp(campaign_id) || lp(global_filtration_id) ||
  raw32(global_filtration_sha256) || lp(ledger_epoch_id) ||
  u64be(i) || raw32(entry_root_(i-1)) ||
  u8(payload_kind) || lp(canonical_payload)
)
```

`entry_commitment_sha256` must equal the independently recomputed `entry_root_i`; `ledger_chain_root_sha256` must equal the final entry root. The verifier rejects gaps, duplicate IDs, invalid revisions, reordered entries, a changed payload with an unchanged hash, a hash-shaped but incorrect entry digest, and a mismatched terminal root.

### Decision-time prefixes

Two separate domain-separated hashes are recomputed:

- `input_prefix_sha256`: the ordered evidence IDs that were actually available before the decision, each paired with its canonical entry root.
- `history_prefix_sha256`: the ordered consumed event IDs, each paired with its canonical entry root; a missing/future ID is encoded with an explicit absent-root tag and still triggers lookahead semantics.

The ordered `available_event_ids` must exactly equal the evidence prefix replayed before that decision. These fields can no longer be made to appear valid by replacing them with any other 64-character string.

Policy, driver, action and feasibility digests remain opaque external receipt references in V1. Their string format is checked and their values are covered by the decision entry hash, but this crate does not recompute them from external receipt bytes. The report must not imply otherwise.

## 4. External terminal checkpoint and truncation

`audit_ledger` returns `InternallyCommittedUnanchored`. An internally consistent prefix is not proof that no later entries existed.

`audit_ledger_against_external_checkpoint` optionally compares:
- campaign and filtration identity;
- ledger epoch;
- terminal sequence;
- terminal chain root;
- the checkpoint's canonical digest.

The caller must supply `independently_pinned_checkpoint_sha256` through a channel independent of the ledger being audited. The verifier returns `MatchesIndependentlyPinnedCheckpoint` only when the expected checkpoint bytes match that pinned digest and the ledger matches the checkpoint fields.

This function does not validate a signature or witness quorum. It cannot distinguish a trustworthy external pin from an attacker-supplied value on its own. Digital signatures, threshold witnesses, epoch rollover and source-inventory completeness remain a separate promotion tranche under the generic CogSec requirements #171 and #191.

A hash chain by itself cannot detect a truncation that is resealed as a new shorter chain. A previously retained external checkpoint can detect that the shorter chain does not match the expected terminal sequence/root.

## 5. As-of replay and filtration admission

For every evidence artifact retain biological event, acquisition, annotation/processing, QC/adjudication and availability times separately. A decision binds:
- the exact ordered evidence IDs delivered before it;
- the subset actually consumed;
- immutable policy, action, history and driver receipt commitments.

Discovery decisions cannot consume confirmation-split evidence. Confirmation decisions require a prior family freeze and post-freeze confirmation data bound to that family.

A local/coarser process is not automatically valid in the global filtration. V1 keeps four explicit receipt modes:
- direct-global supermartingale proof receipt;
- adjuster-lifted proof receipt;
- local-only (NOT_COMBINABLE);
- incomparable/unknown (NOT_COMBINABLE).

The audit validates exact receipt identities and source/target filtration binding. It does not prove the conditional expectation inequality or the adjuster theorem. #11062 and #11065 compose these typed admissions with the closed and flat FWER family gates; those consumers must receive an internally committed ledger before issuing admission-dependent decisions.

## 6. Exact synthetic known answer

The checked-in fixture contains five entries:
1. S1 structural evidence becomes available;
2. D1 consumes S1;
3. T7 behavior evidence arrives later, despite an earlier biological event timestamp;
4. D2 consumes S1 and T7;
5. T8 remains a typed FAILED event.

The fixture stores exact per-entry SHA-256 roots, decision input/history prefix digests, the final chain root and the canonical terminal-checkpoint digest. An independent Python standard-library oracle recomputes those bytes and hashes. The Rust verifier runs separate tamper, prefix, chronology and checkpoint tests.

## 7. Promotion gates and claim ceiling

Synthetic:
- exact byte/digest parity between Rust and Python reference implementations;
- all entry roots and terminal roots recomputed, not shape-checked only;
- payload, sequence, prefix and root mutations rejected;
- a resealed truncated prefix remains explicitly unanchored and fails comparison with the independently pinned full checkpoint;
- no signed/witnessed label without a separately verified signature/witness layer.

This crate does not prove source authenticity, issuer authority, inventory completeness, supermartingale or adjuster validity, FDR/FDR-sup, or any biological mechanism. It establishes canonical ledger commitment and as-of replay semantics only.
