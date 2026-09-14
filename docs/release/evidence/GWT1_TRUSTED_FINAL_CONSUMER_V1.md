# GWT-1 Trusted Final Consumer V1

This tranche creates the reporting-side authority boundary for the frozen GWT-1 direct + causal evidence program.

## Authority chain

```text
exact final-resolution run + attempt
        ↓
frozen GitHub CLI distribution/executable/version identity
        ↓
fresh final-archive attestation verification
        ↓
bounded 60-member final archive admission
        ↓
internal SHA256SUMS + producer provenance checks
        ↓
bounded 7-member re-admission of preserved direct evidence
        ↓
independent Rust direct + causal reconstruction
        ↓
stored state == reconstructed state
        ↓
opaque non-clonable in-memory verified token
        ↓
Serialize-only GWT-1 report projection
        ↓
deterministic self-contained 26-member report archive
        ↓
separate trusted-consumer attestation
```

The report archive carries the exact final-resolution source archive so replay does not depend on mutable artifact discovery or retention. It also retains both the fresh final-archive admission receipt and the fresh nested-direct admission receipt committed by the consumer verifier profile, so every admission commitment remains auditable from preserved bytes.

## Explicit activation fuse

Public/reporting authority is fail-closed in this tranche:

```text
GWT1_PUBLIC_REPORTING_AUTHORITY_ENABLED_V1 = false
```

The trusted consumer may compile and its structural contracts may qualify, but its reconstruction adapter refuses to emit even a report projection candidate while this fuse is false.

A later, separately reviewed activation tranche must advance all draft/bootstrap authority roots to reviewed/landed immutable revisions and flip the fuse. This keeps code qualification separate from authority activation.

## Deliberate non-claims

A future activated successful consumer run would establish only that the persisted GWT-1 reporting state was produced from an exact cryptographically verified final-resolution artifact and independently reconstructed under the frozen V1 protocol.

It would not by itself establish a positive causal result. The projection preserves negative, contradictory and inconclusive causal dispositions. No V1 path may produce `FunctionallySupported`.

It does not establish GWT-2/3/4, sentience, consciousness or alignment.

## Bootstrap activation blocker

The trusted consumer currently pins draft PR #2743 head `198aaf257de0b4f264f69eb64d813d772f0362d6` as the expected final-resolution signer digest so the dependency graph and qualification surface are explicit.

That SHA is a bootstrap candidate, not an approval claim. Before authoritative activation it must be replaced by the reviewed/landed immutable final-resolution workflow revision, all upstream bootstrap authority constants must likewise be advanced to reviewed/landed revisions, and the explicit public/reporting activation fuse must be flipped in the same reviewable authority transition.
