# Workspace orphan cleanup — increment closed 2026-07-29

Reachability and buildability are **separate dimensions**. This increment
addressed the first. It says nothing about the second, and the tables below keep
them apart deliberately — a passing orphan gate is not a health certificate.

## Result

| | before | after |
|---|---|---|
| orphaned modules (workspace) | 214 | 58 |
| stranded subtrees | 1 | **0** |
| crates under the gate | 3 | 203 |
| quarantined crates | — | 18 |
| files removed | — | 167 |

## What changed, and why each was safe

**148 one-line `// placeholder` files deleted.** Every one verified byte-identical
to `// placeholder` before removal, and none modified by a concurrent session.
They were never declared, never compiled, and unreferenceable. This is the bulk
of what looked like a "27.6K-line backlog" — it was mostly empty filesystem debris.

**`muse_motif_foundry_pilot.rs` deleted** (1,690 lines). Imported 12 types; 8 have
never existed in any commit on any branch. Fictional integration code, not an
outdated client.

**`evidence_calibration/` deleted** (19 files, 5,731 lines). Imported **39 items**
from a crate root that never existed, including 7 verifier types, the entire core
data model, 12 `audit_*` functions, and a whole `sha256` module. The upper half of
a subsystem whose lower half was never written. Design intent archived first at
`docs/design/EVIDENCE_CALIBRATION_STRANDED_DESIGN_2026-07-29.md`, with a
resurrection gate; git history keeps every line.

**The gate was inverted.** It began opt-in over 3 muse crates on the assumption
most of the workspace was dirty. Measurement found 172 of 200 crates *already*
clean, so opt-in had it backwards — it left 172 unprotected to accommodate 28. It
now enforces everywhere minus a named, shrinking quarantine.

## Where the backlog came from

Not 28 independent messes. Roughly half traces to "integrate/apply new domain
patchsets" bulk commits (`0cc1ff8539`, `b55f54ca21`, `30b5d9ab97`, `46d6ffa1ee`),
and most of the rest to two workspace reorganizations (`79d50ca86d` "Reorganize
Symthaea crates by tier", `4a212afc76`). Bulk operations, with nothing checking
that files stayed wired. `82330c0c0a` alone produced four disconnected artifacts
in one commit, three referencing APIs that do not exist.

## Build status — tracked separately, NOT addressed here

| crate | reachability | compilation | note |
|---|---|---|---|
| `symthaea-therapeutic` | clean | **failing** | see below |
| all other de-quarantined crates | clean | verified clean | `cargo check` run on statistics, acoustics, coding-theory, music-theory, wisdom, manipulator |

`symthaea-therapeutic` has not compiled since `30b5d9ab97`. `uncertainty.rs`
imports `crate::model_registry::ModelExecutionReceipt`; neither module nor type
has ever existed. Unrelated to this cleanup — `model_registry.rs` was never on
disk to delete.

**It was deliberately not "fixed."** `EstimateEnvelope::model_receipt` gates
`AbstentionReason::ProvenanceRequired`: a `ModelInference` or `Simulation`
estimate without a receipt must abstain. Because the type never existed the field
can only ever be `None`, so that gate currently rejects every model-inferred and
simulated estimate unconditionally — fail-closed and safe, but the feature is
unusable. Deleting the field would have made the crate compile by removing a
clinical provenance check: trading a safe failure for an unsafe success. Full
status is recorded in that crate's `lib.rs` header. Resolving it is a separately
authorized clinical-safety design task.

## Found while closing, NOT acted on: 35 `evidence_*` examples

`symthaea-music-theory/examples/` contains **35 `evidence_*.rs` example binaries**
(vs 11 other examples) from the same `0cc1ff8539` patch bundle. At least 15
reference `Calibration*` types at the crate root, or `support::publication_io`,
none of which exist — so `cargo check -p symthaea-music-theory --examples` fails,
and failed before this cleanup too. `examples/support/` contains only
`checkpoint_verifier.rs` and `mod.rs`; the `publication_io` module it imports was
never written.

They are a third layer of the same artifact: the subtree (deleted), the module
root (deleted), and these examples. Deleting them would arguably complete the
removal.

**They were left in place deliberately.** Discovering a 35-file layer while
closing an increment is precisely the dynamic that turns cleanup into a
self-perpetuating campaign, and the instruction for this increment was to stop.
The other 20 `evidence_*` examples were not individually assessed — the count of
15 is a lower bound from a pattern match, not an audit. Resolving them is a
separate, explicitly-scoped decision.

## What was deliberately NOT done

The remaining **58 orphans are real code** and need individual judgement, not a
sweep. `evidence_calibration` is the template for what that judgement looks like:
measure the dependencies first, and if the foundation was never written, archive
the intent and delete rather than manufacture architecture to make inherited
artifacts look legitimate.

This increment is closed. Grinding through the remaining 58 would turn cleanup
into another self-perpetuating campaign — the exact failure mode this audit
was diagnosing.
