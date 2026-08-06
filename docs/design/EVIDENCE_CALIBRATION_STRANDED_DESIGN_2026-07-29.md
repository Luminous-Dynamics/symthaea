# Stranded design: `symthaea-music-theory::evidence_calibration`

**Status: archived and deleted from the source tree, 2026-07-29.** The code never
compiled and was never reachable. This document exists so the *intent* stays
discoverable; the code itself is in git history, not in the tree.

## Recovery

```bash
# Browse it:
git show 0cc1ff8539:symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/publication/witness_policy.rs

# Restore the whole subtree at the commit before deletion:
git checkout 3b92b20899 -- symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/
```

Historical path: `symthaea/crates/domains/symthaea-music-theory/src/evidence_calibration/`
(18 files, 5,690 lines, 28 `#[test]`s at time of deletion).

## Origin

| commit | date | what it did |
|---|---|---|
| `0cc1ff8539` | 2026-07-21 | "apply new wave of domain patchsets (Subterranean 18-20, Music Theory 16-19, …)" — introduced the subtree |
| `b55f54ca21` | later | another bulk patchset touching it |
| `9fe8626cd3` | later | "apply and commit SMT series 14 patches" |

Never touched by hand-written work. Three bulk applications, no wiring.

## What it was for

A **certificate-transparency-style publication-integrity layer for calibration
evidence**: making published calibration catalogs externally auditable, so a
third party can verify that a published record was not altered or forked after
the fact, and that the authority publishing it rotated its keys legitimately.

Conceptual components, all under `publication/`:

- **witness policy** (1,132 lines) — append-only witness-policy epochs with
  *dual-quorum rotation*: every non-genesis epoch must be authorized by **both**
  the outgoing and incoming witness quorums, so continuity survives key expiry,
  departing organizations, and compromised-witness removal. Signature
  algorithms, key custody, and witness trust deliberately external to the crate.
- **lineage** (373) — exact multi-hop catalog lineage *composed from direct
  consistency proofs*, without weakening them: every intermediate catalog and
  checkpoint stays explicit and each link is re-audited independently.
- **recovery authority** (590 + 191 test lines) — recovery-authority genesis and
  rotation, with an append-only rotation ledger.
- **incident closure** (751 + 173) — closure policy, closure planning, signed
  closure statements, authorization sets, closure bundles.
- **re-entry** (600 + 171) — post-recovery certification: build / audit / verify,
  plus required-limitations disclosure before a recovered authority may resume.
- **authenticated gossip** (585 + 262) — gossip ledger, conflict proofs, and
  persisted gossip models, for detecting split views between witnesses.
- **continuity** (377 + 202) — portable continuity bundles carrying witness
  policy, head, and gossip state together.

The ideas worth revisiting are the **dual-quorum rotation** and the
**compose-don't-weaken lineage** approach. Both are sound, and neither depends
on the rest of the subsystem.

## Why it could not be wired

`lib.rs` declared `pub mod evidence_calibration;`, which resolved to a one-line
`// placeholder` module root. The subtree beneath had no `mod.rs` and no
declarations, so none of it was part of the crate.

It could not be connected by adding `mod` lines. Its files import **39 items**
from a crate root that has never existed — verified with `git log -S` across all
refs on 2026-07-29 — including:

- 7 verifier types (`CalibrationPublicationCheckpointWitnessVerifier`,
  `…GossipVerifier`, `…QuarantineVerifier`, `…RecoveryVerifier`,
  `…WitnessPolicyRotationVerifier`, `…RecoveryAuthorityRotationVerifier`,
  plus `CalibrationSignerIdentity`);
- the core data model (`CalibrationPublicationCatalog`,
  `…CatalogCheckpoint`, `…CatalogConsistencyProof`, `…CatalogHeadBundle`,
  `…GossipLedger`, `…WitnessPolicyLedger`, `…RecoveryAuthorityPolicy`,
  `…PostRecoveryCertification`, `…CheckpointWitnessPolicy`);
- 12 `audit_*` and 5 `build_*` functions;
- 2 schema-version constants and 4 `*_sha256()` helpers;
- an entire `evidence_calibration::sha256` module (`Sha256`, `hex`).

That is not an outdated client against a moved API. It is the **upper half of a
subsystem whose lower half was never written**. Adding declarations would have
converted 5,690 silently-dead lines into several hundred compile errors.

## Resurrection gate

Do **not** restore the subtree wholesale. Before any of this returns:

1. A concrete Muse workflow that actually needs published calibration evidence
   to be externally auditable.
2. A written minimal public data model — what a catalog, checkpoint, and
   signer identity contain — agreed *before* code.
3. One end-to-end caller exercising it.
4. A small implementation built incrementally from tests.
5. No bulk copy-back. Take the two ideas named above and re-derive the rest
   against the model in (2).

## Why it was deleted rather than kept

Keeping unreachable code that cannot compile has a real cost: it inflates the
crate, it reads as available capability to anyone browsing, and — as this audit
found repeatedly — it accumulates further patch bundles piled onto a foundation
that does not exist. Git preserves every line. This document preserves the
intent. Neither requires the tree to carry a subsystem nobody can call.
