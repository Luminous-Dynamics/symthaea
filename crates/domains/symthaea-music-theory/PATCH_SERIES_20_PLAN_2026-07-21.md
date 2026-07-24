# Symthaea Music Theory Patch Series 20 Plan

**Date:** 2026-07-21  
**Base:** Patch Series 19 / Git tree `3136970a475d4e70adb6f0eaf292c1eb7e103910`  
**Theme:** Conservative fork incident attribution, externally governed containment, and recovery when the outgoing witness quorum is unavailable or unsafe

## Executive summary

Series 19 can preserve authenticated rollback, equivocation, and branch-conflict
evidence. It can also rotate witnesses when the outgoing policy remains able and
safe to co-authorize its successor. It deliberately does not solve the harder
case: the outgoing quorum itself may be unavailable, implicated, or suspected
of compromise.

Series 20 adds a bounded incident-response path without rewriting disputed
history or claiming global consensus:

1. derive conservative incident findings from exact conflict proofs;
2. distinguish direct signer contradiction from branch conflict reported by
   observers;
3. contain identities through an append-only externally authenticated
   quarantine ledger;
4. authorize one exact recovery lineage through a predeclared recovery
   authority plus the incoming witness quorum;
5. begin a new witness-policy trust segment at the selected checkpoint; and
6. package the complete recovery and new policy anchor as one portable,
   self-auditing artifact.

The recovery decision is an authorized branch selection. It is not proof of
universal canonicality, signer guilt, or absence of withheld forks.

## Patch groups

### A. Conservative incident reports

- Build incident findings only from authenticated Series-19 conflict proofs.
- Preserve the original proof inside every finding.
- Attribute same-signer rollback or equivocation directly to that signing key.
- Treat authority equivocation and checkpoint forks as branch-level conflicts
  observed by reporters, without automatically blaming those reporters.
- Derive severity, checkpoint identities, policy identities, and reporting
  observers independently during audit.
- Bind every finding and report to canonical SHA-256 identities.

### B. Governed identity quarantine

- Introduce an externally governed quarantine-authority policy.
- Persist signed quarantine and release decisions in one append-only ledger.
- Support witness, observer, and combined scopes.
- Support caller-supplied logical effective and expiry epochs.
- Evaluate active containment against witness policies and gossip statements.
- Report when removing quarantined witnesses makes the configured quorum
  unavailable.
- Preserve containment history after release or expiry.

### C. Exceptional recovery authorization

- Bind recovery to the exact incident report and quarantine ledger.
- Require the disputed witness-policy epoch to appear in the incident.
- Require the selected lineage to begin at a checkpoint in the incident
  history.
- Reject recovery before the incident or selected checkpoint.
- Reject incoming witness policies containing actively quarantined witnesses.
- Require every directly contradictory observer signer to be actively
  contained.
- Require distinct signatures from a predeclared recovery-authority quorum and
  the complete incoming witness quorum.
- Keep signature algorithms, key custody, and signer legitimacy external.

### D. Recovered policy anchor

- Create a new witness-policy genesis at the selected recovery checkpoint.
- Bind the new policy segment to the exact authorized recovery bundle.
- Preserve the disputed incident, quarantine history, and old policy lineage.
- Require fresh checkpoint witnessing under the recovered policy as an
  explicit machine-readable limitation.

### E. Portable incident-response package

- Bind the authorized recovery bundle and recovered policy anchor under one
  canonical package SHA-256.
- Re-run all configured checkpoint, rotation, gossip, quarantine, recovery-
  authority, and incoming-witness verifiers during authenticated audit.
- Keep structural validity and external authorization as separate gates.
- Reject substituted anchors even when outer hashes are recomputed.

### F. Operator workflows

Add examples for:

- incident-report creation, structural audit, and authenticated verification;
- quarantine-policy creation, decision signing, append, release, evaluation,
  audit, and verification;
- recovery-policy creation, recovery planning, statement creation, dual-quorum
  authorization, bundle verification, and recovered-policy anchoring;
- portable incident-response package build, audit, and verification.

All external verifier programs receive JSON over standard input and are invoked
without a shell.

### G. Persistence and API governance

- Export only explicit Series-20 contracts through the crate root.
- Advance the machine-readable schema registry for incident, quarantine,
  recovery, anchor, and response-package records.
- Append every Series-20 role after the complete Series-19 `#[repr(u16)]`
  prefix.
- Freeze the appended role ordinals in regression tests.
- Reject unknown fields on every new persisted model.

### H. Adversarial regression coverage

- Rehashing a report cannot conceal false direct attribution.
- Branch conflict does not directly implicate reporting observers.
- Duplicate governance or recovery signers do not satisfy thresholds.
- Quarantine release and expiry remove active containment without deleting
  history.
- Recovery fails when directly contradictory signers remain uncontained.
- Recovery rejects quarantined incoming witnesses.
- Rehashing a recovery bundle cannot conceal removed containment.
- Rehashing a portable package cannot conceal an anchor substitution.

## Trust boundaries

Series 20 does not establish:

- signer intent, guilt, negligence, or human identity;
- universal branch canonicality;
- absence of withheld or unobserved forks;
- independence of recovery authorities or incoming witnesses;
- legal legitimacy of containment or recovery governance;
- wall-clock freshness;
- certificate-chain, hardware-key, or transparency-log trust;
- secure private-key custody;
- automatic repair of external mirrors or deletion of disputed history;
- distributed consensus.

Those responsibilities remain external and are represented as mandatory
machine-readable limitations.

## Landing order

1. Incident findings and authenticated report audit.
2. Finding-construction and attribution hardening.
3. Quarantine policy, signed decisions, ledger, and evaluation.
4. Recovery authority, explicit lineage selection, and dual authorization.
5. Structural-validity semantics correction inherited from Series 19.
6. Duplicate-signer and recovered-anchor hardening.
7. End-to-end and adversarial regressions.
8. Persistence-role registration and curated crate-root exports.
9. External verifier adapter and three operator tools.
10. Portable incident-response package, schema registration, and tool.
11. Release contract and design integration.
12. Reproducibility plan, application guide, patch archive, and source snapshot.

## Required canonical verification

Run in the project development shell:

```text
cargo fmt --all -- --check
cargo test -p symthaea-music-theory
cargo clippy -p symthaea-music-theory --all-targets -- -D warnings
cargo check -p symthaea-music-theory --examples
```

Recommended focused checks:

```text
cargo test -p symthaea-music-theory incident
cargo test -p symthaea-music-theory quarantine
cargo test -p symthaea-music-theory recovery
cargo test -p symthaea-music-theory incident_response
cargo test -p symthaea-music-theory schema
```

Static syntax, persistence, import, arity, patch-replay, and archive checks are
valuable, but they do not replace Rust compilation and test execution.
