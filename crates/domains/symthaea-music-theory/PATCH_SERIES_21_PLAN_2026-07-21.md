# Symthaea Music Theory Patch Series 21

## Objective

Complete the publication incident lifecycle after Series 20 exceptional recovery by adding:

- append-only, dual-quorum recovery-authority rotation;
- fresh-checkpoint policy re-entry on the selected catalog branch;
- separately governed operational incident closure;
- external-verifier command-line workflows;
- persisted-schema and crate-root API coverage;
- adversarial regressions for tampering, insufficient advance, and quarantine disposition.

## Landing sequence

### A. Recovery-authority policy history

1. Model authority epochs, rotation payloads, signed envelopes, rotation sets, ledgers, audits, and verifier interfaces.
2. Implement genesis, rotation planning, outgoing/incoming threshold verification, transactional append, active-policy lookup, and canonical SHA-256 identities.
3. Test genesis, successful dual quorum, rollback on missing incoming quorum, no-op rejection, and ledger tampering.

### B. Fresh post-recovery re-entry

4. Model the post-recovery certification and machine-readable limitations.
5. Bind the incident-response package, exact continuity lineage, recovered witness policy, recovery-authority ledger, selected checkpoint, new head, catalog advance, and logical epoch.
6. Require authenticated verification across all underlying incident, recovery, continuity, gossip, quarantine, witness, and authority-rotation layers.
7. Test accepted re-entry, minimum catalog advance, and recovered-policy anchor substitution.

### C. Operational closure

8. Model closure policies, canonical plans, signer roles, signed statements, authorization sets, bundles, and audit reports.
9. Require accepted re-entry before closure planning.
10. Require the closure quarantine ledger to extend the recovery containment history.
11. Require signatures from active recovery authorities and recovered witnesses, with optional stricter thresholds and quarantine restrictions.
12. Test dual-quorum closure, forbidden active observer quarantine, and containment-history substitution.

### D. Integration and tooling

13. Register every persisted contract without renumbering previous schema roles.
14. Restore and extend the curated crate-root publication API used by examples.
15. Extend the shell-free external verifier adapter.
16. Add recovery-authority, post-recovery, and incident-closure tools.
17. Fix compile-oriented integration findings discovered during static review.

## Compatibility

- Existing Score, composition, calibration, publication, continuity, incident, and recovery APIs remain unchanged.
- New persisted schema roles are appended after the Series 20 prefix.
- The recovery-authority, re-entry, and closure contracts begin at version 1.
- Existing incident evidence remains append-only and is not rewritten by operational closure.

## Required downstream verification

Run in the canonical project shell:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
```

The patch bundle also includes static parse, import/export, direct-call arity, whitespace, clean-application, tree-identity, archive-reproduction, and SHA-256 evidence. Those checks do not replace Rust compilation.
