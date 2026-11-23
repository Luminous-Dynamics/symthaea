# MEL-AGENCY-001 — replayable proposal, authorization and undo traces

**Date:** 22 November 2026  
**Parent:** [#11319 — co-creative agency, consent and authorship](https://github.com/Luminous-Dynamics/symthaea/issues/11319)  
**Adjacent:** [#10051 — MEL-013A verified Studio artifact transactions](https://github.com/Luminous-Dynamics/symthaea/issues/10051), [#10088 — reciprocal musical interaction](https://github.com/Luminous-Dynamics/symthaea/issues/10088), [#9307 — operation-contract equivalence](https://github.com/Luminous-Dynamics/symthaea/issues/9307)

## Why this increment exists

Muse already has research contracts for multiple valid musical responses, selected preferences, vetoes and ambiguity. RMI-Bench separately evaluates causal reciprocity. MEL-013A defines the production boundary for authenticated actors, purpose-scoped consent, durable artifact lineage and restart-safe transactions.

The gap addressed here is a small but essential executable bridge: can a recorded proposal lifecycle be replayed and rejected when the commit is not the exact thing that was previewed and authorized, when a veto is ignored, or when an undo does not restore the exact source?

This is a reference evaluator under the co-creative agency programme. It does not create a second consent ledger, artifact registry or production Studio transaction path.

## Trace lifecycle

A v1 trace describes one source-bound proposal:

1. **Proposed** — records source/candidate artifact digests and a subject commitment over the exact scope, protected set, requested change and edit plan.
2. **Previewed** — records that the same subject commitment was previewed.
3. **Approved, vetoed or bounded authority granted** — records a decision bound to that exact commitment. Human approval is permitted only in apply_approved mode; a policy grant is permitted only in bounded_exploration mode.
4. **Committed** — references the exact authorization event and exact candidate digest, and carries a preservation-verifier receipt reference.
5. **Undone** — references the exact commit and must restore the original source digest.

SuggestOnly and PreviewOnly traces are valid non-mutating terminal states. A veto is terminal for that proposal; a new proposal must receive a new trace and new subject commitment.

## Independent hard checks

The evaluator rejects:

- empty, duplicated or non-contiguous event identities;
- a broken or reordered SHA-256 event chain;
- a proposal whose source differs from the trace base;
- a no-op candidate with identical source and candidate digest;
- a proposal commitment that does not hash the complete subject;
- a preview or authorization bound to a different proposal commitment;
- approval in a non-approval mode or a bounded grant in a non-bounded mode;
- an expired authorization or a commit referencing the wrong authority event;
- a commit after veto, a commit in suggest-only/preview-only mode, or a commit of a different candidate;
- a commit with no preservation receipt/verifier identity;
- an undo that references the wrong commit or restores a different source digest.

No composite agency score is produced. A trace is either structurally valid for the declared lifecycle or it is rejected with explicit issue codes.

## Reproducible implementation

The isolated Rust 2024 workspace at tools/mel-agency-trace includes:

- a strict versioned JSON contract;
- deterministic subject and event-chain hashing;
- a replay evaluator and JSON report CLI;
- one positive apply/approve/commit/undo fixture;
- one valid suggest-only proposal/preview fixture;
- negative unit tests for stale approvals, wrong-subject approvals, veto bypass, no-op edits, missing preservation evidence, inexact undo, hash tampering, duplicate IDs/sequences and wrong authorization references;
- a bounded-exploration positive fixture constructed in the Rust test module.

The dedicated workflow is exact-head scoped and pins Rust 1.96.0. It performs formatting, lock generation, tests, warnings-denied Clippy, fixture execution and evidence upload without depending on the root workspace.

## Qualification targets (not achieved results)

- 100% acceptance of valid known-answer traces in all declared autonomy modes.
- 100% rejection of registered invalid lifecycle controls.
- 0 accepted commits after veto, after authorization expiry, or in suggest-only/preview-only modes.
- 0 accepted no-op candidates.
- 0 accepted commit without a subject-matched authorization and preservation receipt reference.
- 0 accepted undo that fails to restore the exact source artifact digest.
- 100% event-chain tamper detection in the deterministic test matrix.

Human-facing targets are intentionally not invented here. Proposal usefulness, perceived ownership, workload, musical relevance and preferred alternative presentation require musician pilots, blinded evaluation and preregistered analysis under #11319 / #10088.

## Claim ceiling and required next step

The evaluator checks internally consistent claims and receipt references. It does not:

- authenticate actor_id or verify the signature/issuer of an authority receipt;
- open the score, MIDI or audio bytes and independently recompute their claimed artifact digests;
- rerun the named preservation or undo verifier against those bytes;
- prove a preview was actually displayed to a musician;
- persist, commit or undo a Studio artifact;
- establish that an accepted proposal is musically compelling or preferred.

The authoritative production sequence remains MEL-013A: authenticate the actor, verify purpose- and artifact-scoped consent, resolve bytes from Studio-owned storage, recompute content identities, prepare an idempotent transaction, commit durable artifacts/lineage, and independently reload and verify. This reference harness must be integrated into that path only after the actual Studio transaction contract is available; it must not become a substitute authority.

Next research stage: bind a real Studio proposal/preview/approval/veto/commit/undo trace to actual artifact bytes and authenticated consent receipts, then pilot whether musicians prefer branching, preview-and-select or clarification in matched tasks. Keep synthetic conformance, production transaction verification and human outcomes as separate evidence classes.
