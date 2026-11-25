# SAT Rust Reference Model V0.1 — 25 November 2026

**Status:** source implementation added; compile/test/CI status pending.  
**Parent research programme:** [SAT V0.1, PR #13574](https://github.com/Luminous-Dynamics/symthaea/pull/13574).  
**Implementation PR:** see the linked reference-model PR.

## Purpose

Provide a small, independently authored, dependency-free Rust oracle for
bounded authority and ambiguous-outcome fixtures. The oracle is intended to
challenge production implementations and adapters. It is not a second
production authority kernel and does not replace the fabrication-specific
delegation protocol, #4155 qualification, MACE receipt verification, or
independent evidence infrastructure.

## Implemented model surface

- Exact effect identity: action + resource + caller-computed digest of the
  canonical operation parameters.
- Root grant insertion as an explicitly trusted fixture/upstream-verifier input.
- Parent-to-child delegation with exact parent identity, delegator identity,
  effect-scope subset, validity-window subset and decreasing delegation depth.
- Full-lineage authorization checks and ancestor revocation.
- Half-open validity windows: not-before inclusive, expiry exclusive.
- Per-operation mutation state: unknown outcomes block retries; confirmed
  no-effect permits a retry; confirmed effect blocks duplicate mutation.

## Reference tests

The initial unit-test set covers:
- no matching grant;
- exact effect authorization;
- changed action/resource/parameter digest;
- delegated subset use;
- effect-scope expansion;
- lifetime expansion;
- delegation-depth expansion;
- delegator identity mismatch;
- multi-level delegation depth consumption;
- parent revocation invalidating descendants;
- half-open expiry;
- unknown-outcome reconciliation;
- confirmed-effect duplicate blocking;
- invalid delegation flag/depth combinations.

These are reference-model tests, not execution of the complete public SAT
JSONL corpus. In particular, the model does not implement auditor behavior,
adaptive agent execution, monitor integrity, human deliberation, distributed
revocation, persistence or external sink fencing.

## Why exact operation digests matter

A permission for an action and resource alone can be too broad for consequential
operations. For example, a generic transfer permission could be reused with a
different amount or recipient. The reference model therefore requires an
opaque digest of the complete canonical operation parameters as part of each
effect binding. The caller must supply a versioned canonicalization and digest
contract; this crate does not silently invent one.

## Composition with current work

- [#4143](https://github.com/Luminous-Dynamics/symthaea/issues/4143) and
  [#4155](https://github.com/Luminous-Dynamics/symthaea/issues/4155): no
  authority from individual or collective intelligence.
- [#5439](https://github.com/Luminous-Dynamics/symthaea/issues/5439): separate
  cognition-derived caution from action permission.
- [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064):
  untrusted execution isolation prerequisite.
- [#13565](https://github.com/Luminous-Dynamics/symthaea/pull/13565):
  independent qualification exchange.
- [#13597](https://github.com/Luminous-Dynamics/symthaea/pull/13597):
  signer revalidation at use time.
- [#13599](https://github.com/Luminous-Dynamics/symthaea/pull/13599):
  decision sufficiency and outcome receipt verification.
- [#13600](https://github.com/Luminous-Dynamics/symthaea/pull/13600):
  frozen decision episodes and sealed evaluator manifests.

The reference model does not replace these mechanisms. It supplies a separate
behavioral oracle for a subset of authority and retry invariants.

## No-runner validation record

The current environment has no Rust compiler or Cargo executable. The files
were written to the public branch and read-back is required before PR creation.
No compile, unit-test, Clippy, rustfmt or CI pass is claimed. Next validation
must run on the exact PR head and record the Rust toolchain, command results
and artifact digests. If CI is queued or skipped, status remains pending.
