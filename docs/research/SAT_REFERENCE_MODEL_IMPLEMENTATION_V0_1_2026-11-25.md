# SAT Rust Authority Reference Model V0.1 — 25 November 2026

**Status:** source implementation added; compile/test/CI status pending.  
**Parent research programme:** [SAT V0.1, PR #13574](https://github.com/Luminous-Dynamics/symthaea/pull/13574).  
**Implementation PR:** see the linked reference-model PR.

## Purpose

Provide a small, independently authored, dependency-free Rust oracle for
bounded authority fixtures. The oracle is intended to
challenge production implementations and adapters. It is not a second
production authority kernel and does not replace the fabrication-specific
delegation protocol, #4155 qualification, MACE receipt verification, or
independent evidence infrastructure.

## Implemented model surface

- Exact effect identity: action + resource + sink + purpose + caller-computed
  digest of canonical operation parameters.
- Payload-label containment: request labels must be a subset of the scope's
  allowed labels; child scopes may only narrow that set.
- Root grant insertion as an explicitly trusted fixture/upstream-verifier input.
- Parent-to-child delegation with exact parent identity, delegator identity,
  effect-scope subset, validity-window subset and decreasing delegation depth.
- Full-lineage authorization checks and ancestor revocation.
- Half-open validity windows: not-before inclusive, expiry exclusive.

## Reference tests

The initial unit-test set covers:
- no matching grant;
- exact effect authorization;
- changed action/resource/sink/purpose/parameter digest;
- delegated subset use;
- effect-scope expansion;
- lifetime expansion;
- delegation-depth expansion;
- delegator identity mismatch;
- multi-level delegation depth consumption;
- parent revocation invalidating descendants;
- half-open expiry;
- invalid delegation flag/depth combinations;
- non-delegable parent rejection;
- future grant rejection.

The 17 tests exercise a small reference-model subset. They do not reproduce
DelegateGuard's full dataset, tool adapters, adaptive attack campaign or
end-to-end enterprise workflows.

These are reference-model tests, not execution of the complete public SAT
JSONL corpus. In particular, the model does not implement auditor behavior,
adaptive agent execution, monitor integrity, human deliberation, distributed
revocation, persistence or external sink fencing.

## Comparison boundary: DelegateGuard

Tran-Truong et al., *DelegateGuard: Purpose-Bound Capabilities for Delegated LLM
Agent Workflows* (2026, [arXiv:2610.16005](https://arxiv.org/abs/2610.16005)), is
the closest direct comparator. It already implements purpose-bound attenuable
capabilities over holder, purpose, data labels, operations, sinks, delegation
depth and cascading revocation. SAT's Rust crate should not be described as
inventing those concepts; its current contribution is a narrow, independently
authored Rust oracle with exact operation-parameter binding and half-open
expiry, to be compared against the published semantics.

The paper also establishes a crucial limit: an authority monitor that ignores
payload content and volume cannot prevent arbitrary secret disclosure through
an authorized but externally observable sink. This crate does not solve that
problem. Content-aware declassification, connector-owned label provenance and
output monitoring remain separate qualification responsibilities.

## Why exact operation digests matter

A permission for an action and resource alone can be too broad for consequential
operations. For example, a generic transfer permission could be reused with a
different amount or recipient. The reference model therefore requires an
opaque digest of the complete canonical operation parameters as part of each
effect binding. The caller must supply a versioned canonicalization and digest
contract; this crate does not silently invent one.

## Composition with current work

- Mycelix OPSEC-000 [#2746](https://github.com/Luminous-Dynamics/mycelix/issues/2746) is the shared cross-project information-handling/sink/declassification vocabulary. Symthaea CogSec [#214](https://github.com/Luminous-Dynamics/symthaea/issues/214) owns conservative confidentiality propagation and explicit declassification; INTX-010 [#4453](https://github.com/Luminous-Dynamics/symthaea/issues/4453) owns egress policy distinct from remote-state mutation; INTX-010A [#4543](https://github.com/Luminous-Dynamics/symthaea/issues/4543) binds egress to exact lineage and InteractionIntent; INTX-017 [#4504](https://github.com/Luminous-Dynamics/symthaea/issues/4504) owns cross-connector lineage; INTX-013 / INTX-013A [#4491](https://github.com/Luminous-Dynamics/symthaea/issues/4491) / [#4594](https://github.com/Luminous-Dynamics/symthaea/issues/4594) own policy-decision evidence. The Rust oracle's label-set is only a bounded scope-conformance input; it does not classify, propagate lineage, declassify or authorize network egress.
- Symtropy [MISSION-SUPERVISOR-001 #6498](https://github.com/Luminous-Dynamics/symtropy/issues/6498) owns the mission/subgoal mutation state machine and ambiguity hold. Symtropy [MOBILITY-MUTATION-BOUNDARY-001 #6669](https://github.com/Luminous-Dynamics/symtropy/issues/6669) owns the authority-linearized native mutation boundary. The SAT reference crate deliberately does not duplicate these mutation semantics; SAT consumes them in its operational-control track.
- Mycelix [AGENT-009 #803](https://github.com/Luminous-Dynamics/mycelix/issues/803) and [AGENT-010 #830](https://github.com/Luminous-Dynamics/mycelix/issues/830) own durable effect-attempt and offline authority/evidence reconstruction semantics. Compose those contracts where the Mycelix agent runtime is evaluated.

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
behavioral oracle for bounded capability and delegation invariants. Mutation
reconciliation, sink linearization, and persistent effect-attempt evidence
remain owner-native to Symtropy and Mycelix.

## No-runner validation record

The current environment has no Rust compiler or Cargo executable. The manifest,
source and documentation were fetched back from the public branch and are linked
in draft PR #13602. No compile, unit-test, Clippy, rustfmt or CI pass is claimed. Next validation
must run on the exact PR head and record the Rust toolchain, command results
and artifact digests. If CI is queued or skipped, status remains pending.
