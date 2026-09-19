# Bounded Agency Authority v0.2

## Status and lineage

This document defines the v0.2 successor candidate to draft PR #279 / `agency/authority-core-v0.1`.
It does not retroactively reinterpret v0.1. The v0.1 branch remains historical evidence for the earlier contract.

## Purpose

`symthaea-authority` is a cognition-free, transport-free, I/O-free reference semantics crate for bounded positive authority and runtime admission.
It does not learn, infer intent, decide utility, measure consciousness, execute effects, hold signing keys, or decide scientific truth.

The central separation is:

```text
grant record != runtime admission != live authority != effect
```

A serialized `CapabilityGrant` is authority data. It is not itself a live capability. `LiveGrant` can only be constructed by the admission function after current epoch, context, expiry, use-budget, and negative-authority checks succeed.

## v0.2 invariants

1. **Confidence is not authority.** Phi, confidence, posterior probability, scientific support, utility, urgency, reputation, anomaly score, or predicted benefit never create a grant.
2. **Purpose is exact.** A grant is bound to one semantic `PurposeId`; delegation cannot change it.
3. **Context is exact.** A grant is bound to an `AuthorityContextRef { namespace, digest }`. A changed authority context requires readmission or a separately qualified compatibility transition outside this crate.
4. **Grant record is not live authority.** Deserializing or replaying a `CapabilityGrant` or `AdmissionReceipt` does not recreate `LiveGrant`.
5. **Runtime restart fences live authority.** Admission binds `LiveGrant` to a `RuntimeEpoch`; a different runtime epoch makes the live grant stale even when the underlying grant record remains eligible for fresh readmission.
6. **Delegation only attenuates.** A child grant may narrow audience, task, resources, operations, plan/world bindings, expiry, use count, delegation depth, and risk budget; it may not broaden them.
7. **Negative authority dominates.** Applicable revocation, tombstone, resource freeze, context revocation, or minimum-resource-epoch facts override otherwise valid positive grants.
8. **Epochs never resurrect.** A grant from a non-current authority epoch is denied. Restoring old persisted state does not restore current authority.
9. **Reserved uses are charged.** Durable in-flight reservations count against `max_uses` before dispatch.
10. **Exact resources in v0.2.** Resource matching remains exact. Hierarchical semantics require an explicit future schema.
11. **Integer security budgets.** Authority budgets use integers and do not depend on floating-point canonicalization.
12. **Canonical commitments are domain separated.** Every authority-relevant field of `CapabilityGrant` contributes to its deterministic BLAKE3 commitment.
13. **Unknown schemas fail closed.** Enforcement points reject grant schemas they do not understand.
14. **Authority and effect admission remain separate.** `LiveGrant` means the bounded positive authority record is currently admitted under this reference model. Higher layers may still require safety, information-flow, policy, quorum, physical-health, sandbox, or consequence checks before any effect.

## Authority context

`AuthorityEpoch` and `AuthorityContextRef` solve different problems.

- `AuthorityEpoch` fences stale grant generations and replay.
- `AuthorityContextRef` binds admission to a domain-defined authority snapshot without forcing every domain into one universal authority schema.

For example, swarm may bind a controller-lease context while fabrication may bind its multi-dimensional monotonic authority vector. The shared crate does not interpret either context; it only requires exact binding.

## Runtime admission

`RuntimeAdmissionContext` supplies:

- current runtime epoch;
- current authority epoch;
- exact current authority-context reference;
- trusted wall-clock time for grant expiry when configured;
- durable committed/reserved use accounting.

`admit_live_grant()` first evaluates the grant against current authority state and negative facts, then creates a non-serializable `LiveGrant` bound to the runtime epoch and exact context.

An `AdmissionReceipt` is auditable evidence that an admission occurred. It is serializable by design but carries **zero independent execution authority**.

## Deliberate non-goals

v0.2 does not define signatures, transport, persistence, trusted-time acquisition, hierarchical resource scopes, domain-specific actuation envelopes, motor limits, browser URL policy, scientific evidence rules, multi-party quorum, OS sandbox compilation, OAuth/MCP mapping, or physical effect execution.

It also does not flatten stronger domain authority systems. Swarm controller leases, fabrication authority vectors, HAL policy, and browser capabilities should map into or consume this contract through adapters while retaining domain-specific semantics.

## Qualification seed: AUTH-TINY-001

The crate tests a synthetic `robot-1 / goal-directed-actuation / move` grant and adversarial variants. Required properties include:

- exact context and epoch are required;
- stale runtime admissions are rejected;
- revocation dominates an otherwise valid grant;
- delegation cannot change purpose or authority context;
- resource/operation/risk/use restrictions cannot broaden;
- committed plus reserved use accounting is conservative;
- changing purpose or context changes the grant commitment.

Passing these tests establishes only the reference semantics implemented by this exact crate revision. It does not establish production deployment, cryptographic issuer authenticity, scientific correctness, or safety of any physical system.

## Migration direction

1. qualify this crate on its exact head;
2. add a swarm adapter without changing swarm behavior;
3. add a fabrication context adapter without flattening its authority vector;
4. add an `EmbodimentBridge` v2 shadow path where Phi/safety/ethics are restrictions only and positive actuation authority comes from an admitted grant;
5. differential-test every platform before production activation;
6. only then use the same authority boundary beneath SCI-000 advice/action separation.
