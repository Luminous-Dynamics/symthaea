# Bounded Agency Authority v0.2

## Status and lineage

This document defines the v0.2 successor candidate to draft PR #279 / `agency/authority-core-v0.1`.
It does not retroactively reinterpret v0.1. The v0.1 branch remains historical evidence for the earlier contract.

A second historical lineage must also be preserved: `agency/verified-authority-state-v0.1@9da533c672520b6292b28263d318a509b32639cc` developed substantially stronger trusted-time, verified-current-authority-state, action-accounting, checkpoint, Xenia, and crash-recovery boundaries. v0.2 deliberately does **not** collapse those verifier-owned layers back into the pure core.

## Purpose

`symthaea-authority` is a cognition-free, transport-free, I/O-free reference semantics crate for bounded positive authority records.
It does not learn, infer intent, decide utility, measure consciousness, authenticate current authority state, validate full delegation ancestry, execute effects, hold signing keys, or decide scientific truth.

The central separation is:

```text
capability record
    != verified current authority state
    != verified delegation ancestry
    != execution admission
    != effect
```

A serialized `CapabilityGrant` is authority data. It is not itself a live capability.

Likewise, `AuthorityEvaluationInput` and `NegativeAuthorityFact` are pure evaluator inputs. Constructing them does not prove their authenticity, freshness, completeness, source frontier, or trusted-time provenance.

## v0.2 invariants

1. **Confidence is not authority.** Phi, confidence, posterior probability, scientific support, utility, urgency, reputation, anomaly score, or predicted benefit never create a grant.
2. **Purpose is exact.** A grant is bound to one semantic `PurposeId`; delegation cannot change it.
3. **Context is exact.** A grant is bound to an `AuthorityContextRef { namespace, digest }`; delegation cannot change it.
4. **Evaluation is not verification.** `AuthorityDecision::Allow` means only that a root grant is eligible under the supplied evaluator inputs. It does not prove those inputs are authentic/current and does not create runtime authority.
5. **Verified current state remains external.** Trusted time, fresh challenge-bound current epoch, complete applicable negative facts, source-frontier provenance, and verifier independence remain the job of verifier-owned layers such as the historical `symthaea-authority-time` / `symthaea-authority-state` lineage.
6. **Delegation only attenuates, but attenuation is not ancestry proof.** A child may narrow authority but cannot broaden it. A single delegated record is not independently eligible: the generic evaluator returns `DelegationChainRequired` whenever `parent_digest` is present.
7. **Negative authority dominates.** Applicable revocation, tombstone, resource freeze, context revocation, or minimum-resource-epoch facts override an otherwise eligible positive grant.
8. **Epochs never resurrect.** A grant from a non-current authority epoch is denied by the pure evaluator. Higher layers remain responsible for proving the current epoch.
9. **Reserved uses are charged.** Durable in-flight reservations count against `max_uses` before dispatch. The pure core applies this arithmetic; a runtime/accounting layer must prove the supplied counters are the correct durable counters for the grant.
10. **Exact resources in v0.2.** Resource matching remains exact. Hierarchical semantics require an explicit future schema.
11. **Integer security budgets.** Authority budgets use integers and do not depend on floating-point canonicalization.
12. **Canonical commitments are domain separated.** Every authority-relevant field of `CapabilityGrant` contributes to its deterministic BLAKE3 commitment.
13. **Unknown/invalid schemas fail closed.** Structurally invalid records are denied before positive evaluation. Historical schema v1 does not silently parse as v2.
14. **Authority and execution remain separate.** The pure core exports no motor command, browser dispatch, system call, network action, live capability, or physical effect API.

## Purpose binding

`PurposeId` closes an ambiguity in v0.1 where exact resources and operations could still be reused across semantically different authority classes.

For example, an exact resource/operation pair used for `goal-directed-actuation` cannot be delegated into `safety-fallback` or `software-deployment` merely by retaining the same strings. Purpose is commitment-bound and invariant under delegation.

## Authority context

`AuthorityEpoch` and `AuthorityContextRef` solve different problems.

- `AuthorityEpoch` is the grant-generation value the pure evaluator compares with a supplied current epoch.
- `AuthorityContextRef` binds the grant to an exact domain-defined authority context without forcing every domain into one universal schema.

For example, swarm may bind a controller-lease/admission context while fabrication may bind its multi-dimensional monotonic authority vector. The shared crate does not interpret either context; it only requires exact identity and prevents delegation from silently moving the grant to another context.

A context digest being equal does not by itself prove that the context is current. That is a verifier/admission responsibility. The all-zero context digest is rejected as a placeholder rather than accepted as authority identity.

## Delegation boundary

`validate_attenuation(child, parent)` proves one static edge only:

```text
child <= parent
```

It does not prove:

- that the parent is currently valid;
- that every earlier ancestor is present and valid;
- that no ancestor has been revoked/tombstoned;
- that delegation escrow was durably reserved;
- that the issuing authority was authorized to delegate at the time of issuance;
- that the external authority source still recognizes the chain.

For that reason, the generic single-record evaluator refuses positive evaluation for any grant with `parent_digest.is_some()` and returns `DelegationChainRequired`.

A future/current-main delegation admission slice should consume the exact ancestry plus fresh verified authority state and crash-conservative delegation escrow before producing any opaque delegated execution authority.

## Why runtime/live admission is not in the core

The historical verified-authority-state work established a stronger result than a caller-constructed `current_epoch + negative_facts` object can provide:

```text
caller-provided current state
    != verified current authority state
```

That lineage uses fresh challenges, trusted authority time, multiple independent state witnesses, exact grant binding, complete relevant negative-fact snapshots, source-frontier commitments, and bounded freshness before producing an opaque verified-current-state capability.

v0.2 therefore keeps the pure evaluator deliberately below that boundary. A future/current-main admission tranche should consume freshly verified state and crash-conservative grant accounting, then mint any opaque runtime admission token there—not in this crate.

## Structural identifier discipline

The v0.2 grant schema rejects empty explicit principals, purpose/task identifiers, resource identifiers, operation identifiers, and placeholder authority contexts. This is intentionally stricter than accepting structurally present but semantically empty values at an authority boundary.

## Deliberate non-goals

v0.2 does not define signatures, authority-state witnesses, trusted-time acquisition, transport, persistence, source-frontier verification, full delegation-chain verification, runtime/live admission, reservation state machines, hierarchical resource scopes, domain-specific actuation envelopes, motor limits, browser URL policy, scientific evidence rules, multi-party operational quorum, OS sandbox compilation, OAuth/MCP mapping, or physical effect execution.

It also does not flatten stronger domain authority systems. Swarm controller leases, fabrication authority vectors, HAL policy, browser capabilities, Xenia verification, and the historical verified-authority-state stack should map into or consume this contract through separately qualified adapters while retaining their native semantics.

## Qualification seed: AUTH-TINY-001

The crate tests a synthetic `robot-1 / goal-directed-actuation / move` capability record and adversarial variants. Required properties include:

- exact context and epoch are required by the evaluator;
- revocation and context revocation dominate an otherwise eligible record;
- delegation cannot change purpose or authority context;
- a statically attenuated delegated record still cannot be independently evaluated as eligible;
- resource/operation/risk/use restrictions cannot broaden;
- committed plus reserved use accounting is conservative;
- changing purpose or context changes the grant commitment;
- schema v1 fails closed under the v2 evaluator;
- empty exact identifiers and placeholder context digests fail closed;
- structurally invalid grants fail before positive evaluation;
- issuer tombstones deny the record.

Passing these tests establishes only the deterministic reference semantics implemented by this exact crate revision. It does **not** establish that supplied authority-state inputs are authentic/current, delegated ancestry is verified, production deployment, cryptographic issuer authenticity, execution admission, scientific correctness, or safety of any physical system.

## Migration direction

1. qualify this v0.2 pure core on its exact head;
2. reconcile/import the historical verified-time / verified-authority-state / action-runtime boundaries onto current `main` as separate successor slices rather than reimplementing them inside the core;
3. add a verified delegation-chain + escrow admission boundary before any delegated grant can become executable;
4. add a behavior-preserving swarm adapter and a fabrication-context adapter without flattening either native authority model;
5. add an `EmbodimentBridge` v2 shadow path where Phi/safety/ethics are restrictions only and positive actuation requires separately admitted verified authority;
6. differential-test every platform before production activation;
7. use the same separation beneath SCI-000 so scientific claims can inform policy without ever becoming action capabilities themselves.
