# MEL RMI-Bench v2 — immutable policy identity

**Date:** 23 November 2026  
**Parent protocol:** [#10088 — causal reciprocal musical interaction](https://github.com/Luminous-Dynamics/symthaea/issues/10088)  
**Gap addressed:** [#10206 — immutable model identity separate from episode snapshots](https://github.com/Luminous-Dynamics/symthaea/issues/10206)  
**Implementation:** `crates/domains/symthaea-muse/src/rmi_benchmark.rs`

## Why a second identity boundary is necessary

RMI Track A already forks mutable episode state for the control, control replay, treatment, treatment replay and reversed-order probes. It checks the restored policy ID/version and exact serialized snapshot. That prevents an important class of state-order confounds.

It does not prove that those restored instances use the same immutable policy implementation. Two different model checkpoints, tokenizer tables, grammars, adapters, compiled bundles, configurations or runtimes can report the same human-readable ID and version while the mutable snapshot is byte-identical.

RMI v2 therefore separates:

1. **Immutable policy identity:** the content-addressed implementation and execution contract that should not change across counterfactual arms.
2. **Mutable episode snapshot:** the state that is expected to evolve during a trial.
3. **Run provenance:** exact source revision, compiler, dependency lock, host/target and the evidence bundle that produced the report.

None substitutes for the others.

## Immutable identity receipt

Every `RmiPolicyV1` and `RmiPartnerPolicyV1` implementation must supply an `RmiImmutablePolicyIdentityV1` independently of its mutable snapshot.

The payload contains:

- stable policy ID and version;
- identity kind;
- artifact/checkpoint/executable/deployment SHA-256;
- model weights digest where applicable;
- tokenizer and grammar digests where applicable;
- adapter, effective configuration and runtime digests;
- backend/device identity;
- stochasticity contract and seed authority;
- non-empty source/artifact references.

Identity kinds are explicit:

| Kind | Meaning | Minimum artifact evidence |
|---|---|---|
| `synthetic_rule_fixture` | Finite deterministic control used only to qualify the evaluator | Rule-descriptor digest; source/compiler/lock bound by run manifest |
| `content_addressed_model` | Learned model/checkpoint | Exact artifact and weights digests; optional tokenizer/grammar digests |
| `compiled_policy_artifact` | Executable or adapter bundle | Exact compiled artifact plus adapter/config/runtime digests |
| `external_hosted_service` | Remote service with immutable deployment identity | Deployment/artifact identity, endpoint/backend and effective configuration evidence |

The identity envelope recomputes SHA-256 over its serialized payload. A mutation to the payload without a matching digest is rejected. The policy ID/version must also match the concrete policy instance.

### Important synthetic-fixture boundary

Built-in RMI fixture policies report `synthetic_rule_fixture`. Their artifact digest identifies the named rule descriptor and configuration, **not the Rust source bytes, binary, or model weights**. The exact source revision, Rust compiler and Cargo.lock digest remain bound by the Track A run manifest and qualification workflow.

Production model adapters must not reuse a synthetic descriptor. They must construct a `content_addressed_model`, `compiled_policy_artifact`, or `external_hosted_service` receipt with the real immutable digests appropriate to that adapter. A missing model/checkpoint or unverifiable digest must make replay unavailable; it must not fall back to a newer artifact under the same label.

## Fail-closed evaluation sequence

For each agent and partner evaluation:

1. Read and validate the original immutable identity before invoking the policy.
2. Capture the mutable pre-trial episode snapshot separately.
3. Independently restore control, control replay, treatment and treatment replay.
4. Compare each restored instance's complete immutable identity payload and digest with the original, as well as checking ID/version and snapshot equality.
5. Repeat the immutable identity check for the reversed-order treatment/control probes.
6. Only after every pre-invocation identity and state check passes may the evaluator call `respond`.
7. Immediately after every agent and partner response—including both replay arms and both reversed-order probes—the evaluator recomputes and validates the immutable identity before another response can run or the trial can be scored. A response that changes its checkpoint, adapter, configuration, runtime or backend identity is rejected with a distinct during-response mismatch.
8. Include the immutable identity receipt in the agent evaluation payload. The reciprocal payload binds the agent identity digest and the partner identity receipt.
9. Hash the full evaluation payload, so the immutable identity is included in the replayable result digest.

A pre-invocation mismatch fails before action invocation; a during-response mismatch is detected immediately after that response and before the evaluator continues. A correct mutable snapshot cannot compensate for a different immutable model or adapter.

## Adversarial controls

The Rust tests include:

- same policy ID/version but different static artifact identity on the treatment restore;
- same partner ID/version but different static partner artifact identity;
- both restore mismatches rejected before the first action/partner response;
- an agent that changes its declared artifact identity inside `respond`, with the changed handle deliberately absent from its serializable episode snapshot, is rejected immediately after the control response;
- a partner that changes its declared artifact identity inside `respond`, with the changed handle deliberately absent from its serializable episode snapshot, is rejected immediately after the control response;
- model identity receipt digest recomputation after payload mutation;
- rejection of a content-addressed model without an exact weights digest;
- different adapter digests produce different immutable identity receipt digests;
- existing mutable-state, replay and arm-order negative controls continue to apply.

## Evidence levels and limitations

This increment is **digest-linked**, not signed or authenticated. It proves that the evaluator compared declared immutable identities and that report bytes bind those declarations. It does not prove an external signer controls the declared digest, that a remote service actually ran the declared deployment, or that the runtime prevented network/global-state side channels.

Compose authentication, signer roles, transparency evidence and execution isolation through [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254). Keep device-specific runtime behavior and external service nondeterminism separately qualified.

The new identity receipt is not proof of musical ability, causal relevance, human collaboration, safety, or a better musical outcome.

## Prospective qualification targets

These are design targets, not measured production results:

| Requirement | Target |
|---|---:|
| Counterfactual arms with exactly equal immutable identity receipt | 100% |
| Same-ID/version but different-artifact negative controls rejected | 100% |
| Restored-instance identity mismatch detected before invocation | 100% |
| During-response identity change detected before further invocation or scoring | 100% |
| Scored reports including the identity receipt digest | 100% |
| Missing model artifact silently substituted by a different version | 0 |
| Identity receipt tampering accepted | 0 |

The synthetic suite qualifies the contract. A production Melothaea adapter still needs an exact model/checkpoint resolver, immutable configuration/adapter receipts and an independently controlled runtime identity source.
