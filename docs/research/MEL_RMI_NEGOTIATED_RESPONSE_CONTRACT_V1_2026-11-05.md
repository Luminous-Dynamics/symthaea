# RMI-Bench — negotiated response contracts v1

**Date:** 5 November 2026  
**Implementation:** `tools/mel-rmi-synthetic/src/lib.rs`  
**Parent protocol:** [RMI-Bench v1 preregistration](../domains/symthaea-muse/RMI_BENCH_V1_PREREGISTRATION_2026-11-02.md)  
**Related equivalence work:** [#9307](https://github.com/Luminous-Dynamics/symthaea/issues/9307) and [PR #9325](https://github.com/Luminous-Dynamics/symthaea/pull/9325)

## Research question

Can a musical agent distinguish *a valid response* from *the response the collaborator selected*, preserve plural options when preference is unresolved, and treat veto/clarification as first-class interaction outcomes?

A common evaluation mistake is to assume a single gold output is the unique correct answer. In music, an instruction may admit several valid voicings, motifs, roles or rhythmic responses. A candidate can be contract-valid while still violating a collaborator's explicit choice or prematurely collapsing an unresolved decision.

This module evaluates that interaction/authority distinction. It does not replace the score-level realization equivalence work in #9325, nor does it claim perceptual equivalence from symbolic feature arithmetic.

## Contract model

`NegotiatedResponseContractV1` declares a target feature dimension, a finite cyclic domain, and a non-empty set of acceptable nonzero offsets. Given an input feature, the contract enumerates all acceptable response values. These values form an unordered alternative set; the module makes no aesthetic ranking.

`PartnerAuthorityV1` contains:
- `selected_value`: an explicit partner choice, or `None` when preference remains unresolved;
- `vetoed_values`: options the partner has explicitly ruled out.

The agent returns exactly one disposition:
- `Commit(value)`: commit to one response;
- `Branch(values)`: preserve the complete set of viable alternatives;
- `AskClarification`: ask the partner to resolve an unresolved choice;
- `Abstain`: decline when no admissible response remains.

## Independent predicates

The evaluator reports each property separately:
- **contract_valid:** every proposed value is in the allowed response relation;
- **veto_respected:** no proposed value is vetoed;
- **preference_respected:** an explicit selection is followed exactly;
- **ambiguity_preserved:** unresolved multi-option states are fully branched or sent for clarification; single-option states can be committed; empty option sets require abstention;
- **action_well_formed:** branches contain at least two distinct values;
- **qualified:** all required predicates pass.

No scalar musical-quality score is produced. A contract-valid alternative may fail preference compliance. A high-quality-sounding or valid output cannot override a veto. A single commit in an unresolved multi-option state is marked premature even when the committed value is contract-valid.

## Adversarial coverage

The ten new Rust tests cover:
1. several distinct valid witnesses;
2. preserving all unresolved alternatives by branching;
3. asking for clarification;
4. rejecting silent single-choice commitment under unresolved ambiguity;
5. removing vetoed choices from the branch set;
6. respecting an explicit partner selection even when other alternatives are valid;
7. abstaining when no valid option remains;
8. rejecting duplicate and incomplete branches;
9. rejecting no-op-only and wrong-dimension contracts;
10. preventing clarification from substituting for an already explicit partner selection.

These are finite synthetic conformance tests. The accepted offsets and cyclic feature domain are not a model of general musical equivalence.

## Relationship to existing work

- **#9307 / PR #9325** handles score/event-level edit equivalence: multiple allowed pitch alternatives for a targeted event while preserving all protected score properties and rejecting ambiguous event identity. Reuse that verifier for real Score edits rather than duplicating its event identity logic here.
- **RMI-Bench #10088** asks whether an agent participates in contingent reciprocal interaction, disagreement and branching. This module adds a synthetic agency/authority slice for the "multiple valid next actions" case.
- **#10051 MEL-013A** remains the required path for binding real Studio artifacts, authenticated actors, consent receipts and durable lineage. Synthetic IDs and accepted-value sets are not real participant authority or signed consent.
- **#9184** remains the owner for blinded human calibration and unseen-renderer generalization. No perceptual-validity claim follows from these tests.

## Claim ceiling and next steps

Passing the tests establishes only that the finite negotiation evaluator distinguishes allowed alternatives, vetoes, explicit selections and unresolved ambiguity in the encoded fixtures. It does not establish that a response is musically compelling, that a musician prefers it, or that an agent understands a person's intent.

Next integration:
1. adapt the RMI action trace to the Score-level operation contract from #9325 where a concrete Score edit is involved;
2. store the partner's selection/veto/clarification event with the MEL-013A transaction and exact artifact digests;
3. run symbolic tasks with multiple human-annotated valid response sets;
4. pilot whether musicians prefer full branching, preview-and-select, or clarification under different time/interaction constraints;
5. preregister confirmatory autonomy, workload and preference outcomes before collecting human data.

No live Studio adapter, human study, SOTA comparison or musician outcome is claimed here.
