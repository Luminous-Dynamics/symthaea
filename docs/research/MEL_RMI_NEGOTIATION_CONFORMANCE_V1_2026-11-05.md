# RMI-Bench — negotiated response contracts v1

**Date:** 5 November 2026  
**Implementation:** `tools/mel-rmi-synthetic/src/lib.rs`  
**Parent protocol:** [RMI-Bench v1 preregistration](../../crates/domains/symthaea-muse/RMI_BENCH_V1_PREREGISTRATION_2026-11-02.md)  
**Related score-equivalence work:** [#9307](https://github.com/Luminous-Dynamics/symthaea/issues/9307) and [PR #9325](https://github.com/Luminous-Dynamics/symthaea/pull/9325)

## Research question

Can a musical agent distinguish a valid response from the response a collaborator selected, preserve plural options when preference is unresolved, and treat veto, clarification and abstention as first-class interaction outcomes?

A benchmark that assumes one gold output is the unique correct answer can wrongly penalize valid musical alternatives. Conversely, validity does not imply that a collaborator selected the output. A system can produce an admissible response while overriding an explicit partner preference. Those are different dimensions.

This module evaluates the interaction/authority distinction. It does not replace score-level realization-equivalence work in #9325, and it does not claim perceptual equivalence from symbolic feature arithmetic.

## Contract model

`NegotiatedResponseContractV1` declares a target feature dimension, a finite cyclic domain, and a non-empty set of acceptable nonzero offsets. Given an input feature, it enumerates all acceptable response values. The alternatives form an unordered set; this module does not rank them or claim equal aesthetic quality.

`PartnerAuthorityV1` contains an optional explicit `selected_value` and `vetoed_values`.

Agent dispositions:
- `Commit(value)`: commit to one response;
- `Branch(values)`: preserve the complete set of viable alternatives;
- `AskClarification`: ask the partner to resolve an outstanding preference;
- `Abstain`: decline when no admissible response remains.

## Independent predicates

The evaluator reports each property separately:
- **contract_valid:** every proposed value is in the allowed response relation;
- **veto_respected:** no proposed value is vetoed;
- **preference_respected:** an explicit selection is followed exactly;
- **ambiguity_preserved:** unresolved multi-option states are fully branched or sent for clarification; single-option states can be committed; empty option sets require abstention;
- **action_well_formed:** branches contain at least two distinct values;
- **qualified:** all required predicates pass.

No scalar musical-quality score is produced. A contract-valid alternative may fail preference compliance. A valid output cannot override a veto. A single commitment under unresolved multi-option preference fails the authority gate even when the selected value is contract-valid.

## Executable conformance

The canonical runner emits eleven negotiation rows and the dedicated workflow gates:
- **6 qualified:** complete branch within budget, clarification, veto-compliant branch, following explicit partner selection, abstention after all options are vetoed, and clarification when the viable alternative set exceeds the branch budget.
- **5 rejected:** premature single commitment, a branch containing a vetoed value, fabricated commitment after all options are removed, incomplete branching, and attempted branch explosion above the configured fan-out limit.

These are finite, hand-constructed synthetic controls. They do not estimate how often real musicians prefer branching or clarification.

## Adversarial unit-test coverage

Twelve new Rust tests cover multiple valid witnesses; full branching; clarification; premature single commitment; veto handling; explicit selection; empty-set abstention; duplicate/incomplete branches; no-op/wrong-dimension contracts; clarification after an already explicit partner selection; bounded branch fan-out; and clarification instead of excessive branching.

## Relationship to existing work

- **#9307 / PR #9325** handles Score/event-level edit equivalence. Reuse that verifier for real Score edits; do not duplicate its event-identity semantics here.
- **RMI-Bench #10088 / PR #10093** measures causal adaptation and bilateral interaction. This module adds a synthetic agency/authority slice for multiple valid next actions, disagreement and branching.
- **#10051 MEL-013A** remains the required path for authenticated actors, purpose-specific consent, Studio-owned artifacts and transaction lineage. Synthetic feature IDs and local authority objects are not signed participant authority.
- **#9184** owns blinded human calibration and unseen-renderer generalization. No perceptual-validity claim follows from these tests.

## Claim ceiling and next steps

Passing this finite suite establishes only that the evaluator distinguishes admissible values, vetoes, explicit selections and unresolved ambiguity in the encoded fixtures. It does not establish that a response is musically compelling, that a musician prefers it, or that an agent understands a person's intent.

Next integration: bind the partner selection/veto/clarification event into MEL-013A transactions and actual artifact digests; compose RMI interaction decisions with #9325 score edit verification; then build multi-valid symbolic tasks with blinded musician annotations and preregistered autonomy/workload/preference endpoints.

No live Studio adapter, human study, SOTA comparison or musician outcome is claimed.
