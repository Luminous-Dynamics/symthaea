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

## Scoped authority-event binding

The raw `PartnerAuthorityV1` is a small decision state, not a credential. The harness now wraps it in `PartnerAuthorityEventV1`, binding the preference/veto payload to:
- the exact session and task;
- the exact source observation event;
- a monotonically expected session-control sequence;
- an explicit monotonic clock-origin identifier;
- an effective time and exclusive expiry time;
- an opaque reference to the authority receipt that an external verifier must authenticate.

`assess_scoped_negotiation` refuses to score the decision unless these fields match the current observation and caller-provided sequence/clock context. It rejects cross-session/task/event replay, stale sequence, mismatched clock origin, not-yet-effective events, expired events, invalid time windows, and absent receipt references.

**Important trust boundary:** this is structural scope/time/replay validation, not signature verification or actor authentication. The opaque receipt reference is not verified by this crate. A production adapter must authenticate the referenced receipt, actor, and action/purpose scope through MEL-013A / the trusted identity layer before invoking the structural evaluator. A SHA-256 or opaque reference alone does not establish consent or authority.

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

The canonical runner emits eleven hand-constructed negotiation rows and the dedicated workflow gates:
- **6 qualified:** complete branch within budget, clarification, veto-compliant branch, following explicit partner selection, abstention after all options are vetoed, and clarification when the viable alternative set exceeds the branch budget.
- **5 rejected:** premature single commitment, a branch containing a vetoed value, fabricated commitment after all options are removed, incomplete branching, and attempted branch explosion above the configured fan-out limit.

The Rust unit suite also exhaustively enumerates 3,464 dispositions across every non-empty accepted-offset set for cyclic domains of size 2–5, every source value, every veto subset, and unresolved/explicit partner preference. Expected-valid and expected-invalid outcomes are checked separately and the exact state-space count is pinned. These are finite synthetic controls; they do not estimate how often real musicians prefer branching or clarification.

## Adversarial unit-test coverage

Thirteen new Rust tests cover multiple valid witnesses; full branching; clarification; premature single commitment; veto handling; explicit selection; empty-set abstention; duplicate/incomplete branches; no-op/wrong-dimension contracts; clarification after an already explicit partner selection; bounded branch fan-out; clarification instead of excessive branching; an exhaustive small-state model check of 3,464 dispositions, and six scoped authority-event tests.

## Relationship to existing work

- **#9307 / PR #9325** handles Score/event-level edit equivalence. Reuse that verifier for real Score edits; do not duplicate its event-identity semantics here.
- **RMI-Bench #10088 / PR #10093** measures causal adaptation and bilateral interaction. This module adds a synthetic agency/authority slice for multiple valid next actions, disagreement and branching.
- **#10051 MEL-013A** remains the required path for authenticated actors, purpose-specific consent, Studio-owned artifacts and transaction lineage. Synthetic feature IDs and local authority objects are not signed participant authority.
- **#9184** owns blinded human calibration and unseen-renderer generalization. No perceptual-validity claim follows from these tests.

## Claim ceiling and next steps

Passing this finite suite establishes only that the evaluator distinguishes admissible values, vetoes, explicit selections and unresolved ambiguity in the encoded fixtures. It does not establish that a response is musically compelling, that a musician prefers it, or that an agent understands a person's intent.

Next integration: bind the partner selection/veto/clarification event into MEL-013A transactions and actual artifact digests; compose RMI interaction decisions with #9325 score edit verification; then build multi-valid symbolic tasks with blinded musician annotations and preregistered autonomy/workload/preference endpoints.

No live Studio adapter, human study, SOTA comparison or musician outcome is claimed.
