# QARCH-017: adaptive benchmark exposure protocol

## Status and claim ceiling

This document specifies the first deterministic policy gate for adaptive benchmark exposure. It is a local policy primitive, not a privacy mechanism, sealed-dataset service, cryptographic ledger, authenticated access-control system, or anytime-valid test implementation. The current Rust API returns an authorization decision and the ledger revision against which a caller must atomically reserve it.

A conforming production runner must keep confirmatory data inaccessible to the candidate-development process, authenticate and append reservations before evaluation, reject ledger rollback or concurrent double reservation, and retain tamper-evident receipts. A digest string by itself proves neither identity nor access isolation. The benchmark-family digest must commit to the actual independent family IDs defined by the QARCH-016 sampling design, not just instance rows.

No QARCH-017 policy result is evidence of quantum advantage. Confirmatory claims still require the QARCH-003 end-to-end resource contract, QARCH-014 signed-evidence/attestation requirements, QARCH-015 paired quality/resource gates and Holm correction, QARCH-016 family-level inference, and an independent acceptance verifier.

## Why exposure is a separate validity dimension

A fixed preregistered hypothesis family and a fixed sample do not describe an adaptive development process. Repeatedly inspecting a public benchmark, changing a candidate in response, and then reporting the best observed candidate makes the final score dependent on the feedback history. Holm correction over a final list of claims does not retroactively make the repeatedly queried benchmark an untouched holdout.

A reusable-holdout mechanism (for example, a properly implemented privacy-preserving mechanism or a Ladder-style restricted-feedback procedure) can permit controlled reuse under its own assumptions. This QARCH-017 increment does not implement either. It establishes a simpler boundary that can be audited now: separate public development, budgeted validation, and one-shot confirmation on a fresh family set.

## Evaluation lanes

| Lane | Permitted use | Confirmation status |
| --- | --- | --- |
| Development | Unrestricted exploration on public/development families; unlimited iteration may be allowed by the operator | Development only |
| Validation | Controlled feedback on a non-confirmatory family set, with a preregistered query budget | Validation only |
| Confirmatory | One evaluation of the exact frozen candidate manifest on the reserved, never-before-exposed independent-family set | Eligible for one-shot confirmation; authorization is not a pass |
| Sequential | No evaluation until an explicit anytime-valid method, spending plan, and stopping policy are registered | Fail closed |

Validation budget counts reservations for the same claim family, lane, and family-set digest. It is an operational exposure limit, not a statistical privacy guarantee. The lane and query ordinal are part of each request.

The confirmatory set is reserved by digest in the plan. Development and validation requests are rejected if they name that set. A confirmatory request must use the exact reserved digest, the frozen candidate-manifest digest, and ordinal one. Any earlier record naming that family-set digest makes it exposed, regardless of lane or claim family.

## Candidate and baseline freeze

The candidate manifest digest should cover all items that could be tuned after seeing validation feedback: source commit, build inputs, compiler/transpiler settings, circuit or algorithm parameters, backend selection, error-mitigation configuration, and orchestration policy. For a multi-candidate search, it must commit to the complete candidate family and selection rule, not only the eventual winner.

The baseline digest must bind the strongest relevant classical/hybrid baseline and the matched tuning/resource budget. The comparison-protocol digest binds workload, objective, quality metric, acceptance verifier, resource scope, exclusions, timeout/retry policy, and analysis plan. The verifier digest binds the independent acceptance implementation. Any drift requires a new plan and a new confirmatory family set.

Parent candidate digests record the development lineage. A supplied parent must appear in prior history. This is lineage checking, not code authentication; an external runner must verify that the actual executable bytes match the committed candidate manifest.

## Reservation and concurrency contract

The intended sequence is:

1. Validate the plan, request, and complete append-only reservation history.
2. Ask `authorize_adaptive_evaluation` for a decision.
3. Atomically append the request only if the ledger is still at `expected_ledger_revision` (compare-and-swap).
4. Only after successful reservation, release the corresponding benchmark access.
5. Append a result receipt bound to the reservation, exact candidate, benchmark-family-set digest, evaluator, and output artifacts.

Two callers can receive the same authorization from the same history snapshot. Therefore the API alone cannot prevent a race: only one compare-and-swap append at the expected revision may succeed. The runner must re-read and re-authorize after a lost race. No result may be revealed before the durable reservation is committed.

The ledger must be append-only and tamper evident. Restoring an earlier history, dropping records, editing a family digest, or changing the candidate after authorization must invalidate confirmation. A production implementation should use authenticated receipts and independent audit witnesses; a plain file or caller-supplied in-memory vector is not sufficient.

## Statistical composition

QARCH-017 is complementary to the other gates:

- QARCH-015 tests paired quality non-inferiority and practical resource superiority, then applies Holm to the entire preregistered confirmatory claim family.
- QARCH-016 makes workload family/generator lineage the independent sampling unit and limits generalization to the declared sampling frame.
- QARCH-017 records how candidate selection interacted with benchmark feedback and whether the confirmation families were exposed.

Do not run several candidates on the sealed set and choose the best. If the confirmatory protocol itself compares several candidates, freeze that complete candidate family and correct the full confirmatory claim family using QARCH-015. If evaluation is inspected repeatedly over time, use an explicitly registered anytime-valid method or alpha-spending plan; ordinary fixed-sample p-values must not be repeatedly peeked and selectively reported.

A fresh set of instances from an already-used generator family is not necessarily a fresh independent holdout. Confirmation must rotate at the independent-family level, with the lineage digest and QARCH-016 hierarchy in view. A fixed benchmark census without a sampling model remains descriptive, even when it is sealed.

## Illustrative adaptive-overfit calculation

For a fair-null paired sign test with 16 independent blocks, observing at least 12 wins has exact one-sided probability

`2517 / 65536 = 0.0384063720703125`.

If 50 independent null candidates are each tested against the same feedback set and only the best nominal result is reported, the probability that at least one crosses this threshold is

`1 - (1 - 0.0384063720703125)^50 = 0.8588831611775091`.

This is an analytic toy calculation under independent candidate-level null tests, not a quantum benchmark measurement and not a claim that real adaptive candidates are independent. Real candidates can be correlated, and the exact inflation depends on the adaptation and dependence structure. The example exists to show why a nominal p-value on an adaptively reused holdout is not a confirmatory result.

## Required benchmarks for QARCH-017

Before calling the policy implementation ready for an actual sealed evaluation service, test:

- 10, 100, 1,000, and 10,000 adaptive candidate submissions under null and alternative outcomes.
- Fixed public holdout reuse versus fresh independent-family rotation.
- Independent, positively correlated, and adversarially selected candidate score processes.
- Candidate-digest mutation, baseline retuning, verifier drift, protocol drift, and query-ordinal skips.
- Repeated family IDs disguised as new instances; family lineage aliasing and hash collisions.
- Duplicate reservations, concurrent compare-and-swap races, ledger rollback, record deletion/reordering, and confirmatory access before reservation.
- A confirmatory candidate selected after arbitrary development/validation queries, then tested once on genuinely fresh independent families.
- Any sequential stopping policy against fixed terminal tests, preregistered alpha spending, and an implemented anytime-valid alternative.

Report false-confirmation probability, score optimism, selection regret, exposure count, family-set reuse, unauthorized access rejection, reservation-race rejection, audit completeness, and power under matched query and compute budgets. Separate policy-gate rejection tests from statistical calibration: a correct authorization API does not itself prove inferential validity.

## Acceptance gates

- 100% rejection of the enumerated lane, digest, lineage, query-budget, duplicate-ID, and frozen-candidate negative controls.
- 100% rejection of confirmatory requests after any exposure of the sealed family-set digest in the authoritative ledger.
- Atomic compare-and-swap reservation demonstrated under concurrent callers; exactly one reservation wins.
- Zero confirmatory result release before durable reservation.
- Full history replay, append-only audit, and rollback/tamper detection exercised by an independent verifier.
- Adaptive null experiments report Monte Carlo uncertainty and compare all candidate-exposure policies on identical generated score processes.
- Sequential lane remains disabled until an anytime-valid method and stopping contract pass independent calibration.
- No quantum advantage or SOTA claim from policy tests, synthetic fixtures, or simulation-only results.

## References

- Dwork et al., *The Reusable Holdout: Preserving Validity in Adaptive Data Analysis*, Science (2015), https://doi.org/10.1126/science.aaa9375
- Blum & Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*, ICML (2015), https://proceedings.mlr.press/v37/blum15.html
- Grünwald et al., *Anytime validity is free: inducing sequential tests*, Journal of the Royal Statistical Society Series B (2026), https://doi.org/10.1093/jrsssb/qkag050

The module currently implements the deterministic lane/policy gate and adversarial unit fixtures only. It does not yet implement differential privacy, a reusable-holdout mechanism, an authenticated ledger, the concurrency adapter, secure data access, Monte Carlo adaptive-process simulations, or anytime-valid inference.
