# CMAQ-010: Adversarial Evidence Integrity, Provenance and Replay Robustness

**Status:** protocol v0.1 + dependency-free Rust synthetic smoke harness; draft  
**Tracking issue:** [#9085](https://github.com/Luminous-Dynamics/symthaea/issues/9085)  
**Claim ceiling:** synthetic evidence-channel study only; not a cryptographic attestation, production sensor-security, safety-certification or SOTA result

## Why this layer matters

CMAQ-008 measures delayed/noisy feedback and the trade-off between false revocation and detection delay. CMAQ-009 adds bounded active probes. Both depend on a further premise: observations carry meaningful independent information and are current. CMAQ-010 attacks that premise.

This is the evaluator-of-the-evaluator problem in a more general form. It does not suffice to measure whether a planner detects a shift if the telemetry, probes or witnesses used to detect that shift can be cloned, replayed, correlated or selectively corrupted.

The current literature reinforces the need for this distinction. Closed-loop world-model evaluation is increasingly emphasized over prediction-only scores, while recent judge-robustness studies show that evaluator performance can degrade sharply under style and adversarial distribution shifts. Those papers motivate the protocol; this synthetic harness does not reproduce them.

## Threat model and lanes

The dependency-free Rust CLI runs **512 paired deterministic episodes**, each 128 steps long, with an abrupt shift at step 64 in non-stationary lanes.

- **Clean independent:** three independent provenance groups.
- **Common-mode duplicates:** three claimed identities repeat one underlying observation and share one provenance group.
- **Sybil fan-out:** six claimed identities fan out one underlying observation and share one provenance group.
- **One Byzantine source:** two honest independent groups and one source that inverts its observation.
- **Stale-safe replay:** after the shift, three sources replay safe evidence carrying epoch/context zero and reused nonces.

The hidden regime is never exposed to the policies. Signal probability is 0.01 before the shift and 0.80 after it. A model action has expected utility +0.03 when supported and -0.20 when unsupported; fallback yields zero. Those are synthetic design choices, not empirical domain rates.

The report records false revocation, post-shift detection coverage and p90 delay, model-action coverage, expected utility, evidence multiplicity, independent-group coverage, stale/duplicate rejection, and fail-closed rate.

## Prior art and matched comparator

A directly relevant paper appeared on **24 October 2026**: [Echo chamber dynamics and influence-aware evidence aggregation for epistemic integrity in knowledge-based systems](https://link.springer.com/article/10.1007/s44163-026-02245-5). It proposes an Influence-Indexed Evidence Graph, Effective Independent Evidence (EIE), and bounded confidence propagation (CONFPROP). Its EIE is a weighted sum of evidence independence, and its discount is `gamma = min(1, EIE / K)`. The paper also states a per-observation-window bound on raw evidence increments as an assumption for its tighter propagation bound.

CMAQ-010 adds two deliberately narrower sequential-detection comparators; these are **not reproductions** of the paper's graph-based method:

- `eie_gamma_discount_only`: applies the EIE-style `gamma` to the sum of signed log-likelihood evidence, but does not cap the number of reports in a window. This is an ablation designed to expose what happens when a deployment omits the raw-batch bound.
- `eie_gamma_bounded_batch`: applies the same `gamma`, while capping each observation window at `K=3` report increments. This operationalizes the paper's per-window raw-increment bound in a simple count-limited way.

The comparison matters: a discount factor alone is not the full safety contract. The provenance graph, evidence weights, freshness, and per-window ingestion bound all matter. The present harness uses a single provenance-group label rather than a full influence graph, so it should not be presented as a direct CONFPROP benchmark.

A subtle but important point: the paper's γ discount does **not** make a fully dependent batch equivalent to exactly one observation. With one effective independent source and K=3, six duplicate increments become (1/3) × 6 = 2 single-increment units. CONFPROP bounds the update only under its separate raw-batch ingestion assumption; it still globally caps confidence at 1.0. CMAQ-010 makes this visible: the discount-only arm admits six increments, while the count-capped ablation processes at most three per window and therefore admits at most one effective increment in this six-clone test. This is a sequential log-likelihood stress test, not the paper's positive-only confidence recurrence.

## Policies

1. `naive_claim_sum`: adds a likelihood increment for every report, even when reports share provenance.
2. `naive_claim_majority`: gives each claimed source identity one vote; aliases can still act as Sybils if not lineage-linked.
3. `eie_gamma_discount_only`: EIE-style discount without the per-window raw-batch cap.
4. `eie_gamma_bounded_batch`: EIE-style discount plus a `K=3` report-increment cap per observation window.
5. `provenance_root_dedup_sum`: counts at most one report per provenance group but does not check freshness.
6. `robust_root_majority`: one vote per provenance group and a conservative likelihood envelope intended to tolerate one inverted source; no freshness gate.
7. `freshness_provenance_fail_closed`: rejects stale epoch/context and duplicate nonces, counts one report per provenance group, requires at least two fresh independent groups, then uses the conservative majority statistic. Four consecutive steps without the minimum evidence cause fallback.
8. `oracle_changepoint`: evaluator-only upper bound at the hidden shift time.
9. `direct_fallback`: zero model-use baseline.

The synthetic `provenance_group` field is treated as authenticated metadata to isolate aggregation logic. CMAQ-010 **does not** prove that a production system can authenticate this field. Later work must test signed envelopes, key rotation, revocation, clock uncertainty, lineage registration and compromised roots.

## Interpretation and limits

A robust policy should not convert missing evidence into evidence of safety. A fail-closed fallback caused by insufficient lineage is reported separately from a statistical risk alarm. This distinction prevents conservative behavior from being misreported as successful change-point detection.

A green focused workflow means the source compiles, tests pass and the declared report invariants hold. It does not mean every performance target is met. No comparison to MRS.Q, World-In-World, ReliableBench or commercial systems is implied: these are different tasks and this harness is intentionally synthetic.

## Next experimental layer

- Replace the single Byzantine inversion with adaptive attacks that know the aggregation rule.
- Cross source correlation with independent failure, partial dependence, source dropout, delayed arrival and clock skew.
- Sweep contamination fraction, number of independent roots and minimum quorum.
- Compare clipped likelihoods, robust sequential tests, trimmed aggregation and dynamic quorum selection.
- Bind evidence to exact model/action/context digests and an append-only receipt; separately test cryptographic verification and anti-rollback.
- Port the attack lanes to named closed-loop benchmarks with equal compute, query, probe and intervention budgets.
