# NAS action preflight verifier v0.1

Tracking: [NAS-AUTHZ-001 / #9467](https://github.com/Luminous-Dynamics/symthaea/issues/9467)

Implementation branch: `research/nas-resource-reservation-v0.1`

## 1. Purpose and claim boundary

The NAS pipeline now has independent primitives for Pareto portfolio value-of-computation (#9419), unit-safe vector resource budgets (#9422), anytime bounded-mean evidence (#9445), preregistered confidence-family history (#9465), and run-wide pruning-risk audits (#9329 on top of #9104).

The preflight verifier binds these primitives to one exact candidate action. It recomputes the portfolio measurement decision from the actual posterior, portfolio family, QMC design, and action; validates the exact objective registry, hardware profile, measurement protocol, instrument; recomputes the vector-resource frontier; and verifies the confidence ledger stream derived from the full action scope.

**A successful receipt is preflight eligibility only.** It is not an atomic resource reservation, signed capability, execution authorization, or proof that the upstream measurements are truthful. A concurrent caller could otherwise pass preflight against stale budget state. An execution service must atomically reserve every resource vector and idempotently commit the action before any external measurement or destructive decision.

## 2. Measurement scope binding

The `NasMeasurementActionScope` commits to:

- run ID and exact Git commit object ID;
- action ID, candidate ID, and exact candidate configuration SHA-256;
- selected objective and the complete sorted objective coordinate order;
- objective-registry SHA-256;
- posterior and candidate-portfolio-family SHA-256 values;
- full QMC design digest and disjoint selection/validation replicate-set digests;
- exact physical hardware-profile, measurement-protocol, and instrument-identity digests;
- measurement action's likelihood-model digest;
- resource registry, budget state (including spent values and protected reserves), complete action-cost-vector digest, and the exact compare-and-swap reservation-ledger head observed before preflight.

The confidence stream ID is derived as `measurement:<scope_sha256>`. The manifest family ID must equal the run ID. This prevents replaying an otherwise valid confidence stream against another candidate configuration, device, protocol, posterior, QMC split, or resource budget.

## 3. Recomputed measurement preflight

The verifier performs the following independent checks:

1. Recompute the action-scope digest and validate every digest format.
2. Validate the objective registry and resource registry.
3. Find and revalidate the exact hardware profile; bind firmware, runtime, compiler/toolchain, quantization, board revision, instrument ID/model, and sampling frequency through the profile digest.
4. Require the requested objective to be a physical or imported-physical measurement objective bound to the exact profile and protocol.
5. Recompute objective coordinate order and compare it to the scope and posterior.
6. Match action ID, candidate ID/config SHA, objective, profile, protocol, and observation likelihood digest.
7. Recompute portfolio-class Pareto regret and measurement value-of-computation from the supplied posterior, portfolio family, feasibility config, immutable QMC design, and exact action. Compare all output digests with the scope.
8. Reconcile the objective acquisition-cost unit/quantity with exactly one matching resource-cost component.
9. Recompute resource registry validity, budget digest, and the non-dominated resource action frontier; the action must be present, resource-feasible, and benefit-eligible.
10. Replay the complete anytime confidence ledger. Require a preregistered stream ID derived from the exact scope, a non-empty history, a strictly positive lower bound, an `AT_OR_ABOVE_THRESHOLD` decision, and an estimate that matches the recomputed portfolio regret reduction.
11. Emit a domain-separated preflight receipt binding the evidence digests.

An inconclusive, below-threshold, stale, mismatched, missing, malformed, or non-frontier action is rejected.

The resource frontier currently has a fixed 1.96 normal-approximation LCB. This is still only an allocation prefilter; the independent anytime ledger is an additional gate and does not silently convert the resource frontier's normal interval into a sequential confidence guarantee.

## 4. Pruning preflight

The pruning preflight consumes a live `RunWidePruningRiskLedger` object and:

- independently invokes its integrity replay;
- checks that the run ID matches and the ledger still permits pruning;
- requires the latest risk event to carry an event ID derived from the complete prune scope;
- checks exact target commit, candidate/configuration, selected-frontier digest, and pruned-population digest through that scope-derived event ID;
- checks the event digest and its `pruning_remains_enabled` terminal state.

This composes the exact finite-population sentinel audit from #9104 with the predictable run-wide confidence-spending ledger from #9329. It does not certify top-K/global-optimum retention. The underlying pruning-audit documentation's claim ceiling remains in force.

## 5. Promotion remains blocked

Candidate promotion is deliberately rejected until #9388 supplies a private, non-serializable, exact-scope `VerifiedVoCQualification` capability produced only by successful replay of the empirical qualification receipt from #9384. A boolean, a caller-provided SHA string, or a hand-built JSON receipt cannot substitute for that capability.

This PR does not implement a promotion bypass. It explicitly fails closed.

## 6. Compare-and-swap reservation reference implementation

The follow-on [#9485 / NAS-RESERVE-001](https://github.com/Luminous-Dynamics/symthaea/issues/9485) adds `resource_reservation.rs` and a private-field, non-serializable `VerifiedMeasurementReservation` capability issued only by `prepare_measurement_reservation` after full preflight. The measurement scope now includes the exact reservation-ledger head that the preflight observed.

The reference ledger provides:
- exact-head compare-and-swap reservation;
- action-scope digest as idempotency key;
- complete integer resource-vector accounting after base spend, protected reserves, active reservations and committed consumption;
- one-way COMMITTED state, pre-commit RELEASED state, and monotone-epoch EXPIRED state;
- domain-separated event chain, record/ledger digest, and independent replay;
- stale competing preflight rejection and lost-response idempotency.

This is still an in-process reference state machine. It is not a distributed or durable transaction service, does not authenticate callers, and does not provide trusted time or crash recovery. Production must move the expected-head check and all resource-coordinate reservations into one durable serializable transaction or dedicated service.

## 7. Production transactional adapter is still required

The #9485 reference ledger now performs an expected-head compare-and-swap inside one in-process mutable state machine, atomically applies each transition to a candidate copy, and replays a hash-chained history. It prevents stale concurrent callers in that state machine: only one distinct reservation can mutate from a given revision head. It also provides exact-scope idempotency, bounded leases, release before commit, and irreversible resource consumption at commit.

This does **not** provide cross-process or distributed atomicity. The `&mut self` API serializes operations only within one process; it has no durable database transaction, authentication, trusted clock, write-ahead log, or crash recovery. A production adapter must execute the expected-head compare-and-swap and all resource-coordinate holds in one durable serializable transaction or dedicated service, with authenticated callers, idempotent retries, lease recovery, and a write-ahead log. No production concurrency-safety claim can be made from the reference model alone.

The reference transition is `RESERVED -> COMMITTED | RELEASED | EXPIRED`. COMMITTED consumes the full reservation conservatively and can never be refunded automatically.

## 8. Tests and limitations

Twelve Rust unit tests are authored for scope mutation, commit/candidate replay, event-ID binding, valid and disabled run-wide pruning events, promotion blocking, objective-order hashing, instrument identity mutation, malformed scopes, and receipt digest integrity.

The successful measurement path includes an authored exact-fixture integration test that exercises preflight -> reservation -> commit across the upstream crates. The reservation state machine has a separate 15-test suite. Neither suite has been compiled or executed: the current authoring environment lacks `cargo`, `rustc`, and `rustfmt`; no compile/test/CI result is claimed.

The preflight receipt digest detects accidental or malicious field edits only when the consumer independently recomputes the receipt and trusts the evidence source. It is not a signature or authorization capability. Consumers must re-run the full preflight against exact source evidence, not accept an externally supplied receipt on its hash alone.

## 9. Current research baseline and new RQMC direction

Industry baselines:
- ConNAS (ICLR 2026) reports architectures within 0.14% of the optimal feasible NATS-Bench architecture under its tested constraints: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c9f851081499ed5c60c51e312d3235f-Abstract-Conference.html
- EdgeVolution (Communications Engineering, 23 June 2026) is an end-to-end hardware-in-the-loop microcontroller NAS and deployment workflow: https://doi.org/10.1038/s44172-026-00708-2
- NATS-Bench-MCU v2 physically measures 14,281 successful nRF5340 deployments and records 1,344 flash-infeasible candidates from 15,625 topologies: https://doi.org/10.5281/zenodo.21644470

Important additional statistical result:
- Jain, Hickernell, Owen and Sorokin, *Empirical Bernstein and betting confidence intervals for randomized quasi-Monte Carlo*, Information and Inference: A Journal of the IMA, 15(1), March 2026, DOI 10.1093/imaiai/iaag003; preprint https://arxiv.org/abs/2504.18677.
- The paper studies empirical-Bernstein and hedged-betting confidence intervals for bounded RQMC integrands. For total budget (N=R n) split into (R) independent RQMC replicates of (n) points each, it derives an oracle Bennett-interval allocation (n=Theta(N^{1/(\theta+1)})) under RQMC variance (Theta(n^{-\theta})), and reports HBCIs often narrower than EBCIs in its experiments.
- This directly motivates a two-level optimizer for Symthaea: choose both the number of independent randomized replicates (R) and the point count (n) per replicate, rather than assuming a fixed 16-scramble QMC design. It is not safe to copy the asymptotic rate without verifying smoothness/variance assumptions for our discontinuous hypervolume indicator and the actual randomization scheme. A NAS-specific coverage and cost experiment is required.

## 10. Qualification and comparison targets

Use matched vector-resource budgets and fixed held-out tasks. Report:
- held-out hypervolume and portfolio-class regret;
- feasible-architecture recall and false-feasible rate;
- recall/precision of the physical Pareto set;
- total independent hardware/QMC replicate count;
- total integrand evaluations per replicate;
- wall time, GPU/CPU time, device occupancy, instrument time, energy, money, operator time;
- confidence-sequence width, coverage, false promotion/pruning, and inconclusive rate;
- end-to-end cost to reach a frozen regret target;
- exact deployment constraint satisfaction and reproducibility under a separate hardware session.

Compare random search, regularized evolution/NSGA-II, successive-halving/multi-fidelity NAS, predictor/zero-cost NAS, BoTorch qEHVI/HVKG, ConNAS, and EdgeVolution where protocols permit. Never compare raw headline values across different tasks/search spaces as if they were matched.

Aspirational, not achieved:
- at least 95% simultaneous coverage and <=5% family-wise false decisions under a preregistered 5% family budget;
- at least 20% fewer independent measurement replicates than the Hoeffding reference after a qualified variance-adaptive sequence is added;
- at least 10% lower total vector-resource cost to a fixed held-out Pareto-regret target than the strongest matched-budget baseline;
- no increase in false-feasible rate, constraint violation, or calibration error.

The first physical benchmark target remains the v2 publication export. The current environment could not retrieve its 972 kB `export.tar.gz`; no dataset bytes or local SHA-256 have been acquired here.
