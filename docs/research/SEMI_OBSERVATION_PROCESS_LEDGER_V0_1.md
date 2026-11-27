# SEMI-OBSERVATION-PROCESS-001 — joint acquisition, qualification, tracking and audit ledger v0.1

**Tracked issue:** [#13913](https://github.com/Luminous-Dynamics/symthaea/issues/13913)  
**Implementation:** `symthaea-semiconductor::benchmark::observation_process_ledger`  
**Status:** Rust reference ledger with authored tests; exact-head validation pending.

## Research thesis

The reason a VM system lacks a label is not one generic “missing” event. At least four distinct mechanisms exist:

1. **Physical acquisition:** the wafer was never selected, is still pending, failed acquisition, or was physically measured.
2. **K1 evidence qualification:** a physical result is eligible, UNKNOWN, BLOCKED, or still pending review.
3. **Online feedback retention:** a qualified physical result was retained and delivered to the risk controller, denied by capacity, not selected by the proxy tracker, evicted, expired, or forfeited.
4. **Independent audit:** a separate random audit may observe an otherwise unmeasured or nonqualified population unit through an independent reference method.

Those mechanisms have different probability laws, time semantics, authorities and estimands. Their probabilities must not be collapsed into a single inverse-probability weight.

## Composed components

This branch starts from the exact fixed-delay ACI implementation and includes the bounded-delay CA-OCRC and selection-risk modules:

- Fixed-delay τ-DACI: PR #13614, exact head `9b7d22cbd5eb8aa22549c7d58edf480e8b128041`, SEMI-BENCH-001 run #1340 passed.
- Bounded-delay CA-OCRC: PR #13708, exact head `25bd7733e468f7bceb3d99f27e7e532fcc812393`, run #1497 passed.
- Population partial-identification and independent audit estimator: PR #13650, exact head `ec432fe1934da712ad2e32323e2a057ce9579723`, run #1491 passed.
- Numeric/revision-bound label qualification, streaming adapter and model lineage are retained from the common evidence chain (#13467, #13478, #13465).

ACI and CA-OCRC are alternative risk controllers. A run declares exactly one controller kind. The observation ledger does not apply both recursions to the same risk state or merge their theoretical guarantees.

## Four independent ledgers

### 1. Operational physical acquisition

For every prediction round, record:
- acquisition state: NOT_SELECTED, SELECTED_PENDING, MEASURED or MEASUREMENT_UNAVAILABLE;
- selection mechanism: deterministic/adaptive or randomized with a known inclusion probability;
- selection design digest and per-round probability when the design justifies it;
- acquisition cost and event/measurement times;
- full operational measurement-method identity;
- operational measurement method identity.

A deterministic or active policy must not attach a guessed statistical inclusion probability. The recorded selection probability is descriptive evidence and is not automatically applied as a population-risk weight.

### 2. K1 evidence qualification

Record the exact label ID, revision, full payload digest, measurement-event ID, K1 report digest, report availability time, qualified loss and loss-evidence receipt where eligible.

- ELIGIBLE requires a K1 report and a bounded loss in [0,1].
- UNKNOWN/BLOCKED carries the report identity but no qualified loss.
- PENDING is distinct from UNKNOWN and BLOCKED.
- An invalid or revoked physical measurement is never transformed into a valid label by inverse-probability weighting.

### 3. Controller feedback tracking

The run declares either:
- `fixed_delay_aci`: unlimited fixed-delay feedback with ACI phase state;
- `capacity_aware_ocrc_bounded`: finite-capacity arbitrary-bounded delay with proxy tracking;
- `none`: no online risk controller.

For ACI, the ledger records unlimited pending/received/forfeited states and checks the fixed delay. For CA-OCRC, it records proxy delay, survival probability, tracking slot, actual delay, slot end round and round-action receipt. The evaluator reconstructs maximum concurrent slot occupancy and checks it against K.

A capacity-denied or proxy-not-tracked outcome is not the same as a wafer that was never physically measured. A K1 BLOCKED/UNKNOWN result is not the same as an outcome that was lost to a tracking proxy.

### 4. Independent audit

An independent audit record binds:
- a predeclared audit design digest;
- randomization provenance;
- an independently traceable reference method;
- per-round inclusion probability;
- audit loss and measurement receipt.

The audit method identity must differ from the operational measurement method on an audited row. The design must declare independence from operational qualification. This metadata is necessary but does not itself prove physical independence; that requires a real audit design and reference measurement process.

## Population-risk estimands

The ledger feeds the existing selection-risk estimator and reports separate results:

- qualified-subset risk;
- worst-case population-risk identification interval, assuming only that missing losses lie in [0,1];
- optional independent Bernoulli audit Horvitz–Thompson estimate, design-based variance, effective sample size and conservative Hoeffding interval.

No population point estimate is emitted from a qualified-only subset while missing outcomes remain. A random-audit estimate is available only under the declared independent Bernoulli design and positivity conditions. The normal interval is descriptive; the conservative Hoeffding interval is the finite-sample bound in this implementation.

## Non-composable weights — hard rule

The following quantities are not interchangeable:
- operational physical-metrology inclusion probability (p_{select});
- CA-OCRC proxy survival (F(d));
- audit inclusion probability (p_{audit});
- deterministic K1 evidence qualification.

This V1 ledger records them separately. It does not multiply them. In particular:
- CA-OCRC's inverse-survival correction only addresses its own delayed-feedback tracking mechanism;
- it does not correct active physical-metrology selection;
- it does not correct outcome-dependent K1 qualification failures;
- an independent audit's Horvitz–Thompson estimator is a separate population estimator and is not part of the online controller's update.

Any future joint weighting scheme needs an explicit joint sampling/retention model, dependence analysis, positivity conditions, bounded weights and a new derivation. It must not be introduced as a convenient product of existing probabilities.

## Controller assumption state

The report distinguishes:
- NOT_APPLICABLE;
- COMPLETE_UNDER_DECLARED_ASSUMPTIONS;
- DECLARED_BUT_NOT_VERIFIED;
- INVALIDATED_BY_MISSING_OR_UNQUALIFIED_OUTCOME;
- DELAY_CONTRACT_EXCEEDED;
- TRACKING_CAPACITY_EXCEEDED;
- TRACE_INCOMPLETE.

The delay-contract digest is a declared contract, not proof of real fab latency. Complete controller state means the ledger rows satisfy the declared software constraints; it does not authenticate the physical source or prove the external delay distribution.

## Bitemporal availability contract

The ledger records three distinct clocks for a physically measured row: measurement time, label-availability time and K1-report-availability time. A controller-received outcome has a fourth timestamp. The order is enforced as:

`physical event <= measurement <= label available <= K1 report available <= controller receipt`

Prediction time must not precede the event, and controller receipt must not precede prediction. When a label is available but untracked, its physical/K1 evidence still remains in the offline population ledger; the controller receipt timestamp remains absent. This is how physical availability and online retention remain distinct.

## Frozen adversarial controls

1. Not selected vs selected-pending vs measured-unqualified must produce distinct funnel counts.
2. K1 UNKNOWN/BLOCKED cannot carry a qualified numeric loss.
3. Capacity denied cannot be reclassified as physically unmeasured.
4. Proxy-not-tracked uses proxy delay -1; it cannot be confused with a capacity denial.
5. Tracked feedback must arrive within its sampled proxy horizon.
6. Proxy expiry and early eviction must release capacity at their exact recorded rounds.
7. Reconstructed peak occupancy may not exceed K.
8. Fixed-delay ACI may not use CA-OCRC tracking states and vice versa.
9. ACI missing/unqualified/forfeited feedback invalidates the complete-feedback guarantee state.
10. CA-OCRC with missing physical/K1 outcomes cannot claim its full-population feedback assumptions.
11. Independent audit cannot sample a round already counted as qualified operational evidence.
12. An audit method identical to the failed operational method is rejected.
13. A selection propensity cannot be attached to a deterministic/adaptive rule.
14. Row/input order permutations preserve the sealed report digest.
15. The audit estimator can tighten a population interval only through its own independent design; it cannot rewrite K1 eligibility;
16. no IPW product across selection, qualification, tracking and audit channels is emitted;
17. physical label availability cannot predate measurement;
18. K1 report availability cannot predate physical label availability;
19. controller feedback receipt cannot predate prediction, label or K1 report;
20. capacity overrun, declared-delay violation and incomplete closeout are reported as distinct assumption states.

## Benchmark targets (hypotheses)

- 100% disjoint population partition: each prediction round exactly once.
- 100% consistency between acquisition state and qualification evidence.
- 100% no controller update from UNKNOWN/BLOCKED/PENDING labels.
- 100% rejection of mixed ACI/CA-OCRC controller state.
- 100% capacity compliance for the reconstructed CA-OCRC tracking ledger.
- 0 population point estimates from qualified-only observations when outcomes remain unidentified.
- 100% worst-case [0,1] bounds contain the ground-truth simulated population risk under MNAR selection/qualification failures.
- Independent Bernoulli audit estimates attain their preregistered design-based confidence coverage in simulation.
- Separate reporting for D1/D2 or other fab datasets; no pooling normalized errors or estimands across different physical target units.

## Competitive comparison

The resulting benchmark is not another leaderboard score. It provides a common substrate for fair comparison of:
- GRAPE's zero-delay online adapter and a delayed evidence-gated adapter;
- fixed-delay τ-DACI;
- bounded-delay CA-OCRC with finite K;
- physical metrology policies (fixed/random/EVSI/risk-based);
- independent population audit designs.

It extends the public research direction toward NIST's TIDAL manufacturing architecture, which explicitly treats metrology context, contextualized data, human-machine teaming, integration test benches and digital-twin VVUQ as interdependent parts of trustworthy manufacturing intelligence. <https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing>

## Authored tests

Nineteen unit tests exercise the four funnel stages, capacity denial, proxy nontracking, ACI/CA-OCRC separation, fixed-delay missingness, exact tracked receipt and occupancy, capacity overrun, delay-contract violation, K1 timing, audit-method independence, loss withholding and canonical replay. The exact-head formatter, Clippy, crate tests and SEMI-BENCH-001 reference oracle are still required.

## Claim ceiling

This is synthetic/software ledger validation. It does not show that a production fab's selection probabilities are known, that audit reference methods are independent in practice, that actual feedback delays obey a contract, or that population coverage is guaranteed under selective missingness. It does not authorize skipped inspections, wafer disposition, recipe changes, APC/R2R, equipment control or autonomous process action.
