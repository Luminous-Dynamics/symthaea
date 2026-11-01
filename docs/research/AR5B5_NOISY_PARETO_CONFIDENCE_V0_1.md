# AR-5B5 Noisy Pareto Confidence Analysis v0.1

**Status:** Rust analysis primitive committed; tests authored but not compiled/executed  
**Module:** \`crates/domains/symthaea-architecture-engineering/src/pareto_noisy_observations.rs\`  
**Tracking:** Issue #9659; parent Issue #9469

## 1. Why this exists

A deterministic objective vector is not an adequate representation of a noisy scientific experiment. Repeated measurements can change candidate ordering, feasibility, and Pareto membership. Reporting only sample means or the best observed replicate can therefore turn noise into apparent discovery.

The v0.1 module stores replicate-level objective observations and reports bounded-observation confidence intervals and confidence-aware Pareto relations. It does not collapse objectives into a scalar score.

## 2. Frozen observation contract

Each objective declares:
- objective ID and maximize/minimize direction;
- certified lower and upper bounds for an individual observation.

The bounds are properties of the observation process. They are **not** normalization anchors, and must not be inferred from the observed sample minimum/maximum.

The plan freezes:
- campaign ID;
- objective order and directions;
- family-wise error budget \`alpha\`;
- maximum number of candidate identities;
- minimum replicate count before confidence comparison.

Each replicate records a stable ID, sample seed, raw objective vector, and SHA-256 evidence reference. Duplicate replicate IDs or seed values within one candidate are rejected. A candidate decision digest cannot be aliased to multiple candidate IDs.

Unique seeds are provenance, not proof of statistical independence. The statistical claim is conditional on the actual observation process supplying independent bounded replicates for each candidate/objective sequence (or another justified sampling model). Correlated seeds, reused physical samples, drift, adaptive interference, or unmodeled shared noise invalidate the simple Hoeffding guarantee and must be modeled separately.

## 3. Time-uniform alpha spending

For \(N\) frozen maximum candidates, \(d\) objectives, family-wise error budget \(\alpha\), and replicate prefix \(n \ge 1\), allocate:

\[
\alpha_{\mathrm{coord}} = \frac{\alpha}{N d}
\]

\[
\alpha_n = \frac{\alpha_{\mathrm{coord}}}{n(n+1)}
\]

Since \(\sum_{n=1}^{\infty} 1/(n(n+1))=1\), the sum of the per-prefix error budgets for each candidate/objective is at most \(\alpha_{\mathrm{coord}}\). A union bound across the frozen candidate/objective family therefore spends at most \(\alpha\), conditional on the required bounded-observation concentration assumptions.

For observations \(X_i \in [a,b]\), sample mean \(\bar X_n\), and Hoeffding radius:

\[
r_n = (b-a)\sqrt{\frac{\log(2/\alpha_n)}{2n}}
\]

the interval is:

\[
[\max(a,\bar X_n-r_n),\;\min(b,\bar X_n+r_n)]
\]

This is intended to remain valid under optional stopping: a researcher may inspect the current interval after each replicate without repeatedly resetting a fixed-time confidence level. It is **not** valid merely because seed values are distinct; the observation process must meet the independence/boundedness assumptions.

The implementation caps the candidate family at one million identities to keep the floating-point alpha allocation in a controlled range. It also validates nonzero finite per-coordinate and per-prefix alpha.

## 4. Confidence-aware Pareto relations

Maximize objectives retain their interval direction. Minimize objectives are sign-transformed to maximization utility.

For candidate A to certainly dominate B, A's lower confidence bound must weakly exceed B's upper bound in every transformed objective and strictly exceed it in at least one:

\[
L_A \succeq U_B,\quad L_A \ne U_B
\]

Possible dominance checks whether A's upper interval can dominate B's lower interval.

The reported categories are:
- **InsufficientReplicates:** below the preregistered minimum.
- **CertainlyDominated:** at least one other eligible candidate certainly dominates.
- **CertainlyNondominatedAmongObserved:** no eligible candidate can possibly dominate under the current intervals.
- **Unresolved:** possible dominators exist, but certainty is not established.

"CertainlyNondominatedAmongObserved" is intentionally scoped to the observed, sufficiently replicated candidate set. It is not a claim of global Pareto optimality. The possibly-dominated list may include candidates that also certainly dominate; certainty and possibility are not mutually exclusive.

## 5. Complexity and restart behavior

The study stores candidates, replicates, seed indexes, and decision ownership in canonical ordered maps/sets. Appending a replicate validates the new observation and updates only the relevant indexes. A deserialized study replays and validates the entire retained record once before appending again. Full \`validate()\` and \`report()\` still reconstruct/inspect the retained state.

No runtime performance claim is made. The implementation is an analysis primitive and has not yet been wired into the optimizer or EventTape run loops.

## 6. Authored tests

- Summable alpha-spending schedule and positive-prefix validation.
- Reject candidate-family sizes above the frozen cap.
- Confidence intervals shrink as replicate count increases for fixed bounded observations.
- Separate certain dominance from unresolved trade-offs.
- Reject duplicate replicate IDs, duplicate seeds, decision aliases, and out-of-bound observations.
- Canonical report digests independent of observation insertion order.
- Rebuild validation indexes after serialization and require resumed/uninterrupted studies to produce identical reports.

These tests are authored but have not been compiled or executed.

## 7. Next integration and benchmark ladder

Tracked in Issue #9659:

1. Bind raw replicate artifacts and per-replicate resource receipts to EventTape/AnytimeHistory.
2. Add bounded noisy mathematical fixtures and heteroskedastic/correlated-noise negative controls.
3. Compare NSGA-II/NSGA-III/MOEA-D/SMS-EMOA/RVEA and Sobol/random allocation with Bayesian MOO references.
4. Pin ParEGO/qNParEGO, qEHVI/qNEHVI, JES/PESMO and TS-TCH/NParEGO configurations.
5. Evaluate noisy scaled DTLZ2 at 5 objectives and 10 objectives as separate suites; freeze exact scaling, noise process, reference front, reference point, resource budgets and independent-run design.
6. Add adaptive replication, decoupled evaluation, and expected-value-of-information sampling.
7. Report confidence coverage, target-attainment distributions, empirical attainment, all-in cost, worst-family behavior and held-out transfer.

References:
- COCO official benchmark archive: https://coco-platform.org/data-archive/index.html
- TS-TCH/random scalarization framework: https://proceedings.mlr.press/v115/paria20a.html
- BoTorch noisy, parallel MOO tutorial (qEHVI/qNEHVI/qNParEGO): https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo
- Noisy scaled DTLZ2 five-objective baseline discovery page (30 independent runs): https://www.sota2.com/research/sota/multi-objective-optimization-on-noisy-scaled-dtlz2-5-objectives
- Noisy scaled DTLZ2 ten-objective baseline discovery page: https://www.sota2.com/research/sota/multiobjective-optimization-on-scaled-dtlz2-10-objectives-hv

The leaderboard pages are baseline-discovery aids, not a substitute for reproducing exact task, scaling, noise, budgets, and reference data. No Symthaea benchmark win or SOTA claim is established.


## 8. Sampling, noise, and bound provenance

The plan also freezes:
- a non-empty sampling-protocol ID and SHA-256 digest;
- a non-empty noise-model ID and SHA-256 digest;
- a SHA-256 certificate reference for each objective's observation bounds.

These digests make the assumptions inspectable and prevent protocol/noise/bound metadata from changing without changing the plan digest. They do not authenticate the certificate bytes or prove the assumptions; the separate evidence-artifact resolver and a domain-specific independent reviewer must resolve and assess those certificates.

The plan caps the frozen candidate family at one million identities. This keeps the family allocation within a controlled floating-point range. It rejects per-coordinate or per-prefix alpha allocations that underflow.

## 9. Append and restart complexity

The study uses private candidate/sample/seed indexes. New study construction validates the frozen plan; incremental appends validate only the new replicate and update indexes. A study deserialized from bytes replays its complete contents once before it may append; full \`validate()\` and \`report()\` continue to inspect the complete retained state. This avoids validating the entire sample history on every append. No runtime throughput claim is made.


## 10. Schema version note

The expanded noisy observation plan/report schema is `0.2.0`. It binds sampling-protocol and noise-model identifiers/digests and each objective's bounds-certificate digest. Earlier `0.1.0` plans are not equivalent and must not be silently upgraded.
## 11. Anytime-history replay and resource accounting

Noisy candidate actions are now replayable through `AnytimeHistory`. The frozen history header contains an optional `noisy_observation_plan_sha256`; the noisy plan must be registered before the first event and every noisy observation event must match it. This avoids treating an event-supplied plan digest as post hoc preregistration.

`analyze_noisy_anytime_history(plan, history)` reconstructs the `NoisyParetoStudy` from successful `CandidateNoisyObservation` records. Failed, invalid and timed-out attempts are retained in the hash chain and charged in the generic anytime receipt, but are excluded from sample means and confidence intervals. The joint report binds:
- noisy plan digest;
- history tip digest and run ID;
- full anytime resource receipt;
- attempted, successful and failed replicate counts;
- confidence-aware Pareto report digest.

The adapter checks objective ID, order and direction equality between the noisy plan and history header. Deterministic and noisy evaluation modes cannot be mixed. The generic anytime attempt counter counts both modes.

**Important limitation:** the event chain retains objective vectors and evidence digests, but this adapter does not resolve external raw artifact bytes or independently verify their contents. A later replicate-specific byte-verification path must bind every sample digest to immutable bytes and a verifier receipt. The current joint report is reproducible from the event chain, not a claim of external artifact authenticity.
