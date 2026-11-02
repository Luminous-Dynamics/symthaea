# connectome-query-impact

A dependency-free Rust interval kernel for measuring how a proposed proofreading correction could change pre-registered scientific query endpoints.

## Paradigm shift

A local segmentation edit is not scientifically important merely because a detector is uncertain, a mask is large, or a neuron has high degree. Its importance depends on whether the correction could materially change a *named scientific result*.

For each pre-registered query and admissible structural scenario, supply:
- baseline endpoint interval before the candidate edit;
- hypothetical endpoint interval after the candidate edit;
- metric ID and units;
- a positive materiality threshold in that metric's own units;
- an optional pre-registered decision threshold;
- query-definition digest and structural scenario ID.

The crate returns a vector of query effects. It never blends different query units into a single arbitrary score.

## Interval arithmetic

For baseline interval (B=[b_L,b_U]) and hypothetical post-edit interval (A=[a_L,a_U]), the conservative rectangular outer bound on the change is:

[
A-B=[a_L-b_U, a_U-b_L].
]

For effect interval (E=[e_L,e_U]):

[
\text{minAbs}(E)=
\begin{cases}
0 & 0\in E\\
\min(|e_L|,|e_U|) & \text{otherwise}
\end{cases}
qquad
\text{maxAbs}(E)=\max(|e_L|,|e_U|).
]

The materiality interval is normalized by the query's pre-registered positive threshold (m_q):

[
[\text{minAbs}(E)/m_q, \text{maxAbs}(E)/m_q].
]

- `materiality_exceeded_possible`: upper normalized bound is at least 1.
- `materiality_exceeded_guaranteed`: lower normalized bound is at least 1.

If a decision threshold (t_q) was pre-registered, the component reports whether a threshold-decision change is impossible, possible or guaranteed.

## Important dependence caveat

The interval difference above is a conservative *rectangular outer bound*. It treats any value in the baseline interval as potentially pairable with any value in the after interval. If the same unknown structural parameter, detector state or graph hypothesis jointly constrains both snapshots, this can be wider than the sharp paired effect set.

Do not call this a sharp causal or biological effect bound. A later paired-scenario adapter should carry shared latent state / paired graph hypotheses and calculate the joint feasible effect set. The current primitive is deliberately fail-safe against overstating certainty, at the cost of possible overstatement of effect magnitude.

## Paired shared-world effect bounds

An adapter may optionally supply a `PairedEffectBounds` interval computed from matched baseline and post-edit graph worlds under the same latent structural hypothesis, plus a digest of the paired-world evidence artifact. The paired interval must be contained inside the rectangular outer bound. A pair-derived interval that expands beyond that outer bound is rejected.

This permits a tighter effect estimate when shared uncertainty is modeled jointly. The core only checks containment and digest shape; the adapter must verify that the same world/latent state was used on both sides and that the paired effect artifact is authentic. A paired interval is not trusted merely because it is narrower.

### Paired decision-world evidence

A paired effect artifact may additionally include a non-empty list of `PairedDecisionWorld` rows. Each row has a unique world ID and exact baseline/post-edit endpoint values from the same latent graph world. When a decision threshold is pre-registered, the threshold status is compared within each matched world:

- all supplied worlds flip -> `Guaranteed`;
- no supplied worlds flip -> `Impossible`;
- some supplied worlds flip -> `Possible`.

The report labels the method as `PairedSharedWorlds` and records the paired-world count. If paired decision worlds are absent, the original marginal-interval classification remains and is labelled `MarginalIntervalOuterBound`.

Each paired endpoint must lie inside its corresponding marginal interval, and its before/after difference must lie inside the paired effect interval (with a small numerical tolerance). Duplicate world IDs, empty paired-world lists, or paired decision data without a pre-registered threshold fail closed.

A paired-world result applies only to the supplied worlds. It is not a probability, and guaranteed / impossible mean only within that supplied set. The adapter must prove that the world manifest is complete for the declared admissible structural scenario and that each before/after pair shares the same latent graph world.

## Matched decision-world evidence

If a paired effect artifact supplies `decision_worlds`, each row gives a unique world ID and exact before/after query endpoint values from the same latent structural graph world. For a pre-registered decision threshold, the component compares decisions within each matched pair:

- every paired world crosses the threshold -> `Guaranteed`;
- no paired world crosses -> `Impossible`;
- some cross and some do not -> `Possible`.

The report identifies `PairedSharedWorlds` as the decision-flip method and records the number of worlds. Without paired decision worlds, the existing conservative marginal classification is used and labelled `MarginalIntervalOuterBound`. If no threshold is configured, the result is `NotConfigured` and paired decision worlds are rejected.

Validation requires a non-empty world set, unique world IDs, baseline/post-edit values inside the declared marginal intervals, and each paired difference inside the paired effect interval (with a 1e-12 floating-point tolerance). The evidence digest must be valid. Guarantees and impossibility are **only over the supplied paired worlds**; they do not prove the world set is exhaustive. An independently pinned adapter must authenticate the paired-world manifest and its shared-latent-state mapping.

## Scenario handling

- Each query/scenario row is preserved in the output.
- Scenarios have no weights and are never averaged.
- Scenario rows are sorted by ID for deterministic receipts.
- A query-level "guaranteed flip" is returned only when every supplied scenario guarantees a flip.
- A query-level "impossible flip" is returned only when every supplied scenario makes a flip impossible.
- Mixed outcomes yield "possible"; this does not imply that the flip is likely.
- Missing decision thresholds are reported as NotConfigured.

The crate does not establish that the supplied scenario set is exhaustive. It reports only the supplied admissible scenarios.

## API

- `Interval::new(lower, upper)`: validate a finite ordered interval.
- `Interval::difference(baseline)`: conservative interval-valued after-minus-before effect.
- `analyze_candidate(&CandidateCorrectionInput)`: validate a candidate and return a canonical per-query impact vector.

Every query must be marked pre-registered. The same query must carry the same definition digest, metric, unit, materiality threshold and decision threshold across all its scenarios. The same query/scenario pair cannot be duplicated. A materiality threshold must be finite and strictly positive.

## Synthetic fixture

- Input schema: `research/connectomics/connectome-query-impact-input-v0.1.schema.json`.
- Report schema: `research/connectomics/connectome-query-impact-report-v0.1.schema.json`.
- Fixture schema: `research/connectomics/connectome-query-impact-fixture-v0.1.schema.json`.
- Known-answer cases: `research/connectomics/fixtures/connectome-query-impact-known-answer-v0.1.json`.

The 14-case fixture checks paired-effect interval containment plus matched decision worlds: an impossible flip despite marginal intervals straddling the threshold, guaranteed and possible flips, duplicate world IDs, values outside marginal intervals, differences outside paired effect bounds, missing decision thresholds and empty world sets.

The fixture covers guaranteed threshold flips, scenario disagreement, independent units across different queries and conflicting query-unit declarations.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-query-impact/Cargo.toml
```

Twenty-seven Rust unit tests are authored for interval difference, threshold-flip classifications, materiality bounds, uncertainty/scenario preservation, differing query units, pre-registration, duplicate scenarios, inconsistent query signatures, malformed intervals/hashes and deterministic ordering, paired-world interval containment, paired evidence digests, matched threshold-decision flips, world uniqueness, marginal/effect containment, and paired decision abstention rules.

They have **not** been compiled or executed in the current authoring environment.

## Research integration

- #9984 can consume impact vectors only after an independent, frozen review-cost policy maps query consequences to an explicitly declared loss unit. This crate itself does not rank review work.
- #9987 can acquire additional evidence when the outcome interval spans a decision boundary or materiality threshold.
- #9994 can compare the same query endpoint across observed graph history.
- #10018 ensures a committed correction is independently verified and reversible.
- #10028 binds these receipts into a cross-stage workflow envelope.

## Benchmarks

Synthetic test track:
- inject known false merges and false splits into a fixed graph;
- compute exact query endpoint values before and after each injected correction;
- compare rectangular outer bounds against the known exact paired effect;
- measure interval coverage, width inflation and threshold-flip classification;
- include cases with shared latent structure where rectangular bounds are deliberately conservative.

Prospective track:
- freeze query definitions and materiality thresholds before review;
- compare random, entropy-only, graph-degree, query-impact and hybrid policies at fixed reviewer-minute budgets;
- measure recall of query-material errors, false-priority cost, threshold-flip recall, downstream endpoint error and false query-ready certifications;
- cluster-bootstrap by source volume / neuron / animal as appropriate.

Proposed targets (not achieved):
- at least +15 absolute points of query-material error recall at equal reviewer time versus entropy-only prioritization;
- at least 25% fewer reviewer minutes at matched query-material recall;
- nominal coverage for predicted query-impact intervals on held-out volumes;
- zero false query-ready certifications in synthetic negative controls.

## Non-claims

This component does not decide whether a proposed edit is anatomically correct, assign biological probabilities, estimate neural causality or certify a scientific conclusion. It measures sensitivity of declared graph-derived endpoints to supplied before/after intervals. A graph consequence is not proof that the edit should be made.

See [SCI-CONNECTOMICS-060](https://github.com/Luminous-Dynamics/symthaea/issues/10054).