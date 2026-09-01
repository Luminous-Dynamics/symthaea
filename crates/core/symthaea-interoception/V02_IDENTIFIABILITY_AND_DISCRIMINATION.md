# Affective Emergence v0.2 — Candidate Identifiability and Discrimination Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents v0.2 from comparing many candidate formulas that the locked scenario program cannot actually distinguish.

A finite candidate set is not scientifically useful merely because every candidate has a different digest. Two different formulas may be observationally equivalent over the chosen scenarios, or may differ only in regions the experiment never probes.

## 1. Principle

Before interpreting candidate performance, prove that the experiment has enough **discriminating structure** to separate the candidate from the baselines it is meant to outperform.

A candidate that cannot be distinguished from a simpler baseline under the locked scenario/cut-point set must not receive a stronger interpretation merely because its formula is more elaborate.

Use explicit states such as:

- `StructurallyDistinguishable`;
- `ObservationallyEquivalentWithinDesign`;
- `InsufficientDiscrimination`;
- `NumericallyIndeterminate`.

`InsufficientDiscrimination` is a design result, not candidate support.

## 2. Prospective CandidateDiscriminationManifest

Before exploratory execution, create a canonical manifest binding:

- schema/version;
- exact finite exploratory candidate-set digest;
- exact scenario/cohort digest;
- exact prospective cut points/windows;
- ordered candidate-definition digests;
- ordered baseline-definition digests;
- pairwise candidate/baseline comparisons that must be identifiable;
- discriminator scenario/cut-point references for each required pair;
- expected **structural relation class** where derivable from the candidate definitions;
- numerical/equivalence tolerance contract;
- minimum discrimination coverage requirement;
- canonical SHA-256.

This manifest does not predict which candidate will tell the more interesting scientific story. It predicts only whether the chosen experimental manipulations should make the formulas observationally separable.

## 3. Structural relation classes

A prospective discriminator may declare relations such as:

- `MustEqual` — algebra/contract says outputs should be equal;
- `MustDiffer` — definitions guarantee a difference under the specified fixture when valid;
- `MayDiffer` — the scenario is informative but exact direction depends on state trajectory;
- `GreaterThan` / `LessThan` — direction follows structurally from the locked candidate definitions and scenario;
- `OneUnavailable` — one candidate is intentionally undefined while another is available;
- `DiagnosticOnly` — used to detect implementation or information leakage, not model preference.

Do not preregister empirical outcome claims as structural necessities unless they truly follow from the locked mathematics.

## 4. Pairwise identifiability gate

For every confirmatory primary candidate `C` and every nuisance/simple baseline `B` whose insufficiency is part of the claim, require at least one prospectively locked discriminator where `C` and `B` are expected to be separable under their definitions.

Prefer multiple independent discriminator families when the claim is broad.

If no such discriminator exists, the comparison is not identifiable and confirmatory claims of `C beyond B` are invalid regardless of observed numerical differences elsewhere.

## 5. Candidate equivalence classes

After blinded candidate extraction, construct a deterministic **CandidateFingerprint** over the locked scenario/cut-point set.

A fingerprint should bind at minimum:

- candidate digest;
- ordered scenario/cut-point identities;
- availability state at each coordinate;
- canonical candidate payload/value at each coordinate;
- declared scientific-equivalence tolerance where exact equality is not required;
- fingerprint digest.

Pairwise comparison may cluster candidate definitions into observational equivalence classes under the locked design.

If two candidates remain equivalent across every discriminator:

- preserve both candidate definitions/results;
- report them as observationally equivalent within the tested design;
- do not choose the more complex or affect-intuitive formula as the winner;
- prefer the simpler baseline for explanatory claims unless a separately preregistered criterion justifies otherwise;
- design a new discriminator scenario in a future exploratory lineage if separation matters.

## 6. Mechanism-isolating discriminators

The scenario program should include cases that change one candidate factor while holding others fixed as far as the deterministic model permits.

Required families should cover:

### D1 — weighting discrimination

Hold native state/trajectory and temporal aggregation fixed while changing precision, importance, or denominator inputs according to `V02_WEIGHTING_DECOMPOSITION.md` and the cross-channel aggregation contract.

### D2 — temporal aggregation discrimination

Use trajectories where mean burden and cumulative exposure rank scenarios differently, or where peak/terminal/path exposure disagree.

### D3 — relation discrimination

Use crossed R1/R2/R3/R4 cases so actual change, surprise, outlook revision, and rolling-window change cannot collapse into one signature.

### D4 — forecast-policy discrimination

Construct regimes where zero-input recovery, observed-drive persistence, and kinematic forecasts agree, plus regimes where their locked assumptions force different trajectories.

### D5 — channel-projection discrimination

Use cases where peak-channel, full-vector, weighted-mean, weighted-sum, and breach-breadth projections produce different conclusions.

### D6 — current-state / stimulus nuisance discrimination

Match current burden or drive magnitude while changing already-observed trajectory/internal margin so a richer candidate must show value beyond a trivial current-state or stimulus-only baseline.

### D7 — information-authority diagnostic

Identical-prefix/divergent-future twins must leave every prefix-causal payload identical. This diagnoses information leakage; it must never be interpreted as candidate support.

## 7. Design-matrix coverage

Treat the finite exploratory candidate set and finite scenario set as a deterministic experimental design matrix.

Before execution, compute a **qualitative discrimination matrix** whose rows are locked scenario/cut-point manipulations and whose columns are candidate coordinates or candidate pairs.

The design review should ask:

- does every factor axis have at least one isolating manipulation?
- is every primary-vs-baseline comparison identifiable?
- are any candidates algebraically redundant under all locked scenarios?
- do multiple scenarios test the same distinction while leaving another factor untested?
- is a purportedly rich candidate only distinguishable because of a nuisance difference?

A design with poor discrimination coverage remains `Reviewable` but cannot be frozen for confirmatory use.

## 8. Parsimony rule

When two candidates are observationally equivalent under the locked design, v0.2 should not use interpretive preference as a tiebreaker.

A prospective model-selection rule may prefer, in order:

1. simpler information requirements;
2. fewer free configuration choices;
3. fewer temporal/forecast dependencies;
4. greater prefix availability;
5. stronger numerical stability;
6. easier independent reproduction.

Any alternative complexity penalty must be prospectively defined.

This is not an assertion that simple models are metaphysically true. It is a safeguard against giving extra scientific meaning to complexity that the experiment cannot identify.

## 9. Negative controls and pseudo-candidates

The exploratory suite should include deliberately weak controls, for example:

- current burden only;
- current drive magnitude only;
- elapsed-step index only;
- static constant predictor;
- deterministic hash/noise-like pseudo-candidate that carries no regulatory meaning;
- temporally shifted or shuffled diagnostic where permitted without violating the deterministic information contract.

These controls test whether the scoring/ranking procedure itself manufactures apparent superiority.

Pseudo-candidates must be explicitly labeled as controls and cannot become primary scientific candidates through exploratory ranking.

## 10. Candidate-set pruning before confirmation

Exploratory work may reveal that several candidate definitions are observationally redundant.

Before confirmatory lock:

- preserve the exploratory results for all candidates;
- define equivalence classes;
- choose at most the preregistered number of representatives under the prospective selection/parsimony rule;
- freeze a new finite confirmatory candidate/baseline set;
- do not choose representatives based on semantic/emotional intuitiveness.

If no candidate is distinguishable from the simple baselines, `NoUniqueWinner` / `EquivalentToBaseline` is the correct outcome.

## 11. Implementation gates

The eventual observatory should mechanically test:

1. every required primary-vs-baseline pair has at least one registered discriminator;
2. every discriminator references existing locked scenario/cut-point/candidate identities;
3. no post-execution scenario can be added to repair an identifiability failure inside the same confirmatory lineage;
4. CandidateFingerprint ordering/canonicalization is deterministic;
5. exact-equivalent candidates are detected exactly;
6. tolerance-equivalent candidates use only prospectively declared tolerances;
7. known algebraically equivalent fixture candidates are clustered correctly;
8. known distinguishable fixture candidates are separated by their declared discriminators;
9. candidate ranking cannot promote a pseudo-control to a scientific primary candidate;
10. the realized evidence package preserves the full pairwise/equivalence report.

## 12. Evidence-root consequence

The prospective root should bind:

- CandidateDiscriminationManifest digest;
- discrimination-matrix contract/version;
- finite exploratory/confirmatory candidate-set digest as appropriate;
- pairwise baseline obligations;
- parsimony/model-selection rule identity.

The realized package should bind:

- candidate fingerprint digests;
- pairwise discrimination results;
- equivalence-class report;
- any `InsufficientDiscrimination` findings.

A primary result cannot be `QualifiedSupportedResult` for a claim of superiority over baseline `B` if the locked design never made `C` vs `B` identifiable.

## 13. Claim boundary

Identifiability testing can show that the experiment is capable of distinguishing competing regulatory explanations and can prevent redundant formulas from being overinterpreted.

It does not establish that a distinguishable candidate is affect, emotion, subjective valence, mood, sentience, or consciousness.