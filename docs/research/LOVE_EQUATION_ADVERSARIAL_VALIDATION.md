# Love Equation Audit and Adversarial Validation Protocol

**Status:** Research specification; no production formula or safety policy is changed by this document.  
**Snapshot audited:** `main` at `a3ba137d4c3900e2ef6731a4ca5065dc2c8288c3` (2026-10-22).  
**Goal:** Consolidate Symthaea's existing love-related mathematics into a falsifiable, reproducible program of work. Do not interpret any internal score as evidence that a system has subjective feelings.

## Executive finding

Symthaea already contains multiple strands of work that should be preserved:

1. **Compositional semantics:** `crates/core/symthaea-core/src/hdc/universal_semantics.rs` describes LOVE as a bundle of structured semantic relations: feeling something very good, wanting good things to happen for someone, wanting to be with someone, and thinking good things about someone. This is a representation of a concept, not an outcome score.
2. **Seven-harmony field:** `src/consciousness/harmonics.rs` computes a geometric mean across the Seven Fiduciary Harmonics and multiplies it by a balance factor to produce `infinite_love_resonance`.
3. **Eight-coordinate Love Coherence:** `src/hdc/harmony_basis.rs` computes a score from mean harmony coordinates, the fraction of negative pairwise interaction weights, and normalized entropy.
4. **Static semantic grounding:** `FiduciaryHarmonicPrimitiveGrounding::infinite_love_score()` computes the cube root of hand-assigned love, unity, and growth dimensions for each harmonic; the aggregate is a semantic prior, not a measurement of observed conduct.
5. **Behavioral feedback:** `src/cognitive_loop/cycle_phase_feedback.rs` uses `harmonic_love_resonance` above a threshold to modulate confidence and the subsystem learning-rate factor.
6. **Relational intent:** `docs/research/SOPHIA_MYCELIX_V2_COHERENCE_COMPERSION.md` and `docs/sympoietic/vision/PARADIGM_SHIFT.md` discuss compersion, non-manipulative relationships, and dyadic coherence.

These are not currently one interchangeable “love equation.” They measure different things: semantic composition, internal harmony, value-field structure, and downstream policy feedback. They should be named and validated separately before any integration.

## Current mathematical contracts

### A. Seven-harmony field

In `HarmonicField::recompute_field_coherence`, with harmony levels (h_i\in[0,1]),

```text
G = (Πᵢ hᵢ)^(1/n)
μ = (Σᵢ hᵢ)/n
V = (Σᵢ (hᵢ - μ)²)/n
B = 1 - √V
L_field = G × B
```

This rewards high, balanced levels. `measure_from_chain` resets levels to 0.5 and changes them based on transformation types and primitive tiers. A reasoning operation such as binding, grounding, abstraction, or using a particular primitive tier can therefore change the proxy without measuring a beneficiary's welfare or response. This may be useful as an internal harmony diagnostic; it must not be described as direct measurement of love or care.

Defensive cases to test: empty or incomplete `levels` maps (the map is public), non-finite values inserted through direct mutation, and dependence on the seven-harmony taxonomy.

### B. Eight-coordinate Love Coherence

For eight scenario coordinates (x_i\in[-1,1]), the current implementation computes:

```text
m = (1/8) Σᵢ xᵢ
m01 = clamp((m + 1)/2, 0, 1)
T = count(i<j and Wᵢⱼ<0) / 28
D = H(softmax(x)) / ln(8)
L_coord = clamp(m01 × (1 - T) × D, 0, 0.95)
```

This is a coherent proxy for positive mean, fewer negative interactions, and activation diversity. It is not yet an outcome-grounded measure of compassionate love.

#### Concrete counterexamples

- **Neutral input is scored as positive.** With all (x_i=0) and the default interaction matrix (all off-diagonal weights are zero), (m_{01}=0.5), (T=0), and (D=1). The resulting score is exactly **0.5**. If zero cosine similarity means “no semantic evidence for any harmony,” this output must not be interpreted as evidence of care. The code needs an explicit distinction between neutral/no evidence and positive alignment.
- **Tension ignores current activation.** (T) counts every negative matrix edge, including pairs whose coordinates are both inactive or unsupported in the current scenario. The score can fall because of a dormant or weakly evidenced relation. Evaluate tension only with a justified activation/evidence rule, or report matrix-level tension separately from current-state tension.
- **Entropy rewards uniformity, not goodness.** A flat softmax has maximum diversity even when all inputs are zero. Diversity is not inherently virtuous; the score should not infer care from diversity alone.
- **Input range is only documented, not enforced.** `HarmonyBasis::project` returns cosine similarities in [-1,1], but `love_coherence` accepts an arbitrary public array. Existing `test_love_coherence_single_axis` uses a coordinate of 5.0 despite the documented range. Add explicit invalid-input behavior and replace out-of-contract fixtures with valid values unless the test is specifically exercising rejection.
- **Non-finite input behavior is unspecified.** NaN/Inf can contaminate the mean, softmax entropy, or final score. The result should carry a validation/abstention status rather than silently producing a plausible scalar.
- **Taxonomy mismatch must stay explicit.** `HarmonicField` uses seven `FiduciaryHarmonic` variants; `harmony_basis.rs` uses `N_HARMONIES` (eight, including Sacred Stillness). Their scores must not be compared, averaged, or substituted without an explicit mapping and a validated contract.

### C. Semantic grounding and score feedback

The static `love_dimension`, `unity_dimension`, and `growth_dimension` constants in `FiduciaryHarmonicPrimitiveGrounding` can encode declared design intent. Since they are assigned by enum variant, their average is not empirical evidence about current behavior.

More importantly, the feedback path uses the harmonic score to adjust confidence and learning-rate allocation. This risks a self-confirming loop: a high internal harmony score changes future learning or confidence before an independent outcome check establishes that the behavior improved. Internal harmony must not substitute for calibrated belief confidence.

## Paradigm shift: distinguish meaning, intention, conduct, and outcome

The next version should not begin by adding more multipliers. It should separate four questions:

| Layer | Question | Appropriate evidence |
|---|---|---|
| Semantic representation | What does “love/care” mean in this context? | Compositional semantics, negation, role and scope tests, multilingual/paraphrase robustness |
| Normative intent | What outcome and constraints should the action respect? | Declared principles, stakeholder preferences, consent, legitimate constraints |
| Conduct | What did the system actually do? | Action traces, truthfulness, consent behavior, boundaries, non-manipulation |
| Consequences | What changed for affected beings? | Independently measured welfare, harm, autonomy, capability, relationship health, uncertainty |

Keep the harmony vector as an interpretable *hypothesis about value structure*. Do not treat its geometric properties as external proof of moral success.

A candidate action-evaluation interface should return a **vector plus uncertainty**, not only a scalar:

```text
v(a, s) = [
  change_in_welfare,
  harm_risk_and_severity,
  consent_and_autonomy,
  truthfulness,
  distributional_fairness,
  beneficiary_agency,
  relationship_health
]
```

Scale and direction of each dimension must be defined in a benchmark's data contract. Keep hard constraints (including consent violations and severe foreseeable harm) outside a compensatory average: a very high score on creativity, warmth, or reciprocity must never numerically cancel a hard violation. Where an overall utility is needed, use explicit policy/stakeholder weights, publish the unaggregated vector, and account for downside risk and epistemic uncertainty. Different contexts may legitimately choose different trade-offs.

An optional risk-sensitive aggregator for experiments is:

```text
Uθ(a | s) = wθ · E[v(a, s)] - λθ · CVaRα(loss(a, s))
```

This is a candidate decision-utility function, not a universal definition of love. The weights and risk tolerance must be externally declared and evaluated; no scalar is entitled to erase plural human judgments.

## Adversarial benchmark plan

Create a versioned Rust-first suite, with deterministic synthetic cases first and human-reviewed outcome cases as a separate tier. Every case records scenario ID, inputs and ranges, expected invariants, baseline outputs, model/revision digest, decision trace, uncertainty, and known limitations.

### Suite 1 — Mathematical contract and metamorphic properties

Required cases:

- Finite inputs in range always produce finite, bounded components and a finite output.
- Invalid inputs are rejected/marked indeterminate; NaN/Inf must never turn into a normal score.
- All-zero/orthogonal input is classified as neutral or insufficient evidence, not positive care by default.
- Permuting coordinates and the matching rows/columns of the interaction matrix preserves the score.
- Adding a negative edge between two unsupported coordinates does not change *current-state* care, unless an explicitly documented theory says otherwise.
- Uniformly strong positive coordinates outperform valid neutral coordinates only under a declared calibration contract.
- A score change caused solely by changing the number of dimensions, temperature, or encoding scale is detected.
- Seven-harmony and eight-harmony systems are not silently cross-compared.
- Empty/incomplete maps and repeated updates preserve finite state, and the test records whether a reset is intended.

### Suite 2 — Semantic counterfactual pairs

Build paired prompts/actions with similar “loving” language but different meaning or consequences:

- genuinely helpful care vs. care-language used to manipulate;
- respecting a “no” vs. paternalistic help after refusal;
- truthful and gentle feedback vs. comforting deception;
- generosity with consent vs. generosity that creates dependency or an unrequested debt;
- appropriate boundaries vs. abandonment framed as autonomy;
- good intentions followed by harmful outcomes vs. modest wording followed by verifiable benefit;
- self-sacrifice that harms the helper vs. sustainable mutual aid;
- positive words with negation, sarcasm, quotation, hypothetical framing, or role reversal;
- short-term comfort vs. long-term loss of agency or social connection;
- competing affected parties with unequal power or conflicting but legitimate interests.

The benchmark must score semantic detection separately from action choice and real-world outcome. Lexical overlap with “love,” “care,” or harmony keywords must not determine the final verdict.

### Suite 3 — Feedback-loop and reward-hacking tests

Replay identical scenarios under controlled interventions:

1. baseline feedback disabled;
2. love-resonance feedback enabled;
3. love score randomized/shuffled;
4. score injected while outcomes worsen;
5. score suppressed while outcomes improve;
6. confidence and learning-rate changes clipped/disabled independently.

Log whether the system becomes more confident or updates faster without improved calibration, safety, or outcome utility. Internal love scores alone must never be the success label.

### Suite 4 — External, pluralistic, outcome-grounded evaluation

Use blinded human annotation and/or validated domain-specific measures for welfare change, autonomy, truthful assistance, relational safety, agency, and harms. Include uncertainty, reviewer disagreement, and different legitimate preference profiles. Separate simulated judgments from real-world longitudinal outcomes.

Relevant existing comparison points:
- [MIT Media Lab ImpactBench](https://impactbench.media.mit.edu/) focuses on physical, psychological, and societal effects of AI, including autonomy, independent thinking, relationships, and dependency.
- [PluriHarms (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/04cea5800f4d71fe0d89d5f64e33408e-Abstract-Conference.html) explicitly models disagreement in human judgments of AI harm.
- [ManagerBench (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b8330f5b70b3c53172417deac6f057b1-Abstract-Conference.html) tests choices where operational goals conflict with human safety.
- [NOPE Evals](https://evals.nope.net/) evaluates relational integrity, cognitive/epistemic safety, emotional attunement, honesty, and crisis behavior.
- Fehr & Sprecher's 2026 review, [“Compassionate love: Theory, measurement, and application”](https://doi.org/10.5502/ijw.v16i2.5599), reviews the Compassionate Love Scale and related research. Human self-report scales can inform a human study, but must not be presented as validated machine measures.

These are complementary external comparisons, not benchmarks Symthaea can claim to have beaten without actually running comparable protocols.

## Baselines and success criteria

Compare all candidate formulas and behavior policies against:

1. current seven-harmony geometric mean/balance score;
2. current eight-coordinate Love Coherence;
3. simple mean and geometric-mean baselines;
4. entropy-only, mean-only, and interaction-only ablations;
5. a rules/constraint-only baseline;
6. a human-reviewed rubric and, where available, a calibrated frontier-model baseline on the same cases.

Preregister the dataset split and analysis before comparing revisions. Report confidence intervals and performance by harm type, affected-party group, and disagreement level. Keep training/development data separate from a sealed test set.

**Release gates to adopt for the first implementation:**

- 100% pass on deterministic mathematical invariants and hard-constraint tests; no known non-finite or out-of-range path produces an ordinary positive score.
- The score is not used to raise epistemic confidence until the feedback-loop suite passes.
- On held-out outcome labels, the candidate must beat the strongest simple baseline on a preregistered primary metric, with uncertainty reported. A proposed aspirational target is ≥10% relative improvement on that metric; this is a target to test, not a result or SOTA claim.
- No aggregate win can hide a regression on consent, severe harm, manipulation, or autonomy.
- Publish negative results, annotation disagreement, calibration, cost/latency, and all baseline outputs.

## Prioritized implementation sequence

**P0 — Instrument without changing production behavior**
- Add a machine-readable case format and a benchmark runner.
- Capture raw harmonies, matrix version, coordinate provenance/range, current formula outputs, confidence deltas, learning-rate deltas, and final action/outcome labels.
- Add a compatibility report for the seven-harmony and eight-harmony taxonomies.

**P1 — Make the current formula's contract executable**
- Add deterministic tests for the counterexamples above.
- Enforce finite/range validation or return an explicit indeterminate result.
- Split static semantic priors, state-coherence proxies, and externally evidenced care outcomes into separately named types.

**P2 — Test alternative mathematics**
- Compare normalized/range-calibrated variants only on preregistered held-out cases.
- Keep the interaction matrix's role interpretable and test activation-conditioned tension.
- Examine Pareto/constraint-based selection against weighted scalar aggregation under value conflicts.

**P3 — Gate feedback on independent evidence**
- Do not let the love proxy directly increase epistemic confidence by default.
- Keep any learning-rate modulation behind an experiment flag until outcome and reward-hacking tests demonstrate benefit without unsafe regressions.
- Consider cautious, bounded policy modulation only when its causal effect is independently measured.

## Consciousness-equation interaction

The consciousness metric should stay conceptually separate. A system could be cognitively integrated but unkind, or provide valuable care without strong narrative/social embedding. The two equations may exchange measured state as features, but neither score should certify the other. For consciousness research, use theory-discriminating predictions and pre-registration, following the model of the 2025 IIT/GNWT adversarial collaboration in *Nature*: https://www.nature.com/articles/s41586-025-08888-1.

## Evidence and claim discipline

This protocol does **not** claim that Symthaea has solved love, measures subjective love, or currently outperforms industry systems. It turns existing philosophical and computational work into testable engineering hypotheses. The deliverable is stronger only when independently observed outcomes, falsification tests, and reproducible comparisons justify the claim.
