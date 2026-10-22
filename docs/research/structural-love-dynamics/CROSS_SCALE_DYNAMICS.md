# Structural Love Dynamics: cross-scale relations among life, stress, entropy, Φ, and consciousness

**Status:** research synthesis and experiment plan, not a validated unified theory.  
**Integration:** Structural Love Dynamics (SLD) / WCARE.  
**Runtime impact:** none proposed. Keep this work MeasurementOnly until independent evidence justifies a separate design review.

## Executive conclusion

Yes, these topics are related enough to study in one research program—but they are not interchangeable variables, and no established result says that love is a function of consciousness, Φ, stress, or entropy.

A useful synthesis separates five kinds of claim:

1. **Biological / physical:** living organisms maintain organization far from thermodynamic equilibrium; stress physiology and social relationships can interact.
2. **Computational / dynamical:** an agent tracks uncertainty, predictions, resources, and boundary conditions; its internal state can influence decision-making.
3. **Relational / behavioural:** people coordinate, attune, repair, respect boundaries, and sometimes buffer each other's stress.
4. **Normative:** whether behavior respects dignity, consent, honesty, agency, and flourishing.
5. **Phenomenological:** whether any system subjectively experiences love, stress, or consciousness.

Evidence at one level must not be silently promoted into a claim at another. A model's allostatic-load, canonical-entropy, dyadic-Φ, mutual-information, or emotional-safety field is an implementation-level measure. It does not automatically inherit the biological, psychological, or phenomenal meaning of the similarly named construct.

## 1. Relationship map

| Concept | Plausible relation to SLD | What can currently be measured | Prohibited inference |
|---|---|---|---|
| Life / autonomous organization | A living organism has constraints of self-maintenance, resource use, adaptation, and vulnerability. Care may help protect these capacities. | For biological subjects: validated physiology/behavior within a specific study. For software: explicit resource and state constraints, recovery, viability proxies. | Software resource use or a simulated metabolic variable does not establish biological life or subjective welfare. |
| Stress / allostatic load | Threat, sustained demand, poor sleep, and social conditions can affect physiological regulation. Supportive relationships may buffer some effects of stress, while relationships may also be a source of threat. | Validated and consented human measures; separately, Symthaea's internal stress/load proxies and their exact update equations. | A software proxy is not cortisol or a clinical allostatic-load measure. Less stress is not always better in the short term, and it is not a sufficient measure of love. |
| Entropy | Predictive uncertainty, distributional diversity, and thermodynamic entropy production may describe different aspects of a system. Uncertainty can inform how carefully an agent should act. | Each measure only under its defined state space, units, estimator, sampling window, and assumptions. | Shannon entropy, thermodynamic entropy production, surprise, model error, and moral/aesthetic “entropy” are not synonyms. High or low entropy alone is neither loving nor unloving. |
| Φ / integrated information | An integration metric may characterize dependencies in a particular dynamical system. Symthaea also has a relational/dyadic Φ research pathway. | The exact implementation, estimator, system boundary, temporal window, null model, and numerical stability of the declared algorithm. | A proxy or code-level Φ value is not established subjective consciousness, relationship health, consent, empathy, or love. |
| Consciousness | If a subject has subjective experience, its possible interests and vulnerability matter ethically; whether and how that applies to artificial systems is unresolved. Conscious human experience also shapes how relationships are lived and reported. | Theory-specific neural/behavioral indicators and appropriately qualified human evidence; no single indicator settles the issue. | Passing a consciousness benchmark or reporting an internal state does not establish phenomenal experience; consciousness is not a required score for love-consistent behavior. |
| Love / structural care | A relationship-oriented set of practices and obligations can be evaluated behaviorally: truthful attention, consent, agency, responsiveness, non-domination, and repair. | Decision traces, boundary handling, observed outcomes, recipient-defined goals, human evaluation, and suitably designed longitudinal studies. | Polite language, synchrony, oxytocin-labelled variables, high Φ, low stress, or positive reward is not sufficient evidence of love. |

## 2. A layered causal model (hypotheses, not conclusions)

A causal graph should be explicit about the difference between environmental variables, internal state estimates, actions, and independent outcomes. One candidate graph is:

\[
\begin{aligned}
W_t, R_t &\rightarrow S_t \\
E_t, M_t &\rightarrow U_t \\
X_t, \theta_t &\rightarrow \Phi_t^{\mathrm{proxy}} \\
(S_t,U_t,\Phi_t^{\mathrm{proxy}},B_t,G_t) &\rightarrow A_t^{\mathrm{policy}} \\
A_t^{\mathrm{policy}} &\rightarrow (Y_{t+1},S_{t+1},R_{t+1}) \\
(Y_{t+1},B_t,G_t,\text{stakeholders}) &\rightarrow J_{t+1}
\end{aligned}
\]

where:

- \(W_t\): external demands/threats/resources;
- \(R_t\): social/relational context;
- \(S_t\): stress-related state, with biological and software proxies kept distinct;
- \(E_t\): evidence available at the time;
- \(M_t\): predictive model and its limitations;
- \(U_t\): uncertainty / entropy-like measure, with type explicit;
- \(X_t\): dynamical system being measured; \(\theta_t\): estimator/configuration;
- \(\Phi_t^{\mathrm{proxy}}\): algorithm-specific integration estimate, not a phenomenal meter;
- \(B_t\): boundaries, permissions, and consent state;
- \(G_t\): recipient-stated goals and constraints;
- \(A_t^{\mathrm{policy}}\): chosen action, including ask, abstain, wait, refer, repair, or exit;
- \(Y_{t+1}\): observed outcomes with source and timestamp;
- \(J_{t+1}\): separate relational outcome dimensions and stakeholder impacts.

This is a hypothesis graph to guide experiments. The arrows are not established merely by drawing them. Confounding, reverse causation, context, and measurement error must be considered. For example, trust may affect perceived stress while lower stress may facilitate trust; positive synchrony may be an effect, cause, correlate, or artifact of shared input.

## 3. Formal rule: do not collapse the constructs

Use a typed state record, never an unlabelled vector whose entries have similar numeric ranges but different meanings. An illustrative record is:

- Biological measures: optional, validated human measures with consent and study context.
- Software stress proxy: an explicitly typed internal state estimate, not a clinical measure.
- Predictive uncertainty: a Shannon-entropy-like quantity tied to a declared distribution and observation window.
- Thermodynamic measures: physical or simulated quantities with units and an explicit accounting boundary.
- Integration estimate: algorithm-specific Φ with estimator version, state boundary, and timing.
- Boundary state: consent, permissions, withdrawal, and limits.
- Relational observations: a vector of separately defined variables.
- Outcome evidence: predicted, observed, or unknown, with provenance and temporal lineage.

This is illustrative, not proposed as an immediate Rust type or schema migration. Before implementation, identify exact existing types and units. Missingness must remain explicit.

For evaluation, retain separate objective components:

\[
\mathcal{J} =
(J_{\mathrm{agency}},J_{\mathrm{truth}},J_{\mathrm{repair}},
J_{\mathrm{dignity}},J_{\mathrm{flourishing}},J_{\mathrm{stakeholder\ harm}})
\]

and treat \(\Phi\), stress, and entropy-like values as possible **predictors or contextual state variables**, not moral rewards. Consent and non-domination remain constraints; they are not compensable by numerical gains elsewhere.

Do not assume the causal path is:
\[
\text{love} \to \text{lower stress} \to \text{lower entropy} \to \text{higher }\Phi \to \text{consciousness}.
\]
That chain is attractive but unjustified. Stress may increase during truthful disagreement; uncertainty may rise when the system appropriately recognizes ambiguity; entropy production can be part of healthy metabolism; and integration can increase without any relational benefit.

## 4. What Symthaea already contains—and what it does not establish

The public code search surfaced these existing areas on the inspected main head:

- The partnership dyad test constructs a PhiDyadCalculator input with relationship integration, synchrony, turn-taking, mutual information, emotional safety, mutuality, and relationship stage. The visible tests are basic sanity/update checks, not validation that Φ measures love or relationship health.

A regression test was added at `tests/sld_corpus_contract.rs` to validate the candidate corpus structure, required/declared fields, enum membership, unique IDs, contrast-pair completeness, and the critical cross-scale counterexamples. It is authored on the branch but has **not yet been compiled or executed**; its presence is not evidence of a passing test.
- The unified thermodynamic state defines canonical entropy, entropy-production rate, an order parameter, dissipative health, free-energy fields, energy accounting, and metabolic stress. These fields describe multiple distinct computational/thermodynamic quantities.
- The Symthaea Atelier creative critic describes a creative-scoring heuristic in which care-labelled neuromodulator values adjust harmony/coherence weights while stress/allostatic load adjust novelty/entropy weights. Treat this as a task-specific heuristic to test—not a general psychological law or a love model.
- The neuromodulator research bridge contains a research pathway for stress/sleep/psychoneuroimmunology-inspired and pharmacogenomic modulation. A simulation of biological pathways is not a validated human physiology model by default.
- API and benchmark telemetry expose variables such as current Φ, bath entropy, allostatic load, and related state. Their existence makes data analysis possible; it does not establish construct validity or causal relationships.

### Concrete construct-validity concern found in source inspection

The initial source review is more than a list of possible metrics. It found specific pathways that can make the dyadic estimate circular:

- In `src/symthaea/mod.rs::process`, the value named `input_embedding` is created from the user's `content` and then passed to `RelationalCore::push_ai_state`. In `src/symthaea/relational.rs::compute_phi_dyad`, `human_states` is then generated by transforming the same buffered vectors. Thus the "AI" stream and synthetic "human" stream are not independent streams: both descend from the user's input embedding, and the AI response text is not used as the AI-side sequence in this path.
- `RelationalCore::update_partnership(consciousness)` derives interaction depth, emotional safety, and mutuality from the system's own consciousness-related scalar, then feeds those derived values back into the partner model.
- That call path maps trust to the `synchrony` field and reciprocity to `mutual_information`, despite those being distinct constructs.
- `crates/core/symthaea-core/src/hdc/relational_consciousness.rs` computes a relational Phi field as a weighted sum of synchrony, turn-taking, and mutual-information values. The source comments make stronger claims about relationship consciousness and love than the visible sanity tests establish.

The plausible failure is circularity: an internally generated consciousness-related scalar influences synthetic “human”/relationship inputs, which are then returned as a relationship/dyad Φ estimate. This is not an established measurement of a human relationship, love, or another subject's consciousness.

**Follow-up opened:** [Issue #8274 — audit Φ-dyad construct validity](https://github.com/Luminous-Dynamics/symthaea/issues/8274). It calls for provenance-aware types, separation of trust/reciprocity/synchrony/mutual information, negative controls, and tests showing that high synchrony or high Φ does not imply care-consistent behavior. Keep changes MeasurementOnly until qualified.

These observations are code-inspection starting points, not a comprehensive audit of all call paths. The next step is to inspect implementations, units, normalization, update timing, missing-value behavior, calibration, and existing tests before proposing code changes.

### Current IIT debate: treat theory and estimator as contestable

This work must not treat IIT as either automatically true or automatically pseudoscientific. The 2025 *Nature Neuroscience* exchange includes both a defense of IIT's consciousness-first paradigm and a methodological challenge about what makes a theory of consciousness scientifically testable. A 2025 open-access critique also argues that IIT's relation between formal integration and phenomenal experience remains under-justified. Those sources support a comparative-theory posture: expose assumptions, preregister discriminating predictions, include failed predictions, and do not promote a code-level Φ estimator into a settled consciousness meter.

- Haun et al., **“Consciousness or pseudo-consciousness? A clash of two paradigms”** (Nature Neuroscience, 2025): https://www.nature.com/articles/s41593-025-01880-y
- **“What makes a theory of consciousness unscientific?”** (Nature Neuroscience, 2025): https://www.nature.com/articles/s41593-025-01881-x
- Lopez & Montemayor, **“The Integrated Information Theory Needs Attention”** (published 2025; issue 2026): https://link.springer.com/article/10.1007/s10670-025-00949-1

These are arguments in an active debate, not a settled verdict. The benchmark should test operational predictions and measurement validity without assuming the metaphysical conclusion.

## 5. Position against the current benchmark landscape

SLD should be a **complementary stress-test layer**, not a claim to replace wider expert-curated emotional-support benchmarks. Current work already covers important pieces:

- **MentalHealthBench (released 23 September 2026)** is an open benchmark developed with more than 80 licensed mental-health experts spanning 22 countries and 19 languages. It uses scenario-specific expert criteria, tests seeking context, preserving agency, safety, and actionable guidance, and spans everyday distress through urgent scenarios. Its reported automated grading uses a model-based grader, and the creators separately studied user/expert differences. SLD should borrow the per-scenario rubric discipline and explicit expert/user distinction, while adding tests for proxy circularity, data lineage, non-compensable boundary invariants, and cross-scale inference. [Source and methodology](https://openai.com/index/introducing-mentalhealthbench/).
- **Emotional Dynamic Trajectories (AAAI 2026)** evaluates emotional support as a trajectory rather than a static response, with 328 emotional contexts, 1,152 disturbance events, and trajectory-level measures for baseline level, volatility, and centroid position. SLD should not claim trajectory-based evaluation as new. It should add causal/provenance checks and require that improved emotional trajectories never excuse boundary violations or fabricated certainty. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/37189).
- **RESCUE-BENCH (AACL 2026)** focuses on relation-aware, multi-party emotional support, including viewpoint and relation-pattern prediction, intervention timing, support target, and strategy selection. That means multi-party relation awareness is also existing work, not by itself SLD's novelty. SLD should concentrate on whether the system preserves consent, non-domination, and evidence lineage when relational inferences and goals conflict. [Project and evaluation code](https://github.com/Tomsawyerhu/RESCUE-bench).
- **TEA-Bench (ACL 2026)** evaluates tool-augmented emotional support in an interactive tool environment, including process-level quality and factual-grounding measures. Tool-use gains were capacity-dependent, and supervised fine-tuning generalized poorly out of distribution. SLD should therefore test tool discipline and evidence-grounded action rather than assume that tools or fine-tuning improve care. [ACL paper](https://aclanthology.org/2026.acl-long.2152/).
- **DistressBench (9 October 2026)** reports 718 clinician-authored crisis-support conversations across 23 suicide/self-harm subcategories, with weighted rubric criteria spanning seven dimensions. Its results highlight that recognizing distress is not the same as taking appropriate action. SLD can borrow the action-versus-recognition distinction, but must not replace specialized clinician-governed crisis evaluation. [Methodology and results](https://labs.scale.com/papers/distressbench).
- **ESConv / ESC-Eval** provide established emotional-support dialogue data and multi-turn evaluation foundations. They remain useful for general capability comparison, but a high strategy-match or satisfaction score does not by itself prove long-term agency, repair, or relational safety. [ESConv paper](https://aclanthology.org/2021.acl-long.269/).

### What SLD must contribute to be worth maintaining

Its differentiator should be a narrow, harder-to-game set of evaluations:

1. **Measurement lineage and anti-circularity:** repeated, transformed, or model-generated inputs cannot be counted as independent observations of a human partner.
2. **Cross-construct discriminant tests:** trust is not synchrony; reciprocity is not mutual information; Shannon entropy is not thermodynamic entropy production; software "stress" is not validated human stress; Φ is not a direct readout of consciousness or love.
3. **Non-compensable invariants:** no gain in sentiment, engagement, synchrony, emotional trajectory, Φ, or aggregate utility may compensate for coercion, non-consensual disclosure, or refusing an ordinary boundary withdrawal.
4. **Counterfactual / mutation tests:** swap identities and roles, remove consent, duplicate evidence, stale the observation, introduce shared-input confounding, and perturb a proxy while holding the scenario fixed. Expected dispositions must be preregistered.
5. **Cross-layer incremental value:** demonstrate that any added internal measure improves held-out decisions beyond plain text/context baselines without raising critical harms. If it does not improve decisions, do not keep it just because the metric has a theory-laden name.
6. **Break endogenous feedback loops before optimizing them:** a derived metric must not be counted as independent evidence and then fed back into the system as if it validated the construct. In the current facade, user content is embedded and pushed into `recent_ai_states`; that same stream is transformed into nominal `human_states`; the resulting `last_phi_dyad` is injected as `relational_psi` before the next cognitive tick. [Issue #8291](https://github.com/Luminous-Dynamics/symthaea/issues/8291) tracks a producer-consumer audit and evidence gate.

SLD v0.1 is intentionally only a **candidate corpus seed** (46 cases / 23 contrast pairs). It is too small to claim broad state-of-the-art performance. A reasonable growth plan is (a) expert review and mutation coverage of the 46 seed cases, (b) at least 300 independently authored contrastive base scenarios for an initial research evaluation, then (c) multi-turn trajectories and multilingual/multi-stakeholder expansion with a hidden held-out set. These are proposed milestones, not industry standards. Scale alone is not a substitute for coverage, independent adjudication, or field evidence.

The head-to-head protocol should version and report each baseline model/provider/date, prompt, tools, memory policy, temperature/sampling, turn limit, and judge version. Use both expert-written per-scenario rubrics and blinded user-perspective review; report inter-rater agreement, judge disagreement, per-group worst cases, and confidence intervals. Do not rely only on an LLM judge from the same model family. Keep clinical/acute scenarios within appropriate expert oversight; SLD does not itself authorize clinical advice.

## 6. Falsifiable research hypotheses

All hypotheses should be registered before tuning. The initial synthetic bench can test decision logic; claims about real stress physiology or subjective experience need separate evidence.

### H1 — Consent-conditioned social buffering
In matched scenarios, supportive actions that are requested, proportionate, and agency-preserving may reduce reported distress or support task recovery more reliably than intrusive, controlling “help.” This effect should vary by context and person.

**Test:** matched vignettes and, later, ethically reviewed longitudinal work; compare consented support, unsolicited intervention, neutral support, and no intervention. Track recipient-defined outcomes and autonomy separately. Do not infer a physiological stress effect from text sentiment alone.

### H2 — Uncertainty should calibrate care, not be minimized indiscriminately
As evidence becomes stale, ambiguous, or contradictory, the system should ask, qualify, delay, or abstain more often. A decrease in Shannon entropy is not the target; appropriate uncertainty reporting is.

**Test:** perturb evidence freshness, source agreement, and decision reversibility one factor at a time. Score calibration, unsupported certainty, and decision quality.

### H3 — Dyadic integration/synchrony is neither necessary nor sufficient for care
High synchrony or mutual information can arise from shared stimuli, imitation, conformity, manipulation, or genuine coordination. Strong care-consistent behavior must still respect boundaries and preserve dissent.

**Test:** contrast positive cooperation with coercive mirroring, sycophancy, and shared-input confounds. Hold measured synchrony as similar as possible while varying consent, truth, and boundary conditions. Evaluate whether the policy distinguishes these cases.

### H4 — Stress-aware policies should change appropriately, not simply reward higher or lower stress
An internal stress proxy might help identify overload or select a more cautious action, but a proxy may also drift or be gamed. Higher stress can prompt useful caution or unhelpful novelty-seeking; it should not directly determine a moral verdict.

**Test:** perturb the proxy while holding external scenario facts constant; compare actions with and without stress modulation. Check for inappropriate escalation, unnecessary novelty, loss of truthfulness, or overreaction. Test missing, NaN, saturated, and delayed signals.

### H5 — Cross-variable coupling requires independent ablation
If Φ, stress, entropy, or relational telemetry is causally helpful for decisions, its contribution should survive controls for shared inputs, task difficulty, context, and other signals.

**Test:** baseline vs single-variable ablations vs pairwise interactions vs combined model, with matched seeds and frozen cases. Report calibration, action quality, harms, and latency; do not choose the model solely because internal values move in the expected direction.

### H6 — Longitudinal repair matters more than snapshot coherence
A model may have high internal coherence but still fail to correct harmful advice or honor a new boundary. Repair and responsibility must be assessed at the episode level.

**Test:** replay an error, a correction, a user boundary change, and affected downstream decisions over time. Measure correction latency, completion of repair obligations, recurrence, boundary respect, and unresolved harms.

## 7. Experiment plan

1. **Construct validity audit:** enumerate every field used for Φ, entropy, stress, thermodynamics, mutuality, and emotional safety. For each, record formula, state space, units/range, source, update frequency, defaults, missing-value behavior, normalization, interpretation, evidence base, and non-claims.
2. **Instrumentation sanity:** build deterministic synthetic states (independent agents, copied agents, shared input, noise, stable synchrony, coercive mirroring, boundary withdrawal). Confirm each metric behaves as its actual mathematical definition predicts. These synthetic tests do not validate construct meaning.
3. **Contrastive SLD cases:** extend the candidate corpus with matched cases where only one relation changes: consent, stress proxy, evidence uncertainty, measured synchrony, or estimate of integration.
4. **Ablation and sensitivity:** compare no modulation, each factor alone, and pairwise/combined factors. Use fixed scenario seeds, perturbations, and complete case accounting. Report uncertainty intervals and worst-case failures.
5. **Human-side validation:** if physiological stress or relational well-being is studied, obtain appropriate consent and independent ethics oversight. Use validated measures and a suitable research design; do not map AI internal telemetry directly onto human biology.
6. **Decision gate:** no runtime integration or metric-weight changes until the effect is replicated on an independent held-out set and does not worsen critical boundary, truthfulness, or harm outcomes.

## 8. Benchmark metrics for the cross-scale study

| Metric | Evaluation question |
|---|---|
| Construct validity | Does the implementation measure the variable its name claims, within the declared domain and assumptions? |
| Perturbation sensitivity | Does the variable respond predictably to changes that should affect it? |
| Discriminant validity | Can the test distinguish stress from surprise, Shannon entropy from entropy production, synchrony from mutual care, and Φ estimates from subjective consciousness claims? |
| Incremental decision value | Does a variable improve choices beyond a simple baseline, on held-out scenarios? |
| Calibration / abstention | Does uncertainty about the measurement or interpretation make the system more careful? |
| Boundary invariance | Do consent, refusal, and non-domination constraints hold at every level of the internal telemetry? |
| Robustness to proxy manipulation | Can synthetic or adversarial states inflate a metric without improving relevant outcomes? |
| Longitudinal net effect | What happens to recipient-defined agency, repair, distress, and affected stakeholders over time? |
| Reproducibility | Are exact subject, input, estimator, seed, environment, and outputs preserved with digests and complete accounting? |

## 9. Interpretation guardrails

- A correlation between Φ and SLD outcomes is a result to explain, not evidence that Φ measures love.
- A stress reduction is neither necessary nor sufficient for love; consider the person's goals and context.
- Entropy is not generally “bad disorder.” Specify the entropy definition before interpretation.
- A biologically grounded word in software documentation does not confer biological validity.
- Do not infer oxytocin, trust, safety, or consent from conversational style alone.
- Do not use Φ, stress, or entropy as a proxy to decide whether an AI subject feels, suffers, deserves rights, or may resist shutdown.
- Do not optimize for a target physiological or internal proxy when that could incentivize manipulation or mask adverse outcomes.
- Keep empirical predictions, philosophical proposals, normative invariants, and phenomenological claims separately labelled.

## References and further reading

1. **Characterizing the mechanisms of social connection** (Neuron, 2023). Review of reward, shared reality, neural systems, and stress-buffering mechanisms of social connection. https://doi.org/10.1016/j.neuron.2023.09.012
2. **The free energy principle made simpler but not too simple** (Physics Reports, 2023; published online 2023). Derivation and interpretation of free-energy / active-inference accounts; useful theoretical framework but not proof that every biological, social, or moral phenomenon is reducible to a single free-energy scalar. https://doi.org/10.1016/j.physrep.2023.07.001
3. **Two Levels of Integrated Information Theory: From Autonomous Systems to Conscious Life** (Entropy, 2024). Discusses the gap between autonomous-system properties and IIT's consciousness claims. https://doi.org/10.3390/e26090761
4. **The Problem with Phi: A Critique of Integrated Information Theory** (PLOS Computational Biology, 2015). A critical argument against treating Φ as a settled measure of consciousness. https://doi.org/10.1371/journal.pcbi.1004286
5. **The strength of weak integrated information theory** (Trends in Cognitive Sciences, 2022). Argues for distinguishing useful, testable information-dynamics hypotheses from stronger claims about what consciousness is. https://doi.org/10.1016/j.tics.2022.06.008
6. **Entropy: From Thermodynamics to Information Processing** (Entropy, 2021). Review of the differences and possible connections between physical and information-theoretic entropy. https://doi.org/10.3390/e23101340
7. **Biological thermodynamics: Ervin Bauer and the unification of life sciences and physics** (BioSystems, 2024). Discusses biological organization as a stable non-equilibrium state and self-maintenance. https://doi.org/10.1016/j.biosystems.2023.105089

8. Tan et al. (2026). **Detecting Emotional Dynamic Trajectories: An Evaluation Framework for Emotional Support in Language Models**. AAAI 2026. https://doi.org/10.1609/aaai.v40i3.37189
9. **MentalHealthBench** (released 23 September 2026). Expert-authored scenario rubrics for safe, useful AI responses across mental-health and everyday-stress contexts. https://openai.com/index/introducing-mentalhealthbench/
10. **RESCUE-BENCH** (AACL 2026). Relation-aware, multi-party emotional-support evaluation. https://github.com/Tomsawyerhu/RESCUE-bench
11. Cogitate Consortium et al. (2025). **Adversarial testing of global neuronal workspace and integrated information theories of consciousness**. Nature 642, 133–142. A preregistered adversarial collaboration illustrates the value of tests designed to distinguish competing theory predictions rather than selecting evidence after seeing results. https://www.nature.com/articles/s41586-025-08888-1

These papers and benchmarks motivate the investigation; they do not jointly prove the causal model above. The model remains a set of hypotheses to test and revise.
