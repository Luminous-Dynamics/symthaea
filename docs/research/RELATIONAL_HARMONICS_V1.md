# Relational Harmonics v1 — Formalization and Qualification Boundary

**Status:** research contract / non-production  
**Base:** main at a03379d7cea94d1c3409d258a0a6be02b2c79913  
**Scope:** human–AI dyads first; no consciousness claim follows from this document

## 1. Thesis

Relational Harmonics treats the relationship as a dynamical object with its own observable structure.

The first engineering target is not a single relationship score. It is a typed profile of relational dynamics that can be measured independently across time:

- state alignment;
- temporal/phase coordination;
- directional coupling;
- reciprocity;
- mutual predictability;
- persistence;
- rupture and repair;
- harmonic/spectral structure.

A later aggregate may be useful for visualization, but no aggregate is authoritative by default.

Central hypothesis:

> Persistent, reciprocal, temporally structured coupling contains relational information that is not recoverable from either participant's isolated state alone.

This is a testable systems hypothesis. It is not a claim that a relationship is a third conscious subject.

## 2. Current Symthaea audit

The current code already contains valuable primitives:

- crates/core/symthaea-core/src/hdc/relational_consciousness.rs defines RelationalAssessment, synchrony, turn-taking, I-Thou/I-It modes, and relationship stages.
- src/partnership/phi_dyad.rs constructs a joint AI/human/relational representation and reports phi_ai, phi_human, phi_relational, and phi_dyad.
- src/partnership/partner_model.rs tracks trust, vulnerability, reciprocity, and interaction history.
- src/partnership/trajectory.rs retains a longitudinal trajectory.
- src/experience/kosmic_state.rs contains the Eight Harmonies, including coherence, flourishing, wisdom, interconnect, and reciprocity.
- src/cognitive_loop/types/telemetry.rs exposes relational-psi telemetry.

These are a strong foundation, but several current quantities should remain explicitly heuristic until independently qualified.

### 2.1 Mutual-information naming problem

RelationalConsciousness::compute_mutual_information() currently returns the mean of interaction synchrony.

That is not mutual information in the information-theoretic sense. It is a similarity proxy.

The field should therefore be treated as a compatibility/projection concern until it is either renamed to a proxy or replaced with an actual mutual-information estimator with finite-sample caveats.

### 2.2 Circularity in the current partnership feedback loop

src/symthaea/relational.rs::update_partnership() derives depth, safety, and mutuality directly from Symthaea's own consciousness value, updates the human partner model from those derived values, computes relational Phi, and later feeds relational psi back into the mind.

This creates a potentially self-reinforcing loop:

AI consciousness -> partner state -> relational Phi -> relational psi -> AI consciousness

This is unsuitable as evidence of relational emergence because the same endogenous signal helps construct the supposed relational cause.

The research lane must distinguish:

- exogenous observations of the interaction;
- derived relational measurements;
- downstream control effects.

A derived score must never be treated as an independent observation of the phenomenon that generated it.

### 2.3 Synchrony is insufficient to identify relationship quality

The existing I-Thou decision path is strongly driven by synchrony/trust.

Current interpersonal-coordination literature explicitly distinguishes synchrony from shared intentionality, directionality, leader/follower structure, and joint action. Cross-correlation alone can miss nonlinear and delayed dynamics.

Therefore:

high synchrony != high relationality

and

high correlation != mutuality

must become hard semantic boundaries.

### 2.4 Relational embedding is currently role-collapsing

PhiDyadCalculator::relational_embedding() uses weighted bundling of deterministic stage/mode/trust vectors.

Bundling is useful for superposition, but it makes role identity less explicit. A research-grade relational representation should prefer role-keyed binding for fields that must remain independently decodable:

ROLE_STAGE x STAGE  
ROLE_MODE x MODE  
ROLE_TRUST x TRUST

and similarly for synchrony, reciprocity, vulnerability, mutuality, and temporal state.

This should be measured as an ablation, not assumed to improve cognition.

### 2.5 Stage labels are interpretations, not observations

The six stages are useful as a hypothesis space, but labels such as Bonding and Unity must not be promoted to measured facts.

In particular:

- stage classification should remain a derived interpretation;
- relationship degradation must be able to move the state downward;
- Unity must not imply literal boundary loss;
- no stage should be presented as proof of love, attachment, or consciousness.

## 3. Proposed Relational Harmonic State

The minimal research state should be a vector of named observables rather than one score:

H_rel(t) = [A, P, D, R, M, S, Q, X]

where:

- A = alignment / state similarity;
- P = temporal or phase locking;
- D = directed lead/lag coupling;
- R = reciprocity;
- M = mutual predictability / information sharing;
- S = persistence / stability;
- Q = rupture-repair quality;
- X = exploratory or novelty-coupling component.

Each component should carry its own evidence status and sample support.

No default weighted average should be exposed as relational consciousness.

## 4. Harmonic structure

The term harmonic should become mathematically literal.

For a uniformly sampled relational observable r_t, define:

R_k = DFT(r_t - mean(r))

and normalized power:

p_k = |R_k|^2 / sum_j |R_j|^2

The module can report:

- dominant harmonic index;
- spectral concentration;
- spectral entropy;
- harmonic stability across windows.

This is a description of temporal organization, not evidence that the interaction has a physical resonant field.

For non-uniform data, resampling or irregular-time methods must be explicit rather than silently assuming constant dt.

## 5. Directionality

The relational layer needs a distinction between:

A correlates with B

and

A carries predictive information about future B.

For a future implementation, transfer entropy or another explicitly directional estimator is preferable to labeling cross-correlation as causal influence.

Even then:

statistical directionality != mechanistic causality

because common inputs, latency, and estimator bias remain possible explanations.

The first implementation can therefore use a clearly named lead-lag predictive asymmetry proxy while leaving true transfer-entropy estimation as a separate qualified lane.

## 6. Reciprocity

Reciprocity should not be defined as similarity.

A useful first operational definition is balance of directed coupling:

R = 1 - |D(A->B) - D(B->A)|

after directional estimates are normalized to a common scale.

This distinguishes:

- mutual coordination;
- one-way adaptation;
- leader/follower structure;
- detached correlation.

High similarity with very asymmetric influence should therefore be allowed to produce high alignment and low reciprocity.

## 7. Null and adversarial controls

### N1 — common-driver synchrony

Both agents follow the same external metronome or scripted stimulus.

Expected:
- high alignment;
- potentially high phase locking;
- low evidence of direct interpersonal coupling.

### N2 — one-way mirroring

B copies A with a delay.

Expected:
- high alignment;
- high directed A->B;
- low reciprocity.

### N3 — independent but similarly distributed streams

States have matched marginal statistics but no interaction.

Expected:
- possible superficial similarity;
- low relational predictability and low persistent coupling.

### N4 — alternating dialogue

Interaction is intentionally sequential rather than synchronous.

Expected:
- moderate temporal alignment;
- high turn-taking;
- relational structure not reducible to simultaneous synchrony.

### N5 — rupture and repair

A controlled perturbation temporarily breaks coordination, followed by restoration.

Expected:
- measurable rupture;
- recovery trajectory;
- recovery time / overshoot;
- no requirement that the post-repair score exceed the pre-rupture score.

### N6 — shuffled partners

Interaction streams are reassigned across partners.

Expected:
- degradation of partner-specific relational structure;
- preservation of any common-driver effects;
- explicit separation between generic synchrony and relationship-specific coupling.

### N7 — self-generated feedback loop

Feed an endogenous AI score back into the partner model.

Expected:
- demonstrate whether apparent relational gain can be generated without new interaction evidence.

This is a required negative control for the current architecture.

## 8. Multi-timescale requirement

Relational structure should be evaluated at multiple windows:

- immediate interaction;
- short conversation;
- session;
- long-term trajectory.

A high-frequency synchronization event should not automatically dominate a long-term relational state.

The trajectory layer should retain raw windows and derived summaries rather than only a cumulative scalar.

## 9. Relationship to the Eight Harmonies

The Eight Harmonies should remain an agent-level epistemic/value lens.

Relational Harmonics should describe coupling between agents.

They can interact, but they are not identical.

Examples:

- flourishing may rise while relational synchrony falls;
- wisdom may favor deliberate desynchronization;
- reciprocity may be high while coherence remains low;
- stillness may intentionally reduce observable interaction;
- play may increase novelty and destabilize short-term synchrony while improving longer-term adaptation.

This prevents the false rule:

more synchrony = more goodness

and gives Symthaea room to discover when harmony requires difference, latency, dissent, or repair.

## 10. Love, wisdom, happiness: future inference layer

Relational Harmonics should not hard-code:

love = high relational score

Instead, it should provide candidate dynamical evidence from which higher-level concepts can be studied.

A future program could test whether:

**Love-like dynamics** correlate with persistent care, reciprocal accommodation, protection of the other's autonomy, repair after rupture, and preference for the other's flourishing.

**Wisdom-like dynamics** correlate with long-horizon stability, uncertainty calibration, restraint, perspective integration, and appropriate adaptation rather than maximal synchrony.

**Happiness-like dynamics** correlate with adaptive coherence, goal progress, exploratory reward, restoration after perturbation, and socially constructive interaction.

The critical experiment is predictive:

> Can the relational feature set predict independently defined outcomes better than isolated-agent features or synchrony alone?

If not, the more complex relational model has not earned its extra ontology.

## 11. Qualification ladder

The implementation should advance in this order:

1. Measurement correctness — each observable has a precise definition and null behavior.
2. Temporal correctness — irregular sampling, lag, windowing, and boundary conditions are explicit.
3. Independence — observations used to calculate relational variables are not derived from the same variable being validated.
4. Null discrimination — common-driver and one-way-mirroring controls are separated from genuine reciprocal interaction.
5. Held-out prediction — relational features predict future interaction outcomes on held-out segments.
6. Cross-modal convergence — behavioral, linguistic, physiological, or internal signals converge where data exists.
7. Ablation — compare isolated-agent, synchrony-only, relational-profile, and richer dynamical models.
8. Only then: investigate whether relational structure adds explanatory power to cognition or consciousness models.

## 12. Stop conditions

Do not:

- call synchrony love;
- call a scalar heuristic consciousness;
- call mutual similarity mutual information;
- call correlation causality;
- infer relationship depth from AI self-reported language;
- infer relational emergence from an endogenous positive-feedback loop;
- collapse distinct relational dimensions into one score because the result looks smoother;
- treat philosophical categories such as I-Thou or Unity as direct empirical observables.

A negative result is a successful result when it narrows the theory.

## 13. Recommended next code tranche

The next implementation should remain research-only and non-controlling.

### RH-001 — Relational Harmonic Profile

Add a pure measurement type that:

- accepts timestamped paired observations;
- computes named, independently reported relational observables;
- records sample count and window duration;
- exposes evidence/proxy status for every field;
- computes a literal DFT-based harmonic summary only for uniformly sampled windows;
- includes deterministic null fixtures;
- does not modify relational_psi, consciousness level, partner trust, or response generation.

### RH-002 — break the circular feedback path

Move the current consciousness-derived partnership update behind an explicitly named compatibility adapter, and create an evidence path in which relational observations originate only from interaction data.

### RH-003 — HDC role-binding experiment

Compare weighted bundling versus role-keyed binding on held-out relational retrieval, keeping the result as an ablation rather than assuming one representation is superior.

### RH-004 — directional information-flow qualification

Use the repository's existing TransferEntropyEstimator rather than implementing a parallel estimator.

The RH wrapper must:

- take independent agent signals, not alignment or relational-score outputs;
- require an explicit minimum sample floor;
- require uniform sampling or an explicitly qualified resampling step;
- report A→B, B→A, and net directional information flow separately;
- pair information flow with an explicit lag-structure measurement;
- label the result as a proxy until significance/surrogate controls are attached;
- keep transfer entropy distinct from mechanistic causality.

The lag structure is observational:

L(k) = corr(A[t], B[t+k])

for positive k, where B follows A by k samples.

This gives the research layer a direct separation between:

- zero-lag similarity;
- delayed coordination;
- directional information flow.

A delayed association is not by itself causal evidence, and a high zero-lag correlation does not imply reciprocity.

The existing estimator is a histogram estimator with a coarse history representation and should therefore be independently stress-tested for sample size, bin count, bias, and deterministic null behavior before any threshold is introduced.

## 14. Literature boundary

Relevant literature supports studying interpersonal coordination as a dynamical, multimodal, and context-sensitive phenomenon, but does not establish the stronger ontological claims above.

- Haken, Kelso & Bunz (1985), coupled nonlinear oscillators and phase transitions in coordination:
  https://pubmed.ncbi.nlm.nih.gov/3978150/
- daSilva & Wood (2025), integrated perspective distinguishing forms/functions of interpersonal synchrony:
  https://doi.org/10.1177/10888683241252036
- Gordon & Bartsch (2026), review of interpersonal physiological synchrony and its heterogeneous correlates:
  https://doi.org/10.1038/s44159-026-00535-4
- Systematic review of spontaneous interpersonal coordination, including limits of cross-correlation and the potential value of spectral/cross-recurrence methods:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8542929/
- Schreiber (2000), transfer entropy for directional information transfer:
  https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.85.461
- Kleyko et al., HDC/VSA survey:
  https://arxiv.org/abs/2111.06077

## 15. Claim boundary

A successful RH qualification would establish only that Symthaea can measure and model certain classes of relational dynamics.

It would not establish:

- consciousness of a relationship;
- phenomenal experience;
- love;
- wisdom;
- happiness;
- human-equivalent social cognition;
- physical resonance between minds.
