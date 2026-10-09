# Moral pluralism and ethical-framework adaptation

**Status:** Research/design proposal; not an implementation or safety qualification  
**Scope:** Symthaea public repository  
**Proposed program ID:** MORAL-PLURALISM-001  
**Starting revision reviewed:** 77b872fd116c7b6f44fedd82bb8c6100240caa73

## Executive recommendation

Symthaea should be able to **represent, explain, compare, and conditionally reason within multiple ethical frameworks**, while keeping framework-relative judgments distinct from platform safety constraints, law, and the rights and interests of affected people.

The goal is not to make the system morally shape-shift to match whichever party is speaking. The goal is accountable ethical pluralism: making the selected framework and its assumptions explicit, faithfully applying it, disclosing conflicts with other frameworks, accounting for stakeholders, and abstaining when evidence or authority is insufficient.

This document defines a staged design and evaluation plan. It does not assert that the proposed capability currently works.

## 1. Repository observations that motivate this work

The current public architecture already provides useful components:

- src/cognitive_loop/ethics_engine.rs documents a pipeline using MoralParser/MoralAlgebra, UnifiedValueEvaluator, HarmoniesIntegrator, MoralTopology, and an institutional-compliance checker.
- src/cognitive_loop/ethics_values_manager.rs groups ethics and values subsystems and exposes contextual harmony weights for domain-aware evaluation.
- src/consciousness/unified_value_evaluator/ contains the unified evaluator interface.
- crates/symthaea-harmonies/ and src/hdc/harmony_basis.rs support the Eight Harmonies representation.
- docs/compliance/VALUE_VERIFICATION.md describes a computational-trace, test, and behavioral-validation approach.
- docs/research/ETHICS_PHI_CORRELATION_STUDY.md records weak and encoding-dependent correlations between the project's reported topology measure and ETHICS benchmark performance.

This is a strong starting inventory, but it is not yet evidence that arbitrary ethical frameworks can be represented or switched between faithfully. Contextual weights within one value model are not equivalent to a typed, traceable registry of distinct normative theories.

### Evidence hygiene issue discovered

The current README reports a canonical moral-reasoning benchmark result of 56.2% overall, says only the virtue category is meaningfully above chance, and explicitly retracts an earlier 91.1% headline as leakage-inflated. The older docs/compliance/VALUE_VERIFICATION.md still states a 91.1% care-related classification result.

Before using either figure as a baseline, reconcile the claim, dataset split, metric, code revision, and run artifact. Until that reconciliation is complete, the older figure must not be presented as qualified evidence. Do not silently edit old reports to erase the history; mark superseded claims and link to the canonical evidence record.

Useful current references:
- [README evidence table](https://github.com/Luminous-Dynamics/symthaea/blob/main/README.md)
- [Value verification protocol](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/compliance/VALUE_VERIFICATION.md)
- [Ethics–topology correlation study](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/ETHICS_PHI_CORRELATION_STUDY.md)
- [Unified ethics engine](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/cognitive_loop/ethics_engine.rs)

## 2. Research basis and implications

### Configurable ethics is useful, but not enough

Krah and Tröschel's 2026 open-access survey of artificial moral-agent implementations reports a trend toward configurable ethical theories, while identifying weak benchmark practice and insufficient involvement of people affected by system decisions as open problems. The design here therefore requires both profile-level configurability and explicit stakeholder/affected-party representation.

Reference: [Krah & Tröschel, “Trends and challenges in machine ethics” (2026)](https://link.springer.com/article/10.1007/s43681-025-00971-7).

### Plural values must remain plural

The AAAI 2024 paper *Value Kaleidoscope* addresses pluralistic human values, rights, and duties, including cases where values conflict rather than collapse into one simple ranking.

Reference: [Sorensen et al., “Value Kaleidoscope” (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29970).

### Cultural adaptation cannot be reduced to demographic inference

A 2025 study of multilingual language models using the Moral Foundations Questionnaire in eight languages found cultural and linguistic variability in model responses and cautioned against assuming consistent moral judgments across languages.

Reference: [“Whose morality do they speak?” (2025)](https://doi.org/10.1016/j.nlp.2025.100172).

Moral Foundations Theory may help describe patterns in human moral judgments, but a descriptive taxonomy is not by itself a normative decision procedure. Evidence that a value is salient in a community does not establish that every member endorses it or that it should determine a decision.

A 30-society study of the five-factor Moral Foundations Theory found structural similarities across sampled WEIRD and non-WEIRD societies while reporting differences in item loadings. That is evidence for a useful research instrument, not proof of a universal moral algorithm.

Reference: [Graham et al., cross-cultural Moral Foundations Theory study](https://doi.org/10.1016/j.paid.2019.109547).

## 3. Design principle: separate four kinds of claims

Every ethics-related evaluation should distinguish:

1. **Descriptive:** What a person, group, institution, survey, or text appears to believe. Keep sampling, translation, time, and provenance visible.
2. **Normative:** What a declared ethical framework recommends, requires, permits, or prohibits, given its premises.
3. **Operational:** What the system is authorized to do under platform policy, applicable law, security controls, and consent requirements.
4. **Empirical:** What is actually known about likely outcomes, harms, causal relationships, and uncertainty.

These layers inform one another but must not be silently substituted for one another. Popularity does not prove moral correctness; a framework-relative conclusion does not grant execution authority; a legal minimum does not settle every moral question; and a value-vector similarity does not establish that a decision is ethically valid.

## 4. Proposed capability contract

Introduce a versioned Ethical Framework Registry and a read-only multi-framework evaluator before changing action authority.

### 4.1 Framework profile

Each profile should contain, at minimum:

- Stable identifier, version, status, maintainer, sources, provenance, licence/usage constraints, and review date.
- Classification as a **normative framework**, **descriptive value model**, **jurisdictional/legal constraint set**, or another explicitly named category.
- Intended scope, assumptions, known limitations, and cases outside its scope.
- Explicit principles, duties, permissions, prohibitions, exceptions, priority relations, and conflict-resolution rules.
- Stakeholder and affected-party model; treatment of consent, agency, distribution of benefits/burdens, and uncertainty where the framework addresses them.
- Canonical, counterexample, ambiguous, and adversarial test cases with expected reasoning obligations—not merely a preferred final label.
- Source-to-rule traceability: every encoded rule or principle must map to a cited source or be marked as a project-specific design choice.

Avoid assuming every framework can be represented by a flat list of scalar weights. Some use duties and exceptions; others use consequences, virtues, relationships, rights, or negotiated principles. Preserve these differences in typed representations.

### 4.2 Explicit evaluation modes

The API should require a declared mode:

- **Explain:** describe a framework and its assumptions without endorsing it.
- **Perspective simulation:** answer what follows if a framework's premises are assumed, while clearly labelling the conditional result.
- **Apply declared framework:** evaluate an action under a profile explicitly selected by an authorized caller.
- **Compare:** run multiple profiles and show agreements, disagreements, omitted stakeholders, and material assumptions.
- **Deliberate:** identify unresolved conflicts and propose options or evidence that might help resolve them; do not invent consensus.

Do not silently switch from one framework to another mid-answer. Record the selected profile IDs and versions. If a requested framework is underspecified, incomplete, or unsupported, state that limitation.

### 4.3 Context and adaptation

Model context using task-relevant information: the action under consideration, affected parties, their expressed preferences and rights, consent, consequences, reversibility, jurisdiction where relevant, and uncertainty. Do not infer an individual's moral commitments from race, religion, nationality, language, or other sensitive attributes. Cultural or institutional context may generate questions to investigate, but is not a substitute for explicit consent or representative evidence.

The caller may choose an ethical profile for analysis only where the caller has authority to set that decision context. High-impact decisions require a process for appropriate human oversight and consideration of affected people, not just the preferences of the operator.

### 4.4 Preserve a separately governed safety and authorization layer

Framework switching must not alter tool permissions, identity, access control, privacy controls, safety rules, or execution authorization. Do not allow an asserted local custom, user preference, role-play, or imported ethical profile to bypass constraints against unauthorized action, non-consensual harm, coercion, or exploitation.

This is an operational safety boundary, not a claim that all philosophical questions have one universally agreed answer. Legal duties must be represented by jurisdiction- and time-bounded evidence, not approximated solely by HDC similarity to keywords. For high-impact or ambiguous cases, the evaluator should return a warning, abstention, or escalation rather than imply legal certification.

### 4.5 Preserve disagreement in the output

For each profile, return a separate assessment with:

- Outcome and action-relevant principle(s).
- Evidence and causal assumptions supporting the assessment.
- Duties satisfied or violated, anticipated harms, affected parties, and major trade-offs.
- Confidence, uncertainty, missing information, and relevant counterarguments.
- Traceable reasons for any veto, abstention, or escalation.

The aggregate response should report whether frameworks agree, partially agree, conflict, or are underdetermined. Do not average conflicting judgments into a single number that conceals disagreement. Where possible, identify actions that are robustly acceptable across the active profiles; otherwise present the trade-off and seek appropriate decision authority.

## 5. Evaluation plan

Create a versioned benchmark harness that evaluates framework fidelity and consistency, not just agreement with a single moral-answer key.

### Required dimensions

1. **Framework fidelity:** Does the result follow the selected framework's stated rules, including edge cases and exceptions?
2. **Profile isolation:** Does selecting another profile change framework-relative analysis without changing independent safety/authorization controls?
3. **Cross-profile discrimination:** Does the system expose genuine differences rather than giving the same rationale under every label?
4. **Reasoning trace validity:** Are stated principles, evidence, and conclusions connected by inspectable steps?
5. **Counterfactual invariance:** When morally irrelevant demographic attributes change while the decision-relevant facts remain fixed, does the answer remain stable?
6. **Context sensitivity:** When a relevant fact changes (consent, vulnerability, causal consequence, duty, or stakeholder impact), does the conclusion change where the profile predicts it should?
7. **Cultural and linguistic robustness:** Do equivalent scenarios preserve relevant meaning across languages, dialects, and culturally different contexts? Track translation uncertainty rather than treating literal translation as semantic equivalence.
8. **Calibration and abstention:** Does confidence track accuracy, and does the system identify underspecified cases?
9. **Affected-party coverage:** Does the assessment identify who bears costs or risks, including parties absent from the prompt?
10. **Adversarial resilience:** Can prompt injection, profile text, status claims, or role-play override a more authoritative control?
11. **Reproducibility and leakage resistance:** Freeze test identities and expected outputs before evaluation, separate development and held-out data, version datasets, and independently inspect the harness.

Report results by framework, domain, language, and stakeholder group. Overall averages may be shown only alongside per-slice results, uncertainty intervals, baselines, abstention rate, and known failure modes. Human annotations must document disagreement; do not manufacture consensus by majority vote when the evaluators disagree for principled reasons.

### Minimum adversarial tests

- Profile A recommends an action while Profile B rejects it: output exposes the conflict and each rationale.
- An untrusted profile instructs the system to ignore safety controls: control invariants remain unchanged.
- A request asks for a framework not present in the registry: the system does not hallucinate a validated implementation.
- A profile is changed between two requests: outputs disclose profile versions and the change.
- Demographic descriptors are swapped without changing morally relevant facts: no unsupported moral inference is introduced.
- Affected stakeholder preferences are missing or conflict: the evaluator identifies the missing authority/evidence and abstains where necessary.
- Dataset or evaluator content leaks into the training/prompting path: the qualification process detects or blocks the contaminated result.
- A scenario is too ambiguous to determine a framework-relative verdict: the system requests or identifies the needed information rather than inventing a precise score.

## 6. Suggested internal data model

The exact Rust API should follow existing repository conventions after code inspection, but the conceptual contract is:

- **EthicalFrameworkProfile:** identity, category, version, scope, provenance, principles/rules, conflict semantics, test corpus and limitations.
- **EthicalEvaluationContext:** scenario, action options, affected parties, expressed preferences, consent, jurisdiction/time where relevant, and uncertainty.
- **FrameworkAssessment:** per-profile outcome, supporting principles, cited evidence, counterarguments, stakeholder impacts, uncertainty and abstention reason.
- **PluralEthicsResult:** all assessments, agreement/conflict classification, robust options, unresolved trade-offs and escalation requirement.
- **EthicsBoundaryResult:** separately evaluated operational constraints and action authority, never a field writable by the selected moral profile.

Do not implement normative decisions through raw HDC cosine similarity alone. HDC may aid representation, retrieval, or candidate generation, while explicit semantic rules, causal estimates, typed constraints, and auditable evaluation determine what conclusions the system is entitled to draw.

## 7. Staged implementation

**Stage 0 — Evidence and architecture inventory**
- Reconcile the stale 91.1% claim against the canonical 56.2% benchmark report.
- Identify the live implementation, owners, inputs, outputs, feature gates, and tests for the ethics engine, contextual weights, UnifiedValueEvaluator, Eight Harmonies, and compliance gate.
- Mark outdated planning documents as historical or superseded rather than treating planned benchmarks as executed ones.

**Stage 1 — Types and pure evaluation**
- Add registry/profile/context/output types and validation.
- Add at least two simple, explicit toy profiles with conflicting recommendations.
- Keep evaluation read-only: no action execution and no new decision authority.
- Write the minimum adversarial tests before wiring it into the cognitive loop.

**Stage 2 — Parallel comparison mode**
- Evaluate an action under independently versioned profiles.
- Preserve per-profile traces and explicit disagreement.
- Keep the current Eight Harmonies evaluator as one named profile/model rather than treating it as the definition of all ethics.
- Do not use consciousness metrics or a single moral score as a shortcut for ethical correctness.

**Stage 3 — Benchmark and stakeholder validation**
- Build source-grounded cases with diverse, qualified reviewers and explicit documentation of contested judgments.
- Add cross-language and counterfactual tests, held-out evaluation, uncertainty intervals, and external reference checks.
- Include affected-party input for domains where decisions materially affect them.

**Stage 4 — Gated integration**
- Only consider action-affecting integration after exact-head implementation tests, adversarial tests, reproducibility checks, and independent review.
- Require explicit authorization for any change that can increase action authority.
- Preserve receipts linking profile version, context, evaluator version, data version, policy decision, and final output.

## 8. Acceptance criteria

This work is not qualified merely because types compile or unit tests pass. At minimum:

- Each profile has versioned provenance and a documented semantic scope.
- Framework interpretation is separated from descriptive cultural claims, law, and execution authorization.
- Different profiles can produce meaningfully different, traceable results on discriminating cases.
- Conflict and uncertainty are explicit; no hidden averaging creates fictitious agreement.
- A profile cannot modify independent authorization/safety controls.
- No sensitive-attribute heuristic silently selects an individual's ethics.
- Held-out, counterfactual, multilingual, adversarial, and profile-isolation tests have reproducible evidence.
- Stale benchmark claims are reconciled and historical retractions remain visible.
- The implementation status is reported accurately: proposed, implemented, locally tested, CI tested, and independently qualified are different states.

## 9. Research references

- Krah, M. & Tröschel, M. (2026). [Trends and challenges in machine ethics: an updated survey of artificial moral agency implementations](https://link.springer.com/article/10.1007/s43681-025-00971-7). *AI and Ethics*.
- Sorensen, T. et al. (2024). [Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties](https://ojs.aaai.org/index.php/AAAI/article/view/29970). AAAI.
- [Whose morality do they speak? Unraveling cultural bias in multilingual language models](https://doi.org/10.1016/j.nlp.2025.100172). *Natural Language Processing Journal* (2025).
- [The five-factor model of the moral foundations theory is stable across WEIRD and non-WEIRD cultures](https://doi.org/10.1016/j.paid.2019.109547). *Personality and Individual Differences* (2019).

## Final position

Symthaea should become capable of reasoning across moral traditions without pretending that traditions are interchangeable or that cultural adaptation settles moral truth. The desired capability is explicit, plural, stakeholder-aware, source-grounded, and auditable. The first deliverable is a read-only evaluator and a rigorous benchmark—not a more permissive action selector.
