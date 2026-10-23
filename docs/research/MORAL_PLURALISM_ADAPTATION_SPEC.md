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


## 10. Critical architecture decision: extend the existing ethics engine

**Do not create a second, parallel ethics engine.** The current repository already contains a substantial orchestration and evaluation path. MORAL-PLURALISM-001 should extend that path through a read-only framework-evaluation seam and preserve existing callers until compatibility and safety tests justify changes.

### Existing components and their appropriate roles

- `src/cognitive_loop/ethics_engine.rs` — existing unified orchestrator. Keep it as the coordination boundary; do not introduce a competing orchestrator with different action semantics.
- `src/cognitive_loop/moral.rs` — delegates moral input evaluation to the existing `EthicsEngine`.
- `src/hdc/moral_parser.rs` — extracts moral primitives using pattern matching and keyword analysis. Treat its parse as a fallible interpretation of input, not ground-truth intent, consent, or a complete semantic-role parse.
- `src/hdc/moral_algebra/` — existing HDC moral representations and judgments, including similarity-based prototypes and deontological/obligation judgments. Preserve these as the baseline evaluator; do not treat vector similarity alone as proof of moral validity.
- `src/consciousness/unified_value_evaluator/` — existing Eight Harmonies-based action/value assessment. Expose it as one explicit, versioned evaluation profile, not as the definition of every ethical theory.
- `HarmoniesIntegrator` and `src/hdc/harmony_basis.rs` — keep as the project's current value-model implementation and interaction structure.
- `MoralTopology` — use as a diagnostic for patterns, anomalies, or instability only where its metrics have been validated; do not let topological scores decide moral truth.
- The institutional-compliance checker inside the existing ethics pipeline — treat pattern/similarity flags as candidate risk signals, not legal determinations. Legal controls need explicit, current, jurisdiction-scoped rule sources and traceability.
- `src/cognitive_loop/ethics_values_manager.rs` — existing contextual harmony weights and subsystem ownership should be audited before adding new state or learning loops.

### Proposed additive flow

1. Parse the scenario once using the existing engine and record which fields were detected, which were inferred, and which remain unknown.
2. Construct a versioned normalized context that preserves provenance and uncertainty; never upgrade keyword detections into confirmed consent or intent.
3. Run the current HDC/deontological judgment and Eight Harmonies evaluator unchanged as baseline assessments.
4. Add read-only framework adapters that consume the normalized context and return independent, typed `FrameworkAssessment` values. The current Eight Harmonies model is one adapter; new profiles may include explicitly scoped consequentialist, duty/rights, virtue, or care-ethics models.
5. Compare per-profile results and expose agreement, conflict, missing information, and abstention. Never silently average incompatible conclusions.
6. Return an additive plural-ethics report through the existing orchestration boundary. It must not itself grant tool permissions, alter the execution gate, or rewrite the underlying evaluator results.

### Reuse and qualification rules

- Extend existing test families—especially `tests/adversarial_moral_algebra.rs`, `tests/ethics_gating_integration.rs`, `tests/moral_topology_api_integration.rs`, and `tests/proptest_ethics_gating.rs`—rather than creating redundant suites that repeat the same assertions.
- Add profile-isolation tests proving profile selection affects framework-relative assessments but does not change tool permissions or independent safety/authorization controls.
- Add parser uncertainty tests for negation, implicit coercion, multi-clause scenarios, missing context, and contradictory statements.
- Keep the existing baseline outputs stable until side-by-side tests establish what changes and why.
- Inventory exact-head source and tests before claiming any behavior is implemented. The presence of a module or passing unit tests does not establish faithful support for every ethical framework.
- Reconcile the stale 91.1% care-classification claim against the README's corrected 56.2% headline before using either as qualified evidence.

This decision makes pluralism an extension of Symthaea's existing engine, not a parallel moral authority.


## 11. Implementation and current research update — 2026-10-09

### First additive code seam

A separate draft implementation now exists in [PR #7264](https://github.com/Luminous-Dynamics/symthaea/pull/7264), adding `src/cognitive_loop/ethical_pluralism.rs`. This is deliberately a **comparison contract**, not a complete evaluator or replacement engine. It validates versioned assessment identity, rationale, premise references, optional confidence, and evidence references, then reports agreement, disagreement, incompleteness, or invalid input. It has no tool permission or execution field.

The initial unit tests cover empty input, unanimous support/opposition, explicit disagreement, conditional judgments, duplicate framework/version records, invalid confidence, missing traceability, blank source references, and order-independent aggregate status. They have been authored but not run in this workflow. The module is not yet wired to generate assessments from MoralAlgebra or UnifiedValueEvaluator; those adapters and integration gates remain future work.

### Principle-priority robustness is now a required test dimension

A 2026 study in *Computers in Human Behavior Reports* reports that changing the priority order of ethical principles can reverse model decisions, while transparent ethical-priority policies are often underspecified. See [“What is (More) ethical in AI when ethical principles compete?”](https://doi.org/10.1016/j.chbr.2026.101182).

Implication for Symthaea: profile metadata must distinguish a framework's principles from its conflict-resolution/prioritization rule. The engine must not accidentally treat list order, map iteration order, HDC bundle order, or whichever adapter ran first as moral priority. If a framework specifies priorities, encode and version that rule explicitly. If priority is unspecified and materially affects the answer, return a conflict/underdetermined result and expose a sensitivity analysis.

A new study, [AMULED: Addressing Moral Uncertainty using Large language models for Ethical Decision-making](https://doi.org/10.3389/frai.2026.1754973), is another recent multi-theory proposal. It reinforces that multi-framework ethics is an active area of research, not a unique invention of this project. Symthaea's contribution must therefore be demonstrated through exact-head integration, faithful framework implementation, meaningful held-out tests, and clear evidence—not the mere presence of several labels.

The code-level comparison primitive's order-invariance test checks only that the **aggregation of already-produced assessments** does not depend on input order. It does not prove that framework evaluators or their internal priority rules are order-invariant; those require separate tests.


### Exact subject binding before comparison — 2026-10-09

The first comparison API exposed a correctness seam: it could compare rows from different scenarios or different candidate actions and still report apparent agreement. The follow-on draft [PR #7273](https://github.com/Luminous-Dynamics/symthaea/pull/7273), stacked on [PR #7264](https://github.com/Luminous-Dynamics/symthaea/pull/7264), adds an explicit subject tuple to each assessment:

- `scenario_ref` and `scenario_digest` bind the assessment to the normalized scenario/context.
- `candidate_action_ref` and `candidate_action_digest` bind it to the exact action under evaluation.
- Comparisons with different subject tuples return `SubjectMismatch`; empty binding fields invalidate the input.
- Each premise reference is checked individually, so a blank ID cannot hide beside a valid ID.

**Integrity boundary:** the comparison module checks that supplied IDs/digests are non-empty and exactly equal across rows. It does not compute a digest, authenticate the producer, or prove that the digest corresponds to source bytes. An upstream context constructor must canonicalize and hash the scenario and action; higher-integrity deployments should retain the canonical bytes and their independent verification receipt.

The follow-on tests are authored but have not been executed in this workflow. Workflow metadata for the new head currently reports completed-but-skipped checks, not successful tests. Do not treat this draft contract as qualified until exact-head tests and independent subject-binding checks have run.

This is also aligned with current literature: a September 2026 study argues that case-by-case agreement metrics miss the relational structure of moral judgments, and that structural alignment does not itself prove a model genuinely holds a value ([Tang, 2026](https://link.springer.com/article/10.1007/s43681-026-01368-w)). For Symthaea, this supports both exact case/action binding and later cross-case structural consistency tests. It does not validate this implementation by itself.


### Existing EthicsEngine freshness is a blocker to safe adapters — 2026-10-09

A source-level inspection shows why the plural comparison API must not be wired directly to a current-cycle wrapper over `EthicsEngineOutput`:

- MoralParser/MoralAlgebra updates only when `cycle % 7 == 0`; between those cycles, the last numeric score is carried forward while the verdict strings are empty and several related fields are reset to placeholders.
- UnifiedValueEvaluator updates only when `cycle % 19 == 0`; between those cycles, its numeric score is cached while its decision string is empty. Its current active call evaluates the generic string `cognitive_cycle`, not the current input/action text.
- HarmoniesIntegrator also updates on a 19-cycle cadence and carries cached alignment/approval between fresh evaluations. The output does not expose a uniform per-assessment origin-subject binding for these subsystems.

Those are current implementation facts, not necessarily defects in the cognitive loop's intended schedule. They do mean that a plural-ethics adapter must not label every returned field as a fresh assessment of the current candidate action. The next prerequisite is tracked in [MORAL-PLURALISM-002](https://github.com/Luminous-Dynamics/symthaea/issues/7274): retain the source subject and freshness of each sub-assessment, make stale data explicitly unavailable for cross-framework comparison, and distinguish system-state signals from action-specific moral assessments.

Do not interpret empty verdict strings on skipped cycles as Neutral, do not rebind cached outputs to a new subject, and do not treat the generic `cognitive_cycle` value-evaluator call as a candidate-action evaluator. Required regression cases include cycle boundaries 6/7/8 and 18/19/20, where the input subject changes while some subsystem results are carried forward.

The subject-binding work in [PR #7273](https://github.com/Luminous-Dynamics/symthaea/pull/7273) is a necessary comparison-layer contract, but it cannot compensate for inaccurate provenance from an upstream producer. Tests and CI are still unqualified at this point; the recent workflow records are skipped rather than passing.


### Provenance as an explicit contract — 2026-10-09

The additive provenance contract is being extended in [PR #7278](https://github.com/Luminous-Dynamics/symthaea/pull/7278), stacked on [PR #7273](https://github.com/Luminous-Dynamics/symthaea/pull/7273):

- Each assessment carries the exact source subject, evaluator build/revision identity, evaluation sequence/cycle, and an explicit freshness state.
- A caller can use `compare_assessments_for(expected_subject, assessments)` to bind the comparison to the exact scenario/action it intends to discuss.
- A source-subject mismatch is not silently normalized to the requested subject.
- A carried-forward result is surfaced as `StaleAssessment`, not agreement.
- An unavailable evaluator result is represented explicitly as `Unavailable` and yields `Incomplete`; it does not get a fake neutral stance or invented premise/evidence references.

This follows the general model of [W3C PROV-DM](https://www.w3.org/TR/prov-dm/) and the [PROV Primer](https://www.w3.org/TR/prov-primer/): track the entities and activities involved in producing information, rather than trusting labels alone. Symthaea's current draft does **not** claim full PROV conformance. All provenance fields are still caller-supplied; build identity is not attested, and source digests are not recomputed here. Source integrity must be established by an upstream canonicalizing context constructor and an independent verification/evidence path.

The current comparison module still does not know an expected registry of every framework unless the caller supplies an explicit unavailable row for the framework. Adapter integration must preserve coverage information and must not silently omit a framework that did not run. The comparator does not calculate moral truth or grant execution permission.

Tests in this draft are authored but not run; the workflow records for the latest checked head are skipped, not passing. Next qualification requires exact-head Rust tests in the pinned Nix environment and cadence-boundary tests across the 7-cycle and 19-cycle evaluation schedules. The existing EthicsEngine and action gate remain unchanged.


### Declared framework-roster coverage — 2026-10-09

The comparison now has a strict roster-based entry point in [draft PR #7279](https://github.com/Luminous-Dynamics/symthaea/pull/7279), stacked on the provenance work in [#7278](https://github.com/Luminous-Dynamics/symthaea/pull/7278). Callers provide a versioned list of required framework identities along with the exact subject/action under comparison.

- A missing required framework is reported via `missing_frameworks` and makes the result `Incomplete`, even if all supplied assessments support the action.
- A framework/version outside the declared roster is reported via `unexpected_frameworks` and yields `FrameworkSetMismatch`.
- An explicit `Unavailable` row represents the expected framework without pretending it evaluated the case; it keeps the result incomplete and cannot claim confidence, premise, or evidence references.
- Empty, malformed, or duplicate rosters fail closed. Coverage arrays are sorted for deterministic output, and a roster mismatch must not hide subject/provenance/freshness failures.

This is evaluation **coverage**, not proof of moral adequacy. A caller can still define an incomplete or biased roster; the API cannot decide which frameworks a domain morally requires. The roster should be declared, versioned, justified for the deployment context, and reviewable by affected parties—not inferred silently by the evaluator.

That distinction is consistent with the current research gap described in [“Position: Evaluations of AI Moral Reasoning Still Miss Half of the Picture”](https://arxiv.org/abs/2608.14566) (2026): values-level alignment and norm identification/application are different evaluation targets, and moral competence also depends on identifying relevant contextual features. Therefore later Symthaea qualification should assess both value representation and context-sensitive norm application, rather than scoring only multi-framework agreement.

The [NIST AI RMF Measure guidance](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) calls for documented test sets, metrics, uncertainty and evaluation methods, and recommends independent review and consideration of external stakeholders. The [AI Standards Lab's June 2026 endorsement of the Evals-Consensus recommendations](https://aistandardslab.org/we-have-endorsed-the-recommendations-coming-out-of-the-evals-consensus-project/) further supports treating evaluation coverage, documentation, execution, and reporting as part of the contract, not informal assumptions. Symthaea's current tests are authored but remain unexecuted; these sources motivate the design and do not validate the implementation.


### Framework plurality and roster sufficiency — 2026-10-09

The strict roster API in [draft PR #7279](https://github.com/Luminous-Dynamics/symthaea/pull/7279) now distinguishes **coverage** from **plurality**:

- Missing a required framework/version yields `Incomplete`, with a sorted `missing_frameworks` list.
- Supplying an undeclared framework/version yields `FrameworkSetMismatch`, with a sorted `unexpected_frameworks` list.
- Fewer than two distinct framework IDs yields `InsufficientFrameworks`; multiple versions of one framework do not count as independent perspectives.
- Explicit `Unavailable` rows preserve the fact that a required evaluator did not run and cannot claim a confidence value or used premise/evidence references.
- Subject, provenance, or freshness failures are not masked by a simultaneous roster mismatch.

This makes a declared roster auditable, but does not settle which roster is ethically adequate. Roster choice must remain explicit, versioned and justified for the use case, with affected-party representation considered where impacts warrant it. A technically complete but normatively narrow roster is still inadequate.

This reflects current measurement guidance: [NIST AI RMF Measure](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) calls for documented test sets, metrics and methods, measurement uncertainty, regular assessment, and independent/external perspectives where relevant. A 2026 position paper on moral-reasoning evaluations likewise distinguishes values-level alignment from identifying and applying context-sensitive norms ([arXiv:2608.14566](https://arxiv.org/abs/2608.14566)). Consequently, later qualification should measure not only whether declared frameworks agree, but also whether each framework identifies relevant facts and applies its stated norms faithfully.

The latest draft now has 30 authored unit tests in `ethical_pluralism.rs`; none should be described as passing until executed on the exact head in the pinned environment. PR #7279 remains draft and does not wire the comparator into the existing EthicsEngine or alter authorization.


### Canonical subject fingerprints — 2026-10-09

Draft PR [#7283](https://github.com/Luminous-Dynamics/symthaea/pull/7283), stacked on [#7279](https://github.com/Luminous-Dynamics/symthaea/pull/7279), now adds a BLAKE3 content-binding constructor for assessment subjects and a verifier that recomputes scenario/action digests from canonical byte slices. Scenario and candidate-action hashes use different versioned domain tags. A single-call `compare_assessments_with_canonical_bytes(...)` entry point computes the expected subject from those bytes before checking roster completeness, subject equality, provenance, and freshness.

The root crate already depends on BLAKE3, so this change adds no dependency. Five constructor tests and three direct entry-point tests were added in this pass, including determinism, domain separation, byte changes, forged digest rejection, and matching assessments.

**Important boundary:** Symthaea does not yet define a canonical serialization format in this function. The caller must provide canonical scenario/action bytes and pass the exact bytes the evaluator actually consumed. Hashing arbitrary, differently ordered JSON does not make it semantically canonical. The helper verifies byte-to-digest consistency; it does not establish producer identity, evaluator execution, or truth of the encoded facts. A future upstream context constructor should own a versioned canonical serialization schema and the evaluator adapter must preserve those exact bytes through evaluation.

This supports the NIST AI RMF's Measure function and AI Resource Center's TEVV framing, which emphasize suitable test methods, measurable uncertainty, and documented evaluation practice ([NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework); [NIST AI Resource Center](https://airc.nist.gov/)). It also supports the W3C PROV distinction between a produced data entity and the activity/agent responsible for it ([PROV-N overview](https://www.w3.org/TR/prov-n/)). These standards inform design; the implementation claims neither NIST validation nor full W3C PROV conformance.

Tests remain authored but unexecuted, and the current PR is still draft. The next qualification step is execution against the exact head, followed by a versioned canonical context schema and an adapter whose captured source bytes can be independently compared against the computed subject.


### Descriptive norms are not normative authority — 2026-10-09

A recent paper, [“Large language models outperform humans at estimating society's everyday norms”](https://www.nature.com/articles/s44488-026-00018-8), published 1 September 2026, studies six LLMs across 555 everyday scenarios and compares predictions with 320 human participants. This is useful work on **descriptive norm estimation**—predicting what people in a studied population tend to regard as appropriate—not evidence that those norms are justified, universal, or morally binding.

For Symthaea, represent these as different claim types:

- **Descriptive norm estimate:** a population- and dataset-scoped empirical claim about what respondents tend to approve/disapprove of, with dataset version, geography/culture/population, sampling method, uncertainty, and date.
- **Normative premise/rule:** a value, duty, right, or justification rule that needs its own provenance and declared scope.
- **Application judgment:** the result of applying stated normative rules to explicit facts/context, including uncertainty about which contextual features are morally relevant.
- **Operational authorization:** a separate security/safety decision, never inferred solely from a majority preference or social-norm estimate.

A descriptive estimate can inform context or reveal stakeholder expectations; it must not be silently converted into a premise saying that the majority is right. Likewise, observed cultural or institutional norms should not automatically override consent, rights, or safety boundaries. This separation complements the 2026 argument that evaluations should test both value representation and context-sensitive norm application ([Kierans et al.](https://arxiv.org/abs/2608.14566)).

The subject fingerprint in [PR #7283](https://github.com/Luminous-Dynamics/symthaea/pull/7283) now allows exact scenario/action bytes to be bound to an assessment, but the claim type, dataset scope, and provenance still need to be supplied by a future evidence/context layer. Hashing a norm statement does not validate the norm.


### Context schema identity is part of the assessment subject — 2026-10-09

Draft PR [#7285](https://github.com/Luminous-Dynamics/symthaea/pull/7285), stacked on [#7283](https://github.com/Luminous-Dynamics/symthaea/pull/7283), now binds each subject's digests to explicit `context_schema_id` and `context_schema_version` fields. The schema ID/version are included in the length-prefixed, domain-separated BLAKE3 input for both the scenario and candidate action. Same bytes under different schema versions therefore produce different subject digests. The canonical-byte comparison API takes this schema identity explicitly.

The public subject struct can be constructed without its constructor, so validation also checks schema ID/version on both the reported subject and provenance source subject; the byte verifier refuses subjects whose schema identity is blank. This avoids relying on constructor validation alone.

#### JSON canonicalization decision

For JSON-based context payloads, adopt a reviewed implementation of [RFC 8785, JSON Canonicalization Scheme (JCS)](https://www.rfc-editor.org/rfc/rfc8785.html) rather than implementing number formatting and recursive property sorting by hand. RFC 8785 requires deterministic property sorting and ECMAScript-compatible JSON primitive serialization for hash/signature use. The Rust crate [`serde_json_canonicalizer`](https://docs.rs/serde_json_canonicalizer/latest/serde_json_canonicalizer/) advertises an RFC 8785-compatible `to_vec`/ `to_string` implementation; version 0.3.2 is published as of this research pass. It is a candidate for an isolated dependency review and test-vector comparison, **not yet adopted or qualified in Symthaea**.

Before adopting any canonicalizer, verify published RFC 8785 vectors (including nested object sorting, arrays, Unicode, escaped control characters, and difficult floating-point cases), duplicate-key/input validity policy, dependency tree and licensing, and Rust/toolchain compatibility. If the domain model can avoid floating-point JSON numbers or adopts a schema-specific canonical binary encoding, that should be an explicit versioned choice instead of silently claiming JCS conformance.

The current PR still accepts caller-supplied canonical bytes; it binds those bytes to a schema identity and digest but does not canonicalize JSON itself. Hashing must not be described as canonicalization, schema validation, source authentication, or proof that the evaluator consumed the bytes. The existing EthicsEngine and action gate remain unchanged. The module's 42 tests are authored but unexecuted on this exact head.

### JCS candidate source audit — 2026-10-09

**Disposition: retain as a candidate; do not adopt yet.** This was a static-source review, not a build, dependency-tree execution, or test run.

The published `serde_json_canonicalizer` 0.3.2 manifest declares MIT licensing, Rust edition 2021, and runtime dependencies on `ryu-js ^1.0.1`, `serde ^1.0`, and `serde_json ^1.0` with `float_roundtrip`. Its manifest does **not** declare `rust-version`/MSRV. Symthaea's repository pins Rust 1.96.0 in `rust-toolchain.toml` and uses edition 2024; the candidate's compatibility with that exact pinned toolchain remains unverified until built and tested there. The crate adds a runtime `ryu-js` dependency, while Symthaea already uses `serde_json`; a lockfile/dependency-tree diff and license scan are still required before adoption.

The candidate's checked-in tests include RFC 8785 example/sorting data (sections 3.2.2 and 3.2.3), a reference-implementation test-data suite, number-formatting cases from Appendix B, and non-finite-number rejection tests. It also contains a generated-number stress test marked `#[ignore]`; its source comment says the input file is generated separately and is about 3.7 GB. These test assets are encouraging, but their existence does not establish that tests pass on Symthaea's pinned toolchain or cover the production ingestion path.

#### Input-contract blocker: duplicate JSON object properties

The candidate's `pipe(json)` implementation parses with `serde_json::from_str::<serde_json::Value>` before serializing. The `serde_json` project documents that duplicate keys parsed into `Value` use last-value-wins behavior ([serde_json issue #1112](https://github.com/serde-rs/json/issues/1112); [issue #762](https://github.com/serde-rs/json/issues/762)). RFC 8785 says JCS input must not contain duplicate property names ([RFC 8785 §3.1](https://www.rfc-editor.org/rfc/rfc8785.html)). Therefore, **`pipe` is not by itself an acceptable strict-ingestion boundary for untrusted raw JSON**: it can canonicalize a value after a duplicate property has already been discarded. This is a specific known gap, not merely an untested edge case.

For the first integration, prefer serializing a validated typed envelope directly with `to_vec`. If raw JSON must be accepted, put a duplicate-rejecting parser/visitor in front of canonical serialization and test duplicate names recursively, including escaped aliases that decode to the same key. Preserve the original accepted bytes or the exact resulting canonical byte slice all the way into the evaluator; do not reconstruct the evaluator input after fingerprinting.

#### Required isolated qualification matrix

1. Run the upstream crate tests under the pinned Rust 1.96.0 environment and record the exact lockfile, commit/release, command, exit status, and logs. Separately decide whether to generate and run the ignored 100-million-case number corpus; do not imply that the default suite includes it.
2. Add Symthaea-owned golden vectors for RFC example output, nested key sorting, UTF-16 key ordering, array-order preservation, Unicode and control escapes, signed zero, exponent thresholds, binary64 rounding boundaries, non-finite serialization rejection, malformed JSON, trailing data, and lone-surrogate input.
3. Add fail-closed tests for duplicate property names at the root and nested levels, including alternate escape spellings of the same decoded key. These tests must exercise the exact intended ingestion path, not only the serializer on already-parsed values.
4. Show property-order invariance (semantically equivalent objects serialize to equal bytes and digests) and semantic-change sensitivity (a changed value or array order changes bytes and digest). Document that JCS uses IEEE-754 binary64 number semantics; encode precision-sensitive identifiers and exact quantities as schema-defined strings/integers rather than allowing silent precision loss.
5. Check the dependency graph, license compatibility, and locked versions. Record an explicit canonicalization profile/version alongside schema ID/version. For non-JSON or binary inputs, keep the existing byte-oriented path but require a named/versioned encoding profile rather than labeling arbitrary bytes JCS.
6. Before any `EthicsEngine` adapter, produce a receipt that binds the evaluator result to the exact scenario/action byte digests, schema and canonicalization profile, evaluator build identity, invocation identity, and the bytes actually consumed by that invocation. Caller-supplied metadata alone does not attest evaluator execution.

Until these are evidenced on an exact commit and reviewed, the current subject helper establishes only byte-to-digest consistency. It does not canonicalize JSON, prove input validity, or prove that the evaluator consumed the fingerprinted bytes. The plural-ethics comparator remains read-only and must not alter action authorization.

