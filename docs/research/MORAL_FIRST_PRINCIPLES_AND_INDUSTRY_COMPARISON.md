# First-principles ethics and industry comparison for Symthaea

**Program:** MORAL-FIRST-PRINCIPLES-001  
**Status:** Research/design proposal only — not an implementation or scientific result  
**Repository baseline reviewed:** `Luminous-Dynamics/symthaea` at `77b872fd116c7b6f44fedd82bb8c6100240caa73`  
**Companion work:** [MORAL-PLURALISM-001 specification](https://github.com/Luminous-Dynamics/symthaea/pull/7253) and [tracking issue #7254](https://github.com/Luminous-Dynamics/symthaea/issues/7254)

## Executive position

Symthaea should investigate whether some moral principles can be **derived from explicit foundational assumptions plus empirical facts, causal models, and formal reasoning**. This could become a distinctive research program. We must not claim that thermodynamics already gives us a moral code or that a derivation has eliminated all value assumptions.

The project's target is a formal, auditable **conditional ethics** capability:

> Given a clearly stated set of normative premises, an empirical model with explicit uncertainty, and a context, derive the principles or action constraints that follow; identify contradictions and missing premises; test consequences against evidence; and say when the conclusion is underdetermined.

This program should be pluralistic: it should compare several plausible foundational premise sets and expose where their implications converge or diverge.

## 1. Comparison with current AI practice

The table below compares publicly described approaches and research directions. These systems have different purposes, resources, access to foundation models, and evaluation protocols; the comparison is architectural rather than a head-to-head performance ranking.

| System or research direction | Publicly documented approach | What it establishes | Open gap relevant to Symthaea |
|---|---|---|---|
| OpenAI Model Spec | Explicit target behavior for instruction following, conflict resolution, safety, customization, and helpfulness; intended behavior is a target for training and evaluation | Behavior expectations and priority structure can be made inspectable and iteratively improved | A behavior specification is not itself a proof that moral facts can be derived from natural facts |
| Anthropic Claude Constitution (Jan 2026) | A detailed natural-language account of intended values, judgment, safety, and responses to value conflict; the constitution guides model training | A transparent values-and-judgment framework with explicit tradeoffs and an iterative stance | It is a normative constitution and training guide, not a general axiomatic derivation of ethics from physical law |
| Value Kaleidoscope / ValuePrism (AAAI 2024) | Structured extraction and assessment of contextualized values, rights, duties, and disagreement; ValuePrism reports 218k value records linked to 31k situations | Plural human values and their context can be represented explicitly rather than reduced to one label | Representing human judgments is distinct from establishing which judgments are justified |
| Academic artificial-moral-agent systems | The 2026 Krah–Tröschel survey reports a trend toward configurable ethical theories, especially consequentialist and hybrid/configurable approaches | Configurability is an active research direction, not a wholly new idea | The survey highlights weak benchmarking, unclear authority over profile selection, limited handling of theory conflict, and low involvement of affected people |
| Symthaea (public repository inspected) | A Rust cognitive architecture with a unified ethics pipeline, MoralParser/MoralAlgebra, UnifiedValueEvaluator, Eight Harmonies integration, moral-topology components, contextual harmony weights, and institutional-compliance checks | There is an existing substrate on which to build an explicit, testable plural-ethics and theory-evaluation program | Existing modules and tests do not by themselves establish superior moral reasoning, complete support for distinct normative theories, or successful first-principles derivation |

### Honest competitive assessment

Symthaea is **not presently demonstrated to outperform frontier AI systems on general reasoning or ethics**. Its public documentation reports 56.2% overall on its corrected Hendrycks Ethics run, with only the virtue category described as meaningfully above chance, and retracts an earlier 91.1% headline as leakage-inflated. That evidence cannot support a claim of industry-leading ethical reasoning.

Nor should we claim that Symthaea has already solved configurable ethics: its current components provide a starting point, but a framework registry, faithful per-framework semantics, conflict-preserving comparison, and independent qualification remain proposed work.

A plausible differentiator is not an unsupported claim of being “more ethical.” It is a narrower, potentially valuable research contribution: **an open, machine-checkable pipeline that makes ethical premises explicit, derives conditional consequences, exposes the natural-fact/normative-premise boundary, compares alternative axiom sets, and records reproducible evidence for each conclusion**.

Industry sources and references:
- [OpenAI: Our approach to the Model Spec (2026)](https://openai.com/index/our-approach-to-the-model-spec/)
- [Anthropic: Claude's Constitution (2026)](https://www.anthropic.com/constitution)
- [Sorensen et al.: Value Kaleidoscope (AAAI 2024)](https://doi.org/10.1609/aaai.v38i18.29970)
- [Krah & Tröschel: Trends and challenges in machine ethics (2026)](https://doi.org/10.1007/s43681-025-00971-7)
- [Symthaea public repository and evidence table](https://github.com/Luminous-Dynamics/symthaea)

## 2. Can ethics be derived from first principles like thermodynamics?

### The useful analogy

Thermodynamics connects macroscopic regularities to physical state, constraints, and statistical descriptions of microscopic systems. Statistical mechanics gives explanations of entropy and the second law under specified assumptions about physical states, probability, coarse-graining, and initial conditions. The foundation and interpretation of the second law are themselves subtle; it should not be described as a simple theorem from bare logic with no assumptions.

Ethics could benefit from a similar *method*: specify state variables, mechanisms, constraints, foundational premises, and derivation rules; derive consequences; and compare the predictions to real observations.

But there is a decisive difference. Thermodynamic equations describe what physical systems do under specified conditions. A moral judgment concerns what should matter, what reasons count, or what an agent ought to do. Descriptive facts about suffering, cooperation, scarcity, cognition, or social stability do not by themselves select a normative objective.

This is the family of issues raised by Hume's is–ought problem. Moral naturalism is a serious philosophical position that attempts to understand moral facts in naturalistic terms, but it is contested; it is a research hypothesis, not an established implication of physics.

References:
- [Stanford Encyclopedia of Philosophy: Moral Naturalism](https://plato.stanford.edu/entries/naturalism-moral/)
- [Stanford Encyclopedia of Philosophy: Hume's Moral Philosophy](https://plato.stanford.edu/entries/hume-moral/)
- [Stanford Encyclopedia of Philosophy: Boltzmann's Work in Statistical Physics](https://plato.stanford.edu/entries/statphys-boltzmann/)

### A formal shape for a derivation

Let:

- (F) be empirical facts and a versioned causal model;
- (A) be an explicit set of normative axioms or value premises;
- (C) be legal, safety, consent, and authorization constraints relevant to the context;
- (D) be a declared set of valid inference and decision rules;
- (N) be a candidate norm or action conclusion.

Then the system should attempt to establish a conditional relation:

[
(F, A, C, D) \vdash N
]

This notation means: given these facts, value premises, constraints, and inference rules, conclusion (N) is supported. It does **not** mean that empirical facts alone logically entail (N), nor that the chosen premises have been proven morally correct.

The engine should output the proof or argument trace, factual dependencies, assumptions, uncertainty, counterexamples, and other plausible premise sets under which (N) would change. If the derivation is incomplete, inconsistent, or depends on unsupported empirical assumptions, mark it accordingly.

## 3. Candidate starting points to test — not axioms to smuggle in as facts

The project should study several premise families without treating any one as settled or universally binding.

### A. Moral-patient relevance

Candidate premise: states of a being that matter to that being — for example, experienced pain or well-being if it is sentient — can count as morally relevant.

Empirical work may help identify capabilities and interests. Whether, how, and how much those interests should matter is a normative question. Symthaea must not presume that its own consciousness status is settled.

### B. Consistency and relevant similarity

Candidate premise: if two situations have the same morally relevant features, a judgment should not change merely because of an irrelevant label.

This supports counterfactual tests and some forms of impartiality, but it does not by itself tell us which features are morally relevant or require identical treatment in genuinely different situations.

### C. Agency, consent, and justification

Candidate premise: overriding another agent's expressed agency or consent creates a burden of justification.

This can be formalized in terms of authorization, causal effects, rights claims, and potential coercion. The degree and scope of the obligation still need philosophical argument and context-specific specification.

### D. Harm, benefit, and distribution

Candidate premise: consequences for affected parties matter, and the distribution of burdens cannot always be reduced to a single aggregate total.

Causal modeling can estimate consequences; moral theory must specify how different parties' interests, uncertainty, rights, and unequal burdens are to be considered.

### E. Reciprocal or public justification

Candidate premise: rules that coercively govern multiple agents should be justifiable to those affected using reasons that do not simply privilege the rule-maker.

This is a candidate bridge from agency and interdependence to political morality, but its grounding and scope are disputed and should be evaluated against rival theories.

### F. Epistemic responsibility

Candidate premise: confidence should not exceed evidence, and severe irreversible harms under uncertainty merit additional caution.

Evidence calibration is empirically testable. How much precaution is warranted is itself a value and decision-theoretic commitment and must be declared.

These are **candidate foundations**, not a completed universal ethical system. A rigorous program must compare them with alternatives, examine their tensions, and report which conclusions rely on which premises.

## 4. What empirical science can derive with confidence

Natural and social science can help Symthaea answer questions such as:

- Which actions cause which harms or benefits, under what conditions?
- What capabilities are required for an organism or artificial system to have particular interests or preferences?
- Which cooperative arrangements are stable under specified assumptions about repeated interactions, incentives, information, and institutions?
- Which procedures reduce domination, unfairness, error, or preventable suffering under a specified operational definition?
- What tradeoffs are unavoidable because of physical scarcity, uncertainty, incompatible goals, or competing rights?
- Which policy is robust across multiple plausible causal models?

Evolutionary game theory is useful for studying how cooperation and norms can arise in repeated interactions and how those outcomes depend on assumptions. But a strategy being evolutionarily stable or a convention being widespread does not establish that it is morally good. The same caution applies to thermodynamic metaphors: entropy production, equilibrium, coherence, survival, efficiency, and complexity are physical or mathematical concepts, not ready-made moral rankings.

Sources:
- [Stanford Encyclopedia of Philosophy: Game Theory and Ethics](https://plato.stanford.edu/entries/game-ethics/)
- [Stanford Encyclopedia of Philosophy: Evolutionary Game Theory](https://plato.stanford.edu/entries/game-evolutionary/)

## 5. Proposed Symthaea capability: a Moral Derivation Engine

Build this as a research component next to the plural-ethics evaluator, not as a replacement for it.

### Inputs

1. **Empirical model:** provenance-bearing observations, causal assumptions, distributions, uncertainty, and competing explanatory models.
2. **Normative premise set:** versioned assumptions, their rationale, scope, objections, dependencies, and affected-party relevance.
3. **Context and candidate actions:** intended outcomes, consent/agency facts, reversibility, affected parties, and decision authority.
4. **Inference regime:** formal rules, decision theory, constraint system, or framework-specific reasoning procedure.
5. **Independent operational boundary:** applicable authorization, privacy, safety, and legal controls. This boundary cannot be modified by moral-profile selection.

### Outputs

- A conditional conclusion with a trace of premises and inferences.
- A strict distinction between observed facts, modeled consequences, normative premises, and derived judgments.
- Sensitivity analysis: which changes in factual assumptions or values would change the conclusion?
- Alternative conclusions under other plausible axiom sets.
- Contradictions, underdetermined steps, missing facts, and counterexamples.
- A robust-action set where candidate actions remain acceptable across the selected models, or an explicit report that no such action was established.
- A versioned reproducibility receipt linking the premise-set digest, causal model, inference rules, test corpus, code revision, and output.

The engine must not use an opaque scalar such as “moral free energy” or a high HDC similarity as proof that a conclusion is ethically correct. Those may be research indicators or search aids, but the inference and its limits must be independently inspectable.

## 6. Research program and falsifiable milestones

### Phase 0 — Literature and claim ledger

Inventory foundational ethical theories and their actual formal commitments. Build a claim ledger separating:
- empirical claims;
- philosophical arguments;
- formal theorems under assumptions;
- design choices adopted by Luminous Dynamics;
- speculative hypotheses.

Do not label a candidate axiom “derived” merely because it is encoded in Rust or passes unit tests.

### Phase 1 — Toy worlds with known ground truth

Create small deterministic worlds containing agents with explicit preferences, capabilities, resource constraints, and interaction rules. Compare conclusions from several normative premise sets, then perturb one assumption at a time.

These worlds test internal consistency and derivation correctness; they do not establish universal moral truth.

### Phase 2 — Formal derivations and countermodels

Use a proof checker or independently implemented reference evaluator where the logic supports it. For each proposed normative theorem, actively search for countermodels. For each claimed implication, attempt to remove a premise and see whether the conclusion still follows.

A claim fails if the evaluator cannot reproduce the result, if a conclusion depends on a hidden premise, or if a countermodel satisfies the premises while violating the claimed conclusion.

### Phase 3 — Empirical causal grounding

Evaluate the factual component with held-out observations, sensitivity to confounding, calibrated uncertainty, and alternative causal models. Keep the normative assumptions fixed while testing the causal model, then vary the premises separately. This is essential to discover whether a conclusion follows from evidence or merely from a value choice.

### Phase 4 — Pluralistic comparison

Run the Moral Derivation Engine alongside the MORAL-PLURALISM evaluator. Identify:
- conclusions shared by multiple premise sets;
- conclusions that differ only under specific assumptions;
- genuine conflicts;
- cases where evidence is too weak;
- cases where a proposed moral question is not decidable by the chosen premises.

Agreement between frameworks is interesting evidence of robustness under those frameworks; it is not proof that the shared conclusion is objectively true.

### Phase 5 — Open, adversarial evaluation

Publish specifications, toy-world generators, premise-set definitions, test vectors, held-out protocols, failures, and reproducibility instructions. Invite philosophical, scientific, legal, cross-cultural, and affected-party critique before describing a result as independently qualified.

## 7. Acceptance criteria

Do not qualify the program until:

- Each reported norm states the premises and empirical assumptions needed for its derivation.
- A reference checker reproduces supported formal derivations.
- Negative tests show that removing a necessary premise can make a conclusion underivable.
- Countermodels and contradictory premise sets produce explicit failure or inconsistency results rather than arbitrary confidence.
- Perturbation tests distinguish changes caused by factual evidence from changes caused by normative assumptions.
- Alternative premise sets can yield different conclusions where the problem is genuinely underdetermined.
- A shared conclusion is not advertised as universal morality solely because several frameworks agree.
- No derived conclusion can independently authorize an action or bypass safety, consent, privacy, or access controls.
- Results are reproducible at an exact commit with preserved evidence and retractions.

## 8. Suggested priority

1. Reconcile the current moral-benchmark evidence discrepancy tracked in [issue #7254](https://github.com/Luminous-Dynamics/symthaea/issues/7254).
2. Complete the multi-framework read-only evaluator.
3. Add the Moral Derivation Engine to test toy models and premise sets.
4. Prove narrow conditional results before broad moral claims.
5. Expand to empirical and contested cases only after the formal and evidential boundaries behave correctly.

## Bottom line

Thermodynamics is a useful analogy for rigorous modeling, not a shortcut around normative philosophy. Symthaea should try to discover **which moral conclusions follow from which foundational premises and empirical realities**, where different derivations converge, and where no conclusion follows without making a further value commitment. That is a scientifically tractable and philosophically honest direction—and potentially a meaningful contribution beyond simply adopting another constitution or ethical checklist.
