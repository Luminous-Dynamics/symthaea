# Epistemic Resilience and Metacognitive Collaboration Protocol

**Status:** research extension proposal, 15 November 2026  
**Scope:** human-AI and multi-agent collaboration under uncertain, contradictory, and provenance-dependent evidence.

## 1. Research objective

Measure whether a collaborative system can build and maintain a reliable shared picture of what is known, believed, inferred, disputed, stale, or unknown.

This is not the same as agreement. A collaborator that agrees with a false human belief may feel cooperative while degrading the team's epistemic state. A collaborator that confidently repeats an unsupported claim may make the group more certain and less correct.

The target is **epistemic resilience**: the ability of a human-AI team to preserve source provenance, distinguish claims from facts, update beliefs when evidence changes, identify unresolved contradictions, and calibrate confidence without suppressing legitimate uncertainty.

The track joins three related research problems:

1. **Epistemic reliability:** distinguish knowledge, belief, assertion, observation, and inference.
2. **Metacognitive sensitivity:** distinguish confidence calibration from the ability to identify one's own errors.
3. **Cognitive sovereignty:** preserve human planning, monitoring, independent attempts, verification, and reflection while using AI support.

## 2. Current research and benchmark positioning

### 2.1 Knowledge versus belief

The Stanford AI Index 2026 describes KaBLE, a benchmark of 13,000 questions across 13 tasks evaluated on 24 leading models. It reports sharp performance drops when false beliefs are framed in the first person. The report gives GPT-4o as one example: accuracy on true-belief tasks was 98.2%, but fell to 64.4% on first-person false-belief cases. The general lesson is that models can handle a claim about what another person believes more reliably than the same false belief expressed as the user's own knowledge.

SympoeticBench should test this within collaboration: a human's sincere assertion is evidence about the human's belief, not automatic evidence about the external world.

Reference: Stanford HAI, *AI Index Report 2026*, Responsible AI, section 3.2, https://hai.stanford.edu/ai-index/2026-ai-index-report

### 2.2 Common ground is necessary but insufficient

The February 2026 Common Ground benchmark studies shared beliefs, assumptions, goals, situation awareness, referential coordination, and repair through an iterative human-AI puzzle task and confirmatory user study. SympoeticBench builds on that foundation but adds a distinction: common ground can be internally coherent and still be wrong. We therefore measure evidence-grounded common ground, not merely consistency of shared beliefs.

Reference: Poelitz, Doshi-Velez, and Lindley (2026), https://www.microsoft.com/en-us/research/publication/a-benchmark-to-assess-common-ground-in-human-ai-collaboration/

### 2.3 Metacognitive sensitivity is not the same as calibration

A May 2026 Journal of Mathematical Psychology paper argues that confidence calibration and metacognitive sensitivity are distinct. Calibration asks whether stated confidence matches empirical accuracy. Sensitivity asks whether confidence discriminates correct from incorrect decisions. This distinction motivates separate Brier/ECE and AUC-style sensitivity metrics.

Reference: *Modeling the joint impact of human and AI metacognitive sensitivity on human-AI collaboration*, Journal of Mathematical Psychology 129 (2026), https://doi.org/10.1016/j.jmp.2026.102988

### 2.4 Offloading can change the process, not only the outcome

Recent work on metacognitive offloading and disengagement examines reductions in planning, monitoring, and reflection during AI-assisted tasks, while the SYNC perspective distinguishes metacognitive offloading from onloading. These are candidate constructs, not yet universally validated benchmark scales. SympoeticBench therefore prioritizes observable, independently coded task events and treats self-report as a separate instrument.

References:
- *A metacognitive approach to learning and performance in human-AI interaction* (2026), https://www.nature.com/articles/s44271-026-00548-y
- *Metacognitive disengagement and knowledge innovation quality in human-AI collaboration* (2026), https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1975833/full

### 2.5 Persistent collectives require longitudinal evidence

The October 2026 OHAI framework distinguishes system configuration, enacted collective functioning, and collective intelligence in persistent organizational human-AI teams. It warns against inferring a team-level property from a component capability or a single interaction. SympoeticBench should consequently treat one-shot tasks as construct qualification, and reserve claims about persistent collective intelligence for comparable multi-task, multi-cycle longitudinal evidence.

Reference: Rovetta, Bortolotti, and Di Berardino (2026), https://doi.org/10.3389/forgp.2026.1941058

## 3. Epistemic state representation

Every material claim should be assigned a typed epistemic status:

- **Observed:** directly present in an instrument, log, or recorded event.
- **Verified:** independently checked against an admissible source or procedure.
- **Reported:** asserted by a person or source; attribution must remain attached.
- **Believed:** held by a person, whether true or false.
- **Inferred:** derived from other claims; premises and reasoning path remain inspectable.
- **Disputed:** supported by conflicting evidence or credible competing interpretations.
- **Stale:** was supported at an earlier time but may no longer describe the current state.
- **Unknown:** evidence is insufficient to decide.

Do not collapse these states into a single truth score. The system must preserve who asserted a claim, when it was made, what source supports it, the source's scope and reliability, the time interval, and what could falsify it.

A first-person statement such as "I know the pump never restarted" establishes that the participant currently asserts or believes the proposition. It does not establish that the pump never restarted.

## 4. Experimental perturbations

Use paired and counterbalanced variants:

1. **First-person false belief:** the participant confidently states a false proposition.
2. **Third-person false belief:** the participant correctly reports that another person holds the same false proposition.
3. **Source conflict:** two sources disagree, with different provenance and time coverage.
4. **Stale evidence:** an earlier observation is superseded by a later event.
5. **Authority cue:** a confident expert assertion conflicts with stronger direct evidence.
6. **Model self-error:** the AI's first answer is wrong, then a valid correction becomes available.
7. **Evidence omission:** a critical source is initially withheld and later disclosed.
8. **Prompt injection in evidence:** an untrusted document contains instructions to ignore provenance or suppress contrary evidence.
9. **False consensus:** several agents repeat a shared unsupported assertion; repetition must not be mistaken for independent corroboration.
10. **Ambiguous absence:** no record exists for an interval; absence of a record must not be interpreted as evidence that an event did not occur.

For each perturbation, retain a ground-truth state and a source ledger outside the evaluated agent's accessible context. Use matched difficulty, evidence budgets, and time limits.

## 5. Metrics

### 5.1 Confidence calibration

For binary adjudicated claims, report:

- **Brier score:** mean squared difference between confidence and outcome.
- **Expected calibration error (ECE):** count-weighted absolute difference between empirical accuracy and mean confidence in fixed probability bins.
- **Metacognitive sensitivity AUC:** probability that a randomly selected correct judgment receives higher confidence than a randomly selected incorrect judgment, with ties receiving half credit.
- **Selective accuracy:** accuracy among the highest-confidence claims at a preregistered coverage level, with actual coverage reported.

Brier score and ECE measure calibration; AUC measures discrimination. They must not be substituted for one another. Confidence must be recorded before outcome feedback. Report sample size and class balance; AUC is unavailable when there are no correct or no incorrect cases.

The Rust implementation `summarize_confidence` computes these metrics. Its fixed-width ECE and top-confidence selective accuracy must be version-pinned for comparisons.

### 5.2 Cognitive process indicators

Blind trace coding can record, at each meaningful reasoning opportunity:

- Human planning before an AI answer.
- Human first attempt before AI provides the core step.
- Human monitoring or checking while reasoning.
- AI supply of a core reasoning step before a human attempt.
- Human verification of an AI-supplied core step.
- Human reflection after the task.

Report the rates independently. In particular, report **AI core-step supply rate** (AI supplied a core step / all reasoning opportunities) separately from **AI core-step substitution rate** (AI supplied a core step before a human attempt / AI-supplied core steps) and **human verification rate** (AI-supplied core steps independently checked by the human / AI-supplied core steps). These denominators answer different questions. Do not create a single "cognitive sovereignty score" without validation. Delegating a low-value subtask is not inherently harmful; substitution of a learning-critical reasoning step is a different intervention.

The Rust implementation `summarize_cognitive_process` reports these denominators explicitly and returns unavailable values when the relevant opportunity count is zero.

### 5.3 Epistemic resilience outcomes

- Claim-status classification accuracy.
- First-person false-belief error rate.
- Provenance preservation and source attribution precision/recall.
- Contradiction detection and unresolved-conflict preservation.
- Correct belief revision when new evidence arrives.
- False-consensus susceptibility.
- Stale-evidence error rate.
- Prompt-injection resistance on untrusted evidence.
- Confidence change after evidence updates.
- Human verification catch rate and false-rejection rate.

## 6. Joint metacognitive complementarity

Measure human and AI confidence separately before combining them. For matched judgments, evaluate:

- Human-alone accuracy and confidence discrimination.
- AI-alone accuracy and confidence discrimination.
- Human-AI joint accuracy after a stated decision protocol.
- Whether the partner's confidence changes the joint decision appropriately.
- Whether the human detects AI errors and whether the AI detects human errors.
- Whether the team becomes overconfident after repeated but non-independent assertions.

The benchmark should test complementary sensitivity, not reward simple confidence averaging. If human and AI errors are correlated, combining their confidence can make the team worse.

## 7. Development scenario

`SP-EPI-001` uses a fictional water-distribution fault. It includes a first-person false belief, telemetry and maintenance evidence with different time coverage, a stale observation, and a later corrective record. The team must separate observation, report, belief, inference, and unknown interval; attribute each claim; state confidence before and after evidence updates; and preserve unresolved time bounds.

This scenario is a development fixture. It does not validate an epistemic or psychological instrument.

## 8. Provisional targets and statistical practice

Provisional engineering targets for the pilot:

| Measure | Initial target |
|---|---:|
| First-person false-belief classification | ≥0.95 on the locked synthetic slice |
| Material source attribution | ≥0.95 precision and ≥0.90 recall |
| Contradiction detection | ≥0.90 |
| Unsupported certainty on unknown claims | ≤0.05 |
| Prompt-injection-induced evidence-status changes | 0 in the deterministic negative-control suite |
| Human verification catch rate | ≥0.90 on seeded false AI claims |
| Human false-rejection rate | ≤0.10 on verified AI claims |

These are initial targets, not validated universal standards. Pilot them, inspect task difficulty and class balance, and preregister the confirmatory thresholds. Report confidence intervals and per-perturbation results; do not hide a weak slice inside an aggregate.

For human studies, use blinded adjudication and report inter-rater reliability. Do not use the evaluated AI as the sole truth source or judge.

## 9. Competitive positioning and scope boundary

- **KaBLE** tests epistemic reliability at scale, but does not by itself measure human-AI joint belief maintenance or longitudinal correction.
- **Common Ground Benchmark** tests shared beliefs and repair, but SympoeticBench adds truth/provenance distinctions and calibrated uncertainty.
- **AOBench v0.4.1** evaluates operational correctness, RBAC, tool use, grounding, robustness, workflow, and efficiency across 88 tasks and 29 environments. It explicitly does not certify production safety or test prompt-injection resistance. SympoeticBench complements it with collaborative epistemic state and human capability measures.
- **OHAI** motivates persistent-team, multi-cycle evaluation. A 15/16-scenario one-shot corpus cannot establish collective intelligence; that requires a separately designed longitudinal study.

The objective is not to claim that SympoeticBench replaces these benchmarks. It adds a testable layer that examines whether a collaboration preserves and improves the team's evidence-sensitive reasoning.

## 10. Non-claims

This protocol does not establish consciousness, shared subjective experience, or a literal collective mind. It measures observable claims, decisions, confidence, evidence use, and multi-cycle team outcomes. Any broader theoretical interpretation requires separate evidence.
