# HABITAT-ASSUMPTION-FALSIFICATION-001 — test the assumptions before paying for the factory

**Parent:** [HABITAT-VERTICAL-001 #8051](https://github.com/Luminous-Dynamics/symthaea/issues/8051)  
**Resilience-option benchmark:** [HABITAT-RESILIENCE-OPTIONS-001](./HABITAT_RESILIENCE_OPTIONS_001_OPTION_VALUE.md)  
**Machine-readable experiment definitions:** [habitat-assumption-falsification-v0.1.json](./fixtures/habitat-assumption-falsification-v0.1.json)  
**Shared assumption-falsification proposal:** [MANUFACTURING_ASSUMPTION_FALSIFICATION.md on PR #8080](https://github.com/Luminous-Dynamics/symthaea/blob/5a9a55066b961bf2b6239d6ba69518276d53657c/docs/architecture/MANUFACTURING_ASSUMPTION_FALSIFICATION.md)  
**Shared decision-value benchmark:** [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803)

## Paradigm shift: don't optimize a plan whose deciding assumptions are still guesses

The previous slices freeze disruptions and enumerate resilience options. The next step is to ask which assumptions make each option worth considering, then design a safe observation that could disprove those assumptions before a capital or production decision.

This is the transition from “the design looks plausible” to an auditable engineering investigation:

`claim -> dependency -> falsifier -> discriminating test -> admitted observation -> changed decision or explicit unresolved state`

The shared Symthaea assumption-falsification proposal (#8080) already defines typed assumptions, competing hypotheses, test candidates, observations, decision receipts and dependency invalidation. This factory profile is a domain-specific consumer, not a second general-purpose falsification engine or experimental selector.

## What is now represented

The fixture declares nine assumptions and ten candidate experiment definitions covering the dependencies that currently prevent a real resilience decision:

- qualified alternative material supply and actual delivery windows;
- alternate workcell capability and availability;
- utility, inspection and restart common modes;
- worker skills and shift coverage;
- product-variant compatibility and setup benefit;
- repair-spare identity, availability, repair capability and acceptance;
- the distinction between synthetic challenge bounds and physical predictions;
- complete and comparable delivered-cost accounting; and
- a preregistered fair baseline/candidate comparison.

Every assumption has scope, status, consequence if wrong, a falsifier and links to relevant options/scenarios. Tests range from existing-record audits and protocol review to a controlled incoming-material test, variant sample assembly and bounded repair trial. The fixture deliberately records execution status as not executed, experiment selection as not selected, measured costs/time as not measured, and outcome probabilities as not supplied.

The documentary-first ordering is a *sequence of evidence classes*, not a claim that a specific test is already the cheapest or highest-value choice. Relative cost/time and ability to discriminate must be admitted before a test selector can legitimately rank candidates.

## Why this is materially better than a generic risk register

A conventional risk register can list “supplier delay” or “machine failure” without stating which claim an investment relies on, what result would contradict it, or which downstream decision must be reopened. This profile binds a specific falsifier and follow-up test to each claim, making it possible to answer:

1. Which unknown assumptions are necessary for the proposed option to work?
2. What evidence already exists, and what remains missing or stale?
3. What observation distinguishes the “works” and “does not work” hypotheses?
4. What qualifies the test itself—instrument/calibration, sample, worker, site and acceptance method?
5. Which decision receipts become invalid if the assumption or its revision changes?
6. What can be learned safely and cheaply before a physical trial or procurement commitment?

The objective is not to maximize the count of assumptions marked supported. A failed or inconclusive test is useful if it prevents an unsupported decision or reveals a new dependency.

## Candidate test ladder

Use this progression where appropriate, subject to the discriminating power of each step:

1. **Existing-evidence audit:** review exact source identity, revisions, grade/capability qualification, availability, calibration freshness and contradictions.
2. **Analytical/protocol review:** check dependency closure, load boundaries, BOM compatibility, cost denominators and accepted service definitions.
3. **Offline or shadow evaluation:** compare frozen plans without issuing machine commands; simulated feasibility remains simulation evidence.
4. **Controlled low-risk bench test:** only when the relevant qualified owner approves sample, instrumentation, procedure, acceptance thresholds and stop criteria.
5. **Bounded prospective pilot:** only after the previous evidence is adequate and required professional/organizational approvals exist.

The steps are not universal. A cheap document review that cannot distinguish two competing hypotheses is not the correct test. A high-information physical test that lacks safety qualification is not an eligible candidate.

## Decision and uncertainty semantics

- Unknown is not zero; missing evidence stays explicit.
- An interval is not a probability distribution.
- No expected-value or probability-of-failure ranking is allowed without an admitted, appropriately calibrated model.
- Cost/time of an experiment remains unknown until sourced; do not manufacture numeric values to make a selector run.
- Predicted outcomes and acceptance thresholds must be frozen before seeing the observation.
- Missing, failed, censored, inconclusive and invalid-measurement outcomes remain in the record.
- A hash identifies exact bytes; it does not make a claim true or independently authenticate a physical observation.
- Only the accountable qualified owner can authorize tests or physical execution. This benchmark never actuates equipment.

If the measurements cannot distinguish the options, the valid result is “insufficient decision evidence,” not an arbitrary winner.

## Proposed benchmarks

These are protocol acceptance targets, not measured outcomes:

- 100% of critical assumptions have a scope, source/status, explicit falsifier, consequence-if-wrong and dependent decisions/options before the decision review.
- Zero missing/unknown critical values silently coerced to zero or success.
- Zero false contradictions when admitted measurement uncertainty overlaps the predicted interval.
- Zero missed contradictions in frozen controls where a valid observation interval is disjoint from the declared predicted interval.
- 100% of experiment candidates have explicit intervention, controlled variables, required qualifications, measurement method, predicted discriminators, failure/indeterminate outcome and authority boundary before execution.
- 100% of dependent decision receipts are re-evaluated if a critical assumption revision changes.
- Compare decision-aware selection with fixed-sequence, random and uncertainty-only baselines only when their input models and evaluation budget are admitted. Use the existing #7803 contract; do not create another selector.
- Keep the existing #8080 proposal's target of at least 30% fewer experiments than random on a pinned synthetic oracle as a future hypothesis, not a present result. Use at least 30 seeds per stochastic case if the selector/policies are stochastic, and do not apply a statistical claim to a deterministic single run.
- At each stage, separately report failed/inconclusive experiments, false-feasible decisions, hard-gate violations, evaluation cost/time, and decisions changed. No composite score may offset a hard-gate failure.

## Interfaces and ownership

- [MANUFACTURING_ASSUMPTION_FALSIFICATION.md / PR #8080](https://github.com/Luminous-Dynamics/symthaea/pull/8080) is the reusable assumption/observation/receipt architecture.
- [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803) owns decision-value experiment-selection comparison.
- [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608) owns fair production-value baseline/candidate comparison.
- [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677) owns deterministic capability-constrained scheduling.
- [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) owns manufacturing-process semantics; Mycelix #3084/#3085 own relevant provenance and distributed process/coordination records.
- Robust-frontier and probability-ambiguity analysis remain owned by existing SE-OPT / ICCB work. Do not add a second optimizer or probability engine.

## Current limits

This is a synthetic test plan, not an executed investigation. There are no admitted supplier or resource qualifications, utility load measurements, worker rosters, variant changeover studies, repair trials, historical failure distributions, or real price/lead-time records attached to these assumptions. No experiment has been selected. The exact API and revision integration with the shared #8080/#7803 work remains to be inspected before implementation is claimed.

## External grounding

NIST's 2026 AI/ML roadmap for smart manufacturing discusses industrial-data complexity, cross-system integration, trustworthy operation and supply-chain/logistics optimization. NIST's advanced-manufacturing digital-twin program emphasizes validation, uncertainty and interoperability. NIST's September 2026 circular-manufacturing paper identifies systems-level modeling, circular design and digital threads as research needs. These references support the method's evidence and lifecycle focus; they do not validate the project data or any proposed option:

- https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing
- https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing
- https://www.nist.gov/publications/manufacturing-circular-economy-research-needs-design-systems-modeling-and-digital

## Claim ceiling

This profile establishes a structured set of assumptions and candidate tests only. It does not establish that any assumption is true or false, identify a cost-minimal experiment, qualify any equipment/material/process/person/site, prove a factory's safety or resilience, or authorize procurement or physical operation.
