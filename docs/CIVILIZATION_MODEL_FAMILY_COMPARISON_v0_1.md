# Civilization Model-Family Comparison v0.1

**Status:** research design; not an implemented or validated simulator  
**Date:** 2026-10-11  
**Proposed pilot:** reliable, affordable essential-energy access under physical and institutional constraints  
**Related work:** Symthaea issues [#7188](https://github.com/Luminous-Dynamics/symthaea/issues/7188), [#7190](https://github.com/Luminous-Dynamics/symthaea/issues/7190), [#7193](https://github.com/Luminous-Dynamics/symthaea/issues/7193), [#7201](https://github.com/Luminous-Dynamics/symthaea/issues/7201), [#7203](https://github.com/Luminous-Dynamics/symthaea/issues/7203), [#7205](https://github.com/Luminous-Dynamics/symthaea/issues/7205); Sol-Atlas [#79](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/79), [#82](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/82), [#83](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/83), [#86](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/86), [#87](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/87).

## 1. Purpose and non-goals

Design a reproducible experiment that compares alternative decision mechanisms inside a shared, multiscale human–infrastructure–ecology model. The purpose is to learn which mechanisms explain or predict which outcomes under declared conditions, and where evidence is insufficient.

This is **not** a claim that one theory explains humanity, that the current projects can predict civilization, or that a simulation can authorize policy. Passing a software test establishes only the property tested under the tested fixture.

The first task is deliberately bounded: explore how household and institutional decisions interact with energy affordability, essential-service continuity, infrastructure limits, distribution, and ecological constraints. The pilot is a testbed for architecture, not a proxy for all human life.

## 2. Theoretical contributions: hypotheses, not axioms

### Karl Friston: active inference

Implement active inference as one candidate policy family for agents that maintain beliefs about partially observed states and select actions using declared preferences and expected free-energy formulations. Specify the generative model, state spaces, likelihoods, priors/preferences, precision assumptions, learning rules and action-selection algorithm for each experiment.

Do not equate the Free Energy Principle with a validated social theory. A 2024 review finds promising behavioral fits but calls for direct comparison with competing accounts, including reinforcement-learning approaches. See [Hodson, Mehta & Smith (2024), *The empirical status of predictive coding and active inference*](https://doi.org/10.1016/j.neubiorev.2023.105473).

### Michael Levin: multiscale collective intelligence

Use Levin and McMillen's work to motivate *testable hypotheses* about interactions across levels: local components can coordinate; a higher-level organization can constrain component behavior; and collective capabilities can arise without requiring a single centralized controller. Define these mechanisms concretely for social systems rather than assuming a community, company or nation is literally an organism or possesses a unitary mind.

Reference: [McMillen & Levin (2024), *Collective intelligence: A unifying concept for integrating biology across scales and substrates*](https://doi.org/10.1038/s42003-024-06037-4).

### Ken Wilber: coverage checklist

Use the four AQAL quadrants—individual interior, individual exterior, collective interior and collective exterior—as a **coverage checklist** to prevent the model from omitting subjective experience, observable behavior, shared culture, or institutional/material structure. A quadrant is not itself a measurable variable or causal mechanism. Only operationalize constructs that have declared definitions and defensible measurements. Do not treat Integral developmental levels as ground truth, diagnose individuals from aggregate proxies, or derive rights and authority from a developmental score.

### Current research update (2026)

Two current papers refine the division of labor above:

- Friston et al., [*Active inference and artificial reasoning* (Nature Communications, 2026)](https://www.nature.com/articles/s41467-026-77209-5), studies selecting informative outcomes to discover rules in partially observed discrete models. It may inform an *epistemic-exploration* policy variant or experimental-design component. Its demonstration assumes a known generative-model family and a structured task; it is not evidence that active inference predicts social systems or human populations. Treat it as a candidate mechanism to test, not a universal cognition engine.
- Haase & Pokutta, [*Beyond static responses: multi-agent LLM systems as a new paradigm for social science research* (Humanities and Social Sciences Communications, 2026)](https://www.nature.com/articles/s41599-026-08832-2), reviews agentic LLM architectures for social-science research and highlights reproducibility, bias, ethical oversight and the lack of standardized evaluation. The proposed framework itself has not yet been systematically validated across diverse implementations. This reinforces the choice to keep LLM-driven simulated populations out of the primary model-family benchmark until behavioral fidelity and replication are demonstrated.

- Reissl et al., [*The DSK stock-flow consistent agent-based integrated assessment model* (Ecological Economics, 2025)](https://doi.org/10.1016/j.ecolecon.2025.108641), integrates an agent-based integrated-assessment model with explicit stock-flow-consistent accounting and makes code/input data available. This is a practical pattern for our physical/financial accounting backbone. Fevereiro et al.'s 2025 systematic review of macroeconomic circular-economy models reports that model assumptions materially influence outcomes and identifies gaps around regional distribution, rebound effects, social justice, and real-financial interactions ([DOI](https://doi.org/10.1016/j.ecolecon.2025.108669)). Our implication is to reconcile stocks and flows before adding cognitive sophistication, and to treat boundary assumptions as first-class scenario inputs.

- Chou et al., [*A systematic empirical comparison of active inference and reinforcement learning models in accounting for decision-making under uncertainty* (Journal of Mathematical Psychology, August 2026)](https://doi.org/10.1016/j.jmp.2026.103011), fit complexity-matched models to behavior from two samples. Model comparison favored active inference, but predictive accuracy was similar and external-validity tests did not consistently favor either framework. This is a strong argument for our design: compare the models on held-out outcomes, retain uncertainty about which is appropriate, and avoid equating model-selection preference with a large practical prediction advantage.
- A 2025 replication study using ODD and NetLogo reproduced 15 of 18 agent-based models across disciplines and emphasizes explicit process scheduling, state, inputs and submodels ([Ecological Modelling](https://doi.org/10.1016/j.ecolmodel.2024.110967)). For the Humanity Atlas, we should make the run contract easy to independently reimplement at least for small reference scenarios, even if the full-scale engine stays Rust-only.

Practical consequence: separate **belief/model learning**, **action/policy selection**, **social interaction**, and **institutional rules** into independently switchable mechanisms. A model may use active inference for epistemic exploration while a simpler policy governs allocation; the experiment must be able to compare these components without conflating them.

### Additional modeling traditions

Use agent-based modeling to represent heterogeneity and interaction; stock-flow consistent/system-dynamics models for accumulation, depletion and delays; network science for topology and dependencies; institutional/social-ecological frameworks for rules and shared resources; and causal inference for separating explanation from prediction. These methods answer different questions and should be coupled only where a documented mechanism connects them.

For reproducible model description and replication, use the ODD protocol as a reference: [Grimm et al. and ODD protocol resources](https://www.jasss.org/23/2/7.html). For the rationale for treating wellbeing as a first-class output in coupled environment–society–economy models, see [*Inclusion of wellbeing impacts of climate change in integrated environment–society–economy models* (The Lancet Planetary Health, 2025)](https://doi.org/10.1016/j.lanplh.2025.101375).

## 3. Division of responsibilities

These are proposed boundaries, not statements that all integrations already exist.

- **Symthaea:** candidate agent policies and cognitive-state models; explicit decision traces; model-family adapters; deterministic experiment orchestration where supported; ablations and evaluation.
- **Sol-Atlas:** spatial and time-indexed physical-world state, infrastructure and resource constraints, scenario visualization, and the already-proposed strict run-envelope consumer. It must preserve the evidence classification provided by the run contract.
- **Mycelix:** provenance, consent/permission and governance receipts, source-attributed claims, and bounded coordination between parties. A signed or content-addressed claim can prove binding to bytes or an identity; it cannot by itself prove factual truth or representativeness.
- **Independent evaluation:** a verifier/evaluator distinct from the component proposing a policy or reporting the result. Use an independent oracle for conservation, balance, and invariants where feasible.

Do not make the simulator, evidence repository, user interface and policy authority one trust domain. Preserve the boundary:

\`simulation output != recommendation != authorization != execution != observed outcome\`

Align producer/consumer schemas with the public Sol-Atlas Humanity Atlas work rather than creating a second incompatible envelope.

### Existing simulator audit — 2026-10-12

The repository already contains a multi-world civilization simulator; the implementation task must not create a competing simulator. A source audit found two layers of experimental-validity problems in `mycelix-multiworld-sim/src/bin/governance_comparison.rs` and its execution path:

1. The five-row comparator mapped `Sortition`, `Meritocratic` and `ElderCouncil` to the same `trust_weighted_governance = false` configuration as the equal-weight row. Those rows were not distinct interventions.
2. Even the remaining true/false comparison is **not a governance-policy comparison**. In the current `lib.rs` execution path, the flag changes a resource-waste assumption; in `consciousness.rs`, it changes the consciousness-growth amplifier. The governance tick receives `phase2_enabled`, not a selected `GovernanceModel`, and still computes profile/tier-based eligibility. Thus no governance policy arm is verified as wired end-to-end, and the flag's effects are confounded with non-governance model changes.

The corrective [PR #3765](https://github.com/Luminous-Dynamics/luminous-dynamics/pull/3765) now keeps `GovernanceModel::comparison_ready()` empty and makes the main comparator plus the legacy control, multi-seed and diagnostic executables exit with every governance arm marked `NOT RUN` until the actual tick loop consumes a policy selector. The same patch relabels the boolean in configuration/live metrics/ablation output as a legacy growth-resource bundle, removes the obsolete +6.9% assertion from the active default scenario, marks the historical gating report as superseded, and adds a visible source-audit correction to the planetary-civilization paper. These edits have been fetched back and checked at source level, but Rust compilation, tests and simulation runs have not been executed here; the PR remains draft with no available CI run in this session. Follow-up implementation requirements are tracked in [issue #3766](https://github.com/Luminous-Dynamics/luminous-dynamics/issues/3766).

The same draft also synchronizes the duplicate planetary-civilization paper and relabels the WASM demo's legacy toggle as `Legacy Growth/Resource Bundle (not governance)`. The old serialized config key remains accepted for backward compatibility; the new browser-facing alias uses the accurate bundle name.

This changes the implementation order: first add a typed, versioned selector to the actual governance transition and tests proving that each selected arm changes only its declared mechanism; then run a frozen multi-seed comparison. Do not cite any prior one-seed or five-row result as evidence that distinct governance policies were compared, or as evidence that consciousness metrics should determine political authority.

### Post-hoc scoring binaries audit — 2026-10-12

A second audit found a distinct but related interpretability problem in `scoring_model_comparison` and `multi_seed_scoring`. Each runner simulates one baseline trajectory per seed and only afterwards applies the built-in 4D, 8D and 3D scoring functions to the same final population. The selected scoring model does not alter that trajectory through the governance tick. Consequently, a run-level CVS is repeated across model rows by construction; identical CVS values are not evidence that three scoring policies produce identical outcomes, and per-model score/tier/vote-weight statistics are post-hoc profile measurements, not dynamic governance effects.

The correction now labels these binaries as post-hoc profile analyses, emits an explicit note that CVS is shared/repeated, and changes `FINDINGS.md` to preserve the historical numbers while withdrawing causal statements about governance quality, constitutional infrastructure, or which model performs better. These artifacts can still be useful to compare how candidate profiles classify a fixed population. To infer behavioral or civilizational effects, a selector must feed the candidate profile into the actual decision path and the downstream simulator must respond to that intervention.

Related implementation is tracked in [PR #3765](https://github.com/Luminous-Dynamics/luminous-dynamics/pull/3765). The source edits were fetched back and inspected, but have not been compiled or executed in this environment.


## 4. First pilot: essential-energy access

### Research question

Under the same physical supply, demand, infrastructure, information, and governance constraints, which decision mechanisms and institutional arrangements produce robust improvements in essential-energy service, affordability and resilience—and which trade-offs do they expose?

### Proposed model entities

1. **Households:** essential and discretionary demand, available budget, service priority, local observations, beliefs/expectations where applicable, and permitted actions.
2. **Producers/operators:** generation and storage capacity, operating and maintenance costs, repair backlog, fuel/material limits and reliability.
3. **Communities/institutions:** ownership and allocation rules, investment authority, affordability support, settlement or trade rules, and dispute/exit mechanisms.
4. **Physical resources and infrastructure:** generation, storage, grid links, fuel/material inventories, service queues, and explicitly declared environmental constraints.
5. **External environment:** exogenous weather, supply, demand and price shocks, with provenance and scenario-relative timestamps.

The public socioeconomic laboratory already describes a 100-agent / 20-project / 8-resource / 4-community profile. Reuse that profile only if its current producer and parameters are inspected and bound to the run. Do not silently claim that it represents a real population. For the energy pilot, identify exactly which of those entities map to the energy schema; record any extension as a versioned change.

### Common observation/action contract

Every compared agent policy must receive the same declared information set and have the same feasible-action contract for the target experiment. If the model family requires additional state (for example, a belief distribution), that state is declared as internal model state and must not leak information unavailable to other agents.

At minimum, record the policy's observations, available actions, chosen action, declared state/parameter/configuration identity, random seed, model version, scenario time, and the resulting transition. The run must distinguish an exogenous shock from an agent action and must preserve action failures and no-op decisions.

### Candidate model families

Run comparable baselines before adding model complexity:

- **B0 — deterministic rule baseline:** fixed priority and threshold policies. Establishes a transparent floor.
- **B1 — bounded-rational heuristic:** local information, limited planning horizon and explicit switching costs.
- **B2 — discrete-choice / utility-based policy:** declared preference weights and response stochasticity, fitted only from defensible data or clearly labeled synthetic parameters.
- **B3 — model-free reinforcement learning:** a reward-based comparison with declared reward, training distribution and held-out scenarios.
- **B4 — model-based / belief-state planning:** beliefs about uncertain supply or demand; planning under partial observability without requiring active-inference semantics.
- **B5 — active inference:** declared generative model, prior preferences, epistemic/extrinsic terms and policy-selection configuration.
- **B6 — network/institution ablations:** these are cross-cutting structural variants, not competing individual psychologies. Toggle network topology, local vs centralized allocation, and rule/authority boundaries while holding other conditions fixed.

Do not add an LLM-agent family to the primary benchmark by default. If later explored, isolate model/provider/prompt/version, sampling settings, tool affordances, memory, and stochasticity; avoid interpreting fluent explanations as evidence that simulated agents reproduce human cognition.

## 5. Shared outcomes: a vector, never a winner score

Report these separately, disaggregated by household/community and relevant exposure where data permits:

- essential demand served, unserved essential energy, outage duration and time to recovery;
- affordability burden, arrears/failed settlement and distribution of service access;
- generation/storage/infrastructure utilization, maintenance backlog and lifecycle cost;
- fuel/material use, emissions and explicitly modeled ecological-limit violations;
- concentration of productive assets, infrastructure chokepoints, decision authority and verification authority;
- feasible choices, switching costs, meaningful exit, and dependence on a single provider;
- model-calibration quality, predictive error, uncertainty, sensitivity and compute cost.

Do not use one scalar "humanity score." A model may optimize one metric by shifting harms to another household, community, future cohort, ecosystem, or unmodeled domain. Keep rights and essential-service floors as explicit constraints, not rewards that can be traded off against aggregate gains. Report both violations and uncertainty; never infer a right from a preference weight.

## 6. Experiment protocol

### E0 — specification freeze

Version and hash the model, schemas, parameter configuration, scenario, exogenous shock schedule, seed schedule, policy rules and evaluation code. Record data source, publication/update date, geographic and population coverage, licensing and known biases. Missing data remains missing; don't impute zero or a confident value without a declared method.

### E1 — mechanical invariants

Use deterministic hand-checked scenarios to independently verify, where applicable:

- energy/power balances and unit consistency;
- per-interval essential-load service and priority allocation;
- capacity, storage, stock and flow bounds;
- no creation of physical resources from monetary balances or reputation;
- transition timing, action scope and state continuity;
- deterministic replay for fixed configuration and seed (or a documented explanation for controlled nondeterminism).

Include adverse cases: insufficient storage, grid disconnection, demand surge, failed repairs, stale information, rejected actions and malformed evidence. A verifier must derive essential-load service from interval receipts, not infer it from aggregate energy balance.

### E2 — mechanism and ablation tests

Freeze a small scenario corpus with positive and negative controls. Toggle one mechanism at a time: belief updating, preference representation, planning horizon, social information, network centrality, maintenance, market/credit rule, and local governance. Confirm that output changes are consistent with the stated causal path; a sensitivity by itself is not proof of a mechanism.

### E3 — comparative model-family benchmark

Use the same exogenous trajectories, feasible actions, initial states and seed corpus across model variants as far as each formulation allows. Report any irreducible mismatches. Separate parameter tuning from evaluation. Compare:
- decision/log-likelihood metrics where behavioral observations exist;
- prospective predictive error for held-out transitions/outcomes;
- calibration of probability forecasts;
- constraint violations and resource/service accounting;
- Pareto trade-offs among the outcome vector;
- sensitivity to seeds, parameter uncertainty and model assumptions.

A baseline that equals or beats active inference on a task is a valid result. Do not alter benchmark definitions after inspecting test-set outcomes without versioning the new experiment.

### E4 — retrospective and transfer evaluation

Only when suitable, legally usable observations have been sourced: train/calibrate on a declared period and hold out a later period; separately report transfer to different regions or system conditions. Compare to simple statistical and domain baselines. Distinguish reproduction of a historical series from prediction of a causal intervention.

Do not call synthetic-fixture tests empirical validation. Do not label a model calibrated, or a run a prospective forecast, unless the existing run-envelope contract's evidence requirements are satisfied.

### E5 — intervention and policy analysis

Begin with sensitivity analysis and counterfactuals, not live action. Where causal identification is weak, present scenario comparisons as conditional ("under assumptions A–D") rather than estimates of real-world policy effects. Operational deployment would require separate authority, safety, privacy, legal and impact review as specified by the simulation-to-authority firewall.

## 7. Evidence classes and fail-closed reporting

Reuse the existing Humanity Atlas run-envelope classification and receipts. At minimum, the UI/reporting boundary must distinguish:

- structural/schema acceptance;
- deterministic/mechanical verification;
- synthetic mechanism test;
- retrospective evaluation;
- calibration supported by declared evaluation evidence;
- prospective forecast supported by a forecast receipt.

These are not interchangeable maturity badges. A hash establishes byte identity, not publisher identity or scientific validity; a parser validates structure, not truth; a review is not a holdout; and a queued CI job is not a pass.

If a required dataset is missing, stale, partial or rejected, report the exact state, affected metrics and downstream consequences. Never silently promote a partially supported global result to a full-population claim.

## 8. Privacy, governance and anti-Goodhart requirements

Do not build an individually identifying, continuously updated "humanity telemetry" store. Prefer public aggregate data, consented research datasets, local processing, minimization, privacy-preserving aggregation where appropriate, and explicit purpose limitation. Model synthetic agents unless real person-level data is necessary, lawfully obtained, and justified.

A modeled score must not grant rights, deny services, determine an individual's status, or authorize institutional action. Do not infer authority from reputation, cognition, developmental stage, engagement, contribution, inferred preferences or simulator confidence. Document who selects objectives, whose interests are absent, and who can contest data and conclusions.

Test gaming and displacement explicitly: output growth with service deprivation; average improvement with subgroup harm; maintenance deferral; boundary shifting; privacy loss; power concentration; and future resource depletion. Improvement in a target metric is not sufficient evidence of improvement in the system.

## 9. Exit criteria for v0.1

The design moves to an implemented synthetic prototype only after all of the following are available:

1. A versioned ODD-style model description and typed scenario contract.
2. At least two simple baselines plus an independently implemented mechanical oracle.
3. Frozen deterministic scenarios, seeds, expected invariants and mutation cases.
4. A declared outcome vector and explicit constraints.
5. A run envelope that binds source bytes, model/configuration, scenario, seed, time basis and evaluation receipts.
6. A report separating authored tests, executed local checks, CI outcomes, and real-data qualification.
7. An explicit list of claims the current evidence does **not** support.

The first acceptance is a reproducible synthetic research instrument. It is not "a validated model of humanity."

## 10. Reference and integration links

- Active inference evidence review: https://doi.org/10.1016/j.neubiorev.2023.105473
- Active inference vs reinforcement-learning comparison: https://arxiv.org/abs/1909.10863
- Levin & McMillen on collective intelligence: https://doi.org/10.1038/s42003-024-06037-4
- ODD protocol/model documentation: https://www.jasss.org/23/2/7.html
- Coupled environment–society–economy and wellbeing review (2025): https://doi.org/10.1016/j.lanplh.2025.101375
- Symthaea comparative socioeconomic lab: https://github.com/Luminous-Dynamics/symthaea/issues/7188
- Stock-flow consistency: https://github.com/Luminous-Dynamics/symthaea/issues/7193
- Power/agency/rights-floor stress lab: https://github.com/Luminous-Dynamics/symthaea/issues/7190
- Goodhart/adversarial metric lab: https://github.com/Luminous-Dynamics/symthaea/issues/7201
- Semantic boundaries: https://github.com/Luminous-Dynamics/symthaea/issues/7203
- Simulation-to-authority firewall: https://github.com/Luminous-Dynamics/symthaea/issues/7205
- Sol-Atlas run-envelope consumer and existing qualifications: https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/79
- Sol-Atlas evidence display and dataset-load health: https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/86 and https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/87

---

**Claim ceiling:** this document specifies a research design. It adds no implementation, test pass, empirical validation, policy recommendation, or forecast.
