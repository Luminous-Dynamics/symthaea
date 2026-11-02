# CivBench: intentional teaching under constrained bandwidth v0.1

**Status:** synthetic mechanism benchmark and experiment protocol.  
**Not a reproduction:** no human participant data or published model parameters are fitted here.  
**Core question:** which information should a knowledgeable agent deliberately teach to a particular learner, given limited bandwidth, uncertain evidence and competing goals?

## 1. Why this is a separate problem

Teacher selection asks **who should a learner learn from?** Intentional teaching asks **which parts of a teacher's knowledge should be transmitted, in what form, to which learner, and at what cost?** These are different decisions.

A teacher may possess many useful capabilities but have limited time, bandwidth and learner attention. Sending everything is not optimal. Popularity, claimed payoff, verified benefit, learner-context fit, uncertainty and lineage diversity can point to different items.

Recent empirical work motivates this separation:

- Mackintosh et al. (2026), *Intentional transmission of knowledge optimises cumulative cultural evolution*, Scientific Reports (19 September 2026), studies strategic selection of useful information for teaching. The paper reports that adults preferentially transmitted beneficial discoveries and that this selective transmission improved cultural accumulation relative to random information selection; the effects were not uniform across age groups. DOI: 10.1038/s41598-026-62734-6. Article: https://www.nature.com/articles/s41598-026-62734-6.
- Saral et al. (2026), *The roles of cultural transmission and causal reasoning in the cultural evolution of technology*, PNAS Nexus 5(4), pgag090, uses a four-slider optimization task with 12 discrete settings per slider and compares pure cultural transmission, causal explanation and technical representation. The reported results suggest that cumulative improvement can arise without explicit causal structure; causal reasoning can accelerate early improvement, but its marginal advantage changes over generations. DOI: 10.1093/pnasnexus/pgag090. Article: https://academic.oup.com/pnasnexus/article/5/4/pgag090/.
- Derex et al. (2024) show why social transmission needs matched individual-learning controls: in their combinatorial task, repeated individual learning outperformed chain/group conditions under the studied budgets. https://pmc.ncbi.nlm.nih.gov/articles/PMC11621820/

These findings are task- and population-specific. They motivate controlled hypotheses, not universal claims that intentional teaching always wins.

## 2. Model entities and information boundary

Each knowledge item contains:

- stable item ID and lineage ID;
- task family and declared learner-context fit;
- claimed/declared benefit;
- evidence reliability and uncertainty;
- intrinsic difficulty;
- learning-resource cost;
- evaluator-only true benefit, validity and held-out transferability.

The teacher-selection policy can inspect declared benefit, evidence reliability, uncertainty, learner-context fit and lineage representation. It **cannot** read true benefit, validity or held-out transferability. These are evaluator-only variables. This separation permits tests of overclaiming and noisy teacher metadata without oracle leakage.

The model represents the difference between:
- **resource cost:** how much of a fixed time/bandwidth budget the item consumes;
- **intrinsic difficulty:** how likely a learner is to acquire it after receiving it.

They must not be collapsed into one parameter.

## 3. Mechanisms compared

Given a fixed bandwidth budget, each teaching policy selects a packet of whole items whose total learning cost fits the budget:

1. **Random:** random legal packet selection.
2. **DeclaredBenefit:** rank by claimed benefit only. This is a deliberately weak, optimism-prone baseline.
3. **EvidenceCalibrated:** rank by declared benefit × evidence reliability × (1 − uncertainty).
4. **LearnerBenefit:** multiply the evidence-calibrated score by declared context-fit.
5. **DiversityAware:** add a bounded lineage-rarity bonus to evidence-calibrated, context-fit utility.

The diversity bonus is an experimental intervention, not a universal good. It can help preserve alternatives but can also spend scarce bandwidth on low-benefit novelties. Compare capability and lineage diversity on a Pareto frontier.

A sixth condition is **IndividualOnly**, with the same total learning-resource budget and no teacher, demonstration or social exposure. Its outcomes are not counted as teacher offers.

## 4. Learning stages and required metrics

For each item, record separately:

1. packet selection / individual attempt;
2. demonstration observation (social condition only);
3. acquisition;
4. integration into active repertoire;
5. correct execution against evaluator validity;
6. retention;
7. transfer to a held-out context.

Aggregate metrics include:
- number offered by teacher and number attempted individually;
- observation, acquisition and integration rates;
- correct execution rate and false-adoption rate;
- retention and held-out transfer;
- learning cost, unused budget and realized true benefit;
- unique lineages offered.

Do not use adoption rate as the primary objective. A strategy can be observed, acquired and integrated yet fail on execution. A useful strategy can also be acquired but not integrated, or integrated but later forgotten.

## 5. Initial experiment design

The Rust module `src/intentional_teaching.rs` implements the deterministic reference mechanics. The binary `intentional-teaching-sweep` will run 256 fixed seeds for six conditions:

- individual-only control;
- social random;
- social declared-benefit;
- social evidence-calibrated;
- social learner-context-aware;
- social diversity-aware.

All conditions use a declared budget of 8 abstract resource units. Each condition is evaluated against the same synthetic item bank and learner profile. The current item bank deliberately contains:
- high-claimed, low-reliability items;
- high-benefit, evidence-supported items;
- items with different intrinsic difficulty and resource cost;
- cross-context items;
- multiple lineages with unequal representation.

This is a mechanism-validation fixture. It is not participant evidence, not a parameter fit to the cited human studies, and not a result about universal human teaching behavior.

## 6. Adversarial and counterfactual experiments to add

- **Benefit inflation:** raise declared benefit while holding true benefit/reliability fixed.
- **Evidence corruption:** lower reliability and raise uncertainty for a teacher's favorite items.
- **Context shift:** move the learner to a different task family after teaching.
- **Bandwidth collapse:** reduce the packet budget and measure which capabilities disappear.
- **Rare capability preservation:** create one low-frequency, high-benefit lineage and apply memory pressure.
- **Prestige conflict:** high-prestige teacher holds a weak item; lower-prestige teacher holds a strong one.
- **Teacher collusion:** several demonstrators amplify the same false item.
- **Causal explanation vs replay:** keep the same item, resource budget and demonstration examples while varying whether the teacher transmits causal structure, executable rules or examples only.
- **Teach-back:** learner re-expresses the skill as a new executable artifact, evaluated independently.
- **Correction:** send a later contradiction or correction and measure detection/recovery and descendant contamination.

## 7. Proposed benchmark targets

These are proposed engineering goals, not achieved results or community standards:

| Area | Proposed initial gate |
|---|---|
| Budget integrity | No packet or learner attempt exceeds its declared budget; 100% of fixtures tested |
| Stage accounting | Counts obey offered ≥ observed ≥ acquired ≥ integrated ≥ retained ≥ transferred where denominators and stages apply |
| Oracle isolation | No policy code path reads evaluator-only true benefit, validity or transferability |
| Overclaim resistance | Evidence-calibrated selection produces fewer false adoptions than declared-benefit selection at a matched bandwidth budget |
| Learner fit | Context-aware selection improves held-out learner utility on task-family shifts without increasing false adoption |
| Diversity | Diversity-aware selection reports a Pareto frontier; no forced single composite score |
| Individual-learning fairness | Same resource budget, same item universe, same evaluator and same seed schedule |
| Transfer | Compare on task families not used to define teacher selection or calibrate the policy |
| Calibration | If reliability metadata is supplied, report its calibration and sensitivity; do not assume the declared estimate is true |

For final claims, use paired seeds and paired differences, uncertainty intervals, effect sizes, and ablations. A result should be called superior only on a named held-out task and budget, with the strongest matched baseline included.

## 8. Integration boundaries

- **Symthaea:** cognitive learning, teach-back, skill representation and adaptive selection.
- **Mycelix:** provenance, attestations, rights/consent and dispute records for transmitted items.
- **Symtropy:** population-scale and network-topology experiments.
- **Sol-Atlas:** visualization of knowledge-resource geography and transfer scope; not a validator.
- **CivBench:** task manifests, matched budgets, baselines, scoring, uncertainty and independent evaluation.

The intentional-teaching module does not introduce a new knowledge ledger or network transport protocol.
