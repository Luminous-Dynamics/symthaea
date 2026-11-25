# Decision-qualified scientific discovery for the 5D EGB response atlas

**Version:** 0.1.0  
**Date:** 2026-11-24  
**Status:** research protocol / pilot design; no performance claim

## Thesis

The useful objective is not to maximize model disagreement, posterior entropy, number of simulations, or papers written. It is to **spend the next unit of research effort on evidence that can change a scientifically important decision, then independently qualify that evidence**.

The campaign should therefore optimize a *decision-qualified discovery frontier*: the relationship between (a) correctness of claim disposition, (b) cost to independently qualified evidence, (c) false-promotion risk, and (d) coverage of relevant physical regimes. It should not hide these dimensions in one weighted leaderboard score.

## Paradigm shifts

### 1. From information gain to decision sufficiency

A calculation can be informative but irrelevant to the question. Each action must name the claim or branch decision it can change, the possible dispositions, the consequence of each disposition, and a prospective stopping/pivot rule. Stage A chooses a research action using the existing Symthaea value-of-computation owners (#8721 / #11171). Stage B chooses the numerical design inside that action using an experiment-design method (#8034 after qualification, then covariance-aware extensions if justified).

For finite enumerable toy episodes, evaluate policy regret against an exact action oracle. For larger episodes, report the best known qualified decision frontier rather than pretending an oracle exists.

### 2. From closed-world discrimination to model criticism

A selector that separates H1 from H2 can still fail when both are wrong. Every campaign episode therefore needs a **model-adequacy challenge channel** in addition to rival-model discrimination. Suitable EGB controls include independently re-derived equation rows, held-out original equations, alternate endpoint propagation, chart changes, Noether/constraint checks, higher precision, and residual components withheld from model fitting. A design is not successful merely because it confidently chooses one member of a misspecified hypothesis family.

This is motivated by 2026 active-learning research showing that informativeness alone can amplify errors under model misspecification; the R-IDeA approach explicitly adds representativeness and de-amplification to informativeness. The campaign should implement the underlying adversarial test, not claim to reproduce that paper's method.

### 3. From scalar error bars to structured uncertainty

Do not merge the following into one standard deviation:

- measurement / observation noise;
- numerical discretization and solver error;
- finite-precision / conditioning error;
- perturbative truncation error;
- parameter uncertainty;
- model-form / omitted-equation uncertainty;
- source, fixture, or provenance mismatch.

Keep a provenance record for each component and combine only under declared assumptions. Where no defensible distribution exists, carry intervals or bounded sets rather than inventing a Gaussian posterior.

### 4. From many observables to independent information

The response matrix, determinant, singular values, condition number, and derived endpoint summaries are not independent measurements. The design contract must declare a canonical primary observable vector and the transformations that generate derived values. Carry the full covariance of the rival-prediction difference, including cross-terms; record why any cross-covariance is zero before assuming independence.

For a full-column-rank linear observable transform **L** and positive-definite base covariance **S**, an augmented covariance **A = L S Lᵀ** is singular when redundant outputs are added. Its exact pseudoinverse quadratic form must preserve the base Mahalanobis score on the supported subspace. No silent diagonalization, jitter, ridge, or pseudo-inverse tolerance may turn an unsupported direction into evidence.

This is a known methodological challenge, not an already solved general algorithm: a 2026 review of optimal designs for correlated data notes that correlated designs need stronger assumptions, that generally effective algorithms remain limited, and that design optimality is not routinely certified.

### 5. From post-hoc stories to prospective, auditable policies

Before executing an action, freeze the evidence-prefix digest, policy/model identity, predicted outcomes, expected decision changes, resource budget, stop/pivot/abstain rule and forecast. After execution, append the actual outcome and an independent receipt; never revise the original forecast. Use hidden holdouts and a separate evaluator so the proposing policy cannot select or score its own test set.

## EGB-specific implementation contract

Every candidate design should include:

1. immutable hypothesis/model IDs, source digests and exact equation/AST identities;
2. parameter point, gauge, ensemble, chart, horizon/infinity basis and units;
3. canonical primary observables plus a derived-observable dependency map;
4. predicted mean-difference vector and full covariance/cross-covariance provenance, or a bounded uncertainty set;
5. basis and unit transformation, including metric/Gram transport from Symtropy #7131;
6. numerical error decomposition, solver identity, precision, resolution and independent route;
7. model-adequacy probes that can reject the entire candidate family;
8. claim-specific action outcomes and a predeclared stopping / abstention rule;
9. inclusive costs: wall time, CPU/GPU, memory, precision, solver calls, model tokens, monetary cost and human review.

A covariance-aware score is a screening statistic, not automatically EIG, a Bayes factor, or a calibrated posterior. EIG is eligible only if the likelihood, prior and predictive distribution are independently justified and calibration is demonstrated.

## Benchmark programme

### External agent benchmarks — harness and cost context

| Suite | Current public reference point | What it tests | What it does not establish for EGB |
|---|---|---|---|
| ASI-Bench | 60 tasks / 11 domains; B1–B4 guidance gradient; 26 aggregate rows in the 10 Nov 2026 snapshot. Highest listed row: GPT-6.1 Sol + Codex, ultra, 63.90. | Project-level scientific execution as method guidance is removed; sandboxed artifacts and scored outputs. | Correctness of 5D EGB equations, interval theorems or physical stability. |
| SciWorkBench | 98 workflow tasks; strongest reported model–harness mean 57.38/100 in the 17 Nov 2026 preprint. | Real science workflow across biology, chemistry/materials and physics. | Domain-specific numerical-gravity qualification or calibrated decision value. |
| FIRE-Bench | 30 rediscovery tasks; official page reports best F1 46.7 and high run-to-run variance. | Full-cycle rediscovery of previously verified findings, with experimental design and evidence-backed conclusions. | Novel EGB discoveries or completeness of an EGB branch. |
| NewtonBench | 324 interactive scientific-law tasks across 12 physics domains; its study reports cases where tool availability can prompt premature exploration-to-exploitation transitions. | Causal evaluation of when tools help or harm law discovery under varying complexity/noise and changing laws. | A direct EGB numerical-gravity solver benchmark or a universal conclusion that tools hurt. |
| AstaBench | 2,400+ tasks across literature, coding/execution, data analysis and end-to-end discovery; Spring 2026 update reports Opus 4.7 at 58.0% / $3.54 per problem and GPT-5.5 at 52.9% / $1.61. | Breadth, end-to-end completion and quality-cost trade-offs. | A matched physics solver comparison; its E2E scorer and general scientific tasks are distinct. |

Treat each score as a dated model+harness+effort+scorer result, not a property of a model alone. Re-run official protocols and preserve the execution bundle before claiming an external SOTA comparison.

### Numerical-gravity comparator

Akribeia's paper was accepted by Physical Review D on 12 November 2026. It combines physics-informed neural fields and pseudo-spectral refinement for rotating higher-curvature black holes, including multiple non-zero angular momenta. This is a serious direct numerical-gravity comparator. It is not yet a matched leaderboard comparison to Symtropy's perturbative equal-spin O(a²) response map: dimensions, theory/couplings, spin order, ensembles, gauges, observables, residual normalization, precision and compute budgets must be aligned first. Compare exact 5D Einstein-limit controls immediately; compare nonlinear finite-coupling solutions only after both teams publish matched problem definitions and independent residual/constraint evaluation.

## Required benchmark arms

1. exact fixed schedule;
2. random matched-cost action and design;
3. static residual/uncertainty threshold;
4. existing prediction-variance disagreement baseline;
5. PR #8034 independent-observable maximin baseline, only after package-specific qualification;
6. covariance-aware robust maximin design;
7. calibrated Monte Carlo EIG only on episodes with defensible priors and likelihoods;
8. Stage-A VoC + Stage-B numerical design;
9. Stage-A only and Stage-B only, to isolate effects;
10. no-HDC, no-CfC and no-VoC ablations of the actual #11171 policy incumbent;
11. model-adequacy/adversarial-discrepancy probe;
12. exact tiny-world oracle for enumerated episodes only.

Keep backbone, harness, prompt, tools, solver and inclusive resource budgets matched. Report results separately by episode family and difficulty; use paired seeds and uncertainty intervals. Do not mix pilot episodes into the sealed confirmatory set.

## Metrics: report a frontier, not a vanity score

- **Claim disposition:** false promotions, false rejections, correct abstentions and unsupported-singleton rate.
- **Independent qualification:** exact-source replay, original-row/Noether checks, interval enclosure and theorem-obligation acceptance.
- **Decision value:** prospective outcome calibration, realized decision change, regret versus tiny-world oracle and cost to first qualified decision.
- **Experiment quality:** worst-pair discrimination, covariance calibration/coverage, redundancy invariance, unit/basis invariance and model-adequacy rejection power.
- **Physics quality:** independent solver agreement, spectral/precision convergence, endpoint overlap, validated parameter/arclength coverage, residual norms by original equation and interval branch coverage.
- **Cost:** CPU/GPU hours, wall time, solver calls, precision, memory, tokens, dollars and human review.
- **Reliability:** failures, timeouts, retries, missing outputs and abstentions remain in denominators.

## Proposed gates and aspirational targets (not achieved)

- Zero false promotions on all frozen hostile controls; every malformed covariance, stale digest, unsupported mean, false basis transport and source mutation must fail closed.
- 100% correct handling of exact rational and Einstein-limit anchors in the frozen corpus.
- Exactly zero acquisition-score gain from algebraic duplicate observables; invariance under declared basis and unit transformations.
- Nominal predictive intervals must demonstrate their predeclared empirical coverage on independent controls before any probabilistic design score is used.
- The Stage-A policy must reduce median cost-to-qualified-decision by at least 20% versus the strongest simple policy at matched inclusive cost, with paired confidence intervals and no increase in false promotion, before claiming a practical VoC gain.
- Stage B must improve the qualified-result/cost Pareto frontier over #8034 and fixed/random baselines, without degrading model-adequacy rejection.
- For external agent suites, first reproduce official baselines. Only then target at least +5 absolute points on an EGB-relevant frozen slice, across at least three seeds, with no higher inclusive cost per task and no evaluator leakage. This is a target, not a current result.

## Immediate execution sequence

1. Run **python3 docs/research/verify_bh_egb_covariance_fixture.py**; it must reproduce the exact 4/4 Mahalanobis invariance and reject all four built-in hostile mutations.
2. Use that fixture as a regression gate when #8034 is unblocked by the workspace repair; do not yet retrofit covariance code into the unqualified crate.
3. Extend #13418's pilot with explicit model-adequacy probes (whole-family failure detection), claim-specific utility and source-independent replay actions.
4. Keep Symtropy #7131 as the owner of response/basis semantics and #7135 as the owner of local theorem obligations.
5. Pin the actual #11171 policy callable and freeze evaluator, resources, holdouts, power/stopping rules before confirmatory execution.
6. Compare external agent suites and Akribeia only at the boundaries described above.

## Reproducibility artifacts and claim ceiling

- Synthetic known-answer fixture: **BH_EGB_COVARIANCE_REDUNDANCY_CONTROL_V0_1.json**.
- Exact executable regression: **verify_bh_egb_covariance_fixture.py**.
- Dedicated exact replay receipt: **BH_EGB_COVARIANCE_VERIFICATION_RECEIPT_V0_1.json**.

A passing fixture proves only this exact synthetic linear-algebra construction and its declared negative controls. It does not qualify PR #8034, establish a calibrated covariance model, solve an EGB equation, validate a Symtropy response operator, prove a local branch theorem, or show SOTA performance.

## References

- ASI-Bench, leaderboard and project-level benchmark: https://asibench.apexin.ai/ (10 Nov 2026 snapshot).
- SciWorkBench, arXiv:2611.22490 (17 Nov 2026): https://arxiv.org/abs/2611.22490.
- FIRE-Bench, ICML 2026 / PMLR 306: https://proceedings.mlr.press/v306/wang26d.html and https://firebench.github.io/.
- NewtonBench, 324 interactive scientific-law tasks / 12 physics domains (ICLR 2026): https://arxiv.org/abs/2510.07172 and https://github.com/HKUST-KnowComp/NewtonBench.
- AstaBench update (Spring 2026): https://allenai.org/blog/astabench-update-spring-2026.
- López-Fidalgo & Wong, “Optimal Designs for Correlated Data,” Annual Review of Statistics and Its Application 13 (2026): https://doi.org/10.1146/annurev-statistics-042324-012947.
- Tang, Sloman & Kaski, “Representative, Informative, and De-Amplifying…,” AISTATS 2026 / PMLR 300: https://proceedings.mlr.press/v300/tang26d.html.
- Agurto-Sepúlveda et al., “Neural-Spectral Discovery of Rotating Black Holes Beyond General Relativity,” Phys. Rev. D accepted 12 Nov 2026: https://doi.org/10.1103/3kz2-6dvv.


## Open-world model-adequacy regression

A second independent synthetic control now tests a different failure mode: candidate models can be highly distinguishable from each other while *all* fail the same predeclared held-out adequacy gate.

- Fixture: **BH_EGB_OPEN_WORLD_MODEL_ADEQUACY_CONTROL_V0_1.json**.
- Exact verifier: **verify_bh_egb_open_world_model_adequacy.py**.
- Replay receipt: **BH_EGB_OPEN_WORLD_MODEL_ADEQUACY_RECEIPT_V0_1.json**.

The frozen held-out observation is \([0,0]\), with identity covariance and maximum admissible squared residual \(1/2\). The two rival models predict \([1,0]\) and \([0,1]\): their pairwise squared separation is \(2\), yet each has held-out residual squared \(1\). The only admissible family-level decision is \`ABSTAIN_AND_EXPAND_MODEL_CLASS\`. An outside-family witness predicts \([0,0]\) and passes with residual zero. Four mutations (post-hoc gate relaxation, held-out observation mutation, candidate mutation, and outside-family witness mutation) are rejected.

This is deterministic exact arithmetic and decision-logic coverage—not a chi-square test, predictive calibration, or physical observation. In actual EGB work, the adequacy channel must be grounded in separately withheld original equation rows, independent derivations, alternate endpoint propagation, Noether/constraint checks, chart and unit metamorphisms, and precision/solver refinement. It must not inspect a holdout selected or scored by the proposing policy.

The two synthetic controls now test orthogonal acquisition invariants:

1. **No fake information from redundancy:** full covariance transport and singular-support-aware pseudoinverse preserve the exact score under a duplicated observable.
2. **No false confidence from a closed candidate set:** pairwise separation cannot promote any candidate when the entire family fails the frozen adequacy gate.

The first test is an exact rational linear-algebra control; the second is an exact deterministic model-adequacy control. Neither establishes an EGB physical result, policy performance, or SOTA.



## Coordinate and unit metamorphic regression

The exact covariance verifier has now been extended beyond redundant-output invariance. It also transports a nonsingular mixed basis change \(P\) and a nontrivial unit scaling \(U\), checking in exact rational arithmetic that:

\[
\Delta\mu' = P\Delta\mu,\qquad
S' = P S P^{\mathsf T},\qquad
(\Delta\mu')^{\mathsf T}(S')^{-1}\Delta\mu'
=\Delta\mu^{\mathsf T}S^{-1}\Delta\mu=4.
\]

The synthetic mixed-basis transform is \(P=\begin{bmatrix}2&1\\1&1\end{bmatrix}\), with transported mean \([4,3]\) and covariance \(\begin{bmatrix}7&9/2\\9/2&3\end{bmatrix}\). The unit transform is \(U=\mathrm{diag}(1000,1/100)\), with mean \([1000,1/50]\) and covariance \(\begin{bmatrix}1000000&5\\5&1/10000\end{bmatrix}\). Both preserve the exact score 4. These controls test the algebraic rule that coordinates, units and covariance must move together; they do not verify any physical endpoint metric. Symtropy #7131 remains authoritative for the actual horizon/infinity Gram metrics and their transport.

The covariance fixture now has four exact properties: correlated covariance is respected; duplicate observables do not create information; invertible basis changes preserve the score; and unit rescaling preserves the score. All values are synthetic known answers.


## Decision value versus information gain: exact Stage-A oracle

A third finite synthetic control distinguishes research-action allocation (Stage A) from numerical design (Stage B).

Four equiprobable latent states fall into two decision classes, A and B. The pre-action Bayes risk under zero-one loss is \(1/2\). Two equally costly actions each reveal exactly one bit:

- **A_entropy_only** reveals a within-class subtype. It distinguishes states but leaves the posterior decision-class risk at \(1/2\). Gross decision value is zero; with cost \(1/10\), net VoC is \(-1/10\).
- **B_decision_relevant** reveals the decision class. It reduces posterior Bayes risk to zero. Gross decision value is \(1/2\); with the same cost, net VoC is \(2/5\).

The exact finite oracle selects B even though both actions have identical one-bit information gain and identical cost. This is the concrete reason to optimize decision value / cost-to-qualified-decision rather than maximize entropy reduction or model disagreement.

Artifacts:

- **BH_EGB_DECISION_VALUE_CONTROL_V0_1.json** — fully enumerated exact state/action oracle.
- **verify_bh_egb_decision_value.py** — exact-rational replay and four hostile mutations.
- **BH_EGB_DECISION_VALUE_VERIFICATION_RECEIPT_V0_1.json** — replay receipt.

This is a deterministic toy oracle, not a learned policy, real-world utility function, or calibrated EGB action model. Before using this logic in the live #8721/#11171 Stage-A policy, actions must bind to actual claim dispositions and independently measured resource costs; any probability forecast must be calibrated prospectively.


### NewtonBench as the Stage-A tool-allocation comparator

NewtonBench is particularly relevant to the research-action policy, not just the Stage-B numerical design. Its 324 interactive tasks span 12 physics domains and include controlled changes in laws, systems, complexity and observation noise. The reported finding that optional tool access can sometimes encourage premature exploitation motivates testing **tool availability versus tool allocation** as separate causal factors. It does not imply tools are generally harmful.

Use the existing Symthaea #8721 adapter/owner rather than duplicate its benchmark. Keep the NewtonBench scorecard separate from the EGB numerical-gravity scorecard. For an EGB-specific slice, transplant only compatible task mechanics (prospective probe choice, law/parameter shifts, noisy and misleading early evidence, non-identifiability, stopping and cost); the evaluator must preserve the EGB truth/holdout outside the policy process. The finite Stage-A decision-value fixture in this campaign is a tiny known-answer contract test, not a NewtonBench reproduction.
