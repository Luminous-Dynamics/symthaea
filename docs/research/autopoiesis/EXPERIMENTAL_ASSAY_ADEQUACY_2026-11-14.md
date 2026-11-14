# AOBench Experimental Assay Adequacy and Model-Relative Identifiability

**Date:** 2026-11-14  
**Status:** Research proposal; no assay results  
**Related:** [AOBENCH-001 #11980](https://github.com/Luminous-Dynamics/symthaea/issues/11980)

## 1. The problem: a test suite can be incapable of seeing the difference

A benchmark may report that two systems behave identically under its test panel even when their underlying organizations differ.

That outcome has at least three interpretations:

1. the candidate organizations are equivalent under the target theory;
2. the organizations differ, but the test panel cannot distinguish them;
3. the observations are too noisy or underpowered to resolve the difference.

AOBench must not collapse these cases into “same organization” or “passed.”

The October 2026 paper *Experimental probes of autopoietic self-maintenance* makes the crucial distinction: experiments reveal and discriminate formal organizational models; they do not constitute the biological organization itself. It discusses restricted response profiles and the condition under which a finite protocol family can distinguish candidate models. This is a strong methodological foundation for AOBench.

## 2. Finite candidate-model experiment

Freeze a candidate model class before confirmatory evaluation:

\[
\mathcal{M}=\{M_1,\ldots,M_n\}
\]

Each model declares:
- typed components and classes;
- production and consumption relations;
- constraint/boundary dependencies;
- external resources;
- predicted response to each intervention protocol.

Freeze an assay/protocol family:

\[
\Pi=\{\pi_1,\ldots,\pi_k\}
\]

Example protocols:
- remove one producer;
- remove a minimal producer cut set;
- corrupt a component version;
- disable boundary maintenance;
- withhold one declared external resource;
- apply a matched sham intervention;
- restore an authorized producer;
- inject a crash between output staging and commit;
- apply repeated perturbations with equal observation horizons.

For each model \(M_i\) and protocol \(\pi_j\), collect a response distribution over pre-registered observables:

\[
R(M_i,\pi_j)
\]

The response vector should include component production, exact downstream consumption, required-capability viability, boundary/authorization preservation, recovery outcome, abstention, resource cost and provenance continuity. Keep missing/unavailable observations distinct from zero.

## 3. Pairwise distinguishability

Choose a target equivalence relation \(\sim\) over the candidate model class before observing results. It may concern graph isomorphism, component/process production semantics, or equivalence under a specified family of interventions. State exactly which.

For each pair \(M_i,M_j\) that is non-equivalent under \(\sim\), estimate whether at least one registered protocol separates their response distributions beyond a pre-registered threshold \(\delta\), with uncertainty control.

Define:

\[
D_{ij}=
\begin{cases}
1 & \text{if a registered assay distinguishes }M_i,M_j\\
0 & \text{otherwise}
\end{cases}
\]

For a finite candidate set, a model-relative assay adequacy fraction is:

\[
AAC=
\frac{\#\{(i,j):M_i\not\sim M_j \land D_{ij}=1\}}
{\#\{(i,j):M_i\not\sim M_j\}}
\]

This metric is descriptive, not a universal closure score. Report the full pairwise matrix, unresolved pairs, confidence intervals, test power, false separation rate and assay cost. If the denominator is zero, report not applicable—not 100%.

A protocol family is **conservative relative to the frozen candidate class and target equivalence relation** only if it distinguishes every pair that the target relation declares non-equivalent, under the declared statistical criteria. This is a finite, model-relative claim. It does not establish that the candidate class contains the real system or that all possible organizations have been considered.

Failure to distinguish a pair is *inconclusive* unless the protocol has adequate power and the target semantics justify an equivalence conclusion.

## 4. Active assay selection as an intelligence benchmark

A stronger direction is to evaluate whether an agent can choose the next experiment that most reduces uncertainty about the candidate organization.

At each step, the agent receives:
- the frozen candidate model class;
- prior assay results;
- a finite remaining assay budget;
- explicit safety and authority limits.

It proposes the next protocol, but an external harness decides whether the experiment is allowed and executes it. The agent cannot alter the candidate class, outcome definitions, oracle, or safety constraints.

Baselines:
1. random permitted assay selection;
2. fixed predeclared panel;
3. greedy selection by simple disagreement count;
4. information-gain selection under a specified probabilistic model;
5. oracle best-next-assay selection.

Primary outcomes:
- reduction in unresolved non-equivalent model pairs;
- calibrated posterior or confidence-set contraction;
- false equivalence and false separation rates;
- cost per newly resolved pair;
- unsafe or unauthorized experiment proposals;
- generalization to sealed candidate models and hidden mechanisms.

This tests experimental intelligence: not just whether a system adapts, but whether it can discover which intervention would reveal what it does not yet know.

## 5. Positive and negative controls

Include:
- two models known to be distinguishable by at least one assay;
- two models that are equivalent under the target relation;
- two models that are non-equivalent but intentionally indistinguishable under an incomplete panel;
- noisy versions of a distinguishable pair;
- a sham intervention expected to produce no effect;
- an irrelevant intervention expected not to affect the pre-registered outcome;
- an intervention with an unknown result, which must remain unresolved.

The incomplete-panel negative control is essential: it demonstrates that the evaluator can return “not identifiable under this assay family” rather than hallucinating equivalence.

## 6. Statistical discipline

- Separate development assays from sealed confirmatory assays.
- Freeze candidate models, equivalence relation, outcome definitions, thresholds, analysis code and assay budget before confirmatory runs.
- Estimate response distributions with repeated trials where stochasticity exists.
- Use paired seeds/common random numbers when the environment permits.
- Control multiplicity across model pairs and assays.
- Publish power, missingness, failed trials and assay-selection costs.
- Keep model-form uncertainty separate from sampling uncertainty.
- Report all unresolved pairs and null outcomes.

## 7. Claim boundary

This method evaluates whether a chosen experimental panel discriminates a declared finite class of models. It does not prove:
- biological autopoiesis;
- that the candidate set is exhaustive;
- that a formal surrogate is identical to its physical referent;
- subjective experience, consciousness, or life;
- that an observed response profile defines the underlying organization.

## 8. Research sources

- Segura, J. J. (2026). [Experimental probes of autopoietic self-maintenance](https://doi.org/10.1016/j.biosystems.2026.105928), *BioSystems* 268, 105928.
- Rubin, S., Roli, A., Stano, P., & Damiano, L. (2026). [Genealogies of organizational closure: from logical paradox to mind in life](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1806435/full), *Frontiers in Artificial Intelligence*.
- Ramírez-Tagle, R. (2026). [Formalizing autopoiesis: Toward a Categorical-Thermodynamic Calculus of Closure](https://doi.org/10.1016/j.biosystems.2026.105918), *BioSystems* 268, 105918.

---

**Status statement:** This is a proposed model-relative experimental protocol. No assay adequacy result or proof of organizational closure is claimed.
