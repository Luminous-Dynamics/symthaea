# SPECTRA Preference-Order-Aware Best-Arm Identification V0.1

**Status:** research specification; no production inference or stopping-rule change  
**Owner issue:** [#10838 — SPECTRA-ORDER-BAI-001](https://github.com/Luminous-Dynamics/symthaea/issues/10838)  
**Primary paper:** Xue, Cheng, Wan & Qiu (13 Oct 2026), [When More Objectives Make Identification Easier: The Role of Preferences and Feedback](https://arxiv.org/abs/2610.18577)

## 1. Research question

Can SPECTRA identify the *complete set of decisions that are optimal under a registered preference relation* with fewer observations than matched uniform, coordinate-feedback, or preference-specific baselines—while preserving the declared error guarantee?

This is not equivalent to finding a scalar winner, the Pareto set under every decision context, a set satisfying conjunctive thresholds, or a small portfolio covering many objectives. The preference relation determines the answer map; the answer map determines which alternative instances must be ruled out; those alternatives determine the information-optimal allocation.

## 2. Exact theorem-matched reference problem

The first reproduction must match the paper rather than generalize it prematurely.

Let there be K finite arms and M reward coordinates. Pulling arm i returns the full vector

\[
X_t = \mu_{I_t} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}(0,I_M),
\]

with noise vectors independent across rounds. One pull is one vector observation, not M independently purchased observations.

A known model class \(\Theta\) and a registered preference relation \(\preceq\) induce the complete maximal-arm answer set

\[
A_{\preceq}(\mu)
= \{i\in[K]: \nexists j\in[K]\text{ such that }\mu_i\prec\mu_j\}.
\]

The answer-changing alternatives are

\[
\operatorname{Alt}_{\preceq}(\mu)
= \{\nu\in\Theta:
 A_{\preceq}(\nu)\ne A_{\preceq}(\mu)\}.
\]

For an allocation \(w\in\Delta_K\), the information rate under the paper's isotropic Gaussian model is

\[
\mathcal I(\mu,\nu;w)
= \frac12\sum_{i=1}^{K}w_i\|\mu_i-\nu_i\|_2^2.
\]

The characteristic information and characteristic time are

\[
\mathcal I^*_{\preceq}(\mu)
= \sup_{w\in\Delta_K}
  \inf_{\nu\in\operatorname{Alt}_{\preceq}(\mu)}
  \mathcal I(\mu,\nu;w),
\qquad
T_{\preceq}(\mu)=1/\mathcal I^*_{\preceq}(\mu).
\]

The fixed-confidence lower bound is

\[
\mathbb E_\mu[\tau_\delta]
\ge T_{\preceq}(\mu)\,
\operatorname{kl}(1-\delta,\delta),
\]

for methods that are \(\delta\)-correct over the stated model class. The paper assumes local stability of the answer map at the true instance and a nonempty answer-changing alternative set. These are eligibility conditions, not implementation details that can be silently ignored.

### Preference families

| Preference | Registered decision semantics | Required output |
|---|---|---|
| Pareto | Coordinatewise dominance | Entire nondominated set |
| Linear | Fixed nonzero nonnegative vector \(v\); maximize \(v^\top\mu_i\) | Complete maximal set under the scalarized order |
| Lexicographic | Frozen priority order; compare the first coordinate on which alternatives differ | Complete maximal set, with exact tie behavior |
| Constrained lexicographic | Lexicographic order over a declared feasible model class | Only qualified when the structural restriction is independently established |

A preference is normative input to the experiment, not a latent label to infer from reward data. When the preference is unspecified, contested, or changes during the campaign, the decision contract is unresolved; sensitivity across separately registered preferences may be reported.

### Preference-known and preference-unknown are separate tasks

OATS is a *known-preference* fixed-confidence identification method. SPECTRA must not claim that it also learns an unknown or hidden stakeholder preference.

The 2026 literature adds useful but different comparators:
- Cao, Shi & Shroff, *Provably Efficient Multi-Objective Bandit Algorithms Under Preference-Centric Customization* (AAAI 2026), studies both unknown and hidden preference settings and develops preference-estimation plus preference-aware regret minimization. This is a preference-learning/regret problem, not a drop-in fixed-confidence OATS baseline: https://doi.org/10.1609/aaai.v40i24.39072
- Cao, Shi & Shroff, *Provably Efficient Personalized Multi-Objective Bandits with Proactive Conversational Queries* (UAI 2026), treats structured conversational preference signals using a Plackett–Luce subset-choice model and identifies a shift-invariance limit: query-only observations are insufficient to identify all preference parameters. This is evidence that natural-language requests can constrain preferences without uniquely specifying a utility function: https://proceedings.mlr.press/v337/cao26a.html
- Wang, Sun & Li, *Preference Is More than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback* (AAAI 2026), studies sparse pairwise preference feedback augmented with other human feedback: https://doi.org/10.1609/aaai.v40i31.39852

For SPECTRA, keep three contracts distinct:

1. **Preference supplied:** freeze the preference relation, then use OATS or another matched known-preference identification method.
2. **Preference uncertain:** represent an admissible set/distribution of preferences only where its construction is defensible. Report the union of possibly optimal actions, the intersection of robustly optimal actions, and whether the answer is invariant over the admissible set. Do not average stakeholder preferences into one hidden scalar.
3. **Preference elicitation is actionable:** select a stakeholder query or comparison only under an explicit feedback model, query cost, response-noise model, group/identity scope and stopping rule. A preference query and a candidate-performance experiment are different information channels and must be charged separately.

This can compose with [#10790](https://github.com/Luminous-Dynamics/symthaea/issues/10790) for multi-objective LLM configuration preference feedback, [#7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803) for decision-value-of-information benchmarking, and [#10320](https://github.com/Luminous-Dynamics/symthaea/issues/10320) for decision-sensitive active measurement. Those owners should remain authoritative for their specific domains. The OATS reproduction must remain a known-preference, Gaussian fixed-confidence track.

## 3. Why objective count alone is the wrong complexity proxy

The paper's two-arm analysis gives concrete counterexamples to the intuition that more objectives necessarily make identification harder.

For even M and mixed-sign gap

\[
d=(\Delta\mathbf 1_{M/2},-\Delta\mathbf 1_{M/2}),
\]

the unrestricted construction has

\[
T_{\mathrm{lex}}(d)/T_{\mathrm P}(d)=M/2.
\]

For the aligned gap \(d=\Delta\mathbf 1_M\), the paper obtains a vector-versus-coordinate characteristic-time ratio of 1 for lexicographic identification and M for Pareto identification. More generally,

\[
T_{\preceq}(\mu)
\le T^{\mathrm{co}}_{\preceq}(\mu)
\le M T_{\preceq}(\mu),
\]

and the realized benefit depends on the preference order and geometry of answer-changing alternatives.

The paper also shows regimes in which multi-objective identification is faster than the easiest single-coordinate problem. This is a comparison between different decision tasks, not a claim that vector feedback is always more efficient for the same task.

**SPECTRA implication:** benchmark preference geometry, vector/coordinate feedback and answer-set size as separate factors. Do not claim a universal “multi-objective speedup.”

## 4. OATS comparator

Order-Aware Track-and-Stop (OATS) repeatedly:

1. estimates the mean vector for each arm;
2. selects the candidate answer region closest to the current estimate;
3. evaluates accumulated information against the nearest answer-changing alternatives;
4. computes an allocation maximizing the estimated information rate;
5. mixes in vanishing forced exploration and tracks target allocation deficits; and
6. stops once its time-uniform evidence threshold is crossed.

Under the paper's stated model conditions, OATS is \(\delta\)-correct and asymptotically matches the characteristic-time lower bound as \(\delta\downarrow0\).

### Identification cost is not always the whole cost

Yang, Tan & Jin (JMLR 2026), *Best Arm Identification with Minimal Regret*, studies fixed-confidence best-arm identification while minimizing cumulative regret. It derives an instance-dependent lower bound and an impossibility result exposing a tension between reducing the expected sample count and reducing regret; its Double KL-UCB method is asymptotically optimal for the paper's single-parameter exponential-family setting: https://www.jmlr.org/beta/papers/v27/24-1612.html

This is not a direct multi-objective OATS comparator. It motivates a separately reported live-experiment objective: **cumulative decision loss incurred while obtaining evidence**. For offline simulation, sampling an arm may only consume compute; for live interventions, choosing an exploratory arm may affect participants, customers, operators or physical systems. Do not combine these into one generic regret number. Report (i) sample/compute cost to certificate, (ii) cumulative outcome regret under a declared decision-loss model, (iii) hard-constraint exposure, and (iv) human/participant burden separately. Compose this with the opportunity-cost outcomes already in [#10601](https://github.com/Luminous-Dynamics/symthaea/issues/10601); any regret-optimal sequential policy needs its own estimand, feedback model and proof.

OATS is the primary comparator, not a claim of SPECTRA novelty. A SPECTRA implementation must first reproduce its exact assumptions and reference behavior. Any gain must be measured against an exact, independently checked OATS implementation.

### What the published OATS experiment does and does not establish

The paper's appendix reports 500 independent runs for its numerical experiments. Its HELM-derived case begins with ratings for four models over five scenarios, aggregated from 678 prompts, then rescales the resulting mean vectors using a median Euclidean gap and generates observations from the Gaussian reference model. This is a useful controlled transfer fixture, but it is not a direct replay of raw prompt-level online evaluations or a field experiment.

In one reported four-arm, linear-preference configuration at confidence level 0.05, the paper reports mean stopping counts of 61,797 for OATS, 87,600 for uniform allocation, and 61,938 for the oracle allocation, with zero observed errors among 500 runs. This is a strong result for that specified Gaussian instance (roughly 29.5% fewer pulls than uniform), not a universal SOTA gain over all preferences, arm geometries, distributions or industrial experimentation systems. SPECTRA should reproduce the table and its setup before making any comparative claim, then extend the confirmation campaign to the larger run counts needed for tighter error-rate precision.

**Stopping-threshold fairness is theorem-sensitive.** The paper's threshold depends on the number of independent experiments and the dimension of the mean parameter. For the theorem-matched vector-feedback construction, the experiment count is K and the mean dimension is KM; for coordinate feedback, the count is KM and the mean dimension remains KM. Therefore use the paper's proof-correct threshold for each feedback mode at the same target δ. A common numeric threshold across vector and coordinate feedback is only a secondary ablation, and only if its validity is independently established for both modes. “Same δ and same guarantees” is the primary comparison; “same numeric threshold” is not automatically a fair or valid comparison.

## 5. Reproduction campaign

### Analytic and numerical fixtures

Begin with two arms, where the closest-alternative geometry admits analytic checks:

- mixed-sign gaps with even M; verify \(T_{\mathrm{lex}}/T_{\mathrm P}=M/2\);
- aligned gaps; verify the vector/coordinate feedback ratios;
- linear preferences with reinforcing and cancelling weighted gaps;
- common-ranking lexicographic cases, plus a negative control where common ranking is false;
- exact ties, near-ties, and answer-map instability controls.

Then move to K > 2. The allocation and closest-alternative optimization must be checked against an independent solver or a certified bracket. A point-estimate sampling heuristic is not a replacement for the information optimization.

### Dimensions and data-generating processes

| Factor | Initial levels |
|---|---|
| Arms K | 2, 3, 8, 16, 32 |
| Objectives M | 2, 4, 8, 16, 32 |
| Pareto set size | 1, small, large |
| Gap geometry | aligned, mixed-sign, orthogonal, correlated |
| Gap magnitudes | balanced, imbalanced, near-zero, exact tie |
| Preference | Pareto, fixed linear, lexicographic, qualified constrained lexicographic |
| Feedback | full vector, one-coordinate-at-a-time, partial coordinates |
| Confidence delta | 0.1, 0.05, 0.01, 0.001 |
| Model class | correctly specified, near-boundary, deliberate violation |

The initial confirmatory family must match independent \(\mathcal N(\mu_i,I_M)\) observations exactly. Correlated Gaussian noise, unknown covariance, bounded rewards, heavy tails, contextual drift and scenario-valued robust objectives are extension studies, each with a distinct method identity and proof obligation.

### Baselines

- OATS under the paper's exact model and preference semantics.
- Uniform allocation with the same valid stopping evidence.
- Fixed-confidence scalar Track-and-Stop when the preference reduces exactly to a scalar problem.
- Preference-specific Pareto / lexicographic algorithms where their assumptions match.
- OATS with coordinate feedback versus vector feedback.
- Oracle allocation from known truth, labelled as unattainable reference only.

All methods must share the same problem instance, confidence target, feedback model, resource boundary, stopping cap and random seed schedule. Do not compare nominal correctness guarantees that differ.

### Primary outcomes

- Exact answer-set recovery and empirical \(\delta\)-correctness, with Clopper–Pearson intervals.
- Expected, median and p90/p95/p99 arm pulls to certificate.
- Censored runs and nontermination, reported explicitly.
- Observed expected stopping time divided by characteristic-time lower bound, only where the theorem applies.
- Vector-feedback gain relative to coordinate feedback under the same answer map.
- False certification, unresolved rate, preference-instability sensitivity and model-mismatch rejection.
- Fully charged CPU, GPU, wall time, memory, human/participant time and currency.
- Exact seeds, DGP/version digests, answer-oracle outputs, trial accounting, replay and negative controls.

Use at least 10,000 independent runs per primary regime as an initial floor; increase sample counts when required to achieve the preregistered error-rate precision. Do not drop capped or failed trials from stopping-time summaries.

## 6. SPECTRA integration boundaries

This workstream complements but does not replace:

| Existing owner | Distinct role |
|---|---|
| [#10776](https://github.com/Luminous-Dynamics/symthaea/issues/10776) | Conjunctive threshold feasibility |
| [#10730](https://github.com/Luminous-Dynamics/symthaea/issues/10730) | Few-for-many compact portfolio coverage |
| [#10769](https://github.com/Luminous-Dynamics/symthaea/issues/10769) | Contextual and nuisance-orthogonal experimental design |
| [#10817](https://github.com/Luminous-Dynamics/symthaea/issues/10817) | Covariance-adaptive vector confidence sets |
| [#10773](https://github.com/Luminous-Dynamics/symthaea/pull/10773) | Componentwise resource-feasible Pareto frontier |
| [#10103](https://github.com/Luminous-Dynamics/symthaea/issues/10103) | Generic experiment kernel |
| [#10645](https://github.com/Luminous-Dynamics/symthaea/issues/10645) | Evidence and decision receipt composition |

The decision preference, statistical estimand, confidence method, scenario aggregation and resource objective are different layers. Compose them only with explicit typed contracts and independently validated adapters.

### Industry platform comparison is a separate qualification axis

OATS is an algorithmic fixed-confidence comparator; it is not an end-to-end experimentation-platform comparator. Industrial platforms already include experiment-integrity and causal-analysis capabilities that do not follow from a bandit identification theorem:

- Microsoft ExP documents sample-ratio mismatch and data-quality diagnostics (including data loss and join quality), guardrail monitoring and severe-regression shutdown.
- Uber XP combines fixed-horizon and sequential testing, causal methods and universal holdouts for long-term effects.
- Netflix describes a unified causal-analysis abstraction and workflows that support consistent production and local analysis.
- Statsig documents sequential testing, CUPED / variance reduction, stratification, guardrail metrics and multiple-comparison settings.

The live repository already assigns this platform-level comparison to [#10601](https://github.com/Luminous-Dynamics/symthaea/issues/10601): it includes assignment/exposure reconciliation, telemetry integrity, SRM, contamination/interference, guardrail pause/rollback, long-term holdouts and custody/replay. [#10103](https://github.com/Luminous-Dynamics/symthaea/issues/10103) owns the generic experiment kernel and assignment/custody mechanics; [#9215](https://github.com/Luminous-Dynamics/symthaea/issues/9215) owns the experiment-protocol DSL. Reuse those owners; do not duplicate a platform inside OATS.

Therefore SPECTRA needs two different demonstrations:
1. **Algorithmic result:** matched δ-correct identification with lower sample/resource cost than the best applicable statistical baselines.
2. **Platform result:** lower fully charged cost to a defensible, applicable, integrity-checked decision while meeting industrial data-quality, causal and guardrail requirements.

A result on (1) alone does not establish (2). The differentiator should be evidence-qualified decision assurance across the full chain, not just a smaller stopping time.

A valid OATS answer under its Gaussian benchmark does not establish causal effect, source truth, model adequacy, target-population validity, operational safety, or authority to deploy. Field experimentation additionally requires assignment/exposure reconciliation, SRM and telemetry checks, contamination and missingness diagnostics, long-term holdouts, and guardrail intervention evidence.

## 7. Prospective gates (not achieved results)

- At least 95% exact answer-set recovery at registered \(\delta\le0.05\) on every in-scope theorem-matched regime.
- Zero false certifications in deterministic order, tie, rescaling, answer-map and model-mismatch negative controls.
- Analytic characteristic-time ratios reproduced within 1%; independent numerical allocation solver agreement within a preregistered tolerance.
- On held-out matched regimes, improve the cost-to-certificate frontier over uniform allocation while retaining correctness.
- Demonstrate vector-feedback efficiency gains only where the preference geometry predicts them.
- Zero candidates marked resource-feasible when any resource upper bound is unknown or exceeds its budget.

A claim of beating OATS requires a matched correctness guarantee, identical budget/feedback conditions, exact source revisions, preregistered primary outcomes, repeated campaigns and independent reproduction. Fewer pulls with worse correctness is not an improvement.

## 8. Nonclaims

This note does not implement OATS, certify the paper's theorem independently, prove a new SPECTRA bound, validate non-Gaussian or adaptive vector inference, or claim a performance improvement. It is a research specification and reproduction plan. Production stopping rules remain unchanged until exact-head implementation, independent verification and benchmark qualification are complete.

## References

1. Xue, B., Cheng, J., Wan, Y. & Qiu, S. (2026). [When More Objectives Make Identification Easier: The Role of Preferences and Feedback](https://arxiv.org/abs/2610.18577), arXiv:2610.18577v1.
2. Jiang, X., Hashima, S., Hatano, K. & Takimoto, E. (2025). [Multi-thresholding Good Arm Identification with Bandit Feedback](https://proceedings.mlr.press/v304/jiang26a.html), PMLR 304.
3. Xue, B. et al. (2026). [Few-for-Many Best-Arm Identification in Multi-Objective Bandits](https://arxiv.org/abs/2610.16840), arXiv:2610.16840.
4. Kato, M. & Ariu, K. (2026). [The Role of Contextual Information in Best Arm Identification](https://www.jmlr.org/beta/papers/v27/22-0358.html), JMLR 27(51).
5. Cao, L., Shi, M. & Shroff, N. (2026). [Provably Efficient Multi-Objective Bandit Algorithms Under Preference-Centric Customization](https://doi.org/10.1609/aaai.v40i24.39072), AAAI 2026.
6. Cao, L., Shi, M. & Shroff, N. (2026). [Provably Efficient Personalized Multi-Objective Bandits with Proactive Conversational Queries](https://proceedings.mlr.press/v337/cao26a.html), UAI 2026.
7. Wang, S., Sun, H. & Li, K. (2026). [Preference Is More than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](https://doi.org/10.1609/aaai.v40i31.39852), AAAI 2026.
8. Yang, J., Tan, V. Y. F. & Jin, T. (2026). [Best Arm Identification with Minimal Regret](https://www.jmlr.org/beta/papers/v27/24-1612.html), JMLR 27(86).
9. Microsoft Research. [Patterns of Trustworthy Experimentation During Experiment](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-during-experiment/).
10. Uber Engineering. [XP: A Platform for Experimentation](https://www.uber.com/us/en/blog/xp/).
11. Netflix. [A Science-Centric Experimentation Platform](https://ieeexplore.ieee.org/document/9276550/).
12. Statsig. [Experiment options and advanced statistical settings](https://docs.statsig.com/statsig-warehouse-native/features/experiment-options).
