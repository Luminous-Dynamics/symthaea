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

OATS is the primary comparator, not a claim of SPECTRA novelty. A SPECTRA implementation must first reproduce its exact assumptions and reference behavior. Any gain must be measured against an exact, independently checked OATS implementation.

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
