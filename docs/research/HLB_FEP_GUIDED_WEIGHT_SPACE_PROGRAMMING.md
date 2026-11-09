# HLB: FEP-Guided Weight-Space Programming
## Research specification and implementation contract v0.1

**Status:** Research design and benchmark contract. No integrated runtime result is claimed.
**Parent:** `research/swi-001-weight-space-introspection` at `1dd6a93452a941bb46dba20afb2bf530716e5a9f`.
**Related work:** SWI-001 (#10627 / PR #10628), HLB-002 (#10712 / PR #10694), HLB factorial v1 (PR #8375), model-information-gain primitive (PR #8394), FEP action-information audit (#8391).

---

## 1. Decision

Symthaea should pursue both the Free Energy Principle (FEP) / active inference and weight-space programming (WSP), but they should occupy different control planes.

- **HDC/VSA and HRR** represent and compose state, relations, hypotheses, diagnostic traces, and reusable program fragments.
- **CfC/LTC liquid dynamics** evolve temporal state under irregular elapsed time. They remain a dynamical substrate, not a controller that is allowed to rewrite itself.
- **FEP / active inference** maintains beliefs over hidden states and competing model or fault hypotheses, predicts consequences of candidate observations/actions, and chooses information-gathering actions under preferences, cost, and safety constraints.
- **Weight-space programming** proposes explicit, bounded edits to typed parameter groups in response to a specification and evidence. It is a candidate generator, not an authority.
- **SWI** validates, stages, fingerprints, evaluates, accepts or rejects, and rolls back candidate parameter states.
- **Independent verification** determines whether the proposed repair fixed the target fault without damaging healthy behavior.

This creates a testable closed loop:

```text
observations / failure traces
        |
        v
HDC-HRR evidence + associative episode store
        |
        v
FEP belief over competing failure/model hypotheses
        |
        v
select diagnostic probe by expected information value
        |
        v
WSP proposes typed WeightProgram (candidate only)
        |
        v
SWI checks manifest, base digest, bounds and invariants
        |
        v
isolated evaluation on frozen development + sealed holdout
        |
        +---- reject / rollback <---- independent verifier
        |
        +---- accept as versioned candidate with receipt
```

The critical separation is that FEP selects *what to inspect or test*, WSP proposes *what to change*, and SWI/verifiers decide *whether the change is admissible*. No mechanism should silently certify its own proposal.

## 2. Why these two additions are worth doing

### 2.1 FEP is useful as belief-space control, not as a magic loss

Active inference offers a principled language for uncertainty over hidden causes, model identity, preferences, and the value of observations. It can help decide whether the next useful operation is to inspect a trace, run a counterfactual, probe a model, or defer a repair.

But Symthaea's existing source audit found that the production epistemic score was not action-dependent for a fixed belief/model: predicted precision was updated identically for every action and mode probabilities were copied unchanged, while the entropy term depended on those values rather than the predicted mean. Thus the current score did not establish expected information seeking. See issue #8391.

PR #8394 adds an isolated model-hypothesis mutual-information primitive. It is mathematically useful, but it is not wired to `ActiveInferenceAgent::select_action`, does not learn the model set, and does not include pragmatic preferences, observation cost, safety, or execution authority. The next work must bind real, provenance-identified outcome models and verify information gain against realized disambiguation.

For hypotheses (M), candidate diagnostic action (a), and possible outcome (o):

```text
p(o | a) = sum_m p(m) p(o | m, a)

p(m | o, a) = p(m) p(o | m, a) / p(o | a)

I(M; O | a) = H(M) - sum_o p(o | a) H(M | o, a)
```

### Three different quantities must remain separate

A diagnostic can reveal a great deal about a model without changing the best action. Conversely, a relatively low-information observation can be valuable if it changes a consequential downstream decision. We therefore distinguish:

1. **Model information gain:** \(I(M;O\mid a)\), information about which generative/fault model is true.
2. **Optimal-decision information gain:** \(I(D^*;O\mid a)\), information about which downstream decision is optimal.
3. **Expected value of sample information (EVSI):** expected reduction in downstream Bayes risk, measured in the task's declared loss units.

For a fixed loss table \(L(d,m)\), define the hypothesis-conditioned optimal decision with a deterministic tie-break:

```text
D*(m) = argmin_d L(d, m)

p(d* | a) = sum_m p(m) 1[D*(m) = d*]

I(D*; O | a) = H(D*) - sum_o p(o | a) H(D* | o, a)
```

Let \(q(m)\) denote the current model posterior and define Bayes risk as:

```text
R(q) = min_d sum_m q(m) L(d, m)

EVSI(a) = R(q) - sum_o p(o | a) R(q(m | o, a))
```

If \(D^*\) is a deterministic function of model hypothesis \(M\), the data-processing inequality requires:

```text
0 <= I(D*; O | a) <= I(M; O | a)
```

The implementation checks this bound (within a declared numerical tolerance). It is a mathematical invariant, not an empirical performance target.

These quantities answer different questions. Model EIG asks whether the observation distinguishes models. Decision EIG asks whether it distinguishes which decision is optimal. EVSI asks whether observing it is expected to reduce the actual loss. Decision EIG does not encode loss magnitudes or probe costs; EVSI does. Even EVSI must be adjusted for operational cost and hard safety eligibility before a probe is selected.

A reproducible synthetic counterexample in the branch separates them: an irrelevant nuisance probe has model EIG \(\ln 2\) nats but zero decision EIG and zero EVSI. A noisier fault-relevant probe has model/decision information gain \(\ln 2-H(0.3)\) nats and EVSI 0.2 loss units. At cost 0.05, the EVSI policy may choose it; at cost 0.25, the cost-aware policy declines it even though targeted decision information remains positive. These are analytical fixture values, not measured Symthaea results.

The targeted decision-information criterion is related to published *Targeted Active Learning for Bayesian Decision-Making* (Filstroff et al., TMLR 2024, https://openreview.net/forum?id=KxPjuiMgmm). The current branch implements a one-step finite-hypothesis criterion analogue for a controlled comparison; it is not a full reproduction of that paper's sequential algorithm.

A candidate cost-aware decision score may therefore be written as:

```text
J_EVSI(a) = sum_o p(o | a) R(q(m | o, a))
          + lambda_cost * C(a)
          + lambda_risk * R_safety(a)

          = R(q) - EVSI(a)
          + lambda_cost * C(a)
          + lambda_risk * R_safety(a)
```

where lower is better, \(C(a)\) is observation/action cost, and \(R_{safety}(a)\) is a separately measured risk term. Pure model EIG and targeted decision EIG remain comparator policies and diagnostics, not substitutes for this utility calculation. This is a benchmark decision rule, not a claim that the equation alone is a complete expected-free-energy implementation. The full policy must declare preferences, horizon, model uncertainty, likelihood provenance, cost units, and action constraints.


### Executable synthetic policy result

The branch now includes `crates/core/symthaea-fep/examples/hlb_decision_diagnostic_benchmark.rs`, a standalone 10,000-episode runner for the five fixed policies: no-probe, pure model-EIG, targeted decision-EIG, cost-aware EVSI, and always-inspect. It records the exact synthetic scenario, loss table, probe costs, safety limits, seed, source SHA, policy aggregates, and four paired EVSI-versus-baseline contrasts. The contrast artifact now carries both descriptive paired normal-approximation intervals and conservative finite-sample two-sided Hoeffding intervals for task loss, decision regret, and total cost. Task-loss/regret differences are bounded in [-1, 1]; total-cost differences use the scenario-derived bound 1 + max(weighted probe cost) + max(weighted safety penalty). The normal interval is descriptive only; the Hoeffding interval is distribution-free for independent paired episodes under the declared bound. It also carries simultaneous family-wise 95% Hoeffding intervals with Bonferroni alpha 0.05/12 across four reference-policy contrasts and three co-reported outcomes. This union-bound adjustment does not require independence among the 12 claims; it does require the paired episode observations underlying each interval to satisfy the declared independence/range assumptions.

The HLB workflow executes this runner on the exact checked-out PR head and validates a JSON schema/source binding before uploading `research-results/hlb-decision-diagnostic.json` and including it in the SHA-256 receipt. The output carries the status `synthetic_fixture_result_not_environment_result`: it is a reproducible controlled counterexample, **not** a production `ActiveInferenceAgent` result, trained-RL comparison, or field/deployment evaluation. The factorial profiles remain unexecuted until their adapters and task suites are implemented.

**FEP acceptance condition:** an action must be preferred because its prospective observation distributions distinguish plausible hypotheses, and the selected observation must improve posterior calibration, realized diagnosis, or downstream task utility on held-out episodes. Novelty alone is not information gain.

### 2.2 WSP is not the same as weight-space learning

Keep these terms distinct:

- **Weight-space programming (WSP):** an agent inspects a specific model and task, reasons about its parameter structure, and proposes direct parameter edits.
- **Weight-space learning (WSL):** learns representations, invariants, geometries, or generative distributions over families of parameterized models.
- **Gradient / local learning:** optimizes parameters through an explicitly defined objective and credit path.
- **Mechanistic interpretation:** proposes explanations of internal computational features or circuits; causal intervention is required to establish that an explanation matters.

The recent MIT CSAIL WSP report is an early research blog post, not a peer-reviewed benchmark. It demonstrates promising small-policy diagnosis/repair and a narrow large-model editing result, but also reports negative cases where edits did not create a missing capability. A separate early audit of small clinical decision networks found false alarms on healthy networks and that protocol/black-box tests could outperform the editor on some flaws. The right lesson is to evaluate proposal quality, healthy-model false edits, and collateral damage—not to treat parameter editing as an oracle.

SWI-001 already supplies the important safety foundations: typed parameter manifests, exact model fingerprints, transactional digest-bound edits, parameter constraints, and rejection of malformed updates. This programme adds the *proposal and verification protocol* around that boundary.

### 2.3 FEP should reason over three kinds of uncertainty

The strongest current research direction is not just action-dependent mutual information over one flat set of model IDs. Friston et al. (2026) distinguish uncertainty over **latent states**, **generative-model parameters**, and **generative-model structure**. Their active-reasoning work adds model-structure information gain to expected free energy and uses Bayesian Model Reduction (BMR) to compare reduced models efficiently. In their three-ball rule-discovery task, the state-, parameter-, and model-level information terms play complementary roles; removing model information can leave plausible hypotheses unresolved.

Research source: Friston et al., *Active inference and artificial reasoning*, Nature Communications 17, 10125 (2026), https://doi.org/10.1038/s41467-026-77209-5.

For Symthaea, keep these terms separately observable:

- **State information:** will this observation disambiguate the current hidden state?
- **Parameter information:** will it improve beliefs over dynamics, likelihoods, transition rates, or other generative parameters?
- **Structure information:** will it distinguish competing causal, transition, or computational models?
- **Pragmatic value:** does the predicted outcome satisfy declared preferences?
- **Cost and risk:** what is the observation/action burden, feasibility, and safety exposure?

A useful decomposition, when the model's conditional structure is explicit, is the mutual-information chain rule:

```text
I(M, Θ, S; O | a)
  = I(M; O | a)
  + I(Θ; O | M, a)
  + I(S; O | M, Θ, a)
```

Here M is model structure, Θ its parameters, S latent state, O the prospective observation, and a a candidate information action. This is a decomposition of a joint information quantity; it is **not** permission to add arbitrary entropy deltas that double-count uncertainty. The adapter must declare its factorization, posterior approximation, and which terms it can calculate. Unavailable terms remain unavailable rather than being replaced by novelty.

The current Symthaea model-information-gain primitive computes `I(Model; Outcome | action)` from supplied hypothesis priors and action-conditioned outcome distributions. The branch now also contains a separate diagonal-Gaussian BMR primitive for prior-only reductions under a shared parameterization. Together, these provide research primitives for prospective model-structure information and post-observation model evidence comparison; they do not yet implement a learned model-family bridge, state/parameter information-gain adapters, or a complete expected-free-energy policy. Production integration still requires provenance-bound predictive and posterior contracts.

**Proposed research shift:** implement a hierarchical *epistemic controller* with explicit fast state inference, slower parameter learning, and slower still model-structure selection. Use the structure-level information term to choose discriminating probes; use evidence thresholds and model evidence to decide whether to retain, reduce, or replace a model. A probe ranking does not automatically authorize model replacement.

### 2.3.1 Bayesian Model Reduction as a separate model-selection operator

The branch now contains an isolated diagonal-Gaussian Bayesian Model Reduction primitive in `crates/core/symthaea-fep/src/bayesian_model_reduction.rs`.

Given an approximate or exact posterior under a full model, a full-model Gaussian prior, and reduced models that differ only by Gaussian parameter priors over the same parameterization, it computes:

```text
log p(data | reduced model) / p(data | full model)
  = log E_q(theta | data, full)[p(theta | reduced) / p(theta | full)]
```

The implementation currently supports independent diagonal-Gaussian coordinates. Each posterior/prior object carries a 64-character lowercase hexadecimal 256-bit `parameterization_digest` (e.g. SHA-256 or BLAKE3); the full posterior, full prior, and every reduced prior must match exactly. The digest is required to bind a canonical ordered parameterization (coordinate names, roles, shapes, and order), so equal vector lengths cannot silently compare different parameter spaces. The implementation uses the analytic Gaussian integral, rejects non-integrable ratios, malformed dimensions/variances, invalid digests, mismatched parameterization digests, and unnormalized model priors, then returns posterior model probabilities plus model-posterior entropy. Tests include an independent one-dimensional quadrature reference, additive independent-coordinate checks, parameter-order invariance under a separately consistent coordinate digest, zero-prior behavior, and same-width/different-coordinate negative controls.

**Scope matters:**
- With an exact full-model posterior and nested prior-only reductions, this is an exact marginal-likelihood ratio.
- With a variational Gaussian posterior, it is a variational BMR approximation.
- It is not a general comparison between arbitrary architectures with different parameterizations.
- It is posterior model selection after evidence, not prospective expected information gain.
- It does not choose observations, include preferences/cost/risk, or authorize actions.
- It is not connected to the production FEP selector.

The intended combination is: prospective action selection evaluates the expected information in possible outcomes; after observing an outcome, the posterior is updated and BMR can compare prior-restricted model structures. Do not add BMR log evidence ratios to mutual information as though they were independent reward terms. They operate at different stages and answer different questions.

The branch now provides `canonical_parameterization_digest` and a versioned `ParameterizationDescriptor` over architecture identity, ordered parameter-group IDs/roles, tensor shapes, and every flattened coordinate name. It uses domain-separated, length-prefixed SHA-256 encoding and rejects duplicate IDs/coordinates, zero or overflowing shapes, and coordinate-count mismatches. Native CfC/HDC adapters still need to derive these descriptors directly from their canonical manifests rather than asking callers to supply them manually. Future work should add correlated Gaussian priors/posteriors and numerical stability comparisons against Cholesky-based reference calculations. Only then should the operator become part of a production FEP model-structure adapter.

### 2.3.2 Dirichlet BMR for categorical active-inference models

The diagonal-Gaussian BMR primitive above is appropriate only when the reduced hypotheses are Gaussian-prior restrictions over one shared continuous parameterization. It is **not** the correct conjugate BMR primitive for categorical likelihood tensors such as the three-ball task.

A separate implementation now exists in `crates/core/symthaea-fep/src/dirichlet_model_reduction.rs`. For full prior concentrations \\(a_0\\), full posterior concentrations \\(a\\), and reduced prior concentrations \\(a_m^0\\), the reduced posterior is

\\[
a_m = a + a_m^0 - a_0
\\]

For independent categorical likelihood columns, the log evidence ratio is

\\[
\\log \\frac{p(D\\mid m)}{p(D\\mid M)}
= \\sum_j
\\left[
\\log B(a_{0,j}) + \\log B(a_{m,j})
- \\log B(a_j) - \\log B(a^0_{m,j})
\\right]
\\]

where \\(B\\) is the multivariate beta function. The implementation accumulates these log ratios across columns, incorporates prior model probabilities in log space, and returns posterior model probabilities, entropy, and winner-to-runner-up log odds.

**Contract and boundary:**
- Full posterior, full prior, and every reduced prior must carry the same 64-character parameterization digest, binding the ordered likelihood columns, factor dimensions, outcome order, and coordinates.
- Full-model prior and posterior concentrations must be finite and strictly positive. Reduced priors carry an explicit per-column `active_support` mask: active coordinates require positive concentrations; inactive coordinates require exactly zero concentration and define a proper lower-dimensional simplex face. If posterior evidence contains any count in an excluded coordinate, that reduced model receives exactly zero marginal likelihood. Empty support, implicit zeros, negative values, and non-integrable reduced posteriors are rejected.
- Reduced posterior concentrations must remain strictly positive. If the data accumulated under the full model make a proposed reduction non-integrable, the candidate is rejected.
- Independent columns factorize exactly. Correlated or hierarchical Dirichlet structures are outside the present implementation.
- The Gaussian and Dirichlet BMR modules are complementary primitives, not interchangeable approximations.

Tests include the exact Beta-Binomial evidence ratio for a Beta(1,1) full prior and Beta(2,1) reduced prior (reduced/full evidence ratio \\(4/3\\)), explicit support-restricted evidence when excluded outcomes are unobserved, zero evidence when an excluded outcome is observed, additivity across independent likelihood columns, model-prior odds, ordered-parameterization binding, and negative controls for malformed support and incompatible reduced posteriors.

**Two distinct Dirichlet semantics are now implemented and named:**

1. `reduce_dirichlet_models` is the strict-support variant described above. An outcome excluded by a reduced model has exactly zero probability; observing it gives that model \\(-\\infty\\) log evidence.
2. `reduce_dirichlet_models_spm_compatible` reproduces the pinned SPM `spm_MDP_log_evidence` convention. It adds \\(p=1/32\\) to every coordinate of the posterior, full prior, and reduced prior, forms \\(s=q+r-p+p_0\\) after smoothing (equivalently \\(q'+r'-p'\\)), and computes the negative of SPM's free-energy expression. Previously zero coordinates are therefore smoothed, not treated as impossible. The matching model-average prior is the **globally normalized** weighted sum of reduced priors, not a per-column arithmetic mean. `evaluate_dirichlet_model_information_gain_spm_compatible` uses this same BMR semantics both before and after each hypothetical observation.

The distinction is anchored to the pinned sources: [`spm_MDP_BMR.m`](https://github.com/spm/spm/blob/28d1f103b021f729266cb7ea07baaf4d0285b22a/toolbox/DEM/spm_MDP_BMR.m) globally normalizes the sum of reduced priors; [`spm_MDP_log_evidence.m`](https://github.com/spm/spm/blob/28d1f103b021f729266cb7ea07baaf4d0285b22a/toolbox/DEM/spm_MDP_log_evidence.m) adds 1/32 to all three tensors; and [`spm_betaln.m`](https://github.com/spm/spm/blob/28d1f103b021f729266cb7ea07baaf4d0285b22a/spm_betaln.m) applies the multivariate beta function down the first (outcome) dimension independently for each remaining tensor column. These are not cosmetic implementation choices; they change reduced-model evidence and posterior odds.

The three-ball exact-source path now carries both the older strict-support prior and `spm_model_average_prior`, which matches `DEM_reasoning.m`'s global normalization. It uses the SPM-compatible BMR/EIG functions through the separate `*_spm_compatible` prediction and action-scoring entry points. The strict-support variant remains useful as an explicit counterfactual, but it is not the source reproduction mode.


This is the appropriate categorical BMR substrate for the [2026 three-ball active structure-learning paper](https://doi.org/10.1038/s41467-026-77209-5), whose likelihood parameters are represented by Dirichlet counts. A source-pinned reconstruction of the public 81-rule generator and a one-step model-EIG scorer now exist; the exact paper's reported 79-unique-rule count remains unreconciled with the audited public generator's 75 exact likelihood classes. The branch now tests prospective feedback model-EIG and post-outcome class posteriors under the 81-label and 75-class representations for several choices at a known ball state. This is a feedback-modality quotient-invariance fixture, not the complete three-ball task: the full POMDP, visual+feedback action-conditioned state posterior, exact archived rule-family reconciliation, full expected-free-energy policy evaluator, and production `ActiveInferenceAgent::select_action` integration are not yet implemented.

### 2.3.3 One-step prospective model EIG from Dirichlet BMR

The categorical BMR module now exposes both `evaluate_dirichlet_model_information_gain` (strict-support semantics) and `evaluate_dirichlet_model_information_gain_spm_compatible` (pinned SPM semantics). The latter is the relevant source-reproduction path for three-ball. Each evaluator implements the model-structure component of the paper's one-step expected information gain:

```text
IG_model(action)
  = H(Q(m | current evidence))
    - sum_o Q(o | action, current full model)
             * H(Q(m | current evidence, hypothetical outcome o))
```

For each possible outcome, the caller supplies:
- The posterior-predictive probability under each reduced model, keyed by exact model ID.
- The aggregate full / Bayesian-model-average predictive probability, which is independently reconstructed and checked.
- The expected sufficient-statistic increment for the Dirichlet likelihood tensor (typically outcome-modality outer-product posterior expectations over the parent-state factors).

The evaluator obtains the **current posterior over model identity** from BMR, then derives:

```text
Q(o | a, D) = sum_m Q(m | D) P(o | m, a, D)
```

It rejects missing, duplicate, or unknown model IDs; invalid per-model probabilities; any per-model outcome distribution that does not sum to one; and any supplied aggregate that disagrees with the posterior-weighted mixture. This matters after evidence has changed model odds: using the original model priors instead of the current posterior can produce the wrong action ranking even when every individual probability is normalized.

The evaluator adds the hypothetical sufficient-statistic increment to the current full posterior, runs BMR to obtain the posterior over reduced models for that outcome, and averages posterior model entropy under the reconstructed posterior-predictive mixture. It does **not** multiply the outcome probability by another Bayes-factor mixture term; that would double-count evidence and would not match the model-EIG expression in the paper.

The evaluator enforces:
- In strict-support mode, the full Dirichlet prior is the per-column model-prior-weighted arithmetic average of the reduced-model priors.
- In SPM-compatible mode, the full Dirichlet prior must be the globally normalized weighted sum of all reduced-prior tensor entries, matching `spm_MDP_BMR` and `DEM_reasoning`; it is not a per-column mean.
- Each model-specific action-outcome predictive distribution is finite, non-negative, keyed to the current model set, and sums to one.
- The caller-supplied aggregate outcome distribution matches the mixture weighted by the **current posterior model probabilities**.
- Every hypothetical posterior-count increment matches the exact likelihood tensor dimensions and is finite/non-negative.
- Current and post-outcome model beliefs are derived from the same parameterization and support-bound BMR contract.

**Important current limitation:** the count increment is presently one shared expected sufficient-statistic increment for the nominated parent-state posterior, not a separate state-posterior/count update for each candidate model. This is exact for the registered known-ball-state feedback fixture, where the parent state is fixed and the feedback likelihood column is known. In the full partially observed three-ball task, gaze and choice actions change beliefs over hidden ball colours; a faithful extension must compute action/outcome-conditioned posterior expectations over parent states, bind their provenance, and establish whether the increment is under the full Bayesian-model-average parent or a model-specific parent for each BMR comparison. Do not label the current feedback-only known-state test a full POMDP reproduction.

A deterministic Beta/Bernoulli unit test reproduces the analytic two-model case: equal-prior Beta(4,1) and Beta(1,4) hypotheses have parent Beta(2.5,2.5); a success moves the posterior model belief to 0.8/0.2, a failure to 0.2/0.8, and the expected model EIG is approximately 0.1927448 nats. An uninformative outcome with no sufficient-statistic increment yields zero model EIG. A malformed parent prior and non-normalized predictive mass are rejected.

**Scope boundary:** this is a one-step model-EIG scorer, not a complete expected-free-energy planner. The caller must still produce calibrated action-conditioned predictive outcomes and state/posterior increments. Parameter EIG, state EIG, risk/preference cost, ambiguity, action costs, multi-step planning, the full three-ball POMDP, and production `ActiveInferenceAgent::select_action` integration remain separate work. The full expected-free-energy score should add the components once, with consistent sign conventions; do not double-count BMR log evidence as a separate reward term.

### 2.3.4 A scoped known-state feedback decision score

The branch now adds an executable one-step action evaluator in `crates/core/symthaea-fep/src/three_ball_rule_space.rs`:

- `build_three_ball_feedback_action_prediction` derives each reduced model's posterior predictive categorical distribution from the full posterior and reduced prior using (a_m = a + a_m^0 - a_0). It weights these distributions by the current posterior model belief, not the original model priors.
- `score_three_ball_known_state_feedback_actions` evaluates legal colour choices plus no-choice for a fixed known ball-colour state and fixed gaze.
- `ThreeBallFeedbackDecisionConfig::default()` uses the paper's relative log preferences `[0, 2, -6]` nats in `[no-feedback, correct, incorrect]` order, normalized to a categorical preference distribution, and an explicit epistemic weight of one.
- `ThreeBallFeedbackActionScore` returns the pragmatic-risk KL, model-structure information gain, weighted epistemic value, action cost, predictive feedback distribution, and combined score for each action. The decision chooses the minimum score with deterministic tie-breaking.

The scoped objective is:

```text
G(a) = KL[Q(feedback | a, D) || C(feedback)]
       - lambda_info * I(model; feedback | a, D)
       + action_cost(a)
```

This is a **known-state feedback-only, one-step FEP-style decision slice**. It is not the full expected-free-energy functional: state uncertainty/visual information gain, ambiguity, gaze/vision transitions, multi-step planning, and safety constraints are not present. It is not wired to production `ActiveInferenceAgent::select_action`. The earlier shared count-increment limitation still applies: it is exact for the fixed known parent state, not a full partially observed parent-state posterior update.

Tests now check the decomposition, normalized predictive feedback, the no-choice zero-information control, rejection of negative action cost and invalid epistemic weight, and a posterior-evidence example in which one observed correct feedback removes rules incompatible with that event and a pragmatic-only policy selects the planted green choice. The full EIG-plus-pragmatics trade-off remains a benchmark to compare across `lambda_info`, action costs, and matched random/pragmatic-only/model-EIG-only controls—not a demonstrated advantage over other policies yet.

### 2.4 WSP must pass exact-protocol replication, not headline comparison

An important reproducibility update appeared on 6 November 2026. The independent [OpenWSP replication](https://github.com/openwsp/openwsp) reports that its recorded live Experiment 8 edit scored **12.9% on MATH-500**, versus the MIT authors' reported 48.1% and OpenWSP's local calibration of the released edit at 50.6%. OpenWSP reports that 84.25% of its live MATH-500 generations hit the token limit, and that its live evaluation used four samples per problem while the local calibration used one. GSM8K and IFEval were still pending in that repository snapshot.

This is not a controlled refutation of the MIT result: the live run and local calibration use different evaluation multiplicity and possibly other execution differences. It is, however, strong evidence that raw headline scores are not portable without protocol identity. A direct WSP baseline can vary with prompts, editor model/version, tool permissions, number of passes, forward-pass opportunities, token limits, sampling, evaluation multiplicity, and the checkpoint revision.

For every WSP result, bind and report:

1. Exact base checkpoint revision and tensor digests.
2. Editor model/version, prompt and system-instruction digests, context budget, tools, and sandbox restrictions.
3. Number of edit rounds, forward-pass opportunities, tool calls, candidate count, and token/run budget.
4. Exact evaluation code/version, test split, sampling count, pass@k/aggregation rule, decoding settings, and token cutoff.
5. Local calibration results and live/independent replication results as separate rows.
6. Independent live replications, variance and confidence intervals; never select only the best run.
7. Every intermediate and final weight diff, rejected candidate, and rollback.

The SWI benchmark must reproduce the published WSP protocol where it can, then compare it with SWI-constrained typed edits under equal budgets. The original MIT result is a motivating baseline, not an assumed performance floor or a verified guarantee.

### 2.5 Weight-space reasoning must account for parameter symmetries

A raw parameter vector is not a unique description of a learned function. Hidden-unit permutations can produce different serialized weight arrays with exactly equivalent behavior when the corresponding incoming, outgoing, bias, and state-associated parameters are permuted consistently. Similar equivalences can arise from architecture-specific rescalings or other reparameterizations.

Consequences for SWI:

- A large parameter-space distance does not necessarily imply a large behavioral change.
- A small sparse edit can have a large behavioral effect.
- Parameter index similarity across two checkpoints is not semantic correspondence unless architecture and neuron/group alignment are established.
- A WSP editor that interprets raw indices without alignment can mistake an equivalent healthy model for a faulty one.

Add a **function-space and symmetry-aware** lane alongside raw weight introspection:

1. Compute exact architecture/manifest/model digests as today.
2. Collect a small, frozen set of functional probes (input/output and, where permitted, internal activation/temporal traces).
3. Detect allowed parameter symmetries for the nominated architecture; start with exact hidden-unit permutation equivalence for small CfC networks.
4. Align candidate groups using behavior/activation correspondence, then diagnose/edit the aligned typed groups.
5. Validate the proposal in both parameter space and function space. Never use a symmetry alignment to rewrite the source checkpoint silently.

The WeightProgram IR should remain tied to the exact source model digest. An aligned view is diagnostic metadata, not a new authority to mutate a different checkpoint.

**Executable alignment primitive now present:** `src/dynamics/cfc/weight_program_adapter.rs` exposes `cfc_hidden_unit_activation_signatures` and `align_cfc_hidden_unit_signatures`. It runs a frozen sequence of functional probes on a detached network clone, records per-unit activation signatures for a selected layer, and returns a serializable signature set bound to the native model fingerprint, layer index, and BLAKE3 digest of the ordered probe sequence. The aligner requires source/candidate probe and layer identities to match, validates both model/probe digests, and uses a deterministic Hungarian assignment to map source units to candidate units by mean-squared signature distance. The serializable alignment receipt binds the source and candidate model fingerprints and the probe digest. It also reports the minimum cost increase of any alternative assignment (computed through alternating-cycle shortest paths in O(n³)); `None` means no alternative complete assignment exists, while a zero or near-zero numeric margin means competing assignments are indistinguishable. Consumers must abstain below a threshold calibrated on development data. Malformed/non-finite probes and signatures fail closed.

The primitive does not reorder parameters, rewrite native fingerprints, change WeightProgram target identities, or authorize edits. Current tests cover digest-bound signature receipts, known permutation recovery, ambiguous duplicate signatures, malformed receipts, Hungarian/alternating-cycle margin comparison against exhaustive assignments, deterministic probe selection, and candidate-only canonicalization. A multi-layer feed-forward canonicalizer is implemented for stacked CfC networks without residual connections, bidirectionality, or backbone paths. Residual, bidirectional, and backbone topologies remain fail-closed and unsupported. This is an architecture-specific symmetry tool, not a general weight-space solver.

This creates a distinctive benchmark: can SWI distinguish **parameter reordering with identical behavior** from a real fault, and can a repair learned in one parameterization transfer to a functionally equivalent permuted network?

## 3. Proposed architecture: FEP chooses probes; WSP writes candidates

### 3.1 Typed WeightProgram intermediate representation

Do not start with unrestricted raw floating-point writes. WSP should emit a serializable, reviewable intermediate representation (IR), conceptually:

```text
WeightProgram {
  schema_version,
  architecture_fingerprint,
  manifest_fingerprint,
  base_parameter_digest,
  target_specification_digest,
  fault_hypothesis_ids[],
  evidence_receipt_ids[],
  operations[],
  declared_budget,
  expected_effects[],
  preservation_constraints[],
  rationale
}
```

Every operation identifies a typed parameter group and a bounded transformation. Initial operation families:

1. `ScaleSubspace`: scale a declared, bounded coordinate/subspace range.
2. `AddLowRankDelta`: add a rank-limited factorized update with a norm budget.
3. `SetTimeConstantRange`: propose bounded positive tau changes that respect the cell's tau floor and configured ceiling.
4. `ReplaceRoleVector`: replace a typed HDC role vector only if dimensions, normalization, orthogonality/interference limits, and algebra contract pass.
5. `ProjectOutDirection`: remove a diagnosed nuisance direction only when the target subspace and protected subspaces are identified and independently tested.

These are candidate operations, not a new public API promise. The first implementation should support fewer operations rather than silently falling back to arbitrary writes. Every operation must declare preconditions, expected tensor shape, maximum norm/delta, and reversible before/after values.

A program is rejected before mutation when its architecture/manifest/base digest is stale, a group is duplicated without an explicit composition rule, a value is non-finite, a shape or range is invalid, an invariant is broken, or the declared budget is exceeded.

### 3.2 Candidate lifecycle

1. Freeze a baseline model, manifest, task revision, and split.
2. Produce a diagnostic belief from allowed evidence; keep hidden fault labels outside the agent.
3. Choose a diagnostic probe using prospective expected information gain and declared costs.
4. Update beliefs from the observed probe result; record calibration and evidence provenance.
5. Let WSP propose one or more typed programs against the unchanged baseline digest.
6. Apply each candidate to an isolated clone only after SWI preflight checks.
7. Evaluate against a fixed budget of development episodes and explicit preservation tests.
8. Ask an independent verifier to assess target repair, collateral behavior, calibration, and safety.
9. Accept only a passing candidate; retain original, candidate, operation log, receipts, and rollback pointer.
10. Unlock held-out tests once for final evaluation. Holdout outcomes must never feed the editor or candidate selection.

No online self-editing of the live cognitive loop is in scope. No proposal may directly modify the production model. WSP is initially a tool-assisted research workflow with explicit review.

### 3.3 FEP–WSP coupling hypotheses

- **H1 — diagnostic efficiency:** active information-seeking probes reduce observations/forward passes needed to identify planted faults versus fixed, greedy, random, and bounded Bayesian baselines.
- **H2 — repair quality:** WSP proposals conditioned on diagnostic evidence repair more injected faults than weights-only proposals under equal candidate and evaluation budgets.
- **H3 — safety:** SWI constraints plus an independent verifier reduce harmful edits and false edits to healthy networks versus unconstrained WSP or random matched edits.
- **H4 — complementary value:** FEP + WSP improves a joint cost/repair/retention frontier beyond either component alone.
- **H5 — representational utility:** HDC/HRR memory improves retrieval and composition of evidence/program fragments under corruption or limited storage, without assuming it improves parameter editing itself.

All five are falsifiable. A null result is a valid outcome; do not change the task after seeing held-out results.

### 3.4 Initial executable slice on the research branch

The current branch now contains a deliberately limited, candidate-only implementation:

- `crates/core/symthaea-core/src/weight_space_program.rs` defines typed CfC/HDC-LTC target identities, `ReplaceGroup`, `AddDelta`, and `ScaleRange` operations, immutable model/group snapshots, target-specification and model-digest preconditions, per-group/global edit budgets, parameter constraints, deterministic program/candidate content digests, and a validation receipt.
- `src/dynamics/cfc/weight_program_adapter.rs` adapts the native CfC fingerprint and parameter manifest into a snapshot, maps typed groups back to native CfC identifiers, and materializes the proposal only on a cloned network using `try_apply_parameter_group_replacements`. It recomputes the candidate's native fingerprint after the transactional edit.
- `crates/core/symthaea-core/src/hdc/weight_program_adapter.rs` adapts a unified HDC-LTC neuron's complete `UnifiedConfig`, five typed hypervector groups, and exact vector bytes into the same generic snapshot. It uses the native per-group edit validator on a cloned neuron, resets candidate hidden state, time/update counters, momentum and running statistics via `reset_after_parameter_edit()`, and recomputes its native digest.
- Unit tests cover detached candidate creation, stale model/task rejection, role and tau constraints, duplicate-target rejection, per-group/global budgets, deterministic candidate digests, and a planted 3× tau fault repaired on an isolated clone. The planted-fault test also checks that the same proposal is rejected against the healthy model's different base digest.
- Additional HDC-LTC adapter tests cover detached neuron edits, state reset on the candidate only, norm-bound rejection, and stale task-specification rejection. The CfC adapter's planted-fault test now seeds a nonzero recurrent state and stale Adam counters, then checks that only the candidate resets them and that repaired outputs match the healthy model on a fixed held-out input sequence.

This is a real CfC candidate adapter, but it is **not** an autonomous WSP proposer, a production FEP integration, or a promotion authority. HDC-LTC is represented in the typed IR but does not yet have a live-model adapter. The new Rust tests and formatting checks must pass on the exact branch head before implementation qualification; queued workflows are not passes.

### 3.5 Function-space quotienting and epistemic alignment

A neural parameter vector is not a unique description of the function it computes. For supported feed-forward CfC topologies, hidden-unit relabellings act on the parameter tensors without changing the input/output function. Let \(G=S_h^L\) be the product of hidden-layer permutation groups for a network with \(L\) layers of width \(h\). The relevant model is then not an isolated parameter point \(\theta\), but an equivalence class:

\`\`\`text
[theta] = { g · theta : g in G }
\`\`\`

This changes the learning problem. Raw Euclidean distance between two parameter vectors is not a reliable measure of functional distance; coordinate permutations can make it large while preserving the function exactly. Conversely, similar weights do not prove equivalent behavior. Use three distinct quantities:

- **Parameter distance:** distance between a chosen pair of coordinate representatives.
- **Functional distance:** discrepancy between outputs or internal activations over a frozen, explicitly declared probe distribution.
- **Alignment uncertainty:** uncertainty over the group element that relates the two parameterizations.

The CfC alignment/canonicalization code implements a deliberately bounded first slice. It measures source-bound activation signatures, solves a one-to-one Hungarian assignment, reports an alternating-cycle assignment margin, and can materialize a canonicalizing \`WeightProgram\` on a candidate clone. The single-layer transform is restricted to no-backbone, no-residual, unidirectional CfC. The multi-layer transform additionally permutes adjacent input-projection columns and is restricted to feed-forward/no-residual/no-backbone/no-bidirectional topologies. Other graph structures must be rejected until their complete symmetry action is derived.

**New paradigm to test: FEP over representation correspondence.** Treat the hidden-unit map \(\Pi\) as an epistemic latent variable. Given frozen evidence \(D\), a candidate functional probe \(a\), and measured signature \(Y\), choose a probe by:

\`\`\`text
a* = argmax_a I(Pi; Y | a, D) - lambda_cost * probe_cost(a)
\`\`\`

For a finite small-width research fixture, one possible posterior is:

\`\`\`text
q(pi | D) proportional to exp(- L_probe(pi; D) / (2 sigma^2))
L_probe(pi; D) = sum_(unit,probe) (source[unit,probe] - candidate[pi(unit),probe])^2
\`\`\`

This is a *candidate observation model*, not an empirical noise estimate. `sigma` must be calibrated from repeated probe executions, hardware/precision effects, stochastic layers, or a separately justified observation-noise protocol. Do not manufacture epistemic uncertainty by adding arbitrary noise solely to make EIG nonzero. If execution is deterministic and activations are fully observable, use the deterministic margin objective or first establish a defensible measurement-noise model.

**Executable FEP probe selector now exists on the research branch.** `crates/core/symthaea-fep/src/gaussian_model_information_gain.rs` computes scalar Gaussian-channel mutual information (I(H;Y\mid a)) by deterministic composite-Simpson quadrature and log-sum-exp. `src/dynamics/cfc/weight_program_adapter.rs::select_cfc_disambiguating_probe_by_gaussian_eig` enumerates exact hidden-unit permutations, forms a deterministic BLAKE3-domain-separated projection over the target layer, updates the permutation posterior from existing source/candidate activation probes under the caller-specified Gaussian noise, then scores new input probes by prospective information gain. The receipt binds source/candidate model digests, architecture and manifest digests, projection digest, probe digests, posterior over permutation mappings, quadrature settings, and every candidate action score.

The exact enumerator is intentionally bounded to **120 mappings (up to five hidden units)** and **32 candidate probes** per selection call. It checks the factorial count before allocating permutations. Wider layers continue to use the Hungarian assignment and ambiguity-margin path; scaling to Sinkhorn/soft assignments or amortized permutation-equivariant WSNs is separate work. The observation standard deviation is no longer a free caller scalar: `calibrate_cfc_projection_observation_noise` derives the unbiased sample standard deviation from repeated scalar-projection measurements and records the ordered samples, declared method, exact source/candidate model digests, architecture/manifest, layer, deterministic projection, calibration-probe digest, and receipt digest. The EIG selector recomputes the sample statistics and digest and rejects a calibration bound to any other model pair/projection. Zero-variance and degenerate calibrations are rejected. The seeded unit fixture uses synthetic repeated samples only to test the contract; it does not establish an empirical production noise calibration.

Compare random probe selection, deterministic accumulated-margin gain, minimum functional-fit error, calibrated exact Gaussian permutation EIG (small widths), Sinkhorn/soft-assignment approximations (larger widths), Deep-Align, and permutation-equivariant WSN baselines. Include a no-noise/deterministic control, a calibrated-noise control, and a deliberately misspecified-noise condition. Report posterior calibration and abstention as first-class outcomes. A probe is useful only if its selected alignment generalizes to a disjoint held-out probe sequence.

Primary metrics: probes to a unique/accepted alignment, permutation posterior Brier score, held-out mapping accuracy, held-out functional equivalence, ambiguity abstention precision/recall, realized entropy reduction, expected information gain, noise-calibration error, and probe cost. Add scaling curves over width (4, 5 exact; 8, 16, 32, 64 for approximate methods). Exact permutation enumeration is a small-width oracle/control only; it is factorial in width and must not be presented as a scalable algorithm. The returned EIG scores information about the permutation hypothesis only: preferences, action/probe cost, risk, and multi-step policy planning remain separate terms in a complete expected-free-energy controller.
**Safety invariant:** alignment metadata never authorizes a live-model edit. The source/candidate native digests, architecture/manifest digests, probe digest, layer ID, bijection, fit error, and ambiguity margin must all be checked again when constructing a candidate-only WSP program. A stale model or changed probe sequence invalidates the alignment. Independent held-out functional verification remains mandatory after canonicalization.

## 4. Four-factor factorial study

Add a new 2×2×2×2 factorial, separate from HLB factorial v1 so that the existing three-factor protocol remains stable:

- H: native HDC/HRR representation and associative operations vs fixed-seed random projection / conventional representation control.
- C: nominated CfC/LTC temporal substrate vs parameter-matched GRU control (with memoryless and leaky-state controls on temporal tasks).
- F: action-dependent, prospective FEP information-seeking policy vs pragmatic-greedy control with identical observation/world-model interfaces.
- W: constrained WSP proposal generation vs no-edit / conventional repair baseline. The W-disabled condition must not silently remove SWI safety checks from the evaluation harness.

Sixteen profiles are required exactly once. Every profile records the actual adapter that ran; a config label is not proof that a factor was activated or disabled.

The machine-readable contract is `docs/research/hlb-fep-wsp-factorial-v1.json`. It is protocol-only. Runtime adapters and experiment receipts are future work.

## 5. Benchmark tasks

### A. Information-selective diagnosis
A seeded partially observable task has several plausible fault/model hypotheses and diagnostic probes with different outcome distributions, cost, and downstream impact. Some probes reveal discriminating signals; others are novel but uninformative. Include identical-outcome and misleading-model controls.

Primary outcomes: diagnosis accuracy, normalized regret, information acquisition cost, posterior calibration, number of probes, and realized information gain.

### B. Injected-fault repair with healthy controls

Create small, fully specified CfC and HDC-LTC networks. Inject known faults in controlled groups: input path, recurrent path, tau/gate, mixing operator, and HDC role vector. Include untouched healthy models. Publish the fault generator and ground truth to the verifier, not the candidate proposer.

Primary outcomes: fault detection precision/recall, fault localization precision, repair success, collateral regression, edit sparsity/norm, candidate evaluation cost, and healthy-model false-edit rate.

Add a symmetry negative-control cohort: generate behaviorally equivalent CfC networks by consistent hidden-unit permutations, without changing the represented function. Measure false fault declarations and false repairs on these equivalents. Add a transfer cohort where one repair is mapped between equivalent parameterizations through an explicit alignment procedure.

### C. Compositional and associative transfer
Use role–filler, nested relation, temporal order, and held-out composition tasks. Probe binding/unbinding, retrieval under crosstalk, role swaps, missing elements, and controlled corruption. Compare Hadamard/MAP, bipolar XOR, HRR, random projection, tensor-product small-scale oracle, and classical models where appropriate.

Primary outcomes: held-out compositional success, retrieval precision/recall, capacity as a function of dimension and load, false retrieval, calibration, and memory cost.

### D. Irregular-time dynamics and regime changes

Use fixed datasets and synthetic systems with nonuniform `\Delta t`, missing observations, input interpolation, and controlled regime changes. Report error by elapsed interval and wall-clock cost.

Primary outcomes: forecast error, rollout stability, time-to-recover after shifts, error against high-accuracy references, and end-to-end latency.

### E. Function-preserving parameter symmetries

Generate small CfC networks and exact hidden-unit permutations that preserve their input/output function when every coupled parameter group is permuted consistently. Pair these healthy equivalents with genuinely faulty models and held-out permutations.

Primary outcomes: functional-equivalence error, false fault/false edit rate on permutation-only controls, and repair transfer across parameterizations.

Diagnostics: alignment precision, activation/functional probe budget, weight-distance versus function-distance, changed parameter count, candidate evaluation cost, and preservation on independent input sequences.

Compare raw flattened-weight inspection, random functional probes, activation-probe alignment, greedy accumulated-margin probe selection, permutation-equivariant weight-space editors, SWI's typed group alignment, Deep-Align (when the exact architecture is supported), and a labelled oracle-alignment upper bound. The oracle is verifier-only. A permutation-only change must not be called a defect; a repair must still be evaluated using native model fingerprints and actual behavior.

**Executable anchor now present on this branch:** `src/dynamics/cfc/weight_program_adapter.rs::consistent_hidden_unit_permutation_preserves_cfc_function` materializes the equivalence-preserving permutation through the actual `WeightProgram → materialize_cfc_weight_program → cloned SWI candidate` path. It checks that the native fingerprint changes while a four-step temporal probe sequence remains equivalent. The paired partial-permutation negative control changes only the input-projection rows and must change the output sequence. The first fixture remains intentionally narrow: one small, one-layer, no-backbone CfC network. The branch now also has a multi-layer canonicalizer for feed-forward, non-residual, unidirectional CfC stacks without backbones. It consumes one fingerprint-bound alignment receipt per layer and propagates each permutation through that layer's input-projection rows, the adjacent layer's input-projection columns, both axes of the recurrent projection, hidden bias, time constants, and the final output projection. The new two-layer seeded regression test constructs two independent permutations, recovers each layer's inverse correspondence from functional probes, builds the typed WeightProgram, materializes it on a candidate clone, verifies that the canonicalized native fingerprint returns to the source fingerprint, and checks held-out sequence equivalence. Residual, bidirectional, and backbone configurations remain fail-closed and unsupported; this is not yet a general symmetry solver for arbitrary CfC topologies.

**Second executable increment: functional-signature alignment and adaptive probe design.** The adapter now builds model- and probe-digest-bound activation-signature receipts, aligns units with a deterministic Hungarian assignment, and reports an accumulated assignment margin against the best alternative. Receipts bind model digest, architecture digest, parameter-manifest digest, layer index, and ordered probe-set digest; source and candidate must share architecture, manifest, layer, and exact probe sequence. A fail-closed acceptance gate rejects malformed digests, non-bijective mappings, excessive activation error, and low/absent assignment margins. The margin uses *summed* probe evidence, while the fit metric remains mean-squared activation error; duplicating a consistent probe therefore increases confidence evidence without changing fit error.

The new `select_cfc_disambiguating_probe` evaluates a frozen candidate-probe pool and chooses the probe with the largest accumulated assignment margin (ties: lower mean-squared activation error, then original candidate index). It records all scores and changes neither model. This is a deterministic design heuristic—not Bayesian expected information gain. It is not FEP until a calibrated stochastic observation model and posterior over correspondence hypotheses are specified. The tests cover deterministic receipt generation, probe-order binding, architecture/manifest mismatch rejection, ambiguous-unit abstention, brute-force assignment-margin comparison, disjoint held-out probes, and deterministic candidate selection. The disambiguating-probe selection fixture itself is currently one-layer/no-backbone; the separate multi-layer canonicalizer has a two-layer seeded correctness fixture, while broader layer counts, stochastic CfC cells, and residual/backbone/bidirectional topologies require separate coverage.

**Third executable increment: symmetry-orbit canonicalization through WSP.** The one-layer adapter constructs a typed five-group WeightProgram from a digest-bound alignment. The stacked feed-forward adapter builds the corresponding per-layer replacements, including both row and previous-layer column permutations in each internal projection, recurrent rows/columns, biases, time constants, and final output columns. It refuses stale source/candidate fingerprints, mismatched architecture/manifest, malformed or ambiguous alignments, no-op transforms, and unsupported topologies. The program edits only a cloned candidate through the native SWI transaction boundary; it never copies source weights into the candidate. The new exact fixture checks that the canonical candidate recovers the source parameter fingerprint while preserving the permuted candidate's function, and that a stale correspondence cannot authorize a rewrite.

This is the beginning of a **quotient-space WSP** approach: regard parameter vectors related by known function-preserving symmetries as different coordinate descriptions of one functional model, and canonicalize that orbit before comparing, transferring, or repairing weights. It is not yet a general quotient construction. The supported group action covers one-layer or stacked feed-forward CfC networks with no residual connections, bidirectionality, or backbone paths. The multi-layer canonicalizer explicitly couples adjacent layer projections; independent per-layer permutations are not valid. Other architecture-specific symmetries (including residual-path, gated-backbone, bidirectional, scale, and neuron-splitting symmetries) remain unsupported. Exact-head Rust tests and held-out validation are still required before this capability can be called qualified.


### F. Partial-observation rule discovery: the three-ball reproduction target

Use Friston et al. (2026), [*Active inference and artificial reasoning*](https://doi.org/10.1038/s41467-026-77209-5), as a bounded model-structure learning benchmark. The exact task includes:

- Five latent factors: the colours of left, centre, and right balls (red/green/blue each); a controllable gaze location (start/left/centre/right); and a controllable choice (no-choice/red/blue/green).
- Two outcome modalities: vision (nothing/red/green/blue) and feedback (no-feedback/correct/incorrect).
- Gaze and choice are action-controlled; ball colours are sampled randomly at trial start. Each trial has six time points (five moves).
- Relative log preferences for the feedback modality are [0, 2, -6] nats in the order no-feedback/correct/incorrect.
- The planted context-dependent rule maps centre-ball red to the left ball, green to the centre ball, and blue to the right ball as the criterion for the correct colour choice.
- The paper reports 81 candidate mappings, 79 unique, with rule 65 as the planted ground truth. The pinned public SPM implementation was independently reconstructed in `crates/core/symthaea-fep/src/three_ball_rule_space.rs`: it yields 81 labelled hypotheses, but exact equality of the complete 432-column feedback likelihood tensors produces 75 semantic classes. The discrepancy is now analytically localized in the pinned `spm_MDP_rules.m` generator: there are three context-factor choices and three criterion-location choices for each of the three context colours (3 × 3³ = 81). When the criterion-location mapping is constant across all three context colours, the chosen context factor is irrelevant. There are exactly three such constant mappings (always choose left, always choose centre, always choose right), and each appears under all three context-factor labels. Thus 9 labelled hypotheses collapse into 3 observable likelihood tensors; the other 72 hypotheses remain distinct, giving 75 classes (6 duplicate labels). An executable invariant, `audits_public_generator_semantic_duplicates_without_forcing_paper_count` plus `public_generator_duplicates_are_exactly_the_three_constant_criterion_mappings`, verifies that all and only those three triplicate groups are duplicates. This proves the behavior of the audited public generator, but it does **not** reconcile with the paper's 79 count. Keep the 81 source labels and their equal 1/81 prior mass intact in the source-faithful track; do not silently deduplicate or relabel the family. The separate `semantic_unique_model_hypotheses()` quotient view aggregates the prior mass of the 75 exact-likelihood classes. Its tests compare the compressed mixture prior, post-evidence BMR posterior, and prospective feedback model-EIG/action ranking against the corresponding aggregated quantities from all 81 labelled hypotheses. The known-state feedback action fixture does not claim to reproduce the full visual observation or gaze-information policy. Until the paper's reported uniqueness count is reconciled against the exact archived source or a different equivalence criterion, call the implementation a source-pinned reconstruction/audit, not an exact numerical reproduction.
- Crucially, the generative model is assumed known except for the feedback likelihood tensor. The paper evaluates active selection among candidate rule structures; it does **not** solve perception-to-world-model construction or unconstrained ARC-style rule invention.

The paper reports model commitment when the winning model's log-odds crosses 16 nats in its illustrative setup. Treat this as a reproduction metric, not a universal confidence threshold. Report both time to threshold and false-commitment rate under alternative planted rules and prior misspecification.

Two tracks remain separate:

1. **Exact reproduction:** independently verify the state factors, observation/feedback likelihoods, transition structure, candidate model-generation algorithm, Dirichlet prior counts, action horizon, preference vector, and selection rule from the paper and supplement. Pin the source/supplement revision and all implementation choices.
2. **Task adaptation:** any change to factors, outcomes, ball count, rule family, preferences, horizon, likelihood model, or hypothesis-space constraints must be labelled an adaptation.

Primary outcomes: ground-truth rule posterior probability, rule-identification accuracy, trials to 16-nat model log-odds, model-choice Brier score, decision regret, and observation cost. Diagnostics: state/parameter/model-structure information gain, realized model-entropy reduction, Dirichlet BMR log evidence, posterior winner/runner-up odds, false commitment, and calibration under prior misspecification.

The categorical likelihood BMR baseline must use `reduce_dirichlet_models_spm_compatible` and `evaluate_dirichlet_model_information_gain_spm_compatible` for source-faithful three-ball reproduction. The strict-support functions are separate ablation baselines, not substitutes for the SPM evidence convention.

**Source-pinned model-space audit:** the implementation cites [the paper-pinned SPM demo revision](https://github.com/spm/spm/blob/28d1f103b021f729266cb7ea07baaf4d0285b22a/toolbox/DEM/DEM_reasoning.m), [the public rule generator revision](https://github.com/spm/spm/blob/5852cadb300a81fd3f30e7056ee329bca20f0efe/toolbox/DEM/spm_MDP_rules.m), and [the combination-order implementation](https://github.com/spm/spm/blob/main/spm_combinations.m). The audited SPM source lineage confirms 81 labelled hypotheses and rule 65's context/criterion mapping. Its complete likelihood tensors collapse to 75 equivalence classes under exact tensor equality. The paper's 79 count may reflect a different generator snapshot or a different notion of semantic equivalence; the current evidence does not resolve which. The exact-reproduction hard gate therefore remains closed pending a pinned-source reconciliation. Compare random legal observation, reward/pragmatic-greedy without model information, state-only and parameter-only information policies, hierarchical model EIG with Dirichlet BMR, and an analysis-only oracle. Keep preferences, observation cost, legal actions, model priors, and horizon identical across ablations.



**Explicit quotient experiment:** run the same diagnostic policies over both the 81-label family and the 75-class quotient. Before any outcome, verify that the prior-predictive feedback distributions are identical. After each possible and realized evidence increment, verify that the posterior probability of each quotient class equals the sum of posterior probabilities of its labelled members. Compare model-identity entropy only after accounting for within-class label entropy; otherwise the raw 81-label entropy can make label redundancy look like real uncertainty. Report model evidence, action ranking, and decision regret as invariant checks. The 75-class quotient is an ablation/optimization view, never the sole reproduction representation.

BMR is evaluated after evidence updates the full-model Dirichlet posterior. It must not be added directly to prospective expected information gain or pragmatic value. The state/parameter/model information terms remain separate and their chain rule must be checked.

### G. Causal weight reading under semantic interventions

A model's weights may contain recognizable statistical signatures without revealing the function it actually computes. Recent 2026 weight-reading experiments and independent audits show why familiar dataset priors, class order, and local weight statistics must be separated from causal understanding.

Ahmed's 5 November 2026 exploratory study reports that a weight-only LLM can infer local batch-normalization and output-layer patterns, but often overgeneralizes familiar dataset labels; after a class-label permutation, its interpretations follow memorized label order and become incorrect. The paper uses mostly single runs and is explicitly exploratory, so this should motivate controls rather than be treated as definitive evidence that LLM weight reading cannot work.

The Alan Turing Institute replication reports that a written Breakout policy could be independently decompiled to an exactly action-equivalent program over 49,189 states. However, its weight-only handwritten-digit classification performance averaged 50%, below an 82% fitted-template baseline. This reinforces a key distinction: compiling a policy, identifying a task, interpreting a trained representation, and diagnosing causal computation are different capabilities.

Design a matched family of small networks with identical architecture and dimensions but distinct hidden task functions, randomized class/action names, and procedural task families absent from familiar public benchmarks. Give the reader raw weights and a fixed set of frozen functional probes. Compare weight-only interpretation against architecture-only priors, activation-probe-assisted interpretation, intervention-assisted interpretation, and a verifier-only full-function oracle.

The primary question is not whether a reader can guess a task label. It is whether it can predict what happens under a controlled intervention. The evaluator should ask the reader to predict outputs after selected parameter-group ablations, hidden-unit permutations, output-label permutations, or scale-preserving weight shuffles, then execute those interventions and compare the predictions with actual outputs.

Primary outcomes: task-semantic identification accuracy, held-out function prediction, label-mapping recovery, causal intervention prediction, and false semantic-claim rate. Diagnostics: calibration, selective accuracy at matched coverage, performance after output-label permutation, unseen-task generalization, probe budget, and accuracy versus parameter count.

Negative controls must include architecture-only guessing, shuffled weights that preserve marginal distributions, randomized class/action names, misleading metadata, and task families generated only after the reader's method/prompt is frozen. Score task names and semantic explanations only when they imply a testable behavioural prediction. A confident explanation that fails the held-out intervention is a false claim, not a partial success.

Keep four claims distinct:
1. **Pattern reading:** recognition of local statistical structure in weights.
2. **Task identification:** correct prediction of the training/task family.
3. **Functional reconstruction:** prediction of outputs on unseen inputs.
4. **Causal understanding:** correct prediction of how specific interventions change the function.

Only the fourth supports a strong claim of weight-space diagnosis. This task is initially a benchmark specification; no Symthaea semantic-reading capability result is claimed.

### F.1 Joint-belief foveation and gaze/choice lookahead

The new `crates/core/symthaea-fep/src/three_ball_partial_observation.rs` module adds the partial-observation layer that the earlier known-state feedback evaluator explicitly omitted.

It maintains a normalized joint posterior (P(M,S)) over rule identity (M) and the 27 possible left/centre/right colour assignments (S). The initial model prior can be source-labelled (all 81 hypotheses) or quotient-compressed (75 semantic feedback tensors in the audited public SPM generator); hidden-state priors are independently supplied. Following observations, the full joint matrix is conditioned directly, so model/state correlations are not discarded by separately updating two marginals.

For a gaze action (g), the sensor deterministically reveals the selected ball's colour in this first task adapter. The module computes:

```text
I(M; O_g)
I(S; O_g | M)
I((M,S); O_g) = I(M; O_g) + I(S; O_g | M)
```

For each possible visual colour, it conditions the joint belief, evaluates the four feedback actions (red, green, blue, no response), and chooses the lowest expected-free-energy response. It compares each gaze-then-choice contingency with the immediate-choice baseline. The two-step policy score contains the expected downstream choice free energy, gaze cost, and the gaze observation's model/conditional-state information value.

**Scope and claim boundary:** this is a finite-hypothesis, deterministic-sensor, two-step receding-horizon approximation. It does not yet represent SPM's complete multi-step POMDP, continuous Dirichlet parameter uncertainty, noisy vision, gaze dynamics, or a learned visual generative model. It is useful as an executable intermediate and a target for exact comparison with the published task, not as an exact reproduction by itself.

Qualification tests cover the `ln(3)` nats of state information from one uniformly distributed ball colour, zero model information under an independent model/state prior, preservation of model/state correlations after feedback, impossible-observation rejection, and invariance of gaze/choice scores between labelled and semantically quotient rule families.

### G. Multi-trial finite-hypothesis rule-discovery adaptation

The branch now includes `crates/core/symthaea-fep/src/three_ball_sequential_discovery.rs`, a deterministic multi-trial evaluator around the joint model/state belief and two-step gaze policy.

Each trial independently samples a new left/centre/right colour arrangement, executes the current policy (one gaze → visual observation → one colour choice), observes correct/incorrect/no-feedback, updates the joint rule/state posterior, then marginalizes and carries the semantic rule posterior into the next trial while resetting only the hidden-scene prior. The runner has a versioned SplitMix64 seed stream and produces per-trial receipts containing the evaluator-only hidden scene, gaze, observed colour, choice, feedback, expected information terms, pre/post model entropy, true semantic-class posterior, winning-class odds, cumulative reward, and commitment state. The episode summary binds the pinned SPM demo commit, pinned rule-generator commit, canonical parameterization digest, seed, and source/quotient model-family setting. The complete discovery and post-commitment preference schedules are validated before the first trial. An exactly certain model posterior is represented in serializable output using an explicit absence marker for the mathematically infinite log-odds value.

The runner uses neutral feedback log preferences `[0,0,0]` during evidence gathering, then restores the source preference vector `[0,2,-6]` once the winning semantic-class log odds exceed 16 nats. The threshold is measured after aggregating source labels into semantic likelihood classes. The evaluator can run against either the 81 source-labelled hypotheses or the semantic quotient, and the benchmark compares their predictions and commitment traces.

**Exact claim boundary:** this is a repeated two-step finite-hypothesis adaptation, not the published SPM simulation. It has one gaze/choice pair per independent scene, not the source's six time points / five moves per trial. It performs exact discrete Bayesian conditioning over known deterministic rule hypotheses; it does not learn a continuous Dirichlet likelihood posterior or reproduce SPM's BMR parameter-update sequence. The SPM-compatible Dirichlet BMR and known-state model-EIG path remain separate. The public source generator also yields 75 distinct complete feedback tensors against the paper's reported 79 unique rules, so 81 labelled / 75 semantic source-audit results must remain explicit pending reconciliation.

The evaluator reports: trials to any 16-nat commitment, trials to *correct* semantic-rule commitment, final posterior on the planted semantic class, per-trial model entropy, Brier score, negative log probability, correct/incorrect/no-feedback counts, gaze count, and cumulative reward. It supports controlled ablations by setting model-information and conditional-state-information weights to zero individually or together. The paired policy matrix now includes full expected-free-energy, state-only, model-only, pragmatic-only, immediate-choice-only, fixed-centre-gaze, and uniform-random legal-action arms. Environment and policy randomness use separate domain-separated SplitMix64 streams, so all arms see the same sampled hidden-scene sequence for each paired seed. The matrix returns raw per-seed episode receipts rather than only aggregate scores, allowing downstream analysis to recompute paired effect sizes, bootstrap confidence intervals, calibration, and retention metrics. Source-labelled versus quotient runs must preserve the aggregated semantic-class posterior and commitment decision under matched seeds.

Do not demand that posterior entropy decreases after every realised observation. A surprising observation can increase posterior entropy; the mutual-information identity constrains the expectation over all possible outcomes. Qualification should check finite traces and expected information accounting, not false per-sample monotonicity.

### F.1 Source-fidelity boundary and remaining epistemic layer

The [2026 Nature Communications three-ball paper](https://www.nature.com/articles/s41467-026-77209-5) specifies five latent factors (left, centre, right, gaze, choice), a visual and feedback likelihood, six within-trial time points, 81 candidate likelihood mappings, and a reported 79 unique rules. It uses relative log preferences `[0, 2, -6]`, normalized as a categorical preference distribution, and accumulates Dirichlet evidence before model reduction and model commitment.

The current Symthaea implementation is intentionally **not an exact reproduction**:
- `three_ball_partial_observation.rs` performs a two-step receding-horizon approximation: gaze → observe one colour → choose. It maintains a finite joint posterior over rule identity and hidden ball colours; it does not represent the full six-time-point policy tree.
- `three_ball_sequential_discovery.rs` repeats this two-step policy over fresh hidden scenes, retaining model belief while resetting trial-specific state belief. It records deterministic paired policy arms and seed-level receipts.
- `three_ball_rule_space.rs` reconstructs the 81 ordered hypotheses from pinned public SPM sources. Its exact feedback-tensor equivalence check finds 75 distinct tensors, whereas the paper reports 79 unique rules. This discrepancy is retained and source-bound. It is not resolved by changing the published count or silently altering the model family.
- `dirichlet_model_reduction.rs` contains both strict structural-zero Dirichlet BMR and an explicitly SPM-compatible smoothing mode, plus prospective model information gain from action-conditioned predicted outcomes. These are separate from the finite-hypothesis two-step policy; there is no claim that the current episode runner carries the source Dirichlet posterior through every trial or reproduces the paper's model-commitment trajectory.

A material correctness fix was made in the two-step policy: the source preferences are *relative* log weights, not normalized log probabilities. Before calculating pragmatic risk, the implementation now normalizes them with log-sum-exp and computes (D_{KL}(Q(o)\|C(o))). A regression test shifts all three log preferences by +13 nats and requires identical pragmatic risk, expected-free-energy scores, and selected policy. This tests the required invariance of categorical probabilities to a common additive constant.

**Next research increment:** connect the action-conditioned Dirichlet predictive model and BMR into the partial-observation policy, so the agent can distinguish state information, parameter/likelihood novelty, and model-structure information without collapsing them into one entropy term. Preserve the current finite-hypothesis policy as a transparent baseline. Exact reproduction must wait until the 79-versus-75 model-space discrepancy and six-time-point planning path are reconciled.

## 6. Baselines and fairness requirements

### Temporal models
- Original LTC and CfC.
- Parameter-matched GRU and LSTM.
- ODE-RNN / Neural CDE where data and continuous-time observation interfaces match.
- Simple leaky-state and memoryless controls.
- State-space models where a fair implementation and budget are available.
- NIALIM / DFA-CfN / PRDFA-CfN from the 2026 rederived-CfC line. Reproduce the cited task/split and report errors and compute under the exact same conditions before making comparisons.

### HDC and associative memory
- OnlineHD / representative online HDC classifier.
- MAP/bipolar XOR, continuous Hadamard with its actual inverse/cleanup contract, and HRR.
- SVM and RVFL on identical features; include a strong conventional tabular baseline such as gradient-boosted trees where applicable.
- Random projection and a small tensor-product oracle for tractable dimensions.

A 2025 comparative HDC review across 121 UCI datasets found that conventional SVM/RVFL baselines can outperform HDC on selected classification workloads. Therefore HDC should earn its place through compositional transfer, online updates, robustness, interpretability, privacy/proof compatibility, or resource tradeoffs—not assumed classification superiority.

### FEP / active inference
- Pragmatic greedy, random, and fixed-probe policies.
- Always-observe and cue-greedy policies.
- Bounded Bayesian value-of-information policy.
- Model-information-gain-only policy.
- BMR / model-structure selection policy where the same reduced-model family and posterior information are available.

- Matched RL exploration baseline and a labelled perfect-information oracle upper bound.

Recent complexity-matched active-inference/RL comparisons do not establish a universal accuracy advantage for active inference. Compare both on identical observations, actions, horizons, model information, tuning budgets, and reward/preference structures. Report behavioral/interpretive differences separately from task accuracy.

### Weight-space editing
- No edit.
- Random edit with matched sparsity and norm.
- Analytical/oracle repair when the injected fault admits one.
- Gradient/Adam repair with matched forward/backward and evaluation budgets.
- Black-box search / evolutionary or coordinate search with matched candidate budgets.
- WSP weights-only.
- Published WSP prompt/tool/protocol replication with the original editor budget.
- Independent WSP reproduction; report calibration and live-run rows separately.
- WSP trace-assisted.
- SWI-constrained WSP plus independent verifier.

The analytical/oracle baseline is an upper bound, never a tuning signal. Weights-only and trace-assisted WSP are scored separately.

## 6.1 Current field position and what a fair comparison means

The field is moving quickly, and there is no single "SOTA" across these different problem classes. The relevant external reference points are:

| Area | Current external reference | What it establishes | Symthaea's required comparison |
|---|---|---|---|
| Weight-space learning | Dayan, Eitan & Maron, ICML 2026, permutation-equivariant weight-space networks | A theoretical expressivity analysis, universality results under stated assumptions, and a reported 34% improvement over prior SOTA on their own benchmarks | Include their permutation-equivariant family as a baseline where implementable. Compare function-space generalization, symmetry handling, task transfer, parameter count, and compute. Do not equate SWI's candidate editor with a trained weight-space learner. |
| Continuous-time networks | NIALIM / DFA-CfN / PRDFA-CfN, IEEE TNNLS 2026 | A revised closed-form approximation with an error upper bound and reported results across six sequence domains | Reproduce at least two appropriate public tasks with exact splits, missingness, sampling intervals, parameter count, and reference solver accuracy before making a CfC superiority claim. |
| Active inference / decision-focused acquisition | Friston et al., Nature Communications 2026; Filstroff et al., *Targeted Active Learning for Bayesian Decision-Making*, TMLR 2024 | Model-structure information gain can guide observation selection; targeted active learning maximizes information about the optimal downstream decision rather than generic model entropy | Compare model EIG, optimal-decision information gain, EVSI, cost-aware EVSI and matched RL on identical episodes. Enforce I(D*;O|a) <= I(M;O|a). The Friston task assumes a known model family; the TMLR result is a targeted-acquisition baseline, not evidence that Symthaea's implementation is novel. |
| Weight-space programming | Takehi et al., MIT CSAIL 2026; OpenWSP; Alan Turing Institute replication (2026) | Small-network policy compilation into weights can work, but new-task classification and deeper weight-only interpretation are much harder: the Turing replication reports 50% average handwritten-digit accuracy versus 82% for a fitted template baseline; on Breakout, two independent decompilers matched the network over 49,189 states | Separate program-to-weights compilation from learning new representations and from weight-only auditing. Include executable-program, non-program classification, input-layer-only, activation-probe and full functional baselines. |
| Sequential weight-space adaptation | Li et al., WeightLoop (arXiv:2611.07319, 5 Nov 2026) | A fixed editor writes a reusable bounded editing program; successive assay batches supply evidence for persistent predictor updates. Authors report targeted edits improved mean ranking utility in 7/8 endpoint-budget settings (all 8 vs random edits) and discovery across 3 sequential tasks in offline replay | Reproduce the sequential batch protocol and separate fixed-program reuse from per-batch LLM re-prompting, online/gradient updates and random matched edits. Record endpoint budgets, assays, editor calls, candidate evaluations, drift and held-out retention. This is a fresh preprint claim, not yet an independent replication. |
| WSP trajectory reliability and transfer | Hu et al., *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations* (arXiv:2611.09382, 6 Nov 2026) | Reports 16 editing trajectories across seven benchmarks and two Qwen3.5 model sizes, with strong but uneven held-out gains and six trajectories retaining the original weights when candidate improvements could not be reliably validated | Compare trajectory variance, qualified no-op retention, cross-task transfer/regression matrices, and gains at matched evaluation budgets. Do not select only the best trajectory or treat reported gains as reproduced. |
| Weight-only interpretation / causal audit | Weight Oracles (arXiv:2610.07334, Oct 2026) and Ahmed (arXiv:2611.08070, 5 Nov 2026) | Weight Oracles reports 99% holdout accuracy when simulating small transformers with a staged curriculum and external deterministic computation, plus AUROC 0.93 for attention-routed backdoors and 0.81 over a diversified threat distribution; Ahmed's exploratory label-permutation experiments show that local weight arithmetic can coexist with incorrect semantic labels. Neither establishes general scaling to realistic model sizes or causal interpretation without behavioral checks. | Compare raw weights, weight-only priors, frozen activation probes, controlled semantic interventions, and verifier-only full-function oracles. Include label remapping, hidden-unit permutation, within-matrix shuffling, unseen tasks, and positive/negative causal intervention controls. Score intervention predictions, not guessed dataset names. |\n| HDC/VSA | 2025 comparative review across popular HD/VSA tasks and 121 UCI datasets | HDC has resource and compositional advantages in some settings, but conventional SVM/RVFL baselines can win selected classification tasks | Require identical feature inputs and conventional baselines. Win on compositional OOD transfer, robust online learning, memory/latency, or proof/privacy compatibility rather than assuming HDC wins ordinary classification. |

A recent independent WSP replication illustrates why protocol identity matters: OpenWSP reports 12.9% on a live MATH-500 evaluation against the MIT authors' reported 48.1% and a 50.6% local calibration of the released edit. Its live evaluation used four samples per problem, compared with one for local calibration, and 84.25% of live generations hit the token limit. This is not a controlled refutation of the MIT result, but it demonstrates that raw scores from different evaluation conditions are not interchangeable. The benchmark should preserve the full evaluation configuration and keep all attempt/calibration rows visible.

**Honest position today:** Symthaea has a distinctive, increasingly auditable implementation foundation—typed candidate-only parameter programs, native transactional adapters, explicit epistemic information primitives, and a symmetry-aware test. It does not yet have a production epistemic controller, an autonomous WSP proposer, a measured integrated factorial result, or a demonstrated advantage over the above external baselines. The near-term research contribution should be reproducibility, functional-equivalence-aware weight repair, and a well-controlled comparison—not a premature general-intelligence or SOTA claim.

## 6.2 Evidence-feedback WSP: from one-shot edits to persistent, qualified adaptation

The most important recent extension is the distinction between **writing a weight edit** and **writing a bounded program that keeps adapting a predictor as new evidence arrives**.

Li et al.'s WeightLoop (arXiv:2611.07319, submitted 5 November 2026) proposes this second form: an LLM inspects the task and predictor once, writes a bounded reusable editing program, then measurement batches drive updates without another LLM call or gradient retraining. The authors report targeted improvements in seven of eight endpoint-budget settings and across three sequential discovery tasks in offline replay. These are author-reported preprint results, not an independently reproduced capability claim. See https://arxiv.org/abs/2611.07319.

The Alan Turing Institute's 2026 replication provides a useful counterweight. Its working note reports faithful Breakout policy compilation into weights and exact agreement by two independent decompilers over 49,189 game states, but only 50% average accuracy on handwritten digits versus 82% for a fitted template baseline. It also finds that weight-only reading largely recovers input layout while missing much of the task's underlying computation. See https://www.turing.ac.uk/news/publications/replicating-weight-space-programming-writing-and-reading-neural-network-weights. The practical implication: **program compilation, representation learning, behavioral identification, and causal fault diagnosis are four different benchmarks**. Weight Oracles is an important positive result on small transformers, but its own paper identifies realistic-scale deployment as the main open challenge; Ahmed's label-permutation results motivate semantic interventions that distinguish local arithmetic from correct task interpretation. Success on one must not be used to claim the others.

### The Symthaea architecture to test

1. **Compile once:** a WSP proposer may create a typed, bounded WeightProgram with exact target groups, task/fault hypothesis IDs, evidence receipt IDs, base/architecture/manifest digests, parameter budgets and native invariants.
2. **Acquire evidence:** a separate FEP policy chooses which observation or assay to acquire from action-conditioned predictive outcomes and expected state/parameter/model information. Probe cost and the value/preference channel remain explicit.
3. **Stage candidate updates:** each evidence batch produces a new candidate from an immutable model snapshot. Candidate changes must be bound to the exact evidence-batch digest and to the edit-program digest. The live predictor remains unchanged during proposal and evaluation.
4. **Verify and promote:** an independent verifier evaluates the candidate on predeclared held-out observations, task-retention sets and negative controls. Only a distinct, explicit promotion boundary can replace the deployed predictor; rollback remains available.

This composition is stronger than simply calling an LLM after every batch: the editor can be amortized into a reusable program, FEP can spend the measurement budget on disambiguating evidence, and SWI can constrain and audit every edit. But the pieces must be compared separately before the joint system is credited with an improvement.

### Registered sequential-feedback experiment

The manifest task sequential_evidence_weight_space_programming adds seven controls:

- no edit;
- random edit matched for parameter count and norm at each batch;
- feedback-blind fixed program;
- fixed WeightProgram compiled once and reused across batches;
- editor re-prompted after every evidence batch;
- a matched gradient/online update;
- FEP-selected measurements plus a fixed WSP edit program.

Use both stationary tasks and declared regime shifts. Include shuffled feedback, delayed feedback, and a held-out future-batch leakage canary. The proposer must not see future assay labels or final discovery endpoints.

Primary metrics: cumulative ranking utility, final discovery utility, held-out prediction loss, cumulative regret, and discovery per endpoint budget. Diagnostics: editor-call count, program reuse, candidate evaluations, changed-parameter count, cumulative parameter drift, task retention, calibration after feedback, rollback rate and time to detect a regime shift.

**Required comparison:** fixed editor reuse must be measured against the full per-batch editor cost, not just final model quality. Report total LLM calls, tokens, inference and candidate-evaluation compute, assay/query count, latency, parameter drift and final held-out utility. A fixed edit program that improves one endpoint while damaging prior tasks is not successful continual adaptation.

### Executable sequential session contract now present

The branch now includes `crates/core/symthaea-core/src/sequential_weight_program.rs`. It separates a frozen edit template from one-step programs bound to individual evidence batches:

1. `begin_sequential_edit_session` binds the template to exact architecture, manifest, task, session, and initial native model digests.
2. `prepare_sequential_candidate` requires the next in-order batch, the latest accepted native candidate digest, a unique evidence-batch digest, and an observation-manifest digest. It rebinds the frozen operations into an ordinary digest-bound `WeightProgram` and returns a detached `ValidatedWeightCandidate`. Preparing does not advance the session.
3. A model-specific SWI adapter must materialize the candidate on a clone, recompute its native fingerprint, and evaluate held-out task behaviour, retention, and negative controls.
4. `accept_verified_sequential_candidate` advances only when the caller supplies the native candidate digest, verifier receipt digest, and a positive verifier result. It checks that the prepared candidate is still bound to the current session state, rejects replay/out-of-order/stale batches, enforces cumulative L2-path and parameter-change-event budgets, and appends a hash-linked step receipt.

**Trust boundary:** the session module records a verifier attestation; it does not cryptographically authenticate an external verifier. The verifier receipt digest must be checked by the caller against the actual verifier artifact/signature, and the adapter must bind it to the exact native candidate fingerprint and evaluation results. The module does not promote the live model. A separate deployment-controlled promotion/rollback boundary remains necessary.

Targeted unit tests cover template freezing, prepare-without-advance, verifier rejection, successful sequential advancement, stale parent, out-of-order batch, tampered prepared candidate, and cumulative edit-path budget enforcement. The exact-head Rust workflow is responsible for compiling and executing these tests; do not infer their pass status from the design.

**Native CfC integration test added:** `src/dynamics/cfc/weight_program_adapter.rs::sequential_evidence_program_executes_through_native_cfc_candidate_boundary` runs a frozen tau-edit template for two successive evidence batches against real CfC snapshots. It materializes both batches through the native clone-only adapter, verifies that a rejected attestation leaves the session at batch 0, then accepts two separately attested candidates in order, checks that each next batch is bound to the preceding native fingerprint, and confirms that the original source network remains unchanged throughout. This closes the adapter/session contract at the test-fixture level; it is not a WSP proposer, autonomous feedback learner, cryptographic verifier, or deployment promotion path. Its Rust compile/test result remains subject to the exact-head workflow.

### Claim boundary

The present branch has a typed WeightProgram foundation, candidate-only CfC/HDC-LTC adapters, and a generic sequential evidence-batch session contract. It does **not** yet contain a WSP proposer that authors reusable programs from task descriptions, production FEP diagnostic/outcome-model adapters, a trust-authenticated verifier integration, or a production promotion/rollback controller. The sequential module is a generic candidate-session and receipt-chain primitive, not an implemented WeightLoop reproduction or autonomous continual-learning agent. Do not label the current SWI work as recursive self-improvement.

## 7. Metrics and hard acceptance gates

### Dynamics
- Zero-duration identity: (|F(0,x)-x|) at numerical precision.
- Semigroup/composition error: compare (F(t_1+t_2,x)) with (F(t_2,F(t_1,x))) under controlled constant-input conditions.
- Local and long-horizon error against a high-accuracy solver.
- Boundedness, stability, and error as a function of the dimensionless elapsed interval `Delta t / tau`.
- Latency, allocation count, state bytes, and energy only with documented device/method.

### HDC
- Binding/unbinding reconstruction and algebra-specific exactness.
- Retrieval precision/recall and false-retrieval rate.
- Capacity vs dimension, stored items, noise, query count and cleanup strategy.
- Compositional OOD accuracy and swap/order sensitivity.
- Memory bytes and operations.

### FEP
- Action-dependent expected information gain must vary under controlled changes to observation likelihoods.
- Report state-, parameter-, and model-structure information terms separately when each has a valid generative/posterior contract; unavailable terms are omitted, not fabricated.
- Predicted information gain must correlate with realized posterior reduction / correct model disambiguation.
- Do not sum marginal information terms without a declared factorization and a check against the joint chain-rule quantity.

- Brier score, log loss, posterior calibration, regret, probe count and observation cost.
- Information-channel-off negative control must remove the information-selection benefit when outcome distributions become non-discriminating.
- Hold novelty, reward preference, and action costs fixed in causal ablations.

### WSP / repair
- Fault detection precision/recall and localization precision.
- Target repair success and recovery of held-out task performance.
- Collateral regression on unrelated tasks and protected groups.
- Healthy-model false-edit rate with confidence interval.
- Parameter edit count, sparsity, norm, candidate count and forward-pass/compute budget.
- Time and evidence required to repair, verifier disagreement, rollback frequency.
- WSP replication variance, pass@k/evaluation multiplicity, token-limit fraction, and calibration-to-live gap.
- Weights-only vs trace-assisted performance.

**Non-negotiable gates:**
1. Exact source SHA, model/manifest digests, task revision, seed, split, configuration, budget and artifact digest are recorded.
2. Development and holdout sets are disjoint; holdout is sealed from proposer and selector.
3. Every enabled/disabled factor has a named adapter and evidence of activation or bypass.
4. Malformed, stale, over-budget, non-finite or invariant-breaking edits fail closed before mutation.
5. Healthy models are included; false edits are reported, not filtered out.
6. All repair outcomes include target improvement and collateral-regression measurements.
7. Information gain is computed from model-conditioned predictive outcomes/posteriors, not entropy of a deterministic state or novelty.
8. A negative control disables information-bearing observations while holding reward, costs, and novelty accounting constant.
9. Primary contrasts and multiplicity correction are preregistered; paired seeds and effect sizes with 95% confidence intervals are reported.
10. WSP headline results are not compared unless checkpoint, editor, prompt, tool, pass budget, decoding, evaluation multiplicity, and token-limit protocol are matched or explicitly reported as deviations.
11. Protocol validation never emits an accuracy, capability, efficiency, SOTA, safety, or consciousness claim.

For healthy-model false-edit rate, report exact binomial confidence intervals. Do not declare a low false-edit rate from a tiny control sample. A target such as an upper 95% confidence bound below 1% requires hundreds of independent healthy trials (approximately 300 zero-false-edit trials as a rough planning minimum); confirm the exact interval before stating that gate is met.

## 7.1 Competitive targets and claim ladder

These are preregistered **aspiration thresholds**, not current results. Freeze the task, data, seeds, model sizes, and budget before assessing them; adapt the threshold only when a task-specific meaningful-effect margin has been justified in advance.

| Capability | Qualification target | Competitive target |
|---|---|---|
| CfC/LTC temporal modeling | Reproduce a published baseline under the exact dataset, split, missingness, and elapsed-time protocol; report error against a high-accuracy numerical reference | At matched parameters and compute, either reduce forecast error by at least 5% or achieve non-inferior error (within a predeclared 1% relative margin) with at least 20% lower p95 latency or memory |
| HDC/VSA | Exact algebra contracts, calibrated retrieval precision/recall, and corruption curves against MAP, HRR, OnlineHD and conventional controls | Show a reproducible Pareto advantage: at least 20% lower memory or query latency at no more than a 1 percentage-point task-performance loss, or a meaningful held-out compositional-transfer gain |
| FEP diagnosis | Prospective information gain predicts realized disambiguation; posterior calibration and regret beat random/greedy controls | At the same diagnosis accuracy (within 1 percentage point), reduce median diagnostic probes by at least 20% versus the strongest budget-matched baseline |
| WSP repair | At least 90% repair success on the registered planted-fault suite, with no hidden-label leakage and independently verified preservation | Demonstrate a healthy-model false-edit upper 95% exact-binomial bound below 1% (requires roughly 300 independent healthy cases with zero false edits), while keeping collateral regression within the predeclared task margin |
| Integrated H×C×F×W | Every factor has verified activation/bypass receipts; report all main effects and interactions | Improve the joint repair–retention–probe-cost frontier by at least 10% versus the strongest single-component and conventional baselines, with paired 95% confidence intervals excluding no improvement |

**Claim ladder:**
1. *Correctness*: numerical contracts, candidate isolation, exact fingerprints, and negative controls pass.
2. *Task qualification*: the declared tasks and all baselines execute with complete source-bound receipts.
3. *Competitive*: the relevant capability reaches its predeclared target under matched compute and data.
4. *Field-leading*: the advantage survives independent replication, multiple public datasets or fault families, multiplicity correction, and disclosure of all failed/negative runs.

A single synthetic fixture, one seed, one successful PR workflow, or an aggregate score is not evidence of SOTA. Report latency, memory, energy, and accuracy as a Pareto surface rather than compressing them into an arbitrary scalar. An integrated win cannot be inferred from the full profile alone; estimate the registered factor interactions.

## 8. A staged implementation programme

### Stage 0 — contracts and reproducibility
- Preserve the existing SWI-001 and HLB-002 semantics; do not merge their two HDC-LTC parameterizations by name alone.
- Freeze typed manifests, parameter role metadata and exact digest preconditions.
- Keep the hierarchical information-gain and diagonal-Gaussian BMR primitives research-only until real outcome-model and posterior adapters exist; do not wire either directly into production action selection.
- Add this 4-factor protocol and validate it independently of the runtime.

### Stage 1 — deterministic fault and diagnostic environment
- Build fault injectors with hidden ground truth.
- Build model hypothesis and diagnostic observation likelihood adapters.
- Verify model-relative information gain against exact Bayesian calculation in small enumerable tasks.
- Add healthy controls and negative controls.
- Compare model EIG, optimal-decision information gain, and EVSI as separate objectives against no-probe, pure-EIG, targeted-decision-EIG, cost-aware EVSI, and always-inspect.
- Emit a deterministic source-bound JSON result with paired episode seeds and paired policy contrasts; preserve the result artifact and hashes for independent reanalysis.

### Stage 2 — constrained WSP candidate API
- Define the typed WeightProgram IR.
- Convert only a small set of operation kinds into SWI transactional edit batches.
- Add candidate-only cloning, manifest/base-digest checks, edit budgets, rollback, and independent preservation tests.
- Compare weights-only and trace-assisted WSP.

### Stage 2.1 — Sequential evidence-feedback contract

The generic sequential session module is implemented and receives focused tests. The remaining integration task is to connect a concrete native adapter and independent verifier so that the verifier receipt is not merely passed as an opaque digest and boolean. The first experiment should use a small CfC task with planted tau/gate fault batches, plus shuffled/delayed-feedback and future-batch leakage controls. Measure cumulative utility, prediction loss, parameter drift path, held-out retention, query cost, candidate count, and editor-call amortization.

### Stage 3 — connect HDC/HLB
- Use HDC/HRR only where role/filler compositionality and associative retrieval improve measured task performance.
- Evaluate structured cross-dimensional mixing separately from coordinate-wise baseline.
- Compare scalar-tau and per-coordinate-tau liquid variants as distinct models.
- Measure temporal credit assignment (local targets vs eligibility traces/e-prop/BPTT) only after the task harness is qualified.

### Stage 3.1 — Multi-layer symmetry quotient

The next concrete expansion is now implemented for one restricted topology class: stacked feed-forward CfC networks with no residual connections, no backbone, and no bidirectional path. A layerwise alignment bundle is accepted only when each layer has one fresh receipt bound to the same source model digest, candidate model digest, architecture digest, and manifest digest, with the layer-specific probe digest and mapping retained independently.

The group action is compositional. For layer i, its hidden permutation acts on:
- rows of its input projection;
- columns of its input projection using layer i-1's permutation (except at the external input);
- rows and columns of its recurrent projection;
- its hidden bias and time-constant vectors;
- the final output projection's columns, using the last layer's permutation.

This is a quotient-space operation: it changes coordinates while preserving the realized network function. The resulting edit is still a typed candidate-only WeightProgram; the native fingerprint and held-out functional probes are checked independently. The current two-layer test is a correctness fixture, not a performance benchmark and not support for residual/backbone/bidirectional models.

### Stage 4 — generalization and publication
- Expand from synthetic planted faults to held-out fault families and small real networks.
- Run all factorial profiles with paired seeds and compute-matched baselines.
- Release protocol, task generator, raw source-bound receipts, negative results and analysis scripts.
- Seek independent replication before any industry-leading claim.

## 9. Research references

- Navon et al., *Equivariant Deep Weight Space Alignment*, ICML 2024, for learned permutation-equivariant weight alignment; compare its learned alignment with this branch's deterministic functional-probe baseline.
- Dayan, Eitan & Maron, *On the Expressive Power of Permutation-Equivariant Weight-Space Networks*, ICML 2026, for expressivity/universality results and the reported 34% benchmark improvement under that paper's protocols.
- Friston et al., *Active inference and artificial reasoning*, Nature Communications 17, 10125 (2026), for active model-structure information gain and post-observation BMR in the three-ball paradigm.
- Filstroff et al., *Targeted Active Learning for Bayesian Decision-Making*, Transactions on Machine Learning Research (2024): https://openreview.net/forum?id=KxPjuiMgmm.
- Li et al., *WeightLoop: Weight-Space Programming with Experimental Feedback* (arXiv:2611.07319, 5 Nov 2026): https://arxiv.org/abs/2611.07319. Treat the paper's offline-replay claims as author-reported until independently replicated.
- Liang et al., *Rederived Closed-Form Continuous-Time Neural Networks*, IEEE TNNLS 37(9) (2026), for NIALIM and the DFA-CfN / PRDFA-CfN baselines.
- Vergés et al., *Classification using hyperdimensional computing: a review with comparative analysis*, Artificial Intelligence Review 58, 173 (2025), for a shared seven-task and 121-UCI-dataset comparative HDC evaluation.
- OpenWSP, independent replication records for the 2026 weight-space programming paper; treat each run and calibration row with its own prompt, sample-count, token-limit, and finish-reason metadata.

- Takehi, Teoh, Zhang & Khattab (2026), *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* (MIT CSAIL research blog, 2 Nov 2026): https://rikiyat.github.io/blog/weight-space-programming/
- OpenWSP, independent replication and versioned experiment records for WSP (updated 6 Nov 2026): https://github.com/openwsp/openwsp
- Han et al. (2026), *A Survey of Weight Space Learning: Understanding, Representation, and Generation*: https://arxiv.org/abs/2603.10090
- Dayan, Eitan & Maron (2026), *On the Expressive Power of Permutation-Equivariant Weight-Space Networks*, ICML 2026: https://proceedings.mlr.press/v306/dayan26a.html
- Wang, Wang & Wang (2026), *Position: Weight Space Should Be a First-Class Generative AI Modality*, ICML 2026: https://proceedings.mlr.press/v306/wang26mc.html
- Nusrat & Nusrat (2026), *Can LLMs Reliably Audit Clinical Decision Networks?*: https://arxiv.org/abs/2611.05341
- Vergés et al. (2025), *Classification using hyperdimensional computing: a review with comparative analysis*: https://doi.org/10.1007/s10462-025-11181-2
- Hasani et al. (2022), *Closed-form continuous-time neural networks*: https://doi.org/10.1038/s42256-022-00556-7
- Liang et al. (2026), *Rederived Closed-Form Continuous-Time Neural Networks*: https://doi.org/10.1109/TNNLS.2026.3669345
- Friston et al. (2026), *Active inference and artificial reasoning*: https://doi.org/10.1038/s41467-026-77209-5
- Chou et al. (2026), complexity-matched empirical comparison of active inference and reinforcement learning: https://doi.org/10.1016/j.jmp.2026.103011
- Symthaea FEP qualification blocker: https://github.com/Luminous-Dynamics/symthaea/issues/8391
- Symthaea SWI-001: https://github.com/Luminous-Dynamics/symthaea/pull/10628
- Symthaea HLB factorial v1: https://github.com/Luminous-Dynamics/symthaea/pull/8375
- Symthaea model information gain primitive: https://github.com/Luminous-Dynamics/symthaea/pull/8394
- Symthaea decision-focused FEP integration: https://github.com/Luminous-Dynamics/symthaea/issues/11122.
- Symthaea WeightLoop reproduction baseline: https://github.com/Luminous-Dynamics/symthaea/issues/11159.

---

## 10. Sequential decision-focused diagnostics

The one-step EVSI kernel is now complemented by a bounded finite-horizon planner in `crates/core/symthaea-fep/src/sequential_diagnostic_planning.rs`.

The planner recomputes Bayesian model beliefs after every candidate observation, chooses follow-up diagnostics conditionally, prohibits action reuse on a path, and includes an explicit stop action at every node. It minimizes expected terminal Bayes risk plus observation-cost and safety-risk penalties. Search is bounded by maximum hypothesis, action, outcome, and belief-node counts; budget exhaustion fails closed.

The critical fixture is a four-hypothesis XOR task. Two diagnostic probes independently reveal two latent fault bits, while the downstream repair decision depends on their XOR. Each probe alone has positive model information gain but zero one-step decision value. A two-probe adaptive sequence identifies the correct downstream decision. This is a mathematical conformance fixture, not an empirical performance result.

Tracking: [issue #11179](https://github.com/Luminous-Dynamics/symthaea/issues/11179). Required next step: extend the paired diagnostic benchmark to compare myopic EVSI and bounded sequential EVSI under identical probe, compute, and cost budgets, including model-likelihood misspecification.

## 11. Current industry and hardware comparison requirements

The HLB programme must compare against current deployable systems, not only academic neural-network baselines.

### Liquid AI LFM3

Liquid AI's 27 October 2026 LFM3 release is a relevant on-device/agentic baseline. Its LFM3-1.7B uses a tri-hybrid mix of gated short convolutions, sliding-window attention, and global attention; the vendor reports a 256K context window, a Q4 footprint around 1 GB, and hardware-in-the-loop architecture search across 14 device/backend combinations including Raspberry Pi 5, mobile devices, CPUs, GPUs, and Qualcomm NPU. Its published benchmark table includes agent, tool-use, coding, knowledge, and instruction-following tasks.

This is **not** a continuous-time LTC/CfC network merely because the product family is called Liquid Foundation Models. It is a hybrid sequence architecture. It belongs in the deployment/agent capability comparison, while NIALIM/DFA-CfN/PRDFA-CfN remain direct continuous-time sequence competitors.

Sources:
- Liquid AI, *Introducing LFM3: Every Modality for Specialized Intelligence* (27 Oct 2026): https://www.liquid.ai/blog/lfm3
- Liquid AI, model release and model cards: https://www.liquid.ai/news/models

### Liquid AI d1-ultra decision model

The 15 October 2026 d1-ultra announcement describes a multimodal decision model that returns probabilities for yes/no, multiple-choice, and scalar-score questions in one forward pass, with explicit emphasis on confidence calibration. Liquid AI reports comparisons across 112 evaluations and calibrated decision outputs.

This is a direct industry comparator for the **decision-scoring/calibration** slice of HLB-FEP, not for sequential experiment selection, causal intervention, or model repair. Compare Brier score, log score, calibration error, selective risk, accuracy, latency, and cost. Then compare whether Symthaea's active diagnostic loop improves downstream decisions after spending the same observation budget.

Source: Liquid AI, *d1-ultra: Introducing the World's Best Decision Model* (15 Oct 2026): https://www.liquid.ai/blog/d1-ultra

### GHRR and dedicated hardware

The DATE 2026 GHRR work reports non-commutative matrix-based binding with a vector-space flattening method and a custom ASIC. The published abstract reports 1.36×–1.56× throughput and 16.2×–18.6× energy-efficiency improvements relative to a PyTorch/RTX 4090 baseline. This is a serious competitor for order-aware holographic operations and hardware acceleration.

The comparison is not automatically apples-to-apples because the hardware and software baselines differ. Require end-to-end task quality, implementation details, device power measurement, and equivalent workload definitions before drawing energy or throughput conclusions.

Source: DATE 2026, *Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration*: https://doi.org/10.23919/DATE69613.2026.11539261

### WeightLoop

WeightLoop (arXiv:2611.07319, 5 Nov 2026) is a direct baseline for fixed-editor, experimental-feedback weight-space programming. Its authors report offline replay gains over endpoint-budget settings and sequential discovery tasks. Treat those findings as author-reported until independently reproduced.

Symthaea must not claim the fixed reusable editor / experimental feedback loop as unique. Its potential differentiation is the integration of typed architecture-aware programs, decision-relevant and sequential diagnostics, candidate-only transactional editing, independent verification, and capability preservation.

Source: https://arxiv.org/abs/2611.07319

### WSP-Scope: multi-trajectory generalization and reliability

Hu et al. (arXiv:2611.09382, 6 Nov 2026) evaluate 16 editing trajectories across seven benchmarks and Qwen3.5-0.8B/9B. They report sizeable gains on several held-out tasks but also uneven cross-task transfer and six trajectories that preserve original weights when candidate improvements cannot be reliably validated. This is an important counterweight to single-trajectory headline results. Symthaea's sequential WSP benchmark now registers trajectory-level variance, qualified no-op retention, cross-task transfer and regression matrices, and protected-task canaries. The study is an external comparator; none of its results are claimed as reproduced here.

Source: https://arxiv.org/abs/2611.09382

### Required fair deployment protocol

For every industry comparison, record:
- exact model/checkpoint and quantization;
- hardware, firmware, compiler/runtime, driver, and backend;
- input/output or observation budget;
- concurrency, batch size, warm/cold start, context length, and cache state;
- p50/p95/p99 latency, throughput, peak memory, allocations, and energy per completed task;
- task success, decision loss, calibration, abstention, and failure/recovery behavior;
- tool/editor calls, forward passes, tokens, and total wall-clock cost.

Use both a shared-device comparison and a best-supported deployment comparison. A model's benchmark score is not an end-to-end systems result.

## 12. Claim boundary

This document defines hypotheses and an experiment. It does not claim that current Symthaea FEP action selection performs valid expected-information planning, that WSP is implemented as an autonomous editor, that HDC/CfC/FEP/WSP adapters are integrated, that the architecture exceeds any SOTA system, that weight edits are safe in general, or that any consciousness property has been demonstrated.
