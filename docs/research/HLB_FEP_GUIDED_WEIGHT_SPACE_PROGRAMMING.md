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

A candidate action score may take the explicitly scoped form:

```text
J(a) = E[L_task | a]
     + lambda_cost * C(a)
     + lambda_risk * R_safety(a)
     - lambda_info * I(M; O | a)
```

where lower is better, (C(a)) is observation/action cost, and (R_{mathrm{safety}}) is a separately measured risk term. This is a benchmark decision rule, not a claim that the equation alone is a complete expected-free-energy implementation. The full policy must declare preferences, horizon, model uncertainty, and action constraints.

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

Future work should provide a canonical digest-construction helper over native manifests/configuration (the present function requires callers to supply a digest), add correlated Gaussian priors/posteriors, and prove numerical stability against Cholesky-based reference calculations. Only then should the operator become part of an FEP model-structure adapter.

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

This is the appropriate BMR substrate for the [2026 three-ball active structure-learning paper](https://doi.org/10.1038/s41467-026-77209-5), whose categorical likelihood parameters are represented by Dirichlet counts. It remains a primitive: the complete POMDP, policy evaluator, model-space construction, and prospective model expected-information-gain adapter are not yet wired into Symthaea's production selector.

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

The primitive does not reorder parameters, rewrite native fingerprints, change WeightProgram target identities, or authorize edits. Current tests cover known permutation recovery, ambiguous duplicate signatures, malformed inputs, and integration into the one-layer no-backbone CfC permutation fixture. Multi-layer/backbone correspondence and any transform application remain unimplemented. This is a cautious first bridge between behavioral evidence and parameter-space editing, not a general weight-space solver.

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

Compare raw flattened-weight inspection, activation-probe alignment, permutation-equivariant weight-space editors, SWI's typed group alignment, and a labelled oracle-alignment upper bound. The oracle is verifier-only. A permutation-only change must not be called a defect; a repair must still be evaluated using native model fingerprints and actual behavior.

**Executable anchor now present on this branch:** `src/dynamics/cfc/weight_program_adapter.rs::consistent_hidden_unit_permutation_preserves_cfc_function` materializes the equivalence-preserving permutation through the actual `WeightProgram → materialize_cfc_weight_program → cloned SWI candidate` path. It checks that the native fingerprint changes while a four-step temporal probe sequence remains equivalent. The paired partial-permutation negative control changes only the input-projection rows and must change the output sequence. The fixture is intentionally narrow: one small, one-layer, no-backbone CfC network. It validates one exact symmetry family, not arbitrary multi-layer/backbone transformations or an alignment algorithm.


### F. Partial-observation rule discovery: the three-ball reproduction target

Use Friston et al. (2026), [*Active inference and artificial reasoning*](https://doi.org/10.1038/s41467-026-77209-5), as a bounded model-structure learning benchmark. The exact task includes:

- Five latent factors: the colours of left, centre, and right balls (red/green/blue each); a controllable gaze location (start/left/centre/right); and a controllable choice (no-choice/red/blue/green).
- Two outcome modalities: vision (nothing/red/green/blue) and feedback (no-feedback/correct/incorrect).
- Gaze and choice are action-controlled; ball colours are sampled randomly at trial start. Each trial has six time points (five moves).
- Relative log preferences for the feedback modality are [0, 2, -6] nats in the order no-feedback/correct/incorrect.
- The planted context-dependent rule maps centre-ball red to the left ball, green to the centre ball, and blue to the right ball as the criterion for the correct colour choice.
- The symmetry-constrained model generator produces 81 candidate likelihood mappings, of which two are degenerate duplicates, leaving 79 unique rules; rule 65 is the planted ground truth.
- Crucially, the generative model is assumed known except for the feedback likelihood tensor. The paper evaluates active selection among candidate rule structures; it does **not** solve perception-to-world-model construction or unconstrained ARC-style rule invention.

The paper reports model commitment when the winning model's log-odds crosses 16 nats in its illustrative setup. Treat this as a reproduction metric, not a universal confidence threshold. Report both time to threshold and false-commitment rate under alternative planted rules and prior misspecification.

Two tracks remain separate:

1. **Exact reproduction:** independently verify the state factors, observation/feedback likelihoods, transition structure, candidate model-generation algorithm, Dirichlet prior counts, action horizon, preference vector, and selection rule from the paper and supplement. Pin the source/supplement revision and all implementation choices.
2. **Task adaptation:** any change to factors, outcomes, ball count, rule family, preferences, horizon, likelihood model, or hypothesis-space constraints must be labelled an adaptation.

Primary outcomes: ground-truth rule posterior probability, rule-identification accuracy, trials to 16-nat model log-odds, model-choice Brier score, decision regret, and observation cost. Diagnostics: state/parameter/model-structure information gain, realized model-entropy reduction, Dirichlet BMR log evidence, posterior winner/runner-up odds, false commitment, and calibration under prior misspecification.

The categorical likelihood BMR baseline must use the Dirichlet primitive in `crates/core/symthaea-fep/src/dirichlet_model_reduction.rs`, not the diagonal-Gaussian BMR primitive. Compare random legal observation, reward/pragmatic-greedy without model information, state-only and parameter-only information policies, hierarchical model EIG with Dirichlet BMR, and an analysis-only oracle. Keep preferences, observation cost, legal actions, model priors, and horizon identical across ablations.

BMR is evaluated after evidence updates the full-model Dirichlet posterior. It must not be added directly to prospective expected information gain or pragmatic value. The state/parameter/model information terms remain separate and their chain rule must be checked.

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
| Active inference | Friston et al., Nature Communications 2026, active structure learning and the three-ball rule-discovery paradigm | Model-structure information gain can guide observation selection; BMR can efficiently compare reduced models after evidence accumulation | Reproduce the bounded task, then test whether Symthaea's state/parameter/model decomposition improves observations-to-correct-model and calibration. The paper assumes a known generative model family; it is not a result for general perception or open-ended reasoning. |
| Weight-space programming | Takehi et al., MIT CSAIL 2026, plus OpenWSP's independent records | Direct weight inspection/editing can produce striking small-model results, but outcomes depend heavily on prompts, editor budgets, sampling, token limits, and evaluation protocol | Pin every checkpoint, prompt/editor revision, tool permission, edit budget, and evaluation. Report live and calibration runs separately; do not cite one headline as portable capability. |
| HDC/VSA | 2025 comparative review across popular HD/VSA tasks and 121 UCI datasets | HDC has resource and compositional advantages in some settings, but conventional SVM/RVFL baselines can win selected classification tasks | Require identical feature inputs and conventional baselines. Win on compositional OOD transfer, robust online learning, memory/latency, or proof/privacy compatibility rather than assuming HDC wins ordinary classification. |

A recent independent WSP replication illustrates why protocol identity matters: OpenWSP reports 12.9% on a live MATH-500 evaluation against the MIT authors' reported 48.1% and a 50.6% local calibration of the released edit. Its live evaluation used four samples per problem, compared with one for local calibration, and 84.25% of live generations hit the token limit. This is not a controlled refutation of the MIT result, but it demonstrates that raw scores from different evaluation conditions are not interchangeable. The benchmark should preserve the full evaluation configuration and keep all attempt/calibration rows visible.

**Honest position today:** Symthaea has a distinctive, increasingly auditable implementation foundation—typed candidate-only parameter programs, native transactional adapters, explicit epistemic information primitives, and a symmetry-aware test. It does not yet have a production epistemic controller, an autonomous WSP proposer, a measured integrated factorial result, or a demonstrated advantage over the above external baselines. The near-term research contribution should be reproducibility, functional-equivalence-aware weight repair, and a well-controlled comparison—not a premature general-intelligence or SOTA claim.

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
- Compare diagnostic information gain with realized information and regret.

### Stage 2 — constrained WSP candidate API
- Define the typed WeightProgram IR.
- Convert only a small set of operation kinds into SWI transactional edit batches.
- Add candidate-only cloning, manifest/base-digest checks, edit budgets, rollback, and independent preservation tests.
- Compare weights-only and trace-assisted WSP.

### Stage 3 — connect HDC/HLB
- Use HDC/HRR only where role/filler compositionality and associative retrieval improve measured task performance.
- Evaluate structured cross-dimensional mixing separately from coordinate-wise baseline.
- Compare scalar-tau and per-coordinate-tau liquid variants as distinct models.
- Measure temporal credit assignment (local targets vs eligibility traces/e-prop/BPTT) only after the task harness is qualified.

### Stage 4 — generalization and publication
- Expand from synthetic planted faults to held-out fault families and small real networks.
- Run all factorial profiles with paired seeds and compute-matched baselines.
- Release protocol, task generator, raw source-bound receipts, negative results and analysis scripts.
- Seek independent replication before any industry-leading claim.

## 9. Research references

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

---

## 10. Claim boundary

This document defines hypotheses and an experiment. It does not claim that current Symthaea FEP action selection performs valid expected-information planning, that WSP is implemented as an autonomous editor, that HDC/CfC/FEP/WSP adapters are integrated, that the architecture exceeds any SOTA system, that weight edits are safe in general, or that any consciousness property has been demonstrated.
