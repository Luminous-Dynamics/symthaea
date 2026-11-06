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

[
p(omid a)=sum_m p(m)p(omid m,a)
]

[
p(mmid o,a)=rac{p(m)p(omid m,a)}{p(omid a)}
]

[
I(M;Omid a)=H(M)-sum_o p(omid a)H(Mmid o,a)
]

A candidate action score may take the explicitly scoped form

[
J(a)=mathbb{E}[L_{mathrm{task}}mid a]
+lambda_c C(a)+lambda_s R_{mathrm{safety}}(a)
-lambda_i I(M;Omid a)
]

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
- Unit tests cover detached candidate creation, stale model/task rejection, role and tau constraints, duplicate-target rejection, per-group/global budgets, deterministic candidate digests, and a planted 3× tau fault repaired on an isolated clone. The planted-fault test also checks that the same proposal is rejected against the healthy model's different base digest.

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

### C. Compositional and associative transfer
Use role–filler, nested relation, temporal order, and held-out composition tasks. Probe binding/unbinding, retrieval under crosstalk, role swaps, missing elements, and controlled corruption. Compare Hadamard/MAP, bipolar XOR, HRR, random projection, tensor-product small-scale oracle, and classical models where appropriate.

Primary outcomes: held-out compositional success, retrieval precision/recall, capacity as a function of dimension and load, false retrieval, calibration, and memory cost.

### D. Irregular-time dynamics and regime changes
Use fixed datasets and synthetic systems with nonuniform (Delta t), missing observations, input interpolation, and controlled regime changes. Report error by elapsed interval and wall-clock cost.

Primary outcomes: forecast error, rollout stability, time-to-recover after shifts, error against high-accuracy references, and end-to-end latency.

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
- Matched RL exploration baseline and a labelled perfect-information oracle upper bound.

Recent complexity-matched active-inference/RL comparisons do not establish a universal accuracy advantage for active inference. Compare both on identical observations, actions, horizons, model information, tuning budgets, and reward/preference structures. Report behavioral/interpretive differences separately from task accuracy.

### Weight-space editing
- No edit.
- Random edit with matched sparsity and norm.
- Analytical/oracle repair when the injected fault admits one.
- Gradient/Adam repair with matched forward/backward and evaluation budgets.
- Black-box search / evolutionary or coordinate search with matched candidate budgets.
- WSP weights-only.
- WSP trace-assisted.
- SWI-constrained WSP plus independent verifier.

The analytical/oracle baseline is an upper bound, never a tuning signal. Weights-only and trace-assisted WSP are scored separately.

## 7. Metrics and hard acceptance gates

### Dynamics
- Zero-duration identity: (|F(0,x)-x|) at numerical precision.
- Semigroup/composition error: compare (F(t_1+t_2,x)) with (F(t_2,F(t_1,x))) under controlled constant-input conditions.
- Local and long-horizon error against a high-accuracy solver.
- Boundedness, stability, and error by (Delta t/	au).
- Latency, allocation count, state bytes, and energy only with documented device/method.

### HDC
- Binding/unbinding reconstruction and algebra-specific exactness.
- Retrieval precision/recall and false-retrieval rate.
- Capacity vs dimension, stored items, noise, query count and cleanup strategy.
- Compositional OOD accuracy and swap/order sensitivity.
- Memory bytes and operations.

### FEP
- Action-dependent expected information gain must vary under controlled changes to observation likelihoods.
- Predicted information gain must correlate with realized posterior reduction / correct model disambiguation.
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
10. Protocol validation never emits an accuracy, capability, efficiency, SOTA, safety, or consciousness claim.

For healthy-model false-edit rate, report exact binomial confidence intervals. Do not declare a low false-edit rate from a tiny control sample. A target such as an upper 95% confidence bound below 1% requires hundreds of independent healthy trials (approximately 300 zero-false-edit trials as a rough planning minimum); confirm the exact interval before stating that gate is met.

## 8. A staged implementation programme

### Stage 0 — contracts and reproducibility
- Preserve the existing SWI-001 and HLB-002 semantics; do not merge their two HDC-LTC parameterizations by name alone.
- Freeze typed manifests, parameter role metadata and exact digest preconditions.
- Keep the FEP information-gain primitive isolated until real outcome-model adapters exist.
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
- Han et al. (2026), *A Survey of Weight Space Learning: Understanding, Representation, and Generation*: https://arxiv.org/abs/2603.10090
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
