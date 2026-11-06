# SWI-001: Symthaea Weight-Space Intelligence
## Research programme, implementation contract, and benchmark specification v0.1

**Status:** Research programme approved for implementation; CfC and HDC-LTC introspection, transactional weight editing, content-addressed fingerprints, corrected HDC-LTC local gradients, deterministic initialization, and bounded parameter inspection committed on the working branch.  
**Branch:** research/swi-001-weight-space-introspection  
**Base:** main at ed9d3462e73cd3d435c89f65fe4b5834f34bc184  
**Scope:** weight-space programming (WSP), weight-space learning (WSL), architecture-aware introspection, controlled repair, and eventually staged substrate evolution.  
**Evidence status:** design and code audit; the new Rust tests have been added but have not been run in this environment.

---

## 1. Executive decision

Symthaea should pursue weight-space intelligence, but it should not replace gradient learning, Hebbian learning, predictive learning, or its existing HDC/CfC dynamics.

The recommended direction is a complementary capability that can:

1. Inspect a model's named, typed parameter structure and its operational traces.
2. Form falsifiable hypotheses about specific computational failures.
3. Propose bounded and reviewable parameter-space interventions.
4. Evaluate candidate interventions against frozen, held-out tests and explicit preservation constraints.
5. Commit only qualified candidates, retaining immutable baselines and complete rollback evidence.

The first target is the CfC network because it has explicit named parameter groups, analytic gradients, deterministic-seeding support elsewhere in the architecture, and a flat weight serialization boundary that can be hardened.

The first deliverable is not autonomous self-improvement. It is a reproducible fault-injection and repair benchmark in which the system can identify a known defect, localize it, propose a minimal edit, and demonstrate that the edit repairs the intended behavior without unacceptable collateral regression.

## 2. Terminology: do not collapse adjacent fields

**Weight-Space Programming (WSP)** is the task-time use of a general-purpose model or agent to inspect and directly modify a neural network's parameters around a goal or specification, rather than relying exclusively on a gradient optimizer. The November 2, 2026 MIT CSAIL report is an important early demonstration, published as a research blog post with released experiment logs, not yet as a peer-reviewed benchmark paper.

**Weight-Space Learning (WSL)** is the broader study of distributions, geometry, symmetries, representations, comparison, and generation over neural-network weights. The March 2026 survey groups it into weight-space understanding, representation, and generation. It is not the same task as asking an LLM to repair a network.

**Mechanistic interpretability** studies internal features, circuits, and causal pathways. It can inform WSP, but a plausible parameter explanation is not proof of a causal mechanism.

**Model editing** changes a model to alter or remove a behavior or knowledge. It is a relevant baseline and related field, but WSP is distinguished by directly reasoning over parameter values and structures.

For SWI, these become four separate evaluation tracks:
- WSP: can an agent inspect and propose useful edits?
- WSL: can a learned representation compare or generate useful parameter structures?
- Mechanistic audit: can a proposed explanation survive interventions?
- Verified repair: does the candidate actually improve held-out behavior while preserving unrelated capabilities?

## 3. Literature and state of the art as of 2026-11-05

### 3.1 MIT CSAIL: Sparks of Weight-Space Programming

Takehi, Teoh, Zhang, and Khattab report that GPT-6 Astra identified the task of small neural policies from weights, inferred characteristics of small character transformers, repaired a failing car-ramp policy, generated small game-playing networks, and edited Qwen3.5-0.8B to reduce reasoning loops.

The Qwen result is striking but narrow:
- The reported edit touched 131,072 of approximately 0.8 billion parameters (0.016%).
- MATH-500 rose from 7.9% to 48.1% in the reported setup.
- GSM8K rose from 13.8% to 67.1%.
- IFEval rose from 21.0% to 62.3%.
- The editor used 50 batches of practice prompts / forward-pass opportunities.
- The authors also tested Gemma 3 270M, where ordinary reasoning errors did not expose one repairable failure direction and the attempted edits did not establish an improvement.

The authors' own negative result is essential: removing an obstacle that hides an existing capability is not equivalent to creating a missing capability. The report also notes context and harness limitations and says the team is building a benchmark.

SWI should reproduce the *methodological discipline*, not copy the headline score or claim that 0.016% edits are inherently safe.

Primary source: https://rikiyat.github.io/blog/weight-space-programming/

### 3.2 Weight-Space Learning survey and generative modeling

Han et al., *A Survey of Weight Space Learning: Understanding, Representation, and Generation* (March 2026), provides a useful taxonomy:
- Understanding: properties, structure, and symmetries of weights.
- Representation: embeddings or latent representations of whole models.
- Generation: generating model parameters.

The ICML 2026 position paper *Weight Space Should Be a First-Class Generative AI Modality* argues that checkpoints form structured distributions and that adapter-scale / conditional generation is advancing, while unrestricted frontier-scale checkpoint synthesis remains open.

Implication: SWI should eventually learn a representation of *parameterized computational systems*, but should not begin with an unrestricted hypernetwork or large-checkpoint generator. Begin with named parameter groups and known mathematical invariants.

Sources:
- https://arxiv.org/abs/2603.10090
- https://proceedings.mlr.press/v306/wang26mc.html

### 3.3 Independent checks and early replications

An NYU Shanghai Research Institute of Trustworthy AI summary of early replications reports that a quantized Qwen 3.8 27B model identified three of four small game policies from weights alone but misidentified the fourth. A separate language-editing experiment reported targeted behavior changes in Gemma 4 12B-it with comparisons against a LoRA baseline. The same summary notes that the MIT report's large-language-model experiments use tools and forward passes, and that the main report is a blog post rather than a peer-reviewed paper.

Implication: separate "read weights without executing the network" from "edit weights with a tool harness and behavioral feedback." They are different capabilities and must receive different scores.

Source: https://rits.shanghai.nyu.edu/ai/weight-space-programming-works-beyond-gpt-6-astra-replications-show/

### 3.4 A safety-critical small-network audit

Nusrat and Nusrat's November 3, 2026 clinical-decision-network preprint tested networks with 433–4,801 parameters. The LLM identified 16 of 18 planted flaws, but also flagged 2 of 5 healthy networks. Protocol-based testing caught every flaw, and black-box testing found some flaws faster.

Implication: WSP is a candidate auditor and repair proposer, not a correctness certificate. False edits to healthy networks must be a first-class metric. Conventional black-box testing and explicit specifications remain mandatory.

Source: https://arxiv.org/abs/2611.05341

### 3.5 Mechanistic interpretability and model-editing baselines

MIB (ICML 2025) evaluates circuit localization and causal-variable localization, giving SWI useful neighboring evaluation concepts: precision, concision, causal relevance, and comparison across methods.

The ICML 2025 work *Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization* finds that mechanistic localization can improve robustness and reduce side effects compared with methods based mainly on output preservation.

Implication: SWI must not score an edit only by target-output change. It must measure whether the hypothesized mechanism was changed and whether preservation constraints remain satisfied.

Sources:
- https://proceedings.mlr.press/v267/mueller25a.html
- https://proceedings.mlr.press/v267/guo25k.html

## 4. Symthaea code audit: current integration points

The following audit is against the public main branch snapshot identified above and files returned by the connected repository.

### 4.1 CfC: best first target

Relevant files:
- src/dynamics/cfc/cell.rs
- src/dynamics/cfc/network.rs
- src/dynamics/cfc/types.rs
- src/dynamics/cfc/mod.rs

The CfC cell stores named groups including input weights, recurrent weights, biases, learnable time constants, and optional backbone weights/biases. The network's get_weights() serializes a deterministic sequence of these groups plus output projection weights and bias. The reserved per-cell w_out field is not part of that serialized trainable vector.

The network has analytic gradient / Adam paths. This gives SWI an unusually useful environment for comparing direct parameter edits with gradient-based repair.

**Concrete issue found:** the previous set_weights() copied the payload into live arrays before its final total-length assertion. A short payload could panic after partial mutation; an overlong payload could mutate the complete model and then panic at the final assertion. It also did not reject NaN or infinity before applying the payload.

**Change made on this branch:** set_weights() now delegates to a new try_set_weights() boundary. The new path:
- verifies exact total parameter count before the first write;
- rejects non-finite values;
- rejects time constants below MIN_TAU;
- preflights destination array contiguity;
- applies validated values only after all checks;
- reports typed CfCWeightLoadError variants;
- exposes a versioned CfC parameter manifest with stable typed group IDs, readable names, semantic roles, shapes, offsets, and lengths in get_weights() order;
- exposes transactional single- and multi-group replacement APIs using typed CfCParameterGroupId values, an exact base-model digest precondition, duplicate-target rejection, exact length/range checks, and one full-candidate validation before mutation;
- exposes a BLAKE3 fingerprint binding serialized architecture configuration, manifest, and exact little-endian f32 parameter bytes into separate and combined digests;
- routes asynchronous cognitive-loop weight swaps through try_set_weights() and rejects invalid updates without panicking;
- preserves the legacy set_weights() signature as a compatibility wrapper.

Regression tests are included across the CfC, HDC-LTC, structured-mixer/HRR, and async training modules. They cover manifest/layout; model fingerprints; typed and transactional edits; malformed payloads; initialization scale/orthogonality; HDC parameter profiling and bounded windows; adaptive one-step gradient finite differences (including Fourier, gate, tau, input and state paths); local target updates; and valid/invalid asynchronous swaps. The tests have not been executed here; do not describe them as passing until run on the exact branch head.

### 4.2 HDC-LTC: novel representation-space target

Relevant file:
- crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs

The unified HDC-LTC neuron stores hypervector parameters including:
- weight_hv
- input_mask
- tau_modulator
- gate_weight
- gate_bias
- momentum and running-statistic state

It exposes closed-form evolution and Hebbian / contrastive updates, and now has one-step local gradients for its equilibrium, adaptive gate, cosine-based tau modulation, input, and pre-evolution state. These are **not temporal BPTT**. The core and standalone `symthaea-hdc-ltc` implementations have different scalar versus per-coordinate tau/gate semantics and should remain explicit variants until a shared mathematical specification exists.

SWI-001 now exposes `HdcLtcParameterGroupId`, `HdcLtcNeuronParameterProfile`, bounded read-only parameter windows, and a validated replacement API for the five learnable hypervectors. The profile reports dimension, norm, mean, variance, extrema, non-finite count, and saturation count. Genesis initialization is now normalized and orthogonalized to match the unit-norm seeded constructor.

Do not flatten these hypervectors and pretend every coordinate has an independent neuron-level interpretation. The manifest/profile must record algebraic role, dimension, normalization contract, and binding position. See [SWI-002 Holographic Liquid Brain](./SYMTHAEA_HOLOGRAPHIC_LIQUID_BRAIN_SWI_002.md) for the full HDC/CfC architecture and benchmark programme, including comparisons to NIALIM/DFA-CfN, low-rank RNNs and the systematic 121-UCI HDC benchmark.

### 4.3 Broca: projection parameters and metamorphic updates

Relevant file:
- crates/domains/symthaea-broca/src/projection.rs

Broca's projection layer contains down/up and backward projection weights, optional deeper bottlenecks, expert-specific weights, and gradient diagnostics. It also has a metamorphic update function that targets named projection groups through a string selector.

CfC now exposes typed parameter-group IDs at the SWI boundary. Broca's existing metamorphic API still uses string selectors; a later Broca adapter should replace those at the SWI boundary with typed projection/expert identifiers. Diagnostics already track gradient norms, clipping, bottleneck norms, and variance; those are useful evidence signals, but they must not substitute for held-out language and semantic evaluations.

### 4.4 Existing interventions and source rewriting are adjacent, not equivalent

The existing causal ablation example (examples/causal_ablation_lobotomy.rs) performs activation-oriented interventions such as zero-out, noise, and shuffle. It is a useful precedent for counterfactual measurement, but it does not implement a general weight-space fault benchmark.

The Broca substrate rewriter (crates/domains/symthaea-broca-tools/src/substrate_rewriter.rs) modifies source code and gates writes behind an environment variable. Source-code rewriting and numerical parameter editing are different control planes. SWI must not bypass the source rewriter's guard or conflate a code patch with a weight patch.

## 5. Paradigm shifts worth pursuing

### Shift 1: from flat arrays to a typed computational substrate

A flat vector is an interchange format, not a model of what the parameters mean.

Build an architecture-aware manifest that names each parameter group and records:
- model and architecture fingerprints;
- parameter-group name, role, shape, storage order, dtype, and digest;
- which groups are trainable, frozen, optimizer state, recurrent state, or derived;
- numerical constraints (finite values, positive time constants, normalization bounds);
- algebraic role (matrix projection, recurrent map, HDC binding vector, gate, bias);
- known symmetries / equivalence transformations;
- allowed edit operators and maximum edit budgets.

This creates a small "parameter language" with explicit semantics. An LLM should not receive an undifferentiated billion-float dump when it can instead request typed summaries and targeted slices through a bounded inspection tool.

### Shift 2: from correlational diagnosis to causal repair

A candidate repair must state a mechanism hypothesis and a counterfactual prediction:
- What behavior is faulty?
- Which group is implicated?
- What intervention should change the behavior?
- Which unrelated behaviors should remain unchanged?
- What observation would falsify the hypothesis?

The verifier should use interventions and held-out traces to test the claim. Activation patching, ablation, permutation controls, and parameter edits should be composed rather than treated as competing philosophies.

### Shift 3: from raw weight distance to function- and symmetry-aware geometry

Raw Euclidean distance between weights is not generally a meaningful measure of behavioral change. Neural networks can have function-preserving permutations and rescalings. HDC representations have their own binding and normalization symmetries.

SWI should compare:
- parameter-space change;
- function-space change on probe distributions;
- activation/state-space change;
- architecture-specific invariants;
- downstream task and preservation changes.

A small edit is not automatically a small functional change. A large parameter edit is not automatically a large behavioral change.

### Shift 4: from patches to proof-carrying edit proposals

Every candidate should be a self-contained, replayable object containing:
- exact base model digest and architecture fingerprint;
- parameter manifest version;
- typed edit operations and before/after digests;
- a bounded mutation budget;
- causal hypothesis and expected effects;
- frozen validation-split identifiers and digests;
- positive, negative, and neutral controls;
- verifier outputs and acceptance decision;
- rollback pointer.

The edit proposer must not control the acceptance decision. A separate deterministic verifier applies candidates to a disposable clone, runs required checks, and emits a signed or content-addressed receipt.

### Shift 5: from brute-force testing to value-of-information allocation

The next research step after basic repair is not to maximize the number of forward passes. It is to spend evaluation budget where another observation could change the diagnosis or acceptance decision.

For each probe, estimate:
- expected reduction in uncertainty over fault hypotheses;
- probability the result changes the candidate ranking;
- cost of the probe;
- risk of overfitting the diagnostic set.

Stop when the evidence is sufficient to accept, reject, or abstain. Preserve anytime-valid uncertainty or a conservative fixed-look design; do not repeatedly peek at a test set and report an unadjusted best result.

### Shift 6: from self-optimization to bounded self-design

Longer term, SWI can compare alternative parameterizations and small architecture variants, transfer a verified functional module, or propose an architecture redesign. But this is a later track. A system that cannot reliably identify a planted defect in its current network has not earned permission to redesign or mutate its production substrate.

## 6. Proposed SWI architecture

### 6.1 Components

1. **Parameter Manifest** — architecture-aware schemas, group metadata, constraints, symmetries, and digests.
2. **Introspection Service** — bounded summaries, statistics, selected tensor slices, and activation/state-trace access.
3. **Fault Atlas** — deterministic fault injection, clean controls, and known ground truth.
4. **Mechanistic Auditor** — generates ranked hypotheses and abstains when evidence is insufficient.
5. **Edit Planner** — proposes typed, bounded changes without direct write access to the production model.
6. **Candidate Executor** — applies a proposal to a disposable copy through validated setters.
7. **Independent Verifier** — evaluates target improvement, preservation, stability, and negative controls.
8. **Evolution Ledger** — stores immutable baselines, proposal digests, receipts, and rollback references.

### 6.2 Contract objects

**ModelFingerprint**
- format version;
- model architecture identifier;
- configuration digest;
- parameter-manifest digest;
- complete parameter-value digest;
- source revision / build identity.

**ParameterGroup**
- stable group identifier;
- semantic role and mathematical type;
- shape and order;
- constraints;
- allowed edit operators;
- parameter digest.

**DiagnosticPacket**
- base fingerprint;
- task specification;
- failure description;
- activation and state traces;
- probe-set digest;
- summarized parameter statistics;
- explicit information withheld from the editor (including hidden fault seed and held-out tests).

**EditProposal**
- exact base fingerprint;
- list of typed operations;
- preconditions and postconditions;
- claimed mechanism;
- expected target effect;
- preservation constraints;
- maximum L0/L2 or group-level budget;
- deterministic replay metadata.

**VerifierReceipt**
- exact candidate fingerprint;
- evaluator and harness version;
- dataset/probe digests;
- baseline comparisons;
- target metrics and confidence intervals;
- collateral-regression metrics;
- stability and numerical checks;
- accepted / rejected / abstained verdict;
- rollback reference.

### 6.3 Initial edit operator set

Start with a very small allowlist:
- scale a named channel or row within a bounded factor;
- add or subtract a bounded low-rank direction;
- adjust one named bias group;
- rescale a time constant within explicit limits;
- apply an HDC direction update with norm and normalization constraints;
- restore a named tensor/group from an immutable known-good checkpoint.

Do not start with arbitrary tensor code, shell access, unrestricted file writes, or unconstrained direct mutation. All edit plans should be serializable, validated, and independently replayable.

## 7. Benchmark suite: SWI-Repair and extensions

The public field does not yet have a mature, universally accepted WSP benchmark. The MIT authors explicitly say they are building a benchmark and harness. SWI should publish a transparent benchmark specification and reproducible seeds, while avoiding premature claims of being first.

### 7.1 SWI-Repair: primary benchmark

**Protocol**

1. Train or construct a clean small network for a specified task.
2. Save an immutable clean checkpoint and compute its digest.
3. Generate a deterministic fault from a held-out fault seed.
4. Give the editor the faulty weights, architecture manifest, task specification, and permitted observations. Do not reveal the fault label, seed, clean checkpoint, or held-out cases.
5. Ask for fault localization, a mechanism hypothesis, and an edit proposal.
6. Apply the proposal only to a disposable candidate copy.
7. Run independent positive, negative, neutral, and held-out evaluations.
8. Compare with no-edit, random-edit, analytic, gradient, black-box, and oracle-repair baselines.
9. Preserve all prompts, outputs, artifacts, digests, metrics, and verdicts.

**First task families**

- delayed impulse / memory retention;
- damped oscillator or smooth dynamical prediction;
- piecewise temporal dynamics with irregular sampling intervals;
- input-channel discrimination;
- recurrent-state stability.

Use multiple task families. A repair system that only fixes one hand-designed pattern has not demonstrated general repair.

**Fault families for CfC**

- input-row ablation or attenuation;
- recurrent-row/column attenuation or a planted interfering direction;
- bias offset;
- time-constant collapse or extreme slowdown;
- dead or saturated hidden unit;
- backbone-layer perturbation;
- output-projection corruption;
- neutral hidden-unit permutation or other function-preserving reparameterization as a negative control;
- invalid payloads (short, long, NaN, infinity, below-minimum tau) for ingestion-hardening tests.

The last two categories are important: the system should distinguish functional faults from benign parameter rearrangements and should reject malformed data before mutation.

### 7.2 SWI-HDC: hypervector and dynamical-algebra track

Evaluate:
- binding/unbinding fidelity;
- bundling similarity and retrieval;
- semantic-neighbor preservation;
- interference / capacity under multiple bindings;
- norm and distribution stability;
- state evolution under irregular time steps;
- multi-horizon prediction;
- temporal-order preservation;
- robustness to coordinate permutations that should preserve the defined algebra.

Seeded HDC faults should be designed from the actual operation semantics, not from an arbitrary notion that one coordinate is a human-interpretable neuron.

### 7.3 SWI-Broca: projection and multilingual track

Evaluate:
- projection reconstruction / round-trip error;
- semantic retrieval and contrastive separation;
- expert-routing balance;
- representation collapse;
- language-specific transfer;
- preservation across isiZulu, isiXhosa, Sesotho, Afrikaans, and English;
- catastrophic forgetting and cross-language interference;
- text-level repetition and semantic consistency.

Separate missing knowledge from a repairable obstruction. A weight edit should not receive credit for "repairing" a capability that the base system did not possess.

### 7.4 SWI-Read: weights-only interpretation

Keep a separate track for claims made without forward passes or activation traces:
- infer architecture/task family where possible;
- identify known structural properties;
- detect planted parameter-level defects;
- distinguish a healthy model from a faulty one;
- calibrate uncertainty and abstain.

Never merge this score with a trace-assisted repair score. The MIT report itself uses different information budgets across its experiments.

### 7.5 SWI-Longitudinal: multi-round improvement

Only after SWI-Repair is reliable:
- run sequential diagnosis/repair rounds;
- record every accepted and rejected candidate;
- evaluate cumulative target improvement;
- measure regression accumulation;
- test rollback and recovery;
- use an untouched final evaluation set;
- compare against one-shot repair and ordinary retraining.

Do not let a sequence of edits repeatedly optimize against the same visible validation set.

## 8. Metrics and proposed research gates

### Localization and diagnosis
- fault-class precision, recall, F1, AUROC/AUPRC;
- exact parameter-group localization and top-k recall;
- distance between predicted and planted fault support;
- mechanism-hypothesis accuracy;
- confidence calibration (Brier score / calibration error);
- abstention coverage-risk curve.

### Repair quality
- repair success rate on held-out fault seeds;
- target metric improvement against the clean reference;
- false-edit rate on healthy controls;
- false-repair rate (edits that change behavior without correcting the planted cause);
- collateral regression across unrelated tasks;
- numerical stability and bounded-state violations.

### Edit efficiency
- changed-parameter count (L0);
- relative L1/L2 movement;
- changed parameter groups;
- number of forward passes / probes;
- total compute, latency, memory, and model-call cost;
- performance improvement per unit cost.

### Reproducibility and auditability
- exact base and candidate digests;
- bitwise or tolerance-defined replay agreement;
- complete proposal and evaluator logs;
- seed and split integrity;
- independent rerun agreement;
- rollback correctness.

### Proposed initial gates (targets, not results)

| Gate | Initial target |
|---|---:|
| Fault recall | ≥90% point estimate; report confidence interval by fault family |
| Fault localization precision | ≥95% point estimate |
| Repair success | ≥80% on held-out injected faults |
| Healthy-control false-edit rate | ≤1%; use at least 300 independent healthy instances and report an exact binomial interval |
| Catastrophic preservation regressions | 0 observed; any occurrence blocks acceptance pending root-cause analysis |
| Target improvement | Statistically supported on held-out paired seeds, versus the strongest relevant baseline |
| Replay | 100% agreement for deterministic transformations and exact artifact digests |
| Malformed parameter input | 100% rejected before any live parameter mutation |

These targets must be preregistered before final evaluation. Report confidence intervals and sample counts, not percentages alone. If the data do not support a gate, say so.

## 9. Baselines and industry comparison

Every benchmark report should include the following baselines on the same architecture, splits, and compute budget where possible:

1. **No edit** — establish the underlying failure and natural variance.
2. **Random matched edit** — match edit count and norm to test whether the editor's reasoning matters.
3. **Simple analytic detector/repair** — especially for planted faults with known mathematical signatures.
4. **Gradient-based fine-tuning / BPTT** — conventional supervised repair when training data and gradients are permitted.
5. **Black-box search** — random search, NES, or CMA-ES when parameter count permits.
6. **Mechanistic localization and intervention** — use activation/state traces and causal ablations.
7. **LLM-only WSP** — parameter summaries or raw weights with no causal trace assistance.
8. **Trace-assisted WSP** — parameter inspection plus permitted forward-pass observations.
9. **Hybrid SWI** — typed manifest, adaptive probes, causal tests, bounded edit proposal, independent verifier.
10. **Oracle repair** — the smallest known inverse edit or clean checkpoint, used only as an upper bound.

For Broca, add LoRA/fine-tuning and existing metamorphic/gradient updates. For HDC-LTC, compare to the existing Hebbian and contrastive update methods. For CfC, compare to its analytic gradient/Adam update and black-box parameter search.

**What the industry currently demonstrates:** the MIT report is the most visible public direct-WSP demonstration in this narrow period; WSL survey and generative-weight work provide the wider technical context; MIB and mechanistic model-editing work provide evaluation methodology. The early results do not establish a mature industrial standard or a robust general-purpose weight programmer. SWI should compete on reproducibility, causal validity, domain breadth, and lower regression—not on a cherry-picked headline score.

## 10. Research hypotheses

- **H1 — typed manifests:** Architecture-aware parameter metadata improves localization and edit success relative to flat weight dumps.
- **H2 — causal evidence:** Adding targeted activation/state traces and counterfactual interventions reduces false edits relative to weights-only diagnosis.
- **H3 — temporal priors:** Explicit continuous-time constraints improve repair of CfC temporal defects and generalization to irregular dt.
- **H4 — HDC-native geometry:** HDC-specific operators outperform generic coordinate edits on HDC algebraic invariants.
- **H5 — selective evaluation:** Adaptive probe allocation reaches equal or better repair decisions with fewer forward passes than fixed-budget evaluation, while retaining valid uncertainty.
- **H6 — proof-carrying changes:** Independent verification and immutable receipts reduce undetected regressions in sequential repair.
- **H7 — cross-task transfer:** A parameter-space representation learned across small networks can improve diagnosis on unseen seeds and related architectures without increasing healthy-model false edits.

H1–H6 are the initial programme. H7 is a later WSL research track and should not be presented as an expected near-term result.

## 11. Implementation plan

### P0 — safe parameter boundary (started)
- Add typed CfC weight-load validation.
- Validate exact length, finite values, tau floor, and contiguous storage before mutation.
- Preserve existing infallible API as a compatibility wrapper.
- Add a versioned named parameter manifest matching the flat serialization layout.
- Add bounded typed named-group replacement and digest-bound transactional batches through the validated loader.
- Add content-addressed architecture, manifest, parameter, and combined model fingerprints.
- Route asynchronous weight swaps through the fallible API and test both valid and malformed updates.
- Add regression tests.
- Next: execute targeted unit tests and full relevant crate checks on the exact branch head.

### P1 — manifest and immutable snapshots
- Add a CfC parameter manifest with stable group IDs and exact offsets/shapes.
- Add architecture/configuration fingerprint and value digest.
- Expose read-only summaries and bounded parameter slices.
- Ensure manifest offsets reconstruct get_weights() exactly.
- Add round-trip tests and digest stability tests.

### P2 — fault atlas and evaluator
- Create deterministic task generators and planted faults.
- Keep hidden fault seeds and clean checkpoints out of the editor input.
- Add negative and neutral controls.
- Add no-edit, analytic, random, gradient, and oracle baselines.
- Report confidence intervals and held-out results.

### P3 — extend architecture-aware editing
- HDC-LTC now has typed parameter-group IDs, profiles, bounded windows, and validated replacement for its five learnable hypervectors.
- Structured orthogonal mixing and unitary HRR circular convolution are now available as independent primitives; benchmark and task-level ablations are next.
- Validate HDC local gradients, zero-time identity across integrators, and FFT/HRR round trips on the exact branch head.
- Define typed Broca projection/expert group IDs.
- Replace string target selection at the SWI boundary with typed selectors.
- Add model-specific preservation suites.

### P4 — editor interface
- Define a provider-agnostic proposal interface.
- Allow LLMs to inspect summaries and request bounded tensor windows.
- Do not require any one proprietary model.
- Keep execution and verification in deterministic Rust.
- Enforce size, time, token, probe, and mutation budgets.

### P5 — staged evolution
- Add candidate queues, acceptance policies, receipts, and rollback.
- Permit only offline candidate generation and evaluation initially.
- Do not enable online production self-editing.
- Graduate one subsystem at a time after repeated independent validation.

## 12. Safety and release policy

- Production weights are immutable to the editor.
- An edit proposal must match the exact current model digest; stale proposals are rejected.
- Every proposal is applied to an isolated candidate copy.
- Exact base digest is a hard precondition; stale proposals are rejected.
- All values must be finite and all architecture-specific invariants must pass.
- No silent clamping of a malformed candidate; reject and report.
- The proposer cannot alter tests, evaluation data, acceptance thresholds, or the verifier.
- Use separate train/diagnostic/validation/final-test splits.
- Keep at least one hidden evaluation set.
- Include neutral transformations and healthy controls.
- A verifier must be independent of the candidate proposer; use separate code paths and preferably separate models/teams for scientific audits.
- Any high-severity regression blocks acceptance, even if the target score rises.
- Rollback must restore the exact prior digest.
- IIT/Φ, consciousness proxies, or self-reported confidence are not acceptance objectives for parameter editing. They may be separately reported as exploratory telemetry, never as substitutes for task and safety outcomes.

## 13. First concrete experiment

**SWI-CFC-001: repair a planted temporal-constant collapse**

1. Construct a small seeded CfC network on a synthetic temporal prediction task.
2. Train or configure the clean network to meet a preregistered minimum held-out performance gate.
3. Save the clean network and its digest as an oracle-only artifact.
4. Create independent corrupted copies by collapsing selected tau values toward the minimum stability floor or by driving a subset to an excessive timescale.
5. Give the editor the faulty model, manifest, task description, and a small diagnostic trace budget. Keep the clean weights and fault labels hidden.
6. Require it to localize affected cells/neurons and provide an edit proposal with a predicted effect on fast and slow temporal response.
7. Execute the candidate in isolation through try_set_weights().
8. Evaluate on new sequences, irregular sampling intervals, multiple horizons, and unrelated temporal tasks.
9. Compare to no edit, random matched edits, simple tau-statistic repair, gradient repair, and oracle inverse edit.
10. Repeat over preregistered clean seeds and fault seeds.

**Primary question:** can a typed, evidence-assisted weight-space programmer repair temporal behavior better than simple statistical and gradient baselines while avoiding changes to healthy networks?

This is a deliberately modest first experiment. It tests whether SWI has a genuine advantage rather than assuming it does.

## 14. Acceptance checklist for the first benchmark release

- [ ] Exact code revision and build identity recorded.
- [ ] Clean checkpoint and faulty checkpoint digests recorded.
- [ ] Parameter manifest versioned and independently reconstructed.
- [ ] Fault generators deterministic and validated against clean controls.
- [ ] Fault labels and final test cases hidden from the editor.
- [ ] All baselines use the same split and comparable information budget.
- [ ] Candidate edits are bounded and replayable.
- [ ] Invalid candidates are rejected before mutation.
- [ ] Target improvement and collateral regression both reported.
- [ ] Confidence intervals and sample sizes included.
- [ ] Full prompt/tool/evaluation logs retained.
- [ ] Independent rerun succeeds.
- [ ] No claim of SOTA or recursive self-improvement until evidence supports it.

## 15. References

1. Takehi, R., Teoh, J., Zhang, A., and Khattab, O. (2026-11-02). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* https://rikiyat.github.io/blog/weight-space-programming/
2. Han, X. et al. (2026-03-10). *A Survey of Weight Space Learning: Understanding, Representation, and Generation.* https://arxiv.org/abs/2603.10090
3. Wang, Z., Wang, P., and Wang, K. (ICML 2026). *Position: Weight Space Should Be a First-Class Generative AI Modality.* https://proceedings.mlr.press/v306/wang26mc.html
4. Nusrat, H. and Nusrat, O. (2026-11-03). *Weight-space programming for auditing clinical decision networks.* https://arxiv.org/abs/2611.05341
5. Mueller, A. et al. (ICML 2025). *MIB: A Mechanistic Interpretability Benchmark.* https://proceedings.mlr.press/v267/mueller25a.html
6. Guo, P. H. et al. (ICML 2025). *Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization.* https://proceedings.mlr.press/v267/guo25k.html
7. NYU Shanghai RITS (2026-11-04). *Weight-Space Programming Works Beyond GPT-6 Astra, Replications Show.* https://rits.shanghai.nyu.edu/ai/weight-space-programming-works-beyond-gpt-6-astra-replications-show/

---

**Decision:** proceed with SWI as a research programme, not as an unrestricted self-modification feature. The next decisive milestone is a reproducible CfC/HDC-LTC fault-and-dynamics benchmark with architecture-aware manifests, transactional typed edits, finite-difference verification, and an independent verifier.