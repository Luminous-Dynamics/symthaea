# Evidence-Gated Belief Transitions for Scientific Experiments (EGBT) v1

**Status:** executable reference contract / research prototype  
**Parent:** [SDCB scientific-discovery interoperability v1](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md)  
**Implementation:** [scripts/research/sdcb_belief_transition.rs](../../scripts/research/sdcb_belief_transition.rs)  
**Wire contract:** [SDCB_EXPERIMENT_TRANSITION_V1.schema.json](SDCB_EXPERIMENT_TRANSITION_V1.schema.json)

## 1. Research thesis

Active experiment selection is only half of scientific intelligence. A system can choose a useful experiment and still corrupt its knowledge state by accepting a stale, misbound, duplicated, unverified, or unauthorized observation.

EGBT treats a scientific belief update as a typed state transition:

**frozen hypotheses → explicit experiment proposal → source-bound observation → independent verification → recomputable posterior update**

The transition is accepted only when the observation is bound to the exact run, task, system snapshot, ordered hypothesis set, experiment proposal, prior version, and cost budget. Rejected transitions leave the belief state and budget unchanged.

This changes the unit of scientific evaluation. We do not evaluate only the answer or the experiment selected; we evaluate whether the path from a hypothesis to a revised belief is justified and replayable.

## 2. Core mathematical contract

Let the finite hypothesis set be H = {h_1, ..., h_n} with prior distribution p_i.

An experiment e has candidate outcomes o_1, ..., o_m and a likelihood matrix:

\[
L_{ij} = P(o_j \mid h_i,e)
\]

Each row must be a valid probability distribution. The predicted outcome probability is:

\[
P(o_j \mid e)=\sum_i p_i L_{ij}
\]

For an outcome o_j, the posterior is:

\[
p_i' = \frac{p_i L_{ij}}{\sum_k p_k L_{kj}}
\]

The reference selector computes expected information gain:

\[
IG(e)=H(p)-\sum_j P(o_j\mid e)H(p(\cdot\mid o_j,e))
\]

and selects the candidate with greatest IG(e) / cost(e). Ties are deterministic. If all candidate experiments have zero information gain, the selector abstains rather than inventing a useful probe.

This is a deliberately small finite-hypothesis oracle. It is not a general Bayesian inference engine, causal discovery algorithm, or decision-theoretic planner. A later extension should compare information gain with expected decision value, because information can be scientifically interesting but irrelevant to the decision at hand.

## 3. State-transition and evidence-binding rules

An accepted proposal and observation must agree on:

- run ID and task ID;
- task, system, and ordered hypothesis-set digests;
- experiment ID and proposal digest;
- prior ledger version;
- experiment cost and charged cost;
- observed outcome index.

The verifier identity must be present and the receipt must indicate independent verification. The ledger checks all bindings before mutation. It also enforces:

- posterior normalization and valid likelihood rows;
- observation has nonzero predictive probability under the current model set;
- no repeated experiment ID (consume-once);
- no stale proposal version;
- no budget underflow;
- monotonically incremented ledger version.

**Atomicity invariant:** any rejected transition leaves posterior, version, budget, and accepted-transition count unchanged.

The implementation records the prior, posterior, predictive probability, verifier identity, observation digest, and budget before/after in the returned transition record. A replay tool can independently recompute the posterior from the frozen prior, likelihood table, and outcome.

## 4. Explicit trust boundary

The Rust reference harness is an analytic contract, not production attestation infrastructure.

- The independently_verified boolean is a test seam representing a decision made by a separately governed verifier. An untrusted agent must never be allowed to set it.
- Digest fields are checked for lowercase SHA-256 hexadecimal shape. The harness does not hash observation bytes, verify a signature, authenticate a verifier, or prove custody.
- A verifier saying an observation is authentic does not prove that the sensor is calibrated, the experimental protocol was scientifically valid, or the causal interpretation is correct.
- The implementation does not include persistent append-only storage, revocation, key rotation, trusted timestamps, or cross-machine replay.
- A production system must consume a typed verifier attestation with authenticated verifier identity, signed source binding, custody metadata, calibration/protocol references, and revocation checks. It must persist both accepted transitions and rejected attempts in an append-only audit stream.

The intended production boundary is:

**Symthaea proposes and predicts; an authorized executor performs; an independent verifier authenticates evidence; the belief service applies the update.**

No component should be permitted to both fabricate an observation and unilaterally certify it.

## 5. Why this is distinct from existing scientific-agent benchmarks

The current research landscape already evaluates adaptive experiment planning, mechanism recovery, scientific campaigns, and active causal discovery. EGBT does not attempt to replace those suites.

It adds a cross-cutting assurance layer:

1. Was the observation captured from the exact experiment that was authorized?
2. Does the receipt refer to the exact task, system, and hypothesis version?
3. Was the evidence independently verified before it influenced the posterior?
4. Can the belief update be recomputed from the declared prior, likelihood model, and observation?
5. Are invalid, stale, duplicate, misbound, or over-budget observations rejected without state mutation?

This contract can be applied to synthetic dynamical systems, kinetic mechanisms, gene-regulatory networks, material campaigns, causal discovery, and physical testbeds.

## 6. Benchmark integration and external comparators

Do not create a standalone leaderboard that duplicates upstream datasets. Use the same tasks and budgets where licensing and API terms allow, and publish EGBT results as a separate evidence-transition report.

| Upstream benchmark | What it already tests | EGBT contribution |
|---|---|---|
| Beyond Accuracy (2026) | Belief update, hypothesis discrimination, experiment design, predictive reasoning under intervention | Evidence-to-posterior binding, stale/duplicate rejection, recomputable update |
| ActiveSciBench / LLM-AutoSciLab | Budgeted mechanism discovery in enzyme kinetics and gene-regulatory networks | Verified observation binding and update integrity |
| Autonomous Science Bench 0.1 | Real-lab design-build-test-learn campaigns; discovery and held-out learning gates | Source-bound measurement receipts and independently replayable belief changes |
| Active Causal Discovery Benchmark | Skeleton/DAG recovery, interventions, intervention efficiency, Markov-equivalence limits | Explicit boundary between identifiable structure and unresolved mechanism; authenticated intervention evidence |
| AutoSciBench | Adaptive generation/refinement of scientific evaluation tasks | Regenerate assurance challenges as agents saturate public cases |
| CoDBench | Data-driven learning of continuous physical dynamics across PDE tasks | Apply transition receipts to physical observations; retain CoDBench as the physical-model performance suite |

Preserve each upstream benchmark's task definition and score. Any EGBT metric is an additional result, not a redefinition of upstream performance.

## 7. Metrics: evaluate the transition, not just the answer

Report these independently:

### Scientific quality
- mechanism identification accuracy, conditioned on identifiability;
- false mechanism-claim rate on non-identifiable cases;
- observable prediction error when mechanism recovery is impossible;
- expected information gain versus realized information gain;
- posterior predictive log score and Brier score;
- calibration and selective risk/coverage;
- held-out counterfactual prediction and intervention fidelity.

### Experiment efficiency
- oracle-normalized experiment-selection regret;
- measurement cost to reach a preregistered decision threshold;
- simple regret and cumulative regret versus budget;
- discovery and unmeasured-candidate learning as separate gates.

### Evidence assurance
- unauthorized belief transitions (must be zero on the frozen negative-control suite);
- stale, duplicate, wrong-task, wrong-system, wrong-proposal, unverified, or over-budget receipt detection;
- false rejection rate for valid independently verified receipts;
- exact replay agreement for posterior and budget transitions;
- time and cost to reconcile ambiguous or conflicting evidence.

### Deployment
- wall-clock latency, CPU/GPU time, peak memory, tool calls, and cost;
- calibrated energy measurement only where hardware telemetry supports it;
- failure and recovery costs, not only successful-run costs.

Do not collapse these metrics into one weighted score. A perfect integrity score must not hide useless experimentation; high discovery performance must not compensate for accepting unauthorized evidence.

## 8. Provisional acceptance targets

These are project targets, not measured outcomes or industry standards:

- 100% exact posterior agreement on analytic reference fixtures.
- 100% rejection of the frozen invalid-receipt controls with zero state mutation.
- Zero unauthorized belief updates on confirmatory negative controls.
- At least 99% acceptance of valid receipts in a separately generated validity-control set (to prevent trivial reject-all behavior).
- No false mechanistic claims on exact observational-equivalence controls.
- Experiment regret reported against an exact planner on every tractable finite fixture.
- At least five independent seeds and one held-out generating family before promoting an exploratory result.
- Independent replay by a second implementation or machine before making a reproducibility claim.

A target miss remains a result. Do not weaken the gate after observing test outcomes.

## 9. Paradigm shifts enabled

### Scientific state transitions become first-class objects
A paper, model response, or chain-of-thought trace is not the durable unit of knowledge. The durable unit is a typed transition from a versioned hypothesis state to a new state, justified by an independently verifiable observation.

### Uncertainty becomes operational
The system can distinguish "unknown because no experiment has been run", "unidentifiable under the current sensors", "uncertain because evidence is noisy", and "contradicted by a verified observation". These states should not be collapsed into a single confidence score.

### Falsification becomes a planning primitive
Every hypothesis should carry predictions that could disconfirm it. The system can then choose an experiment that maximizes disagreement among plausible hypotheses, not merely one that gathers more data.

### Evidence integrity becomes orthogonal to scientific correctness
A correctly authenticated observation can still support a bad model; a good-looking model can still be based on corrupted evidence. Score both axes separately.

## 10. Luminous integration and evidence-class separation

The contract is intended to span the Luminous architecture, but this PR does not claim that these integrations are implemented.

| Component | Candidate responsibility | Explicit boundary |
|---|---|---|
| Symthaea | Maintain competing hypotheses, predict outcomes, select a probe, compute posterior beliefs | Must not certify its own observation or silently change the frozen task/system identity |
| Symtropy | Execute isolated simulations and, where separately integrated, coordinate controlled physical experiments | A simulation result must remain labelled simulation; it is not a physical measurement |
| Mycelix | Principal identity, delegated authority, verifier/key registry, revocation and auditable attestation references | Trust/identity consensus does not establish scientific truth or authorize a model to self-verify |
| Independent verifier | Validate run identity, experiment binding, instrument/protocol receipt, observation provenance and signer authority | Must be separated from the proposing agent and execution principal |
| Evidence execution architecture | Carry intent → authority → execution → observation → independent verification → replay | Unknown execution outcome blocks new mutation until reconciliation |

### Evidence classes must never collapse

The next wire-contract version should explicitly carry an evidence-origin classification:

- **analytic** — exact mathematical controls and hand-computable oracles;
- **synthetic** — generated observations with known ground truth;
- **simulated** — outputs from a pinned simulator and configuration;
- **physical** — instrument-captured observations with calibration, acquisition and custody metadata.

The class, environment digest, authorized-execution receipt, protocol digest, and calibration/instrument digest should be bound into the accepted transition. A simulated pass cannot satisfy a physical-evidence requirement. A physical observation can still be scientifically inconclusive or inconsistent with the model.

For production, the verifier's signed attestation should bind these fields, including a verifier key identifier and revocation check. The current v1 Rust source and schema do not yet encode this complete attestation or evidence-origin model; it is a versioned integration requirement, not a completed feature.

## 11. Implementation and validation status

The reference source is standard-library-only and independent of the production Symthaea workspace. It implements:
- finite-hypothesis expected information gain;
- cost-normalized experiment selection with deterministic tie-breaking and abstention;
- exact Bayesian posterior update for a discrete outcome;
- identity, digest-shape, hypothesis-order, version, cost, budget, and consume-once checks;
- atomic rejection semantics;
- unit tests for positive, negative, stale, duplicate, misbound, and impossible-observation cases.

**Current source state:** 23 Rust unit tests are authored. The container used for this work has no rustc or cargo; compilation and execution have not been performed. The Draft 2020-12 schema passed meta-schema validation; one valid receipt fixture passed, an invalid digest fixture was rejected, and posterior/version/budget cross-field checks passed in a separate reference script. The existing SDCB reference receipt runner has not yet been extended to execute this new harness.

This prototype does not establish that Symthaea has active scientific intelligence, beats any SOTA system, or can verify physical evidence. It defines and tests a narrow contract once a Rust toolchain is available.
