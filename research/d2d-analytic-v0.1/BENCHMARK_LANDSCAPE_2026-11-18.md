# D2D Benchmark Landscape and Competitive Plan
## Research snapshot: 2026-11-18

**Purpose:** choose external comparators and define what Dream-to-Discovery must measure beyond existing leaderboards.  
**Status:** research synthesis; reported results are source-reported, not Luminous reproductions.

---

## 1. Central conclusion

Do not compete with every world-model laboratory on visual realism or one manipulation success percentage. The leading systems and benchmarks measure different capabilities.

The defensible opportunity for Symthaea + Symtropy + Mycelix is to evaluate a complete loop:

> identify a knowledge gap → choose a discriminating experiment → seal predictions → execute an intervention → evaluate against independent outcomes → update a candidate model or skill → measure transfer, retention and cost.

Do not claim that D2D is the first autonomous experiment-selection or closed-loop science system. Autonomous Science Bench 0.1 already evaluates agents that choose budgeted experiments, receive measurements, improve decisions, and submit models scored on designs that were never measured. The defensible D2D research question is narrower and more technical: can we integrate action-conditioned physical counterfactuals, certified uncertainty-bounded branch discrimination, prediction-before-reveal, decision utility, recurrent-memory retention, held-out transfer, and reproducible evidence into one benchmarked loop? Novelty must be established against these adjacent systems, not asserted from the architecture diagram.

## 1.1 Existing Luminous capabilities to reuse

The repository audit found specialized owners that make the proposed programme more precise:

- **Symtropy #3406 — PHYS-CONTACT-ACTIVE-EXPERIMENT-001:** exact worst-case experiment design to separate declared contact branches using branch-conditioned interval sensitivities, a declared observable/noise/resolution profile, finite-step validity, and explicit NOT_SEPARATED outcomes. Its P0 scope is the two certified double-slide branches in #3365. It does not claim global branch completeness or arbitrary physical parameter identification.
- **Symthaea #9817 — EDGE-ACTIVE-SENSING-001:** decision-focused value-of-information sampling under a complete resource vector, with hidden outcomes kept from the candidate.
- **Symthaea #9010 — CMAQ-004:** matched direct-policy versus model-assisted decision utility, including stale-model and shuffled-model controls.
- **Symtropy #3183 — PHYS-CONTACT-ASSURANCE-BENCH-001** and #3749 — PHYS-VALID-07: contact assurance and physics verification authorities.

D2D must consume these owners rather than creating another local sensitivity optimizer, active-sensing scheduler, contact solver, or physics verifier.

### The key distinction

There are two complementary experiment-selection problems:

1. **Probabilistic decision-focused value of information:** is another observation expected to change the decision enough to justify its resource cost? This is the Symthaea #9817 / existing EVoI line.
2. **Deterministic certified branch separation:** is there an admissible perturbation for which the entire declared uncertainty-bounded branch outcomes become measurably disjoint? This is the Symtropy #3406 line.

They are not interchangeable. D2D can compare both, but it must never call a heuristic score or probabilistic information gain an exact interval guarantee. If interval sets overlap or noise consumes the gap, the correct result is INCONCLUSIVE / NOT_SEPARATED.

Also preserve the physical correspondence boundary: a local branch-gradient certificate is not itself a finite executed trajectory or a real-world measurement. Finite-step separation requires the branch-tube/remainder path (Symtropy #3386); time-domain rigid-body execution and replay must be evaluated in a separately matched lane.

## 1.2 First Symtropy adapter and immediate engineering gates

A concrete downstream execution task is now open:

- [Symtropy #5975 — D2D-SYMTROPY-ADAPTER-001](https://github.com/Luminous-Dynamics/symtropy/issues/5975) will use the existing native `symtropy-physics::PhysicsWorld<3>` plus `ReplayTape`, `WorldSnapshot` and the bitwise replay harness for a small paired contact-probe experiment.
- It is an execution/replay adapter, not a new simulator. The first controlled world uses a declared floor/body geometry, mass, material/friction, timestep, solver settings and intervention. Rebuilt branches must start from the same frozen manifest.
- The existing replay snapshot asserts equality for body state and last-step collision events. It does not capture every private solver/contact-cache field; the adapter's claim must be scoped to the public snapshot contract.
- D2D uses the analytic toy fixture as a protocol baseline; the Symtropy adapter then replaces only the analytic executor. The analytic fixture must not be treated as physical validation.

For certified branch-separating contact experiments, use Symtropy #3406. Important status distinction: #3365, which provides the two-branch six-coordinate sensitivity witness used by #3406's P0 plan, is currently an open draft PR with no passing combined status reported in the audit. It is not yet a qualified main-branch dependency. The finite-step branch-tube owner #3386 remains distinct.

HLB recurrent-learning continuity is another gate: Symthaea #1269 remains the canonical owner for versioned HDC-LTC episode-state checkpointing. Draft PR #12757 adds an atomic in-memory restore API that records ordered layer sizes and validates topology, counts, dimensions, and finite state before mutation. It deliberately preserves the legacy best-effort scratch-buffer path for existing callers. The new API is not yet Rust-tested, is not serialized, and does not bind the recurrent state to learned-weight/optimizer identity. Portable checkpoints and fresh-process replay remain unqualified; do not claim them.

The existing Symtropy manipulator-demo Monte Carlo example is not an acceptable D2D SOTA baseline as-is: its low-cost Adaptive arm is explicitly a proximity-keyed deterministic stand-in, not the actual cognitive policy, and its default comparison uses 100 s for two arms versus 40 s for the Φ arm. A future use must first be reworked into a matched-policy, matched-horizon protocol.

## 2. Current external benchmark and industry map

| System / benchmark | Main capability measured | What it does not establish by itself | D2D use |
|---|---|---|---|
| **Cosmos Policy (NVIDIA, ICLR 2026)** | Video-model post-training for robot action generation, future-state/value prediction and test-time planning | A generally better research curriculum, verified discovery or cross-domain learning | Strong learned-policy/world-model baseline on compatible robot tasks |
| **World-In-World (ICLR 2026 Oral)** | Closed-loop utility of generated futures across four embodied environments, with a shared action API | Hidden mechanism discovery, independent scientific replication or memory retention after learning | Main external protocol anchor for prediction-to-action value |
| **MiraBench (2026)** | Physical consistency, action-following fidelity and optimism bias in action-conditioned robotic world models; more than 16,000 human judgments and 12 model configurations | Whether a model improves a complete closed-loop controller or selects informative experiments | Reliability diagnostics and negative controls for over-optimistic imagined outcomes |
| **RoboWM-Bench (CVPR 2026 Workshop)** | Converts predicted manipulation videos into executable action sequences and validates them in reconstructed simulation / real-to-sim cases | General-purpose scientific discovery or broad sim-to-real qualification | Test whether dreamed trajectories are physically executable, not just plausible |
| **MIND (2026)** | Long-context scene/memory consistency and action control across 250 videos, viewpoints and action spaces | Physical robot dynamics or verified causal intervention effects | Long-horizon memory and action-space consistency tests |
| **XEWorld (2026)** | Action-conditioned world-model generalization to held-out robot embodiments | General scientific learning or broad real-robot competence | Held-out embodiment and catastrophic-forgetting tests |
| **Autonomous Science Bench 0.1** | End-to-end agent performance on six real scientific workflow tasks | Broad scientific intelligence; its initial pilot is small and domain-limited | External research-agent comparator and cost/time reporting reference |
| **Genie 4 (Google DeepMind, Oct 27 2026)** | Real-time interactive, multimodal world generation and user-directed scene changes | Stable long-session physics, verified numerical state or independent execution truth | Generative scenario/observation proposal baseline, not the physics or evaluator authority |

### 2.1 Reported comparator figures and caveats

- NVIDIA's Cosmos Policy page reports **98.5% mean success on LIBERO** and **67.1% on RoboCasa** for the reported configuration. NVIDIA reports strong results using 50 RoboCasa demonstrations per task; the comparison must still match task splits, demonstration sources, action interface, compute and evaluator before any direct head-to-head claim.
- World-In-World reports that action-conditioned post-training improved example success rates (for example, Wan2.1 from 60.25% to 63.34%) and that increasing inference rollouts from 3 to 11 improved an example SVD result from 53.36% to 60.98%. These are results on its own tasks and protocol, not D2D baselines.
- Autonomous Science Bench 0.1 lists six pilot tasks. Its current leaderboard reports 72.2% resolution for GPT-6 Astra + Codex at $190.23 and 557 agent-minutes, and 66.7% for Opus 5.5 + Claude Code at $172.91 and 1,142 agent-minutes. This is a pilot leaderboard, not a universal ranking of research intelligence.
- Genie 4 is a research preview. Google DeepMind explicitly identifies long-session visual drift, object permanence, complex physical interaction, spatial accuracy and audio accuracy as active limitations.

Every number above is comparator context, not a reproduced Luminous result. Do not pool percentages from incompatible benchmarks.

### 2.2 Version-aware frontier snapshot and benchmark coverage

- **Cosmos Policy versioning matters.** NVIDIA's current Cosmos Policy results page reports 98.5% mean LIBERO success and 67.1% RoboCasa success for its displayed configuration. NVIDIA's Cosmos Cookbook separately reports a Predict2.5 configuration at 98.33% LIBERO and 71.1% RoboCasa, compared with its Predict2 entry at 98.5% and 67.1%. Preserve these as distinct published configuration rows; do not silently relabel the 71.1% result as the same model/configuration as the 67.1% page result. Any reproduction must pin the exact checkpoint, training demonstrations, tasks, evaluation code, action interface and seeds.
- **The world-model/action gap is independently corroborated.** A separate 2026 survey maps 160 embodied-prediction benchmarks and reports that only 11 make an explicit VLA-versus-world-model contrast and only four turn prediction into executed action. This is a different corpus and methodology from the 130-benchmark world-model survey above; the two counts must not be added or treated as one dataset.
- **Autonomous Science Bench is a direct comparator, not just inspiration.** Its v0.1 leaderboard covers six pilot campaigns, ten model-agent configurations and 180 trials (three trials per campaign/configuration). The top listed GPT-6 Astra + Codex configuration resolves 72.2% of trials at $190.23 and 557 agent-minutes; the authors caution that three trials per task make small leaderboard gaps non-significant, and the hardest campaign is solved in only five of 30 attempts. Its two gates explicitly separate discovery among measured designs from predictive ranking on unmeasured designs. D2D should adopt that discovery/learning separation while adding the physics/action and evidence-assurance dimensions.

**Competitive interpretation:** we should not try to beat Cosmos on LIBERO/RoboCasa by citing a different small toy simulation. The first credible win is a matched, held-out experiment-design result: better decision quality per total resource cost than random, tuned Bayesian optimization/active learning, and the strongest reproducible model-based baseline—while maintaining calibrated uncertainty and complete branch accounting. Only then should the same frozen method be tested as a policy/world-model component on a named robotics benchmark.

## 3. The important research gaps

The 2026 survey of 130 world-model benchmarks classifies evaluation by target, open- versus closed-loop protocol, prediction- versus outcome-level metrics, and data source. Its snapshot reports:

- Of 74 tracks for control fidelity and interactive dynamics, 6 include closed-loop protocols; 68 are open-loop only.
- Of 22 interactive-action-fidelity tracks, 3 include closed-loop protocols.
- Functional-utility tracks are more often outcome-level and closed-loop, but there are only 15 tracks in that category.

The gap is not that closed-loop world-model evaluation or autonomous experiment selection do not exist. They do. The 130-benchmark survey and the separate 160-benchmark embodied-prediction survey both indicate that prediction-to-action coverage remains sparse; Autonomous Science Bench demonstrates budgeted real-science campaigns with discovery and held-out-learning gates. The narrower gap D2D should test is whether physical action-counterfactual reliability and certified branch-separating experiment design can improve those campaigns' decisions while preserving recurrent capability, transfer and independently auditable evidence.

1. The agent chooses which experiment to run.
2. It commits predictions before seeing the result.
3. The selected intervention is actually executed in a simulator or physical test fixture.
4. Outcomes are scored independently of the learner and world generator.
5. The learner adapts from results.
6. A fresh, held-out generator family tests transfer.
7. Protected skills and prior capabilities are retested.
8. The complete compute, energy, experiment and human-review costs are charged.

D2D should complement, not rename or duplicate, existing world-model benchmarks.

## 4. The D2D benchmark stack

### Layer A — exact-known-truth experiment design

Use analytic systems with identifiable parameters and a controlled observation/noise model.

Measure:
- hypothesis identification accuracy;
- expected versus realized discrimination;
- experiment cost and budget feasibility;
- abstention when candidate hypotheses are observationally indistinguishable;
- prediction-before-reveal compliance;
- exact expected/observed branch accounting.

The first scalar friction fixture is the initial software protocol demonstration, not the whole benchmark.

### Layer B — action-conditioned world-model reliability

Use MiraBench-style physical/action/optimism diagnostics and World-In-World's closed-loop action API.

Measure:
- action-following fidelity;
- paired intervention sensitivity;
- physically inconsistent but visually plausible predictions;
- optimism under failure-inducing actions;
- action-ranking regret;
- closed-loop task utility;
- p50/p95/p99 inference and rollout cost.

### Layer C — executable imagined behaviour

Use RoboWM-Bench-style trajectory-to-action conversion where licensed implementations and adapters are compatible.

Measure:
- fraction of predicted behaviours that can be translated into legal action sequences;
- execution success versus predicted success;
- contact, geometry and dynamics failure categories;
- uncertainty calibration on non-executable predictions;
- false-safe / optimistic prediction rate.

### Layer D — memory, temporal persistence and embodiment transfer

Use MIND and XEWorld as external anchors, plus the existing HLB retention programme.

Measure:
- long-context state and relation consistency;
- action-space transfer;
- held-out morphology / embodiment transfer;
- memory retrieval relevance;
- repeated failure after memory consolidation;
- protected-skill retention;
- catastrophic forgetting after adaptation.

### Layer E — autonomous research and independent replication

Use Autonomous Science Bench as an external whole-agent research-workflow comparator where tasks are compatible; extend D2D to sealed, held-out experiment families.

Measure:
- validated discoveries per experiment budget;
- reduction in decision regret;
- expected versus realized information gain;
- transfer to a generator family not seen during training;
- independent fresh-process reproduction;
- complete source, tool, data, model and evaluator lineage.

No single scalar should erase the separate metrics. Publish a Pareto frontier over verified utility, transfer, risk, cost and latency.

## 5. Fair baseline matrix

Every confirmatory D2D run should include:

1. Direct policy without a learned world model.
2. Direct policy with the same memory and observation history.
3. Random experiment selection with the same experiment budget.
4. Tuned classical experiment design (Bayesian optimization, expected improvement or active learning where assumptions fit).
5. A conventional learned action-conditioned world model and planner.
6. A frontier VLA/video-world-model baseline where its weights, code, license, action API, data and compute can be reproduced.
7. HLB without dream execution.
8. HLB with Symtropy-executed dreams and active experiment selection.
9. Evaluator-only oracle upper bound, never available to the learner.

Separate representation ablations (HDC, liquid dynamics, memory, uncertainty, active sensing) from orchestration ablations (experiment selection, simulator fidelity, inference budget and evaluator feedback). Otherwise the HLB contribution will be impossible to identify.

## 6. Competitive strategy: how to beat the state of the art honestly

### Do not start by chasing the largest public score

Cosmos Policy's reported LIBERO and RoboCasa results are already very high. Reproducing those setups is useful, but an isolated higher number on a different configuration would not establish superiority.

### Win on a capability that the benchmark actually measures

Prioritize four plausible competitive wedges:

1. **Experiment efficiency:** fewer real/simulated trials to reach the same or better decision quality.
2. **Counterfactual reliability:** better intervention-conditioned prediction and less optimism when a branch fails.
3. **Transfer:** stronger retention and generalization under held-out generator families and embodiments.
4. **Evidence quality:** independently reconstructable experiments, complete denominators, and no hidden evaluator access.

### Prospective gates

- 100% expected-branch accounting, including all incomplete and unsupported cases.
- Zero accepted hidden evaluator / future-outcome / teacher-state leaks under planted negative controls.
- At least 10% relative action-ranking-regret reduction versus the strongest matched baseline, with a preregistered paired confidence interval excluding zero.
- At least +5 percentage points on one genuine held-out generator family, without a statistically credible protected-task regression.
- Calibration compatible with nominal coverage on held-out regimes.
- Positive validated decision value per total compute, energy, experiment and human-review cost.
- At least one independent fresh-process replication.

These are prospective research gates, not achieved performance. Set sample sizes, power and non-inferiority margins before confirmatory evaluation.

## 7. Recommended next experimental domain

Start with **contact-rich manipulation**, because the existing external benchmark portfolio is relatively mature:

- World-In-World for closed-loop planning and task value.
- MiraBench for action-conditioned reliability and optimism.
- RoboWM-Bench for executable behaviour.
- RoboCasa / LIBERO only with benchmark-validity and shortcut diagnostics.
- MIND / XEWorld for temporal consistency and embodiment transfer.

Use distributed-systems resilience as the second domain, where Symtropy can generate partitions, churn, correlated failures and Byzantine conditions, and Mycelix can preserve the attestations. Do not mix the two domains' headline percentages.

## 8. Source register

- [NVIDIA Cosmos Policy — official results and protocol](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- [World-In-World — ICLR 2026 closed-loop world-model benchmark](https://world-in-world.github.io/)
- [MiraBench — action-conditioned reliability benchmark (arXiv:2605.29360)](https://arxiv.org/abs/2605.29360)
- [RoboWM-Bench — CVPR 2026 Workshop](https://robowm-bench.github.io/RoboWM-Bench/)
- [MIND — memory consistency and action control (arXiv:2602.08025)](https://arxiv.org/abs/2602.08025)
- [XEWorld — held-out robot embodiment generalization (arXiv:2608.05799)](https://arxiv.org/abs/2608.05799)
- [Autonomous Science Bench 0.1 — leaderboard](https://asb.open-ssi.org/leaderboard)
- [Autonomous Science Bench — overview, six tasks, scoring gates and pilot caveats](https://asb.open-ssi.org/about)
- [NVIDIA Cosmos Cookbook — Predict2.5 vs Predict2 benchmark tables](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/recipes/post_training/predict2/cosmos_policy/post_training.md)
- [Do World Models Make Better Robots? — separate 160-benchmark survey (arXiv:2609.29669)](https://arxiv.org/abs/2609.29669)
- [Google DeepMind Genie 4 — official announcement and limitations](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)
- [A Survey of World Model Benchmarks — 130-benchmark snapshot](https://world-model-benchmarks.github.io/World-Model-Benchmarks/)
- [Robot Manipulation Benchmark Audit — CoRL 2026](https://ripl.github.io/manipulation_benchmark_audit/)
