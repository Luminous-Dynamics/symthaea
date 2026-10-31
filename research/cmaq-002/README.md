# CMAQ-002: Active Causal Learning and Value of Information

**Status:** executable synthetic research benchmark v0.3  
**Version:** report contract v0.3  
**Validation:** source and test suite not yet Rust-compiled in this authoring environment  
**Claim ceiling:** synthetic finite-hypothesis benchmark only; no SOTA or real-world causal-reasoning claim

## Research question

Does choosing experiments by expected improvement in downstream decisions outperform fixed probing, seeded random probing, and maximum-information probing under the same interaction budget?

This tranche instantiates a tiny exact simulator so the research can proceed independently of model runners. It is deliberately small enough for exhaustive reasoning. The candidate's declared hypothesis library has nine mechanisms across three structurally different families:

- additive effects;
- interaction-dominated effects;
- threshold effects.

An additional evaluator-only parity family is used as a model-misspecification control and in a unit test. The held-out family is not included in the candidate hypothesis library or in the decision-value calibration aggregate. The CLI reports its detection coverage separately so the model-misspecification check is observable beyond unit-test output.

Each mechanism yields deterministic reward and safety outcomes for three actions across four contexts. A candidate starts with a uniform belief over the hypothesis library, probes one context/action at a time, and receives only that intervention's outcome. The evaluator knows the true mechanism and scores post-learning decisions.

## Run

    cargo test --locked
    cargo run --locked -- --seed 4242 --probe-budget 2 --decision-horizon 1 --probe-cost 0.25 --allocator-cost 0.001

The CLI prints a deterministic JSON report comparing no probe, fixed probes, seeded random probes, maximum expected information gain, expected value of information (EVoI), and an evaluator-only oracle that knows the true mechanism. The `--decision-horizon N` option (default 1) is the number of downstream decision rounds that can reuse the learned mechanism. EVoI buys a probe only when expected per-round decision improvement multiplied by this horizon exceeds probe cost; reported probe costs are amortized over the same horizon. This explicitly models information reuse rather than treating every problem as one-shot. The report also includes an open-class control: three evaluator-only parity mechanisms that the candidate catalog does not contain, with the number of probes required before the posterior becomes inconsistent with every candidate hypothesis. A separate Python mirror of the small finite-state equations produced a preliminary calibration-panel result of mean net regret 0.9444 for EVoI versus 1.5833 for maximum information gain and 3.5278 for no probes. This is exploratory validation of the toy design only, not execution of the Rust binary, not statistical evidence, and not an external benchmark score.

The oracle is an upper bound, never a deployable baseline. The report scores all nine possible true mechanisms, not a single lucky hidden draw. The seed changes the random-probe policy and is recorded in the report; it does not change mechanism outcomes.

## Decision and information semantics

The candidate retains all hypotheses consistent with the observation history. A probe reveals only a binary positive/non-positive reward signal plus the safety bit for one selected context/action; exact reward magnitude remains evaluator-side and is used only for scoring. This deliberately makes information incomplete: multiple mechanisms can yield the same observed signal while having different utility. If a new signal matches no remaining hypothesis, the update fails closed with an explicit mechanism-change/contradiction signal; it does not silently create an empty belief.

Probe queries are simulated interventions inside the benchmark sandbox; they do not actuate a physical system. The safety bit describes whether the same context/action is safe in the represented world, and the candidate uses only posterior predictions when selecting a deployed action. The current prototype therefore tests decision-time safety filtering, not the safety of real-world experiment execution; physical experimentation would require an independent, pre-authorized safe-probe envelope outside the learned model.

For each test context, the candidate may choose an action only if that action is safe in every remaining hypothesis. It then selects the action with the highest posterior mean reward, abstaining when no robustly safe action has positive expected reward. This is intentionally conservative and makes the safety assumptions explicit.

- **Decision value:** mean utility over the four evaluation contexts.
- **Constrained regret:** oracle utility for the true mechanism minus candidate utility.
- **Expected information gain:** expected entropy reduction over the current mechanism posterior.
- **EVoI:** expected increase in the candidate's posterior decision value after a possible probe.
- **Net EVoI:** EVoI minus the declared probe cost; the EVoI policy stops when no remaining probe clears cost.

The four evaluation contexts are known to the candidate as a task grid, but the true mechanism is hidden. The full hypothesis library is visible by design. This tests inference and experiment selection within a finite model class; it does not test discovery outside that class. A real qualification needs hidden mechanism families or functional forms.

## Required interpretation

This prototype tests an important null hypothesis: information gain and decision value are not interchangeable. The maximum-information policy may choose an observation that sharply distinguishes mechanisms without improving safe action selection. EVoI instead scores the expected improvement to the downstream decision policy.

Do not assume EVoI will win. The goal is an auditable comparison whose output can falsify the value-of-information design. A stronger next release should preregister additional mechanism families, independently generated hidden laws, sequential environment changes, noisy observations, and task-level bootstrap analysis.

## Limits and next gates

- No Symthaea HDC, CfC/LTC, production memory, learned world model, or planner is connected yet.
- The simulator is deterministic and fully specified. It is a protocol unit test, not an external benchmark score.
- The current family library has nine hand-defined hypotheses. It does not justify claims of general causal discovery.
- The regime-shift signal detects an observation inconsistent with the current hypothesis set; a full change-point detector and adaptation benchmark are future work. The Rust unit tests also require the fixed calibration panel to distinguish decision-value probing from information-only probing, and verify that an evaluator-only held-out mechanism family cannot be silently absorbed by the candidate hypothesis set. These are protocol sanity checks specific to the hand-defined toy catalog, not general performance guarantees.
- Rust CI must compile and run tests at the exact head before this is promoted beyond draft status.

External benchmark alignment:
- [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) evaluates long-horizon agent memory.
- [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) evaluates memory in interdependent multi-session agent/environment loops.
- [World Model Benchmarks survey](https://world-model-benchmarks.github.io/) distinguishes prediction-level evaluation from closed-loop downstream utility.
- [World-In-World (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html) evaluates world models in closed-loop embodied settings.

CMAQ-002 complements those suites; its synthetic scores are not comparable to their published leaderboard numbers.


## v0.3: safe information acquisition and decision-value accounting

This extension adds three distinct mechanisms to the earlier EVoI comparison.

### 1. Two-layer probe safety

An experiment is an intervention. It must be safe before its information value is considered.

Every candidate probe must now satisfy both:
- an **externally pre-authorized probe envelope** fixed in the task contract, independent of the candidate's learned posterior; and
- a **posterior-robust safety check**: the probe must be safe in every remaining candidate mechanism.

The synthetic task's six authorized context/action pairs are tested against every in-catalog mechanism and the evaluator-only parity family. Tests verify that all six remain safe across those fixtures. The report separately counts authorization-blocked candidates and candidates blocked by posterior safety. Selected unsafe probes are counted explicitly and the focused workflow requires zero in this controlled panel.

This is a synthetic task guarantee only. It is not proof that a learned model can certify a real physical probe as safe. Real deployment needs a separately reviewed actuator/sensor envelope, independent interlocks and task-specific authorization. Under arbitrary unseen mechanisms, no posterior-only filter can establish safety.

### 2. Charge the allocator

\`--allocator-cost X\` specifies a **synthetic utility cost per normalized allocator work unit**. The allocator cost is charged even when the allocator evaluates candidates and then decides not to probe.

- EVoI and maximum-information strategies incur one work unit for each eligible probe candidate they score.
- Fixed and random strategies incur one selection work unit.
- EVoI only purchases a probe when expected downstream decision value covers both physical probe cost and the work required to select the probe.
- Reported net utility and net regret charge both probe cost and allocator cost, amortized over the declared downstream decision horizon.

This is not measured FLOPs, latency, memory or energy. The work unit is a controlled synthetic accounting parameter, included so the benchmark can falsify systems that claim an allocator is free. Real hardware comparison should use the resource-profile/profiling-receipt approach from CMAQ-012 and charge measured allocator inference.

The focused workflow runs both a low-cost case and an intentionally expensive allocator case. The latter must stop EVoI probing while still recording the work/cost spent to reach that decision.

### 3. Did information change the decision—and was the change good?

The report now separates:
- \`mean_decision_gain_vs_no_probe\`: realized utility change after acquisition, against the same mechanism's pre-probe decision;
- \`mean_decision_flip_fraction\`: fraction of context decisions changed by acquisition;
- \`beneficial_flip_fraction\` and \`harmful_flip_fraction\`: evaluator-only classification of changed decisions using the true outcome;
- \`mean_allocator_work_units\` and \`mean_allocator_cost\`;
- authorization-blocked and posterior-safety-blocked candidate counts;
- \`unsafe_probes\` independently of \`unsafe_actions\`.

This makes it possible to identify an information policy that learns a great deal yet does not change the downstream action, or that changes an action in the wrong direction. The evaluator-only benefit/harm labels are never candidate-visible.

## External frontier: DEEP and ActiveArena

Two 2026 benchmarks inform the next adapters:

- [DEEP: The Decision Value of Perception Compute](https://arxiv.org/abs/2609.35910) scores perception escalation by its effect on the downstream decision, under latency and energy budgets, and charges the allocator for its own computation. Its published experiments show that improved perception can produce harmful decision changes; allocator cost can eliminate the apparent benefit of heavier routing. CMAQ-002 adopts the decision-value and allocator-accounting principles, but its current synthetic work unit is not a reproduction of DEEP.
- [ActiveArena](https://arxiv.org/abs/2609.24124) provides 35 active-perception manipulation tasks across five categories, with multi-round evidence acquisition, memory reasoning and ID/OOD scene splits. Its published ActiveArena-OFT results are 62.97% ID / 47.60% OOD macro success, while ActiveArena-Plan reports 60.97% / 45.54%. Those are task-specific published results, not numbers comparable to CMAQ's finite-mechanism simulator.

The interfaces must remain distinct. CMAQ-002 currently chooses a causal intervention \`(context, action)\` and observes a safety/reward signal. ActiveArena chooses viewpoints, acquires images and manages memory; it is not the same action space or oracle. The correct next step is a separate visual active-perception adapter that reuses the report's decision-change, net-value, budget and provenance concepts, rather than pretending that a synthetic mechanism probe is a camera-motion result.

## v0.3 claim ceiling

- The external authorization envelope is a fixed synthetic task contract; it is not a real safety case.
- Allocator work units and costs are declared synthetic quantities, not hardware-measured operations or energy.
- Decision-flip benefit/harm uses evaluator-only true outcomes and does not imply candidate access to counterfactual labels.
- The current benchmark still uses a finite, deterministic hypothesis catalogue. It is not a live Symthaea HDC/CfC-LTC/memory/planner integration, and it does not establish external generalization or SOTA.


## Pinned ActiveArena reference

The external active-perception bridge is now accompanied by [activearena-baseline.lock.json](activearena-baseline.lock.json) and [activearena-baseline.schema.json](activearena-baseline.schema.json). The lock pins the ActiveArena simulator source and ActiveArena-VLA source separately, records the five native task categories, the 18D action space, head-camera-only candidate observation, fixed ID/OOD seed protocol, upstream simulator/asset/policy dependency revisions, and paper-reported OFT/Plan reference scores.

Important reproducibility boundaries:
- ActiveArena code is MIT; the public ActiveArena-Data Hugging Face card reports Apache-2.0 and about 581k LeRobot rows. Its immutable dataset revision and per-file checksum manifest have not been pinned in this lock, so it is metadata-only and no dataset has been downloaded.
- The base RoboTwin assets and ActiveArena supplementary assets are pinned to upstream revisions, but their separate dataset/asset licenses still require review. No asset archives have been downloaded.
- ActiveArena-VLA source is pinned separately. Its released checkpoint bundles do not yet have an immutable revision and manifest in this lock; the Qwen3-VL base model is pinned to the model-snapshot revision but has not been downloaded or licensed for this run.
- The simulator runtime requires Linux/Python 3.10, CUDA/Vulkan/NVIDIA GPU, and its own Python/PyTorch/SAPIEN/CuRobo/PyTorch3D stack. This must remain an isolated external runtime, not production Symthaea code.

The focused workflow schema-validates this lock and asserts that the dataset, asset archives and policy weights remain un-downloaded until the outstanding version/license/checksum gates are resolved. Paper-reported ID/OOD scores are comparison targets only; they are not Symthaea measurements.
