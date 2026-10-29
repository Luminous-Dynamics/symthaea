# CMAQ-002: Active Causal Learning and Value of Information

**Status:** small executable research prototype  
**Version:** 0.1.0  
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
    cargo run --locked -- --seed 4242 --probe-budget 2 --decision-horizon 1 --probe-cost 0.25

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
