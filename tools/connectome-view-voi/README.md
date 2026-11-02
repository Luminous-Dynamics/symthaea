# connectome-view-voi

A dependency-free Rust primitive for choosing the next leakage-safe evidence view by **robust downstream decision value**, not entropy alone.

## Objective

For state Y, current evidence E, candidate view V, action a, declared loss L and acquisition cost c(V):

```text
VOI(V | E) =
    min_a E[L(a,Y) | E]
  - E_V[min_a E[L(a,Y) | E,V]]
  - c(V)
```

The implementation computes this quantity for each scenario in a finite, unweighted set of admissible belief/likelihood models. It reports the minimum and maximum net VOI across scenarios and selects a view only if the lower bound is strictly positive (above a small numerical tolerance). When scenarios disagree about whether acquisition is worth its cost, it stops rather than assigning them arbitrary weights.

## API

- `DecisionProblem`: task route, ordered latent states, candidate decisions, belief scenarios and view options.
- `DecisionAction`: nonnegative loss per latent state.
- `BeliefScenario`: a normalized prior over the latent states.
- `ScenarioViewLikelihood`: P(signal | state), one row per signal and one column per state.
- `ViewOption`: cost and an explicit cost-unit ID that must exactly match the decision-loss unit, allowed task routes, explicit leakage-safe flag, and scenario-specific likelihoods.
- `choose_next_view`: validated, deterministic `ViewDecision` with per-scenario risk/VOI records and an explicit stop reason.

All probabilities must be normalized and finite. Every view needs one likelihood matrix for every scenario. Signal order, scenario order and view input order are canonicalized in the receipt.

## Safety and scientific boundaries

- A view marked `leakage_safe = false` fails closed before likelihood evaluation.
- Every candidate must list the current task route in `allowed_task_routes`. The upstream manifest must independently establish this compatibility; the boolean is not a cryptographic proof.
- For CB2 merge-error identification, `em_best` must be rejected because its orientation depends on post-split masks. Split-error and other tasks need their own declared view policies.
- Scenarios are an **unweighted uncertainty set**, not random draws from a known population. The planner uses worst-case lower net VOI; it does not average or silently assign weights.
- VOI requires a declared action-loss matrix and calibrated likelihoods. If either is not supportable, the caller must not invent point values; the correct result is a bounded analysis elsewhere or `NOT_ESTIMABLE`.
- Acquisition cost and decision loss must use the same declared unit; the Rust API rejects a unit-ID mismatch. Real inference, data I/O, reviewer time, and physical acquisition costs must be measured or explicitly converted before use.
- The library chooses an evidence view only. It does not label a neuron, infer synaptic causality, propose a segmentation edit, or certify an edit as safe.
- Cached-view replay estimates policy behaviour under a simulation protocol; it is not a prospective acquisition or human-workflow study.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-view-voi/Cargo.toml
```

Fourteen unit tests are authored for perfect information, redundant views, entropy/decision-utility divergence, scenario disagreement, robust selection, empty candidate sets, target-leak rejection, task-route rejection, malformed distributions, missing scenarios and deterministic order. They have **not** been compiled or executed in the current environment.

## Research plan

See [SCI-CONNECTOMICS-055](https://github.com/Luminous-Dynamics/symthaea/issues/9987). The next steps are to add property tests against an exhaustive finite decision oracle, connect calibrated view models and acquisition costs, replay cached CB2 views with masks that preserve operation identity, and compare random / entropy / margin / VOI / robust-VOI / full-stack policies at fixed acquisition and reviewer-minute budgets.
