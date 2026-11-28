# SDCB pole-swap decision-sufficiency adapter v1

**Status:** analytic known-answer task adapter; Rust source/tests authored but not compiled or CI-qualified.

This fixture reuses the exact world and probe identities from #14003 / SDCB_RESPONSE_IDENTIFIABILITY_CHALLENGE_V1.json. Source identity:
- commit 32ed97333ce383ab4c604ee558bb642341872f3e
- blob SHA edf6da2ec4e4e5effb4638d76fcdcaffa16ff218

## The worlds and probe families

World A has physical pole lambda=0.25, sensor pole mu=4, variance 9. World B has lambda=4, mu=0.25, variance 144.

The original challenge freezes five probe types:
- passive single-channel autocovariance: equal complete law under both worlds;
- upstream first-order cascade excitation: equal transfer magnitude/phase under pole swap;
- traceable sensor-inlet step and frequency sweep: distinct responses and sensor-pole identification for the declared two-world model;
- independent characterized reference on the latent process: distinct responses and physical-pole identification.

A separate task-only nuisance control has Bernoulli(1/2) outcomes in both worlds. Its outcome entropy is 1 bit under either world, but its mutual information about the world is exactly zero.

## Two terminal decision tasks over the same worlds

### INV — decision invariant

| Action | World A loss | World B loss |
|---|---:|---:|
| maintain_safe_configuration | 0 | 0 |
| replace_sensor | 1 | 1 |

The unique minimax terminal action is maintain_safe_configuration with worst-case loss zero. No probe can reduce the terminal decision loss. A sensor-inlet test or external reference could identify the mechanism, but this task assigns it zero decision value. Correct policy: do not acquire more evidence merely to identify a parameter that cannot change the action.

### SENS — decision sensitive

| Action | World A loss | World B loss |
|---|---:|---:|
| select_fast_sensor_path | 0 | 1 |
| select_slow_sensor_path | 1 | 0 |

With support {A,B}, prior-free minimax terminal loss is 1. The optimal action changes with the world, so passive/upstream data cannot settle the decision.

| Probe | Remaining support | Worst-case terminal loss | Decision value |
|---|---|---:|---:|
| passive autocovariance | {A,B} | 1 | 0 |
| upstream cascade | {A,B} | 1 | 0 |
| sensor-inlet step | singleton | 0 | 1 |
| sensor-inlet frequency | singleton | 0 | 1 |
| independent latent reference | singleton | 0 | 1 |
| one-bit nuisance control | {A,B} on either outcome | 1 | 0 |

The result demonstrates that mechanism information and decision value are separate quantities. High outcome entropy is not decision value; mechanism identification can also have zero terminal decision value.

## Synthetic resource vector

All values below are challenge units, not measured hardware costs. Preserve them as separate dimensions:
- passive evidence already available: (0,0,0,0);
- upstream excitation: (1,0,1,0);
- sensor-inlet step: (1,1,1,0);
- frequency sweep: (2,1,2,0);
- independent reference: (1,2,0,0);
- nuisance entropy control: (1,0,0,0).

The dimensions are elapsed slots, reference units, actuator energy units and compute units. No scalarization is allowed in this benchmark.

## Implementation boundary

- scripts/research/sdcb_pole_swap_decision_sufficiency.rs verifies the finite action-loss/support partitions and known-answer decision values.
- scripts/research/run_sdcb_pole_swap_decision_sufficiency_tests.sh is a standalone rustc test/executable harness.
- Nine authored tests cover invariant action/no acquisition, zero decision value of mechanism-only information, unresolved sensitive task, passive/upstream equivalence, disambiguating probe value, entropy-vs-information negative control, world-specific action correctness, resource-vector preservation and the no-physical-authorization boundary.

This is a task adapter to #13567, not a second policy-tree implementation. The #13567 exact prior-free policy oracle remains the authority for complete adaptive policies and terminal regret. #14000 owns uncertainty enclosures and #13805 owns verified evidence transitions.

## Validation boundary

The original pole-swap asset was fetched from its exact branch/commit and all response strings/probe IDs are carried into the versioned manifest. Rust tests are authored but not compiled/executed; no #13567 run, physical validation, metrological traceability or SOTA claim is made.
