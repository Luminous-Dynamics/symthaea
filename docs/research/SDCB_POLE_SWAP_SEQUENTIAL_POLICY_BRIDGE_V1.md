# SDCB pole-swap → sequential policy oracle bridge v1

**Status:** authored Rust integration fixture; not compiled, executed or CI-qualified. Synthetic finite-support challenge only.

## Purpose

Bridge the frozen two-world process/sensor pole-swap task from #14243 into the existing prior-free finite-horizon policy oracle owned by #13567 / PR #13571.

This adds no second planner. It constructs a \`SequentialRobustProblem\`, then calls both existing references:

- \`evaluate_sequential_robust\` — backward dynamic programming;
- \`enumerate_sequential_robust\` — separately implemented exhaustive policy-tree enumeration.

The adapter checks that they agree on value, selected policy, branch supports and per-terminal-path resource receipts.

## Provenance and frozen worlds

The task is bound to:

- #14243, \`SDCB_POLE_SWAP_DECISION_SUFFICIENCY_V1.json\`, blob \`d09d8b9c2acd4d2c3ee392f19d4a1e25aacfad17\`;
- #14003 source response challenge, commit \`32ed97333ce383ab4c604ee558bb642341872f3e\`, source blob \`edf6da2ec4e4e5effb4638d76fcdcaffa16ff218\`;
- #13571 policy oracle head \`7ed8f823ef1c5bcf4a3e2d7e01d5f95320f3cda7\`.

The worlds are:

| World | Physical decay λ | Sensor response μ | Process variance |
|---|---:|---:|---:|
| A: physical slow / sensor fast | 0.25 | 4.0 | 9 |
| B: physical fast / sensor slow | 4.0 | 0.25 | 144 |

The passive single-channel law and the upstream unit-DC-gain first-order relaxation cascade are symmetric under the pole swap. A generic upstream step or frequency command therefore does not automatically identify which pole belongs to the physical process and which belongs to the sensor. The sensor-inlet calibration and independent latent reference are distinct probes with different response contracts.

## Tasks and expected policy

### INV — decision invariant

\`maintain_safe_configuration\` has loss 0 in both worlds; \`replace_sensor\` has loss 1 in both.

Expected exact policy: **STOP**. Current minimax loss is already zero. No mechanism-identifying probe can improve the terminal decision, so buying more identification is strictly unnecessary under this task loss.

### SENS — decision sensitive

\`select_fast_sensor_path\` has losses (0,1) over (A,B), and \`select_slow_sensor_path\` has losses (1,0).

Expected exact policy: use a genuinely disambiguating probe, then select the world-specific action. Current minimax loss is 1; worst-case terminal loss after the selected policy is 0. Passive data, the symmetric upstream cascade and the nuisance-only probe cannot reduce the decision loss.

The default problem admits all three resolvers. They all achieve terminal loss zero. The existing #13567 oracle uses deterministic lexicographic action-ID tie-breaking, so this frozen run selects \`independent_reference_of_latent_process\`. **That is not a least-cost claim.** Cost remains a component-wise vector; the adapter additionally restricts budgets to demonstrate the inlet-step and independent-reference paths separately.

## Probe and resource contract

Resource dimensions remain separate:

\`{elapsed_slots, reference_units, actuator_energy_units, compute_units}\`

| Probe | Elapsed | Reference | Actuator energy | Compute | Support result |
|---|---:|---:|---:|---:|---|
| Passive autocovariance | 0 | 0 | 0 | 0 | {A,B} |
| Upstream relaxation cascade | 1 | 0 | 1 | 0 | {A,B} |
| Traceable inlet step | 1 | 1 | 1 | 0 | {A} or {B} |
| Traceable inlet frequency sweep | 2 | 1 | 2 | 0 | {A} or {B} |
| Independent latent reference | 1 | 2 | 0 | 0 | {A} or {B} |
| World-independent nuisance control | 1 | 0 | 0 | 0 | {A,B} on either outcome |

The nuisance control has a Bernoulli(1/2) outcome in both worlds: one bit of outcome entropy, zero mutual information about the world and zero decision value. The policy oracle itself consumes set-valued support, not probabilities; its adapter checks that each nuisance outcome preserves \`{A,B}\`. Probability and entropy facts remain in the frozen source manifest and are not inferred from this support-only solver.

## Authored Rust checks

Seven tests are included:

1. INV stops without acquisition and returns zero loss.
2. SENS acquires a resolving probe, maps each singleton support to the correct terminal action, and returns exact resource-path receipts.
3. Passive, upstream and one-bit nuisance controls preserve support and cannot resolve SENS.
4. Each of the three declared disambiguators independently resolves SENS.
5. A one-slot / one-reference / one-actuator budget with a one-step deadline selects the inlet step without scalarizing cost.
6. A one-slot / two-reference / zero-actuator budget selects the independent reference.
7. An unaffordable or late frequency sweep cannot create a resolution claim.

Each known-answer test calls both the dynamic-programming and exhaustive policy references and compares their outputs.

## What this establishes—and what it does not

This is a composition check for the existing finite set-valued policy oracle. It does **not** establish:

- that a physical sensor's noisy observation distributions have disjoint support;
- finite-sample type-I error or worst-case power;
- robustness to continuous nuisance parameters;
- traceability of a real calibration reference;
- superior policy performance on real experiments;
- a new policy-planning theorem or production execution authority.

The next mathematical bridge is to replace the idealized singleton-vs-overlap support outcomes with intervention-dependent *noisy* trajectory laws from #14002 / #13904, uncertainty enclosures from #14000, and terminal test certificates from #14185 / #14218. The next policy benchmark should then compare the same fixed arms under identical component-wise budgets and report regret against #13567.

## Tournament to run after exact-head qualification

Freeze and compare, on identical worlds, candidate actions, costs and deadlines:

- no acquisition;
- passive-only;
- upstream response-gap greedy;
- outcome-entropy greedy;
- sensor-inlet response-gap greedy;
- decision-value greedy;
- fixed inlet-step design;
- fixed independent-reference design;
- exact prior-free sequential oracle (evaluator-only upper reference).

Report worst-case terminal loss and regret, false mechanism-resolution rate, unnecessary acquisition on INV, failure to acquire on SENS, risk-gate status, per-path resource vectors, deadline/budget failures, unresolved/abstention denominators and independent replay.

A superiority claim requires held-out mechanisms and device families, matched costs, paired uncertainty intervals and complete failure denominators. This synthetic fixture is only the exact known-answer control for that campaign.
