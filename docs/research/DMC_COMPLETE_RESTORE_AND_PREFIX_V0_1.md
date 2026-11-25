# DMC Complete Restore and Prefix Qualification v0.1

**Status:** evaluator-side restore-manifest validator; exact-head Rust qualification pending.  
**Parent:** #13530 — DMC individual-effect pilot.  
**Related:** #13566 — DMC-RESTORE-001; #13375/#13377 causal trajectory lineage; Symtropy #3690/#3741/#3743 stochastic event correspondence.

## 1. Exact pinned-source finding

The DMC pilot pins google-deepmind/dm_control commit d7e52e3602a555df6cc20cc928ea5b41e9920158. At that exact commit, setup.py declares dm_control 1.0.52 and requirements.txt pins mujoco==4.0.1.

The pinned dm_control source establishes:

1. Physics.get_state(sig=None) uses _physics_state_items(): data.qpos, data.qvel, data.act, and data.plugin_state only if model.nplugin > 0. The exact engine_test.py labels these the three/four legacy state arrays.
2. Physics.get_state(sig=...) and set_state(..., sig=...) delegate to mujoco.mj_getState / mj_setState.
3. Physics.copy(share_model=False) copies the underlying MjData and model by default; this does not clone the Python dm_env.Environment or task object.
4. dm_control.rl.control.Environment separately stores _step_count, _reset_next_step, _n_sub_steps, _step_limit, _task and physics configuration such as legacy_step.
5. dm_control.suite.base.Task stores its own NumPy RandomState. Its source documentation separately notes that MuJoCo sensor noise, when enabled, uses an independent MuJoCo RNG.

**The default Physics.get_state() is not a complete branch-point snapshot.** It is a legacy compact state, not the full integration state needed for non-initial branch restore.

Pinned source:
- https://github.com/google-deepmind/dm_control/blob/d7e52e3602a555df6cc20cc928ea5b41e9920158/dm_control/mujoco/engine.py
- https://github.com/google-deepmind/dm_control/blob/d7e52e3602a555df6cc20cc928ea5b41e9920158/dm_control/rl/control.py
- https://github.com/google-deepmind/dm_control/blob/d7e52e3602a555df6cc20cc928ea5b41e9920158/dm_control/suite/base.py
- https://github.com/google-deepmind/dm_control/blob/d7e52e3602a555df6cc20cc928ea5b41e9920158/dm_control/mujoco/engine_test.py
- https://github.com/google-deepmind/mujoco/blob/4.0.1/include/mujoco/mjtype.h
- https://mujoco.readthedocs.io/en/stable/computation.html

## 2. Pinned integration-state signature

In the exact MuJoCo 4.0.1 header, mjSTATE_INTEGRATION is the union of mjSTATE_FULLPHYSICS, mjSTATE_USER and mjSTATE_WARMSTART. The state flags occupy bits 0 through 14, making the exact numeric mask **32767 (0x7fff)**.

The manifest binds both the symbolic name and the numeric value. The Python adapter must obtain the runtime enum value and compare it with this pinned value rather than hard-code a guess.

The documented integration state includes the inputs to forward dynamics: time, positions, velocities, actuator activations/history/plastic state, controls, applied forces, equality activation, mocap state, userdata, plugin state and warmstart state for this API. It does not replace Python task, wrapper, driver/controller or independent sensor-noise RNG state.

MuJoCo's exact reproducibility guarantee is scoped to the same version and architecture. A snapshot from a different release or CPU architecture is not silently interchangeable.

## 3. Snapshot manifest

The DMC restore manifest binds:

**Physics / model:** dm_control commit and package version; MuJoCo exact package/version/build; runtime platform and architecture; MJCF/assets/model options; explicit mjSTATE_INTEGRATION name, numeric value, vector size and digest or a deep Physics.copy(share_model=False) snapshot; plugin inventory and plugin-state inclusion; common exogenous schedule.

**Python task / wrapper / driver:** task class/configuration and private-state digest; full task.random state digest; wrapper state digest plus explicit _step_count, _reset_next_step, _n_sub_steps, _step_limit raw IEEE-754 bits and legacy_step; control timestep; driver/controller state digest; separate MuJoCo sensor-noise RNG state when sensor noise is enabled.

If sensor noise is enabled and its independent RNG state cannot be captured and verified for the exact pinned runtime, the same-u stochastic individual-effect estimand is unsupported. Do not infer that the MuJoCo integration signature covers an RNG state absent from the documented state mask.

**Factual prefix / interventions:** common branch-point step; prefix trace/state/observation digests; rest/settling operation identity and whether it belongs to the common factual prefix; expected action IDs, run IDs and action-sequence digests; prefix state and observation maximum absolute errors in integer nanounits (one nanounit = 1e-9 source units); fresh-process replay disposition.

## 4. Admission semantics

The new Rust validator requires:

1. Exact dm_control commit, package version 1.0.52 and MuJoCo version 4.0.1.
2. A non-legacy complete physics snapshot.
3. The exact pinned integration signature name and numeric value 32767, or a deep non-aliasing physics copy.
4. Nonzero model, task, task-RNG, wrapper, driver, exogenous, prefix and snapshot digests.
5. Python task/wrapper state at the exact branch point; _reset_next_step false and _step_count equal to the declared branch step.
6. Nonzero action repeat and control timestep; the episode step limit has not been reached.
7. Plugin state for plugin-bearing models, and separate MuJoCo RNG state when sensor noise is enabled.
8. Any auxiliary rest/settling operation belongs to the common factual prefix. Branch-only rest is rejected.
9. Exact expected action coverage and unique run/action identities.
10. The same snapshot, exogenous schedule, factual-prefix digest, prefix step count and branch-point step on each sibling.
11. Prefix state and observation maximum errors no greater than 1e-9.
12. A passing fresh-process replay for each branch.
13. A nonempty post-intervention trajectory for every branch.

Post-intervention trajectories are branch-local. They may differ or remain identical: inequality is not itself proof of causal correctness, and equality is valid for a true null/action-invariant case.

The certificate preserves action ID, run ID and post-intervention trajectory digest as one sorted tuple so canonicalization cannot scramble their mapping.

## 5. Restore procedure

1. Freeze the exact simulator/task/model profile, seed domains, action repeat, control timestep and endpoint measurand.
2. Run the factual prefix once.
3. At the branch point, capture a deep physics copy or explicit mjSTATE_INTEGRATION vector, plus task, wrapper, driver/controller and applicable RNG state.
4. Recreate a fresh environment/process per intervention; restore every required component. Do not reuse mutable MjData, task objects, wrapper counters or controllers across sibling actions without a verified full clone.
5. Replay the factual prefix from the saved source in a fresh process. Check internal state and observations before the first action divergence.
6. Begin each alternative action tape from the admitted snapshot. Preserve the underlying exogenous disturbance realization across interventions, using stable event/tick/channel correspondence for stochastic execution.
7. Do not add an auxiliary rest/settling action only at the branch point. If rest is part of the declared experiment, it must belong to the common factual prefix in every branch.

## 6. Falsification controls

The companion JSON fixtures cover:
- legacy compact-state rejection;
- pinned integration signature and numeric-mask mismatch;
- missing plugin-state rejection;
- sensor-noise RNG omission;
- wrapper reset/step-count mismatch;
- branch-only rest rejection;
- valid deep physics copy;
- valid common full-snapshot restore;
- prefix state/observation error above 1e-9;
- missing fresh-process replay;
- branch exogenous schedule mismatch;
- legal post-intervention branch divergence;
- legal identical-trajectory null case.

The Rust module has 27 deterministic unit tests covering snapshot kind/version, mask, plugin/RNG completeness, wrapper state, rest symmetry, action/run uniqueness, prefix identity and tolerance, fresh-process replay, exogenous schedule and legal branch divergence. These tests validate the finite contract only; they are not dm_control simulation results.

A separate pinned Python smoke harness lives at tools/dmc-restore-smoke/restore_smoke.py. The dedicated workflow installs the exact dm_control source commit and MuJoCo 4.0.1, then runs two separate Python invocations: process A exports a real cartpole.swingup integration-state snapshot and expected one-step continuation; process B creates a fresh environment, restores the integration state, task RandomState and wrapper fields, checks the factual-prefix observation, applies the same next action and compares the next state/observation/reward/step counters. The harness also demonstrates that the legacy compact state does not restore a deliberately modified full integration state and that Physics.copy(share_model=False) does not alias MjData.

The smoke harness covers the deterministic cartpole task only. It does not claim restoration of sensor-noise RNG state (sensor noise is disabled for this lane), every possible task-private mutable field, arbitrary plugins, all platforms, or stochastic event correspondence. Those remain explicit unsupported/extension lanes until individually exercised.

The Rust validator does not execute dm_control or prove that every possible Python object field has been captured. The pinned Python smoke harness exercises the declared adapter path; the exact-head Rust workflow qualifies the finite manifest admission. Both are necessary and neither alone proves broad DMC paper replication.

## 7. Ownership and claim boundary

- #13530 owns DMC formulation, task eligibility and exact replication boundary.
- #13520 owns individual-effect arithmetic.
- #13375/#13377 own generic action-coupled trajectory lineage and pairing references.
- Symtropy #3690/#3741/#3743 own stochastic event correspondence and independent draw-receipt verification.
- #9437 owns evaluator process/OS isolation.
- #13566 owns DMC-specific full simulator/task/wrapper restore and prefix qualification.

This module does not implement MuJoCo, Python replay, a second generic branch verifier, RNG generation, effect scoring, calibration, causal identification, regret or action authorization.

Passing establishes only that supplied DMC restore/prefix evidence conforms to this pinned manifest contract. It does not establish physics fidelity, DMC eligibility, exact paper replication, world-model performance, real-robot counterfactual truth, closed-loop improvement, physical safety or SOTA.
