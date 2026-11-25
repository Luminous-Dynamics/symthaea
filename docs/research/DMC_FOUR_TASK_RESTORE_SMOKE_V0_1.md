# Four-task DMC restore/replay extension v0.1

**Status:** harness/workflow extension; exact-head four-task replay pending.
**Base:** #13568 at `fb98f789141c9d79af7a790efa52a7b5b0747704`.
**Pinned source:** dm_control `d7e52e3602a555df6cc20cc928ea5b41e9920158` (1.0.52), MuJoCo 4.0.1.

## Motivation

The published DMC family in the individual-effect paper includes four environment/task constructors:

- `cartpole.swingup`
- `cartpole.two_poles`
- `cartpole.three_poles`
- `walker.walk`

The parent #13568 qualification exercised deterministic `cartpole.swingup` only. This extension runs the same isolated two-process checkpoint/replay procedure for all four task constructors. It increases task-API and state-dimension coverage; it does **not** recreate the paper's 150-individual, six-intervention, 15-pair, 12-history-block experiment or resolve its missing generator/seeds/parameter distributions.

## Harness contract

The existing test-only Python adapter is parameterized by an explicit finite task allowlist. Each task run:

1. Uses the same pinned dm_control/MuJoCo release and a deterministic task RNG seed.
2. Runs a 12-simulator-step factual prefix.
3. Captures `mjSTATE_INTEGRATION` and the task RNG state plus the Python wrapper state used in #13568.
4. Starts an independent fresh Python process, restores the integration state, task RNG and wrapper fields, and checks the prefix observation.
5. Applies the same next action and compares the next integration state, observations, reward, discount, step type, RNG digest and wrapper step/reset counters.
6. Separately demonstrates that legacy compact `Physics.get_state()` does not restore the complete integration vector, and that `Physics.copy(share_model=False)` does not alias `MjData.qpos`.

The four tasks do not have identical state/action dimensions. The harness obtains action specifications from each environment rather than hard-coding a scalar action vector. The walker task's constructor applies its own `control_timestep`; this is preserved by the pinned API rather than replaced by a cartpole-specific assumption.

## Workflow

A non-cancelling four-entry GitHub Actions matrix runs the same export process and fresh-process replay for each task. Every matrix entry has its own Python 3.11 environment and checkpoint. The inherited Rust manifest evaluator is unchanged and was qualified with 108 passing crate tests plus strict Clippy in parent #13568; this PR changes only the Python task-coverage adapter and workflow matrix.

## Scope and limitations

- This is deterministic restore/replay smoke coverage, not world-model training or a causal-effect benchmark.
- Twelve simulator steps in this smoke are not the article's 12 history blocks.
- Sensor noise remains disabled; independent MuJoCo sensor RNG restoration is not exercised.
- This does not prove all task-private fields or arbitrary plugins are captured for every possible dm_control suite environment.
- The task constructors are the four named in the article; the `walker.walk` smoke does not imply that it passes the paper's C4 measurability criterion.
- No paper replication, trained HDC/CfC performance, SOTA, or physical safety result is claimed.
