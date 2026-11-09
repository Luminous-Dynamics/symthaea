# XPolicyLab policy-action contract harness

This is a standalone, dependency-free Rust conformance harness for the Luminous adapter boundary around the XPolicyLab policy server.

It checks request/episode/server/schema correlation, ordered batched environment mapping, action-chunk sequence, channel names and dimensions, finite numeric values, and maximum chunk length. It also keeps ambiguous RPC timeout and policy-server restart distinct from a valid action response.

## Why this exists

At the inspected XPolicyLab source revision (`6baccd57ff958570d0db1588370289ee36c8bb70`), the policy model API exposes:
- `update_obs` / `update_obs_batch`;
- `get_action`, returning an ordered list of action dictionaries (a chunk);
- `get_action_batch(env_idx_list)`, returning chunks aligned to the supplied environment indices;
- `reset()`, which clears model-side state between episodes.

The XPolicyLab README also states that a request timeout is fatal for the trial because the server may still be processing, and a server-process restart loses model-side state. The validator encodes those as explicit non-success outcomes.

The separate `call_lifecycle` module models the documented policy-RPC lifecycle:
- an exact same-ID/same-payload reconnect replay is represented as a replay of the same request, not a new model invocation;
- a cached response for the same exact request can be returned idempotently;
- reusing a request ID with a different RPC kind or payload is rejected;
- an ambiguous timeout aborts the episode's policy trial and late responses cannot be used;
- a policy-server process restart aborts the current episode because model-side state was lost;
- request count is bounded.

This is an in-memory policy-call fixture, not the Symtropy controller/ticket ledger. It does not persist replay state across process restart and does not authenticate transport messages.

## Contract boundary

The adapter supplies the Luminous correlation envelope around the native policy output:

- request ID;
- episode ID;
- policy server **process instance** ID;
- pinned adapter/schema revision;
- action-space mode;
- ordered expected environment indices;
- per-environment next chunk index;
- exact action channel/dimension manifest;
- maximum allowed number of actions in one chunk.
- digest of the exact policy-visible observation bundle;
- digest of the policy source/checkpoint/weight manifest;
- digest of deployment configuration, prompt/inference settings and resolved runtime lock;
- digest of the declared tool, skill, oracle and human-assistance profile.

The four digest fields are attached by the Luminous adapter as a wrapper envelope; they are **not** native XPolicyLab model-response fields. They preserve which immutable inputs a result is associated with. This crate does not compute or authenticate those artifact hashes, verify digital signatures, or prove that the policy server actually loaded the artifact named by the manifest. The all-zero digest is reserved as an invalid/unset sentinel.

These envelope fields are not assumed to be native XPolicyLab response fields. In particular, the environment client must bind batch output slot *i* to the exact `env_idx_list[i]` sent to `get_action_batch`.

The validator deliberately does not:
- reinterpret action values as physical torque, force, velocity or position units;
- enforce joint limits, workspace, speed, collision, deadline or safety constraints;
- create controller authority, authenticate physical hardware, or prove an action was applied;
- establish sensor calibration, clock synchronization, metrology or physical observation provenance;
- replace Symtropy's request ledger, controller epoch fencing, native admission or independent safety authority.

A successfully validated chunk remains a **policy proposal**. It must still pass through the existing Symtropy authority-gated path. The evidence envelope must be attached separately through ROB-EXP / ROB-INSTR / SE-OBS semantics.

## Run

From the repository root:

```bash
cargo fmt --manifest-path experiments/xpolicylab-contract/Cargo.toml -- --check
cargo test --locked --manifest-path experiments/xpolicylab-contract/Cargo.toml
cargo clippy --locked --manifest-path experiments/xpolicylab-contract/Cargo.toml --all-targets -- -D warnings
```

The unit tests use invented synthetic action dictionaries. They establish only the behavior of this local validator; they are not XPolicyLab execution, a benchmark result, a controller-safety proof, or a physical robotics result.
