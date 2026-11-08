# Open-world ansatz executable microcosm v1

This standalone Rust crate adds the first executable world to #11042 and the task-contract fixtures in PR #11044.

It is a finite-state reference system, not a general scientific discovery engine, a PhysicsIR replacement or an Unsatz reproduction.

## World

The controllable input is \(u_t \in \{0,1\}\). The benchmark agent receives the intervention sequence and measured output sequence, but not the generating mechanism, initial hidden state or measurement map.

Two mechanisms are defined:

- **Direct:** \(y_t = u_t\).
- **Hidden lag:** \(h_{t+1}=u_t\), \(y_t=h_t\), with initially unknown \(h_0\).

The measurement channel is independently unknown and belongs to this finite family:

- identity: \(m_t=y_t\).
- invert: \(m_t=1-y_t\).
- constant-zero: \(m_t=0\), an intentionally uninformative channel.

This tests whether a researcher distinguishes direct dependence from latent-state mediation and separates uncertainty about the world from uncertainty about the measurement instrument.

## A0/A2 matched comparison

Both arms consume the same ObservationLog and therefore the same observations, interventions and measurement budget.

- **A0:** the mechanism is supplied; the reference enumerator infers the measurement map and hidden initialization.
- **A2:** the mechanism, measurement map and hidden initialization are withheld.

The only intended treatment is access to the mechanism. The reference implementation does not train an AI agent; it provides the executable world and exact finite hypothesis filter needed to test one.

**Binary v1 limitation:** the original boolean microcosm is a structure-known identification control, not a coefficient-fitting benchmark. The numeric extension below adds the first bounded gain/decay family and matched coefficient-bearing A0/A2 control.

## Coefficient-bearing numeric extension

The public Rust module numeric adds a second finite world that corrects the first slice's A0 limitation.

Its intervention alphabet is {-1, 0, 1}; candidate gains are {1, 2, 3}; leaky-memory decay values are {0, 1, 2}; initial hidden states are {-1, 0, 1}. Sensor maps are identity, negate, double and constant-zero.

Two structures are available:

- Direct response: x_t = gain * u_t.
- Leaky memory: y_t = h_t, followed by h_(t+1) = decay * h_t + gain * u_t.

Measurements pass through a separately unknown sensor map.

A0 receives the mechanism but must recover gain, decay (where applicable), initial state and sensor from the trace. A2 receives the same trace and must also discover the mechanism. This is the first coefficient-bearing matched A0/A2 control in this crate.

The finite exact enumerator deliberately preserves parameter ambiguity: direct gain 2 with an identity sensor is observationally equivalent to direct gain 1 with a double sensor. A unique structural answer must not be presented as unique coefficients or a uniquely identified measurement instrument.

The numeric exhaustive oracle uses a closed-form state response and enumerates all intervention sequences in {-1,0,1} for horizons 1–6. The finite probe selector ranks predicted outcome partitions, worst-case remaining models and query cost. It remains a partition baseline, not EVSI or a globally optimal adaptive strategy.

This model family is intentionally small and integer-valued. It is a protocol/control benchmark, not yet a continuous, stochastic, delayed, spatial or empirical scientific world.

## Independent structural oracle

The world simulator advances an explicit hidden state. The exhaustive oracle does not call the simulator or the hypothesis transition method. Instead, it uses a closed-form trace relation and enumerates every binary intervention sequence at the requested horizon.

Models are grouped when their complete input/output signatures are identical over all enumerated interventions. The constant-zero channel places direct and hidden-lag structures in one observational equivalence class. The oracle therefore preserves structural ambiguity instead of forcing a unique explanation.

The exhaustive oracle is limited to horizons 1–10. This is a small independently enumerable reference, not a scalable solver.

## Probe-selection baseline

recommend_discriminating_probe enumerates finite intervention sequences and chooses one that maximizes the number of distinct predicted outcome partitions, then minimizes the largest remaining candidate set, then minimizes step count.

This is an exact finite partition baseline only. It is not expected value of sample information (EVSI), expected utility, a causal-discovery theorem or a globally optimal adaptive policy. It also exposes when no probe can resolve structure under the current measurement family.

## Run

From the Symthaea repository root:

    cargo fmt --manifest-path research/open-world-ansatz-microcosm/Cargo.toml -- --check
    cargo clippy --manifest-path research/open-world-ansatz-microcosm/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path research/open-world-ansatz-microcosm/Cargo.toml
    cargo run --manifest-path research/open-world-ansatz-microcosm/Cargo.toml -- hidden_lag identity 0 1,0,0 a2

CLI scenario arguments identify the hidden evaluator-side world. They are for an operator-controlled local run; do not expose them to the agent process in a blind evaluation. The JSON output includes an oracle summary and is an evaluator report, not an agent observation packet.

For the matched A0 control, use the same world and intervention sequence, changing only the final argument to a0.

## Authored tests

Twenty-three Rust tests cover the binary and numeric worlds: hidden-state transitions, matched A0/A2 access, coefficient recovery, sensor-gain equivalence, passive and sensor-induced non-identifiability, partition probe selection, exhaustive oracle equivalence classes, distorted measurements, inconsistent evidence, invalid parameter bindings and horizon bounds.

Tests are authored. The binary v1 implementation has passed the focused Rust format, Clippy and test workflow on commit f40e2dd. The numeric extension is newer and awaits its own exact-head workflow result.

## Claim boundaries

- The crate is an executable synthetic microcosm and finite exact enumerator, not an agent result.
- The exhaustive oracle is independent at the trace-construction level, but shares the authored finite model definitions; it is not an external scientific authority.
- The implementation and source-visible tests are not a blind benchmark.
- No AI agent has been evaluated.
- No empirical physical system, unknown natural law or published scientific result has been reproduced.
- No Unsatz, Scientist World or FrontierPhysics benchmark score is claimed.
- The numeric extension's exact-head Rust compilation, tests, formatting and Clippy status must be read from its focused workflow; do not infer it from the earlier binary-v1 pass.
