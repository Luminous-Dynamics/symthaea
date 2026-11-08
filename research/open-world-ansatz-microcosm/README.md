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

**A0 limitation:** this first slice is a structure-known identification control, not yet a coefficient-fitting benchmark. It has no free numerical coefficient family. Add bounded gains/decay parameters next, then require the A0 arm to recover them with structure supplied before interpreting A2 performance.

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

Twelve Rust tests cover hidden-lag transitions, A2 structure recovery, matched A0/A2 access, passive non-identifiability, finite partition probe selection, uninformative sensors, oracle equivalence classes, distorted measurement maps, inconsistent evidence, exhaustive direct-versus-hidden separation, invalid state bindings and horizon bounds.

Tests are authored but have not been compiled or executed in the current editing environment.

## Claim boundaries

- The crate is an executable synthetic microcosm and finite exact enumerator, not an agent result.
- The exhaustive oracle is independent at the trace-construction level, but shares the authored finite model definitions; it is not an external scientific authority.
- The implementation and source-visible tests are not a blind benchmark.
- No AI agent has been evaluated.
- No empirical physical system, unknown natural law or published scientific result has been reproduced.
- No Unsatz, Scientist World or FrontierPhysics benchmark score is claimed.
- No Rust compilation, test, formatting, Clippy or CI pass is claimed by this PR.
