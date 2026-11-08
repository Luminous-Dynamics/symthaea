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

The numeric operator CLI accepts evaluator-side scenario values and prints matched A0/A2 reports plus the oracle class count. Its truth-bearing arguments and oracle report are not an agent-facing interface.

## Two-input variable and interaction microcosm

The public Rust module interaction adds a static two-input world for discovering variable relevance and cross terms.

Its legal intervention pairs are the full factorial set {-1, 0, 1} × {-1, 0, 1}. Candidate structures are:

- U-only: y = bias + a*u. The second input is irrelevant.
- V-only: y = bias + b*v. The first input is irrelevant.
- Additive two-input: y = bias + a*u + b*v.
- Bilinear interaction: y = bias + a*u + b*v + c*u*v.

Bias and nonzero coefficient ranges are finite and explicitly enumerated. A0 supplies the structural family and fits coefficients; A2 receives the same observations and must discover the structural family as well.

The important negative control is a collinear design with only (u,v)=(1,1) and (-1,-1). For a bilinear world, the interaction term is constant on these two points and can be absorbed into the bias. U-only, V-only, additive and bilinear models all fit those two observations. The adaptive one-step selector searches all nine legal input pairs, excludes previously observed pairs and updates the candidate set after each result. In this expanded family, two adaptive probes are required for the supplied collinear example; the first alone does not guarantee uniqueness. The factorial oracle identifies candidate response functions over the complete domain.

This module has no hidden state, stochastic noise or unknown sensor map. Those are covered separately by the other reference worlds. It is a controlled test of variable relevance, interaction discovery and experimental-design aliasing, not a general causal-discovery engine.

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
    cargo run --manifest-path research/open-world-ansatz-microcosm/Cargo.toml --bin numeric_microcosm -- leaky 2 1 1 identity 1,-1,1,0

CLI scenario arguments identify the hidden evaluator-side world. They are for an operator-controlled local run; do not expose them to the agent process in a blind evaluation. The JSON output includes an oracle summary and is an evaluator report, not an agent observation packet.

For the matched A0 control, use the same world and intervention sequence, changing only the final argument to a0.

## Experiment-selection controls

`strategies::ProbeStrategy` supplies six comparable one-step policies: maximum partition count, minimax worst-case survivors, maximum entropy, minimum expected remaining candidates under a uniform prior, fixed grid, and seeded random. Each action is scored over the candidates still compatible with the public evidence. The score exposes partition count, worst-case survivors, exact expected-survivor numerator/denominator, entropy and action cost.

These are deliberately different objectives. Maximum entropy and expected survivors assume a uniform prior over the surviving finite grammar; they are not calibrated Bayesian posteriors or decision-focused EVSI. The random arm is seeded and reproducible. Independent two-input probes already used in the episode are excluded; repeat inputs remain available in stateful Boolean/numeric worlds.

`policy::StrategyEnumeratorBaseline` combines each strategy with the same exact finite inference/submission logic, enabling per-strategy matched A0/A2 episodes. No comparative performance ranking is claimed until the exact-head Rust campaign runs and its full per-world metrics are recorded.

## Finite enumerator agent baseline v0

`policy::FiniteEnumeratorBaseline` consumes only an `AgentView` and emits a probe or terminal submission. It maintains the exact finite compatible-model set and uses the existing partition-maximizing probe heuristic in a receding-horizon loop. It is a transparent algorithmic reference, not a trained research policy. Paired A0/A2 tests use the same hidden model and action budget; the external harness still needs to preregister and randomize task seeds and ensure agent/evaluator process isolation.

## Blind agent episode protocol v0

The public Rust module protocol exposes a bounded, one-action-at-a-time interface over all three worlds. Agents receive a PublicTaskManifest, observations, remaining budget and an evidence-prefix SHA-256; they return typed probes or a terminal version-space submission. The submission must echo the current evidence-prefix digest and cite observation IDs that were actually returned.

The protocol validates A0/A2 structure access, intervention family, action budget, terminal submission, model-family membership, duplicate hypotheses, disposition cardinality and evidence references. Evaluator scoring is available only after submission and reports exact parameterized-model version-space match separately from the projected structural-family set. This prevents parameter or sensor non-identifiability from being mis-scored as structural failure, while preserving cross-family ambiguity under uninformative sensors. Set counts, disposition correctness and evidence-reference validity remain separate. The wire schema is research/open-world-ansatz-microcosm/schemas/agent_episode_v0.schema.json; the full contract and process-isolation caveat are in docs/research/OPEN_WORLD_AGENT_EPISODE_PROTOCOL_V0_2026-11-08.md.

Important boundary: this Rust API is not an OS security sandbox. The agent must run in a separate process and must never receive the evaluator handle or pre-submission score. The v0 digest is computed over this version's serde_json encoding; an agent should echo the digest rather than independently reserialize it in another language.

## Authored tests

Sixty-two Rust tests cover the three reference worlds and blind protocol: hidden-state transitions, matched A0/A2 access, coefficient recovery, sensor-gain equivalence, passive and sensor-induced non-identifiability, variable relevance, interaction aliasing, adaptive probes, exhaustive oracle equivalence classes, evidence-prefix integrity, claim validation, probe budgets, terminal submissions, exact version-space scoring, structure-versus-parameter identifiability and safe-versus-unnecessary abstention scoring.

The previous exact-head focused runs covered 34 tests on commit 58128d025bdb24995c31822a86cd3afcaec28413. The additional protocol tests and wire contract must be qualified on the new exact branch head before this expanded branch is called green.

## Claim boundaries

- The crate is an executable synthetic microcosm and finite exact enumerator, not an agent result.
- The exhaustive oracle is independent at the trace-construction level, but shares the authored finite model definitions; it is not an external scientific authority.
- The implementation and source-visible tests are not a blind benchmark.
- No AI agent has been evaluated.
- No empirical physical system, unknown natural law or published scientific result has been reproduced.
- No Unsatz, Scientist World or FrontierPhysics benchmark score is claimed.
- The numeric extension's exact-head Rust compilation, tests, formatting and Clippy status must be read from its focused workflow; do not infer it from the earlier binary-v1 pass.
