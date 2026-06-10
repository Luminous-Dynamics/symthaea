# Crates

Sub-crates for specific functionality.

## sophia-gym

Reinforcement learning environment for training consciousness-aware agents.

```rust
// Usage
use sophia_gym::SophiaEnv;

let env = SophiaEnv::new();
let obs = env.reset();
let (next_obs, reward, done) = env.step(action);
```

## symthaea-gym

Consciousness training environment - OpenAI Gym-compatible interface for training agents that interact with the consciousness system.

```rust
// Usage
use symthaea_gym::SymthaeaEnv;

let env = SymthaeaEnv::new(config);
// Train with any RL algorithm
```

## Building

```bash
# Build all crates
cargo build -p sophia-gym -p symthaea-gym

# Run tests
cargo test -p sophia-gym
cargo test -p symthaea-gym
```

## Engineering Crates

The engineering track is intentionally split into lightweight crates that keep
external CAD/solver dependencies out of default builds:

- `symthaea-sim-bridge`: normalized simulation request/result types and backend
  traits for FEA, CFD, multibody, circuit, and process tools.
- `symthaea-digital-twin`: engineered asset telemetry, health, and free-energy
  trend tracking.
- `symthaea-formal-safety`: safety cases, proof obligations, and evidence
  records.
- `symthaea-engineering`: facade tying requirements, concepts, simulations,
  twins, and safety gates together.
- `symthaea-mujoco-bridge`: dry-run generic MuJoCo backend boundary.
- `symthaea-opensees-bridge`: dry-run OpenSees structural backend boundary.
- `symthaea-ngspice-bridge`: dry-run ngspice circuit backend boundary.
- `symthaea-openfoam-bridge`: dry-run OpenFOAM CFD backend boundary.

See `docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md` for the 18-month plan.
