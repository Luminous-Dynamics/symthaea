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
