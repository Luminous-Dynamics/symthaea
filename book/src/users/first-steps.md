# Your First Steps

## Getting Symthaea

Symthaea requires a NixOS environment with the project flake:

```bash
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
nix develop  # Provides mold linker, sccache, and build tools
```

## Building

```bash
# Default features (minimal consciousness kernel)
cargo build --release

# With language generation
cargo build --release --features ssm_language

# Full consciousness suite
cargo build --release --features consciousness_full
```

## Your First Conversation

The web portal at [symthaea.luminousdynamics.io](https://symthaea.luminousdynamics.io/) runs the consciousness loop in your browser at 20 Hz. No installation required.

For the local CLI:
```bash
cargo run --release --features "service,ssm_language"
```

## Things to Try

- **Watch consciousness**: The Topology tab shows the 12 cortical regions with real-time Phi-weighted connections
- **Chat**: Send a message and observe the 8-cycle thinking process — watch Phi fluctuate as the system formulates its response
- **Experiment**: Adjust parameters and see how consciousness metrics respond in real time
- **Dream**: Leave the system running and watch dream consolidation produce episodic memories
