# symthaea-hdc-ltc

**O(D) work per temporal update, independent of elapsed time at fixed D.**

A standalone Rust crate implementing the Hyperdimensional Liquid Time-Constant (HDC-LTC) unified neuron architecture. The neuron's *state* is a hypervector (16,384 dimensions by default) evolved with a solver-free Closed-form Continuous (CfC) update. At fixed dimension, the update cost does not grow with the elapsed time horizon: a 1 ms jump and a 100 s jump perform the same fixed number of O(D) passes. The shared state-complexity factor `backbone_tau` modulates the base time constant, while per-coordinate state/input terms preserve heterogeneous time scales. CfC is an efficient closed-form approximation for liquid time-constant dynamics, not an exact integrator for every arbitrary nonlinear LTC ODE.

## The Key Insight

Traditional recurrent neural networks have two scaling problems:
1. **Weight matrices** are O(D^2) parameters for D-dimensional states
2. **Temporal integration** requires O(steps) ODE solver evaluations per time jump

HDC-LTC solves both:
- **Weight hypervectors** replace a dense recurrent matrix with per-dimension HDC binding (element-wise multiply): O(D) parameters per neuron. This is a more compact, structured parameterization—not an equivalent replacement for every possible dense recurrent map.
- **Closed-form exponential update** avoids repeated ODE solver substeps: O(D) per jump, independent of dt
- **Multi-timescale dynamics** combine `backbone_tau` state-complexity modulation with per-coordinate state/input modulation; the checked API rejects non-finite or negative `backbone_tau`

The closed-form solution:
```
x(t + dt) = sigma * x_inf + (1 - sigma) * x(t)
```

where `x_inf = f(W . x + U . u)` is the equilibrium state, and sigma is an adaptive gating factor. A 1 ms jump costs exactly the same as a 100 s jump.

## Quick Example

```rust
use symthaea_hdc_ltc::{ContinuousHV, NeuronConfig, HdcLtcUnifiedNeuron};

let config = NeuronConfig { dim: 1024, ..NeuronConfig::default() };
let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

let input = ContinuousHV::new_random(1024, 123);

// Work remains O(D) for either duration when dimension and activation are fixed.
neuron.try_evolve_closed_form_fused(0.001, &input).expect("valid 1 ms step");
neuron.try_evolve_closed_form_fused(100.0, &input).expect("valid 100 s step");
```

## Architecture

- **ContinuousHV**: Continuous-valued hypervector with bind, bundle, similarity, permute
- **HdcLtcUnifiedNeuron**: Single neuron with closed-form and fused evolution methods
- **HdcLtcUnifiedNetwork**: Multi-layer network with optional layer binding and skip connections
- **Activation**: Tanh (fast rational approximation), Sigmoid, SiLU, Identity, BoundedTanh

## Performance

- **Neuron evolution**: O(D) per step, independent of dt; it is not O(1) in the hypervector dimension
- **Fused hot path**: `evolve_closed_form_fused()` avoids intermediate heap allocations and fuses a scaled sum-of-squares *post-update bound check* into the update pass. The shared state norm used by `backbone_tau` is computed before that pass. The hot path assumes matching dimensions and well-scaled finite values
- **Checked path**: `try_evolve_closed_form_fused()` validates dt, dimensions, configuration and finite inputs/parameters, then verifies the evolved state; if arithmetic overflow produces a non-finite result, it restores the prior state and counters. Validation and rollback safety add O(D) scans and a state clone
- **Small-step stability**: CfC interpolation uses `-expm1(-dt / tau)` to avoid cancellation for tiny positive dt
- **fast_tanh**: Rational approximation (max 0.4% error) that auto-vectorizes on modern CPUs
- **Deterministic**: Initialization is seed-controlled for reproducibility

## Examples

```bash
cargo run --example time_series   # Sine wave prediction
cargo run --example controller    # PD controller with LTC
cargo run --example language      # Mini thought-to-text demo
cargo run --release -p symthaea-hdc-ltc --example scaling_benchmark # CSV scaling measurements
```

## References

- Hasani, R. et al. (2021). "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence.
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation."
- Plate, T. (2003). "Holographic Reduced Representations." CSLI Publications.

## License

AGPL-3.0-or-later. Commercial licensing available -- see COMMERCIAL_LICENSE.md at repository root.

Part of the [Symthaea](https://github.com/Luminous-Dynamics/symthaea) project by Luminous Dynamics.
