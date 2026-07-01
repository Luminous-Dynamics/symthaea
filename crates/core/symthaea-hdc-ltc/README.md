# symthaea-hdc-ltc

**O(1) temporal dynamics in 16,384 dimensions.**

A standalone Rust crate implementing the Hyperdimensional Liquid Time-Constant (HDC-LTC) unified neuron architecture. This neuron's *state* is a 16,384-dimensional hypervector that evolves through Liquid Time-Constant dynamics with a closed-form solution, enabling O(1) temporal jumps to any time horizon.

## The Key Insight

Traditional recurrent neural networks have two scaling problems:
1. **Weight matrices** are O(D^2) parameters for D-dimensional states
2. **Temporal integration** requires O(steps) ODE solver evaluations per time jump

HDC-LTC solves both:
- **Weight hypervectors** replace weight matrices via HDC binding (element-wise multiply): O(D) parameters
- **Closed-form exponential decay** replaces ODE integration: O(1) per temporal jump

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

// O(1) temporal jumps -- both cost the same!
neuron.evolve_closed_form(0.001, &input);  // 1 ms
neuron.evolve_closed_form(100.0, &input);   // 100 seconds
```

## Architecture

- **ContinuousHV**: Continuous-valued hypervector with bind, bundle, similarity, permute
- **HdcLtcUnifiedNeuron**: Single neuron with closed-form and fused evolution methods
- **HdcLtcUnifiedNetwork**: Multi-layer network with optional layer binding and skip connections
- **Activation**: Tanh (fast rational approximation), Sigmoid, SiLU, Identity, BoundedTanh

## Performance

- **Neuron evolution**: O(D) per step, independent of dt
- **Zero-alloc fused path**: `evolve_closed_form_fused()` avoids all intermediate allocations
- **fast_tanh**: Rational approximation (max 0.4% error) that auto-vectorizes on modern CPUs
- **Deterministic**: All operations are seed-controlled for reproducibility

## Examples

```bash
cargo run --example time_series   # Sine wave prediction
cargo run --example controller    # PD controller with LTC
cargo run --example language      # Mini thought-to-text demo
```

## References

- Hasani, R. et al. (2021). "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence.
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation."
- Plate, T. (2003). "Holographic Reduced Representations." CSLI Publications.

## License

AGPL-3.0-or-later. Commercial licensing available -- see COMMERCIAL_LICENSE.md at repository root.

Part of the [Symthaea](https://github.com/Luminous-Dynamics/symthaea) project by Luminous Dynamics.
