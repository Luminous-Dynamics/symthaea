# symthaea-hdc-ltc

**O(D) event updates over a 16,384-dimensional continuous hypervector state, with cost independent of the elapsed `dt`.**

A standalone Rust crate implementing Symthaea's Hyperdimensional Liquid Time-Constant (HDC-LTC) recurrent architecture. The recurrent state itself is a continuous hypervector. Each event evaluates a state/input-dependent liquid field and then applies a solver-free exponential relaxation update.

## The Key Insight

Dense recurrent networks commonly use O(D²) recurrent parameters for a D-dimensional state. Numerical continuous-time models may also require multiple solver evaluations across a time interval.

HDC-LTC explores a different tradeoff:

- **Weight hypervectors** replace dense recurrent matrices with elementwise HDC fields: O(D) recurrent parameters per neuron.
- **Solver-free temporal evolution** uses one exponential relaxation update per event: O(D) work independent of the number of numerical ODE substeps that the elapsed `dt` might otherwise require.
- **Liquid log-timescales** map an unconstrained per-dimension control signal across `[tau_min, tau_max]` in log space while preserving `tau_base` as the neutral timescale.

The frozen-field event update is:

```text
control_i = tau_mod_i * |h_i| + tau_coupling * |input_i|
tau_i     = LogInterpolate(tau_min, tau_base, tau_max, control_i)
alpha_i   = 1 - exp(-dt / tau_i)
h_i'      = (1 - alpha_i) * h_i + alpha_i * h_inf_i
```

This is intentionally described as a **solver-free frozen-field exponential update**, not as the exact solution of a general nonlinear LTC ODE. A 1 ms event interval and a 100 s event interval execute the same number of arithmetic operations for a fixed D; they produce different state transitions because `alpha_i` depends on `dt`.

## Quick Example

```rust
use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

let config = NeuronConfig {
    dim: 1024,
    tau_min: 0.001,
    tau_base: 0.1,
    tau_max: 100.0,
    ..NeuronConfig::default()
};
let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
let input = ContinuousHV::new_random(1024, 123);

neuron.evolve_closed_form(0.001, &input); // 1 ms elapsed time
neuron.evolve_closed_form(100.0, &input); // 100 s elapsed time
```

## Architecture

- **ContinuousHV**: continuous-valued hypervector with bind, bundle, similarity, and permutation.
- **HdcLtcUnifiedNeuron**: one continuous hypervector state with per-dimension gates and liquid timescales.
- **HdcLtcUnifiedNetwork**: multi-layer network with optional layer binding, skip connections, and irregular timestamp support.
- **Activation**: Tanh, Sigmoid, SiLU, Identity, and BoundedTanh.

## Current Research Boundary

This crate establishes a compact continuous-time HDC substrate. It does **not** yet establish that generic continuous binding is unitary, that liquid dynamics preserve VSA algebra, or that the architecture outperforms modern state-space/recurrent baselines. Those are explicit research questions and should be tested through ablations rather than assumed from implementation structure.

The next HLS research stages are:

1. unitary role/binding semantics,
2. algebra-preservation tests (binding/permutation equivariance),
3. structured O(KD) cross-dimensional mixing,
4. independently ablated learning of gates and liquid timescales,
5. state-tracking comparisons against strong recurrent and state-space baselines.

## Performance Contract

- **Neuron evolution**: O(D) per event for a fixed neuron, independent of elapsed `dt`.
- **Zero-allocation fused path**: `evolve_closed_form_fused()` avoids intermediate heap allocations in the recurrent loop.
- **Bounded state**: post-update soft norm bound protects the recurrent state from runaway magnitude.
- **Deterministic initialization/replay**: seed-controlled initialization and property tests cover deterministic replay.

## Examples

```bash
cargo run --example time_series
cargo run --example controller
cargo run --example language
```

## References

- Hasani, R. et al. (2022). "Closed-form Continuous-time Neural Networks." *Nature Machine Intelligence*.
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation."
- Plate, T. (2003). "Holographic Reduced Representations." CSLI Publications.

## License

AGPL-3.0-or-later. Commercial licensing available -- see `COMMERCIAL_LICENSE.md` at repository root.

Part of the [Symthaea](https://github.com/Luminous-Dynamics/symthaea) project by Luminous Dynamics.
