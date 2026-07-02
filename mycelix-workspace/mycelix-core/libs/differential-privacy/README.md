# mycelix-differential-privacy

Differential Privacy primitives for Mycelix Federated Learning.

## License

Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use mycelix_differential_privacy::mechanisms::GaussianMechanism;
use mycelix_differential_privacy::budget::PrivacyBudget;

// Create a privacy budget (epsilon = 1.0, delta = 1e-5)
let budget = PrivacyBudget::new(1.0, 1e-5);

// Apply Gaussian noise to a gradient
let mechanism = GaussianMechanism::new(budget.epsilon, budget.delta, sensitivity);
let noisy_value = mechanism.apply(true_value);
```

## Features

- Gaussian and Laplace noise mechanisms
- Privacy budget tracking and composition
- Gradient clipping for bounded sensitivity
- Advanced composition theorems (Renyi DP)
- Optional Python bindings via PyO3 (`python` feature)

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
