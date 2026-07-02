# mycelix-fl

Rust-native federated learning aggregation with Byzantine fault tolerance.

## Algorithms

| Algorithm | Module | BFT Guarantee | Reference |
|-----------|--------|---------------|-----------|
| FedAvg | `defenses::fedavg` | None (baseline) | McMahan et al., 2017 |
| Trimmed Mean | `defenses::trimmed_mean` | f < (1-2β)n | Yin et al., 2018 |
| Coordinate Median | `defenses::coordinate_median` | f < n/2 | Yin et al., 2018 |
| RFA (Geometric Median) | `defenses::rfa` | Provably robust | Pillutla et al., 2022 |

## Usage

```rust
use mycelix_fl::{Gradient, DefenseConfig};
use mycelix_fl::defenses::{Defense, FedAvg};

let gradients = vec![
    Gradient { values: vec![1.0, 2.0, 3.0], node_id: "a".into(), round: 0 },
    Gradient { values: vec![1.1, 1.9, 3.1], node_id: "b".into(), round: 0 },
];

let config = DefenseConfig::default();
let result = FedAvg.aggregate(&gradients, &config).unwrap();
```

## Features

- `std` (default) — standard library support
- `wasm` — WebAssembly compatibility

## License

AGPL-3.0-or-later. Commercial licensing available — see COMMERCIAL_LICENSE.md at repository root.
