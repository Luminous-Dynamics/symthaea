# mycelix-core-types

Core types for the Mycelix ecosystem - K-Vector, E/N/M Classification, GIS v4 types.

## License

MIT OR Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use mycelix_core_types::k_vector::KVector;
use mycelix_core_types::epistemic::{EAxis, NAxis, MAxis, ENMClassification};

// Create a K-Vector
let kv = KVector::new(vec![0.8, 0.2, 0.5, 0.1]);

// Classify a claim using the Epistemic Cube
let classification = ENMClassification {
    e: EAxis::E3,  // Peer-reproducible
    n: NAxis::N2,  // Network consensus
    m: MAxis::M2,  // Long-term significance
};
```

## Features

- K-Vector type for multi-dimensional trust/quality scoring
- Epistemic Cube (E/N/M) classification for all claims
- GIS v4 spatial types
- Optional Holochain integration (`holochain` feature)
- Optional Serde serialization (`serde` feature)
- Optional Python bindings via PyO3 (`python` feature)

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
