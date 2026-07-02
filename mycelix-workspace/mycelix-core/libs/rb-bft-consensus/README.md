# rb-bft-consensus

Reputation-Based Byzantine Fault Tolerant consensus for Mycelix - 45% Byzantine tolerance.

## License

MIT OR Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use rb_bft_consensus::consensus::RbBftConsensus;
use mycelix_core_types::k_vector::KVector;

// Create a consensus instance with reputation-weighted voting
let mut consensus = RbBftConsensus::new(config);

// Register validators with their reputation scores
consensus.register_validator(validator_id, reputation_kvector);

// Submit a proposal and reach consensus
let result = consensus.propose(block)?;
```

## Features

- Reputation-weighted Byzantine consensus exceeding the classical 33% limit
- Ed25519 cryptographic signatures for all messages
- Optional BLS12-381 aggregate signatures (`bls` feature)
- Optional post-quantum Dilithium signatures (`pq` feature)
- Optional FROST threshold signatures (`threshold` feature)
- Optional async support via Tokio (`async` feature)
- Integration with mycelix-core-types K-Vector reputation scores

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
