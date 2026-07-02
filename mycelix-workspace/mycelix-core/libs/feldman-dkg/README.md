# feldman-dkg

Feldman's Verifiable Secret Sharing for Distributed Key Generation.

## License

MIT OR Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use feldman_dkg::ceremony::DkgCeremony;
use feldman_dkg::participant::Participant;

// Set up a (t, n) threshold ceremony: t=2 of n=3
let mut ceremony = DkgCeremony::new(2, 3);

// Each participant generates and distributes shares
let participants: Vec<Participant> = ceremony.generate_participants();

// Participants verify commitments and reconstruct the shared key
let shared_public_key = ceremony.run()?;
```

## Features

- Feldman's Verifiable Secret Sharing (VSS) with polynomial commitments
- Distributed Key Generation ceremonies with configurable thresholds
- secp256k1 elliptic curve cryptography
- Secure memory handling with zeroize
- Optional ML-KEM-768 post-quantum key encapsulation (`ml-kem-768` feature)
- Optional ML-DSA-65 post-quantum signatures (`ml-dsa-65` feature)
- WASM-compatible (no getrandom activation by default)

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
