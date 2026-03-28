# symthaea-hdc-crypto

**Information-theoretically secure homomorphic encryption on hyperdimensional binary vectors.**

Traditional FHE (fully homomorphic encryption) schemes based on lattice problems impose 1000x+ computational overhead. This crate exploits the algebraic structure of 16,384-bit binary hypervectors to achieve **homomorphic encryption at plaintext speed** -- XOR binding distributes over XOR encryption, and majority-vote bundling approximately commutes with the same mask. The result is a privacy-preserving computation toolkit with nanosecond-scale operations and Shannon-perfect secrecy.

## Security Model

- **Perfect secrecy** (Shannon 1949): One-time pad encryption with D = 16,384 bit masks. Ciphertext is statistically independent of plaintext.
- **Zero collision probability**: XOR binding is a bijection -- distinct inputs always yield distinct MACs.
- **Information-theoretic threshold sharing**: (k,n) secret splitting where fewer than k shares reveal zero bits about the secret.
- **Noise-tolerant verification**: HDC-MAC supports lossy channels (LoRa, BLE) with configurable similarity thresholds and false-positive rate ~ 2^{-4700}.

## Quick Example

```rust
use symthaea_hdc_crypto::{BinaryHV, EncryptedHV, HdcMac, HdcThresholdSharing};

// Encrypt
let plaintext = BinaryHV::new_random(42);
let mask = BinaryHV::new_random(99);
let encrypted = EncryptedHV::encrypt(&plaintext, &mask);

// Homomorphic bind (works on encrypted data!)
let other = BinaryHV::new_random(7);
let other_mask = BinaryHV::new_random(88);
let enc_other = EncryptedHV::encrypt(&other, &other_mask);
let enc_bound = encrypted.hom_bind(&enc_other);

// Decrypt the result
let combined_mask = mask.bind(&other_mask);
let result = enc_bound.decrypt(&combined_mask);
assert_eq!(result, plaintext.bind(&other));

// Authenticate
let key = BinaryHV::new_random(55);
let mac = HdcMac::compute(&plaintext, &key);
assert!(HdcMac::verify(&plaintext, &key, &mac));

// Threshold secret sharing (3-of-5)
let shares = HdcThresholdSharing::split(&plaintext, 3, 5, 1000);
let recovered = HdcThresholdSharing::recover(&shares[..3]);
assert_eq!(recovered, plaintext);
```

## API Overview

### `BinaryHV` -- 16,384-bit hypervector

| Method | Description |
|--------|-------------|
| `new_random(seed)` | Deterministic random vector (BLAKE3 XOF) |
| `zero()` | All-zero vector |
| `bind(&other)` | XOR binding (commutative, self-inverse) |
| `bundle(&[vectors])` | Majority-vote bundling |
| `similarity(&other)` | Hamming similarity in [0, 1] |
| `permute(shift)` | Cyclic bit rotation |
| `density()` | Fraction of 1-bits |

### `EncryptedHV` -- OTP-encrypted hypervector

| Method | Description |
|--------|-------------|
| `encrypt(plaintext, mask)` | One-time pad encryption |
| `decrypt(mask)` | Decryption (XOR is self-inverse) |
| `hom_bind(&other)` | Homomorphic bind on ciphertexts |
| `encrypted_similarity(&other)` | Same-mask similarity preserved |

### `CollectiveWisdomPool` -- Privacy-preserving aggregation

| Method | Description |
|--------|-------------|
| `new()` / `with_capacity(n)` | Create pool |
| `contribute(peer_id, encrypted)` | Add encrypted contribution |
| `aggregate()` | Majority-vote bundle of all contributions |
| `clear()` | Reset for next round |

### `HdcMac` -- Message authentication

| Method | Description |
|--------|-------------|
| `compute(message, key)` | Compute MAC (~5-10 ns) |
| `verify(message, key, mac)` | Exact verification |
| `verify_noisy(message, key, mac, threshold)` | Lossy-channel verification |

### `HdcThresholdSharing` -- (k,n) secret splitting

| Method | Description |
|--------|-------------|
| `split(secret, k, n, seed)` | Split into n shares requiring k |
| `recover(&shares)` | Reconstruct from k+ shares |
| `recovery_quality(&shares)` | Check reconstruction fidelity |

### `HdcContextKey` -- Sensor-derived keys

| Method | Description |
|--------|-------------|
| `derive(&sensors)` | Bind+permute chain from sensor HVs |
| `to_symmetric_key(context)` | Extract 256-bit key via BLAKE3 |
| `derive_symmetric(&sensors)` | Combined derive + extract |

### `HdcCommitment` -- Permutation-based commitments

| Method | Description |
|--------|-------------|
| `commit(secret, offset)` | Create commitment |
| `verify(commitment, secret, offset)` | Exact verification |
| `verify_noisy(...)` | Lossy-channel verification |

## What This Is NOT

This crate supports only the HDC algebra (bind, bundle, similarity). It is not a general-purpose FHE scheme, not a replacement for AES/ChaCha20 for bulk encryption, and not a digital signature system.

## License

AGPL-3.0-or-later. Commercial licensing available -- contact Luminous Dynamics.

## Part of Symthaea

This crate is extracted from the [Symthaea](https://github.com/Luminous-Dynamics/symthaea) consciousness framework. The full `symthaea-core` crate includes SIMD-accelerated operations, weighted bundling, density normalization, and integration with the HDC cognitive pipeline.
