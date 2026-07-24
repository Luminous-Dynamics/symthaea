# symthaea-hdc-crypto

> **Quarantined insecure research code. Do not use for security or production.**

This crate preserves historical hyperdimensional-computing masking and algebra
experiments for compatibility, reproducibility, and adversarial study. Its
security-labelled type names are deprecated because the constructions do not
satisfy the cryptographic definitions their names imply.

## Confirmed failures

- `HdcMac` is a linear tag. One known message/tag pair permits forgery for any
  other message.
- `HdcThresholdSharing` places both `secret XOR mask` and `mask` in every share,
  so one share recovers the secret and `k` is not a security threshold.
- `HdcCommitment` is a cyclic rotation. It is neither hiding nor binding.
- `CollectiveWisdomPool` reuses a shared XOR mask. Ciphertext pairs reveal
  plaintext XORs and Hamming distances; this is not FHE or privacy-preserving
  aggregation.
- `HdcContextKey` hashes deterministic context encodings without independent
  secret entropy. Hashing does not make enumerable sensor states secret.

Executable attack demonstrations live in the crate tests. The full inventory,
impact classification, and remediation gates are recorded in
`docs/security/2026-07-13-cryptographic-claim-integrity.md` in the Symthaea
repository.

## Legitimate research use

The crate can still demonstrate:

- XOR binding and inversion;
- majority-vote bundling;
- Hamming-similarity preservation under a common XOR transform;
- why functional round-trip tests are not cryptographic security evidence.

Use established, independently reviewed constructions for authentication,
secret sharing, commitments, authenticated encryption, key establishment, and
homomorphic encryption.
