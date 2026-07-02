# Hybrid Threshold Signing Protocol

## Problem Statement

Post-quantum threshold signatures do not yet exist as a mature standard. ML-DSA-65 (FIPS 204) is a single-signer scheme; threshold variants are active research (Cozzo & Smart 2023, Damgard et al. 2024). Naively applying Shamir secret sharing to ML-DSA key material is insecure because lattice-based schemes have different algebraic structure than elliptic curves.

Mycelix needs quantum-resistant governance finality today, not after threshold lattice signatures are standardized.

## Solution: Defense-in-Depth Hybrid

We combine two independent signature layers:

```
Governance Decision
        |
        v
  +-----------+     +-----------+
  | Threshold |     | Designated|
  |   ECDSA   |     |  ML-DSA   |
  | (t-of-n)  |     | Attestor  |
  +-----------+     +-----------+
        |                 |
        v                 v
  ThresholdSignature {
    signature:      [ECDSA r||s],      // 64 bytes
    pq_signature:   [ML-DSA-65 sig],   // 3,309 bytes
    algorithm:      HybridEcdsaMlDsa65,
  }
```

### Layer 1: Threshold ECDSA (Classical Security)

Standard Feldman VSS-based DKG produces a shared secp256k1 key. Any t-of-n members can reconstruct the signing key and produce an ECDSA signature. This is the existing, proven protocol.

- **Security**: 128-bit classical (vulnerable to quantum Shor's algorithm)
- **Threshold**: True t-of-n via Shamir secret sharing over a prime field
- **Maturity**: Decades of deployment, well-understood security proofs

### Layer 2: ML-DSA-65 Attestation (Post-Quantum Security)

A designated committee member (the "PQ attestor") holds a full ML-DSA-65 keypair and co-signs the governance decision. The attestor role rotates per-epoch.

- **Security**: NIST Level 3 post-quantum (128-bit quantum security)
- **Threshold**: Single-signer (NOT threshold), but attestor selection requires t-of-n agreement
- **Maturity**: FIPS 204 finalized August 2024

### Why This Works

An attacker must break BOTH layers to forge a governance signature:

| Attack | ECDSA | ML-DSA | Hybrid |
|--------|-------|--------|--------|
| Classical (today) | Secure | Secure | Secure |
| Quantum (future) | Broken | Secure | Secure |
| Compromised attestor | Secure | Broken | Secure |
| t colluding members | Broken | Secure* | Secure |

*Assuming the attestor is not among the t colluding members.

The hybrid is strictly stronger than either layer alone.

## Protocol Details

### Committee Formation

1. `create_committee` with `signature_algorithm: HybridEcdsaMlDsa65`
2. Standard DKG ceremony produces shared ECDSA key
3. Committee elects PQ attestor (highest trust score, or round-robin by epoch)
4. PQ attestor generates ML-DSA-65 keypair, publishes public key to DHT

### Signing Flow

1. Governance decision reaches finality (proposal approved)
2. t-of-n members reconstruct ECDSA signing key, produce threshold signature
3. PQ attestor independently signs the same content hash with ML-DSA-65
4. `combine_signatures` receives both:
   ```json
   {
     "combined_signature": "<ECDSA bytes>",
     "pq_signature": "<ML-DSA-65 bytes>",
     "signers": [1, 3, 4]
   }
   ```
5. Integrity zome validates both signatures meet minimum length requirements
6. Both signatures stored on-chain for future verification

### Verification

Verifiers check BOTH signatures:
- ECDSA: Verify against committee's combined public key (from DKG)
- ML-DSA-65: Verify against attestor's published ML-DSA public key

A signature is valid only if both verifications pass.

### Key Rotation

On epoch rotation (`rotate_committee_key`):
- ECDSA: Proactive secret sharing refreshes all shares (existing `RefreshRound`)
- ML-DSA: New attestor generates fresh ML-DSA-65 keypair
- Old attestor's key is revoked on-chain

### Attestor Selection

The PQ attestor is selected deterministically:
```
attestor_index = hash(committee_id || epoch) % qualified_member_count
```

This prevents any single member from permanently controlling the PQ key while ensuring all members agree on who the attestor is.

## Violation Integration

Protocol violations affect both layers:

| Violation | Severity | Penalty | Effect |
|-----------|----------|---------|--------|
| DealTimeout | Minor | 0.05 | Trust score reduction |
| InvalidShare | Moderate | 0.15 | Trust score reduction |
| Equivocation | Severe | 0.40 | Barred from future committees |
| InvalidCommitment | Moderate | 0.15 | Trust score reduction |
| CommitRevealMismatch | Severe | 0.40 | Barred from future committees |

Participants with cumulative penalty > 0.50 are barred from joining new committees. The PQ attestor role is never assigned to participants with any severe violations.

## Share Encryption with ML-KEM

During DKG deal distribution, shares are encrypted using ML-KEM-768 (FIPS 203) key encapsulation:

1. Each participant publishes an ML-KEM-768 encapsulation key
2. Dealer encapsulates a shared secret per recipient: `(ct, ss) = Encapsulate(ek)`
3. Share bytes are encrypted with `SHA-256(ss)` as XOR key
4. Recipient decapsulates: `ss = Decapsulate(dk, ct)`, recovers share

This protects share confidentiality against both classical and quantum eavesdroppers during the DKG ceremony, complementing the post-quantum signature layer.

## Future: True Threshold ML-DSA

When threshold lattice signatures mature, the protocol can upgrade:

1. Add `ThresholdSignatureAlgorithm::ThresholdMlDsa65` variant
2. Run parallel DKG: one for ECDSA, one for ML-DSA key material
3. Both layers become true t-of-n threshold signatures
4. Remove attestor role (no longer needed)
5. Hybrid validation requires both threshold signatures

The current `signature_algorithm` field on `SigningCommittee` supports this evolution without breaking existing committees.

## References

- NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM), August 2024
- NIST FIPS 204: Module-Lattice-Based Digital Signature Standard (ML-DSA), August 2024
- Cozzo, D. & Smart, N.P. (2023). Sharing the LUOV: Threshold Post-Quantum Signatures. CT-RSA.
- Damgard, I. et al. (2024). Threshold Signatures from Lattice Assumptions. EUROCRYPT.
- Feldman, P. (1987). A Practical Scheme for Non-interactive Verifiable Secret Sharing. FOCS.
- Gennaro, R. et al. (1999). Secure Distributed Key Generation for Discrete-Log Based Cryptosystems. EUROCRYPT.
