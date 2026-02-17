# Security Model for zkSTARK Proof System

This document describes the security assumptions, threat model, and guarantees provided by the fl-aggregator proof system.

## Overview

The proof system uses Winterfell zkSTARK proofs to provide verifiable computation for federated learning. STARKs (Scalable Transparent ARguments of Knowledge) offer:

- **Post-quantum security**: No reliance on elliptic curves or number-theoretic assumptions
- **Transparency**: No trusted setup required
- **Scalability**: Verification time is polylogarithmic in computation size

## Important Limitations

### Not Perfect Zero-Knowledge

**Winterfell proofs are NOT perfectly zero-knowledge.** The proofs may leak information about the secret witness. This is acceptable for our use cases because:

1. **Verification focus**: We prove validity/integrity, not secrecy
2. **Public inputs**: Most data we're proving about is not secret (ranges, norms, etc.)
3. **Practical security**: The leakage is bounded and doesn't reveal full witness

For applications requiring perfect zero-knowledge, consider:
- zk-SNARKs (Groth16, PLONK)
- Bulletproofs
- Adding additional masking/blinding

## Security Levels

| Level | Security Bits | Hash Queries | Use Case |
|-------|--------------|--------------|----------|
| Standard96 | 96 | 27 | Development, low-stakes verification |
| Standard128 | 128 | 36 | Production, general use |
| High256 | 256 | 72 | High-security applications |

### Choosing a Security Level

- **Standard96**: Suitable for testing and development. Provides adequate security for non-critical operations.
- **Standard128** (default): Recommended for production. Matches AES-128 security level.
- **High256**: Use for high-value transactions or long-term security. Higher computational cost.

## Threat Model

### What We Protect Against

1. **Malicious provers**: Cannot create valid proofs for false statements
2. **Proof forgery**: Cannot modify proofs without detection
3. **Replay attacks**: Timestamped proofs prevent reuse
4. **Parameter tampering**: Public inputs are bound to the proof

### What We Do NOT Protect Against

1. **Compromised verifier**: Verifier must be trusted to run verification
2. **Side-channel attacks**: Timing attacks may leak information during proof generation
3. **Quantum computers**: Standard96/128 may be vulnerable to Grover's algorithm (quantum speedup on hash functions)
4. **Witness leakage**: Proofs may leak partial information about the witness

## Circuit-Specific Guarantees

### RangeProof

**Proves**: Value is within [min, max] range

**Guarantees**:
- Prover knows a value V such that min <= V <= max
- Value is committed via bit decomposition

**Assumptions**:
- 64-bit value representation
- Honest bit decomposition (enforced by constraints)

### MembershipProof

**Proves**: Leaf exists in Merkle tree with given root

**Guarantees**:
- Prover knows a valid Merkle path from leaf to root
- Path authenticity verified via Rescue hash

**Assumptions**:
- Hash function is collision-resistant
- Merkle tree was honestly constructed

### GradientIntegrityProof

**Proves**: Gradient vector has bounded L2 norm

**Guarantees**:
- Prover knows gradient G with ||G||_2 <= max_norm
- Gradient commitment binds the values

**Assumptions**:
- Floating-point to field conversion preserves ordering
- Scaling factor (1000) is sufficient for precision

### IdentityAssuranceProof

**Proves**: Identity meets assurance level threshold

**Guarantees**:
- DID commitment binds the identity
- Factor contributions sum to meet threshold
- Only active factors are counted

**Assumptions**:
- Factors were honestly obtained
- Assurance level thresholds are correctly defined

### VoteEligibilityProof

**Proves**: Voter meets eligibility requirements for proposal type

**Guarantees**:
- All eligibility criteria are checked
- Voter profile is committed and bound

**Assumptions**:
- Profile data is accurate
- Eligibility rules are correctly encoded

## Cryptographic Primitives

### Hash Function: Blake3

- 256-bit output
- Collision resistance: 128 bits
- Preimage resistance: 256 bits

### Field: 128-bit Prime Field

- Modulus: 2^128 - 45 * 2^40 + 1
- Efficient arithmetic
- Sufficient security margin

### Rescue Hash (for Merkle trees)

- Algebraic hash function
- STARK-friendly (low degree)
- 256-bit security target

## Operational Security

### Key Management

- HSM integration available for sensitive operations
- Threshold signatures (FROST) for distributed key management
- Key rotation support

### Rate Limiting

- Configurable rate limits prevent DoS
- Token bucket algorithm
- Per-user quotas

### Audit Logging

- All proof operations logged
- Tamper-evident audit trail
- Compliance support (GDPR, SOC2)

## Best Practices

### For Developers

1. **Always verify proofs** before trusting the result
2. **Use Standard128** or higher for production
3. **Validate public inputs** before verification
4. **Implement rate limiting** to prevent abuse
5. **Log all verification failures** for security monitoring

### For Operators

1. **Keep dependencies updated** for security patches
2. **Monitor for unusual patterns** in proof verification
3. **Use HSM** for production deployments
4. **Enable audit logging** and review regularly
5. **Test disaster recovery** procedures

### For Security Auditors

Key areas to review:

1. **Constraint satisfaction**: Do AIR constraints correctly encode the statement?
2. **Field arithmetic**: Are there overflow/underflow issues?
3. **Serialization**: Is deserialization safe against malformed input?
4. **Error handling**: Are errors handled without information leakage?
5. **Timing**: Are operations constant-time where needed?

## Known Vulnerabilities

### Mitigated

- **CVE-XXXX-XXXX**: (None currently known)

### Under Investigation

- Timing side-channels in field arithmetic (low priority)
- Memory safety in large proof handling (fuzzing in progress)

## Reporting Security Issues

Please report security vulnerabilities to: security@luminous-dynamics.com

- Include detailed reproduction steps
- Do not disclose publicly until patch is available
- We follow responsible disclosure practices

## References

- [Winterfell Documentation](https://github.com/facebook/winterfell)
- [STARK Whitepaper](https://eprint.iacr.org/2018/046)
- [Rescue Hash Paper](https://eprint.iacr.org/2020/1143)
- [FROST Threshold Signatures](https://eprint.iacr.org/2020/852)
