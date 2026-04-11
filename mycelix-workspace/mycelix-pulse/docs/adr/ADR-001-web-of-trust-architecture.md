# ADR-001: Web of Trust Architecture

## Status

Accepted

## Date

2024-01-15

## Context

Mycelix Mail needs a decentralized trust system that allows users to establish trust relationships without relying on a central authority. The system must:

1. Allow users to attest trust in other email addresses
2. Propagate trust transitively through the network
3. Calculate trust scores for unknown senders
4. Respect user privacy and prevent manipulation
5. Work across federated Mycelix instances

Traditional approaches include:
- **Centralized reputation systems**: Easy but violates decentralization principles
- **Blockchain-based**: Immutable but slow and costly
- **Pure PGP web of trust**: Proven but has UX issues

## Decision

We implement a **hybrid web of trust** architecture combining:

### 1. Direct Attestations

Users create signed trust attestations:

```rust
struct TrustAttestation {
    from_email: String,      // Attester's email
    to_email: String,        // Subject's email
    level: TrustLevel,       // 1-5 trust level
    context: String,         // Optional context
    created_at: DateTime,
    expires_at: Option<DateTime>,
    signature: Ed25519Signature,
}
```

### 2. Transitive Trust Calculation

Trust propagates through the network with decay:

```
trust(A → C) = trust(A → B) × trust(B → C) × decay_factor
```

Where:
- Maximum path length: 6 hops
- Decay factor: 0.8 per hop
- Multiple paths are combined using weighted average

### 3. Local Trust Store

Each user maintains a local trust database:
- Direct attestations they've made
- Received attestations from others
- Cached calculated scores

### 4. Federation Protocol

Trust attestations sync between instances:
- Pull-based synchronization
- Signed attestation bundles
- Configurable trust import policies

### 5. Anti-Gaming Measures

- Rate limiting on attestation creation
- Sybil resistance through email verification
- Attestation cost (computational proof-of-work)
- Revocation propagation

## Algorithm

```python
def calculate_trust_score(from_user, to_email):
    # Check direct attestation
    direct = get_direct_attestation(from_user, to_email)
    if direct:
        return direct.level / 5.0

    # Find trust paths
    paths = find_trust_paths(from_user, to_email, max_depth=6)

    if not paths:
        return 0.0  # Unknown

    # Calculate path scores
    scores = []
    for path in paths:
        score = 1.0
        for i in range(len(path) - 1):
            attestation = get_attestation(path[i], path[i+1])
            score *= (attestation.level / 5.0) * DECAY_FACTOR
        scores.append(score)

    # Weighted combination (favor shorter paths)
    weights = [1.0 / (len(path) ** 2) for path in paths]
    final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    return min(final_score, 1.0)
```

## Trust Levels

| Level | Name | Description |
|-------|------|-------------|
| 5 | Very High | Personally verified, would vouch completely |
| 4 | High | Known and trusted contact |
| 3 | Medium | Professional relationship, generally reliable |
| 2 | Low | Aware of, limited trust |
| 1 | Very Low | Known but not trusted |
| 0 | Unknown | No attestation exists |

## Consequences

### Positive

1. **Decentralized**: No single point of failure or control
2. **Privacy-preserving**: Trust relationships are user-controlled
3. **Resistant to spam**: Unknown senders are flagged
4. **Gradual trust building**: New contacts can build trust over time
5. **Federation-ready**: Works across Mycelix instances

### Negative

1. **Cold start problem**: New users have no trust network
2. **Computation overhead**: Path finding is O(n²) in worst case
3. **Complexity**: More complex than centralized reputation
4. **Sybil vulnerability**: Motivated attackers can create fake identities

### Mitigations

1. **Cold start**: Import from existing email contacts, PGP keyservers
2. **Performance**: Aggressive caching, limit path depth
3. **Complexity**: Good UX abstracts complexity from users
4. **Sybil**: Email verification, proof-of-work, rate limiting

## Implementation Notes

### Data Storage

- Graph database (Neo4j) for trust relationships
- PostgreSQL for attestation metadata
- Redis for score caching

### API Endpoints

```
GET  /v1/trust/score/{email}           # Get trust score
POST /v1/trust/attestations            # Create attestation
GET  /v1/trust/attestations            # List attestations
DELETE /v1/trust/attestations/{id}     # Revoke attestation
GET  /v1/trust/path                    # Get trust path
```

### Caching Strategy

- Cache trust scores for 24 hours
- Invalidate on new attestations in the path
- Background refresh for frequently accessed scores

## Related Decisions

- ADR-002: Encryption Key Management
- ADR-003: Federation Protocol
- ADR-004: Anti-Spam Strategy

## References

- [PGP Web of Trust](https://www.openpgp.org/)
- [Google Web of Trust Paper](https://research.google/pubs/pub43904/)
- [Sybil Attack Prevention](https://www.cs.rice.edu/~dwallach/pub/sybil-tissec.pdf)
