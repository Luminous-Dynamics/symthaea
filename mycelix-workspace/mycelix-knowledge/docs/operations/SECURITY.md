# Security Considerations

Security practices and considerations for Mycelix Knowledge.

## Threat Model

### Adversaries

| Actor | Motivation | Capabilities |
|-------|------------|--------------|
| **Misinformation Agent** | Spread false claims | Multiple accounts, coordinated campaigns |
| **Reputation Farmer** | Gain illegitimate trust | Sock puppets, collusion |
| **Market Manipulator** | Profit from false resolutions | Capital, insider knowledge |
| **Data Poisoner** | Corrupt the knowledge graph | Volume attacks, subtle errors |
| **Denial of Service** | Disrupt network | Spam, resource exhaustion |

### Attack Surfaces

1. **Claim Submission** - Fraudulent or misleading claims
2. **Oracle Voting** - Collusion or manipulation
3. **Belief Propagation** - Cascade attacks
4. **API Access** - Unauthorized queries or modifications
5. **Network Layer** - DHT attacks, eclipse attacks

## Security Measures

### Byzantine Fault Tolerance

The system tolerates up to 45% malicious actors through MATL-weighted voting:

```
Honest Outcome = Σ(honest_votes × matl²) > Σ(malicious_votes × matl²)

With quadratic weighting:
- High-MATL oracles have proportionally more influence
- Attackers need >45% of weighted stake
- New accounts have minimal impact
```

### Sybil Resistance

Multiple mechanisms prevent Sybil attacks:

1. **MATL Requirement** - Participation requires minimum trust score
2. **Reputation Building** - Trust must be earned over time
3. **Stake Requirements** - Economic cost to participate
4. **Graph Analysis** - Detect coordination patterns

### Claim Validation

```rust
// Validation performed on all claims
fn validate_claim(claim: &Claim) -> ExternResult<ValidateCallbackResult> {
    // 1. Basic field validation
    if claim.content.is_empty() {
        return invalid("Content cannot be empty");
    }

    // 2. Epistemic bounds check
    if !validate_epistemic(&claim.classification) {
        return invalid("Invalid epistemic position");
    }

    // 3. Evidence consistency
    if claim.classification.e > 0.7 && claim.evidence.is_empty() {
        return invalid("High-E claims require evidence");
    }

    // 4. Rate limiting (by author)
    if exceeds_rate_limit(claim.author) {
        return invalid("Rate limit exceeded");
    }

    Ok(ValidateCallbackResult::Valid)
}
```

### Oracle Security

```typescript
// Oracle participation requirements
interface OracleRequirements {
  minMatlScore: 0.3;           // Minimum trust level
  minReputationAge: 30;        // Days of reputation history
  minAccuracy: 0.6;            // Historical accuracy
  maxStakePerMarket: 0.1;      // Max 10% of total stake
}

// Vote validation
function validateOracleVote(vote: OracleVote): boolean {
  // Check MATL eligibility
  if (vote.matlWeight < requirements.minMatlScore) return false;

  // Check evidence requirement
  if (vote.confidence > 0.8 && vote.evidence.length === 0) return false;

  // Check stake limits
  if (vote.reputationStake > 0.2) return false;  // Max 20%

  return true;
}
```

## Data Integrity

### Cryptographic Verification

All entries are cryptographically signed:

```
Entry Hash = blake3(serialize(entry))
Signature = Ed25519.sign(privkey, Entry Hash)
Action = { hash, signature, author, timestamp }
```

### Source Chain Integrity

Each agent maintains a tamper-evident source chain:

```
Chain: [Genesis] → [Action_1] → [Action_2] → ... → [Action_n]
                     ↓              ↓
                prev_hash       prev_hash

Tampering detection:
- Any modification breaks hash chain
- Peers validate chain integrity
- Fork detection algorithms
```

### DHT Validation

All DHT operations are validated:

```rust
// Validation rules enforced by all peers
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry { entry, .. } => validate_entry(entry),
        Op::RegisterCreateLink { create_link, .. } => validate_link(create_link),
        Op::RegisterDeleteLink { delete_link, .. } => validate_delete(delete_link),
    }
}
```

## Privacy

### Data Visibility

| Data Type | Visibility | Protection |
|-----------|------------|------------|
| Public Claims | Network-wide | Content hash only |
| Author Identity | Pseudonymous | Agent pub key |
| Oracle Votes | Transparent | After resolution |
| Query Patterns | Private | No logging |

### Sensitive Claims

For sensitive domains:

```typescript
// Claims can be marked for access control
const sensitiveClaim = await knowledge.claims.createClaim({
  content: "Confidential organizational data...",
  classification: { ... },
  domain: "internal",
  accessControl: {
    visibility: "restricted",
    allowedGroups: ["organization-members"]
  }
});
```

## Rate Limiting

### Claim Submission

| Tier | Claims/Hour | Evidence/Claim |
|------|-------------|----------------|
| New User | 5 | 10 |
| Established | 20 | 50 |
| Trusted | 100 | 200 |
| Unlimited | No limit | No limit |

### API Queries

```typescript
// Query rate limiting
const rateLimits = {
  search: 100,        // per minute
  factCheck: 20,      // per minute
  batchFactCheck: 5,  // per minute
};
```

## Incident Response

### Detection

1. **Automated Monitoring**
   - Unusual claim volumes
   - Coordinated voting patterns
   - Sudden reputation changes
   - Network anomalies

2. **Community Reporting**
   - Flag suspicious claims
   - Report oracle misconduct
   - Identify coordination

### Response Procedures

1. **Immediate**
   - Rate limit affected accounts
   - Freeze disputed markets
   - Alert oracle reviewers

2. **Investigation**
   - Graph analysis
   - Pattern identification
   - Evidence collection

3. **Resolution**
   - Reputation adjustments
   - Claim corrections
   - Market reversals (if warranted)

4. **Prevention**
   - Update validation rules
   - Improve detection
   - Document learnings

## Best Practices

### For Users

- ✅ Verify sources before submitting claims
- ✅ Use strong, unique credentials
- ✅ Report suspicious activity
- ❌ Don't share private keys
- ❌ Don't submit unverified information
- ❌ Don't participate in coordinated campaigns

### For Developers

- ✅ Validate all inputs
- ✅ Use prepared queries
- ✅ Implement rate limiting
- ✅ Log security events
- ❌ Don't trust client-side validation
- ❌ Don't expose internal errors
- ❌ Don't store secrets in code

### For Operators

- ✅ Monitor network health
- ✅ Keep dependencies updated
- ✅ Review access logs
- ✅ Test incident procedures
- ❌ Don't ignore anomalies
- ❌ Don't delay security updates

## Reporting Vulnerabilities

Report security issues to: security@luminousdynamics.org

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We follow responsible disclosure:
1. Acknowledge within 24 hours
2. Initial assessment within 72 hours
3. Regular updates during fix
4. Credit in disclosure

---

*Security is a shared responsibility.*
