# Design Principles

The foundational principles guiding Knowledge hApp development.

## Core Philosophy

Knowledge is built on the belief that **truth is discovered, not decreed**. No central authority should control what is considered true. Instead, truth emerges from:

- Open submission of claims
- Transparent evidence and relationships
- Economic incentives aligned with accuracy
- Community verification and review
- Time-tested belief propagation

## The Seven Design Principles

### 1. Epistemic Humility

**"All knowledge is provisional"**

Every claim in Knowledge is understood to be:
- Subject to revision with new evidence
- Positioned in epistemic space, not labeled "true" or "false"
- Connected to the context of its creation
- Limited by the knowledge of its time

**Implementation**:
- E-N-M classification shows uncertainty
- Claims can always be updated or retracted
- Contradictions are linked, not hidden
- History is preserved

### 2. Decentralized Truth

**"No one controls what is true"**

Knowledge has no:
- Central database that can be corrupted
- Administrator who can delete claims
- Authority that decides truth
- Single point of failure

**Implementation**:
- Holochain's agent-centric DHT
- Local-first with eventual consistency
- No privileged nodes
- Cryptographic integrity

### 3. Incentive Alignment

**"Truth should be profitable"**

The system is designed so that:
- Accurate claims increase author reputation
- False claims decrease reputation
- Verification is economically rewarded
- Gaming is expensive and detectable

**Implementation**:
- MATL² reputation weighting
- Verification markets with real stakes
- Credibility affects claim visibility
- Sybil-resistance through economics

### 4. Transparency by Default

**"Hidden information corrupts"**

Everything in Knowledge is:
- Publicly visible (claims, relationships, scores)
- Auditable (computation can be verified)
- Traceable (provenance is preserved)
- Explained (reasoning is documented)

**Implementation**:
- All entries on public DHT
- Score calculations are deterministic
- Evidence chains are explicit
- Moderation decisions are public

### 5. Progressive Verification

**"Trust is earned, not assumed"**

New claims start with:
- Neutral credibility position
- No assumed truth or falsehood
- Potential for verification
- Path to higher confidence

**Implementation**:
- Information value ranking
- Verification market recommendations
- Credibility increases with evidence
- Time-based maturation

### 6. Epistemic Diversity

**"Multiple ways of knowing"**

Knowledge respects that:
- Empirical facts aren't the only knowledge
- Normative claims have their place
- Foundational beliefs matter
- Different domains have different standards

**Implementation**:
- E-N-M three-dimensional classification
- Domain-specific verification standards
- Mythic dimension for foundational claims
- Cross-domain relationships

### 7. Graceful Degradation

**"The system should fail well"**

When problems occur:
- Partial functionality continues
- Data is never lost
- Recovery is possible
- Users are informed

**Implementation**:
- Local-first architecture
- Offline capability
- Conflict resolution
- Clear error states

## Technical Design Decisions

These principles inform specific technical choices:

### Why Holochain?

| Requirement | Holochain Solution |
|-------------|-------------------|
| Decentralization | Agent-centric DHT |
| No central server | Peer-to-peer network |
| Data integrity | Content-addressed hashes |
| Privacy options | Private/public entries |
| Scalability | Sharded by agent |

### Why E-N-M Classification?

| Need | E-N-M Solution |
|------|---------------|
| Beyond true/false | 3D continuous space |
| Acknowledge subjectivity | Normative dimension |
| Respect foundations | Mythic dimension |
| Clear communication | Visual representation |

### Why MATL Integration?

| Challenge | MATL Solution |
|-----------|--------------|
| Sybil attacks | Quadratic weighting |
| Reputation gaming | Multi-dimensional trust |
| Byzantine actors | 45% fault tolerance |
| Cold start | Transitive trust |

### Why Verification Markets?

| Problem | Market Solution |
|---------|----------------|
| Who decides truth? | Collective intelligence |
| Incentive to verify | Economic reward |
| Quality signals | Price as probability |
| Resolution mechanism | Oracle consensus |

## Anti-Patterns to Avoid

### What Knowledge is NOT

1. **Not a fact-checker that declares truth**
   - We provide evidence and credibility, not verdicts

2. **Not a social network with viral dynamics**
   - No engagement optimization, no addiction mechanics

3. **Not a reputation casino**
   - Stakes are meaningful, not gamified

4. **Not a censorship tool**
   - Low credibility != removed

5. **Not a single source of truth**
   - One of many knowledge sources

### Design Smells

Watch for these warning signs:

| Smell | Problem | Solution |
|-------|---------|----------|
| Central authority | Single point of failure | Distribute responsibility |
| Hidden scores | Opaque = manipulable | Make everything auditable |
| Binary verdicts | Oversimplification | Use continuous spaces |
| Viral mechanics | Attention over truth | Remove engagement metrics |
| Walled gardens | Knowledge hoarding | Open protocols |

## Evolution Process

### How Principles Guide Development

When making design decisions:

1. **State the decision** clearly
2. **List principles** that apply
3. **Evaluate options** against principles
4. **Document tradeoffs** transparently
5. **Choose** the most aligned option

### Principle Conflicts

When principles conflict:
- Epistemic Humility vs. User Clarity → Provide both uncertainty and summaries
- Decentralization vs. Performance → Accept latency for integrity
- Transparency vs. Privacy → Public claims, private identities by choice

### Updating Principles

These principles can evolve through:
1. Community proposal
2. Discussion period (30 days)
3. MATL-weighted vote
4. Supermajority approval (>75%)
5. Documentation update

## Measuring Principle Alignment

### Health Metrics

| Principle | Metric | Target |
|-----------|--------|--------|
| Epistemic Humility | % claims with uncertainty | >80% |
| Decentralized Truth | Node distribution | >1000 nodes |
| Incentive Alignment | False claim profitability | <0 |
| Transparency | Auditable operations | 100% |
| Progressive Verification | Markets per claim | >0.1 |
| Epistemic Diversity | Domain coverage | >10 domains |
| Graceful Degradation | Recovery time | <1 hour |

### Principle Violations

Report potential violations:
1. Document the concern
2. Reference violated principle
3. Propose correction
4. Submit to governance

## Related Documentation

- [Epistemic Charter](./EPISTEMIC_CHARTER.md) - Community covenant
- [Claim Moderation](./CLAIM_MODERATION.md) - Quality processes
- [Epistemic Classification](../concepts/EPISTEMIC_CLASSIFICATION.md) - E-N-M system

---

*Principles over policies, transparency over control.*
