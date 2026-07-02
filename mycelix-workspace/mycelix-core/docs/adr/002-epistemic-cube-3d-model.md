# ADR-002: 3D Epistemic Cube (E/N/M Axes)

**Status**: ✅ Accepted
**Date**: November 2025
**Authors**: Mycelix Core Team
**Supersedes**: N/A (v1.0 used 1D Layered Epistemic Model)
**Superseded By**: N/A

---

## Context

**The Problem with 1D Classification**

The original Layered Epistemic Model (LEM v1.0) used a single dimension (L0-L4) to classify claims based on "epistemic strength." This approach had critical limitations:

1. **Conflated Independent Dimensions**: A claim's **factual verifiability** (is it true?), **normative authority** (who agrees it's binding?), and **materiality** (how long does it matter?) are independent properties, but were forced into a single axis.

2. **Could Not Classify Key Cases**:
   - Constitutional principles: High authority (binding on all) but unverifiable beliefs
   - Scientific evolution: Newton vs. Einstein both had high empirical validity but different consensus
   - Ephemeral vs. permanent data: A "like" vs. a copyright claim have different lifespans regardless of verification method

3. **State Management Confusion**: No clear way to determine which claims should be pruned from the DKG vs. preserved forever.

**Why Now?**

With the v6.0 architecture planning and Constitution v0.25 drafting, we needed a truth framework that could:
- Support scientific evolution (superseding claims)
- Enable intelligent state pruning (M-Axis)
- Separate factual disputes (E-Axis) from governance disputes (N-Axis)
- Classify everything from ephemeral messages to mathematical proofs

## Decision

**We replaced the 1D Layered Epistemic Model with a 3-dimensional "Epistemic Cube" framework.**

All claims are now classified along **three independent axes**:

### 1. E-Axis (Empirical Verifiability)
**Question**: How do we know this claim is true?
- **E0**: Null (unverifiable belief)
- **E1**: Testimonial (personal attestation)
- **E2**: Privately Verifiable (Audit Guild)
- **E3**: Cryptographically Proven (ZKP)
- **E4**: Publicly Reproducible (open data/code)

### 2. N-Axis (Normative Authority)
**Question**: Who agrees this claim is binding?
- **N0**: Personal (self only)
- **N1**: Communal (local DAO)
- **N2**: Network (global consensus)
- **N3**: Axiomatic (constitutional/mathematical)

### 3. M-Axis (Materiality)
**Question**: How long does this claim matter?
- **M0**: Ephemeral (discard immediately)
- **M1**: Temporal (prune after state change)
- **M2**: Persistent (archive after time)
- **M3**: Foundational (preserve forever)

**Example Classifications**:
- A "like": (E0, N0, M0) - Unverifiable, personal, ephemeral
- Passed MIP: (E0, N2, M3) - Unverifiable belief, network consensus, permanent
- Mathematics: (E4, N3, M3) - Reproducible, axiomatic, permanent

## Consequences

### Positive Outcomes ✅

1. **Can Classify Everything**: From ephemeral UI events to constitutional principles, every claim type has a clear coordinate.

2. **Enables Scientific Evolution**: Claims can SUPERCEDE others (Einstein supersedes Newton) with clear graph relationships in the DKG.

3. **Intelligent State Pruning**: M-Axis determines storage strategy (M0=discard, M1=prune, M2=archive, M3=preserve).

4. **Separate Dispute Paths**:
   - E-Axis disputes (fraud, counterfeit) → Member Redress Council
   - N-Axis disputes (constitutional challenges) → Constitutional process
   - M-Axis disputes (archival policy) → Knowledge Council

5. **Clear Governance Integration**: N-Axis maps directly to governance tiers (N1=Local DAO, N2=Global DAO, N3=Constitution).

6. **Backward Compatible Schema**: Upgraded Epistemic Claim Schema v2.0 adds `epistemic_tier_e`, `epistemic_tier_n`, `epistemic_tier_m` without breaking existing fields.

### Negative Outcomes / Trade-offs ⚠️

1. **Increased Complexity**: Requires understanding 3 axes instead of 1. Steeper learning curve for new contributors.

2. **Migration Effort**: All existing claims must be reclassified from 1D (L0-L4) to 3D (E, N, M). Automated but requires validation.

3. **UI Challenge**: Harder to visualize 3 dimensions than 1 in user interfaces. Requires thoughtful UX design.

4. **Schema Upgrade**: Breaking change to Epistemic Claim Schema (v1.1 → v2.0). All clients must upgrade.

### Neutral Impacts 🔄

1. **Documentation Rewrite**: Extensive updates to Epistemic Charter, Architecture docs, and tutorials.

2. **Training Required**: Community education needed to understand and use the 3D model.

3. **Terminology Shift**: "Epistemic tier" becomes "E-Axis coordinate" + "N-Axis coordinate" + "M-Axis coordinate".

## Alternatives Considered

### Alternative 1: Enhanced 1D Model (L0-L9)
**Description**: Keep 1D approach but add more tiers to capture nuance.
**Pros**:
- Simpler conceptual model
- No schema breaking changes
- Easier to visualize

**Cons**:
- Still conflates independent dimensions
- Would need L0-L9 × 3 = 27+ tiers to capture all cases
- Does not solve state pruning or scientific evolution
- **Decision**: Rejected because it doesn't solve the core problem.

### Alternative 2: 2D Model (Empirical + Authority)
**Description**: Use 2 axes (E-Axis and N-Axis) but ignore Materiality.
**Pros**:
- Simpler than 3D (2 axes instead of 3)
- Solves factual vs. governance separation
- Enables scientific evolution

**Cons**:
- No solution for state pruning
- DKG would grow unbounded with ephemeral data
- Still missing a critical dimension
- **Decision**: Rejected because M-Axis is essential for scalability.

### Alternative 3: Do Nothing (Keep 1D Model)
**Description**: Continue using LEM v1.0 with workarounds for edge cases.
**Pros**:
- No migration cost
- No learning curve
- No schema changes

**Cons**:
- Cannot classify constitutional principles properly
- Cannot support scientific evolution
- No state pruning strategy
- Growing technical debt
- **Decision**: Rejected because limitations block v6.0 features.

## Implementation Notes

**Migration Path**:

1. **Phase 1: Schema Upgrade** (Week 1)
   - Deploy Epistemic Claim Schema v2.0
   - Update DKG validation rules
   - Create migration scripts for existing claims

2. **Phase 2: Charter Publication** (Week 2)
   - Publish Epistemic Charter v2.0
   - Update all cross-references
   - Announce to community

3. **Phase 3: Client Upgrades** (Weeks 3-4)
   - Update all official clients
   - Provide migration tools
   - Grace period for 3rd-party clients

4. **Phase 4: Education** (Ongoing)
   - Create tutorials and visualizations
   - Host community workshops
   - Update developer documentation

**Technical Requirements**:
- DKG nodes must support 3-axis queries
- Clients must validate all 3 axes
- UI components for 3D visualization

## Success Metrics

**Quantitative**:
- 100% of constitutional claims properly classified as (E0, N3, M3) by Week 4
- 90% of clients upgraded to v2.0 schema by Week 8
- DKG query performance remains <100ms with 3-axis filtering
- State pruning reduces storage by 40% for M0/M1 claims

**Qualitative**:
- Community feedback: "This finally makes sense" vs. "Too complicated"
- Developer onboarding time for epistemic concepts
- Reduction in classification disputes

**Timeline**: Evaluate after 3 months (February 2026)

## References

**Related Documents**:
- [Epistemic Charter v2.0](../architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md)
- [Epistemic Charter v1.0](../architecture/THE%20EPISTEMIC%20CHARTER%20(v1.0).md) (superseded)
- [Constitution v0.24](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
- [DKG Architecture](../../0TML/docs/06-architecture/README.md)

**Research**:
- "Epistemic Logic and Distributed Knowledge" (Fagin et al., 1995)
- "The Social Construction of Reality" (Berger & Luckmann, 1966)
- "Information Architecture for the World Wide Web" (Rosenfeld & Morville, 2006)

**Discussions**:
- Original proposal: Internal design doc (October 2025)
- Community feedback: (coming in Constitution v0.25 discussion)

---

## Revision History

| Date | Author | Change |
|------|--------|--------|
| 2025-11-10 | Mycelix Core Team | Initial draft |
| 2025-11-10 | Mycelix Core Team | Accepted |

---

📍 **Navigation**: [← ADR Index](./README.md) | [↑ Docs Home](../README.md) | [Next ADR →](./003-matl-composite-trust-scoring.md)
