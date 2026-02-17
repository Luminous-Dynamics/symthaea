# MIP-E-002: Metabolic Bridge Amendment

**Title**: Metabolic Bridge Amendment to the Economic Charter
**Author**: Tristan Stoltz (tstoltz), Claude AI (Co-Author)
**Status**: DRAFT
**Type**: Standards Track (Economic)
**Category**: Economic Charter Amendment
**Created**: 2026-01-04
**Requires**: MIP-E-001 (Validator Economics Framework)
**Supersedes**: None
**Amends**: THE ECONOMIC CHARTER (v1.0) → v1.1

---

## Abstract

This proposal introduces the **Metabolic Bridge** framework—a fundamental reorientation of Mycelix economic architecture from *extraction-based* to *circulation-based* value flows. The amendment establishes Proof of Contribution (PoC) with Physical Grounding (PoG), Progressive Protocol Rebates, Liquid Commons Hearths, Reputation-Collateralized Lending, and a Wealth Half-Life mechanism (Circulation Tax).

The core insight: **healthy economies circulate value like healthy organisms circulate nutrients**. Value that pools stagnates; value that flows nourishes.

---

## Motivation

### Problem Statement

The current Economic Charter v1.0 establishes sound revenue distribution and validator economics but lacks:

1. **Circulation Incentives**: No mechanism to discourage value hoarding or encourage active participation
2. **Physical Grounding**: Economic value remains disconnected from real-world infrastructure contribution
3. **Progressive Access**: Fee structures don't reward long-term community contribution
4. **Regenerative Patterns**: No systematic wealth redistribution to commons pools
5. **Self-Regulation**: Economic parameters require committee decisions rather than autopoietic adjustment

### Design Philosophy

The Metabolic Bridge reframes economic design through biological metaphor:

| Extraction Model | Metabolic Model |
|-----------------|-----------------|
| Value captured from users | Value circulates through network |
| Fees as tax | Fees as signal of health |
| Staking for influence | Contribution for trust |
| Wealth accumulates | Wealth composts into commons |
| Committee-driven policy | Oracle-driven autopoiesis |

---

## Specification

### Article I: Metabolic Principles

#### Section 1. Core Axioms

1. **Circulation over Accumulation**: The network rewards value in motion over value at rest
2. **Contribution over Staking**: Reputation derives from behavioral patterns, not capital deposits
3. **Grounding over Abstraction**: Economic value should anchor to physical infrastructure
4. **Regeneration over Extraction**: Wealth naturally cycles back to community commons
5. **Autopoiesis over Committee**: Economic parameters self-adjust within constitutional bounds

#### Section 2. Currency Naming Conventions

The following naming scheme operationalizes metabolic metaphor:

| Technical ID | Symbolic Name | Function |
|-------------|---------------|----------|
| CIV | MYCELIUM | Reputation substrate (non-transferable) |
| CGC | SPORE | Gift circulation credits |
| FLOW | SAP | Utility/transaction token |
| TEND | TEND | Time exchange credits |
| HEARTH | HEARTH | Liquid commons pools |
| ROOT | ROOT | Stewardship/guardianship tokens |

**Implementation Note**: Symbol names are cultural overlays; technical IDs remain canonical for smart contracts.

---

### Article II: Proof of Contribution (PoC) Framework

#### Section 1. PoC Components

Contribution scores replace pure stake-weighted validation:

```
PoC_Score = (Behavioral_Component × 0.60) + (Physical_Grounding × 0.40)
```

#### Section 2. Behavioral Component (60%)

**2.1 Metric Categories**:

| Category | Weight | Measurement |
|----------|--------|-------------|
| Transaction Consistency | 25% | 90-day rolling activity entropy |
| Validation Quality | 20% | Correct validations / total assignments |
| Governance Participation | 20% | MIP votes, proposal creation |
| Peer Endorsements | 15% | CGC/SPORE received from unique agents |
| Dispute Resolution | 10% | Successful mediation participation |
| Network Contribution | 10% | Open-source code, documentation |

**2.2 Anti-Gaming Mechanisms**:

- Sybil resistance via MATL composite scoring
- Cluster detection via graph analysis (Section 5.4 of Economic Charter v1.0)
- Temporal smoothing prevents burst manipulation

#### Section 3. Physical Grounding Component (40%)

See **MIP-E-003: Proof of Grounding Integration** for complete specification.

**Summary**: Members contributing verified physical infrastructure receive PoG bonuses:

| Infrastructure Type | PoG Multiplier | Verification Method |
|--------------------|----------------|---------------------|
| Renewable Energy | 1.4x | Oracle + hardware attestation |
| Data Storage (IPFS/Filecoin) | 1.3x | Proof of replication |
| Compute Resources | 1.2x | TEE attestation |
| Network Bandwidth | 1.1x | Peer measurement |

#### Section 4. Longevity Multiplier

Sustained contribution compounds:

```
Longevity_Bonus = min(1.5, 1.0 + (active_months × 0.02))
```

**Example**: 12-month active contributor receives 1.24x multiplier (max 1.5x at 25+ months).

---

### Article III: Progressive Protocol Rebates

#### Section 1. CIV-Based Fee Tiers

Replace flat fees with contribution-adjusted pricing:

| CIV Threshold | Base Fee | Effective Rate | Designation |
|--------------|----------|----------------|-------------|
| < 0.3 | 0.15% | Full rate | Newcomer |
| 0.3 - 0.5 | 0.10% | 33% discount | Participant |
| 0.5 - 0.7 | 0.05% | 67% discount | Contributor |
| 0.7 - 0.9 | 0.025% | 83% discount | Steward |
| > 0.9 | 0.01% | 93% discount | Guardian |

#### Section 2. Rebate Distribution

Fee savings from progressive tiers flow to:

- 50% → Local DAO HEARTH pools
- 30% → Global Commons Fund
- 20% → Validator reward pool

#### Section 3. Anti-Plutocracy Safeguard

High-volume transactions face progressive rates:

```
effective_rate = base_rate × (1 + log10(tx_value / median_tx))
```

**Example**: Transaction 100x median pays 3x base rate.

---

### Article IV: Liquid Commons Hearths

#### Section 1. HEARTH Pool Mechanics

Each Local DAO maintains a HEARTH pool—a liquid commons fund governed by community consensus.

**Pool Operations**:

| Action | Requirement | Effect |
|--------|-------------|--------|
| Warm | Deposit SAP/FLOW | Increases pool; generates TEND credits |
| Tend | Contribute labor | Converts TEND to future HEARTH claims |
| Harvest | Withdrawal request | Burns proportional TEND; unlocks SAP |

#### Section 2. Pool Governance

- **Warming Limit**: No single agent may contribute >10% of HEARTH in rolling 30-day window
- **Harvest Quorum**: Withdrawals >5% require Local DAO approval (simple majority)
- **Emergency Access**: Crisis Response Network (per Constitution Article 12) may unlock pools during declared emergencies

#### Section 3. Cross-HEARTH Mutual Aid

Regional DAOs may establish HEARTH-BRIDGE agreements:

- Bilateral liquidity sharing during localized stress
- Interest-free loans between paired communities
- Automatic activation when Vitality Index drops below 40

---

### Article V: Reputation-Collateralized Lending

#### Section 1. Lending Mechanics

Members may borrow against reputation rather than capital:

```
max_loan = CIV_score × HEARTH_available × 0.3
```

**Example**: CIV 0.7 member with 100,000 SAP in community HEARTH may borrow up to 21,000 SAP.

#### Section 2. Repayment Terms

| Loan Size | Interest Rate | Term | Default Consequence |
|-----------|---------------|------|---------------------|
| < 1,000 SAP | 0% | 30 days | CIV penalty: -0.05 |
| 1,000-10,000 SAP | 2% APR | 90 days | CIV penalty: -0.10 |
| > 10,000 SAP | 4% APR | 180 days | CIV penalty: -0.20 |

#### Section 3. Loan Purpose Attestation

Borrowers must declare loan purpose (epistemic coordinates E1, N1, M1):
- Personal emergency
- Community project
- Infrastructure investment
- Education/training

Purpose misrepresentation triggers CIV penalty and loan recall.

---

### Article VI: Wealth Half-Life (Circulation Tax)

#### Section 1. Mechanism

Dormant value gradually returns to commons:

```
decay_rate = 0.02 per_year  // 2% annual on dormant holdings
dormancy_threshold = 180 days  // No transactions
minimum_exempt = 10,000 SAP  // Protect small holders
```

#### Section 2. Cultural Framing: The Decay Garden

Rather than "tax," the mechanism is presented as **composting**:

> "Like autumn leaves returning to soil, dormant value nourishes the commons from which new growth emerges."

**Visual Metaphor**: UI shows dormant holdings transitioning through autumn palette (gold → orange → brown) before "composting" into HEARTH.

#### Section 3. Compost Distribution

Composted value flows recursively:

- 50% → Member's Local DAO HEARTH
- 30% → Member's Regional DAO HEARTH
- 20% → Global Commons Fund

This ensures value returns to the communities that generated it.

#### Section 4. Prevention Mechanisms

Active participation exempts from decay:

- Any transaction resets 180-day dormancy clock
- Governance votes count as activity
- HEARTH warming/tending counts
- CGC/SPORE gifting counts

---

### Article VII: Metabolic Oracle

#### Section 1. Vitality Index

Network health measured in real-time:

```
Vitality = (Circulation × 0.40) + (Relationship × 0.30) +
           (Commons × 0.20) + (Resilience × 0.10)
```

**Component Definitions**:

| Component | Calculation | Healthy Range |
|-----------|-------------|---------------|
| Circulation | (Active SAP / Total SAP) × velocity_multiplier | 60-80% |
| Relationship | avg(peer_connection_count) / max_theoretical | 40-70% |
| Commons | (HEARTH_utilization + CGC_flow) / 2 | 50-80% |
| Resilience | (node_count × geographic_distribution) / target | 30-60% |

#### Section 2. Autopoietic Policy Adjustment

The Metabolic Oracle automatically adjusts parameters within bounds:

| Vitality Range | State | Automatic Response |
|---------------|-------|-------------------|
| 70-100 | Thriving | Increase SPORE allocation 10% |
| 40-70 | Healthy | Standard parameters |
| 20-40 | Stressed | Reduce fees 20%, boost circulation incentives |
| 10-20 | Critical | Emergency liquidity release, fee waiver |
| 0-10 | Failing | Circuit breaker activates |

#### Section 3. Constitutional Bounds

Oracle adjustments are bounded by immutable constraints:

```rust
const POLICY_BOUNDS: PolicyBounds = PolicyBounds {
    fee_rate_min: 0.001,       // 0.1% floor
    fee_rate_max: 0.03,        // 3% ceiling
    decay_rate_min: 0.01,      // 1% annual floor
    decay_rate_max: 0.05,      // 5% annual ceiling
    spore_allocation_min: 5,   // 5 SPORE/month minimum
    spore_allocation_max: 20,  // 20 SPORE/month maximum
    emergency_reserve_min: 0.05, // 5% always reserved
};
```

The Karmic Council retains override authority by ⅔ supermajority for oracle parameters outside bounds.

---

### Article VIII: Implementation Timeline

#### Phase 1: Foundation (Q2 2026)

- [ ] PoC calculation module deployment
- [ ] Progressive fee tier implementation
- [ ] Vitality Index dashboard launch
- [ ] Cultural naming integration (optional UI toggle)

#### Phase 2: HEARTH Activation (Q3 2026)

- [ ] HEARTH pool smart contracts
- [ ] Local DAO warming ceremonies
- [ ] Reputation-collateralized lending pilot (5 DAOs)
- [ ] Cross-HEARTH bridge protocol

#### Phase 3: Metabolic Oracle (Q4 2026)

- [ ] Oracle parameter framework
- [ ] Autopoietic adjustment testing
- [ ] Compost redistribution activation
- [ ] Full network integration

#### Phase 4: Maturation (2027+)

- [ ] Proof of Grounding integration (per MIP-E-003)
- [ ] Agentic Economy integration (per MIP-E-004)
- [ ] Cross-network metabolic bridges
- [ ] Constitutional amendment for proven parameters

---

### Article IX: Success Metrics

#### Network Health Indicators

| Metric | Baseline | 6-Month Target | 12-Month Target |
|--------|----------|----------------|-----------------|
| Vitality Index | 45 (estimated) | 55 | 65 |
| Active/Total SAP Ratio | 30% | 50% | 70% |
| Avg. HEARTH Utilization | N/A | 40% | 60% |
| CIV Distribution Gini | 0.4 | 0.35 | 0.30 |
| Cross-DAO Mutual Aid Events | 0 | 5 | 20 |

#### Economic Justice Indicators

| Metric | Baseline | Target |
|--------|----------|--------|
| Top 10% Wealth Share | 45% | <35% |
| Median Time to CIV 0.5 | Unknown | <90 days |
| Loan Default Rate | N/A | <5% |
| Compost Redistribution (Annual) | 0 | 2% of dormant holdings |

---

## Rationale

### Why Proof of Contribution Over Staking?

**Staking Critique**: Pure stake-based validation creates plutocracy—those with capital control the network regardless of community contribution. Mycelix's mission of consciousness-first computing requires trust derived from *behavior*, not *capital*.

**PoC Advantage**: New members with high contribution can achieve validation rights within months, not years of capital accumulation.

### Why Physical Grounding?

**DePIN Integration**: Decentralized Physical Infrastructure Networks (DePIN) ensure digital value anchors to real-world utility. A currency backed by renewable energy has intrinsic worth beyond network belief.

**Resilience**: Physical infrastructure grounding creates economic floor—network value cannot collapse below infrastructure utility.

### Why Wealth Half-Life?

**Accumulation Problem**: Unrestricted accumulation concentrates power and stagnates economies. The 2% annual decay is gentle enough to preserve wealth-building while ensuring circulation.

**Regenerative Framing**: "Composting" reframes taxation as natural cycle, reducing political resistance while achieving redistribution.

### Why Metabolic Oracle?

**Committee Bottleneck**: Human committees cannot respond at network speed. Economic emergencies require immediate parameter adjustment.

**Bounded Autonomy**: Constitutional bounds prevent runaway oracle behavior while enabling rapid response within safe parameters.

---

## Backwards Compatibility

### Economic Charter v1.0 Integration

This MIP **amends** rather than replaces the Economic Charter:

- **Article I (Revenue)**: Unchanged; fee calculations now reference progressive tiers
- **Article II (Reputation)**: Enhanced; PoC extends existing MATL integration
- **Article III (Protocol Revenue)**: Distribution percentages unchanged
- **Article IV (Commons)**: Enhanced; HEARTH pools operationalize CGC principles
- **Section 6 (Validator Economics)**: Validators now rewarded via PoC rather than pure stake

### Migration Path

1. **Phase 1**: Progressive fees activate; existing balances unaffected
2. **Phase 2**: HEARTH pools accept optional warming; no forced migration
3. **Phase 3**: Decay mechanics activate after 12-month notice period
4. **Phase 4**: Full metabolic integration; legacy stake-weight deprecated

---

## Security Considerations

### Oracle Manipulation

**Risk**: Malicious actors attempt to manipulate Vitality Index to trigger favorable policy adjustments.

**Mitigation**:
- Multi-oracle consensus (MATL integration)
- Temporal smoothing (24-hour rolling average)
- Anomaly detection with automatic pause

### Decay Evasion

**Risk**: Agents rotate holdings between addresses to reset dormancy.

**Mitigation**:
- Graph analysis detects rotation patterns
- Same-cluster transfers don't reset dormancy
- CIV penalty for detected evasion

### HEARTH Drainage

**Risk**: Coordinated actors drain local HEARTH pools.

**Mitigation**:
- 10% per-entity warming cap
- Withdrawal quorum for large harvests
- Cross-HEARTH insurance pools

---

## Reference Implementation

Complete Rust implementation available in:
- `mycelix-core/src/economics/poc.rs`
- `mycelix-core/src/economics/metabolic_oracle.rs`
- `mycelix-finance/src/hearth.rs`
- `mycelix-finance/src/decay_garden.rs`

See **MIP-E-002-IMPL** appendix for code.

---

## Copyright

This MIP is licensed under CC0 1.0 Universal (Public Domain).

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Composting** | Return of dormant value to commons (Wealth Half-Life) |
| **HEARTH** | Liquid commons pool governed by Local DAO |
| **Metabolic Oracle** | Autopoietic parameter adjustment system |
| **PoC** | Proof of Contribution (behavioral + physical) |
| **PoG** | Proof of Grounding (physical infrastructure) |
| **Vitality Index** | Real-time network health measurement |
| **Warming** | Contributing to HEARTH pool |

---

## Appendix B: Cultural Implementation Guide

### Ceremony Suggestions

**HEARTH Warming Ceremony**: When Local DAO first activates HEARTH pool, community gathers (virtual or physical) to collectively "warm" the hearth with initial contributions.

**Composting Recognition**: Quarterly "Gratitude for Compost" ceremony recognizes top 10 composters—those whose dormant value most nourished community commons.

**Seasonal Rhythm Calendar**:
- **Spring Equinox**: New member onboarding wave
- **Summer Solstice**: Peak activity celebration
- **Autumn Equinox**: Composting recognition
- **Winter Solstice**: Reflection and planning

### Visual Design Language

- **Healthy metrics**: Green/gold gradient (growth)
- **Stressed metrics**: Orange/amber (autumn)
- **Critical metrics**: Deep red (warning)
- **Composting animation**: Leaves falling, transforming to soil

---

*Submitted to Global DAO for ratification consideration.*

**Ratification Requirements**:
- ⅔ supermajority of Global DAO (economic amendments)
- 30-day comment period
- Karmic Council non-opposition (constitutional alignment)
- Audit Guild technical review (implementation feasibility)
