# Mycelix Improvement Proposals (MIPs)

This directory contains formal governance proposals for the Mycelix Protocol.

## Active Proposals

### Economic (MIP-E-XXX)

| MIP | Title | Status | Requires | Summary |
|-----|-------|--------|----------|---------|
| [MIP-E-001](../architecture/THE%20ECONOMIC%20CHARTER%20(v1.0).md#section-6-validator-economics-mip-e-001) | Validator Economics Framework | RATIFICATION PENDING | None | Validator incentives, hardware requirements, staking economics |
| [MIP-E-002](./MIP-E-002_METABOLIC_BRIDGE_AMENDMENT.md) | Metabolic Bridge Amendment | DRAFT | MIP-E-001 | Circulation-based economics, PoC, HEARTH, Decay Garden |
| [MIP-E-003](./MIP-E-003_PROOF_OF_GROUNDING.md) | Proof of Grounding (PoG) | DRAFT | MIP-E-002 | Physical infrastructure binding (DePIN integration) |
| [MIP-E-004](./MIP-E-004_AGENTIC_ECONOMY_FRAMEWORK.md) | Agentic Economy Framework | DRAFT | MIP-E-002 | AI agents as economic participants (KREDIT system) |
| [MIP-E-005](./MIP-E-005_TEMPORAL_ECONOMICS.md) | Temporal Economics | DRAFT | MIP-E-001, MIP-E-002 | Patient capital, time-locked commitments, intergenerational trusts |
| [MIP-E-006](./MIP-E-006_PROOF_OF_REGENERATION.md) | Proof of Regeneration (PoR) | DRAFT | MIP-E-002, MIP-E-003 | Ecological impact verification, harm accounting |
| [MIP-E-007](./MIP-E-007_CIVILIZATIONAL_INTENTIONS.md) | Civilizational Intention Framework | DRAFT | MIP-E-001, MIP-E-002 | Goal-oriented economics, intention trees, progress oracles |
| [MIP-E-008](./MIP-E-008_AGENT_COLLECTIVES.md) | Agent Collectives | DRAFT | MIP-E-004 | Coordinated AI swarms with human oversight |

## MIP Process

### Proposal Lifecycle

1. **IDEA** - Initial concept discussion
2. **DRAFT** - Formal specification written
3. **REVIEW** - Community feedback period (30-60 days)
4. **AUDIT** - Technical review by Audit Guild
5. **VOTE** - Global DAO ratification
6. **ACCEPTED** - Ratified and queued for implementation
7. **IMPLEMENTED** - Deployed to network
8. **FINAL** - Complete and in production

### Voting Requirements

| Category | Quorum | Approval | Comment Period |
|----------|--------|----------|----------------|
| Economic | 40% | ⅔ supermajority | 30 days minimum |
| Technical | 30% | Simple majority | 21 days minimum |
| Constitutional | 50% | ⅔ supermajority | 60 days minimum |

### Who Can Propose

- Any member with CIV ≥ 0.5 can submit proposals
- Proposals require 10 endorsements to proceed to review
- Karmic Council may fast-track emergency proposals

## MIP Template

```markdown
# MIP-X-NNN: Title

**Title**: Descriptive Title
**Author**: Name (DID or handle)
**Status**: DRAFT | REVIEW | VOTE | ACCEPTED | IMPLEMENTED | FINAL
**Type**: Standards Track (Economic | Technical | Constitutional)
**Category**: Specific area affected
**Created**: YYYY-MM-DD
**Requires**: List of prerequisite MIPs
**Supersedes**: List of MIPs this replaces

---

## Abstract
Brief summary (2-3 paragraphs)

## Motivation
Why is this needed?

## Specification
Detailed technical specification

## Rationale
Why these design decisions?

## Backwards Compatibility
How does this affect existing systems?

## Security Considerations
What are the risks and mitigations?

## Reference Implementation
Link to code implementation

## Copyright
License statement
```

## Economic MIP Series Overview

The MIP-E-002 through MIP-E-008 series represents a comprehensive economic redesign spanning infrastructure, agents, time, ecology, and civilizational purpose:

### Metabolic Bridge (MIP-E-002)

Transforms Mycelix from extraction-based to circulation-based economics:

- **Proof of Contribution (PoC)**: Replace stake-weighted validation with behavioral + physical contribution
- **Progressive Fee Tiers**: CIV-based fee reductions reward long-term participation
- **HEARTH Pools**: Liquid commons governed by Local DAOs
- **Wealth Half-Life**: 2% annual "composting" of dormant value back to commons
- **Metabolic Oracle**: Autopoietic parameter adjustment within constitutional bounds

### Proof of Grounding (MIP-E-003)

Binds digital value to physical infrastructure:

- **Energy**: Renewable generation (solar, wind, hydro) with hardware attestation
- **Storage**: Decentralized storage (IPFS, Holochain DHT) with proof of replication
- **Compute**: TEE-attested processing capacity
- **Bandwidth**: Peer-verified network connectivity

Creates a value floor based on real-world utility.

### Agentic Economy (MIP-E-004)

Enables AI agents as economic participants:

- **Instrumental Actors**: AI agents with sponsor accountability
- **KREDIT**: Non-transferable, sponsor-collateralized credit
- **Constitutional Constraints**: Agents cannot vote, validate, or govern HEARTH
- **Lifecycle Management**: Creation, suspension, revocation with full audit trail

### Temporal Economics (MIP-E-005)

Aligns incentives with long-term thinking:

- **Time-Locked Commitments**: Graduated governance weight (1x to 7x) for patient capital
- **Covenant Bonds**: Purpose-bound capital that survives beyond individuals
- **Intergenerational Trusts**: Multi-generational asset stewardship
- **Patience Coefficient**: Patience rewarded throughout the protocol
- **Succession Mechanisms**: Orderly value transfer across generations

### Proof of Regeneration (MIP-E-006)

Measures ecological impact, not just capacity:

- **Five Domains**: Carbon, Biodiversity, Water, Soil, Air
- **Baseline-Delta Verification**: Before/after measurement of ecological state
- **Regeneration Multipliers**: PoG × PoR = true grounding score
- **Harm Accounting**: Penalties for degradation, remediation requirements
- **Oracle Networks**: Multi-source verification of regeneration claims

### Civilizational Intentions (MIP-E-007)

Explicit goals that give purpose to economic machinery:

- **Intention Domains**: Universal Basic Services, Ecological Regeneration, Knowledge Commons, etc.
- **Intention Trees**: Hierarchical goals from abstract to concrete
- **Progress Oracles**: Real-world measurement of intention advancement
- **Intention-Aligned Incentives**: PoC includes intention contribution
- **Civilizational Dashboard**: Public progress display embedded in every client

### Agent Collectives (MIP-E-008)

Coordinated AI swarms with human oversight:

- **Shared KREDIT Pools**: Resource pooling with sponsor-backed caps
- **Collective Constraints**: Aggregate limits prevent swarm exploitation
- **Decision Protocols**: Unanimous, majority, weighted, liquid democracy
- **Human Oversight Councils**: Required for all collectives
- **Collective Reputation**: Shared accountability mechanisms

## Implementation Status

### Phase 1: Core Economics (MIP-E-002 to MIP-E-004) ✅

| Component | SDK Module | Test Coverage | Status |
|-----------|------------|---------------|--------|
| PoC Calculation | `economics::poc` | ✅ | Ready |
| Metabolic Oracle | `economics::metabolic_oracle` | ✅ | Ready |
| HEARTH Pools | `economics::hearth` | ✅ | Ready |
| Decay Garden | `economics::decay_garden` | ✅ | Ready |
| Energy PoG | `pog::energy` | ✅ | Ready |
| Storage PoG | `pog::storage` | ✅ | Ready |
| Compute PoG | `pog::compute` | ✅ | Ready |
| Bandwidth PoG | `pog::bandwidth` | ✅ | Ready |
| KREDIT System | `agentic::kredit` | ✅ | Ready |
| Agent Constraints | `agentic::constraints` | ✅ | Ready |
| Agent Lifecycle | `agentic::lifecycle` | ✅ | Ready |

### Phase 2: Long-Term Evolution (MIP-E-005 to MIP-E-008) 🚧

| Component | SDK Module | Status |
|-----------|------------|--------|
| Time-Locked Commitments | `temporal::commitment` | Planned |
| Covenant Bonds | `temporal::covenant` | Planned |
| Intergenerational Trusts | `temporal::trust` | Planned |
| Patience Coefficient | `temporal::patience` | Planned |
| Carbon Regeneration | `por::carbon` | Planned |
| Biodiversity Regeneration | `por::biodiversity` | Planned |
| Water/Soil/Air Regeneration | `por::water`, `por::soil`, `por::air` | Planned |
| Intention Domains | `intentions::domains` | Planned |
| Intention Trees | `intentions::trees` | Planned |
| Progress Oracles | `intentions::oracles` | Planned |
| Agent Collectives | `agentic::collective` | Planned |
| Collective KREDIT | `agentic::collective_kredit` | Planned |
| Oversight Councils | `agentic::oversight` | Planned |

See `/mycelix-workspace/sdk/src/` for complete implementations.

## Related Documents

- [THE ECONOMIC CHARTER (v1.0)](../architecture/THE%20ECONOMIC%20CHARTER%20(v1.0).md)
- [THE GOVERNANCE CHARTER (v1.0)](../architecture/THE%20GOVERNANCE%20CHARTER%20(v1.0).md)
- [THE MYCELIX SPORE CONSTITUTION (v0.24)](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)

## Contributing

1. Discuss your idea in the community forum
2. Draft proposal using the template above
3. Submit to this directory via pull request
4. Gather 10 endorsements
5. Address review feedback
6. Proceed to Global DAO vote

## Contact

- **Governance Working Group**: governance@mycelix.net
- **Technical Questions**: dev@luminousdynamics.org
- **Audit Guild**: audit@mycelix.net

---

*MIP process based on Ethereum EIP and Bitcoin BIP standards, adapted for Mycelix's polycentric governance model.*
