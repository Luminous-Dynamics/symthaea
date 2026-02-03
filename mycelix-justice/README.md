# Mycelix Justice hApp

**Decentralized Dispute Resolution for the Mycelix Civilizational OS**

Part of Phase 3 - Governance Pillar.

## Overview

Mycelix Justice provides a comprehensive dispute resolution system combining traditional adjudication with restorative justice practices. The system emphasizes healing and restoration over punishment, while providing binding enforcement when needed.

## Three-Tier Justice System

### Tier 1: Mediation
- **Voluntary participation** by all parties
- **Neutral mediator** facilitates dialogue
- **14-day resolution window**
- **Non-binding** unless settlement reached
- **Focus**: Understanding and agreement

### Tier 2: Arbitration
- **5 jurors** selected by MATL-weighted random
- **Evidence presentation** with chain of custody
- **Deliberation period** (7 days default)
- **Binding judgment** with reasoned decision
- **Focus**: Fair adjudication

### Tier 3: Appeal
- **30-day appeal window** from decision
- **7 jurors** (different from original panel)
- **Limited to specific grounds** (procedural, new evidence, bias)
- **Maximum 2 appeals**
- **Focus**: Correction of errors

## Restorative Justice Track

An alternative to punitive outcomes:
- **Restorative Circles**: Facilitated dialogue between parties
- **Community Involvement**: Affected community participates
- **Healing Focus**: Understanding harm and making amends
- **Voluntary**: All parties must consent
- **Agreements**: Collaborative action plans

## Architecture

```
justice/
├── dna/
│   ├── integrity/           # Entry & link types
│   │   └── src/lib.rs       # Case, Evidence, Arbitration, etc.
│   └── coordinator/
│       ├── cases/           # Case management
│       ├── arbitration/     # Panel & decision
│       ├── restorative/     # Restorative circles
│       └── enforcement/     # Remedy execution
├── happ.yaml
└── Cargo.toml
```

## Entry Types

### Case
A dispute with full lifecycle:
```rust
pub struct Case {
    pub title: String,
    pub description: String,
    pub case_type: CaseType,  // Contract, Conduct, Property, etc.
    pub complainant: String,  // DID
    pub respondent: String,   // DID
    pub phase: CasePhase,     // Filed → Mediation → Arbitration → Closed
    pub severity: CaseSeverity,
    pub context: CaseContext, // Originating hApp, community
}
```

### Evidence
Tamper-proof evidence with custody chain:
```rust
pub struct Evidence {
    pub case_id: String,
    pub evidence_type: EvidenceType,  // Document, Transaction, Testimony
    pub content: EvidenceContent,     // Hash, reference, encryption
    pub custody: Vec<CustodyEvent>,   // Full chain of custody
    pub verification: EvidenceVerification,
    pub sealed: bool,
}
```

### Arbitration & Decision
Panel formation and judgment:
```rust
pub struct Arbitration {
    pub case_id: String,
    pub arbitrators: Vec<Arbitrator>,
    pub selection_method: ArbitratorSelection,
    pub status: ArbitrationStatus,
}

pub struct Decision {
    pub outcome: DecisionOutcome,  // ForComplainant, ForRespondent, Split
    pub reasoning: String,
    pub remedies: Vec<Remedy>,
    pub votes: Vec<ArbitratorVote>,
    pub dissents: Vec<DissentingOpinion>,
}
```

### RestorativeCircle
Alternative healing process:
```rust
pub struct RestorativeCircle {
    pub case_id: String,
    pub facilitator: String,  // DID
    pub participants: Vec<CircleParticipant>,
    pub sessions: Vec<CircleSession>,
    pub agreements: Vec<String>,
    pub status: CircleStatus,
}
```

## Juror Selection

```
selection_probability = MATL_score × expertise_factor × availability

Where:
- MATL_score: Multi-dimensional trust (minimum 0.4 required)
- expertise_factor: Domain knowledge rating (1.0-2.0)
- availability: Juror's declared availability (0.0-1.0)
```

## Remedy Types

| Remedy | Description |
|--------|-------------|
| Compensation | Monetary payment |
| Restitution | Return of property |
| SpecificPerformance | Complete an obligation |
| Injunction | Cease and desist |
| Apology | Formal acknowledgment |
| CommunityService | Contribution to community |
| ReputationAdjustment | MATL score impact |
| RestorativeCircle | Participate in healing |

## Evidence Standards

All evidence is:
- **Encrypted** at rest (optional)
- **Tamper-evident** with content hashing
- **Timestamped** with Holochain timestamps
- **Chain of custody** fully tracked
- **Retained** for 7 years (configurable)

## Integration with Other hApps

### Governance
- Case escalation from governance disputes
- Policy interpretation
- Constitutional review

### Finance
- Compensation enforcement via Finance hApp
- Asset freezing capabilities
- Escrow for disputed amounts

### Identity
- DID-based party identification
- Credential verification for jurors
- Reputation impacts

### Knowledge
- Evidence links to knowledge claims
- Legal precedent as claims
- Epistemic classification of testimony

## Example Usage

```typescript
// File a case
const case = await justice.fileCase({
  title: "Breach of Service Agreement",
  description: "Provider failed to deliver...",
  caseType: CaseType.ContractDispute,
  respondent: providerDid,
  severity: CaseSeverity.Moderate,
  context: { happ: "mycelix-finance", referenceId: "tx:123" },
});

// Submit evidence
await justice.submitEvidence({
  caseId: case.id,
  evidenceType: EvidenceType.Document,
  content: {
    hash: "sha256:...",
    reference: "ipfs://...",
    mimeType: "application/pdf",
  },
  description: "Original service agreement signed by both parties",
});

// After arbitration, check decision
const decision = await justice.getDecision(case.id);
if (decision.outcome === DecisionOutcome.ForComplainant) {
  // Remedies will be enforced automatically
  console.log("Remedies:", decision.remedies);
}

// Alternative: Propose restorative circle
await justice.proposeRestorativeCircle({
  caseId: case.id,
  facilitatorDid: facilitator.did,
  proposedParticipants: [complainant, respondent, communityRep],
});
```

## Case Flow

```
┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌────────┐
│  Filed   │───►│ Mediation │───►│ Arbitration │───►│ Appeal │
└──────────┘    └───────────┘    └─────────────┘    └────────┘
                      │                  │               │
                      ▼                  ▼               ▼
                ┌──────────┐      ┌──────────┐    ┌──────────┐
                │ Settled  │      │ Decision │    │  Final   │
                └──────────┘      └──────────┘    └──────────┘
                                        │
                      ┌─────────────────┼─────────────────┐
                      ▼                 ▼                 ▼
              ┌────────────┐   ┌─────────────────┐  ┌─────────┐
              │Enforcement │   │Restorative Circle│  │ Closed  │
              └────────────┘   └─────────────────┘  └─────────┘
```

## Development

```bash
# Enter Nix environment
cd /srv/luminous-dynamics/mycelix-workspace
nix develop

# Build
cd happs/justice
cargo build --release --target wasm32-unknown-unknown

# Package
hc dna pack dna/
hc app pack .
```

## Roadmap

- [ ] Case filing & management
- [ ] Evidence submission & custody
- [ ] Mediation workflow
- [ ] Arbitrator selection
- [ ] Decision rendering
- [ ] Appeal process
- [ ] Restorative circles
- [ ] Cross-hApp enforcement

## Related hApps

- **mycelix-governance**: Policy disputes, constitutional review
- **mycelix-finance**: Compensation enforcement
- **mycelix-identity**: Party & juror verification
- **mycelix-knowledge**: Evidence as claims, precedent

## License

Apache-2.0
