# Mycelix Epistemic Markets

**A Living System for Collective Truth-Seeking**

> *"We build this not as a tool to use, but as a partner to grow with."*

Epistemic Markets transcends traditional prediction markets to create a living infrastructure for collective intelligence - a system that learns, grows, heals, and evolves alongside the beings who inhabit it.

---

## Start Here

| Document | Purpose |
|----------|---------|
| [**MANIFESTO**](./MANIFESTO.md) | Our values, beliefs, and invitation to join |
| [**GETTING_STARTED**](./GETTING_STARTED.md) | Practical guide from first prediction to wisdom keeper |
| [**GENESIS_MARKET**](./genesis/GENESIS_MARKET.md) | The first prediction - about the system itself |
| [**FOUNDING_WISDOM**](./genesis/FOUNDING_WISDOM.md) | Wisdom seeds planted at genesis |

## The Vision

| Document | Explores |
|----------|----------|
| [**ECOSYSTEM_INTEGRATION**](./docs/ECOSYSTEM_INTEGRATION.md) | How we connect with all Mycelix hApps |
| [**LONG_TERM_VISION**](./docs/LONG_TERM_VISION.md) | Evolution from tool to civilizational nervous system |
| [**THE_DEEPER_VISION**](./docs/THE_DEEPER_VISION.md) | Spiritual dimensions, Eight Harmonies, healing |
| [**THE_LIVING_PROTOCOL**](./docs/THE_LIVING_PROTOCOL.md) | How the system becomes genuinely alive |
| [**ADVANCED_MECHANISMS**](./docs/ADVANCED_MECHANISMS.md) | Belief graphs, attention markets, anti-manipulation |

## The Design

| Document | Covers |
|----------|--------|
| [**DESIGN_PRINCIPLES**](./docs/DESIGN_PRINCIPLES.md) | 26 principles for human-centered, epistemic, social, technical design |
| [**FAILURE_MODES**](./docs/FAILURE_MODES.md) | What can go wrong, how we detect it, how we heal |
| [**RITUALS**](./docs/RITUALS.md) | Daily, weekly, monthly, annual practices that sustain the culture |
| [**GOVERNANCE**](./docs/GOVERNANCE.md) | How we decide together, checks and balances, evolution |
| [**ECONOMICS**](./docs/ECONOMICS.md) | Value types, flows, staking, fees, sustainability |
| [**EDGE_CASES**](./docs/EDGE_CASES.md) | Weird situations and how we handle them |
| [**PERSONAS**](./docs/PERSONAS.md) | The 8 humans we design for, their journeys and needs |
| [**RESEARCH_AGENDA**](./docs/RESEARCH_AGENDA.md) | Open questions we haven't answered yet |
| [**COMPARATIVE_ANALYSIS**](./docs/COMPARATIVE_ANALYSIS.md) | How we differ from Polymarket, Metaculus, etc. |
| [**SECURITY**](./docs/SECURITY.md) | Threat model, cryptography, attack resistance |
| [**FAQ**](./docs/FAQ.md) | Common questions from newcomers to experts |
| [**CASE_STUDIES**](./docs/CASE_STUDIES.md) | Real-world examples: climate, tech, community, science, crisis |

## Community & Operations

| Document | Addresses |
|----------|-----------|
| [**ACCESSIBILITY**](./docs/ACCESSIBILITY.md) | Universal design, WCAG compliance, assistive technology |
| [**MODERATION**](./docs/MODERATION.md) | Community moderation policies, tools, appeals process |
| [**ONBOARDING**](./docs/ONBOARDING.md) | Journey from first click to wisdom keeper, persona paths |
| [**INTERNATIONALIZATION**](./docs/INTERNATIONALIZATION.md) | Multi-language support, RTL, cultural adaptation |

## Development

| Document | Helps With |
|----------|------------|
| [**DEVELOPER_GUIDE**](./docs/DEVELOPER_GUIDE.md) | Building on the platform, zomes, SDK, deployment |
| [**PERFORMANCE**](./docs/PERFORMANCE.md) | Optimization, caching, profiling, benchmarks |

## Reference

| Document | Provides |
|----------|----------|
| [**GLOSSARY**](./docs/GLOSSARY.md) | Complete vocabulary with definitions and cross-references |
| [**API_REFERENCE**](./docs/API_REFERENCE.md) | Full SDK documentation, zome functions, types, errors |
| [**METRICS**](./docs/METRICS.md) | What we measure, targets, dashboards, anti-metrics |

---

## Key Innovations

| Feature | Traditional Markets | Mycelix Epistemic Markets |
|---------|---------------------|---------------------------|
| **Stakes** | Money only | Money + Reputation + Social + Commitments |
| **Questions** | Given externally | Question Markets discover what's worth knowing |
| **Resolution** | Single oracle | MATL-weighted distributed oracles with 45% Byzantine tolerance |
| **Adversarial** | Competition | Collaboration rewarding synthesis |
| **Privacy** | Public positions | ZK commitments, private aggregation |
| **Integration** | Standalone | Deep hooks into governance, knowledge, identity |

## Architecture

```
epistemic-markets/
├── zomes/
│   ├── markets/           # Core prediction markets
│   ├── question_markets/  # Trade on what's worth knowing
│   ├── resolution/        # MATL-weighted oracle consensus
│   ├── scoring/           # Calibration & Brier scores
│   ├── predictions/       # Prediction submission
│   └── markets_bridge/    # Cross-hApp integration
├── sdk-ts/                # TypeScript client SDK
└── docs/                  # Documentation
```

## Core Concepts

### 1. Epistemic Position (E-N-M Classification)

Every market is classified on three axes from the Mycelix Epistemic Charter:

**E-Axis (Empirical)**: How can we verify the outcome?
- E0: Subjective (no external verification)
- E1: Testimonial (witness testimony)
- E2: Private Verify (ZK proofs, credentials)
- E3: Cryptographic (on-chain events)
- E4: Measurable (scientific reproducibility)

**N-Axis (Normative)**: Who must agree?
- N0: Personal (individual perspective)
- N1: Communal (community consensus)
- N2: Network (network-wide agreement)
- N3: Universal (objective truth)

**M-Axis (Materiality)**: How long does this matter?
- M0: Ephemeral (hours)
- M1: Temporal (days/weeks)
- M2: Persistent (months/years)
- M3: Foundational (permanent)

The epistemic position automatically determines the resolution mechanism.

### 2. Multi-Dimensional Stakes

Beyond monetary stakes, participants can stake:

```typescript
interface MultiDimensionalStake {
  monetary?: { amount: number; currency: string };

  reputation?: {
    domains: string[];      // Which MATL domains at risk
    stakePercentage: number; // How much of score to stake
    confidenceMultiplier: number;
  };

  social?: {
    visibility: "Private" | "Limited" | "Community" | "Public";
    identityLink?: string;  // Link to verified identity
  };

  commitment?: {
    ifCorrect: Commitment[];  // Actions if prediction correct
    ifWrong: Commitment[];    // Actions if prediction wrong
  };

  time?: {
    researchHours: number;
    evidenceSubmitted: EntryHash[];
  };
}
```

This democratizes participation: experts can stake reputation, researchers can stake time, and anyone can make public commitments.

### 3. Question Markets

Before predicting answers, discover which questions matter:

```typescript
// Propose a question
const questionId = await client.questions.proposeQuestion({
  questionText: "What will be the primary cause of the next financial crisis?",
  context: "Seeking to understand systemic risks in global markets",
  domains: ["finance", "economics", "risk"],
});

// Signal curiosity (lightweight interest)
await client.questions.signalCuriosity(questionId, "Critical for portfolio protection");

// Buy value shares (bet the answer is worth knowing)
await client.questions.buyShares(questionId, 100, undefined, "High decision relevance");
```

When a question reaches sufficient value, it automatically spawns a prediction market.

### 4. MATL-Weighted Oracle Networks

Resolution uses Mycelix's revolutionary 45% Byzantine tolerance:

```rust
// Oracle votes are weighted by MATL composite score
oracle_weight = matl_score.composite.powi(2);

// Consensus requires weighted supermajority
if weighted_votes_for_outcome / total_weight >= 0.67 {
    resolve(outcome);
}

// Byzantine analysis runs on every vote
let analysis = ByzantineAnalysis::analyze(&votes, &config);
match analysis.recommendation {
    Halt => abort_resolution(),
    Escalate => escalate_to_governance(),
    _ => continue_voting(),
}
```

### 5. Cross-hApp Integration

Create markets about events in any Mycelix hApp:

```typescript
// Market about governance outcome
const request = await bridge.submitPredictionRequest({
  sourceHapp: "Governance",
  question: "Will MIP-47 pass with >70% approval?",
  context: { proposalId: "mip-47" },
  resolutionCriteria: {
    type: "OnChainEvent",
    eventType: "proposal.resolved",
    eventSource: "Governance",
  },
  urgency: "Medium",
  bounty: 500,
});

// Market auto-resolves when governance vote completes
```

## Quick Start

### Installation

```bash
# Clone the repository
cd mycelix-workspace/happs/epistemic-markets

# Install TypeScript SDK
cd sdk-ts && npm install && npm run build
```

### Usage

```typescript
import { EpistemicMarketsClient, calculateStakeValue } from "@mycelix/epistemic-markets-sdk";

// Initialize client
const client = new EpistemicMarketsClient(appClient);

// Create a prediction market
const marketId = await client.markets.createMarket({
  question: "Will Bitcoin exceed $100k by end of 2025?",
  description: "Resolves YES if BTC/USD exceeds $100,000 at any point before Dec 31, 2025 UTC",
  outcomes: ["Yes", "No"],
  epistemicPosition: {
    empirical: "Cryptographic",  // Price is on-chain verifiable
    normative: "Universal",      // Objective fact
    materiality: "Temporal",     // Matters for ~1 year
  },
  mechanism: { type: "LMSR", liquidityParameter: 100, subsidyPool: 1000 },
  closesAt: Date.parse("2025-12-30T00:00:00Z"),
  resolutionDeadline: Date.parse("2026-01-07T00:00:00Z"),
  tags: ["crypto", "bitcoin", "price"],
});

// Make a prediction with reputation stake
await client.predictions.submitPrediction({
  marketId,
  outcome: "Yes",
  confidence: 0.72,
  stake: {
    reputation: {
      domains: ["crypto", "markets"],
      stakePercentage: 0.05,  // 5% of MATL score at risk
      confidenceMultiplier: 1.0,
    },
    social: {
      visibility: "Public",
    },
  },
  reasoning: "Historical halving cycles suggest continued appreciation",
});
```

## Zome Reference

### markets
Core prediction market functionality.

| Function | Description |
|----------|-------------|
| `create_market` | Create a new prediction market |
| `get_market` | Get market by hash |
| `list_markets` | List all markets |
| `list_open_markets` | List open markets |
| `search_markets_by_tag` | Search by tag |
| `close_market` | Close market for predictions |

### question_markets
Trade on which questions are worth answering.

| Function | Description |
|----------|-------------|
| `propose_question` | Propose a new question |
| `signal_curiosity` | Signal interest in question |
| `buy_question_shares` | Buy value shares |
| `get_question_market` | Get question by hash |
| `list_question_markets` | List all questions |
| `get_top_questions` | Get highest-value questions |

### resolution
MATL-weighted oracle resolution.

| Function | Description |
|----------|-------------|
| `start_resolution` | Begin resolution process |
| `submit_oracle_vote` | Submit oracle vote |
| `finalize_resolution` | Finalize and trigger payouts |
| `dispute_resolution` | Dispute resolved outcome |
| `get_resolution_process` | Get resolution state |

### markets_bridge
Cross-hApp market integration.

| Function | Description |
|----------|-------------|
| `submit_prediction_request` | Request prediction from another hApp |
| `accept_request` | Accept pending request |
| `create_market_from_request` | Create market from request |
| `subscribe_to_event` | Subscribe to resolution events |
| `handle_bridge_event` | Handle incoming bridge events |
| `list_pending_requests` | List pending requests |

## Integration with Mycelix Ecosystem

### MATL (Trust Layer)
- Oracle selection based on composite trust scores
- Reputation staking for predictions
- Byzantine detection using MATL consistency metrics

### Knowledge Graph
- Markets can be anchored to knowledge claims
- Outcomes verified against claim database
- Evidence linking for complex predictions

### Governance
- Market parameters voted via MIPs
- Disputed outcomes escalate to governance
- Futarchy integration for policy decisions

### Identity
- Credential-based market access
- Cross-hApp reputation aggregation
- Verified identity for high-stakes predictions

### Finance
- Collateral management via finance hApp
- Credit-based capital requirements
- Treasury for liquidity pools

## Roadmap

### Phase 1: Foundation (Current)
- [x] Core market data structures
- [x] Question markets mechanism
- [x] Multi-dimensional stakes
- [x] MATL-weighted resolution
- [x] TypeScript SDK
- [x] Cross-hApp bridge

### Phase 2: Advanced Mechanisms
- [ ] LMSR automated market maker
- [ ] Continuous double auction
- [ ] Calibration training games
- [ ] Adversarial collaboration markets

### Phase 3: Deep Integration
- [ ] Knowledge graph anchoring
- [ ] Futarchy for governance
- [ ] AI oracle integration
- [ ] Privacy-preserving aggregation

### Phase 4: Collective Intelligence
- [ ] Prediction teams and swarms
- [ ] Source reliability markets
- [ ] Epistemic debt tracking
- [ ] Meta-prediction markets

## Design Philosophy

Epistemic Markets is designed around several core principles:

1. **Questions before answers**: Before predicting, discover what's worth knowing
2. **Multi-dimensional accountability**: Money isn't the only valuable stake
3. **Truth through collaboration**: Reward synthesis, not just winning
4. **Byzantine resilience**: 45% tolerance for adversarial participants
5. **Cross-domain integration**: Predictions about anything in the ecosystem
6. **Calibration feedback**: Accuracy improves reputation over time

## Complete File Structure

```
epistemic-markets/
├── MANIFESTO.md                    # Our declaration and invitation
├── GETTING_STARTED.md              # Practical onboarding guide
├── CONTRIBUTING.md                 # How to join the work
├── README.md                       # This file
│
├── genesis/                        # The founding ceremony
│   ├── GENESIS_MARKET.md          # The first prediction
│   └── FOUNDING_WISDOM.md         # Seeds for future generations
│
├── docs/                           # Vision, philosophy, design, and reference
│   ├── ACCESSIBILITY.md           # Universal design, WCAG compliance
│   ├── ADVANCED_MECHANISMS.md     # Belief graphs, anti-manipulation
│   ├── API_REFERENCE.md           # Complete SDK and zome documentation
│   ├── CASE_STUDIES.md            # Real-world examples and lessons
│   ├── COMPARATIVE_ANALYSIS.md    # How we differ from alternatives
│   ├── DESIGN_PRINCIPLES.md       # 26 design principles
│   ├── DEVELOPER_GUIDE.md         # Building on the platform
│   ├── ECONOMICS.md               # Value flows, staking, sustainability
│   ├── ECOSYSTEM_INTEGRATION.md   # Cross-hApp patterns
│   ├── EDGE_CASES.md              # Weird situations, boundary handling
│   ├── FAILURE_MODES.md           # What can go wrong, healing
│   ├── FAQ.md                     # Frequently asked questions
│   ├── GLOSSARY.md                # Terms and definitions
│   ├── GOVERNANCE.md              # Decision-making, evolution
│   ├── INTERNATIONALIZATION.md    # Multi-language support
│   ├── LONG_TERM_VISION.md        # Civilizational evolution
│   ├── METRICS.md                 # What we measure and why
│   ├── MODERATION.md              # Community moderation policies
│   ├── ONBOARDING.md              # User journey and personas
│   ├── PERFORMANCE.md             # Optimization and profiling
│   ├── PERSONAS.md                # The humans we design for
│   ├── RESEARCH_AGENDA.md         # Open questions for the future
│   ├── RITUALS.md                 # Practices that sustain culture
│   ├── SECURITY.md                # Threat model, protections
│   ├── THE_DEEPER_VISION.md       # Spiritual dimensions
│   └── THE_LIVING_PROTOCOL.md     # The system as life
│
├── zomes/                          # Holochain zomes (Rust)
│   ├── markets/                   # Core prediction markets
│   ├── predictions/               # Predictions with reasoning
│   ├── resolution/                # MATL-weighted oracles
│   ├── scoring/                   # Calibration & wisdom
│   ├── question_markets/          # What's worth knowing?
│   └── markets_bridge/            # Cross-hApp integration
│
├── sdk-ts/                         # TypeScript SDK
│   └── src/index.ts               # Client library
│
├── tests/                          # Integration tests
│   └── integration.test.ts        # Tests as teaching
│
├── dna.yaml                        # Holochain DNA config
├── happ.yaml                       # hApp bundle manifest
└── workdir/                        # Development config
    └── dna.yaml                   # Dev-friendly settings
```

## Contributing

We welcome contributions at every level:

- **Predictions**: Join markets, stake honestly, share reasoning
- **Wisdom**: Plant seeds for future generations
- **Code**: Improve zomes, SDK, tests
- **Documentation**: Clarify, extend, translate
- **Vision**: Challenge, refine, expand

See [CONTRIBUTING.md](./CONTRIBUTING.md) for technical guidelines.

## The Invitation

This is not a product launch. It is an invitation to a practice.

You join by:
- Making your first prediction
- Staking something you value
- Sharing your reasoning openly
- Celebrating your uncertainty
- Learning from your mistakes
- Teaching what you discover
- Caring for those who come after

The journey begins with a single honest prediction.

**[Read the Manifesto](./MANIFESTO.md)** | **[Get Started](./GETTING_STARTED.md)** | **[Join the Genesis](./genesis/GENESIS_MARKET.md)**

## License

MIT License - see [LICENSE](./LICENSE)

---

> *"In the end, we will not be judged by the systems we built,*
> *but by the life we nurtured within them."*

---

*Building collective intelligence infrastructure for human flourishing*
*With love, for the future*
