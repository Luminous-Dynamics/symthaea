# Mycelix Developer Guide

## Overview

This guide provides everything developers need to build on, extend, and deploy the Mycelix Civilizational OS. Whether you're creating a new hApp, integrating with existing ones, or deploying a community instance, this document is your comprehensive reference.

---

## Architecture Overview

### The Mycelix Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   Web UI    │ │  Mobile UI  │ │   CLI       │ │  AI Agents  │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
└─────────┼───────────────┼───────────────┼───────────────┼───────────────┘
          │               │               │               │
┌─────────┴───────────────┴───────────────┴───────────────┴───────────────┐
│                         MYCELIX SDK LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  @mycelix/sdk - Unified API for all hApps                        │   │
│  │  - TypeScript/JavaScript SDK                                      │   │
│  │  - Rust SDK (native)                                             │   │
│  │  - Python SDK (AI/ML integration)                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─────────┴───────────────────────────────────────────────────────────────┐
│                         BRIDGE PROTOCOL                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Event Bus │ Cross-hApp Calls │ State Sync │ Permission Router   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─────────┴───────────────────────────────────────────────────────────────┐
│                         hAPP LAYER (43 hApps)                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐              │
│  │  Tier 0   │ │  Tier 1   │ │  Tier 2   │ │ Tier 3-4  │              │
│  │  Spine    │ │   Core    │ │ Essential │ │ Advanced  │              │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─────────┴───────────────────────────────────────────────────────────────┐
│                         HOLOCHAIN LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Conductor │ DNA │ DHT │ Source Chain │ Validation │ Gossip     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
mycelix/
├── happs/                      # Individual hApp DNAs
│   ├── tier-0-spine/
│   │   ├── bridge/
│   │   ├── matl/
│   │   ├── attest/
│   │   ├── sentinel/
│   │   └── chronicle/
│   ├── tier-1-core/
│   │   ├── agora/
│   │   ├── arbiter/
│   │   ├── oracle/
│   │   ├── marketplace/
│   │   ├── treasury/
│   │   ├── collab/
│   │   └── covenant/
│   ├── tier-2-essential/
│   │   └── [20 hApps...]
│   ├── tier-2.5-meta/
│   │   ├── nudge/
│   │   └── spiral/
│   └── tier-3-4-advanced/
│       └── [12 hApps...]
├── sdk/
│   ├── typescript/             # @mycelix/sdk
│   ├── rust/                   # mycelix-sdk crate
│   └── python/                 # mycelix-py
├── ui/
│   ├── web/                    # Main web application
│   ├── mobile/                 # React Native app
│   └── components/             # Shared UI components
├── tools/
│   ├── cli/                    # mycelix CLI
│   ├── testing/                # Test framework
│   └── simulation/             # Multi-agent simulator
├── templates/
│   ├── community/              # Community deployment templates
│   └── happ/                   # New hApp scaffolding
└── docs/
    └── civilizational-os/      # This documentation
```

---

## Quick Start

### Prerequisites

```bash
# Install Holochain
curl -L https://holochain.org/install.sh | bash

# Install Mycelix CLI
cargo install mycelix-cli

# Or via npm
npm install -g @mycelix/cli
```

### Create a New Community Instance

```bash
# Initialize a new Mycelix community
mycelix init my-community

# Configure community settings
cd my-community
mycelix configure

# Deploy locally for development
mycelix dev

# Deploy to production
mycelix deploy --network mainnet
```

### Create a New hApp

```bash
# Scaffold a new hApp
mycelix happ new my-happ --tier 2

# This creates:
# my-happ/
# ├── dna/
# │   ├── coordinator/
# │   │   └── src/
# │   │       ├── lib.rs
# │   │       └── handlers.rs
# │   └── integrity/
# │       └── src/
# │           ├── lib.rs
# │           └── entries.rs
# ├── ui/
# │   └── src/
# ├── tests/
# └── mycelix.toml
```

---

## SDK Reference

### TypeScript SDK

```typescript
import { MycelixClient, Attest, Agora, Marketplace } from '@mycelix/sdk';

// Initialize client
const client = new MycelixClient({
  conductorUrl: 'ws://localhost:8888',
  appId: 'my-community',
});

// Connect with identity
await client.connect();

// Use hApp modules
const identity = await client.attest.getMyProfile();
const proposals = await client.agora.getActiveProposals();
const listings = await client.marketplace.search({ category: 'services' });

// Cross-hApp operations via Bridge
const result = await client.bridge.call({
  target: 'treasury',
  method: 'allocate',
  payload: { amount: 100, purpose: 'community-project' },
  requiredApprovals: ['agora'],  // Requires governance approval
});

// Subscribe to events
client.bridge.on('proposal.passed', (event) => {
  console.log('Proposal passed:', event.proposalId);
});

// Stage-aware interactions (via Spiral)
const stageContext = await client.spiral.getMyStageContext();
const nudgedInterface = await client.nudge.getInterfaceConfig({
  happ: 'agora',
  stage: stageContext.primaryStage,
});
```

### Rust SDK

```rust
use mycelix_sdk::{MycelixClient, Result};
use mycelix_sdk::happs::{attest, agora, marketplace};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize client
    let client = MycelixClient::connect("ws://localhost:8888", "my-community").await?;

    // Get identity
    let profile = attest::get_my_profile(&client).await?;

    // Create a proposal
    let proposal = agora::create_proposal(&client, agora::ProposalInput {
        title: "Fund community garden".into(),
        description: "...".into(),
        proposal_type: agora::ProposalType::ResourceAllocation,
        options: vec!["Approve".into(), "Reject".into()],
        voting_period: Duration::days(7),
    }).await?;

    // Listen for events
    let mut events = client.bridge().subscribe("agora.*").await?;
    while let Some(event) = events.next().await {
        println!("Event: {:?}", event);
    }

    Ok(())
}
```

### Python SDK (AI/ML Integration)

```python
from mycelix import MycelixClient, AI
from mycelix.happs import emergence, pulse, chronicle

# Initialize client with AI capabilities
client = MycelixClient(
    conductor_url="ws://localhost:8888",
    app_id="my-community"
)

# AI agent for pattern detection
ai_agent = AI.Agent(
    client=client,
    capabilities=["read", "suggest"],  # No write without human approval
    ethical_bounds=AI.EthicalBounds.STANDARD,
)

# Analyze community patterns
async def detect_patterns():
    # Get collective sentiment
    sentiment = await pulse.get_community_sentiment()

    # Analyze historical patterns
    patterns = await emergence.detect_patterns(
        timeframe="30d",
        domains=["governance", "economics", "social"]
    )

    # AI-assisted insight generation
    insights = await ai_agent.analyze(
        data=patterns,
        prompt="Identify emerging opportunities for community coordination"
    )

    # Insights require human review before action
    await chronicle.submit_for_review(
        content=insights,
        category="ai-generated",
        requires_approval=True
    )
```

---

## hApp Development

### Anatomy of a Mycelix hApp

Every Mycelix hApp follows a consistent structure:

```
my-happ/
├── dna/
│   ├── coordinator/              # Business logic
│   │   └── src/
│   │       ├── lib.rs            # Zome definition
│   │       ├── handlers.rs       # External function handlers
│   │       ├── bridge.rs         # Bridge protocol integration
│   │       ├── nudge.rs          # Behavioral hooks
│   │       └── spiral.rs         # Stage-aware adaptations
│   ├── integrity/                # Data validation
│   │   └── src/
│   │       ├── lib.rs            # Entry/link definitions
│   │       ├── entries.rs        # Entry type definitions
│   │       └── validate.rs       # Validation rules
│   └── workdir/
│       └── dna.yaml              # DNA configuration
├── ui/
│   ├── src/
│   │   ├── components/           # UI components
│   │   ├── hooks/                # React hooks for hApp
│   │   ├── stores/               # State management
│   │   └── stage-variants/       # Stage-specific UI variants
│   └── package.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── simulation/               # Multi-agent scenarios
├── mycelix.toml                  # Mycelix configuration
└── README.md
```

### mycelix.toml Configuration

```toml
[happ]
name = "my-happ"
version = "0.1.0"
tier = 2
category = "social"
description = "A custom hApp for my community"

[dependencies]
# Tier 0 dependencies (automatic)
bridge = "0.6.0"
matl = "0.6.0"
attest = "0.6.0"

# Additional dependencies
agora = "0.6.0"      # For governance integration
treasury = "0.6.0"   # For economic features

[bridge]
# Events this hApp emits
emits = [
    "my-happ.item.created",
    "my-happ.item.updated",
    "my-happ.action.completed"
]

# Events this hApp subscribes to
subscribes = [
    "agora.proposal.passed",
    "treasury.allocation.approved"
]

[nudge]
# Behavioral integration points
decision_points = [
    { name = "create_item", type = "commitment" },
    { name = "large_transaction", type = "cooling_off", delay = "24h" }
]

[spiral]
# Stage-aware features
stage_variants = true
assessment_integration = true

[permissions]
# Capability requirements
required_capabilities = ["identity", "basic_trust"]
optional_capabilities = ["economic", "governance"]

[testing]
# Test configuration
simulation_agents = 100
scenario_files = ["tests/simulation/*.yaml"]
```

### Implementing Bridge Protocol Integration

```rust
// bridge.rs - Bridge protocol integration

use hdk::prelude::*;
use mycelix_bridge::{BridgeEvent, BridgeCall, BridgeResponse};

/// Register with Bridge on init
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Register this hApp with Bridge
    let registration = BridgeRegistration {
        happ_id: "my-happ".into(),
        version: "0.1.0".into(),
        capabilities: vec![
            Capability::Read,
            Capability::Write,
            Capability::Subscribe,
        ],
        event_schema: get_event_schema(),
    };

    call_bridge("register", registration)?;

    Ok(InitCallbackResult::Pass)
}

/// Emit event to Bridge
pub fn emit_event(event: MyHappEvent) -> ExternResult<()> {
    let bridge_event = BridgeEvent {
        source: "my-happ".into(),
        event_type: event.event_type(),
        payload: serialize(&event)?,
        timestamp: sys_time()?,
        requires_ack: event.requires_acknowledgment(),
    };

    call_bridge("emit", bridge_event)?;

    Ok(())
}

/// Handle incoming Bridge calls
#[hdk_extern]
pub fn handle_bridge_call(call: BridgeCall) -> ExternResult<BridgeResponse> {
    // Verify caller permissions via MATL
    let trust_score = verify_caller_trust(&call.caller)?;
    if trust_score < call.required_trust {
        return Ok(BridgeResponse::Unauthorized);
    }

    // Route to appropriate handler
    match call.method.as_str() {
        "get_item" => handle_get_item(call.payload),
        "create_item" => handle_create_item(call.payload),
        "update_item" => handle_update_item(call.payload),
        _ => Ok(BridgeResponse::MethodNotFound),
    }
}

/// Subscribe to events from other hApps
pub fn setup_subscriptions() -> ExternResult<()> {
    // Subscribe to governance decisions
    subscribe_to("agora.proposal.passed", |event| {
        if event.affects_happ("my-happ") {
            apply_governance_decision(event)?;
        }
        Ok(())
    })?;

    // Subscribe to treasury allocations
    subscribe_to("treasury.allocation.approved", |event| {
        if event.recipient == "my-happ" {
            receive_allocation(event)?;
        }
        Ok(())
    })?;

    Ok(())
}
```

### Implementing Nudge Integration

```rust
// nudge.rs - Behavioral design integration

use hdk::prelude::*;
use mycelix_nudge::{NudgeContext, NudgeDecision, DecisionPoint};

/// Wrap actions with nudge decision points
pub fn create_item_with_nudge(input: CreateItemInput) -> ExternResult<Item> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Get nudge context for this decision
    let nudge_context = get_nudge_context(
        &agent,
        DecisionPoint::Commitment {
            action: "create_item".into(),
            reversibility: Reversibility::Moderate,
        },
    )?;

    // Apply any applicable nudges
    match nudge_context.decision {
        NudgeDecision::Proceed => {
            // Normal flow
            create_item_internal(input)
        }
        NudgeDecision::CoolingOff { until, reason } => {
            // Return cooling off response
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cooling off until {}: {}", until, reason
            ))))
        }
        NudgeDecision::ConfirmationRequired { prompt } => {
            // Store pending action, return confirmation request
            store_pending_action(PendingAction {
                action_type: "create_item".into(),
                payload: serialize(&input)?,
                confirmation_prompt: prompt,
                expires: sys_time()? + Duration::hours(24),
            })?;

            Err(wasm_error!(WasmErrorInner::Guest(
                "Confirmation required".into()
            )))
        }
        NudgeDecision::ShowImpact { impact_preview } => {
            // Return impact preview for UI to display
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Impact preview: {:?}", impact_preview
            ))))
        }
    }
}

/// For large transactions, apply cooling off
pub fn large_transaction_with_cooling(input: TransactionInput) -> ExternResult<Transaction> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Check if this is a "large" transaction
    if input.amount > get_large_transaction_threshold(&agent)? {
        let nudge_context = get_nudge_context(
            &agent,
            DecisionPoint::CoolingOff {
                action: "large_transaction".into(),
                delay: Duration::hours(24),
                reason: "Large transactions benefit from reflection time".into(),
            },
        )?;

        if let NudgeDecision::CoolingOff { until, .. } = nudge_context.decision {
            // Store for later execution
            schedule_transaction(input, until)?;
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Transaction scheduled for {}", until
            ))));
        }
    }

    execute_transaction(input)
}
```

### Implementing Spiral Integration

```rust
// spiral.rs - Developmental stage integration

use hdk::prelude::*;
use mycelix_spiral::{StageContext, DevelopmentalStage, StageInterface};

/// Get stage-appropriate interface configuration
#[hdk_extern]
pub fn get_stage_interface(agent: AgentPubKey) -> ExternResult<MyHappInterface> {
    // Get agent's developmental context from Spiral
    let stage_context = call_spiral("get_stage_context", agent)?;

    // Generate stage-appropriate interface
    let interface = match stage_context.primary_stage {
        DevelopmentalStage::Traditional => MyHappInterface {
            language_style: LanguageStyle::Authoritative,
            feature_emphasis: vec!["rules", "roles", "approval_required"],
            hidden_features: vec!["experimental", "self_directed"],
            visual_style: VisualStyle::Structured,
            help_style: HelpStyle::Directive,
        },
        DevelopmentalStage::Modern => MyHappInterface {
            language_style: LanguageStyle::Professional,
            feature_emphasis: vec!["efficiency", "metrics", "personal_dashboard"],
            hidden_features: vec!["consensus_building"],
            visual_style: VisualStyle::DataDriven,
            help_style: HelpStyle::Tutorial,
        },
        DevelopmentalStage::Postmodern => MyHappInterface {
            language_style: LanguageStyle::Inclusive,
            feature_emphasis: vec!["collaboration", "impact", "community"],
            hidden_features: vec!["competitive_features"],
            visual_style: VisualStyle::Warm,
            help_style: HelpStyle::Supportive,
        },
        DevelopmentalStage::Integral => MyHappInterface {
            language_style: LanguageStyle::Contextual,
            feature_emphasis: vec!["all_features", "meta_view", "stage_awareness"],
            hidden_features: vec![],
            visual_style: VisualStyle::Adaptive,
            help_style: HelpStyle::Exploratory,
        },
        _ => default_interface(),
    };

    Ok(interface)
}

/// Adapt content for developmental stage
pub fn adapt_content_for_stage(
    content: &str,
    stage: &DevelopmentalStage,
) -> ExternResult<String> {
    let adaptation = call_spiral("translate_for_stage", TranslationRequest {
        content: content.into(),
        target_stage: stage.clone(),
        content_type: ContentType::Instructional,
    })?;

    Ok(adaptation.translated)
}
```

---

## Testing Framework

### Unit Testing

```rust
// tests/unit/test_handlers.rs

use hdk::prelude::*;
use holochain::sweettest::*;
use my_happ::*;

#[tokio::test(flavor = "multi_thread")]
async fn test_create_item() {
    let (conductor, agent, cell) = setup_test_cell().await;

    // Create item
    let input = CreateItemInput {
        title: "Test Item".into(),
        description: "A test item".into(),
    };

    let result: Item = conductor
        .call(&cell.zome("my_happ"), "create_item", input)
        .await;

    assert_eq!(result.title, "Test Item");
    assert!(result.id.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_bridge_integration() {
    let (conductor, agents, cells) = setup_multi_happ_test(
        vec!["my_happ", "agora", "treasury"]
    ).await;

    // Test cross-hApp call
    let bridge_call = BridgeCall {
        target: "treasury".into(),
        method: "check_balance".into(),
        payload: serialize(&agents[0]).unwrap(),
    };

    let result: BridgeResponse = conductor
        .call(&cells[0].zome("my_happ"), "handle_bridge_call", bridge_call)
        .await;

    assert!(matches!(result, BridgeResponse::Success(_)));
}
```

### Integration Testing

```rust
// tests/integration/test_full_flow.rs

use mycelix_testing::*;

#[tokio::test]
async fn test_governance_to_action_flow() {
    // Setup full Mycelix environment
    let env = MycelixTestEnv::new()
        .with_happs(vec!["bridge", "attest", "agora", "treasury", "my_happ"])
        .with_agents(5)
        .await;

    // Create identities
    for agent in &env.agents {
        env.attest.create_profile(agent, ProfileInput::default()).await;
    }

    // Create proposal that affects my_happ
    let proposal = env.agora.create_proposal(&env.agents[0], ProposalInput {
        title: "Enable new feature in my_happ".into(),
        affects_happs: vec!["my_happ".into()],
        ..Default::default()
    }).await;

    // All agents vote
    for agent in &env.agents {
        env.agora.vote(agent, &proposal.id, Vote::Approve).await;
    }

    // Fast-forward time to end voting
    env.advance_time(Duration::days(7)).await;

    // Execute proposal
    env.agora.execute_proposal(&proposal.id).await;

    // Verify my_happ received the governance decision
    let config = env.my_happ.get_config().await;
    assert!(config.new_feature_enabled);
}
```

### Multi-Agent Simulation

```yaml
# tests/simulation/community_scenario.yaml

name: "Community Resource Allocation"
description: "Simulate a community making collective decisions"

agents:
  - count: 50
    profile:
      stage_distribution:
        traditional: 0.2
        modern: 0.4
        postmodern: 0.3
        integral: 0.1
      trust_range: [0.3, 0.9]

  - count: 10
    profile:
      role: "facilitator"
      stage: "integral"
      trust_range: [0.8, 1.0]

timeline:
  - day: 1
    events:
      - type: "proposal_created"
        agent: "random"
        data:
          title: "Fund community garden"
          amount: 1000

  - day: 1-7
    events:
      - type: "discussion"
        participation_rate: 0.6
        sentiment_evolution: "convergent"

  - day: 7
    events:
      - type: "voting"
        participation_rate: 0.8

  - day: 8
    events:
      - type: "proposal_execution"

assertions:
  - "proposal.passed == true"
  - "treasury.balance_decreased_by == 1000"
  - "community.satisfaction > 0.7"
  - "no_agents.trust_decreased_significantly"

metrics_to_collect:
  - participation_by_stage
  - voting_patterns
  - trust_evolution
  - nudge_effectiveness
```

```bash
# Run simulation
mycelix simulate tests/simulation/community_scenario.yaml --runs 100 --report

# Output:
# ═══════════════════════════════════════════════════════════════
# Simulation Results: Community Resource Allocation
# ═══════════════════════════════════════════════════════════════
# Runs: 100
#
# Outcomes:
#   Proposal Passed: 94%
#   Proposal Failed: 6%
#
# Participation by Stage:
#   Traditional: 72% voted (avg)
#   Modern: 85% voted (avg)
#   Postmodern: 91% voted (avg)
#   Integral: 98% voted (avg)
#
# Nudge Effectiveness:
#   Cooling-off triggered: 12 times
#   Cooling-off respected: 11 times (92%)
#   Stage-appropriate messaging: 98% positive response
#
# Trust Evolution:
#   Average trust change: +0.02
#   Trust decreased significantly: 0 agents
#
# All assertions passed.
# ═══════════════════════════════════════════════════════════════
```

---

## Deployment

### Local Development

```bash
# Start local Holochain conductor with all Tier 0-1 hApps
mycelix dev

# Start with specific hApps only
mycelix dev --happs bridge,attest,agora,my-happ

# Start with UI hot-reload
mycelix dev --ui

# Start with simulation agents
mycelix dev --simulate 10
```

### Staging Deployment

```bash
# Deploy to staging network
mycelix deploy --network staging --config staging.toml

# Run integration tests against staging
mycelix test --target staging --suite integration
```

### Production Deployment

```bash
# Pre-deployment checklist
mycelix preflight --network mainnet

# Output:
# ═══════════════════════════════════════════════════════════════
# Pre-flight Checklist for Mainnet Deployment
# ═══════════════════════════════════════════════════════════════
#
# ✓ All DNAs compiled successfully
# ✓ All unit tests passing
# ✓ All integration tests passing
# ✓ Security audit: No critical issues
# ✓ Simulation tests: 95%+ success rate
# ✓ Backward compatibility: Verified
# ✓ Migration scripts: Ready
# ✓ Rollback plan: Documented
#
# Ready for deployment.
# ═══════════════════════════════════════════════════════════════

# Deploy to mainnet
mycelix deploy --network mainnet --config production.toml

# Monitor deployment
mycelix status --network mainnet --watch
```

### Community Configuration

```toml
# production.toml - Community deployment configuration

[community]
name = "Sunrise Village"
description = "A regenerative community in Vermont"
region = "us-east"

[identity]
# Which identity verification is required
verification_level = "basic"  # basic, verified, or sovereign
allow_pseudonyms = true

[governance]
# Governance configuration
default_voting_period = "7d"
quorum = 0.3
supermajority_threshold = 0.67

[economics]
# Economic configuration
primary_currency = "sunrise-hours"  # Time-based currency
enable_marketplace = true
enable_mutual_aid = true

[features]
# Which hApps to enable
enabled_happs = [
    # Tier 0 (always enabled)
    "bridge", "matl", "attest", "sentinel", "chronicle",
    # Tier 1
    "agora", "arbiter", "treasury", "collab", "covenant",
    # Tier 2
    "sanctuary", "provision", "terroir", "kinship", "ember",
    # Tier 2.5
    "nudge", "spiral"
]

[nudge]
# Behavioral configuration
enable_wise_defaults = true
enable_cooling_off = true
enable_social_proof = true
# Community can customize nudge aggressiveness
nudge_intensity = "gentle"  # minimal, gentle, moderate, strong

[spiral]
# Developmental support
enable_stage_assessment = true
enable_stage_interfaces = true
enable_growth_support = true

[federation]
# Inter-community federation
enable_federation = true
federated_communities = ["valley-collective", "northeast-network"]
```

---

## API Reference

### Core Endpoints

See individual hApp documentation for complete API references:

- [Attest API](../happs/MYCELIX_ATTEST_DESIGN.md#api-reference)
- [Agora API](../happs/MYCELIX_AGORA_DESIGN.md#api-reference)
- [Bridge API](../architecture/INTEGRATION_BLUEPRINT.md#bridge-api)
- [Nudge API](../happs/MYCELIX_NUDGE_DESIGN.md#api-reference)
- [Spiral API](../happs/MYCELIX_SPIRAL_DESIGN.md#api-reference)

### Common Patterns

```typescript
// All hApp methods follow consistent patterns

// Create
const item = await client.happ.create(input);

// Read
const item = await client.happ.get(id);
const items = await client.happ.list(filter);
const items = await client.happ.search(query);

// Update
const updated = await client.happ.update(id, changes);

// Delete (soft delete in most cases)
await client.happ.archive(id);

// Cross-hApp
const result = await client.bridge.call({ target, method, payload });

// Events
client.bridge.on('event.type', handler);
client.bridge.off('event.type', handler);
```

---

## Troubleshooting

### Common Issues

**Issue**: Bridge calls timing out
```bash
# Check bridge status
mycelix status bridge

# Verify target hApp is running
mycelix status <target-happ>

# Check conductor logs
mycelix logs --follow
```

**Issue**: Trust verification failing
```bash
# Check MATL status
mycelix matl status

# View trust network
mycelix matl graph --agent <agent-id>

# Rebuild trust indices
mycelix matl reindex
```

**Issue**: Stage interface not loading
```bash
# Check Spiral integration
mycelix spiral status

# View agent stage profile
mycelix spiral profile --agent <agent-id>

# Reset stage discovery
mycelix spiral reset-discovery --agent <agent-id>
```

### Debug Mode

```bash
# Enable debug logging
export MYCELIX_LOG=debug
mycelix dev

# Trace specific hApp
mycelix trace my-happ --duration 60s --output trace.json

# Profile performance
mycelix profile --happ my-happ --scenario tests/simulation/perf.yaml
```

---

## Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run full test suite: `mycelix test --all`
5. Submit pull request

### Code Standards

- Rust: Follow Rust API guidelines, use `cargo fmt` and `cargo clippy`
- TypeScript: Follow Mycelix style guide, use ESLint config
- Documentation: Update relevant docs with any API changes
- Tests: Maintain >80% coverage, include simulation tests for behavioral features

### Review Process

1. Automated checks (lint, test, security scan)
2. Code review by maintainer
3. Integration test on staging
4. Community review for significant changes
5. Merge and deploy

---

*Document Version: 1.0*
*Last Updated: 2025*
