# Developer Guide

*Building on Epistemic Markets*

---

> "The best platform is one that makes building the next platform possible."

---

## Introduction

Welcome, developer. This guide will help you:

1. **Understand** the architecture and patterns
2. **Set up** your development environment
3. **Build** applications and integrations
4. **Contribute** to the core platform
5. **Deploy** your creations

Whether you're building a custom frontend, integrating with external systems, or contributing to core development, this guide has you covered.

---

## Part I: Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Applications                         │
│  (Web, Mobile, Desktop, CLI, Bots)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TypeScript SDK                                 │
│  @mycelix/epistemic-markets-sdk                                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Holochain Conductor                            │
│  Agent-centric P2P network                                         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DNA (hApp)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────┐ ┌────────┐ │
│ │ markets │ │ predictions  │ │ resolution │ │ scoring │ │ bridge │ │
│ └─────────┘ └──────────────┘ └────────────┘ └─────────┘ └────────┘ │
│ ┌──────────────────┐                                               │
│ │ question_markets │                                               │
│ └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Mycelix Ecosystem                                │
│  MATL | Knowledge | Governance | Identity | Finance                │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Zomes** | Rust | Core business logic |
| **SDK** | TypeScript | Client-side library |
| **Conductor** | Holochain | P2P networking, validation |
| **DHT** | Holochain | Distributed data storage |
| **Bridge** | Zome calls | Cross-hApp communication |

### Data Flow

```
User Action → SDK → Conductor → Zome Function → Validation → DHT Storage
     ↓                                                              │
 UI Update ← SDK ← Conductor ← Signal/Response ← Validation ←──────┘
```

---

## Part II: Development Environment

### Prerequisites

```bash
# Holochain Development Kit
curl --proto '=https' --tlsv1.2 -sSf https://holochain.github.io/install | bash

# Or with Nix (recommended)
nix-shell https://holochain.love

# Node.js (for SDK development)
node --version  # >= 18.0.0

# Rust (for zome development)
rustc --version  # >= 1.70.0
```

### Project Setup

```bash
# Clone the repository
git clone https://github.com/mycelix/epistemic-markets
cd epistemic-markets

# Install dependencies
npm install

# Build zomes
npm run build:zomes

# Build SDK
npm run build:sdk

# Run tests
npm test

# Start development conductor
npm run dev
```

### Directory Structure

```
epistemic-markets/
├── zomes/                    # Rust zomes (core logic)
│   ├── markets/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs       # Entry point
│   │       ├── entry.rs     # Entry definitions
│   │       ├── validation.rs # Validation callbacks
│   │       └── handlers.rs   # Zome functions
│   ├── predictions/
│   ├── resolution/
│   ├── scoring/
│   ├── question_markets/
│   └── markets_bridge/
│
├── sdk-ts/                   # TypeScript SDK
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── index.ts         # Main export
│       ├── client.ts        # Client classes
│       ├── types.ts         # Type definitions
│       └── utils.ts         # Helper functions
│
├── tests/                    # Integration tests
│   ├── integration.test.ts
│   └── fixtures/
│
├── dna.yaml                  # DNA configuration
├── happ.yaml                 # hApp bundle config
└── workdir/                  # Development configs
```

### IDE Setup

**VS Code (Recommended)**:
```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.features": ["mock"],
  "rust-analyzer.checkOnSave.command": "clippy",
  "typescript.tsdk": "node_modules/typescript/lib"
}

// Recommended extensions:
// - rust-analyzer
// - ESLint
// - Prettier
// - Holochain Dev Tools (if available)
```

---

## Part III: Working with Zomes

### Zome Structure

Each zome follows this pattern:

```rust
// lib.rs
use hdk::prelude::*;

mod entry;
mod validation;
mod handlers;

pub use entry::*;
pub use handlers::*;

// Entry point
entry_defs![
    Market::entry_def(),
    Prediction::entry_def()
];

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Initialization logic
    Ok(InitCallbackResult::Pass)
}
```

### Defining Entries

```rust
// entry.rs
use hdk::prelude::*;

#[hdk_entry_helper]
#[derive(Clone)]
pub struct Market {
    pub question: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub epistemic_position: EpistemicPosition,
    pub creator: AgentPubKey,
    pub created_at: Timestamp,
    pub closes_at: Timestamp,
    pub resolution_deadline: Timestamp,
    pub status: MarketStatus,
    pub tags: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct EpistemicPosition {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum MarketStatus {
    Open,
    Closed,
    Resolving,
    Resolved { outcome: String },
    Disputed,
    Voided { reason: String },
}
```

### Implementing Handlers

```rust
// handlers.rs
use hdk::prelude::*;
use crate::entry::*;

#[hdk_extern]
pub fn create_market(input: CreateMarketInput) -> ExternResult<EntryHash> {
    // Validate input
    if input.question.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Question cannot be empty".to_string()
        )));
    }

    // Build entry
    let market = Market {
        question: input.question,
        description: input.description,
        outcomes: input.outcomes,
        epistemic_position: input.epistemic_position,
        creator: agent_info()?.agent_latest_pubkey,
        created_at: sys_time()?,
        closes_at: Timestamp(input.closes_at as i64),
        resolution_deadline: Timestamp(input.resolution_deadline as i64),
        status: MarketStatus::Open,
        tags: input.tags.unwrap_or_default(),
    };

    // Create entry
    let entry_hash = create_entry(&market)?;

    // Add to index (link from known anchor)
    let markets_anchor = anchor("markets".into(), "all".into())?;
    create_link(
        markets_anchor,
        entry_hash.clone(),
        LinkType::MarketIndex,
        ()
    )?;

    // Emit signal for real-time updates
    emit_signal(&Signal::MarketCreated {
        hash: entry_hash.clone(),
        market: market.clone(),
    })?;

    Ok(entry_hash)
}

#[hdk_extern]
pub fn get_market(hash: EntryHash) -> ExternResult<Option<Market>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let market: Market = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(e))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Entry not found".to_string()
                )))?;
            Ok(Some(market))
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn list_markets(_: ()) -> ExternResult<Vec<MarketWithHash>> {
    let markets_anchor = anchor("markets".into(), "all".into())?;
    let links = get_links(markets_anchor, Some(LinkType::MarketIndex), None)?;

    let markets: Vec<MarketWithHash> = links
        .into_iter()
        .filter_map(|link| {
            let hash = link.target.into_entry_hash()?;
            let market = get_market(hash.clone()).ok()??;
            Some(MarketWithHash { hash, market })
        })
        .collect();

    Ok(markets)
}
```

### Validation Callbacks

```rust
// validation.rs
use hdk::prelude::*;
use crate::entry::*;

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry { entry, .. } => validate_entry(entry),
        Op::RegisterCreateLink { create_link, .. } => validate_link(create_link),
        Op::RegisterDeleteLink { delete_link, .. } => validate_delete_link(delete_link),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_entry(entry: Entry) -> ExternResult<ValidateCallbackResult> {
    match entry {
        Entry::App(bytes) => {
            // Try to deserialize as Market
            if let Ok(market) = Market::try_from(bytes.clone()) {
                return validate_market(market);
            }

            // Try other types...
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_market(market: Market) -> ExternResult<ValidateCallbackResult> {
    // Question not empty
    if market.question.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Question cannot be empty".to_string()
        ));
    }

    // At least 2 outcomes
    if market.outcomes.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Market must have at least 2 outcomes".to_string()
        ));
    }

    // Closes before resolution deadline
    if market.closes_at >= market.resolution_deadline {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution deadline must be after close time".to_string()
        ));
    }

    // Closes in the future
    let now = sys_time()?;
    if market.closes_at <= now {
        return Ok(ValidateCallbackResult::Invalid(
            "Market must close in the future".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
```

### Cross-Zome Calls

```rust
// In resolution zome, calling MATL
use crate::bridge::*;

pub fn get_matl_score(agent: AgentPubKey) -> ExternResult<MatlScore> {
    let response: MatlScore = call(
        CallTargetCell::OtherRole("matl".into()),
        ZomeName::new("trust"),
        FunctionName::new("get_composite_score"),
        None,
        agent,
    )?;

    Ok(response)
}
```

---

## Part IV: Working with the SDK

### Installation

```bash
npm install @mycelix/epistemic-markets-sdk
```

### Basic Usage

```typescript
import { EpistemicMarketsClient } from "@mycelix/epistemic-markets-sdk";
import { AppWebsocket } from "@holochain/client";

// Connect to conductor
const appWs = await AppWebsocket.connect("ws://localhost:8888");
const appInfo = await appWs.appInfo({ installed_app_id: "epistemic-markets" });

// Initialize client
const client = new EpistemicMarketsClient(appWs, appInfo.cell_info);

// Create a market
const marketHash = await client.markets.createMarket({
  question: "Will it rain tomorrow?",
  description: "Resolves YES if any measurable precipitation falls in San Francisco",
  outcomes: ["Yes", "No"],
  epistemicPosition: {
    empirical: "Measurable",
    normative: "Universal",
    materiality: "Ephemeral",
  },
  closesAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
  resolutionDeadline: Date.now() + 48 * 60 * 60 * 1000, // 48 hours
});

// Make a prediction
await client.predictions.submitPrediction({
  marketId: marketHash,
  outcome: "Yes",
  confidence: 0.65,
  stake: {
    reputation: {
      domains: ["weather"],
      stakePercentage: 0.02,
    },
  },
  reasoning: "Satellite imagery shows incoming storm system",
});

// Listen for signals
client.on("marketCreated", (signal) => {
  console.log("New market:", signal.market);
});

client.on("predictionSubmitted", (signal) => {
  console.log("New prediction:", signal.prediction);
});
```

### SDK Architecture

```typescript
// Main client composes sub-clients
class EpistemicMarketsClient {
  public readonly markets: MarketsClient;
  public readonly predictions: PredictionsClient;
  public readonly questions: QuestionMarketsClient;
  public readonly resolution: ResolutionClient;
  public readonly scoring: ScoringClient;
  public readonly bridge: BridgeClient;
  public readonly health: HealthClient;

  private eventEmitter: EventEmitter;

  constructor(appClient: AppClient, cellInfo: CellInfo) {
    const zomeCaller = new ZomeCaller(appClient, cellInfo);

    this.markets = new MarketsClient(zomeCaller);
    this.predictions = new PredictionsClient(zomeCaller);
    // ...

    this.setupSignalHandler(appClient);
  }

  on<T extends SignalType>(type: T, handler: SignalHandler<T>): void {
    this.eventEmitter.on(type, handler);
  }

  off<T extends SignalType>(type: T, handler: SignalHandler<T>): void {
    this.eventEmitter.off(type, handler);
  }
}
```

### Type Definitions

```typescript
// Types match Rust structures
export interface Market {
  question: string;
  description: string;
  outcomes: string[];
  epistemicPosition: EpistemicPosition;
  creator: AgentPubKey;
  createdAt: number;
  closesAt: number;
  resolutionDeadline: number;
  status: MarketStatus;
  tags: string[];
}

export interface EpistemicPosition {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
}

export type EmpiricalLevel =
  | "Subjective"
  | "Testimonial"
  | "PrivateVerify"
  | "Cryptographic"
  | "Measurable";

export type MarketStatus =
  | { type: "Open" }
  | { type: "Closed" }
  | { type: "Resolving" }
  | { type: "Resolved"; outcome: string }
  | { type: "Disputed" }
  | { type: "Voided"; reason: string };
```

### Error Handling

```typescript
import { EpistemicMarketsError, ErrorCode } from "@mycelix/epistemic-markets-sdk";

try {
  await client.markets.createMarket(input);
} catch (error) {
  if (error instanceof EpistemicMarketsError) {
    switch (error.code) {
      case ErrorCode.VALIDATION_FAILED:
        console.error("Invalid market data:", error.details);
        break;
      case ErrorCode.UNAUTHORIZED:
        console.error("Not authorized to create markets");
        break;
      case ErrorCode.NETWORK_ERROR:
        console.error("Network issue, retry later");
        break;
      default:
        console.error("Unknown error:", error.message);
    }
  }
}
```

---

## Part V: Building Applications

### Web Application (React)

```tsx
// hooks/useMarkets.ts
import { useState, useEffect } from "react";
import { useEpistemicMarkets } from "../context/EpistemicMarketsContext";
import type { Market, MarketWithHash } from "@mycelix/epistemic-markets-sdk";

export function useMarkets() {
  const client = useEpistemicMarkets();
  const [markets, setMarkets] = useState<MarketWithHash[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMarkets = async () => {
      const data = await client.markets.listOpenMarkets();
      setMarkets(data);
      setLoading(false);
    };

    fetchMarkets();

    // Listen for new markets
    const handleNewMarket = (signal: { hash: string; market: Market }) => {
      setMarkets((prev) => [...prev, { hash: signal.hash, market: signal.market }]);
    };

    client.on("marketCreated", handleNewMarket);

    return () => {
      client.off("marketCreated", handleNewMarket);
    };
  }, [client]);

  return { markets, loading };
}

// components/MarketList.tsx
import { useMarkets } from "../hooks/useMarkets";

export function MarketList() {
  const { markets, loading } = useMarkets();

  if (loading) return <Spinner />;

  return (
    <div className="market-list">
      {markets.map(({ hash, market }) => (
        <MarketCard key={hash} hash={hash} market={market} />
      ))}
    </div>
  );
}
```

### CLI Tool

```typescript
// cli/src/index.ts
import { Command } from "commander";
import { EpistemicMarketsClient } from "@mycelix/epistemic-markets-sdk";
import { connect } from "./connection";

const program = new Command();

program
  .name("em-cli")
  .description("Epistemic Markets CLI")
  .version("1.0.0");

program
  .command("markets")
  .description("List open markets")
  .option("-t, --tag <tag>", "Filter by tag")
  .action(async (options) => {
    const client = await connect();
    const markets = options.tag
      ? await client.markets.searchByTag(options.tag)
      : await client.markets.listOpenMarkets();

    console.table(
      markets.map((m) => ({
        Question: m.market.question.slice(0, 50),
        Outcomes: m.market.outcomes.join(" | "),
        Closes: new Date(m.market.closesAt).toLocaleString(),
      }))
    );
  });

program
  .command("predict <marketHash>")
  .description("Make a prediction")
  .requiredOption("-o, --outcome <outcome>", "Predicted outcome")
  .requiredOption("-c, --confidence <confidence>", "Confidence (0-1)")
  .option("-r, --reasoning <reasoning>", "Your reasoning")
  .action(async (marketHash, options) => {
    const client = await connect();

    await client.predictions.submitPrediction({
      marketId: marketHash,
      outcome: options.outcome,
      confidence: parseFloat(options.confidence),
      reasoning: options.reasoning || "",
      stake: { reputation: { domains: [], stakePercentage: 0.01 } },
    });

    console.log("Prediction submitted successfully!");
  });

program.parse();
```

### Bot/Automation

```typescript
// bot/src/index.ts
import { EpistemicMarketsClient } from "@mycelix/epistemic-markets-sdk";

class PredictionBot {
  private client: EpistemicMarketsClient;

  constructor(client: EpistemicMarketsClient) {
    this.client = client;
  }

  async start() {
    // Listen for new markets
    this.client.on("marketCreated", async (signal) => {
      const analysis = await this.analyzeMarket(signal.market);
      if (analysis.shouldPredict) {
        await this.makePrediction(signal.hash, analysis);
      }
    });

    // Listen for resolution opportunities
    this.client.on("resolutionStarted", async (signal) => {
      if (await this.canProvideOracle(signal.marketId)) {
        await this.submitOracleVote(signal);
      }
    });

    console.log("Bot started, listening for opportunities...");
  }

  private async analyzeMarket(market: Market): Promise<Analysis> {
    // Your analysis logic here
    // Could use external data sources, ML models, etc.
    return {
      shouldPredict: true,
      predictedOutcome: "Yes",
      confidence: 0.72,
      reasoning: "Based on historical data...",
    };
  }

  private async makePrediction(marketHash: string, analysis: Analysis) {
    await this.client.predictions.submitPrediction({
      marketId: marketHash,
      outcome: analysis.predictedOutcome,
      confidence: analysis.confidence,
      reasoning: analysis.reasoning,
      stake: {
        reputation: {
          domains: ["automated"],
          stakePercentage: 0.01,
        },
      },
    });
  }
}
```

---

## Part VI: Testing

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use holochain::test_utils::*;

    #[test]
    fn test_market_validation() {
        let valid_market = Market {
            question: "Test question?".to_string(),
            outcomes: vec!["Yes".to_string(), "No".to_string()],
            // ...
        };

        assert!(validate_market(valid_market).is_ok());
    }

    #[test]
    fn test_market_requires_question() {
        let invalid_market = Market {
            question: "".to_string(),
            // ...
        };

        let result = validate_market(invalid_market);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
```

### Integration Tests (TypeScript)

```typescript
// tests/markets.test.ts
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { EpistemicMarketsClient } from "@mycelix/epistemic-markets-sdk";
import { createTestConductor, TestConductor } from "./helpers";

describe("Markets", () => {
  let conductor: TestConductor;
  let client: EpistemicMarketsClient;

  beforeAll(async () => {
    conductor = await createTestConductor();
    client = new EpistemicMarketsClient(conductor.appClient, conductor.cellInfo);
  });

  afterAll(async () => {
    await conductor.shutdown();
  });

  it("should create a market", async () => {
    const hash = await client.markets.createMarket({
      question: "Test market?",
      description: "A test",
      outcomes: ["Yes", "No"],
      epistemicPosition: {
        empirical: "Subjective",
        normative: "Personal",
        materiality: "Ephemeral",
      },
      closesAt: Date.now() + 100000,
      resolutionDeadline: Date.now() + 200000,
    });

    expect(hash).toBeTruthy();
  });

  it("should retrieve a created market", async () => {
    const hash = await client.markets.createMarket({
      question: "Retrievable market?",
      // ...
    });

    const market = await client.markets.getMarket(hash);

    expect(market).toBeTruthy();
    expect(market.question).toBe("Retrievable market?");
  });

  it("should list all open markets", async () => {
    // Create a few markets
    await client.markets.createMarket({ /* ... */ });
    await client.markets.createMarket({ /* ... */ });

    const markets = await client.markets.listOpenMarkets();

    expect(markets.length).toBeGreaterThanOrEqual(2);
  });
});
```

### End-to-End Tests

```typescript
// e2e/prediction-flow.test.ts
import { test, expect } from "@playwright/test";

test("complete prediction flow", async ({ page }) => {
  // Login
  await page.goto("/login");
  await page.fill('[name="seed"]', "test-seed-phrase");
  await page.click('button[type="submit"]');

  // Navigate to markets
  await page.click('a[href="/markets"]');
  await expect(page.locator("h1")).toHaveText("Open Markets");

  // Create a market
  await page.click('button:has-text("Create Market")');
  await page.fill('[name="question"]', "Will this test pass?");
  await page.fill('[name="description"]', "A test market");
  await page.click('button:has-text("Yes/No")');
  await page.click('button:has-text("Create")');

  // Verify market created
  await expect(page.locator(".market-card")).toContainText("Will this test pass?");

  // Make a prediction
  await page.click(".market-card >> text=Will this test pass?");
  await page.click('button:has-text("Predict")');
  await page.click('button:has-text("Yes")');
  await page.fill('[name="confidence"]', "0.8");
  await page.fill('[name="reasoning"]', "Because I wrote the test");
  await page.click('button:has-text("Submit Prediction")');

  // Verify prediction submitted
  await expect(page.locator(".prediction-success")).toBeVisible();
});
```

---

## Part VII: Deployment

### Building for Production

```bash
# Build optimized zomes
CARGO_PROFILE_RELEASE_LTO=true cargo build --release --target wasm32-unknown-unknown

# Build DNA
hc dna pack workdir

# Build hApp
hc app pack workdir

# Build SDK
npm run build --workspace=sdk-ts
```

### Conductor Configuration

```yaml
# conductor-config.yaml
environment_path: ./databases
keystore:
  type: lair_server_legacy_deprecated

admin_interfaces:
  - driver:
      type: websocket
      port: 8888
    allowed_origins: "*"

network:
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.mycelix.net

  bootstrap_service: https://bootstrap.mycelix.net

  tuning_params:
    gossip_loop_iteration_delay_ms: 1000
    default_notify_remote_agent_count: 5
    default_notify_timeout_ms: 1000
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM holochain/holochain:latest

COPY workdir/epistemic-markets.happ /happ/
COPY conductor-config.yaml /config/

EXPOSE 8888

CMD ["holochain", "-c", "/config/conductor-config.yaml"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  conductor:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - holochain-data:/databases
    environment:
      - RUST_LOG=info

  web:
    build: ./web
    ports:
      - "3000:3000"
    depends_on:
      - conductor
    environment:
      - HOLOCHAIN_URL=ws://conductor:8888

volumes:
  holochain-data:
```

---

## Part VIII: Contributing

### Development Workflow

```
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit pull request
6. Address review feedback
7. Merge!
```

### Code Style

**Rust**:
```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

**TypeScript**:
```bash
# Format code
npm run format

# Run linter
npm run lint
```

### Commit Messages

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(markets): add tag filtering for market search
fix(resolution): correct MATL weight calculation
docs(api): update prediction submission examples
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Documentation updated
- [ ] No new warnings
```

---

## Part IX: Resources

### Documentation
- [Holochain Developer Docs](https://developer.holochain.org/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

### Community
- Discord: [Mycelix Developers](https://discord.gg/mycelix)
- Forum: [forum.mycelix.net](https://forum.mycelix.net)
- GitHub Discussions: [github.com/mycelix/epistemic-markets/discussions](https://github.com/mycelix/epistemic-markets/discussions)

### Examples
- [Example Frontend](https://github.com/mycelix/em-example-frontend)
- [Example Bot](https://github.com/mycelix/em-example-bot)
- [Example Integration](https://github.com/mycelix/em-example-integration)

---

## Conclusion

Building on Epistemic Markets means contributing to collective truth-seeking infrastructure. Whether you're creating a new frontend, building an integration, or improving the core platform, you're helping build tools for humanity's epistemic future.

We're excited to see what you build.

---

> "The best code is code that enables others to build what they imagine."

---

*Happy building. May your predictions be well-calibrated.*

---

## Quick Reference

### Common Commands

```bash
# Development
npm run dev                    # Start development conductor
npm run build                  # Build everything
npm run test                   # Run all tests

# Zomes
cargo build --release          # Build zomes
cargo test                     # Test zomes
cargo clippy                   # Lint zomes

# SDK
npm run build:sdk              # Build SDK
npm run test:sdk               # Test SDK
npm run docs:sdk               # Generate SDK docs
```

### Key Files

| File | Purpose |
|------|---------|
| `dna.yaml` | DNA configuration |
| `happ.yaml` | hApp bundle configuration |
| `workdir/dna.yaml` | Development overrides |
| `sdk-ts/src/index.ts` | SDK entry point |
| `zomes/*/src/lib.rs` | Zome entry points |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `HOLOCHAIN_URL` | Conductor WebSocket URL |
| `RUST_LOG` | Logging level |
| `HC_ADMIN_PORT` | Admin interface port |
