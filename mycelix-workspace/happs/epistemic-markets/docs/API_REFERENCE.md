# API Reference

*Complete Technical Documentation for Developers*

---

> "Good APIs are invisible. Great APIs are a joy to use."
> — Developer wisdom

> "We document thoroughly because clarity is kindness."
> — Us

---

## Overview

This document provides complete API documentation for Epistemic Markets. It covers:

1. **TypeScript SDK** - Client-side library for applications
2. **Zome Functions** - Direct Holochain calls
3. **Data Types** - All structures and enums
4. **Events** - Signals and webhooks
5. **Error Codes** - Comprehensive error handling

---

## Part I: TypeScript SDK

### Installation

```bash
npm install @mycelix/epistemic-markets-sdk
# or
yarn add @mycelix/epistemic-markets-sdk
```

### Quick Start

```typescript
import {
  EpistemicMarketsClient,
  MarketConfig,
  PredictionInput,
  StakeConfig
} from "@mycelix/epistemic-markets-sdk";

// Initialize with Holochain app client
const client = new EpistemicMarketsClient(appClient);

// Check connection
const status = await client.health.check();
console.log(status); // { connected: true, version: "1.0.0" }
```

---

### Client Initialization

#### `EpistemicMarketsClient`

```typescript
class EpistemicMarketsClient {
  constructor(appClient: AppClient, options?: ClientOptions);

  // Sub-clients for different domains
  readonly markets: MarketsClient;
  readonly predictions: PredictionsClient;
  readonly questions: QuestionMarketsClient;
  readonly resolution: ResolutionClient;
  readonly scoring: ScoringClient;
  readonly bridge: BridgeClient;
  readonly health: HealthClient;
}

interface ClientOptions {
  // Timeout for zome calls (default: 30000ms)
  timeout?: number;

  // Retry configuration
  retries?: number;
  retryDelay?: number;

  // Logging
  logLevel?: "debug" | "info" | "warn" | "error" | "none";

  // Custom error handler
  onError?: (error: EpistemicError) => void;
}
```

---

### Markets Client

#### `client.markets.createMarket()`

Creates a new prediction market.

```typescript
async createMarket(config: MarketConfig): Promise<EntryHash>

interface MarketConfig {
  // Required fields
  question: string;           // The question being predicted
  description: string;        // Detailed description
  outcomes: string[];         // Possible outcomes (2-10)
  epistemicPosition: EpistemicPosition;
  closesAt: number;           // Unix timestamp
  resolutionDeadline: number; // Unix timestamp

  // Optional fields
  mechanism?: MarketMechanism;
  tags?: string[];
  minStake?: number;
  maxStake?: number;
  creatorStake?: number;      // Initial liquidity
  visibility?: MarketVisibility;
  accessControl?: AccessControl;
  metadata?: Record<string, unknown>;
}

interface EpistemicPosition {
  empirical: "Subjective" | "Testimonial" | "PrivateVerify" | "Cryptographic" | "Measurable";
  normative: "Personal" | "Communal" | "Network" | "Universal";
  materiality: "Ephemeral" | "Temporal" | "Persistent" | "Foundational";
}

interface MarketMechanism {
  type: "LMSR" | "OrderBook" | "Parimutuel";
  liquidityParameter?: number;  // For LMSR (default: 100)
  subsidyPool?: number;         // Initial liquidity subsidy
}
```

**Example:**

```typescript
const marketId = await client.markets.createMarket({
  question: "Will SpaceX land humans on Mars by 2030?",
  description: "Resolves YES if SpaceX successfully lands at least one human on Mars...",
  outcomes: ["Yes", "No"],
  epistemicPosition: {
    empirical: "Measurable",
    normative: "Universal",
    materiality: "Persistent",
  },
  mechanism: { type: "LMSR", liquidityParameter: 100, subsidyPool: 1000 },
  closesAt: Date.parse("2029-12-31T23:59:59Z"),
  resolutionDeadline: Date.parse("2030-03-31T23:59:59Z"),
  tags: ["space", "spacex", "mars"],
});
```

**Returns:** `EntryHash` - The unique identifier for the created market

**Errors:**
- `InvalidQuestion` - Question is empty or too long
- `InvalidOutcomes` - Fewer than 2 or more than 10 outcomes
- `InvalidTimestamps` - Close time after resolution deadline
- `InsufficientStake` - Creator stake below minimum

---

#### `client.markets.getMarket()`

Retrieves a market by its hash.

```typescript
async getMarket(marketId: EntryHash): Promise<Market | null>

interface Market {
  id: EntryHash;
  question: string;
  description: string;
  outcomes: string[];
  epistemicPosition: EpistemicPosition;
  mechanism: MarketMechanism;
  status: MarketStatus;
  createdAt: number;
  createdBy: AgentPubKey;
  closesAt: number;
  resolutionDeadline: number;
  tags: string[];

  // Computed fields
  currentPrices: number[];        // Current probability per outcome
  totalVolume: number;            // Total trading volume
  predictionCount: number;        // Number of predictions
  liquidityDepth: number;         // Available liquidity
}

type MarketStatus =
  | "Draft"
  | "Open"
  | "Closed"
  | "Resolving"
  | "Resolved"
  | "Disputed"
  | "Voided";
```

---

#### `client.markets.listMarkets()`

Lists markets with filtering and pagination.

```typescript
async listMarkets(options?: ListOptions): Promise<PaginatedResult<MarketSummary>>

interface ListOptions {
  status?: MarketStatus | MarketStatus[];
  tags?: string[];
  createdBy?: AgentPubKey;
  createdAfter?: number;
  createdBefore?: number;
  closesAfter?: number;
  closesBefore?: number;
  minVolume?: number;
  sortBy?: "created" | "closes" | "volume" | "predictions";
  sortOrder?: "asc" | "desc";
  limit?: number;       // Default: 50, max: 200
  offset?: number;      // For pagination
}

interface PaginatedResult<T> {
  items: T[];
  total: number;
  hasMore: boolean;
  nextOffset?: number;
}

interface MarketSummary {
  id: EntryHash;
  question: string;
  outcomes: string[];
  status: MarketStatus;
  currentPrices: number[];
  predictionCount: number;
  totalVolume: number;
  closesAt: number;
  tags: string[];
}
```

---

#### `client.markets.searchMarkets()`

Full-text search across markets.

```typescript
async searchMarkets(query: string, options?: SearchOptions): Promise<MarketSummary[]>

interface SearchOptions {
  fields?: ("question" | "description" | "tags")[];
  status?: MarketStatus[];
  limit?: number;
}
```

---

#### `client.markets.closeMarket()`

Closes a market for new predictions (creator or admin only).

```typescript
async closeMarket(marketId: EntryHash): Promise<void>
```

---

#### `client.markets.getMarketStats()`

Gets detailed statistics for a market.

```typescript
async getMarketStats(marketId: EntryHash): Promise<MarketStats>

interface MarketStats {
  marketId: EntryHash;

  // Volume stats
  totalVolume: number;
  volumeByOutcome: number[];
  volumeHistory: TimeSeriesPoint[];

  // Prediction stats
  totalPredictions: number;
  uniquePredictors: number;
  predictionsByOutcome: number[];
  averageConfidence: number[];

  // Price history
  priceHistory: PricePoint[];

  // Stake breakdown
  monetaryStakeTotal: number;
  reputationStakeTotal: number;
  commitmentCount: number;
}

interface PricePoint {
  timestamp: number;
  prices: number[];
}
```

---

### Predictions Client

#### `client.predictions.submitPrediction()`

Submits a prediction to a market.

```typescript
async submitPrediction(input: PredictionInput): Promise<EntryHash>

interface PredictionInput {
  marketId: EntryHash;
  outcome: string;            // Must match one of market outcomes
  confidence: number;         // 0-1 (e.g., 0.75 for 75%)
  stake: StakeConfig;
  reasoning: string;          // Required length depends on stake

  // Optional
  evidence?: EvidenceLink[];
  visibility?: "Private" | "Limited" | "Community" | "Public";
  updateOf?: EntryHash;       // If updating previous prediction
}

interface StakeConfig {
  monetary?: {
    amount: number;
    currency?: string;        // Default: "HAM"
  };

  reputation?: {
    domains: string[];
    stakePercentage: number;  // 0-1 (e.g., 0.05 for 5%)
    confidenceMultiplier?: number;
  };

  social?: {
    visibility: "Private" | "Limited" | "Community" | "Public";
    identityLink?: string;
  };

  commitment?: {
    ifCorrect: CommitmentAction[];
    ifWrong: CommitmentAction[];
  };

  time?: {
    researchHours: number;
    evidenceSubmitted: EntryHash[];
  };
}

interface CommitmentAction {
  type: "Donation" | "PublicStatement" | "Action" | "Custom";
  description: string;
  verifiable: boolean;
  verificationMethod?: string;
}

interface EvidenceLink {
  type: "URL" | "Document" | "KnowledgeGraphClaim" | "Citation";
  reference: string;
  description?: string;
}
```

**Example:**

```typescript
const predictionId = await client.predictions.submitPrediction({
  marketId: marketId,
  outcome: "Yes",
  confidence: 0.72,
  stake: {
    monetary: { amount: 100 },
    reputation: {
      domains: ["space", "technology"],
      stakePercentage: 0.05,
    },
    commitment: {
      ifWrong: [{
        type: "Donation",
        description: "Donate $50 to Mars Society",
        verifiable: true,
        verificationMethod: "Transaction receipt",
      }],
      ifCorrect: [],
    },
  },
  reasoning: `SpaceX has demonstrated consistent progress with Starship...`,
  evidence: [
    { type: "URL", reference: "https://spacex.com/starship", description: "Official Starship page" },
  ],
  visibility: "Public",
});
```

**Validation Rules:**
- `reasoning` minimum length: 50 chars (low stake), 150 chars (medium), 300 chars (high)
- `confidence` must be between 0.01 and 0.99
- `stake.monetary.amount` must be within market's min/max
- `stake.reputation.stakePercentage` max is 0.25 (25%)

---

#### `client.predictions.getPrediction()`

Retrieves a prediction by hash.

```typescript
async getPrediction(predictionId: EntryHash): Promise<Prediction | null>

interface Prediction {
  id: EntryHash;
  marketId: EntryHash;
  predictor: AgentPubKey;
  outcome: string;
  confidence: number;
  stake: ResolvedStake;
  reasoning: string;
  evidence: EvidenceLink[];
  visibility: string;
  createdAt: number;
  updatedAt?: number;
  updateHistory?: EntryHash[];

  // Computed after resolution
  wasCorrect?: boolean;
  reward?: RewardBreakdown;
  calibrationScore?: number;
}
```

---

#### `client.predictions.updatePrediction()`

Updates an existing prediction (before market closes).

```typescript
async updatePrediction(input: PredictionUpdateInput): Promise<EntryHash>

interface PredictionUpdateInput {
  predictionId: EntryHash;
  newOutcome?: string;
  newConfidence?: number;
  additionalReasoning: string;  // Required: explain the update
  additionalStake?: StakeConfig;
}
```

---

#### `client.predictions.withdrawPrediction()`

Withdraws a prediction (partial stake penalty may apply).

```typescript
async withdrawPrediction(predictionId: EntryHash, reason: string): Promise<WithdrawalResult>

interface WithdrawalResult {
  refundedStake: number;
  penaltyApplied: number;
  reputationImpact: number;
}
```

---

#### `client.predictions.getMyPredictions()`

Gets the current user's predictions.

```typescript
async getMyPredictions(options?: MyPredictionsOptions): Promise<PaginatedResult<Prediction>>

interface MyPredictionsOptions {
  marketId?: EntryHash;
  status?: "pending" | "resolved" | "withdrawn";
  limit?: number;
  offset?: number;
}
```

---

#### `client.predictions.getPredictionsForMarket()`

Gets all predictions for a market (respecting visibility settings).

```typescript
async getPredictionsForMarket(
  marketId: EntryHash,
  options?: PredictionListOptions
): Promise<PaginatedResult<Prediction>>

interface PredictionListOptions {
  outcome?: string;
  minConfidence?: number;
  maxConfidence?: number;
  minStake?: number;
  sortBy?: "confidence" | "stake" | "created";
  limit?: number;
  offset?: number;
}
```

---

### Question Markets Client

#### `client.questions.proposeQuestion()`

Proposes a new question for the question market.

```typescript
async proposeQuestion(input: QuestionProposal): Promise<EntryHash>

interface QuestionProposal {
  questionText: string;
  context: string;
  suggestedOutcomes?: string[];
  suggestedEpistemicPosition?: EpistemicPosition;
  domains: string[];
  initialCuriosityStake?: number;
}
```

---

#### `client.questions.signalCuriosity()`

Signals lightweight interest in a question.

```typescript
async signalCuriosity(
  questionId: EntryHash,
  reason?: string
): Promise<void>
```

---

#### `client.questions.buyShares()`

Buys value shares in a question (betting it's worth answering).

```typescript
async buyShares(
  questionId: EntryHash,
  amount: number,
  maxPrice?: number,
  reason?: string
): Promise<SharePurchase>

interface SharePurchase {
  questionId: EntryHash;
  sharesBought: number;
  totalCost: number;
  newPrice: number;
}
```

---

#### `client.questions.sellShares()`

Sells value shares in a question.

```typescript
async sellShares(
  questionId: EntryHash,
  shares: number,
  minPrice?: number
): Promise<ShareSale>
```

---

#### `client.questions.getQuestionMarket()`

Gets a question market by hash.

```typescript
async getQuestionMarket(questionId: EntryHash): Promise<QuestionMarket | null>

interface QuestionMarket {
  id: EntryHash;
  questionText: string;
  context: string;
  proposer: AgentPubKey;
  proposedAt: number;
  domains: string[];

  // Market state
  currentValue: number;
  totalShares: number;
  curiositySignals: number;
  shareholderCount: number;

  // Spawning info
  spawningThreshold: number;
  spawnedMarketId?: EntryHash;
  status: "Active" | "Spawned" | "Expired" | "Rejected";
}
```

---

#### `client.questions.listQuestionMarkets()`

Lists question markets with filtering.

```typescript
async listQuestionMarkets(options?: QuestionListOptions): Promise<PaginatedResult<QuestionMarket>>

interface QuestionListOptions {
  status?: QuestionStatus[];
  domains?: string[];
  minValue?: number;
  proposedBy?: AgentPubKey;
  sortBy?: "value" | "signals" | "proposed" | "trending";
  limit?: number;
  offset?: number;
}
```

---

#### `client.questions.getTopQuestions()`

Gets the highest-value questions.

```typescript
async getTopQuestions(
  count?: number,
  domains?: string[]
): Promise<QuestionMarket[]>
```

---

### Resolution Client

#### `client.resolution.startResolution()`

Initiates the resolution process for a market (authorized users only).

```typescript
async startResolution(
  marketId: EntryHash,
  proposedOutcome?: string
): Promise<ResolutionProcess>

interface ResolutionProcess {
  id: EntryHash;
  marketId: EntryHash;
  status: ResolutionStatus;
  phase: ResolutionPhase;
  initiatedAt: number;
  initiatedBy: AgentPubKey;
  proposedOutcome?: string;
  oraclePool: OracleInfo[];
  votes: OracleVote[];
  currentConsensus?: ConsensusState;
  disputeWindow: { opens: number; closes: number };
}

type ResolutionStatus = "InProgress" | "ConsensusReached" | "Disputed" | "Finalized" | "Failed";
type ResolutionPhase = "OracleSelection" | "Voting" | "Consensus" | "DisputeWindow" | "Finalization";
```

---

#### `client.resolution.submitOracleVote()`

Submits an oracle vote (qualified oracles only).

```typescript
async submitOracleVote(input: OracleVoteInput): Promise<EntryHash>

interface OracleVoteInput {
  resolutionId: EntryHash;
  outcome: string;
  confidence: number;
  reasoning: string;
  evidence?: EvidenceLink[];
}
```

---

#### `client.resolution.disputeResolution()`

Disputes a resolution (requires stake).

```typescript
async disputeResolution(input: DisputeInput): Promise<EntryHash>

interface DisputeInput {
  resolutionId: EntryHash;
  disputedOutcome: string;
  proposedOutcome: string;
  reasoning: string;
  evidence: EvidenceLink[];
  stake: number;  // Required dispute stake
}
```

---

#### `client.resolution.getResolutionProcess()`

Gets the current state of a resolution process.

```typescript
async getResolutionProcess(resolutionId: EntryHash): Promise<ResolutionProcess | null>
```

---

#### `client.resolution.finalizeResolution()`

Finalizes a resolution and triggers payouts (after dispute window).

```typescript
async finalizeResolution(resolutionId: EntryHash): Promise<FinalizationResult>

interface FinalizationResult {
  resolutionId: EntryHash;
  finalOutcome: string;
  payoutsProcessed: number;
  totalDistributed: number;
  wisdomSeedsCreated: number;
}
```

---

### Scoring Client

#### `client.scoring.getCalibration()`

Gets calibration metrics for a user.

```typescript
async getCalibration(
  agent?: AgentPubKey,  // Default: current user
  options?: CalibrationOptions
): Promise<CalibrationMetrics>

interface CalibrationOptions {
  domains?: string[];
  timeRange?: { from: number; to: number };
  minPredictions?: number;
}

interface CalibrationMetrics {
  agent: AgentPubKey;

  // Overall calibration
  brierScore: number;           // 0 = perfect, 1 = worst
  calibrationCurve: CalibrationPoint[];
  overconfidenceIndex: number;  // Positive = overconfident

  // Breakdown
  totalPredictions: number;
  resolvedPredictions: number;
  correctPredictions: number;
  accuracyRate: number;

  // By confidence bucket
  bucketStats: BucketStat[];

  // By domain
  domainStats: Record<string, DomainCalibration>;
}

interface CalibrationPoint {
  predictedProbability: number;
  actualFrequency: number;
  count: number;
}

interface BucketStat {
  bucket: string;           // e.g., "60-70%"
  count: number;
  correctRate: number;
  calibrationError: number;
}
```

---

#### `client.scoring.getBrierScore()`

Gets Brier score for specific predictions or timeframe.

```typescript
async getBrierScore(options?: BrierOptions): Promise<BrierResult>

interface BrierOptions {
  agent?: AgentPubKey;
  marketIds?: EntryHash[];
  domains?: string[];
  timeRange?: { from: number; to: number };
}

interface BrierResult {
  score: number;
  decomposition: {
    reliability: number;      // Calibration component
    resolution: number;       // Discrimination component
    uncertainty: number;      // Base rate component
  };
  predictionCount: number;
}
```

---

#### `client.scoring.getLeaderboard()`

Gets calibration leaderboard.

```typescript
async getLeaderboard(options?: LeaderboardOptions): Promise<LeaderboardEntry[]>

interface LeaderboardOptions {
  domain?: string;
  timeRange?: { from: number; to: number };
  metric?: "brier" | "accuracy" | "calibration" | "matl";
  limit?: number;
}

interface LeaderboardEntry {
  rank: number;
  agent: AgentPubKey;
  displayName?: string;
  score: number;
  predictionCount: number;
  accuracyRate: number;
  matlScore: number;
}
```

---

#### `client.scoring.getWisdomSeeds()`

Gets wisdom seeds, optionally filtered.

```typescript
async getWisdomSeeds(options?: WisdomSeedOptions): Promise<PaginatedResult<WisdomSeed>>

interface WisdomSeedOptions {
  marketId?: EntryHash;
  author?: AgentPubKey;
  domains?: string[];
  minRating?: number;
  sortBy?: "rating" | "created" | "citations";
  limit?: number;
  offset?: number;
}

interface WisdomSeed {
  id: EntryHash;
  marketId: EntryHash;
  author: AgentPubKey;
  content: string;
  context: string;
  createdAt: number;
  rating: number;
  citationCount: number;
  verified: boolean;
  tags: string[];
}
```

---

#### `client.scoring.plantWisdomSeed()`

Creates a wisdom seed for a resolved market.

```typescript
async plantWisdomSeed(input: WisdomSeedInput): Promise<EntryHash>

interface WisdomSeedInput {
  marketId: EntryHash;
  content: string;
  context: string;
  forFuturePredictors: string;
  tags?: string[];
  references?: EntryHash[];
}
```

---

### Bridge Client

#### `client.bridge.submitPredictionRequest()`

Requests a prediction market from another hApp.

```typescript
async submitPredictionRequest(input: BridgeRequest): Promise<EntryHash>

interface BridgeRequest {
  sourceHapp: string;
  question: string;
  context: Record<string, unknown>;
  resolutionCriteria: ResolutionCriteria;
  urgency: "Low" | "Medium" | "High" | "Critical";
  bounty?: number;
  deadline?: number;
}

interface ResolutionCriteria {
  type: "OnChainEvent" | "OracleVote" | "DataFeed" | "Manual";
  eventType?: string;
  eventSource?: string;
  dataFeedId?: string;
  threshold?: number;
}
```

---

#### `client.bridge.subscribeToEvent()`

Subscribes to resolution events from other hApps.

```typescript
async subscribeToEvent(input: EventSubscription): Promise<EntryHash>

interface EventSubscription {
  eventType: string;
  sourceHapp: string;
  filter?: Record<string, unknown>;
  callback: EventCallback;
}

type EventCallback = (event: BridgeEvent) => void | Promise<void>;
```

---

#### `client.bridge.listPendingRequests()`

Lists pending bridge requests.

```typescript
async listPendingRequests(options?: BridgeListOptions): Promise<BridgeRequest[]>
```

---

### Health Client

#### `client.health.check()`

Checks client and connection health.

```typescript
async check(): Promise<HealthStatus>

interface HealthStatus {
  connected: boolean;
  version: string;
  agentPubKey: AgentPubKey;
  cellId: CellId;
  latency: number;
  capabilities: string[];
}
```

---

#### `client.health.getAgentInfo()`

Gets information about the current agent.

```typescript
async getAgentInfo(): Promise<AgentInfo>

interface AgentInfo {
  agentPubKey: AgentPubKey;
  matlScore: MATLScore;
  predictionCount: number;
  marketCount: number;
  memberSince: number;
  roles: string[];
  badges: Badge[];
}
```

---

## Part II: Events and Signals

### Subscribing to Events

```typescript
// Market events
client.markets.on("marketCreated", (market: Market) => { ... });
client.markets.on("marketClosed", (marketId: EntryHash) => { ... });
client.markets.on("marketResolved", (resolution: Resolution) => { ... });
client.markets.on("priceUpdate", (update: PriceUpdate) => { ... });

// Prediction events
client.predictions.on("predictionSubmitted", (prediction: Prediction) => { ... });
client.predictions.on("predictionUpdated", (update: PredictionUpdate) => { ... });

// Resolution events
client.resolution.on("resolutionStarted", (process: ResolutionProcess) => { ... });
client.resolution.on("oracleVoteSubmitted", (vote: OracleVote) => { ... });
client.resolution.on("consensusReached", (consensus: ConsensusState) => { ... });
client.resolution.on("disputeRaised", (dispute: Dispute) => { ... });
client.resolution.on("resolutionFinalized", (result: FinalizationResult) => { ... });

// Question market events
client.questions.on("questionProposed", (question: QuestionMarket) => { ... });
client.questions.on("questionSpawned", (spawn: SpawnEvent) => { ... });

// Unsubscribe
const unsubscribe = client.markets.on("priceUpdate", handler);
unsubscribe();
```

---

## Part III: Error Handling

### Error Types

```typescript
class EpistemicError extends Error {
  code: ErrorCode;
  details?: Record<string, unknown>;
  retryable: boolean;
}

enum ErrorCode {
  // Client errors (4xx)
  InvalidInput = "INVALID_INPUT",
  NotFound = "NOT_FOUND",
  Unauthorized = "UNAUTHORIZED",
  Forbidden = "FORBIDDEN",
  RateLimited = "RATE_LIMITED",
  InsufficientStake = "INSUFFICIENT_STAKE",
  MarketClosed = "MARKET_CLOSED",
  AlreadyExists = "ALREADY_EXISTS",
  ValidationFailed = "VALIDATION_FAILED",

  // Oracle errors
  NotOracle = "NOT_ORACLE",
  AlreadyVoted = "ALREADY_VOTED",
  VotingClosed = "VOTING_CLOSED",

  // Resolution errors
  NotResolvable = "NOT_RESOLVABLE",
  DisputePending = "DISPUTE_PENDING",
  ConsensusNotReached = "CONSENSUS_NOT_REACHED",

  // Server errors (5xx)
  InternalError = "INTERNAL_ERROR",
  NetworkError = "NETWORK_ERROR",
  Timeout = "TIMEOUT",
  ServiceUnavailable = "SERVICE_UNAVAILABLE",
}
```

### Error Handling Example

```typescript
try {
  await client.predictions.submitPrediction(input);
} catch (error) {
  if (error instanceof EpistemicError) {
    switch (error.code) {
      case ErrorCode.MarketClosed:
        showMessage("This market is no longer accepting predictions");
        break;
      case ErrorCode.InsufficientStake:
        showMessage(`Minimum stake is ${error.details?.minStake}`);
        break;
      case ErrorCode.RateLimited:
        await delay(error.details?.retryAfter || 60000);
        // Retry
        break;
      default:
        showMessage(`Error: ${error.message}`);
    }
  }
}
```

---

## Part IV: Direct Zome Calls

For advanced use cases, you can call zome functions directly.

### Markets Zome

```typescript
// Using Holochain client directly
const result = await appClient.callZome({
  cap_secret: null,
  cell_id: cellId,
  zome_name: "markets",
  fn_name: "create_market",
  payload: marketConfig,
});
```

| Function | Input | Output |
|----------|-------|--------|
| `create_market` | `MarketConfig` | `EntryHash` |
| `get_market` | `EntryHash` | `Option<Market>` |
| `list_markets` | `ListOptions` | `Vec<MarketSummary>` |
| `list_open_markets` | `()` | `Vec<MarketSummary>` |
| `search_markets_by_tag` | `String` | `Vec<MarketSummary>` |
| `close_market` | `EntryHash` | `()` |
| `get_market_stats` | `EntryHash` | `MarketStats` |

### Predictions Zome

| Function | Input | Output |
|----------|-------|--------|
| `submit_prediction` | `PredictionInput` | `EntryHash` |
| `get_prediction` | `EntryHash` | `Option<Prediction>` |
| `update_prediction` | `PredictionUpdateInput` | `EntryHash` |
| `withdraw_prediction` | `EntryHash` | `WithdrawalResult` |
| `list_predictions_for_market` | `(EntryHash, ListOptions)` | `Vec<Prediction>` |
| `get_my_predictions` | `ListOptions` | `Vec<Prediction>` |

### Resolution Zome

| Function | Input | Output |
|----------|-------|--------|
| `start_resolution` | `(EntryHash, Option<String>)` | `ResolutionProcess` |
| `submit_oracle_vote` | `OracleVoteInput` | `EntryHash` |
| `get_resolution_process` | `EntryHash` | `Option<ResolutionProcess>` |
| `dispute_resolution` | `DisputeInput` | `EntryHash` |
| `finalize_resolution` | `EntryHash` | `FinalizationResult` |

### Question Markets Zome

| Function | Input | Output |
|----------|-------|--------|
| `propose_question` | `QuestionProposal` | `EntryHash` |
| `signal_curiosity` | `(EntryHash, Option<String>)` | `()` |
| `buy_question_shares` | `(EntryHash, u64, Option<u64>, Option<String>)` | `SharePurchase` |
| `sell_question_shares` | `(EntryHash, u64, Option<u64>)` | `ShareSale` |
| `get_question_market` | `EntryHash` | `Option<QuestionMarket>` |
| `list_question_markets` | `QuestionListOptions` | `Vec<QuestionMarket>` |
| `get_top_questions` | `(Option<u32>, Option<Vec<String>>)` | `Vec<QuestionMarket>` |

### Scoring Zome

| Function | Input | Output |
|----------|-------|--------|
| `calculate_brier_score` | `(AgentPubKey, BrierOptions)` | `BrierResult` |
| `get_calibration_metrics` | `(AgentPubKey, CalibrationOptions)` | `CalibrationMetrics` |
| `get_leaderboard` | `LeaderboardOptions` | `Vec<LeaderboardEntry>` |
| `plant_wisdom_seed` | `WisdomSeedInput` | `EntryHash` |
| `get_wisdom_seeds` | `WisdomSeedOptions` | `Vec<WisdomSeed>` |
| `rate_wisdom_seed` | `(EntryHash, i8)` | `()` |

### Markets Bridge Zome

| Function | Input | Output |
|----------|-------|--------|
| `submit_prediction_request` | `BridgeRequest` | `EntryHash` |
| `accept_request` | `EntryHash` | `()` |
| `reject_request` | `(EntryHash, String)` | `()` |
| `create_market_from_request` | `EntryHash` | `EntryHash` |
| `subscribe_to_event` | `EventSubscription` | `EntryHash` |
| `handle_bridge_event` | `BridgeEvent` | `()` |
| `list_pending_requests` | `BridgeListOptions` | `Vec<BridgeRequest>` |

---

## Part V: Rate Limits

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Market Creation | 10 | 1 hour |
| Predictions | 100 | 1 hour |
| Question Proposals | 20 | 1 hour |
| Oracle Votes | 50 | 1 hour |
| Read Operations | 1000 | 1 minute |
| Search Operations | 100 | 1 minute |

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## Part VI: Webhooks (Future)

For server-to-server integrations:

```typescript
// Register webhook
POST /api/webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["market.resolved", "prediction.submitted"],
  "secret": "your-webhook-secret"
}

// Webhook payload
{
  "id": "evt_123",
  "type": "market.resolved",
  "created": 1704067200,
  "data": {
    "marketId": "uhCkk...",
    "outcome": "Yes",
    "...": "..."
  }
}
```

---

## Appendix: Complete Type Definitions

For complete TypeScript type definitions, see the SDK package:

```typescript
import type {
  Market,
  MarketConfig,
  MarketStatus,
  Prediction,
  PredictionInput,
  StakeConfig,
  EpistemicPosition,
  QuestionMarket,
  ResolutionProcess,
  CalibrationMetrics,
  WisdomSeed,
  // ... all types
} from "@mycelix/epistemic-markets-sdk";
```

---

*"A good API is like a good map: it shows you exactly where to go."*

*May your integrations be smooth and your errors informative.*

