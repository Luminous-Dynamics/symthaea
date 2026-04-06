# @mycelix/sdk

> TypeScript SDK for the Mycelix Ecosystem - Decentralized Trust, Federated Learning & Byzantine-Resistant Infrastructure on Holochain

[![npm version](https://badge.fury.io/js/@mycelix%2Fsdk.svg)](https://www.npmjs.com/package/@mycelix/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-blue.svg)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/tests-1556%20passing-brightgreen.svg)]()

## Overview

The Mycelix SDK provides TypeScript/JavaScript bindings for building applications on the Mycelix ecosystem - a Holochain-based infrastructure for decentralized trust, federated learning, epistemic classification, and cross-hApp communication with **34% validated Byzantine fault tolerance**.

### Key Features

| Module | Description |
|--------|-------------|
| **MATL** | Mycelix Adaptive Trust Layer - 34% validated Byzantine tolerance, PoGQ, reputation scoring |
| **FL** | Federated Learning - Byzantine-resistant aggregation (FedAvg, Krum, TrimmedMean) |
| **FL Hub** | Federated Learning Hub - Session management, model registry, differential privacy |
| **FHE** | Fully Homomorphic Encryption - Privacy-preserving computation on encrypted data |
| **Epistemic** | Truth classification system (Empirical × Normative × Materiality) |
| **Bridge** | Cross-hApp communication and aggregate reputation |
| **Security** | Cryptographic utilities, rate limiting, input sanitization |
| **Config** | Centralized configuration management with presets and validation |
| **Utils** | Convenience functions, helpers, and type aliases |
| **Client** | Holochain conductor WebSocket connectivity |
| **Errors** | Structured error handling with recovery suggestions |
| **Validation** | Zod schemas for all bridge payloads with runtime validation |
| **Signals** | Typed event handlers for real-time Holochain signals |
| **CLI** | Developer tools for scaffolding, generation, and validation |
| **Integrations** | hApp-specific adapters for all 12 Civilizational OS domains |
| **Resilience** | Circuit breakers, retry policies, timeouts, rate limiting |
| **Observability** | OpenTelemetry metrics, tracing, and spans |
| **Mobile** | Mobile Wallet SDK - Wallet management, biometrics, QR codes |
| **AI** | AI Integration - Model inference, embeddings, RAG pipelines |
| **RTC** | Real-Time Communication - WebRTC, signaling, media handling |
| **React** | React hooks for MATL, Epistemic, Client, and GraphQL |
| **Svelte** | Svelte stores for reactive state management |
| **GraphQL** | Schema, resolvers, and subscriptions for all types |
| **Innovations** | Trust markets, private queries, epistemic agents, civic feedback, workflows, constitutional AI |

## Installation

```bash
npm install @mycelix/sdk
```

## Quick Start

```typescript
import { matl, epistemic, bridge, fl, security } from '@mycelix/sdk';

// === MATL: Trust Layer ===
// Create Proof of Gradient Quality measurement
const pogq = matl.createPoGQ(0.95, 0.88, 0.12); // quality, consistency, entropy
console.log(`Byzantine? ${matl.isByzantine(pogq)}`); // false (good participant)

// Reputation tracking
let reputation = matl.createReputation('agent_alice');
reputation = matl.recordPositive(reputation); // Good interaction
reputation = matl.recordPositive(reputation);
console.log(`Reputation: ${matl.reputationValue(reputation)}`); // ~0.67

// Composite trust score
const composite = matl.calculateComposite(pogq, reputation);
console.log(`Trustworthy? ${matl.isTrustworthy(composite, 0.5)}`); // true

// === Epistemic: Truth Classification ===
const claim = epistemic.claim('Transaction verified by 3 validators')
  .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
  .withNormative(epistemic.NormativeLevel.N2_Network)
  .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
  .withIssuer('trust_layer')
  .build();

console.log(`Classification: ${epistemic.classificationCode(claim.classification)}`); // E3-N2-M2
console.log(`Meets high trust? ${epistemic.meetsStandard(claim,
  epistemic.EmpiricalLevel.E2_PrivateVerify,
  epistemic.NormativeLevel.N2_Network
)}`); // true

// === Bridge: Cross-hApp Communication ===
const localBridge = new bridge.LocalBridge();
localBridge.registerHapp('marketplace');
localBridge.registerHapp('governance');

// Set reputation across hApps
localBridge.setReputation('marketplace', 'agent_bob', reputation);

// Get aggregate reputation
const aggregate = localBridge.getAggregateReputation('agent_bob');
console.log(`Cross-hApp reputation: ${aggregate}`);

// === FL: Federated Learning ===
const coordinator = new fl.FLCoordinator({
  minParticipants: 3,
  aggregationMethod: 'fedavg',
});

// Register participants
coordinator.registerParticipant('hospital_a');
coordinator.registerParticipant('hospital_b');
coordinator.registerParticipant('hospital_c');

// Start training round
coordinator.startRound();

// Submit gradient updates
coordinator.submitUpdate({
  participantId: 'hospital_a',
  modelVersion: 1,
  gradients: new Float64Array([0.1, 0.2, 0.15]),
  metadata: { batchSize: 100, loss: 0.25, timestamp: Date.now() },
});
// ... more updates

// Aggregate (Byzantine-resistant)
const success = coordinator.aggregateRound();

// === Security: Cryptographic Utilities ===
// Secure random generation
const uuid = security.secureUUID();
const randomBytes = security.secureRandomBytes(32);

// Hashing
const hash = await security.hashHex('sensitive data');

// Rate limiting
const registry = new security.RateLimiterRegistry({
  maxRequests: 100,
  windowMs: 60000, // 1 minute
});
const { allowed, remaining } = registry.check('user_123');

// Input sanitization
const sanitized = security.sanitizeString('<script>alert("xss")</script>Hello');
// Result: 'alert("xss")Hello'
```

## Modules

### MATL - Mycelix Adaptive Trust Layer

The MATL module implements **34% validated Byzantine fault tolerance** through Proof of Gradient Quality (PoGQ) and adaptive reputation scoring.

```typescript
import { matl } from '@mycelix/sdk';

// PoGQ: Measures participant quality in federated systems
const pogq = matl.createPoGQ(
  0.95,  // quality: gradient quality score [0,1]
  0.88,  // consistency: behavior consistency [0,1]
  0.12   // entropy: randomness/noise level [0,1]
);

// Check if participant is Byzantine (malicious)
if (matl.isByzantine(pogq)) {
  console.log('Participant flagged as potentially Byzantine');
}

// Reputation management
let rep = matl.createReputation('agent_id');
rep = matl.recordPositive(rep);  // Good interaction
rep = matl.recordNegative(rep);  // Bad interaction
const score = matl.reputationValue(rep);  // [0, 1]

// Composite scoring (PoGQ + Reputation)
const composite = matl.calculateComposite(pogq, rep);
const trusted = matl.isTrustworthy(composite, 0.5);  // threshold = 0.5

// Adaptive thresholds for anomaly detection
let threshold = matl.createAdaptiveThreshold('node_1', 100, 0.5, 2.0);
for (const observation of historicalData) {
  threshold = matl.observe(threshold, observation);
}
const isAnomalous = matl.isAnomalous(threshold, newObservation);

// Constants
console.log(matl.MAX_BYZANTINE_TOLERANCE);  // 0.34 (34% validated)
console.log(matl.DEFAULT_BYZANTINE_THRESHOLD);  // 0.5
```

### FL - Federated Learning

Byzantine-resistant federated learning with multiple aggregation methods.

```typescript
import { fl } from '@mycelix/sdk';

// Aggregation methods
const updates = [
  { participantId: 'p1', modelVersion: 1, gradients: new Float64Array([0.1, 0.2]),
    metadata: { batchSize: 100, loss: 0.3, timestamp: Date.now() } },
  { participantId: 'p2', modelVersion: 1, gradients: new Float64Array([0.12, 0.18]),
    metadata: { batchSize: 80, loss: 0.32, timestamp: Date.now() } },
  // ... more updates
];

// FedAvg - Standard weighted average
const avgResult = fl.fedAvg(updates);

// Trimmed Mean - Remove outliers before averaging
const trimmedResult = fl.trimmedMean(updates, 0.1);  // trim 10% extremes

// Coordinate Median - Robust to Byzantine attacks
const medianResult = fl.coordinateMedian(updates);

// Krum - Select most representative update
const krumResult = fl.krum(updates);

// Trust-weighted aggregation (uses MATL reputation)
const participants = new Map();
participants.set('p1', { id: 'p1', reputation: rep1, roundsParticipated: 10 });
participants.set('p2', { id: 'p2', reputation: rep2, roundsParticipated: 5 });
const trustResult = fl.trustWeightedAggregation(updates, participants, 0.3);

// FL Coordinator for managing rounds
const coordinator = new fl.FLCoordinator({
  minParticipants: 5,
  maxParticipants: 100,
  aggregationMethod: 'trust_weighted',
  trustThreshold: 0.3,
  byzantineTolerance: 0.2,
});

coordinator.registerParticipant('hospital_1');
coordinator.startRound();
coordinator.submitUpdate(update);
coordinator.aggregateRound();

const stats = coordinator.getRoundStats();
console.log(`Completed ${stats.totalRounds} rounds`);

// Serialization for network transfer
const serialized = fl.serializeGradients(gradients);
const deserialized = fl.deserializeGradients(serialized);
```

### Epistemic - Truth Classification

3D classification system for verifiable claims based on the Epistemic Charter v2.0.

```typescript
import { epistemic, EmpiricalLevel, NormativeLevel, MaterialityLevel } from '@mycelix/sdk';

// Empirical levels (evidence quality)
// E0_Unverified   - No verification
// E1_Testimonial  - Personal attestation
// E2_PrivateVerify- Privately verifiable
// E3_Cryptographic- Cryptographically proven
// E4_Consensus    - Consensus-verified

// Normative levels (scope of agreement)
// N0_Personal     - Individual only
// N1_Communal     - Small group
// N2_Network      - Network-wide
// N3_Universal    - Universal/institutional

// Materiality levels (persistence)
// M0_Ephemeral    - Temporary
// M1_Versioned    - Version-tracked
// M2_Persistent   - Permanently stored
// M3_Immutable    - Cannot be changed

// Build a claim
const claim = epistemic.claim('Smart contract audited by CertiK')
  .withEmpirical(EmpiricalLevel.E3_Cryptographic)
  .withNormative(NormativeLevel.N3_Universal)
  .withMateriality(MaterialityLevel.M3_Immutable)
  .withIssuer('certik_audits')
  .withExpiration(Date.now() + 365 * 24 * 60 * 60 * 1000)  // 1 year
  .build();

// Classification code
const code = epistemic.classificationCode(claim.classification);  // "E3-N3-M3"

// Parse classification code
const parsed = epistemic.parseClassificationCode('E2-N2-M2');

// Check against standards
const meetsHighTrust = epistemic.meetsStandard(
  claim,
  EmpiricalLevel.E3_Cryptographic,
  NormativeLevel.N2_Network
);

// Pre-defined standards
const { minE, minN, minM } = epistemic.Standards.HighTrust;
const meetsHigh = epistemic.meetsMinimum(claim.classification, minE, minN, minM);

// Add evidence
const withEvidence = epistemic.addEvidence(claim, {
  type: 'audit_report',
  data: 'https://certik.com/audit/12345',
  source: 'certik',
  timestamp: Date.now(),
});

// Check expiration
if (epistemic.isExpired(claim)) {
  console.log('Claim has expired');
}
```

### Bridge - Cross-hApp Communication

Event routing and reputation aggregation across Holochain applications.

```typescript
import { bridge } from '@mycelix/sdk';

// Local bridge for testing (no conductor needed)
const localBridge = new bridge.LocalBridge();

// Register hApps
localBridge.registerHapp('marketplace');
localBridge.registerHapp('governance');
localBridge.registerHapp('social');

// Set reputation for an agent in a specific hApp
localBridge.setReputation('marketplace', 'agent_alice', reputation);

// Get cross-hApp reputation scores
const scores = localBridge.getCrossHappReputation('agent_alice');
// [{ happ: 'marketplace', score: 0.85 }, ...]

// Get aggregate reputation (weighted by interactions)
const aggregate = localBridge.getAggregateReputation('agent_alice');

// Subscribe to events
localBridge.on('governance', bridge.BridgeMessageType.BroadcastEvent, (msg) => {
  console.log('Received:', msg);
});

// Send messages
const verificationResult = bridge.createVerificationResult(
  'identity',          // source hApp
  'credential_123',    // credential ID
  true,                // valid
  'identity_verifier', // verifier
  ['kyc_complete', 'biometric_verified']  // claims
);
localBridge.send('marketplace', verificationResult);

// Broadcast events to all hApps
const event = bridge.createBroadcastEvent(
  'trust_layer',
  'reputation_updated',
  new TextEncoder().encode(JSON.stringify({ agent: 'alice', score: 0.9 }))
);
localBridge.broadcast(event);

// Calculate aggregate reputation manually
const aggregated = bridge.calculateAggregateReputation([
  { happ: 'marketplace', score: 0.9, weight: 0.4 },
  { happ: 'governance', score: 0.85, weight: 0.35 },
  { happ: 'social', score: 0.7, weight: 0.25 },
]);
```

### Civilizational Workflows - Pre-Built Cross-hApp Operations

The SDK includes pre-built workflows that coordinate multiple hApps for common civilizational operations. Each workflow handles verification, validation, state management, and step-by-step progress tracking.

```typescript
import {
  executeProposalWorkflow,
  executeLendingWorkflow,
  executePropertyTransferWorkflow,
  executeEnergyTradeWorkflow,
  executeEnforcementWorkflow,
  executePublicationWorkflow,
  executeComprehensiveIdentityCheck,
  type WorkflowResult,
} from '@mycelix/sdk/bridge/workflows';

// === Governance Proposal ===
// Coordinates: Identity → Governance → Knowledge
const proposalResult = await executeProposalWorkflow({
  proposerDid: 'did:mycelix:alice-verified',
  daoId: 'solar-coop-sf',
  title: 'Expand Community Solar Array by 50kW',
  description: 'Proposal to install additional solar panels...',
  votingPeriodHours: 168,  // 1 week
  quorumPercentage: 0.51,
});

console.log(`Proposal ${proposalResult.success ? 'created' : 'failed'}`);
console.log(`Steps: ${proposalResult.steps.map(s => s.name).join(' → ')}`);

// === Credit/Lending Application ===
// Coordinates: Identity → Property → Finance → Knowledge
const lendingResult = await executeLendingWorkflow({
  applicantDid: 'did:mycelix:borrower-charlie',
  amount: 10000,
  currency: 'MCX',
  purpose: 'Purchase solar installation equipment',
  termMonths: 24,
  collateralAssetId: 'asset:property:warehouse-lot-7',
});

if (lendingResult.data?.approved) {
  console.log(`Loan approved: ${lendingResult.data.loanId}`);
  console.log(`Interest rate: ${lendingResult.data.interestRate}%`);
}

// === Property Transfer with Escrow ===
// Coordinates: Property → Finance → Identity → Justice
const transferResult = await executePropertyTransferWorkflow({
  assetId: 'asset:property:solar-farm-lot-12',
  fromOwner: 'did:mycelix:seller-dana',
  toOwner: 'did:mycelix:buyer-evan',
  considerationAmount: 50000,
  currency: 'MCX',
  escrowDurationDays: 30,
});

console.log(`Transfer status: ${transferResult.data?.status}`);
console.log(`Escrow ID: ${transferResult.data?.escrowId}`);

// === P2P Energy Trade ===
// Coordinates: Energy → Finance → Identity → Knowledge
const tradeResult = await executeEnergyTradeWorkflow({
  sellerDid: 'did:mycelix:solar-producer-frank',
  buyerDid: 'did:mycelix:consumer-grace',
  amountKwh: 500,
  pricePerKwh: 0.12,
  source: 'solar',  // or 'wind', 'hydro', etc.
});

console.log(`Trade ID: ${tradeResult.data?.tradeId}`);
console.log(`REC Credit: ${tradeResult.data?.creditId}`);

// === Content Publication with Fact-Check ===
// Coordinates: Media → Knowledge → Identity → Governance
const publicationResult = await executePublicationWorkflow({
  authorDid: 'did:mycelix:journalist-henry',
  contentHash: 'QmXyz123abc456def789...',
  title: 'Community Solar Reduces Grid Costs by 15%',
  tags: ['energy', 'solar', 'community'],
  requestFactCheck: true,
});

// === Justice Enforcement ===
// Coordinates: Justice → Finance → Governance → Identity
const enforcementResult = await executeEnforcementWorkflow({
  decisionId: 'decision:case-001-final',
  targetDid: 'did:mycelix:respondent-iris',
  remedies: [{
    type: 'financial',
    happId: 'finance',
    amount: 500,
    currency: 'MCX',
    description: 'Damages for breach of contract',
  }],
});

// === Comprehensive Identity Check ===
// Full verification across reputation, credentials, standing
const identityResult = await executeComprehensiveIdentityCheck({
  subjectDid: 'did:mycelix:member-jack',
  requesterHapp: 'governance',
  checkCategories: ['reputation', 'credentials', 'standing'],
});

if (identityResult.data?.verified) {
  console.log(`Agent verified with MATL score: ${identityResult.data.matlScore}`);
}

// === WorkflowResult Structure ===
// All workflows return WorkflowResult<T>:
interface WorkflowResult<T> {
  success: boolean;
  duration: number;  // milliseconds
  error?: string;
  steps: Array<{
    name: string;
    happ: string;
    status: 'pending' | 'completed' | 'failed' | 'skipped';
    result?: Record<string, unknown>;
    error?: string;
  }>;
  data?: T;
}
```

**Available Workflows:**

| Workflow | hApps Coordinated | Use Case |
|----------|-------------------|----------|
| `executeProposalWorkflow` | Identity, Governance, Knowledge | Democratic decision making |
| `executeLendingWorkflow` | Identity, Property, Finance | MATL-based credit applications |
| `executePropertyTransferWorkflow` | Property, Finance, Identity, Justice | Escrowed ownership changes |
| `executeEnergyTradeWorkflow` | Energy, Finance, Identity | P2P renewable energy exchange |
| `executePublicationWorkflow` | Media, Knowledge, Identity | Content with fact-checking |
| `executeEnforcementWorkflow` | Justice, Finance, Governance | Cross-hApp penalty execution |
| `executeComprehensiveIdentityCheck` | Identity, all registered hApps | Full agent verification |

### Security - Cryptographic Utilities

Comprehensive security module for Byzantine-resistant systems.

```typescript
import { security } from '@mycelix/sdk';

// === Secure Random Generation ===
const randomBytes = security.secureRandomBytes(32);
const randomFloat = security.secureRandomFloat();  // [0, 1)
const randomInt = security.secureRandomInt(1, 100);  // [1, 100]
const uuid = security.secureUUID();  // UUID v4
const shuffled = security.secureShufffle([1, 2, 3, 4, 5]);

// === Cryptographic Hashing ===
const hash = await security.hash('data', 'SHA-256');  // Uint8Array
const hashHex = await security.hashHex('data');  // hex string

// HMAC
const key = security.secureRandomBytes(32);
const mac = await security.hmac(key, 'message');
const isValid = await security.verifyHmac(key, 'message', mac);

// Constant-time comparison (timing-attack safe)
const equal = security.constantTimeEqual(bytes1, bytes2);

// === Rate Limiting ===
const registry = new security.RateLimiterRegistry({
  maxRequests: 100,
  windowMs: 60000,  // 1 minute
  slidingWindow: true,
});

const result = registry.check('user_123');
if (!result.allowed) {
  console.log(`Rate limited. Reset at: ${new Date(result.resetAt)}`);
}

// === Input Sanitization ===
const sanitized = security.sanitizeString(userInput, {
  maxLength: 1000,
  stripHtml: true,
  stripControl: true,
});

const safeId = security.sanitizeId(userId);  // alphanumeric + _ - only

const parsed = security.sanitizeJson<MyType>(jsonString, {
  maxDepth: 10,
  maxSize: 1000000,
});

// === Secure Secrets ===
const secret = new security.SecureSecret('api-key-12345', 60000);  // 1 min TTL
await secret.use(async (bytes) => {
  // Use secret here - auto-cleared after callback
  await makeApiCall(new TextDecoder().decode(bytes));
});
// Secret is now cleared from memory

// === Byzantine Resistance Utilities ===
// Validate BFT parameters (n >= 3f + 1)
const { valid, message } = security.validateBftParams(10, 3);  // 10 nodes, 3 Byzantine

// Calculate max Byzantine failures
const maxF = security.maxByzantineFailures(10);  // 3

// Check quorum
const hasQuorum = security.hasQuorum(10, 2, 6);  // 10 total, 2 Byzantine, need 6

// === Timing Attack Protection ===
await security.randomDelay(10, 100);  // Random delay 10-100ms

const result = await security.constantTime(async () => {
  return await sensitiveOperation();
}, 50);  // Ensure minimum 50ms execution

// === Security Audit Logging ===
const auditLog = new security.SecurityAuditLog(1000);  // Keep 1000 events

auditLog.onEvent((event) => {
  console.log(`[${event.severity}] ${event.type}: ${JSON.stringify(event.details)}`);
});

auditLog.log(
  security.SecurityEventType.RATE_LIMIT_EXCEEDED,
  { userId: 'user_123', requests: 150 },
  'high',
  'user_123'
);

const criticalEvents = auditLog.getEvents({ severity: 'critical' });
```

### Integrations - hApp-Specific Adapters

Pre-built adapters for each Mycelix ecosystem hApp with domain-specific trust logic.

```typescript
// === Mail Integration ===
import { getMailTrustService, MailTrustService } from '@mycelix/sdk/integrations/mail';

const mailService = getMailTrustService();

// Record sender interactions for learning
mailService.recordInteraction('sender@example.com', true);  // positive
mailService.recordInteraction('spam@bad.com', false);       // negative

// Check sender trust
const trust = mailService.getSenderTrust('sender@example.com');
console.log(`Level: ${trust.level}, Score: ${trust.score}`);  // "high", 0.85

// Create epistemic claim for email content
const emailClaim = mailService.createEmailClaim({
  id: 'email-123',
  subject: 'Important Update',
  from: 'verified@example.com',
  to: ['recipient@example.com'],
  body: '...',
  timestamp: Date.now(),
  verification: {
    dkimVerified: true,
    spfPassed: true,
    dmarcPassed: true,
  },
});

// === Marketplace Integration ===
import { getMarketplaceService } from '@mycelix/sdk/integrations/marketplace';

const marketplace = getMarketplaceService();

// Record transaction outcomes
const result = marketplace.recordTransaction({
  id: 'tx-123',
  type: 'purchase',
  buyerId: 'buyer-1',
  sellerId: 'seller-1',
  amount: 100,
  currency: 'USD',
  itemId: 'item-1',
  timestamp: Date.now(),
  success: true,
});

// Get seller profile
const seller = marketplace.getSellerProfile('seller-1');
console.log(`Trust: ${seller.trustScore}, Verified: ${seller.verified}`);

// Verify listing safety
const verification = marketplace.verifyListing('listing-1', 'seller-1');
console.log(`Safe: ${verification.verified}, Risk: ${verification.scamRiskScore}`);

// === Praxis Integration ===
import { getPraxisService } from '@mycelix/sdk/integrations/praxis';

const praxis = getPraxisService();

// Issue course completion certificate
const credential = praxis.issueCertificate({
  studentId: 'student-123',
  courseId: 'nix-fundamentals',
  courseName: 'NixOS Fundamentals',
  grade: 95,
});

// Verify credential
const { valid, credential: cred, reason } = praxis.verifyCredential(credential.id);

// Get learner profile
const learner = praxis.getLearnerProfile('student-123');
console.log(`Completed: ${learner.coursesCompleted}, Trust: ${learner.trustScore}`);

// === SupplyChain Integration ===
import { getSupplyChainService } from '@mycelix/sdk/integrations/supplychain';

const supplychain = getSupplyChainService();

// Record checkpoint
const checkpoint = supplychain.recordCheckpoint({
  productId: 'product-123',
  location: 'Warehouse A',
  handler: 'handler-456',
  action: 'received',
  evidence: [{
    type: 'iot_sensor',
    data: { temperature: 4.5, humidity: 45 },
    timestamp: Date.now(),
    verified: true,
  }],
});

// Get full provenance chain
const chain = supplychain.getProvenanceChain('product-123');
console.log(`Integrity: ${chain?.chainIntegrity}, Distance: ${chain?.totalDistance}km`);

// Verify chain integrity
const verification = supplychain.verifyChain('product-123');
console.log(`Verified: ${verification.verified}, Weak links: ${verification.weakLinks.length}`);
```

### Client - Holochain Connectivity

WebSocket client for Holochain conductor communication.

```typescript
import { createClient, createMockClient, MycelixClient } from '@mycelix/sdk';

// For testing (no conductor required)
const mockClient = createMockClient();

// For production
const client = createClient({
  installedAppId: 'mycelix_ecosystem',
  appUrl: 'ws://localhost:8888',
});

await client.connect();

// Register hApp with bridge
await client.registerHapp({
  happId: 'marketplace',
  subscriptions: ['trust_updated', 'reputation_changed'],
  metadata: { version: '1.0.0' },
});

// Record reputation
await client.recordReputation({
  targetAgent: 'uhCAk_agent_hash',
  delta: 1,  // positive
  context: 'successful_trade',
});

// Query cross-hApp reputation
const reputation = await client.queryCrossHappReputation('uhCAk_agent_hash');
console.log(`Aggregate: ${reputation.aggregate}`);
console.log(`Interactions: ${reputation.total_interactions}`);

// Check trustworthiness
const trusted = await client.isAgentTrustworthy({
  agent: 'uhCAk_agent_hash',
  threshold: 0.7,
});

// Broadcast event
await client.broadcastEvent({
  eventType: 'price_updated',
  payload: { item: 'item_123', price: 99.99 },
  targetHapps: ['marketplace', 'analytics'],
});

// Get events
const events = await client.getEvents({
  eventTypes: ['trust_updated'],
  since: Date.now() - 3600000,  // Last hour
  limit: 100,
});

// Verify credential
const verification = await client.verifyCredential({
  credentialId: 'cred_123',
  claims: ['kyc_verified'],
});

await client.disconnect();
```

### Errors - Structured Error Handling

Comprehensive error handling with codes, context, and recovery suggestions.

```typescript
import {
  MycelixError,
  ValidationError,
  ErrorCode,
  validate,
  withRetry,
  withErrorHandling,
} from '@mycelix/sdk';

// Validation
const validator = validate()
  .required('agentId', agentId)
  .inRange('score', score, 0, 1)
  .notEmpty('name', name)
  .pattern('email', email, /^[^\s@]+@[^\s@]+\.[^\s@]+$/, 'must be valid email');

const result = validator.result();
if (!result.valid) {
  console.log('Errors:', result.errors);
}

// Or throw immediately
validator.throwIfInvalid();

// Custom errors with context
throw new MycelixError(
  'Participant not found',
  ErrorCode.FL_PARTICIPANT_NOT_FOUND,
  { participantId: 'unknown_participant' }
);

// Error handling wrapper
const safeFunction = withErrorHandling(riskyFunction, 'processing data');

// Retry with exponential backoff
const result = await withRetry(
  async () => {
    return await unreliableApiCall();
  },
  {
    maxRetries: 3,
    initialDelay: 100,
    maxDelay: 5000,
    backoffFactor: 2,
    retryOn: (error) => error instanceof ConnectionError,
  }
);

// Error codes by category
// 1xxx - General (UNKNOWN, INVALID_ARGUMENT, NOT_FOUND, TIMEOUT)
// 2xxx - MATL (INVALID_SCORE, AGENT_NOT_FOUND, THRESHOLD_OUT_OF_RANGE)
// 3xxx - Epistemic (INVALID_LEVEL, CLAIM_EXPIRED, INSUFFICIENT_EVIDENCE)
// 4xxx - FL (NOT_ENOUGH_PARTICIPANTS, ROUND_NOT_STARTED, AGGREGATION_FAILED)
// 5xxx - Bridge (HAPP_NOT_FOUND, CONNECTION_FAILED, INVALID_WEIGHT)
// 6xxx - Connection (CONNECTION_FAILED, TIMEOUT, AUTH_FAILED)
// 7xxx - Security (RATE_LIMITED, INVALID_SIGNATURE, SECRET_EXPIRED)
```

### Validation - Runtime Schema Validation

Comprehensive Zod schemas for all bridge payloads with runtime validation.

```typescript
import {
  validateOrThrow,
  validateSafe,
  withValidation,
  IdentitySchemas,
  FinanceSchemas,
  PropertySchemas,
  EnergySchemas,
  MediaSchemas,
  GovernanceSchemas,
  JusticeSchemas,
  KnowledgeSchemas,
} from '@mycelix/sdk/validation';

// === Validate Bridge Payloads ===

// Identity domain
const identityInput = {
  happ_id: 'identity-happ',
  agent: 'did:mycelix:abc123def456',
};
const validated = validateOrThrow(IdentitySchemas.QueryIdentityInput, identityInput);

// Finance domain - payment processing
const paymentInput = {
  payer_did: 'did:mycelix:payer123',
  payee_did: 'did:mycelix:payee456',
  amount: 100.50,
  currency: 'MCX',
  reference_id: 'order-789',
};
validateOrThrow(FinanceSchemas.ProcessPaymentInput, paymentInput);

// Energy domain - query available energy
const energyQuery = {
  location: { lat: 37.7749, lng: -122.4194 },
  radius_km: 50,
  min_kwh: 100,
  energy_types: ['solar', 'wind'],
};
validateOrThrow(EnergySchemas.QueryAvailableEnergyInput, energyQuery);

// === Safe Validation (no throwing) ===
const result = validateSafe(IdentitySchemas.RegisterHappInput, userInput);
if (result.success) {
  console.log('Valid:', result.data);
} else {
  console.log('Errors:', result.error.issues);
}

// === Higher-Order Function Wrapper ===
const processPayment = withValidation(
  FinanceSchemas.ProcessPaymentInput,
  async (input) => {
    // Input is already validated and typed
    return await paymentService.process(input);
  }
);

// Will throw if input is invalid
await processPayment({ payer_did: '...', payee_did: '...', amount: 100 });

// === Common Schemas ===
import { didSchema, happIdSchema, matlScoreSchema, timestampSchema } from '@mycelix/sdk/validation';

// DID validation (must start with 'did:mycelix:')
didSchema.parse('did:mycelix:abc123');

// MATL score (0 to 1)
matlScoreSchema.parse(0.85);

// hApp ID (lowercase alphanumeric with hyphens)
happIdSchema.parse('my-cool-happ');

// === All Domain Schemas ===
// IdentitySchemas: RegisterHappInput, QueryIdentityInput, VerifyCredentialInput, ...
// FinanceSchemas: QueryCreditInput, ProcessPaymentInput, RequestLoanInput, ...
// PropertySchemas: VerifyOwnershipInput, PledgeCollateralInput, TransferPropertyInput, ...
// EnergySchemas: QueryAvailableEnergyInput, ListEnergyInput, PurchaseEnergyInput, ...
// MediaSchemas: QueryContentInput, RequestLicenseInput, ReportContentInput, ...
// GovernanceSchemas: QueryGovernanceInput, RequestExecutionInput, SubmitVoteInput, ...
// JusticeSchemas: FileCrossHappDisputeInput, RequestArbitrationInput, SubmitEvidenceInput, ...
// KnowledgeSchemas: QueryKnowledgeInput, FactCheckInput, CreateClaimInput, ...
```

### Signals - Real-Time Event Handling

Typed signal handlers for real-time events from Holochain conductor.

```typescript
import {
  createSignalManager,
  createIdentitySignals,
  createFinanceSignals,
  createEnergySignals,
  createGovernanceSignals,
  createKnowledgeSignals,
  BridgeSignalManager,
} from '@mycelix/sdk/signals';
import { createClient } from '@mycelix/sdk';

// === Full Signal Manager ===
const client = createClient({ installedAppId: 'mycelix' });
await client.connect();

// Create manager and connect to client
const signals = createSignalManager();
signals.connect(client);

// Subscribe to identity events
signals.identity.onIdentityCreated((data) => {
  console.log(`New identity: ${data.did}, MATL: ${data.initial_matl_score}`);
});

signals.identity.onCredentialIssued((data) => {
  console.log(`Credential ${data.credential_id} issued by ${data.issuer_did}`);
});

// Subscribe to finance events
signals.finance.onPaymentProcessed((data) => {
  console.log(`Payment ${data.payment_id}: ${data.amount} ${data.currency}`);
});

signals.finance.onCreditScoreUpdated((data) => {
  console.log(`Credit score changed: ${data.old_score} → ${data.new_score}`);
});

// Subscribe to energy events
signals.energy.onEnergyListed((data) => {
  console.log(`${data.amount_kwh} kWh of ${data.energy_source} @ $${data.price_per_kwh}/kWh`);
});

signals.energy.onGridBalanceRequest((data) => {
  console.log(`Grid needs ${data.needed_kwh} kWh by ${new Date(data.deadline)}`);
});

// Subscribe to governance events
signals.governance.onProposalCreated((data) => {
  console.log(`New proposal: "${data.title}" by ${data.proposer_did}`);
});

signals.governance.onVoteCast((data) => {
  console.log(`Vote: ${data.vote} with weight ${data.weight}`);
});

// Subscribe to knowledge events
signals.knowledge.onClaimCreated((data) => {
  console.log(`Claim: ${data.title} [E:${data.empirical} N:${data.normative} M:${data.mythic}]`);
});

signals.knowledge.onFactCheckResult((data) => {
  console.log(`Verdict: ${data.verdict} (${data.confidence * 100}% confidence)`);
});

// === Unsubscribe ===
const unsubscribe = signals.identity.onIdentityCreated((data) => { ... });
// Later...
unsubscribe();

// === Wildcard Subscriptions ===
signals.identity.onAny((signal) => {
  console.log('Identity event:', signal.signalName, signal.payload);
});

// === Signal Buffering ===
// Recent signals are buffered for late subscribers
const recentIdentitySignals = signals.identity.getBuffer();
console.log(`${recentIdentitySignals.length} recent signals buffered`);

// === Standalone Domain Handlers ===
// Create handlers for specific domains only
const identitySignals = createIdentitySignals({ bufferSize: 50 });
const financeSignals = createFinanceSignals();
const energySignals = createEnergySignals();

// Connect to client's signal stream
client.onSignal((signal) => {
  identitySignals.handle(signal);
  financeSignals.handle(signal);
});
```

### CLI - Developer Tools

Command-line tools for scaffolding new hApps and working with the Mycelix ecosystem.

```bash
# Install globally or use npx
npm install -g @mycelix/sdk
# or
npx @mycelix/sdk <command>

# === Initialize a New hApp ===
npx @mycelix/sdk init my-marketplace

# With UI scaffolding
npx @mycelix/sdk init my-marketplace --with-ui

# With specific bridge integrations
npx @mycelix/sdk init my-marketplace --integrations=identity,finance,property

# === Generated Structure ===
# my-marketplace/
# ├── mycelix.json          # Project configuration
# ├── happ.yaml             # Holochain hApp manifest
# ├── dna.yaml              # DNA manifest
# ├── zomes/
# │   ├── integrity/        # Integrity zome (validation rules)
# │   │   ├── Cargo.toml
# │   │   └── src/lib.rs
# │   ├── coordinator/      # Coordinator zome (business logic)
# │   │   ├── Cargo.toml
# │   │   └── src/lib.rs
# │   └── bridge/           # Bridge zome (cross-hApp communication)
# │       ├── Cargo.toml
# │       └── src/lib.rs
# ├── client/               # TypeScript client
# │   ├── package.json
# │   ├── tsconfig.json
# │   └── src/
# │       └── index.ts
# ├── tests/                # Integration tests
# │   └── src/
# │       └── index.ts
# └── ui/                   # (if --with-ui) SvelteKit UI
#     └── ...

# === Generate Bridge Methods ===
npx @mycelix/sdk generate bridge verify-seller

# === Validate Configuration ===
npx @mycelix/sdk validate
# ✓ mycelix.json is valid
# ✓ happ.yaml is valid
# ✓ All integrity zomes found
# ✓ All coordinator zomes found

# === Check Project Status ===
npx @mycelix/sdk status
# Project: my-marketplace
# Version: 0.1.0
# Integrations: identity, finance, property
# Zomes: integrity (✓), coordinator (✓), bridge (✓)
# Tests: 23 passing

# === Help ===
npx @mycelix/sdk help
npx @mycelix/sdk help init
```

```typescript
// === Programmatic CLI Usage ===
import { runCli, initProject, validateProject } from '@mycelix/sdk/cli';

// Initialize project programmatically
await initProject('my-happ', {
  withUI: true,
  integrations: ['identity', 'finance'],
});

// Validate project
const validation = await validateProject('/path/to/project');
if (!validation.valid) {
  console.log('Issues:', validation.errors);
}
```

### Resilience - Production Hardening

Circuit breakers, retry policies, timeouts, and rate limiting for production deployments.

```typescript
import {
  createCircuitBreaker,
  createRetryPolicy,
  withTimeout,
  createRateLimiter,
} from '@mycelix/sdk/resilience';

// === Circuit Breaker ===
// Prevents cascading failures by opening after threshold failures
const breaker = createCircuitBreaker({
  failureThreshold: 5,        // Open after 5 failures
  recoveryTimeout: 30000,     // Try half-open after 30s
  monitoringWindow: 60000,    // Rolling window for failure count
});

const result = await breaker.execute(async () => {
  return await riskyOperation();
});

console.log(`Circuit state: ${breaker.getState()}`); // 'closed' | 'open' | 'half-open'

// === Retry Policy ===
// Exponential backoff with jitter
const retry = createRetryPolicy({
  maxRetries: 3,
  initialDelay: 100,
  maxDelay: 5000,
  backoffMultiplier: 2,
  jitter: true,
  retryOn: (error) => error.code === 'NETWORK_ERROR',
});

const result = await retry.execute(async () => {
  return await unreliableApiCall();
});

// === Timeout ===
// Abort operations that take too long
const result = await withTimeout(
  async () => await slowOperation(),
  5000, // 5 second timeout
  'Operation timed out'
);

// === Rate Limiter ===
// Token bucket algorithm for request throttling
const limiter = createRateLimiter({
  maxTokens: 100,        // Bucket capacity
  refillRate: 10,        // Tokens per second
  refillInterval: 1000,  // Refill every second
});

if (limiter.tryAcquire()) {
  await processRequest();
} else {
  throw new Error('Rate limit exceeded');
}

// === Combined Usage ===
// Wrap operations with all patterns
const resilientCall = async () => {
  return await breaker.execute(() =>
    retry.execute(() =>
      withTimeout(
        () => apiCall(),
        5000
      )
    )
  );
};
```

### Observability - OpenTelemetry Integration

Metrics, tracing, and spans for monitoring and debugging.

```typescript
import {
  createMetrics,
  createTracer,
  withSpan,
  MetricType,
} from '@mycelix/sdk/observability';

// === Metrics ===
const metrics = createMetrics('mycelix-sdk');

// Counter - monotonically increasing
const requestCounter = metrics.createCounter('requests_total', {
  description: 'Total number of requests',
});
requestCounter.add(1, { method: 'POST', status: '200' });

// Histogram - distribution of values
const latencyHistogram = metrics.createHistogram('request_latency_ms', {
  description: 'Request latency in milliseconds',
  boundaries: [10, 50, 100, 250, 500, 1000],
});
latencyHistogram.record(42, { endpoint: '/api/trust' });

// Gauge - point-in-time value
const activeConnections = metrics.createGauge('active_connections', {
  description: 'Number of active connections',
});
activeConnections.set(5);

// === Tracing ===
const tracer = createTracer('mycelix-sdk');

// Create spans for distributed tracing
const span = tracer.startSpan('process-gradient');
try {
  span.setAttribute('participant.id', 'agent-123');
  span.setAttribute('round', 5);

  const result = await processGradient(input);

  span.setStatus({ code: 'OK' });
  return result;
} catch (error) {
  span.setStatus({ code: 'ERROR', message: error.message });
  span.recordException(error);
  throw error;
} finally {
  span.end();
}

// === Convenience Wrapper ===
const result = await withSpan(tracer, 'aggregate-round', async (span) => {
  span.setAttribute('participants', participants.length);
  return await aggregateGradients(participants);
});

// === Export Configuration ===
// Configure exporters (OTLP, Jaeger, Prometheus, etc.)
import { configureExporters } from '@mycelix/sdk/observability';

configureExporters({
  metrics: {
    endpoint: 'http://localhost:4318/v1/metrics',
    interval: 10000,
  },
  traces: {
    endpoint: 'http://localhost:4318/v1/traces',
    sampler: 'parentbased_traceidratio',
    samplerArg: 0.1, // 10% sampling
  },
});
```

### React - Hooks for React Applications

React hooks for integrating Mycelix SDK into React applications.

```typescript
import {
  MycelixProvider,
  usePoGQ,
  useReputation,
  useClaim,
  useClient,
  useEpistemicQuery,
  useBridge,
} from '@mycelix/sdk/react';

// === Provider Setup ===
function App() {
  return (
    <MycelixProvider
      config={{
        conductorUrl: 'ws://localhost:8888',
        installedAppId: 'mycelix',
      }}
    >
      <TrustDashboard />
    </MycelixProvider>
  );
}

// === usePoGQ Hook ===
function ParticipantScore({ quality, consistency, entropy }) {
  const { pogq, isByzantine, score } = usePoGQ(quality, consistency, entropy);

  return (
    <div>
      <p>Score: {score.toFixed(2)}</p>
      <p>Byzantine: {isByzantine ? 'Yes ⚠️' : 'No ✓'}</p>
    </div>
  );
}

// === useReputation Hook ===
function AgentReputation({ agentId }) {
  const {
    reputation,
    score,
    recordPositive,
    recordNegative,
    isLoading,
  } = useReputation(agentId);

  return (
    <div>
      <p>Reputation: {(score * 100).toFixed(1)}%</p>
      <button onClick={recordPositive}>+</button>
      <button onClick={recordNegative}>-</button>
    </div>
  );
}

// === useClaim Hook ===
function ClaimBuilder() {
  const { claim, setClaim, classification, isValid } = useClaim();

  return (
    <div>
      <input
        value={claim.content}
        onChange={(e) => setClaim({ ...claim, content: e.target.value })}
      />
      <p>Classification: {classification}</p>
      <p>Valid: {isValid ? '✓' : '✗'}</p>
    </div>
  );
}

// === useClient Hook ===
function ConnectionStatus() {
  const { client, isConnected, connect, disconnect } = useClient();

  return (
    <div>
      <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <button onClick={isConnected ? disconnect : connect}>
        {isConnected ? 'Disconnect' : 'Connect'}
      </button>
    </div>
  );
}

// === useEpistemicQuery Hook ===
function ClaimSearch({ minEmpirical }) {
  const { claims, isLoading, error, refetch } = useEpistemicQuery({
    minEmpirical,
    minNormative: 2,
  });

  if (isLoading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <ul>
      {claims.map((c) => (
        <li key={c.id}>{c.content} [{c.classification}]</li>
      ))}
    </ul>
  );
}
```

### Svelte - Stores for Svelte Applications

Svelte stores for reactive state management with Mycelix SDK.

```typescript
import {
  createPoGQStore,
  createReputationStore,
  createClaimStore,
  createClientStore,
  createBridgeStore,
} from '@mycelix/sdk/svelte';

// === PoGQ Store ===
const pogqStore = createPoGQStore(0.9, 0.85, 0.1);

// In Svelte component:
// <script>
//   import { pogqStore } from './stores';
//   $: ({ pogq, isByzantine, score } = $pogqStore);
// </script>
// <p>Score: {score}</p>

pogqStore.update(0.95, 0.9, 0.08); // Update values

// === Reputation Store ===
const reputationStore = createReputationStore('agent-123');

reputationStore.recordPositive();
reputationStore.recordNegative();
// $reputationStore.score -> current score

// === Claim Store ===
const claimStore = createClaimStore();

claimStore.setContent('Verified transaction');
claimStore.setEmpirical(3);
claimStore.setNormative(2);
claimStore.setMateriality(2);
claimStore.setIssuer('trust-layer');
// $claimStore.claim -> built claim
// $claimStore.classification -> 'E3-N2-M2'

// === Client Store ===
const clientStore = createClientStore({
  conductorUrl: 'ws://localhost:8888',
  installedAppId: 'mycelix',
});

await clientStore.connect();
// $clientStore.isConnected -> true
// $clientStore.client -> MycelixClient instance

// === Bridge Store ===
const bridgeStore = createBridgeStore();

bridgeStore.registerHapp('marketplace');
bridgeStore.setReputation('marketplace', 'agent-1', reputation);
// $bridgeStore.happs -> registered hApps
// $bridgeStore.getAggregate('agent-1') -> aggregate score
```

### GraphQL - Schema and Resolvers

GraphQL schema, resolvers, and subscriptions for Mycelix SDK.

```typescript
import {
  mycelixSchema,
  mycelixResolvers,
  createMycelixContext,
  MycelixGraphQL,
} from '@mycelix/sdk/graphql';

// === Full GraphQL Server ===
import { createYoga } from 'graphql-yoga';
import { createServer } from 'http';

const yoga = createYoga({
  schema: mycelixSchema,
  context: createMycelixContext({
    conductorUrl: 'ws://localhost:8888',
  }),
});

const server = createServer(yoga);
server.listen(4000, () => {
  console.log('GraphQL server running at http://localhost:4000/graphql');
});

// === Schema Overview ===
// Types: PoGQ, Reputation, Claim, Classification, Evidence, Participant, Gradient
// Queries: pogq, reputation, claim, claims, participant, bridge
// Mutations: createClaim, recordPositive, recordNegative, submitGradient
// Subscriptions: reputationUpdated, claimCreated, gradientSubmitted

// === Example Queries ===
const GET_REPUTATION = `
  query GetReputation($agentId: ID!) {
    reputation(agentId: $agentId) {
      agentId
      score
      positive
      negative
      updatedAt
    }
  }
`;

const CREATE_CLAIM = `
  mutation CreateClaim($input: CreateClaimInput!) {
    createClaim(input: $input) {
      id
      content
      classification {
        code
        empirical
        normative
        materiality
      }
      issuer
      createdAt
    }
  }
`;

const REPUTATION_SUBSCRIPTION = `
  subscription OnReputationUpdate($agentId: ID!) {
    reputationUpdated(agentId: $agentId) {
      agentId
      score
      delta
      reason
    }
  }
`;

// === Standalone Usage ===
import { execute, parse } from 'graphql';

const result = await execute({
  schema: mycelixSchema,
  document: parse(GET_REPUTATION),
  variableValues: { agentId: 'agent-123' },
  contextValue: createMycelixContext(),
});
```

### FHE - Fully Homomorphic Encryption

Privacy-preserving computation on encrypted data for voting, analytics, and secure aggregation.

```typescript
import { fhe } from '@mycelix/sdk';

// === Basic FHE Client ===
const client = new fhe.FHEClient({ preset: 'voting' });
await client.initialize();

// Encrypt values
const encryptedA = await client.encrypt(42);
const encryptedB = await client.encrypt(8);

// Compute on encrypted data (no decryption needed!)
const encryptedSum = await client.add(encryptedA, encryptedB);
const encryptedProduct = await client.multiply(encryptedA, encryptedB);

// Only the key holder can decrypt
const sum = await client.decrypt(encryptedSum);  // [50]
const product = await client.decrypt(encryptedProduct);  // [336]

// === Privacy-Preserving Voting ===
const votingClient = await fhe.createVotingClient();

// Create encrypted votes (voters can't see each other's votes)
const vote1 = await votingClient.createEncryptedVote('proposal-123', 'voter-1', 1);  // Yes
const vote2 = await votingClient.createEncryptedVote('proposal-123', 'voter-2', 0);  // No
const vote3 = await votingClient.createEncryptedVote('proposal-123', 'voter-3', 1);  // Yes

// Aggregate encrypted votes (tallier never sees individual votes!)
const aggregation = await votingClient.aggregateVotes([vote1, vote2, vote3]);

// Only the decryption committee can reveal final tally
const finalTally = await votingClient.decrypt(aggregation.sum);  // [2] - 2 yes votes

// === Secure Aggregation for FL ===
const analyticsClient = await fhe.createAnalyticsClient();

// Participants encrypt their local gradients
const encGradient1 = await analyticsClient.encrypt([0.1, 0.2, 0.15]);
const encGradient2 = await analyticsClient.encrypt([0.12, 0.18, 0.16]);
const encGradient3 = await analyticsClient.encrypt([0.09, 0.22, 0.14]);

// Server aggregates without seeing raw gradients
const encryptedSum = await analyticsClient.sum([encGradient1, encGradient2, encGradient3]);
const encryptedAvg = await analyticsClient.computeEncryptedAverage(
  [encGradient1, encGradient2, encGradient3],
  3
);

// === Threshold Decryption ===
// Requires k-of-n parties to decrypt (Shamir Secret Sharing)
const aggregator = new fhe.SecureAggregator({
  threshold: 3,  // Need 3 parties
  totalParties: 5,
  scheme: 'BFV',
});

await aggregator.initialize();

// Distribute key shares to 5 parties
const shares = await aggregator.distributeShares();

// Participants submit encrypted values
await aggregator.submitEncrypted('participant-1', encryptedValue1);
await aggregator.submitEncrypted('participant-2', encryptedValue2);

// Aggregate (still encrypted)
const result = await aggregator.aggregate();

// Threshold decryption - requires 3 shares
const partialDecrypts = await Promise.all([
  aggregator.partialDecrypt(result, shares[0]),
  aggregator.partialDecrypt(result, shares[2]),
  aggregator.partialDecrypt(result, shares[4]),
]);
const finalResult = await aggregator.combinePartialDecryptions(partialDecrypts);

// === FHE Presets ===
// 'development' - Fast, less secure (for testing)
// 'voting' - Optimized for tallying operations
// 'analytics' - Optimized for aggregation and statistics
// 'production' - Maximum security (slower)
```

### FL Hub - Federated Learning Hub

Comprehensive session management, model registry, and differential privacy for production FL deployments.

```typescript
import { flHub } from '@mycelix/sdk';

// === Session Coordinator ===
const coordinator = new flHub.FLHubCoordinator({
  maxConcurrentSessions: 10,
  defaultRoundsPerSession: 100,
  privacyBudget: { epsilon: 1.0, delta: 1e-5 },
});

// Create a new training session
const session = await coordinator.createSession({
  name: 'medical-imaging-v2',
  modelId: 'resnet50-chest-xray',
  minParticipants: 5,
  maxParticipants: 50,
  targetRounds: 100,
  aggregationMethod: 'trust_weighted',
  privacyConfig: {
    enableDifferentialPrivacy: true,
    epsilon: 0.5,
    delta: 1e-6,
    clipNorm: 1.0,
  },
});

console.log(`Session created: ${session.sessionId}`);

// Register participants
await coordinator.registerParticipant(session.sessionId, {
  participantId: 'hospital-stanford',
  dataSize: 10000,
  computeCapability: 'gpu',
});

// Start training
await coordinator.startSession(session.sessionId);

// Submit updates (with automatic privacy noise)
await coordinator.submitUpdate(session.sessionId, {
  participantId: 'hospital-stanford',
  round: 1,
  gradients: new Float64Array([0.1, 0.2, 0.15, ...]),
  metrics: { loss: 0.25, accuracy: 0.89 },
});

// Get session status
const status = coordinator.getSessionStatus(session.sessionId);
console.log(`Round ${status.currentRound}/${status.targetRounds}`);
console.log(`Participants: ${status.participantCount}`);
console.log(`Privacy budget remaining: ${status.privacyBudget.remaining}`);

// === Model Registry ===
const registry = new flHub.ModelRegistry();

// Register a trained model
const modelEntry = await registry.registerModel({
  modelId: 'resnet50-chest-xray-v2',
  version: '2.0.0',
  architecture: 'ResNet50',
  trainingSessionId: session.sessionId,
  metrics: {
    accuracy: 0.94,
    f1Score: 0.92,
    auc: 0.96,
  },
  parameters: modelWeights,  // serialized weights
});

// Save checkpoints during training
await registry.saveCheckpoint(session.sessionId, {
  round: 50,
  weights: intermediateWeights,
  metrics: { loss: 0.15 },
});

// List all model versions
const models = registry.listModels('resnet50-chest-xray');
console.log(models.map(m => `v${m.version}: ${m.metrics.accuracy}`));

// Get latest model
const latest = registry.getLatestModel('resnet50-chest-xray');

// Deploy model
await registry.deployModel(latest.modelId, {
  environment: 'production',
  replicas: 3,
});

// === Differential Privacy Manager ===
const privacyManager = new flHub.PrivacyManager({
  totalEpsilon: 10.0,
  totalDelta: 1e-5,
  accountingMethod: 'rdp',  // Rényi DP for tighter bounds
});

// Check if operation is allowed
const canProceed = privacyManager.canSpend(0.1, 1e-7);

// Record privacy consumption
if (canProceed) {
  privacyManager.recordConsumption(0.1, 1e-7, 'round_50_aggregation');
}

// Get remaining budget
const remaining = privacyManager.getRemainingBudget();
console.log(`Epsilon remaining: ${remaining.epsilon}`);
console.log(`Operations: ${remaining.history.length}`);

// === Private Aggregator ===
// Adds calibrated noise for differential privacy
const aggregator = new flHub.PrivateAggregator({
  epsilon: 0.5,
  delta: 1e-6,
  clipNorm: 1.0,
  noiseMechanism: 'gaussian',
});

// Aggregate with automatic DP noise
const privateResult = await aggregator.aggregate(participantUpdates);
console.log(`Aggregated ${privateResult.participantCount} updates`);
console.log(`Noise scale: ${privateResult.noiseScale}`);
```

### Mobile - Mobile Wallet SDK

Complete mobile wallet with biometric authentication, credential storage, and QR code utilities.

```typescript
import { mobile } from '@mycelix/sdk';

// === Wallet Manager ===
const wallet = mobile.createWalletManager();

// Create a new wallet with biometric protection
const metadata = await wallet.createWallet('My Mycelix Wallet', '123456', {
  enableBiometrics: true,
  autoLockTimeout: 300000,  // 5 minutes
  backupEnabled: true,
});

console.log(`Wallet created: ${metadata.walletId}`);
console.log(`Accounts: ${metadata.accounts.length}`);

// Unlock wallet
await wallet.unlock('123456');
// Or with biometrics
await wallet.unlockWithBiometrics();

// Get default account
const account = wallet.getDefaultAccount();
console.log(`Agent ID: ${account.agentId}`);
console.log(`Public Key: ${account.publicKey}`);

// Sign transactions
const txData = new TextEncoder().encode(JSON.stringify({
  action: 'transfer',
  amount: 100,
  recipient: 'did:mycelix:recipient123',
}));
const signature = await wallet.sign(account.accountId, txData);

// Verify signatures
const publicKey = wallet.getPublicKey(account.accountId);
const isValid = await wallet.verify(publicKey!, txData, signature);

// Create additional accounts
const newAccount = await wallet.createAccount('Trading Account');

// Export/Import for backup
const backup = await wallet.export('backup-password');
// Store backup securely...

// Restore from backup
await wallet.import(backup, 'backup-password');

// Lock wallet
wallet.lock();

// === Biometric Authentication ===
const biometrics = mobile.createBiometricManager();

// Check availability
const available = await biometrics.isAvailable();
const capabilities = await biometrics.getCapabilities();
console.log(`Biometrics available: ${available}`);
console.log(`Types: ${capabilities.availableTypes.join(', ')}`);

// Authenticate
const result = await biometrics.authenticate('Confirm transaction');
if (result.success) {
  console.log(`Authenticated with: ${result.biometricType}`);
} else {
  console.log(`Failed: ${mobile.biometricErrorToMessage(result.error!)}`);
}

// === QR Code Utilities ===

// Create wallet connect QR for dApp pairing
const connectPayload = mobile.createWalletConnectPayload(
  'marketplace-happ',
  'Mycelix Marketplace',
  ['read_profile', 'sign_transactions', 'view_balance'],
  'https://marketplace.mycelix.net/callback'
);
const qrData = mobile.encodeQRPayload(connectPayload);
// Display qrData as QR code...

// Create credential request QR
const credentialRequest = new mobile.CredentialRequestBuilder()
  .request('IdentityCredential', true)  // required
  .requestClaims('EducationCredential', ['degree', 'institution'], true)
  .requestWithPredicate('ReputationCredential', [
    { claim: 'trustScore', operator: '>=', value: 0.7 }
  ], false)  // optional
  .build();

const requestPayload = mobile.createCredentialRequestPayload(
  'employer-verifier',
  'Job Application Portal',
  credentialRequest
);
const requestQR = mobile.encodeQRPayload(requestPayload);

// Create payment request QR
const paymentPayload = mobile.createPaymentRequestPayload(
  'merchant-did',
  100.50,
  'MCX',
  'Coffee Shop Payment',
  'order-12345'
);
const paymentQR = mobile.encodeQRPayload(paymentPayload);

// Process scanned QR codes
const scanResult = mobile.processScannedQR(scannedData);
if (scanResult.success) {
  const action = mobile.getQRPayloadAction(scanResult.payload.type);
  console.log(`Action required: ${action}`);

  switch (scanResult.payload.type) {
    case 'wallet_connect':
      // Show permission dialog for dApp connection
      break;
    case 'credential_request':
      // Show credential selection UI
      break;
    case 'payment_request':
      // Show payment confirmation
      break;
  }
} else {
  console.log(`Scan failed: ${scanResult.error}`);
}

// Validate QR payloads
const validation = mobile.validateQRPayload(payload);
if (!validation.valid) {
  console.log(`Invalid: ${validation.errors.join(', ')}`);
}

// Check expiration
if (mobile.isPayloadExpired(payload)) {
  console.log('This QR code has expired');
}

// === Contact Exchange ===
const contactPayload = mobile.createContactExchangePayload({
  agentId: 'my-agent-id',
  displayName: 'Alice',
  publicKey: myPublicKey,
  profilePicture: 'ipfs://...',
  metadata: { organization: 'Mycelix Labs' },
});
```

## Innovations

Revolutionary new systems that extend the Mycelix ecosystem with advanced capabilities:

| Module | Description |
|--------|-------------|
| **Trust Markets** | Prediction markets for trust claims - "Will Alice's reputation stay > 0.7?" |
| **Private Queries** | Privacy-preserving cross-hApp analytics via FHE and differential privacy |
| **Epistemic Agents** | Transform AI agents into verified knowledge sources with calibrated confidence |
| **Civic Feedback Loop** | Auto-propagation of decisions across Justice-Knowledge-Governance |
| **Self-Healing Workflows** | Resilient workflow orchestration with saga compensation patterns |
| **Constitutional AI** | DAO-governed rules for AI agent behavior with violation tracking |

### Trust Markets

Create prediction markets for trust-related claims:

```typescript
import { innovations } from '@mycelix/sdk';

const trustMarkets = new innovations.TrustMarketService();

// Create a market
const market = trustMarkets.createMarket({
  subjectDid: 'did:mycelix:alice',
  claim: {
    type: 'reputation_maintenance',
    threshold: 0.7,
    contextHapps: ['governance', 'finance'],
    durationMs: 30 * 24 * 60 * 60 * 1000,
    operator: 'gte',
  },
  title: "Will Alice's reputation stay >= 0.7?",
  creatorStake: 100,
});

// Submit trades
await trustMarkets.submitTrade({
  marketId: market.id,
  participantId: 'did:mycelix:bob',
  outcome: 'yes',
  shares: 50,
  stake: { monetary: 100, reputation: 0.05 },
});

// Get market-implied probability
const prob = trustMarkets.getMarket(market.id)?.state.impliedProbabilities.get('yes');
```

### Epistemic Agents

Create agents that produce verified, calibrated knowledge claims:

```typescript
import { innovations, EmpiricalLevel, NormativeLevel, MaterialityLevel } from '@mycelix/sdk';

const agent = innovations.createEpistemicAgentFromDomain('benefits', {
  name: 'SNAP Navigator',
});

// Agent claims are classified on a 3D epistemic grid
const claim: innovations.EpistemicClaim = {
  id: 'claim-001',
  text: 'SNAP eligibility is based on income, not employment status.',
  position: {
    empirical: EmpiricalLevel.Cryptographic,
    normative: NormativeLevel.Universal,
    materiality: MaterialityLevel.Persistent,
  },
  classificationCode: 'E3-N3-M2',
  confidence: 0.88,
  calibratedConfidence: 0.85,
  // ... additional fields
};
```

### Civic Feedback Loop

Automatically propagate justice decisions to create legal precedents:

```typescript
import { innovations } from '@mycelix/sdk';

const feedbackLoop = innovations.createCivicFeedbackLoop({
  autoPropagate: true,
  enableConflictDetection: true,
});
await feedbackLoop.start();

// Propagate a justice decision
const propagation = await feedbackLoop.propagateJusticeDecision({
  caseId: 'case-001',
  decisionId: 'decision-001',
  type: 'final_judgment',
  summary: 'Employment does not disqualify SNAP eligibility.',
  reasoning: 'Based on federal income guidelines.',
  principles: ['SNAP is income-based'],
  legalDomain: 'administrative',
  confidence: 0.95,
  setsPrecedent: true,
  // ... additional fields
});

console.log(`Precedents created: ${propagation.precedentsEstablished}`);
```

### Self-Healing Workflows

Create resilient workflows with automatic compensation:

```typescript
import { innovations } from '@mycelix/sdk';

const workflow = innovations.createWorkflow('benefit-application', '1.0.0')
  .description('Process benefit application')
  .timeout(60000)
  .compensateOnFailure(true)
  .step('verify-identity', 'identity', async (input, ctx) => {
    ctx.log('Verifying identity...');
    return { verified: true };
  }, { critical: true })
  .step('check-eligibility', 'eligibility', async (input, ctx) => {
    ctx.log('Checking eligibility...');
    return { eligible: true };
  }, { dependsOn: ['verify-identity'] })
  .build();

const orchestrator = innovations.createOrchestrator();
const result = await orchestrator.execute(workflow, {
  initiatorDid: 'did:mycelix:caseworker',
  input: { applicantId: '001' },
});
```

### Constitutional AI

Enforce constitutional rules on AI agent behavior:

```typescript
import { innovations } from '@mycelix/sdk';

const governor = await innovations.getConstitutionalGovernor();

// Check content against constitutional rules
const check = governor.checkContent(
  'agent-001',
  'Here is sensitive financial information...',
  { domain: 'finance' }
);

if (!check.allowed) {
  console.log(`Blocked: ${check.reason}`);
  console.log(`Violations: ${check.violations.length}`);
}

// Generate compliance audit
const audit = governor.generateAudit(Date.now() - 3600000);
console.log(`Compliance rate: ${audit.complianceRate * 100}%`);
```

### Integration Examples

See the integration examples demonstrating all modules working together:

```typescript
import { innovations } from '@mycelix/sdk';

// Run comprehensive housing voucher scenario
await innovations.comprehensiveCivicScenario();

// Run individual integration examples
await innovations.trustMarketCalibrationExample();
await innovations.epistemicCivicIntegrationExample();
await innovations.workflowConstitutionalExample();
```

## Testing

```bash
# Run all tests
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage

# Run benchmarks
npm run bench

# Type checking
npm run typecheck

# Lint
npm run lint
```

## Documentation

```bash
# Generate API documentation
npm run docs

# Generate and serve locally
npm run docs:serve
# Then open http://localhost:8080
```

API documentation is generated using [TypeDoc](https://typedoc.org/) and includes:
- Complete API reference for all modules
- Type definitions and interfaces
- Code examples from JSDoc comments
- Module hierarchy and relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Application                            │
├─────────────────────────────────────────────────────────────────┤
│                      Developer Tools                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  CLI: npx @mycelix/sdk init | generate | validate | status │ │
│  └────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│              hApp Integrations (12 Civilizational OS Domains)    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │Identity│ │Finance │ │Property│ │ Energy │ │ Media  │        │
│  ├────────┤ ├────────┤ ├────────┤ ├────────┤ ├────────┤        │
│  │Governan│ │Justice │ │Knowledg│ │  Mail  │ │Market  │ + more │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
├─────────────────────────────────────────────────────────────────┤
│                   @mycelix/sdk (TypeScript)                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  MATL  │ │   FL   │ │Epistemic│ │ Bridge │ │Security│        │
│  │ 34% BFT│ │FedAvg  │ │ E×N×M  │ │Cross-  │ │Crypto  │        │
│  │  PoGQ  │ │Krum    │ │ Claims │ │  hApp  │ │RateLimit│       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐                  │
│  │ Client │ │ Errors │ │Validation│ │Signals │                  │
│  │Holochain│ │Handling│ │Zod Schemas│ │Events │                  │
│  └────────┘ └────────┘ └──────────┘ └────────┘                  │
├─────────────────────────────────────────────────────────────────┤
│                    WebSocket Connection                          │
├─────────────────────────────────────────────────────────────────┤
│                   Holochain Conductor (0.6.x)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │   Bridge DNAs   │  │   Domain DNAs   │  │   Trust DNA    │   │
│  │ (8 bridge zomes)│  │ (hApp-specific) │  │  (FL + MATL)   │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│               Mycelix SDK (Rust) - Shared Types                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Imports

```typescript
// Named imports (recommended)
import { matl, fl, epistemic, bridge, security, errors } from '@mycelix/sdk';

// Direct function imports
import {
  createPoGQ,
  isByzantine,
  createReputation,
  MAX_BYZANTINE_TOLERANCE,
  // Validation
  validateOrThrow,
  validateSafe,
  IdentitySchemas,
  FinanceSchemas,
  // Signals
  createSignalManager,
  createIdentitySignals,
} from '@mycelix/sdk';

// Submodule imports (tree-shakeable)
import * as matl from '@mycelix/sdk/matl';
import * as fl from '@mycelix/sdk/fl';
import * as flHub from '@mycelix/sdk/fl-hub';
import * as fhe from '@mycelix/sdk/fhe';
import * as mobile from '@mycelix/sdk/mobile';
import * as ai from '@mycelix/sdk/ai';
import * as rtc from '@mycelix/sdk/rtc';
import * as security from '@mycelix/sdk/security';
import * as validation from '@mycelix/sdk/validation';
import * as signals from '@mycelix/sdk/signals';
import * as cli from '@mycelix/sdk/cli';
import * as resilience from '@mycelix/sdk/resilience';
import * as observability from '@mycelix/sdk/observability';
import * as react from '@mycelix/sdk/react';
import * as svelte from '@mycelix/sdk/svelte';
import * as graphql from '@mycelix/sdk/graphql';

// hApp Integration imports (all 12 Civilizational OS domains)
import { getMailTrustService } from '@mycelix/sdk/integrations/mail';
import { getMarketplaceService } from '@mycelix/sdk/integrations/marketplace';
import { getPraxisService } from '@mycelix/sdk/integrations/praxis';
import { getSupplyChainService } from '@mycelix/sdk/integrations/supplychain';
import { IdentityBridgeClient } from '@mycelix/sdk/integrations/identity';
import { FinanceBridgeClient } from '@mycelix/sdk/integrations/finance';
import { PropertyBridgeClient } from '@mycelix/sdk/integrations/property';
import { EnergyBridgeClient } from '@mycelix/sdk/integrations/energy';
import { MediaBridgeClient } from '@mycelix/sdk/integrations/media';
import { GovernanceBridgeClient } from '@mycelix/sdk/integrations/governance';
import { JusticeBridgeClient } from '@mycelix/sdk/integrations/justice';
import { KnowledgeBridgeClient } from '@mycelix/sdk/integrations/knowledge';
```

## Requirements

- Node.js 20+
- TypeScript 5.3+ (for development)
- Holochain 0.6.x (for conductor tests)

## Contributing

Contributions welcome! Please read our contributing guidelines.

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass: `npm test`
5. Submit a pull request

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/Luminous-Dynamics/mycelix)
- [Holochain](https://holochain.org)
- [Luminous Dynamics](https://luminousdynamics.org)
