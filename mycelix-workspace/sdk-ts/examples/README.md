# Mycelix SDK Examples

This directory contains practical examples demonstrating how to use the Mycelix
SDK.

## Prerequisites

Make sure you have built the SDK first:

```bash
npm run build
```

## Running Examples

Examples can be run with `ts-node` or by compiling first:

```bash
# Using npx ts-node
npx ts-node examples/01-quick-start.ts

# Or compile and run
npx tsc examples/01-quick-start.ts --outDir examples/dist --esModuleInterop
node examples/dist/01-quick-start.js
```

## Example Overview

### 01 - Quick Start

**File:** `01-quick-start.ts`

Basic introduction to the SDK:

- SDK health checks
- Quick trust assessment with `checkTrust()`
- Building reputation over time
- Low-level MATL operations

Best for: Getting started with the SDK.

### 02 - Federated Learning

**File:** `02-federated-learning.ts`

Byzantine-fault-tolerant federated learning:

- Setting up FL coordinators
- Registering participants with reputation
- Running training rounds
- Byzantine attack simulation with Krum
- Different aggregation methods

Best for: Understanding distributed ML with trust.

### 03 - Cross-hApp Reputation

**File:** `03-cross-happ-reputation.ts`

Multi-hApp reputation aggregation:

- Setting up `ReputationAggregator`
- Registering multiple hApps
- Aggregating reputation across hApps
- Bridge message protocol
- Trust decisions based on aggregate reputation

Best for: Building Holochain ecosystems.

### 04 - Epistemic Claims

**File:** `04-epistemic-claims.ts`

3D truth classification system:

- Understanding Empirical/Normative/Materiality levels
- Creating claims with the fluent builder
- Adding evidence to claims
- Validating against standards
- Expiration handling

Best for: Implementing verifiable claims.

### 05 - Secure Messaging

**File:** `05-secure-messaging.ts`

Security primitives and patterns:

- Secure random generation
- Cryptographic hashing and HMAC
- Signed messages with verification
- Rate limiting
- Timing-safe comparisons
- Secure secrets management
- BFT validation utilities
- Security audit logging

Best for: Building secure applications.

### 06 - Full Ecosystem

**File:** `06-full-ecosystem.ts`

Complete integration example:

- Healthcare FL network scenario
- Configuration management
- Cross-hApp reputation setup
- Compliance claims
- Trust-verified participation
- Federated learning round
- Secure result signing
- Rate-limited API access

Best for: Understanding how all modules work together.

---

## hApp Integration Examples

### 07 - Mail Integration

**File:** `07-mail-integration.ts`

Email sender trust and verification:

- Recording positive/negative email interactions
- Sender trust levels (unknown → verified)
- Email claims with DKIM/SPF/DMARC verification
- Email filtering patterns based on trust

Best for: Building email trust systems.

### 08 - Marketplace Integration

**File:** `08-marketplace-integration.ts`

Transaction reputation and scam detection:

- Recording transaction outcomes
- Seller/buyer profile management
- Listing verification with scam risk scores
- Purchase decision flows
- Verified seller badges

Best for: Building marketplace trust systems.

### 09 - EduNet Integration

**File:** `09-edunet-integration.ts`

Educational credentials and verification:

- Issuing course completion certificates
- Skill certifications with proficiency levels
- Credential verification and expiration
- Learner profiles with trust scores
- Hiring verification patterns

Best for: Building credentialing systems.

### 10 - SupplyChain Integration

**File:** `10-supplychain-integration.ts`

Product provenance and chain verification:

- Recording supply chain checkpoints
- Multiple evidence types (IoT, GPS, blockchain, photos)
- Full provenance chain retrieval
- Chain integrity verification
- Handler trust profiles
- Consumer product verification

Best for: Building traceability systems.

### Cross-hApp Ecosystem

**File:** `cross-happ-ecosystem.ts`

All integrations working together:

- Complete supply chain journey
- All 4 hApp integrations (Mail, Marketplace, EduNet, SupplyChain)
- Bridge-based reputation sharing
- Federated Learning integration

Best for: Understanding the full Mycelix ecosystem.

---

## Civilizational OS Bridge Examples

### 11 - Civilizational Bridge

**File:** `11-civilizational-bridge.ts`

Full Civilizational OS integration with all 8 domains:

- **Identity**: Verified identity creation with MATL reputation
- **Finance**: Trust-weighted payment processing
- **Property**: Asset ownership verification
- **Energy**: Renewable energy listing and purchase
- **Media**: Content verification (referenced)
- **Governance**: Community proposal creation and voting
- **Justice**: Dispute resolution framework (referenced)
- **Knowledge**: Epistemic claims with fact-checking

Plus new SDK features:

- **Validation**: Runtime Zod schema validation for all inputs
- **Signals**: Real-time event subscriptions from conductor
- **Cross-hApp**: Aggregate reputation across all domains

Best for: Building complete Civilizational OS applications.

---

## UESS Storage Examples

### 13 - Epistemic Storage

**File:** `13-epistemic-storage.ts`

Classification-aware data storage:

- E/N/M classification system explained
- Storing ephemeral data (M0 → Memory)
- Storing temporal data (M1 → Local)
- Storing persistent data (M2 → DHT)
- Storing immutable data (M3 → IPFS)
- Querying by classification
- Data verification and deletion

Best for: Understanding classification-based storage routing.

### 14 - GDPR Compliance

**File:** `14-gdpr-compliance.ts`

Cryptographic shredding for "right to be forgotten":

- Per-subject encryption keys
- Key shredding for data erasure
- Audit trail for compliance proof
- Secure key overwriting (DoD 5220.22-M)
- Compliance reports for regulators
- Subject isolation verification

Best for: Building GDPR-compliant applications.

### 15 - Advanced Storage Patterns

**File:** `15-storage-advanced.ts`

High-performance and distributed data patterns:

- **Batch Operations**: High-throughput bulk storage
- **CRDT Merge Strategies**: LWW, MVR, OR-Set, G-Counter, PN-Counter
- **Observability**: Metrics, tracing, health checks, Prometheus export
- **Cross-hApp Storage**: Shared namespaces, credential sharing, reputation aggregation

Best for: Production-grade storage implementations.

---

## Module Quick Reference

### Core Modules

| Module       | Import                                                                       | Key Features                          |
| ------------ | ---------------------------------------------------------------------------- | ------------------------------------- |
| `matl`       | `import { matl } from '@mycelix/sdk'`                                        | PoGQ, Reputation, Byzantine detection |
| `fl`         | `import { fl } from '@mycelix/sdk'`                                          | FL coordination, aggregation methods  |
| `epistemic`  | `import { epistemic } from '@mycelix/sdk'`                                   | Claims, evidence, standards           |
| `bridge`     | `import { bridge } from '@mycelix/sdk'`                                      | Cross-hApp communication              |
| `security`   | `import { security } from '@mycelix/sdk'`                                    | Crypto, rate limiting, audit          |
| `config`     | `import { config } from '@mycelix/sdk'`                                      | Configuration management              |
| `utils`      | `import { utils } from '@mycelix/sdk'`                                       | Helper functions                      |
| `storage`    | `import { storage } from '@mycelix/sdk'`                                     | UESS epistemic storage, backends      |
| `validation` | `import { validateOrThrow, IdentitySchemas } from '@mycelix/sdk/validation'` | Zod schemas for all domains           |
| `signals`    | `import { createSignalManager } from '@mycelix/sdk/signals'`                 | Real-time event handlers              |
| `cli`        | `npx @mycelix/sdk init`                                                      | Project scaffolding tools             |

### Domain Bridge Integrations (Civilizational OS)

| Domain       | Import                                                                          | Key Features                        |
| ------------ | ------------------------------------------------------------------------------- | ----------------------------------- |
| `identity`   | `import { IdentityBridgeClient } from '@mycelix/sdk/integrations/identity'`     | DID, credentials, trust attestation |
| `finance`    | `import { FinanceBridgeClient } from '@mycelix/sdk/integrations/finance'`       | Payments, credit scores, loans      |
| `property`   | `import { PropertyBridgeClient } from '@mycelix/sdk/integrations/property'`     | Ownership, collateral, transfers    |
| `energy`     | `import { EnergyBridgeClient } from '@mycelix/sdk/integrations/energy'`         | Energy trading, grid balance        |
| `media`      | `import { MediaBridgeClient } from '@mycelix/sdk/integrations/media'`           | Content, licensing, royalties       |
| `governance` | `import { GovernanceBridgeClient } from '@mycelix/sdk/integrations/governance'` | Proposals, voting, delegation       |
| `justice`    | `import { JusticeBridgeClient } from '@mycelix/sdk/integrations/justice'`       | Disputes, arbitration, evidence     |
| `knowledge`  | `import { KnowledgeBridgeClient } from '@mycelix/sdk/integrations/knowledge'`   | Claims, fact-checking, endorsements |

### hApp-Specific Adapters

| Adapter       | Import                                                                          | Key Features            |
| ------------- | ------------------------------------------------------------------------------- | ----------------------- |
| `mail`        | `import { getMailTrustService } from '@mycelix/sdk/integrations/mail'`          | Email sender trust      |
| `marketplace` | `import { getMarketplaceService } from '@mycelix/sdk/integrations/marketplace'` | Transaction reputation  |
| `edunet`      | `import { getEduNetService } from '@mycelix/sdk/integrations/edunet'`           | Educational credentials |
| `supplychain` | `import { getSupplyChainService } from '@mycelix/sdk/integrations/supplychain'` | Product provenance      |

## Common Patterns

### Quick Trust Check

```typescript
import { checkTrust } from '@mycelix/sdk';

const result = checkTrust('agent-id', 0.85, 0.9, 0.15);
if (result.trustworthy) {
  // Allow operation
}
```

### Signed Messages

```typescript
import { signMessage, verifyMessage, security } from '@mycelix/sdk';

const key = security.secureRandomBytes(32);
const signed = await signMessage({ action: 'transfer' }, key);
const { valid } = await verifyMessage(signed, key);
```

### FL Round

```typescript
import {
  createSimpleFLCoordinator,
  createGradientUpdate,
  runFLRound,
} from '@mycelix/sdk';

const coordinator = createSimpleFLCoordinator({ minParticipants: 3 });
coordinator.registerParticipant('p1');
// ...register more

const updates = [
  createGradientUpdate('p1', 1, [0.1, 0.2], { batchSize: 32, loss: 0.5 }),
  // ...more updates
];

const summary = runFLRound(coordinator, updates);
```

### Epistemic Claims

```typescript
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '@mycelix/sdk';

const myClaim = claim('Agent verified identity')
  .withClassification(
    EmpiricalLevel.E3_Cryptographic,
    NormativeLevel.N2_Network,
    MaterialityLevel.M2_Persistent
  )
  .withIssuer('verification-service')
  .build();
```

### Epistemic Storage (UESS)

```typescript
import { storage, EmpiricalLevel, NormativeLevel, MaterialityLevel } from '@mycelix/sdk';

// Create storage instance
const store = storage.createEpistemicStorage({ agentId: 'agent:alice' });

// Store with classification (routes to appropriate backend)
await store.store('profile:alice', { name: 'Alice' }, {
  empirical: EmpiricalLevel.E3_Cryptographic,
  normative: NormativeLevel.N2_Network,
  materiality: MaterialityLevel.M2_Persistent, // → DHT
});

// Retrieve
const profile = await store.retrieve('profile:alice');
```

### GDPR Cryptographic Shredding

```typescript
import { storage } from '@mycelix/sdk';

const manager = await storage.createCryptoShreddingManager();
const keyId = await manager.generateKey('user:alice');

// Encrypt user data
const encrypted = await manager.encrypt(keyId, userData);
manager.registerResource(keyId, 'profile:alice');

// GDPR erasure request - destroy the key
await storage.executeGDPRErasure(manager, 'user:alice', 'legal-team');
// All data encrypted with this key is now unrecoverable
```

---

## Privacy & Security Examples

### 17 - FHE Privacy-Preserving Voting

**File:** `17-fhe-privacy-voting.ts`

Fully Homomorphic Encryption for privacy-preserving computation:

- Basic FHE operations (add, multiply on encrypted data)
- Privacy-preserving voting with encrypted ballots
- Vote aggregation without revealing individual votes
- Threshold decryption (3-of-5 authorities)
- Secure FL gradient aggregation with FHE

Best for: Building privacy-preserving voting and analytics systems.

### 18 - FL Hub Session Management

**File:** `18-fl-hub-session-management.ts`

Production-grade Federated Learning infrastructure:

- FL Hub coordinator for session lifecycle management
- Participant registration with compute capabilities
- Training round orchestration with progress tracking
- Model registry for versioning and checkpoints
- Differential privacy budget management
- Private aggregation with calibrated noise injection

Best for: Deploying production FL systems with privacy guarantees.

### 19 - Mobile Wallet SDK

**File:** `19-mobile-wallet-sdk.ts`

Complete mobile wallet implementation:

- Wallet creation with PIN and biometric protection
- Multi-account management
- Transaction signing and verification
- Biometric authentication (fingerprint, face)
- QR code generation (wallet connect, payments, credentials)
- Credential request builder for selective disclosure
- Contact exchange and deep links
- Wallet backup and restore

Best for: Building mobile dApps with wallet integration.

---

## Advanced Module Reference

### Privacy & Security Modules

| Module   | Import                               | Key Features                             |
| -------- | ------------------------------------ | ---------------------------------------- |
| `fhe`    | `import { fhe } from '@mycelix/sdk'` | Homomorphic encryption, secure voting    |
| `flHub`  | `import { flHub } from '@mycelix/sdk'` | Session management, model registry, DP   |
| `mobile` | `import { mobile } from '@mycelix/sdk'` | Wallet, biometrics, QR codes            |
| `ai`     | `import { ai } from '@mycelix/sdk'`  | Model inference, embeddings, RAG         |
| `rtc`    | `import { rtc } from '@mycelix/sdk'` | WebRTC, signaling, media handling        |

### FHE Quick Start

```typescript
import { fhe } from '@mycelix/sdk';

// Create voting client
const client = await fhe.createVotingClient();

// Encrypt votes
const vote = await client.createEncryptedVote('prop-1', 'voter-1', 1);

// Aggregate without seeing votes
const result = await client.aggregateVotes([vote1, vote2, vote3]);

// Only authorized parties can decrypt
const tally = await client.decrypt(result.sum);
```

### FL Hub Quick Start

```typescript
import { flHub } from '@mycelix/sdk';

// Create coordinator
const coordinator = new flHub.FLHubCoordinator({
  privacyBudget: { epsilon: 1.0, delta: 1e-5 },
});

// Create and run session
const session = await coordinator.createSession({
  name: 'medical-imaging',
  minParticipants: 3,
});

await coordinator.registerParticipant(session.sessionId, { participantId: 'hospital-1' });
await coordinator.startSession(session.sessionId);
```

### Mobile Wallet Quick Start

```typescript
import { mobile } from '@mycelix/sdk';

// Create wallet
const wallet = mobile.createWalletManager();
await wallet.createWallet('My Wallet', '123456', { enableBiometrics: true });

// Sign transactions
await wallet.unlock('123456');
const signature = await wallet.sign(account.accountId, txData);

// Generate QR codes
const qr = mobile.createPaymentRequestPayload('merchant', 10.50, 'USD');
```

## Need Help?

- [Full API Documentation](../docs/API.md)
- [README](../README.md)
- [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix/issues)
