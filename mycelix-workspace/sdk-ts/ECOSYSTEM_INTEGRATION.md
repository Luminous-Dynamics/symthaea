# @mycelix/sdk Ecosystem Integration Guide

## Overview

The `@mycelix/sdk` TypeScript SDK provides client-side implementations of the Mycelix protocols, designed to integrate seamlessly with the existing Holochain-based ecosystem.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MYCELIX ECOSYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ Mycelix-Mail│  │ Marketplace │  │   Praxis    │  │  SupplyChain    │    │
│  │  (React)    │  │   (Next.js) │  │  (SvelteKit)│  │  (Dashboard)    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────┬─────────┘    │
│         │                │                │                  │              │
│         └────────────────┴────────────────┴──────────────────┘              │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      @mycelix/sdk (TypeScript)                       │   │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────┐ ┌──────────┐ ┌───────┐  │   │
│  │  │  MATL   │ │ Epistemic│ │ Bridge │ │ FL │ │ Security │ │ Utils │  │   │
│  │  │ (Trust) │ │ (Claims) │ │ (Msgs) │ │    │ │          │ │       │  │   │
│  │  └─────────┘ └──────────┘ └────────┘ └────┘ └──────────┘ └───────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Holochain Conductor (hc 0.6+)                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ mail.dna     │  │ market.dna   │  │ praxis.dna   │               │   │
│  │  │ trust_zome   │  │ listing_zome │  │ course_zome  │               │   │
│  │  │ mail_zome    │  │ review_zome  │  │ cert_zome    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points by hApp

### 1. Mycelix-Mail Integration

**Location**: `/srv/luminous-dynamics/Mycelix-Mail/`
**Status**: Production Ready
**Frontend**: React
**Backend**: Axum (Rust)

#### SDK Usage

```typescript
// ui/src/services/trust.ts
import {
  createClient,
  claim,
  EmpiricalLevel,
  NormativeLevel,
  meetsStandard,
  Standards,
  LocalBridge,
  createReputationQuery,
} from '@mycelix/sdk';

// Connect to Holochain conductor
const client = createClient({
  conductorUrl: 'ws://localhost:8888',
  appId: 'mycelix-mail',
  roleName: 'mail',
  zomeName: 'trust',
});

await client.connect();

// Verify sender trust before displaying email
async function verifySenderTrust(senderId: string): Promise<TrustLevel> {
  const result = await client.checkTrust({
    agentId: senderId,
    minimumScore: 0.7,
    happContext: 'mail',
  });

  return {
    trustworthy: result.trustworthy,
    score: result.score,
    level: getAssuranceLevel(result.score),
  };
}

// Create epistemic claim for email content
function createEmailClaim(email: Email): EpistemicClaim {
  return claim(email.subject)
    .withEmpirical(determineEmpiricalLevel(email))
    .withNormative(NormativeLevel.N2_Network)
    .addEvidence({
      type: 'email_headers',
      data: {
        dkim: email.dkimVerified,
        spf: email.spfPassed,
        senderDid: email.senderDid,
      },
    })
    .build();
}

// Map email assurance to Epistemic levels (E0-E4)
function determineEmpiricalLevel(email: Email): EmpiricalLevel {
  if (email.zkProofVerified) return EmpiricalLevel.E4_Consensus;
  if (email.credentialVerified) return EmpiricalLevel.E3_Cryptographic;
  if (email.dkimVerified && email.spfPassed) return EmpiricalLevel.E2_PrivateVerify;
  if (email.senderKnown) return EmpiricalLevel.E1_Testimonial;
  return EmpiricalLevel.E0_Unverified;
}
```

#### Integration File Structure

```
Mycelix-Mail/
├── ui/
│   ├── src/
│   │   ├── services/
│   │   │   ├── trust.ts          # SDK trust integration
│   │   │   ├── claims.ts         # Epistemic claims
│   │   │   └── bridge.ts         # Cross-hApp messaging
│   │   └── hooks/
│   │       ├── useTrust.ts       # React hook for trust
│   │       └── useClaims.ts      # React hook for claims
│   └── package.json              # Add @mycelix/sdk dependency
```

---

### 2. Mycelix-Marketplace Integration

**Location**: `/srv/luminous-dynamics/mycelix-marketplace/`
**Status**: Development
**Frontend**: Next.js (likely)
**Backend**: Rust + WASM

#### SDK Usage

```typescript
// src/lib/reputation.ts
import {
  createClient,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  FLCoordinator,
  AggregationMethod,
} from '@mycelix/sdk';

// Marketplace reputation service
class MarketplaceReputation {
  private client: MycelixClient;
  private flCoordinator: FLCoordinator;

  async initialize() {
    this.client = createClient({
      conductorUrl: process.env.CONDUCTOR_URL,
      appId: 'mycelix-marketplace',
      roleName: 'marketplace',
      zomeName: 'reputation',
    });

    // FL for Byzantine-resistant reputation aggregation
    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34, // Mycelix's 34% validated BFT
    });
  }

  // Record transaction outcome
  async recordTransaction(
    buyerId: string,
    sellerId: string,
    success: boolean,
    amount: number
  ) {
    // Record buyer's view of seller
    await this.client.recordReputation({
      targetAgentId: sellerId,
      happId: 'marketplace',
      score: success ? 1.0 : 0.0,
      evidence: {
        type: 'transaction',
        amount,
        success,
        timestamp: Date.now(),
      },
    });

    // Record seller's view of buyer
    await this.client.recordReputation({
      targetAgentId: buyerId,
      happId: 'marketplace',
      score: success ? 1.0 : 0.0,
      evidence: {
        type: 'transaction',
        amount,
        success,
        timestamp: Date.now(),
      },
    });
  }

  // Get seller trust score with cross-hApp data
  async getSellerTrust(sellerId: string) {
    return this.client.getCrossHappReputation({
      agentId: sellerId,
      happs: ['marketplace', 'mail', 'praxis'], // Cross-reference
      aggregationMethod: 'weighted',
    });
  }
}
```

---

### 3. Mycelix-Praxis Integration

**Location**: `/srv/luminous-dynamics/mycelix-praxis/`
**Status**: HC 0.6 Migration
**Frontend**: Web app

#### SDK Usage

```typescript
// apps/web/src/lib/credentials.ts
import {
  claim,
  EmpiricalLevel,
  MaterialityLevel,
  ClaimVerifier,
  SchemaRegistry,
} from '@mycelix/sdk';

// Educational credential verification
class PraxisCredentials {
  private verifier: ClaimVerifier;
  private schemas: SchemaRegistry;

  constructor() {
    this.verifier = new ClaimVerifier();
    this.schemas = new SchemaRegistry();

    // Register credential schemas
    this.schemas.registerSchema({
      id: 'course-completion',
      version: { major: 1, minor: 0, patch: 0 },
      validate: (data) => this.validateCourseCompletion(data),
    });
  }

  // Issue course completion credential
  async issueCertificate(
    studentId: string,
    courseId: string,
    grade: number
  ): Promise<EpistemicClaim> {
    const certificate = claim(`Completed ${courseId} with grade ${grade}`)
      .withEmpirical(EmpiricalLevel.E3_Cryptographic) // Cryptographically signed
      .withMateriality(MaterialityLevel.M3_Critical)   // Educational credential
      .addEvidence({
        type: 'course_completion',
        data: {
          studentId,
          courseId,
          grade,
          completedAt: Date.now(),
          issuerDid: 'did:mycelix:praxis',
        },
      })
      .withTtl(365 * 24 * 60 * 60 * 1000) // 1 year validity
      .build();

    return certificate;
  }

  // Verify credential with privacy preservation
  async verifyCredential(credentialId: string, requiredLevel: number) {
    const result = await this.verifier.verify(credentialId, {
      minimumEmpiricalLevel: requiredLevel,
      requireProof: true,
    });

    return result;
  }
}
```

---

### 4. Mycelix-SupplyChain Integration

**Location**: `/srv/luminous-dynamics/mycelix-supplychain/`
**Status**: Alpha
**Dashboard**: React/Vue

#### SDK Usage

```typescript
// dashboard/src/services/provenance.ts
import {
  createClient,
  claim,
  EmpiricalLevel,
  EventPipeline,
  TimeSeriesAnalytics,
  MetricsCollector,
} from '@mycelix/sdk';

// Supply chain provenance tracking
class ProvenanceTracker {
  private client: MycelixClient;
  private analytics: TimeSeriesAnalytics;
  private metrics: MetricsCollector;
  private eventPipeline: EventPipeline<ProvenanceEvent>;

  async initialize() {
    this.client = createClient({
      conductorUrl: process.env.CONDUCTOR_URL,
      appId: 'mycelix-supplychain',
      roleName: 'supplychain',
      zomeName: 'provenance',
    });

    this.analytics = new TimeSeriesAnalytics();
    this.metrics = new MetricsCollector();

    // Set up event pipeline for provenance tracking
    this.eventPipeline = new EventPipeline<ProvenanceEvent>()
      .filter((e) => e.verified)
      .map((e) => this.enrichWithLocation(e))
      .tap((e) => this.recordMetrics(e));
  }

  // Record provenance checkpoint
  async recordCheckpoint(
    productId: string,
    location: string,
    handler: string,
    evidence: any
  ) {
    const checkpoint = claim(`Product ${productId} at ${location}`)
      .withEmpirical(this.getEvidenceLevel(evidence))
      .addEvidence({
        type: 'checkpoint',
        data: {
          productId,
          location,
          handler,
          timestamp: Date.now(),
          ...evidence,
        },
      })
      .build();

    await this.client.recordReputation({
      targetAgentId: handler,
      happId: 'supplychain',
      score: 1.0,
      evidence: {
        checkpointId: checkpoint.id,
        location,
      },
    });

    this.analytics.addPoint(Date.now(), 1);
    return checkpoint;
  }

  private getEvidenceLevel(evidence: any): EmpiricalLevel {
    if (evidence.iotSensor && evidence.gps && evidence.photo) {
      return EmpiricalLevel.E3_Cryptographic;
    }
    if (evidence.signature) {
      return EmpiricalLevel.E2_PrivateVerify;
    }
    return EmpiricalLevel.E1_Testimonial;
  }
}
```

---

## Shared Configuration

### Workspace package.json

Add to `/srv/luminous-dynamics/mycelix-workspace/package.json`:

```json
{
  "workspaces": [
    "sdk-ts",
    "happs/*"
  ],
  "dependencies": {
    "@mycelix/sdk": "workspace:*"
  }
}
```

### TypeScript Path Aliases

Add to each hApp's `tsconfig.json`:

```json
{
  "compilerOptions": {
    "paths": {
      "@mycelix/sdk": ["../../sdk-ts/dist"],
      "@mycelix/sdk/*": ["../../sdk-ts/dist/*"]
    }
  }
}
```

---

## API Parity with Rust SDK

The TypeScript SDK mirrors the Rust SDK (`/srv/luminous-dynamics/mycelix-workspace/sdk/`):

| Rust Module | TypeScript Module | Status |
|-------------|-------------------|--------|
| `matl/` | `matl/` | ✅ Complete |
| `epistemic/` | `epistemic/` | ✅ Complete |
| `bridge/` | `bridge/` | ✅ Complete |
| `credentials/` | (in epistemic) | ✅ Integrated |

### Type Mappings

```typescript
// Rust -> TypeScript type mappings

// matl/mod.rs -> matl/index.ts
ProofOfGradientQuality  ->  ProofOfGradientQuality
ReputationScore         ->  ReputationScore
CompositeScore          ->  CompositeScore

// epistemic/mod.rs -> epistemic/index.ts
EmpiricalLevel (E0-E4)  ->  EmpiricalLevel (enum)
NormativeLevel (N0-N3)  ->  NormativeLevel (enum)
MaterialityLevel (M0-M3) -> MaterialityLevel (enum)

// bridge/mod.rs -> bridge/index.ts
BridgeMessage           ->  BridgeMessage
MessageType             ->  BridgeMessageType
```

---

## Deployment Patterns

### Development

```bash
# In each hApp directory
npm link @mycelix/sdk

# Or with workspace
cd mycelix-workspace
npm install
```

### Production

```bash
# Publish SDK
cd sdk-ts
npm publish --access public

# Install in hApps
npm install @mycelix/sdk
```

### Docker

```dockerfile
# In hApp Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
```

---

## Testing Integration

### Mock Mode (No Conductor)

```typescript
import { createMockClient } from '@mycelix/sdk';

const client = createMockClient();
await client.connect();
// All operations work without real conductor
```

### Real Conductor

```bash
# Start conductor
cd Mycelix-Core
just dev

# Run integration tests
cd sdk-ts
CONDUCTOR_AVAILABLE=true npm run test:conductor
```

---

## Migration Guide

### From Direct Holochain Calls

Before:
```typescript
const result = await client.callZome({
  cell_id: cellId,
  zome_name: 'trust',
  fn_name: 'get_reputation',
  payload: { agent_id: agentId },
});
```

After:
```typescript
import { createClient } from '@mycelix/sdk';

const client = createClient({ /* config */ });
const result = await client.checkTrust({
  agentId,
  minimumScore: 0.5,
  happContext: 'my-happ',
});
```

---

## Next Steps

1. **Install SDK in each hApp** - Add `@mycelix/sdk` dependency
2. **Create adapters** - hApp-specific wrapper modules
3. **Update CI/CD** - Include SDK in build pipelines
4. **Add E2E tests** - Cross-hApp integration tests
5. **Documentation** - hApp-specific usage guides

---

## Support

- **SDK Issues**: https://github.com/Luminous-Dynamics/mycelix/issues
- **Discord**: #sdk-support channel
- **Docs**: https://docs.mycelix.net/sdk
