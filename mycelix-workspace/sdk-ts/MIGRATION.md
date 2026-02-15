# Mycelix SDK Migration Guide

This guide helps you migrate from legacy service-based patterns to the modern unified client pattern.

## Overview

The Mycelix SDK has evolved to use a unified client pattern across all domains. This provides:

- **Consistent API**: Same patterns across Identity, Finance, Property, Energy, Media, Governance, Justice, Knowledge, and Health
- **Better Developer Experience**: Single entry point per domain with all functionality accessible
- **Built-in Resilience**: Retry policies and circuit breakers integrated by default
- **Type Safety**: Full TypeScript support with comprehensive type exports

## Quick Migration

### Before (Legacy Pattern)

```typescript
// Old: Multiple service imports and manual setup
import { IdentityService } from '@mycelix/sdk/identity/services';
import { ProfileService } from '@mycelix/sdk/identity/services/profile';
import { CredentialService } from '@mycelix/sdk/identity/services/credentials';

const identityService = new IdentityService(client);
const profileService = new ProfileService(client);
const credentialService = new CredentialService(client);

// Manual error handling and retry logic
try {
  const profile = await profileService.getProfile(did);
} catch (error) {
  // Manual retry implementation
}
```

### After (Unified Client Pattern)

```typescript
// New: Single unified client with all functionality
import { MycelixIdentityClient } from '@mycelix/sdk';

const identity = await MycelixIdentityClient.connect({
  url: 'ws://localhost:8888',
});

// All services accessible through the unified client
const profile = await identity.profiles.getProfile(did);
const credentials = await identity.credentials.getCredentials(did);

// Built-in retry with configurable policy
const identityWithRetry = identity.withRetry({ maxAttempts: 5 });
```

## Domain-Specific Migration

### Energy SDK

#### Legacy Pattern
```typescript
import { EnergyService } from '@mycelix/sdk/integrations/energy/legacy';
import { ProjectService } from '@mycelix/sdk/integrations/energy/legacy';

const energyService = new EnergyService(client);
const projectService = new ProjectService(client);

const project = await projectService.createProject({
  name: 'Solar Farm',
  source: 'Solar',
  capacity_kw: 500,
});
```

#### Unified Pattern
```typescript
import { MycelixEnergyClient } from '@mycelix/sdk';

const energy = await MycelixEnergyClient.connect({
  url: 'ws://localhost:8888',
});

// All functionality through sub-clients
const project = await energy.projects.registerProject({
  name: 'Solar Farm',
  description: 'Community solar installation',
  source: 'Solar',
  capacity_kw: 500,
  location: { lat: 32.95, lng: -96.73 },
  investment_goal: 750000,
});

// Convenience methods for common operations
const status = await energy.getEnergyStatus('did:mycelix:user');
const overview = await energy.getGridOverview('ERCOT');
```

### Knowledge SDK

#### Legacy Pattern
```typescript
import { ClaimsService } from '@mycelix/sdk/integrations/knowledge/services';
import { GraphService } from '@mycelix/sdk/integrations/knowledge/services';

const claimsService = new ClaimsService(client);
const graphService = new GraphService(client);

const claim = await claimsService.submitClaim({ ... });
const related = await graphService.getRelatedClaims(claimId);
```

#### Unified Pattern
```typescript
import { MycelixKnowledgeClient } from '@mycelix/sdk';

const knowledge = await MycelixKnowledgeClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients for specific operations
const claim = await knowledge.claims.submitClaim({ ... });
const related = await knowledge.graph.getSupportingClaims(claimId);

// Convenience methods
const evidence = await knowledge.getClaimWithEvidence(claimId);
const results = await knowledge.queryKnowledgeGraph({
  query: 'climate change',
  filters: { min_empirical: 0.7 },
  includeRelated: true,
});
```

### Justice SDK

#### Legacy Pattern
```typescript
import { CasesService } from '@mycelix/sdk/justice/services';
import { EvidenceService } from '@mycelix/sdk/justice/services';
import { ArbitrationService } from '@mycelix/sdk/justice/services';

const casesService = new CasesService(client);
const evidenceService = new EvidenceService(client);

const case_ = await casesService.fileCase({ ... });
await evidenceService.submitEvidence(case_.id, { ... });
```

#### Unified Pattern
```typescript
import { MycelixJusticeClient } from '@mycelix/sdk';

const justice = await MycelixJusticeClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients
const case_ = await justice.cases.fileCase({ ... });
await justice.evidence.submitEvidence({ case_id: case_.id, ... });

// Convenience methods
const { case_, evidence } = await justice.fileWithEvidence(
  caseInput,
  [evidenceItem1, evidenceItem2]
);

const timeline = await justice.getArbitrationTimeline(caseId);
```

### Media SDK

#### Legacy Pattern
```typescript
import { PublicationService } from '@mycelix/sdk/media/services';
import { FactCheckService } from '@mycelix/sdk/media/services';

const publicationService = new PublicationService(client);
const factCheckService = new FactCheckService(client);

const content = await publicationService.publish({ ... });
```

#### Unified Pattern
```typescript
import { MycelixMediaClient } from '@mycelix/sdk';

const media = await MycelixMediaClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients with consistent naming
const content = await media.publication.publish({ ... });
await media.attribution.addAttribution({ ... });
await media.factCheck.submitFactCheck({ ... });
await media.curation.endorse(contentId);

// Convenience methods
const { content, attribution } = await media.publishWithAttribution(authorDid, input);
const complete = await media.getContentComplete(contentId);
```

### Finance SDK

#### Legacy Pattern
```typescript
import { WalletService } from '@mycelix/sdk/integrations/finance/services';
import { TransactionService } from '@mycelix/sdk/integrations/finance/services';

const walletService = new WalletService(client);
const transactionService = new TransactionService(client);
```

#### Unified Pattern
```typescript
import { MycelixFinanceClient } from '@mycelix/sdk';

const finance = await MycelixFinanceClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients for all finance operations
const wallet = await finance.wallets.createWallet({ ... });
const tx = await finance.transactions.transfer({ ... });
const credit = await finance.credit.getCreditScore(did);
const escrow = await finance.escrow.createEscrow({ ... });

// Convenience methods
const { creditScore, creditLimit } = await finance.getCreditStatus(did);
```

### Governance SDK

#### Legacy Pattern
```typescript
import { DAOService } from '@mycelix/sdk/integrations/governance/services';
import { VotingService } from '@mycelix/sdk/integrations/governance/services';

const daoService = new DAOService(client);
const votingService = new VotingService(client);
```

#### Unified Pattern
```typescript
import { MycelixGovernanceClient } from '@mycelix/sdk';

const governance = await MycelixGovernanceClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients
const dao = await governance.daos.createDAO({ ... });
const proposal = await governance.proposals.createProposal({ ... });
await governance.voting.castVote(proposalId, 'For');

// Convenience methods
const daoStats = await governance.getDAOStatistics(daoId);
const activeProposals = await governance.getActiveProposals(daoId);
```

### Property SDK

#### Legacy Pattern
```typescript
import { AssetService } from '@mycelix/sdk/integrations/property/services';
import { TransferService } from '@mycelix/sdk/integrations/property/services';

const assetService = new AssetService(client);
const transferService = new TransferService(client);
```

#### Unified Pattern
```typescript
import { MycelixPropertyClient } from '@mycelix/sdk';

const property = await MycelixPropertyClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients
const asset = await property.assets.registerAsset({ ... });
const transfer = await property.transfers.initiateTransfer({ ... });
const history = await property.provenance.getAssetHistory(assetId);

// Convenience methods
const portfolio = await property.getPortfolioValue(ownerDid);
const complete = await property.getAssetComplete(assetId);
```

### Identity SDK

#### Legacy Pattern
```typescript
import { DidService } from '@mycelix/sdk/integrations/identity/services';
import { CredentialService } from '@mycelix/sdk/integrations/identity/services';

const didService = new DidService(client);
const credentialService = new CredentialService(client);
```

#### Unified Pattern
```typescript
import { MycelixIdentityClient } from '@mycelix/sdk';

const identity = await MycelixIdentityClient.connect({
  url: 'ws://localhost:8888',
});

// Sub-clients
const did = await identity.dids.createDid();
const credential = await identity.credentials.issueCredential({ ... });
const verification = await identity.verification.verifyCredential(credId);
const recovery = await identity.recovery.initiateRecovery(did);

// Convenience methods
const profile = await identity.getIdentityComplete(did);
const reputation = await identity.getReputationSummary(did);
```

### Health SDK

#### Legacy Pattern
```typescript
import { PatientService } from '@mycelix/sdk/integrations/health/services';
import { RecordsService } from '@mycelix/sdk/integrations/health/services';

const patientService = new PatientService(client);
const recordsService = new RecordsService(client);
```

#### Unified Pattern
```typescript
import { MycelixHealthClient } from '@mycelix/sdk';

const health = await MycelixHealthClient.connect({
  url: 'ws://localhost:8888',
});

// 7 sub-clients for comprehensive health operations
const patientHash = await health.patient.createPatient({ ... });
const encounters = await health.records.getPatientEncounters(patientHash);
const prescriptions = await health.prescriptions.getActivePrescriptions(patientHash);
const consents = await health.consent.getActiveConsents(patientHash);
const trials = await health.trials.getRecruitingTrials();
const insurance = await health.insurance.getPatientInsurance(patientHash);

// Convenience methods
const summary = await health.getPatientSummary(patientHash);
const eligibleTrials = await health.findEligibleTrials(patientHash, patientAge);
```

## Retry Configuration

All unified clients support configurable retry policies:

### Using Preset Policies

```typescript
import { MycelixEnergyClient, RetryPolicies } from '@mycelix/sdk';

// Use a preset policy
const energy = await MycelixEnergyClient.connect({
  url: 'ws://localhost:8888',
  retry: RetryPolicies.aggressive, // 5 attempts, slower backoff
});

// Available presets:
// - RetryPolicies.none      - No retries
// - RetryPolicies.light     - 2 attempts, fast backoff
// - RetryPolicies.standard  - 3 attempts, moderate backoff (default)
// - RetryPolicies.aggressive - 5 attempts, slower backoff
// - RetryPolicies.network   - 4 attempts, optimized for unstable connections
```

### Custom Retry Configuration

```typescript
const energy = await MycelixEnergyClient.connect({
  url: 'ws://localhost:8888',
  retry: {
    maxAttempts: 5,
    initialDelayMs: 500,
    backoffMultiplier: 2,
    maxDelayMs: 30000,
    jitter: true,
    onRetry: (attempt, error, delay) => {
      console.log(`Retry ${attempt} after ${delay}ms: ${error.message}`);
    },
  },
});
```

### Changing Retry Policy at Runtime

```typescript
// Create a new client with different retry settings
const aggressiveClient = energy.withRetry(RetryPolicies.aggressive);

// Or with custom options
const customClient = energy.withRetry({ maxAttempts: 10 });
```

## Circuit Breaker Integration

For high-availability scenarios, use circuit breakers:

```typescript
import {
  MycelixEnergyClient,
  CircuitBreaker,
  CircuitBreakers,
  withCircuitBreaker,
} from '@mycelix/sdk';

// Create a circuit breaker
const breaker = CircuitBreakers.standard('energy-service');

// Or customize it
const customBreaker = new CircuitBreaker({
  failureThreshold: 3,
  resetTimeoutMs: 30000,
  successThreshold: 2,
  name: 'my-service',
  onStateChange: (from, to) => console.log(`Circuit ${from} -> ${to}`),
});

// Wrap operations
const getProject = withCircuitBreaker(
  (id: string) => energy.projects.getProject(id),
  breaker
);

// Use with fallback
const project = await breaker.executeWithFallback(
  () => energy.projects.getProject(projectId),
  null // Fallback value when circuit is open
);
```

## Error Handling

Each domain has its own error class:

```typescript
import {
  EnergySdkError,
  KnowledgeSdkError,
  JusticeSdkError,
  MediaSdkError,
  HealthSdkError,
} from '@mycelix/sdk';

try {
  await energy.projects.getProject(projectId);
} catch (error) {
  if (error instanceof EnergySdkError) {
    console.log(`Energy error [${error.code}]: ${error.message}`);
    // Error codes: CONNECTION_ERROR, ZOME_ERROR, NOT_FOUND, etc.
  }
}
```

## Framework Integration

### React

```typescript
import { createQueryHook, createMutationHook } from '@mycelix/sdk/react';

const useProject = createQueryHook(
  (id: string) => energy.projects.getProject(id)
);

const useCreateProject = createMutationHook(
  (input) => energy.projects.registerProject(input)
);

// In your component
function ProjectView({ projectId }) {
  const { data, loading, error, refetch } = useProject(projectId);
  // ...
}
```

### Svelte

```typescript
import { createQueryStore, createMutationStore } from '@mycelix/sdk/svelte';

const projectStore = createQueryStore(
  () => energy.projects.getProject(projectId),
  projectId
);

// In your Svelte component
$: ({ data, loading, error } = $projectStore);
```

### GraphQL

```typescript
import { createResolvers, createSubscriptions } from '@mycelix/sdk/graphql';

const resolvers = createResolvers(energy);
const subscriptions = createSubscriptions(energy);
```

## Deprecated APIs

The following legacy exports are deprecated and will be removed in a future version:

### Energy
- `EnergyService` -> Use `MycelixEnergyClient.projects`
- `LegacyEnergySource` -> Use `EnergySource`
- `LegacyEnergyParticipant` -> Use `EnergyParticipant`

Import deprecated types from the legacy module if needed during migration:
```typescript
import { EnergyService, LegacyEnergySource } from '@mycelix/sdk/integrations/energy/legacy';
```

## Type Exports

All types are exported from the main entry point:

```typescript
import type {
  // Identity types
  DidDocument,
  VerifiableCredential,
  VerificationMethod,
  CredentialSchema,

  // Finance types
  Wallet,
  Transaction,
  CreditScore,
  Currency,
  EscrowContract,

  // Governance types
  DAO,
  Proposal,
  Vote,
  DAOMember,
  VotingPower,

  // Property types
  Asset,
  Transfer,
  Ownership,
  AssetType,
  PropertyTitle,

  // Energy types
  EnergyProject,
  EnergyParticipant,
  EnergyTrade,
  EnergyCredit,
  EnergySource,

  // Knowledge types
  Claim,
  ClaimCluster,
  SearchResult,
  Citation,

  // Justice types
  Case,
  Evidence,
  Decision,
  CasePhase,
  CaseStatus,

  // Media types
  Content,
  Attribution,
  FactCheck,
  ContentType,

  // Health types
  Patient,
  Provider,
  Encounter,
  Prescription,
  Consent,

  // Common types
  RetryOptions,
  CircuitBreakerConfig,
} from '@mycelix/sdk';
```

## Getting Help

- Check the [API Documentation](./docs/index.html) for detailed type information
- File issues at the project repository
- Join the community Discord for support
