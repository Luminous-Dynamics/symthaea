// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Clients
 *
 * Master SDK clients for all Civilizational OS hApps.
 * Each hApp has a dedicated client module with type-safe access to all zomes.
 *
 * ## Available Client Modules
 *
 * | Module | hApp | Zomes | Description |
 * |--------|------|-------|-------------|
 * | `identity` | mycelix-identity | 8 | DID, credentials, recovery, trust |
 * | `finance` | mycelix-finance | 9 | Payments, CGC, TEND, HEARTH |
 * | `governance` | mycelix-governance | 7 | Proposals, voting, delegation |
 * | `health` | mycelix-health | - | Patient records, credentials |
 * | `energy` | mycelix-energy | - | Energy trading, grid balance |
 * | `knowledge` | mycelix-knowledge | - | DKG, fact-checking |
 * | `property` | mycelix-commons | 4 | Asset registry, transfers, liens, commons |
 * | `housing` | mycelix-commons | 4 | Units, membership, finances, governance |
 * | `care` | mycelix-commons | 3 | Timebank, circles, matching |
 * | `water` | mycelix-commons | 3 | Flow, purity, stewardship |
 * | `justice` | mycelix-civic | 5 | Cases, evidence, arbitration, enforcement |
 * | `emergency` | mycelix-civic | 3 | Incidents, coordination, shelters |
 * | `media` | mycelix-civic | 4 | Publication, attribution, factcheck, curation |
 * | `hearth` | mycelix-hearth | 10 | Kinship, gratitude, stories, care, autonomy, emergency, decisions, resources, milestones, rhythms |
 *
 * @example
 * ```typescript
 * import { identity, finance, governance } from '@mycelix/sdk/clients';
 * import { AppWebsocket } from '@holochain/client';
 *
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 *
 * // Create unified clients for each hApp
 * const id = identity.createIdentityClient(client);
 * const fin = finance.createFinanceClient(client);
 *
 * // Access individual zome clients
 * const myDid = await id.did.createDid();
 * const credential = await id.credentials.issueCredential({...});
 * const cgcAllocation = await fin.cgc.getOrCreateAllocation(myDid.document.id);
 * ```
 *
 * @module @mycelix/sdk/clients
 */

// =============================================================================
// HAPP CLIENT MODULES
// =============================================================================

export * as identity from './identity/index.js';
export * as finance from './finance/index.js';
export * as governance from './governance/index.js';
export * as health from './health/index.js';
export * as energy from './energy/index.js';
export * as knowledge from './knowledge/index.js';
export * as justice from './justice/index.js';
export * as media from './media/index.js';
export * as property from './property/index.js';
export * as care from './care/index.js';
export * as emergency from './emergency/index.js';
export * as water from './water/index.js';
export * as housing from './housing/index.js';
export * as hearth from './hearth/index.js';

// =============================================================================
// IDENTITY HAPP EXPORTS
// =============================================================================

export {
  // Unified client
  createIdentityClient,
  IdentityClient,
  type UnifiedIdentityClient,

  // Individual zome clients
  DidClient,
  CredentialsClient,
  RecoveryClient,
  BridgeClient as IdentityBridgeClient,
  SchemaClient,
  RevocationClient,
  TrustClient,
  EducationClient,

  // Error types
  IdentitySdkError,
  IdentitySdkErrorCode,

  // Configuration
  DEFAULT_IDENTITY_CLIENT_CONFIG,
  type IdentityClientConfig,
} from './identity/index.js';

// =============================================================================
// FINANCE HAPP EXPORTS
// =============================================================================

export {
  FinanceClient,
  createFinanceClient,
  createFinanceClients,
  type FinanceClientConfig,
} from './finance/index.js';

// =============================================================================
// CARE HAPP EXPORTS
// =============================================================================

export {
  CareClient,
  createCareClient,
  TimebankClient,
  CirclesClient,
  MatchingClient,
  CareError,
  type CareClientConfig,
} from './care/index.js';

// =============================================================================
// EMERGENCY HAPP EXPORTS
// =============================================================================

export {
  EmergencyClient,
  createEmergencyClient,
  IncidentsClient,
  CoordinationClient,
  SheltersClient,
  EmergencyError,
  type EmergencyClientConfig,
} from './emergency/index.js';

// =============================================================================
// WATER HAPP EXPORTS
// =============================================================================

export {
  WaterClient,
  createWaterClient,
  FlowClient,
  PurityClient,
  StewardClient,
  WaterError,
  type WaterClientConfig,
} from './water/index.js';

// =============================================================================
// HOUSING HAPP EXPORTS
// =============================================================================

export {
  HousingClient,
  createHousingClient,
  UnitsClient,
  MembershipClient,
  FinancesClient,
  HousingGovernanceClient,
  HousingError,
  type HousingClientConfig,
} from './housing/index.js';

// =============================================================================
// HEARTH HAPP EXPORTS
// =============================================================================

export {
  HearthClient,
  createHearthClient,
  KinshipClient,
  GratitudeClient,
  StoriesClient,
  HearthCareClient,
  AutonomyClient,
  EmergencyClient as HearthEmergencyClient,
  DecisionsClient,
  ResourcesClient,
  MilestonesClient,
  RhythmsClient,
  HearthError,
  type HearthClientConfig,
} from './hearth/index.js';
