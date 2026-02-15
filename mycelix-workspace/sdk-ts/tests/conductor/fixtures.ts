/**
 * Test Fixtures for Conductor Integration Tests
 *
 * Provides reusable test data generators and fixture management
 * for Holochain conductor integration tests.
 */

// ============================================================================
// DID and Identity Fixtures
// ============================================================================

export function generateTestDID(prefix = 'test'): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).slice(2, 11);
  return `did:mycelix:${prefix}-${timestamp}-${random}`;
}

export function generateTestPublicKey(length = 64): string {
  const chars = 'abcdef0123456789';
  let key = '';
  for (let i = 0; i < length; i++) {
    key += chars[Math.floor(Math.random() * chars.length)];
  }
  return key;
}

export function generateTestAgentKey(): string {
  return `uhCAk${generateTestPublicKey(39)}`; // Holochain agent key format
}

// ============================================================================
// Hash and Content Fixtures
// ============================================================================

export function generateTestHash(prefix = 'Qm'): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let hash = prefix;
  for (let i = 0; i < 44; i++) {
    hash += chars[Math.floor(Math.random() * chars.length)];
  }
  return hash;
}

export function generateTestDnaHash(): string {
  return `uhC0k${generateTestPublicKey(39)}`;
}

// ============================================================================
// Location Fixtures
// ============================================================================

export interface TestLocation {
  lat: number;
  lng: number;
  radius_km?: number;
  name?: string;
}

export const TEST_LOCATIONS: TestLocation[] = [
  { lat: 37.7749, lng: -122.4194, name: 'San Francisco' },
  { lat: 40.7128, lng: -74.006, name: 'New York' },
  { lat: 51.5074, lng: -0.1278, name: 'London' },
  { lat: 35.6762, lng: 139.6503, name: 'Tokyo' },
  { lat: -33.8688, lng: 151.2093, name: 'Sydney' },
  { lat: 52.52, lng: 13.405, name: 'Berlin' },
  { lat: 48.8566, lng: 2.3522, name: 'Paris' },
  { lat: 55.7558, lng: 37.6173, name: 'Moscow' },
];

export function getRandomLocation(): TestLocation {
  const loc = TEST_LOCATIONS[Math.floor(Math.random() * TEST_LOCATIONS.length)];
  return {
    ...loc,
    // Add some randomness to coordinates
    lat: loc.lat + (Math.random() - 0.5) * 0.1,
    lng: loc.lng + (Math.random() - 0.5) * 0.1,
    radius_km: Math.floor(Math.random() * 100) + 10,
  };
}

// ============================================================================
// Energy Fixtures
// ============================================================================

export type EnergySource = 'solar' | 'wind' | 'hydro' | 'battery' | 'grid' | 'other_renewable';

export const ENERGY_SOURCES: EnergySource[] = [
  'solar',
  'wind',
  'hydro',
  'battery',
  'grid',
  'other_renewable',
];

export function generateEnergyReading() {
  return {
    production_kwh: Math.floor(Math.random() * 1000) + 100,
    consumption_kwh: Math.floor(Math.random() * 500) + 50,
    energy_source: ENERGY_SOURCES[Math.floor(Math.random() * ENERGY_SOURCES.length)],
    timestamp: Date.now() * 1000, // Microseconds
  };
}

// ============================================================================
// Finance Fixtures
// ============================================================================

export type CreditPurpose =
  | 'LoanApplication'
  | 'TrustVerification'
  | 'MarketplaceTransaction'
  | 'PropertyPurchase'
  | 'EnergyInvestment';

export const CREDIT_PURPOSES: CreditPurpose[] = [
  'LoanApplication',
  'TrustVerification',
  'MarketplaceTransaction',
  'PropertyPurchase',
  'EnergyInvestment',
];

export function generateLoanApplication() {
  return {
    borrower_did: generateTestDID('borrower'),
    amount: Math.floor(Math.random() * 100000) + 1000,
    term_months: [6, 12, 24, 36, 48, 60][Math.floor(Math.random() * 6)],
    interest_rate: Math.random() * 0.15 + 0.02,
    purpose: CREDIT_PURPOSES[Math.floor(Math.random() * CREDIT_PURPOSES.length)],
  };
}

export function generatePayment() {
  return {
    payee_did: generateTestDID('payee'),
    amount: Math.floor(Math.random() * 10000) + 10,
    currency: ['MCX', 'USD', 'EUR', 'BTC'][Math.floor(Math.random() * 4)],
    reference_id: `payment-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  };
}

// ============================================================================
// Media Fixtures
// ============================================================================

export type ContentType =
  | 'Article'
  | 'Opinion'
  | 'Investigation'
  | 'Review'
  | 'Analysis'
  | 'Interview'
  | 'Report'
  | 'Editorial'
  | 'Other';
export type LicenseType = 'CC0' | 'CCBY' | 'CCBYSA' | 'CCBYNC' | 'CCBYNCSA' | 'AllRightsReserved' | 'Custom';

export const CONTENT_TYPES: ContentType[] = [
  'Article',
  'Opinion',
  'Investigation',
  'Review',
  'Analysis',
  'Interview',
  'Report',
  'Editorial',
  'Other',
];
export const LICENSE_TYPES: LicenseType[] = [
  'CC0',
  'CCBY',
  'CCBYSA',
  'CCBYNC',
  'CCBYNCSA',
  'AllRightsReserved',
  'Custom',
];

export function generateContent() {
  return {
    id: `content-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    author_did: generateTestDID('author'),
    title: `Test Content ${Date.now()}`,
    content_type: CONTENT_TYPES[Math.floor(Math.random() * CONTENT_TYPES.length)],
    content_hash: generateTestHash(),
    license: LICENSE_TYPES[Math.floor(Math.random() * LICENSE_TYPES.length)],
  };
}

// ============================================================================
// Governance Fixtures
// ============================================================================

export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional';

export function generateProposal() {
  return {
    id: `proposal-${Date.now()}`,
    proposer_did: generateTestDID('proposer'),
    title: `Test Proposal ${Date.now()}`,
    description: 'This is a test proposal for conductor testing.',
    proposal_type: ['Standard', 'Emergency', 'Constitutional'][
      Math.floor(Math.random() * 3)
    ] as ProposalType,
    voting_period_hours: [24, 72, 168, 720][Math.floor(Math.random() * 4)],
    quorum_percentage: 0.3 + Math.random() * 0.4,
  };
}

// ============================================================================
// Justice Fixtures
// ============================================================================

export type DisputeType =
  | 'ContractDispute'
  | 'CopyrightInfringement'
  | 'FraudulentActivity'
  | 'ServiceFailure'
  | 'Other';

export const DISPUTE_TYPES: DisputeType[] = [
  'ContractDispute',
  'CopyrightInfringement',
  'FraudulentActivity',
  'ServiceFailure',
  'Other',
];

export function generateDispute() {
  return {
    complainant_did: generateTestDID('complainant'),
    respondent_did: generateTestDID('respondent'),
    dispute_type: DISPUTE_TYPES[Math.floor(Math.random() * DISPUTE_TYPES.length)],
    title: `Test Dispute ${Date.now()}`,
    description: 'This is a test dispute for conductor testing.',
    evidence_hashes: [generateTestHash(), generateTestHash()],
    related_happs: ['marketplace', 'finance'],
  };
}

// ============================================================================
// Knowledge Fixtures
// ============================================================================

export function generateClaim() {
  return {
    id: `claim-${Date.now()}`,
    author_did: generateTestDID('researcher'),
    title: `Research Finding ${Date.now()}`,
    content: 'This is a test claim for conductor testing.',
    claim_hash: generateTestHash(),
    empirical: Math.random(),
    normative: Math.random(),
    mythic: Math.random(),
  };
}

// ============================================================================
// Property Fixtures
// ============================================================================

export type AssetType = 'RealEstate' | 'Vehicle' | 'Equipment' | 'Intellectual' | 'Digital';

export const ASSET_TYPES: AssetType[] = [
  'RealEstate',
  'Vehicle',
  'Equipment',
  'Intellectual',
  'Digital',
];

export function generateAsset() {
  return {
    id: `asset-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    owner_did: generateTestDID('owner'),
    asset_type: ASSET_TYPES[Math.floor(Math.random() * ASSET_TYPES.length)],
    valuation: Math.floor(Math.random() * 1000000) + 10000,
    location: getRandomLocation(),
  };
}

// ============================================================================
// Batch Fixture Generators
// ============================================================================

export function generateTestAgents(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    did: generateTestDID(`agent-${i}`),
    publicKey: generateTestPublicKey(),
    agentKey: generateTestAgentKey(),
  }));
}

export function generateTestScenario() {
  const agents = generateTestAgents(5);
  const assets = Array.from({ length: 3 }, () => generateAsset());
  const content = Array.from({ length: 2 }, () => generateContent());
  const claims = Array.from({ length: 3 }, () => generateClaim());

  return {
    agents,
    assets,
    content,
    claims,
    proposal: generateProposal(),
    dispute: generateDispute(),
  };
}

// ============================================================================
// Fixture Cleanup
// ============================================================================

export interface CleanupRegistry {
  dids: string[];
  assetIds: string[];
  contentIds: string[];
  proposalIds: string[];
  disputeIds: string[];
}

export function createCleanupRegistry(): CleanupRegistry {
  return {
    dids: [],
    assetIds: [],
    contentIds: [],
    proposalIds: [],
    disputeIds: [],
  };
}

export function registerForCleanup(
  registry: CleanupRegistry,
  type: keyof CleanupRegistry,
  id: string
): void {
  registry[type].push(id);
}

// Note: Actual cleanup would require conductor API calls to remove test data
// For now, test data is ephemeral and clears when conductor restarts
