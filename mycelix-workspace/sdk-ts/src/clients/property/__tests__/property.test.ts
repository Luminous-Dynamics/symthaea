/**
 * Property Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * RegistryClient, TransfersClient, DisputesClient, and CommonsClient.
 *
 * Property clients use ZomeCallable interface (not ZomeClient base class)
 * and return HolochainRecord<T> directly (no extractEntry).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  RegistryClient,
  TransfersClient,
  DisputesClient,
  CommonsClient,
  type ZomeCallable,
} from '../index';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockCallable(): ZomeCallable {
  return {
    callZome: vi.fn(),
  } as unknown as ZomeCallable;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: 'mock-hash', content: {} }, signature: 'mock-sig' },
  };
}

// ============================================================================
// MOCK ENTRIES (snake_case, matching Rust serde output)
// ============================================================================

const PROPERTY_ENTRY = {
  id: 'prop-001',
  property_type: 'Land',
  title: 'Community Garden Plot',
  description: 'Shared garden space in Richardson',
  owner_did: 'did:mycelix:alice',
  co_owners: [],
  geolocation: { latitude: 32.948, longitude: -96.729 },
  address: { street: '123 Garden Ln', city: 'Richardson', state: 'TX', country: 'US' },
  metadata: { parcel_id: 'TX-1234', zoning: 'Residential' },
  registered: 1708200000,
};

const TITLE_DEED_ENTRY = {
  id: 'deed-001',
  property_id: 'prop-001',
  owner_did: 'did:mycelix:alice',
  deed_type: 'Original',
  issued: 1708200000,
  encumbrances: [],
};

const TRANSFER_ENTRY = {
  id: 'xfer-001',
  property_id: 'prop-001',
  from_did: 'did:mycelix:alice',
  to_did: 'did:mycelix:bob',
  transfer_type: 'Sale',
  price: 150000,
  currency: 'USD',
  conditions: [],
  status: 'Initiated',
  initiated: 1708200000,
};

const ESCROW_ENTRY = {
  id: 'escrow-001',
  transfer_id: 'xfer-001',
  amount: 150000,
  currency: 'USD',
  funded: false,
  release_conditions: ['Title cleared', 'Inspection passed'],
  created: 1708200000,
};

const DISPUTE_ENTRY = {
  id: 'dispute-001',
  property_id: 'prop-001',
  dispute_type: 'Boundary',
  claimant_did: 'did:mycelix:alice',
  respondent_did: 'did:mycelix:bob',
  description: 'Fence encroaches 2 feet',
  evidence_ids: ['ev-001'],
  status: 'Filed',
  filed: 1708200000,
};

const CLAIM_ENTRY = {
  id: 'claim-001',
  property_id: 'prop-001',
  claimant_did: 'did:mycelix:carol',
  claim_basis: 'Inheritance',
  supporting_documents: ['doc-001'],
  status: 'Pending',
  filed: 1708200000,
};

const RESOURCE_ENTRY = {
  id: 'res-001',
  name: 'Community Garden',
  description: 'Shared garden for local residents',
  resource_type: 'Land',
  property_id: 'prop-001',
  stewards: ['did:mycelix:alice'],
  governance_rules: { decision_threshold: 0.66, voting_period_days: 7, quorum_percentage: 0.5 },
  created: 1708200000,
};

const USAGE_RIGHT_ENTRY = {
  id: 'right-001',
  resource_id: 'res-001',
  holder_did: 'did:mycelix:bob',
  right_type: 'Access',
  quota: 100,
  granted: 1708200000,
  active: true,
};

const USAGE_LOG_ENTRY = {
  id: 'log-001',
  resource_id: 'res-001',
  user_did: 'did:mycelix:bob',
  usage_type: 'Irrigation',
  quantity: 50,
  unit: 'liters',
  timestamp: 1708200000,
};

// ============================================================================
// REGISTRY CLIENT TESTS
// ============================================================================

describe('RegistryClient', () => {
  let mockCallable: ZomeCallable;
  let registry: RegistryClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    registry = new RegistryClient(mockCallable);
  });

  it('registerProperty calls correct zome', async () => {
    const record = mockRecord(PROPERTY_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      property_type: 'Land' as const,
      title: 'Community Garden Plot',
      description: 'Shared garden space in Richardson',
      owner_did: 'did:mycelix:alice',
      co_owners: [],
      metadata: { parcel_id: 'TX-1234', zoning: 'Residential' },
    };
    const result = await registry.registerProperty(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_registry',
      fn_name: 'register_property',
      payload: input,
    });
    expect(result.entry.Present.id).toBe('prop-001');
    expect(result.entry.Present.property_type).toBe('Land');
  });

  it('getOwnerProperties returns property list', async () => {
    const record = mockRecord(PROPERTY_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await registry.getOwnerProperties('did:mycelix:alice');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_registry',
      fn_name: 'get_owner_properties',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].entry.Present.owner_did).toBe('did:mycelix:alice');
  });

  it('hasClearTitle returns boolean', async () => {
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

    const result = await registry.hasClearTitle('prop-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_registry',
      fn_name: 'has_clear_title',
      payload: 'prop-001',
    });
    expect(result).toBe(true);
  });

  it('getTitleDeed returns deed record', async () => {
    const record = mockRecord(TITLE_DEED_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await registry.getTitleDeed('prop-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_registry',
      fn_name: 'get_title_deed',
      payload: 'prop-001',
    });
    expect(result!.entry.Present.deed_type).toBe('Original');
  });
});

// ============================================================================
// TRANSFERS CLIENT TESTS
// ============================================================================

describe('TransfersClient', () => {
  let mockCallable: ZomeCallable;
  let transfers: TransfersClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    transfers = new TransfersClient(mockCallable);
  });

  it('initiateTransfer calls correct zome', async () => {
    const record = mockRecord(TRANSFER_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      property_id: 'prop-001',
      from_did: 'did:mycelix:alice',
      to_did: 'did:mycelix:bob',
      transfer_type: 'Sale' as const,
      price: 150000,
      currency: 'USD',
      conditions: [],
    };
    const result = await transfers.initiateTransfer(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_transfer',
      fn_name: 'initiate_transfer',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Initiated');
  });

  it('getSellerTransfers returns transfer list', async () => {
    const record = mockRecord(TRANSFER_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await transfers.getSellerTransfers('did:mycelix:alice');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_transfer',
      fn_name: 'get_seller_transfers',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].entry.Present.from_did).toBe('did:mycelix:alice');
  });

  it('createEscrow returns escrow record', async () => {
    const record = mockRecord(ESCROW_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      transfer_id: 'xfer-001',
      amount: 150000,
      currency: 'USD',
      release_conditions: ['Title cleared', 'Inspection passed'],
    };
    const result = await transfers.createEscrow(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_transfer',
      fn_name: 'create_escrow',
      payload: input,
    });
    expect(result.entry.Present.funded).toBe(false);
  });
});

// ============================================================================
// DISPUTES CLIENT TESTS
// ============================================================================

describe('DisputesClient', () => {
  let mockCallable: ZomeCallable;
  let disputes: DisputesClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    disputes = new DisputesClient(mockCallable);
  });

  it('fileDispute calls correct zome', async () => {
    const record = mockRecord(DISPUTE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      property_id: 'prop-001',
      dispute_type: 'Boundary' as const,
      claimant_did: 'did:mycelix:alice',
      respondent_did: 'did:mycelix:bob',
      description: 'Fence encroaches 2 feet',
      evidence_ids: ['ev-001'],
    };
    const result = await disputes.fileDispute(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_disputes',
      fn_name: 'file_dispute',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Filed');
  });

  it('getPropertyDisputes returns dispute list', async () => {
    const record = mockRecord(DISPUTE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await disputes.getPropertyDisputes('prop-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_disputes',
      fn_name: 'get_property_disputes',
      payload: 'prop-001',
    });
    expect(result).toHaveLength(1);
  });

  it('fileOwnershipClaim returns claim record', async () => {
    const record = mockRecord(CLAIM_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      property_id: 'prop-001',
      claimant_did: 'did:mycelix:carol',
      claim_basis: 'Inheritance' as const,
      supporting_documents: ['doc-001'],
    };
    const result = await disputes.fileOwnershipClaim(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_disputes',
      fn_name: 'file_ownership_claim',
      payload: input,
    });
    expect(result.entry.Present.claim_basis).toBe('Inheritance');
  });
});

// ============================================================================
// COMMONS CLIENT TESTS
// ============================================================================

describe('CommonsClient', () => {
  let mockCallable: ZomeCallable;
  let commons: CommonsClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    commons = new CommonsClient(mockCallable);
  });

  it('createCommonResource calls correct zome', async () => {
    const record = mockRecord(RESOURCE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      name: 'Community Garden',
      description: 'Shared garden for local residents',
      resource_type: 'Land' as const,
      property_id: 'prop-001',
      stewards: ['did:mycelix:alice'],
      governance_rules: { decision_threshold: 0.66, voting_period_days: 7, quorum_percentage: 0.5 },
    };
    const result = await commons.createCommonResource(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_commons',
      fn_name: 'create_common_resource',
      payload: input,
    });
    expect(result.entry.Present.name).toBe('Community Garden');
  });

  it('grantUsageRight returns right record', async () => {
    const record = mockRecord(USAGE_RIGHT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      resource_id: 'res-001',
      holder_did: 'did:mycelix:bob',
      right_type: 'Access' as const,
      quota: 100,
    };
    const result = await commons.grantUsageRight(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_commons',
      fn_name: 'grant_usage_right',
      payload: input,
    });
    expect(result.entry.Present.active).toBe(true);
  });

  it('logUsage records usage log', async () => {
    const record = mockRecord(USAGE_LOG_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      resource_id: 'res-001',
      user_did: 'did:mycelix:bob',
      usage_type: 'Irrigation',
      quantity: 50,
      unit: 'liters',
    };
    const result = await commons.logUsage(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_commons',
      fn_name: 'log_usage',
      payload: input,
    });
    expect(result.entry.Present.quantity).toBe(50);
  });

  it('checkUsageQuota returns boolean', async () => {
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

    const input = {
      resource_id: 'res-001',
      user_did: 'did:mycelix:bob',
      requested_amount: 25,
    };
    const result = await commons.checkUsageQuota(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'property_commons',
      fn_name: 'check_usage_quota',
      payload: input,
    });
    expect(result).toBe(true);
  });
});
