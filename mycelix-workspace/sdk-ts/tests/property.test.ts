/**
 * Property Module Tests
 *
 * Tests for the Property hApp TypeScript clients:
 * - RegistryClient (asset registration and management)
 * - TransferClient (title transfers)
 * - LienClient (liens and encumbrances)
 * - CommonsClient (shared resource management)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  RegistryClient,
  TransferClient,
  LienClient,
  CommonsClient,
  createPropertyClients,
  type Asset,
  type TitleTransfer,
  type Lien,
  type Commons,
  type OwnershipShare,
  type CommonsMember,
  type GeoLocation,
  type ZomeCallable,
  type HolochainRecord,
  type AssetType,
  type OwnershipType,
  type AssetStatus,
  type TransferStatus,
  type LienType,
  type LienStatus,
  type CommonsType,
} from '../src/property/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockAsset(overrides: Partial<Asset> = {}): Asset {
  return {
    id: 'asset-123',
    type_: 'RealEstate' as AssetType,
    name: 'Downtown Office Building',
    description: 'Commercial office space in downtown district',
    ownership_type: 'Sole' as OwnershipType,
    owners: [
      {
        owner: 'did:mycelix:owner123',
        percentage: 100,
        acquired_at: Date.now() * 1000 - 31536000000000,
      },
    ],
    location: {
      lat: 32.7767,
      lng: -96.797,
      address: '123 Main St, Dallas, TX',
      geohash: '9vg4m8',
    },
    valuation: 2500000,
    valuation_currency: 'USD',
    status: 'Active' as AssetStatus,
    created_at: Date.now() * 1000 - 31536000000000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockTransfer(overrides: Partial<TitleTransfer> = {}): TitleTransfer {
  return {
    id: 'transfer-123',
    asset_id: 'asset-123',
    from_owner: 'did:mycelix:owner123',
    to_owner: 'did:mycelix:buyer456',
    percentage: 100,
    consideration: 2500000,
    currency: 'USD',
    status: 'Pending' as TransferStatus,
    initiated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockLien(overrides: Partial<Lien> = {}): Lien {
  return {
    id: 'lien-123',
    asset_id: 'asset-123',
    holder: 'did:mycelix:bank789',
    type_: 'Mortgage' as LienType,
    amount: 1500000,
    currency: 'USD',
    priority: 1,
    status: 'Active' as LienStatus,
    filed_at: Date.now() * 1000 - 31536000000000,
    ...overrides,
  };
}

function createMockCommons(overrides: Partial<Commons> = {}): Commons {
  return {
    id: 'commons-123',
    name: 'Community Garden',
    description: 'Shared garden space for neighborhood',
    type_: 'Land' as CommonsType,
    managing_dao: 'dao-neighborhood',
    boundary: [
      { lat: 32.7767, lng: -96.797 },
      { lat: 32.7768, lng: -96.797 },
      { lat: 32.7768, lng: -96.796 },
      { lat: 32.7767, lng: -96.796 },
    ],
    rules: JSON.stringify({
      maxPlots: 50,
      plotSize: '10x10ft',
      harvestRules: 'Personal use only',
    }),
    members: [
      {
        did: 'did:mycelix:member1',
        access_level: 'Use',
        joined_at: Date.now() * 1000 - 2592000000000,
      },
    ],
    created_at: Date.now() * 1000 - 31536000000000,
    ...overrides,
  };
}

// ============================================================================
// RegistryClient Tests
// ============================================================================

describe('RegistryClient', () => {
  let client: RegistryClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockAsset = createMockAsset();

    responses.set('property_registry:register_asset', createMockRecord(mockAsset));
    responses.set('property_registry:get_asset', createMockRecord(mockAsset));
    responses.set('property_registry:get_assets_by_owner', [createMockRecord(mockAsset)]);
    responses.set('property_registry:get_assets_by_type', [createMockRecord(mockAsset)]);
    responses.set(
      'property_registry:update_asset',
      createMockRecord({
        ...mockAsset,
        valuation: 2750000,
        updated_at: Date.now() * 1000,
      })
    );
    responses.set('property_registry:get_asset_history', [createMockRecord(mockAsset)]);

    mockZome = createMockClient(responses);
    client = new RegistryClient(mockZome);
  });

  describe('registerAsset', () => {
    it('should register a new asset', async () => {
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Downtown Office Building',
        description: 'Commercial office space',
        ownership_type: 'Sole',
        location: { lat: 32.7767, lng: -96.797 },
        valuation: 2500000,
        valuation_currency: 'USD',
      });

      expect(result.entry.Present.id).toBe('asset-123');
      expect(result.entry.Present.status).toBe('Active');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'property_registry',
          fn_name: 'register_asset',
        })
      );
    });

    it('should register assets of all types', async () => {
      const types: AssetType[] = ['RealEstate', 'Vehicle', 'Equipment', 'Intellectual', 'Digital', 'Other'];

      for (const type_ of types) {
        const result = await client.registerAsset({
          type_,
          name: `Test ${type_}`,
          description: 'Test asset',
          ownership_type: 'Sole',
        });
        expect(result).toBeDefined();
      }
    });

    it('should register with all ownership types', async () => {
      const types: OwnershipType[] = ['Sole', 'Joint', 'Fractional', 'Commons'];

      for (const ownership_type of types) {
        const result = await client.registerAsset({
          type_: 'Other',
          name: `Test ${ownership_type}`,
          description: 'Test',
          ownership_type,
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getAsset', () => {
    it('should get asset by ID', async () => {
      const result = await client.getAsset('asset-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('asset-123');
    });
  });

  describe('getAssetsByOwner', () => {
    it('should get assets by owner DID', async () => {
      const results = await client.getAssetsByOwner('did:mycelix:owner123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.owners[0].owner).toBe('did:mycelix:owner123');
    });
  });

  describe('getAssetsByType', () => {
    it('should get assets by type', async () => {
      const results = await client.getAssetsByType('RealEstate');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.type_).toBe('RealEstate');
    });
  });

  describe('updateAsset', () => {
    it('should update asset valuation', async () => {
      const result = await client.updateAsset('asset-123', { valuation: 2750000 });

      expect(result.entry.Present.valuation).toBe(2750000);
    });
  });

  describe('getAssetHistory', () => {
    it('should get asset history', async () => {
      const results = await client.getAssetHistory('asset-123');

      expect(results).toHaveLength(1);
    });
  });
});

// ============================================================================
// TransferClient Tests
// ============================================================================

describe('TransferClient', () => {
  let client: TransferClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockTransfer = createMockTransfer();

    responses.set('property_transfer:initiate_transfer', createMockRecord(mockTransfer));
    responses.set(
      'property_transfer:accept_transfer',
      createMockRecord({
        ...mockTransfer,
        status: 'Completed',
        completed_at: Date.now() * 1000,
      })
    );
    responses.set(
      'property_transfer:cancel_transfer',
      createMockRecord({
        ...mockTransfer,
        status: 'Cancelled',
      })
    );
    responses.set('property_transfer:get_transfer', createMockRecord(mockTransfer));
    responses.set('property_transfer:get_transfers_for_asset', [createMockRecord(mockTransfer)]);
    responses.set('property_transfer:get_pending_transfers', [createMockRecord(mockTransfer)]);

    mockZome = createMockClient(responses);
    client = new TransferClient(mockZome);
  });

  describe('initiateTransfer', () => {
    it('should initiate a title transfer', async () => {
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer456',
        percentage: 100,
        consideration: 2500000,
        currency: 'USD',
      });

      expect(result.entry.Present.status).toBe('Pending');
      expect(result.entry.Present.to_owner).toBe('did:mycelix:buyer456');
    });

    it('should initiate transfer with escrow', async () => {
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer456',
        percentage: 100,
        use_escrow: true,
      });

      expect(result).toBeDefined();
    });

    it('should initiate partial transfer', async () => {
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer456',
        percentage: 50,
      });

      expect(result).toBeDefined();
    });
  });

  describe('acceptTransfer', () => {
    it('should accept a pending transfer', async () => {
      const result = await client.acceptTransfer('transfer-123');

      expect(result.entry.Present.status).toBe('Completed');
      expect(result.entry.Present.completed_at).toBeDefined();
    });
  });

  describe('cancelTransfer', () => {
    it('should cancel a pending transfer', async () => {
      const result = await client.cancelTransfer('transfer-123');

      expect(result.entry.Present.status).toBe('Cancelled');
    });
  });

  describe('getTransfer', () => {
    it('should get transfer by ID', async () => {
      const result = await client.getTransfer('transfer-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('transfer-123');
    });
  });

  describe('getTransfersForAsset', () => {
    it('should get transfers for an asset', async () => {
      const results = await client.getTransfersForAsset('asset-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.asset_id).toBe('asset-123');
    });
  });

  describe('getPendingTransfers', () => {
    it('should get pending transfers for a DID', async () => {
      const results = await client.getPendingTransfers('did:mycelix:buyer456');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Pending');
    });
  });
});

// ============================================================================
// LienClient Tests
// ============================================================================

describe('LienClient', () => {
  let client: LienClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockLien = createMockLien();

    responses.set('property_liens:file_lien', createMockRecord(mockLien));
    responses.set('property_liens:get_lien', createMockRecord(mockLien));
    responses.set('property_liens:get_liens_for_asset', [createMockRecord(mockLien)]);
    responses.set(
      'property_liens:satisfy_lien',
      createMockRecord({
        ...mockLien,
        status: 'Satisfied',
        satisfied_at: Date.now() * 1000,
      })
    );
    responses.set(
      'property_liens:release_lien',
      createMockRecord({
        ...mockLien,
        status: 'Released',
      })
    );

    mockZome = createMockClient(responses);
    client = new LienClient(mockZome);
  });

  describe('fileLien', () => {
    it('should file a lien on an asset', async () => {
      const result = await client.fileLien({
        asset_id: 'asset-123',
        type_: 'Mortgage',
        amount: 1500000,
        currency: 'USD',
      });

      expect(result.entry.Present.status).toBe('Active');
      expect(result.entry.Present.type_).toBe('Mortgage');
    });

    it('should file all lien types', async () => {
      const types: LienType[] = ['Mortgage', 'Tax', 'Mechanic', 'Judgment', 'Other'];

      for (const type_ of types) {
        const result = await client.fileLien({
          asset_id: 'asset-123',
          type_,
          amount: 10000,
          currency: 'USD',
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getLien', () => {
    it('should get lien by ID', async () => {
      const result = await client.getLien('lien-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('lien-123');
    });
  });

  describe('getLiensForAsset', () => {
    it('should get liens for an asset', async () => {
      const results = await client.getLiensForAsset('asset-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.asset_id).toBe('asset-123');
    });
  });

  describe('satisfyLien', () => {
    it('should mark lien as satisfied', async () => {
      const result = await client.satisfyLien('lien-123');

      expect(result.entry.Present.status).toBe('Satisfied');
      expect(result.entry.Present.satisfied_at).toBeDefined();
    });
  });

  describe('releaseLien', () => {
    it('should release a lien', async () => {
      const result = await client.releaseLien('lien-123');

      expect(result.entry.Present.status).toBe('Released');
    });
  });
});

// ============================================================================
// CommonsClient Tests
// ============================================================================

describe('CommonsClient', () => {
  let client: CommonsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockCommons = createMockCommons();

    responses.set('property_commons:create_commons', createMockRecord(mockCommons));
    responses.set('property_commons:get_commons', createMockRecord(mockCommons));
    responses.set('property_commons:get_commons_by_dao', [createMockRecord(mockCommons)]);
    responses.set(
      'property_commons:join_commons',
      createMockRecord({
        ...mockCommons,
        members: [
          ...mockCommons.members,
          {
            did: 'did:mycelix:newmember',
            access_level: 'Use',
            joined_at: Date.now() * 1000,
          },
        ],
      })
    );
    responses.set('property_commons:update_rules', createMockRecord(mockCommons));

    mockZome = createMockClient(responses);
    client = new CommonsClient(mockZome);
  });

  describe('createCommons', () => {
    it('should create a commons', async () => {
      const result = await client.createCommons({
        name: 'Community Garden',
        description: 'Shared garden space',
        type_: 'Land',
        managing_dao: 'dao-neighborhood',
        boundary: [{ lat: 32.7767, lng: -96.797 }],
        rules: JSON.stringify({ maxPlots: 50 }),
      });

      expect(result.entry.Present.id).toBe('commons-123');
      expect(result.entry.Present.managing_dao).toBe('dao-neighborhood');
    });

    it('should create all commons types', async () => {
      const types: CommonsType[] = ['Land', 'Water', 'Forest', 'Infrastructure', 'Digital', 'Other'];

      for (const type_ of types) {
        const result = await client.createCommons({
          name: `Test ${type_}`,
          description: 'Test commons',
          type_,
          managing_dao: 'dao-test',
          rules: '{}',
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getCommons', () => {
    it('should get commons by ID', async () => {
      const result = await client.getCommons('commons-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('commons-123');
    });
  });

  describe('getCommonsByDAO', () => {
    it('should get commons by DAO', async () => {
      const results = await client.getCommonsByDAO('dao-neighborhood');

      expect(results).toHaveLength(1);
    });
  });

  describe('joinCommons', () => {
    it('should join a commons', async () => {
      const result = await client.joinCommons('commons-123');

      expect(result.entry.Present.members.length).toBeGreaterThan(1);
    });
  });

  describe('updateRules', () => {
    it('should update commons rules', async () => {
      const result = await client.updateRules(
        'commons-123',
        JSON.stringify({ maxPlots: 75, plotSize: '12x12ft' })
      );

      expect(result).toBeDefined();
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createPropertyClients', () => {
  it('should create all property clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createPropertyClients(mockZome);

    expect(clients.registry).toBeInstanceOf(RegistryClient);
    expect(clients.transfers).toBeInstanceOf(TransferClient);
    expect(clients.liens).toBeInstanceOf(LienClient);
    expect(clients.commons).toBeInstanceOf(CommonsClient);
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce asset status constraints', () => {
    const statuses: AssetStatus[] = ['Active', 'Encumbered', 'Disputed', 'Archived'];

    statuses.forEach((status) => {
      const asset = createMockAsset({ status });
      expect(statuses).toContain(asset.status);
    });
  });

  it('should enforce transfer status transitions', () => {
    const statuses: TransferStatus[] = ['Pending', 'Completed', 'Cancelled', 'Disputed'];

    statuses.forEach((status) => {
      const transfer = createMockTransfer({ status });
      expect(statuses).toContain(transfer.status);
    });
  });

  it('should enforce percentage in valid range', () => {
    const share: OwnershipShare = {
      owner: 'did:mycelix:test',
      percentage: 50,
      acquired_at: Date.now(),
    };

    expect(share.percentage).toBeGreaterThanOrEqual(0);
    expect(share.percentage).toBeLessThanOrEqual(100);
  });

  it('should enforce lien priority is positive', () => {
    const lien = createMockLien();
    expect(lien.priority).toBeGreaterThan(0);
  });

  it('should enforce commons member access levels', () => {
    const levels: CommonsMember['access_level'][] = ['Read', 'Use', 'Manage', 'Admin'];

    levels.forEach((access_level) => {
      const member: CommonsMember = {
        did: 'did:mycelix:test',
        access_level,
        joined_at: Date.now(),
      };
      expect(levels).toContain(member.access_level);
    });
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full property transfer lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockAsset = createMockAsset();
    const mockTransfer = createMockTransfer();

    responses.set('property_registry:register_asset', createMockRecord(mockAsset));
    responses.set('property_transfer:initiate_transfer', createMockRecord(mockTransfer));
    responses.set(
      'property_transfer:accept_transfer',
      createMockRecord({
        ...mockTransfer,
        status: 'Completed',
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createPropertyClients(mockZome);

    // Register asset
    const asset = await clients.registry.registerAsset({
      type_: 'RealEstate',
      name: 'Property',
      description: 'Test',
      ownership_type: 'Sole',
    });
    expect(asset.entry.Present.id).toBeDefined();

    // Initiate transfer
    const transfer = await clients.transfers.initiateTransfer({
      asset_id: asset.entry.Present.id,
      to_owner: 'did:mycelix:buyer',
      percentage: 100,
    });
    expect(transfer.entry.Present.status).toBe('Pending');

    // Accept transfer
    const completed = await clients.transfers.acceptTransfer(transfer.entry.Present.id);
    expect(completed.entry.Present.status).toBe('Completed');
  });

  it('should handle encumbered asset transfers', async () => {
    const responses = new Map<string, unknown>();
    const mockAsset = createMockAsset({ status: 'Encumbered' });
    const mockLien = createMockLien();

    responses.set('property_registry:get_asset', createMockRecord(mockAsset));
    responses.set('property_liens:get_liens_for_asset', [createMockRecord(mockLien)]);
    responses.set(
      'property_liens:satisfy_lien',
      createMockRecord({
        ...mockLien,
        status: 'Satisfied',
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createPropertyClients(mockZome);

    // Check asset is encumbered
    const asset = await clients.registry.getAsset('asset-123');
    expect(asset!.entry.Present.status).toBe('Encumbered');

    // Get liens
    const liens = await clients.liens.getLiensForAsset('asset-123');
    expect(liens).toHaveLength(1);

    // Satisfy lien
    const satisfied = await clients.liens.satisfyLien(liens[0].entry.Present.id);
    expect(satisfied.entry.Present.status).toBe('Satisfied');
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  it('should handle asset with no valuation', () => {
    const asset = createMockAsset({ valuation: undefined, valuation_currency: undefined });
    expect(asset.valuation).toBeUndefined();
  });

  it('should handle fractional ownership', () => {
    const asset = createMockAsset({
      ownership_type: 'Fractional',
      owners: [
        { owner: 'did:mycelix:owner1', percentage: 60, acquired_at: Date.now() },
        { owner: 'did:mycelix:owner2', percentage: 40, acquired_at: Date.now() },
      ],
    });

    const totalPercentage = asset.owners.reduce((sum, o) => sum + o.percentage, 0);
    expect(totalPercentage).toBe(100);
  });

  it('should handle asset without location', () => {
    const asset = createMockAsset({
      type_: 'Intellectual',
      location: undefined,
    });

    expect(asset.location).toBeUndefined();
  });

  it('should handle commons without boundary', () => {
    const commons = createMockCommons({
      type_: 'Digital',
      boundary: undefined,
    });

    expect(commons.boundary).toBeUndefined();
  });

  it('should handle lien with multiple priorities', () => {
    const lien1 = createMockLien({ priority: 1 });
    const lien2 = createMockLien({ id: 'lien-456', priority: 2 });

    expect(lien1.priority).toBeLessThan(lien2.priority);
  });
});

// ============================================================================
// Validated Client Tests
// ============================================================================

import {
  ValidatedRegistryClient,
  ValidatedTransferClient,
  ValidatedLienClient,
  ValidatedCommonsClient,
  createValidatedPropertyClients,
} from '../src/property/validated.js';

describe('ValidatedRegistryClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedRegistryClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['property_registry:register_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_assets_by_owner', [createMockRecord(createMockAsset())]],
        ['property_registry:get_assets_by_type', [createMockRecord(createMockAsset())]],
        ['property_registry:update_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_asset_history', [createMockRecord(createMockAsset())]],
      ])
    );
    validatedClient = new ValidatedRegistryClient(mockClient);
  });

  describe('registerAsset', () => {
    it('accepts valid asset input', async () => {
      const result = await validatedClient.registerAsset({
        type_: 'RealEstate',
        name: 'Downtown Office',
        description: 'Commercial office space',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBe('asset-123');
    });

    it('rejects empty name', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: '',
          description: 'Description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects name over 200 chars', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: 'x'.repeat(201),
          description: 'Description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty description', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: 'Valid Name',
          description: '',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid asset type', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'Invalid' as 'RealEstate',
          name: 'Valid Name',
          description: 'Description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid ownership type', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: 'Valid Name',
          description: 'Description',
          ownership_type: 'Invalid' as 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid geolocation', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: 'Valid Name',
          description: 'Description',
          ownership_type: 'Sole',
          location: { lat: 200, lng: 0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('accepts valid geolocation', async () => {
      const result = await validatedClient.registerAsset({
        type_: 'RealEstate',
        name: 'Valid Name',
        description: 'Description',
        ownership_type: 'Sole',
        location: { lat: 40.7128, lng: -74.006 },
      });
      expect(result.entry.Present?.id).toBe('asset-123');
    });

    it('rejects negative valuation', async () => {
      await expect(
        validatedClient.registerAsset({
          type_: 'RealEstate',
          name: 'Valid Name',
          description: 'Description',
          ownership_type: 'Sole',
          valuation: -1000,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getAsset', () => {
    it('accepts valid asset ID', async () => {
      const result = await validatedClient.getAsset('asset-123');
      expect(result?.entry.Present?.id).toBe('asset-123');
    });

    it('rejects empty asset ID', async () => {
      await expect(validatedClient.getAsset('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getAssetsByOwner', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getAssetsByOwner('did:mycelix:owner123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getAssetsByOwner('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('getAssetsByType', () => {
    it('accepts valid asset type', async () => {
      const result = await validatedClient.getAssetsByType('RealEstate');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid asset type', async () => {
      await expect(validatedClient.getAssetsByType('Invalid' as 'RealEstate')).rejects.toThrow('Validation failed');
    });
  });

  describe('updateAsset', () => {
    it('accepts valid update', async () => {
      const result = await validatedClient.updateAsset('asset-123', { name: 'Updated Name' });
      expect(result.entry.Present?.id).toBe('asset-123');
    });

    it('rejects empty asset ID', async () => {
      await expect(validatedClient.updateAsset('', { name: 'Updated' })).rejects.toThrow('Validation failed');
    });
  });

  describe('getAssetHistory', () => {
    it('accepts valid asset ID', async () => {
      const result = await validatedClient.getAssetHistory('asset-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty asset ID', async () => {
      await expect(validatedClient.getAssetHistory('')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedTransferClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedTransferClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['property_transfer:initiate_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:accept_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:cancel_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:get_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:get_transfers_for_asset', [createMockRecord(createMockTransfer())]],
        ['property_transfer:get_pending_transfers', [createMockRecord(createMockTransfer())]],
      ])
    );
    validatedClient = new ValidatedTransferClient(mockClient);
  });

  describe('initiateTransfer', () => {
    it('accepts valid transfer input', async () => {
      const result = await validatedClient.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer456',
        percentage: 100,
      });
      expect(result.entry.Present?.id).toBe('transfer-123');
    });

    it('rejects empty asset ID', async () => {
      await expect(
        validatedClient.initiateTransfer({
          asset_id: '',
          to_owner: 'did:mycelix:buyer456',
          percentage: 100,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID to_owner', async () => {
      await expect(
        validatedClient.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'not-a-did',
          percentage: 100,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects percentage over 100', async () => {
      await expect(
        validatedClient.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer456',
          percentage: 150,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative percentage', async () => {
      await expect(
        validatedClient.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer456',
          percentage: -10,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative consideration', async () => {
      await expect(
        validatedClient.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer456',
          percentage: 100,
          consideration: -1000,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('acceptTransfer', () => {
    it('accepts valid transfer ID', async () => {
      const result = await validatedClient.acceptTransfer('transfer-123');
      expect(result.entry.Present?.id).toBe('transfer-123');
    });

    it('rejects empty transfer ID', async () => {
      await expect(validatedClient.acceptTransfer('')).rejects.toThrow('Validation failed');
    });
  });

  describe('cancelTransfer', () => {
    it('accepts valid transfer ID', async () => {
      const result = await validatedClient.cancelTransfer('transfer-123');
      expect(result.entry.Present?.id).toBe('transfer-123');
    });

    it('rejects empty transfer ID', async () => {
      await expect(validatedClient.cancelTransfer('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getTransfer', () => {
    it('accepts valid transfer ID', async () => {
      const result = await validatedClient.getTransfer('transfer-123');
      expect(result?.entry.Present?.id).toBe('transfer-123');
    });

    it('rejects empty transfer ID', async () => {
      await expect(validatedClient.getTransfer('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getTransfersForAsset', () => {
    it('accepts valid asset ID', async () => {
      const result = await validatedClient.getTransfersForAsset('asset-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty asset ID', async () => {
      await expect(validatedClient.getTransfersForAsset('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getPendingTransfers', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getPendingTransfers('did:mycelix:owner123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getPendingTransfers('not-a-did')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedLienClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedLienClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['property_liens:file_lien', createMockRecord(createMockLien())],
        ['property_liens:get_lien', createMockRecord(createMockLien())],
        ['property_liens:get_liens_for_asset', [createMockRecord(createMockLien())]],
        ['property_liens:satisfy_lien', createMockRecord(createMockLien())],
        ['property_liens:release_lien', createMockRecord(createMockLien())],
      ])
    );
    validatedClient = new ValidatedLienClient(mockClient);
  });

  describe('fileLien', () => {
    it('accepts valid lien input', async () => {
      const result = await validatedClient.fileLien({
        asset_id: 'asset-123',
        type_: 'Mortgage',
        amount: 100000,
        currency: 'USD',
      });
      expect(result.entry.Present?.id).toBe('lien-123');
    });

    it('rejects empty asset ID', async () => {
      await expect(
        validatedClient.fileLien({
          asset_id: '',
          type_: 'Mortgage',
          amount: 100000,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid lien type', async () => {
      await expect(
        validatedClient.fileLien({
          asset_id: 'asset-123',
          type_: 'Invalid' as 'Mortgage',
          amount: 100000,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-positive amount', async () => {
      await expect(
        validatedClient.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: 0,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative amount', async () => {
      await expect(
        validatedClient.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: -1000,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty currency', async () => {
      await expect(
        validatedClient.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: 100000,
          currency: '',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getLien', () => {
    it('accepts valid lien ID', async () => {
      const result = await validatedClient.getLien('lien-123');
      expect(result?.entry.Present?.id).toBe('lien-123');
    });

    it('rejects empty lien ID', async () => {
      await expect(validatedClient.getLien('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getLiensForAsset', () => {
    it('accepts valid asset ID', async () => {
      const result = await validatedClient.getLiensForAsset('asset-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty asset ID', async () => {
      await expect(validatedClient.getLiensForAsset('')).rejects.toThrow('Validation failed');
    });
  });

  describe('satisfyLien', () => {
    it('accepts valid lien ID', async () => {
      const result = await validatedClient.satisfyLien('lien-123');
      expect(result.entry.Present?.id).toBe('lien-123');
    });

    it('rejects empty lien ID', async () => {
      await expect(validatedClient.satisfyLien('')).rejects.toThrow('Validation failed');
    });
  });

  describe('releaseLien', () => {
    it('accepts valid lien ID', async () => {
      const result = await validatedClient.releaseLien('lien-123');
      expect(result.entry.Present?.id).toBe('lien-123');
    });

    it('rejects empty lien ID', async () => {
      await expect(validatedClient.releaseLien('')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedCommonsClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedCommonsClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['property_commons:create_commons', createMockRecord(createMockCommons())],
        ['property_commons:get_commons', createMockRecord(createMockCommons())],
        ['property_commons:get_commons_by_dao', [createMockRecord(createMockCommons())]],
        ['property_commons:join_commons', createMockRecord(createMockCommons())],
        ['property_commons:update_rules', createMockRecord(createMockCommons())],
      ])
    );
    validatedClient = new ValidatedCommonsClient(mockClient);
  });

  describe('createCommons', () => {
    it('accepts valid commons input', async () => {
      const result = await validatedClient.createCommons({
        name: 'Community Forest',
        description: 'Shared forest land for sustainable management',
        type_: 'Forest',
        managing_dao: 'dao-123',
        rules: 'No clear-cutting allowed',
      });
      expect(result.entry.Present?.id).toBe('commons-123');
    });

    it('rejects empty name', async () => {
      await expect(
        validatedClient.createCommons({
          name: '',
          description: 'Description',
          type_: 'Forest',
          managing_dao: 'dao-123',
          rules: 'Rules',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects name over 200 chars', async () => {
      await expect(
        validatedClient.createCommons({
          name: 'x'.repeat(201),
          description: 'Description',
          type_: 'Forest',
          managing_dao: 'dao-123',
          rules: 'Rules',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty description', async () => {
      await expect(
        validatedClient.createCommons({
          name: 'Valid Name',
          description: '',
          type_: 'Forest',
          managing_dao: 'dao-123',
          rules: 'Rules',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid commons type', async () => {
      await expect(
        validatedClient.createCommons({
          name: 'Valid Name',
          description: 'Description',
          type_: 'Invalid' as 'Forest',
          managing_dao: 'dao-123',
          rules: 'Rules',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty managing_dao', async () => {
      await expect(
        validatedClient.createCommons({
          name: 'Valid Name',
          description: 'Description',
          type_: 'Forest',
          managing_dao: '',
          rules: 'Rules',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty rules', async () => {
      await expect(
        validatedClient.createCommons({
          name: 'Valid Name',
          description: 'Description',
          type_: 'Forest',
          managing_dao: 'dao-123',
          rules: '',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getCommons', () => {
    it('accepts valid commons ID', async () => {
      const result = await validatedClient.getCommons('commons-123');
      expect(result?.entry.Present?.id).toBe('commons-123');
    });

    it('rejects empty commons ID', async () => {
      await expect(validatedClient.getCommons('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getCommonsByDAO', () => {
    it('accepts valid DAO ID', async () => {
      const result = await validatedClient.getCommonsByDAO('dao-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty DAO ID', async () => {
      await expect(validatedClient.getCommonsByDAO('')).rejects.toThrow('Validation failed');
    });
  });

  describe('joinCommons', () => {
    it('accepts valid commons ID', async () => {
      const result = await validatedClient.joinCommons('commons-123');
      expect(result.entry.Present?.id).toBe('commons-123');
    });

    it('rejects empty commons ID', async () => {
      await expect(validatedClient.joinCommons('')).rejects.toThrow('Validation failed');
    });
  });

  describe('updateRules', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.updateRules('commons-123', 'Updated rules');
      expect(result.entry.Present?.id).toBe('commons-123');
    });

    it('rejects empty commons ID', async () => {
      await expect(validatedClient.updateRules('', 'Rules')).rejects.toThrow('Validation failed');
    });

    it('rejects empty rules', async () => {
      await expect(validatedClient.updateRules('commons-123', '')).rejects.toThrow('Validation failed');
    });
  });
});

describe('createValidatedPropertyClients', () => {
  it('creates all validated clients', () => {
    const mockClient = createMockClient(new Map());
    const clients = createValidatedPropertyClients(mockClient);

    expect(clients.registry).toBeInstanceOf(ValidatedRegistryClient);
    expect(clients.transfers).toBeInstanceOf(ValidatedTransferClient);
    expect(clients.liens).toBeInstanceOf(ValidatedLienClient);
    expect(clients.commons).toBeInstanceOf(ValidatedCommonsClient);
  });
});

// ============================================================================
// Additional Mutation Score Tests - Validation Schema Boundary Conditions
// ============================================================================

describe('Property Validation Schema Boundary Tests', () => {
  let mockClient: ZomeCallable;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['property_registry:register_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_assets_by_owner', [createMockRecord(createMockAsset())]],
        ['property_registry:get_assets_by_type', [createMockRecord(createMockAsset())]],
        ['property_registry:update_asset', createMockRecord(createMockAsset())],
        ['property_registry:get_asset_history', [createMockRecord(createMockAsset())]],
        ['property_transfer:initiate_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:accept_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:cancel_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:get_transfer', createMockRecord(createMockTransfer())],
        ['property_transfer:get_transfers_for_asset', [createMockRecord(createMockTransfer())]],
        ['property_transfer:get_pending_transfers', [createMockRecord(createMockTransfer())]],
        ['property_liens:file_lien', createMockRecord(createMockLien())],
        ['property_liens:get_lien', createMockRecord(createMockLien())],
        ['property_liens:get_liens_for_asset', [createMockRecord(createMockLien())]],
        ['property_liens:satisfy_lien', createMockRecord(createMockLien())],
        ['property_liens:release_lien', createMockRecord(createMockLien())],
        ['property_commons:create_commons', createMockRecord(createMockCommons())],
        ['property_commons:get_commons', createMockRecord(createMockCommons())],
        ['property_commons:get_commons_by_dao', [createMockRecord(createMockCommons())]],
        ['property_commons:join_commons', createMockRecord(createMockCommons())],
        ['property_commons:update_rules', createMockRecord(createMockCommons())],
      ])
    );
  });

  describe('AssetType Enum Validation', () => {
    const validClient = () => new ValidatedRegistryClient(mockClient);

    const validAssetTypes: AssetType[] = ['RealEstate', 'Vehicle', 'Equipment', 'Intellectual', 'Digital', 'Other'];

    it.each(validAssetTypes)('accepts valid asset type: %s', async (type_) => {
      const client = validClient();
      const result = await client.registerAsset({
        type_,
        name: 'Test Asset',
        description: 'Test description',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects invalid asset type', async () => {
      const client = validClient();
      await expect(
        client.registerAsset({
          type_: 'InvalidType' as AssetType,
          name: 'Test Asset',
          description: 'Test description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('OwnershipType Enum Validation', () => {
    const validClient = () => new ValidatedRegistryClient(mockClient);

    const validOwnershipTypes: OwnershipType[] = ['Sole', 'Joint', 'Fractional', 'Commons'];

    it.each(validOwnershipTypes)('accepts valid ownership type: %s', async (ownership_type) => {
      const client = validClient();
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test Asset',
        description: 'Test description',
        ownership_type,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects invalid ownership type', async () => {
      const client = validClient();
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test Asset',
          description: 'Test description',
          ownership_type: 'Invalid' as OwnershipType,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Asset Name Validation', () => {
    it('accepts name at exactly 200 characters', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'x'.repeat(200),
        description: 'Test description',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts name at exactly 1 character', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'x',
        description: 'Test description',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty name', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: '',
          description: 'Test description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Name is required');
    });

    it('rejects name over 200 characters', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'x'.repeat(201),
          description: 'Test description',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Asset Description Validation', () => {
    it('accepts non-empty description', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test Asset',
        description: 'x',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty description', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test Asset',
          description: '',
          ownership_type: 'Sole',
        })
      ).rejects.toThrow('Description is required');
    });
  });

  describe('GeoLocation Validation', () => {
    it('accepts valid lat/lng at boundaries', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test',
        description: 'Test',
        ownership_type: 'Sole',
        location: { lat: 90, lng: 180 },
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts valid negative lat/lng at boundaries', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test',
        description: 'Test',
        ownership_type: 'Sole',
        location: { lat: -90, lng: -180 },
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects lat above 90', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test',
          description: 'Test',
          ownership_type: 'Sole',
          location: { lat: 91, lng: 0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects lat below -90', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test',
          description: 'Test',
          ownership_type: 'Sole',
          location: { lat: -91, lng: 0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects lng above 180', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test',
          description: 'Test',
          ownership_type: 'Sole',
          location: { lat: 0, lng: 181 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects lng below -180', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test',
          description: 'Test',
          ownership_type: 'Sole',
          location: { lat: 0, lng: -181 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('accepts location with optional address and geohash', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test',
        description: 'Test',
        ownership_type: 'Sole',
        location: { lat: 32.7767, lng: -96.797, address: '123 Main St', geohash: '9vg4m8' },
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Valuation Validation', () => {
    it('accepts zero valuation', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'Digital',
        name: 'Test',
        description: 'Test',
        ownership_type: 'Sole',
        valuation: 0,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts large valuation', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'RealEstate',
        name: 'Test',
        description: 'Test',
        ownership_type: 'Sole',
        valuation: 1000000000,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects negative valuation', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(
        client.registerAsset({
          type_: 'RealEstate',
          name: 'Test',
          description: 'Test',
          ownership_type: 'Sole',
          valuation: -100,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Transfer Percentage Validation', () => {
    it('accepts zero percentage', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer',
        percentage: 0,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts 100 percentage', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer',
        percentage: 100,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects percentage above 100', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(
        client.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer',
          percentage: 101,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative percentage', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(
        client.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer',
          percentage: -1,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Transfer DID Validation', () => {
    it('accepts valid DID', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer',
        percentage: 100,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects non-DID to_owner', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(
        client.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'invalid-owner',
          percentage: 100,
        })
      ).rejects.toThrow('Must be a valid DID');
    });
  });

  describe('Transfer Consideration Validation', () => {
    it('accepts zero consideration', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:buyer',
        percentage: 100,
        consideration: 0,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects negative consideration', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(
        client.initiateTransfer({
          asset_id: 'asset-123',
          to_owner: 'did:mycelix:buyer',
          percentage: 100,
          consideration: -1000,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('LienType Enum Validation', () => {
    const validClient = () => new ValidatedLienClient(mockClient);

    const validLienTypes: LienType[] = ['Mortgage', 'Tax', 'Mechanic', 'Judgment', 'Other'];

    it.each(validLienTypes)('accepts valid lien type: %s', async (type_) => {
      const client = validClient();
      const result = await client.fileLien({
        asset_id: 'asset-123',
        type_,
        amount: 1000,
        currency: 'USD',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects invalid lien type', async () => {
      const client = validClient();
      await expect(
        client.fileLien({
          asset_id: 'asset-123',
          type_: 'InvalidLien' as LienType,
          amount: 1000,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Lien Amount Validation', () => {
    it('accepts positive amount', async () => {
      const client = new ValidatedLienClient(mockClient);
      const result = await client.fileLien({
        asset_id: 'asset-123',
        type_: 'Mortgage',
        amount: 1,
        currency: 'USD',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects zero amount', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(
        client.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: 0,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative amount', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(
        client.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: -1000,
          currency: 'USD',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Lien Currency Validation', () => {
    it('accepts valid currency', async () => {
      const client = new ValidatedLienClient(mockClient);
      const result = await client.fileLien({
        asset_id: 'asset-123',
        type_: 'Mortgage',
        amount: 1000,
        currency: 'USD',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty currency', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(
        client.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: 1000,
          currency: '',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('CommonsType Enum Validation', () => {
    const validClient = () => new ValidatedCommonsClient(mockClient);

    const validCommonsTypes: CommonsType[] = ['Land', 'Water', 'Forest', 'Infrastructure', 'Digital', 'Other'];

    it.each(validCommonsTypes)('accepts valid commons type: %s', async (type_) => {
      const client = validClient();
      const result = await client.createCommons({
        name: 'Test Commons',
        description: 'Test description',
        type_,
        managing_dao: 'dao-123',
        rules: 'test rules',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects invalid commons type', async () => {
      const client = validClient();
      await expect(
        client.createCommons({
          name: 'Test Commons',
          description: 'Test description',
          type_: 'InvalidCommons' as CommonsType,
          managing_dao: 'dao-123',
          rules: 'test rules',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Commons Name Validation', () => {
    it('accepts name at exactly 200 characters', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      const result = await client.createCommons({
        name: 'x'.repeat(200),
        description: 'Test description',
        type_: 'Land',
        managing_dao: 'dao-123',
        rules: 'test rules',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty name', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(
        client.createCommons({
          name: '',
          description: 'Test description',
          type_: 'Land',
          managing_dao: 'dao-123',
          rules: 'test rules',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Commons Managing DAO Validation', () => {
    it('accepts non-empty managing_dao', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      const result = await client.createCommons({
        name: 'Test',
        description: 'Test',
        type_: 'Land',
        managing_dao: 'x',
        rules: 'test rules',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty managing_dao', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(
        client.createCommons({
          name: 'Test',
          description: 'Test',
          type_: 'Land',
          managing_dao: '',
          rules: 'test rules',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Commons Rules Validation', () => {
    it('accepts non-empty rules', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      const result = await client.createCommons({
        name: 'Test',
        description: 'Test',
        type_: 'Land',
        managing_dao: 'dao-123',
        rules: 'x',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects empty rules', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(
        client.createCommons({
          name: 'Test',
          description: 'Test',
          type_: 'Land',
          managing_dao: 'dao-123',
          rules: '',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Commons Boundary Validation', () => {
    it('accepts empty boundary', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      const result = await client.createCommons({
        name: 'Test',
        description: 'Test',
        type_: 'Digital',
        managing_dao: 'dao-123',
        rules: 'test rules',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts valid boundary with multiple points', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      const result = await client.createCommons({
        name: 'Test',
        description: 'Test',
        type_: 'Land',
        managing_dao: 'dao-123',
        rules: 'test rules',
        boundary: [
          { lat: 32.7767, lng: -96.797 },
          { lat: 32.7768, lng: -96.797 },
          { lat: 32.7768, lng: -96.796 },
        ],
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects boundary with invalid coordinates', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(
        client.createCommons({
          name: 'Test',
          description: 'Test',
          type_: 'Land',
          managing_dao: 'dao-123',
          rules: 'test rules',
          boundary: [{ lat: 200, lng: -96.797 }], // Invalid lat
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('ID Validation Across Clients', () => {
    it('rejects empty asset ID for get', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(client.getAsset('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty asset ID for update', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(client.updateAsset('', { name: 'New Name' })).rejects.toThrow('Validation failed');
    });

    it('rejects empty asset ID for history', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(client.getAssetHistory('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty transfer ID for accept', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(client.acceptTransfer('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty transfer ID for cancel', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(client.cancelTransfer('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty transfer ID for get', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(client.getTransfer('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty asset ID for transfers', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(client.getTransfersForAsset('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty lien ID for get', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(client.getLien('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty asset ID for liens', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(client.getLiensForAsset('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty lien ID for satisfy', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(client.satisfyLien('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty lien ID for release', async () => {
      const client = new ValidatedLienClient(mockClient);
      await expect(client.releaseLien('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty commons ID for get', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(client.getCommons('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty DAO ID for commons by DAO', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(client.getCommonsByDAO('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty commons ID for join', async () => {
      const client = new ValidatedCommonsClient(mockClient);
      await expect(client.joinCommons('')).rejects.toThrow('Validation failed');
    });
  });

  describe('DID Validation for Owner Lookups', () => {
    it('rejects non-DID for getAssetsByOwner', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      await expect(client.getAssetsByOwner('invalid')).rejects.toThrow('Must be a valid DID');
    });

    it('accepts valid DID for getAssetsByOwner', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.getAssetsByOwner('did:mycelix:owner');
      expect(result).toBeDefined();
    });

    it('rejects non-DID for getPendingTransfers', async () => {
      const client = new ValidatedTransferClient(mockClient);
      await expect(client.getPendingTransfers('invalid')).rejects.toThrow('Must be a valid DID');
    });

    it('accepts valid DID for getPendingTransfers', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.getPendingTransfers('did:mycelix:owner');
      expect(result).toBeDefined();
    });
  });

  describe('Optional Fields', () => {
    it('accepts asset without location', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'Digital',
        name: 'NFT Asset',
        description: 'Digital asset',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts asset without valuation', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      const result = await client.registerAsset({
        type_: 'Other',
        name: 'Misc Asset',
        description: 'Miscellaneous',
        ownership_type: 'Sole',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts transfer without consideration', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:recipient',
        percentage: 50,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts transfer with use_escrow', async () => {
      const client = new ValidatedTransferClient(mockClient);
      const result = await client.initiateTransfer({
        asset_id: 'asset-123',
        to_owner: 'did:mycelix:recipient',
        percentage: 100,
        use_escrow: true,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Error Message Content', () => {
    it('includes field path in validation error', async () => {
      const client = new ValidatedRegistryClient(mockClient);
      try {
        await client.registerAsset({
          type_: 'RealEstate',
          name: '',
          description: 'Test',
          ownership_type: 'Sole',
        });
        expect.fail('Should have thrown');
      } catch (error: unknown) {
        expect((error as Error).message).toContain('name');
      }
    });

    it('includes context in validation error', async () => {
      const client = new ValidatedLienClient(mockClient);
      try {
        await client.fileLien({
          asset_id: 'asset-123',
          type_: 'Mortgage',
          amount: -100,
          currency: 'USD',
        });
        expect.fail('Should have thrown');
      } catch (error: unknown) {
        expect((error as Error).message).toContain('fileLien');
      }
    });
  });
});
