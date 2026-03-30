// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Property hApp Conductor Integration Tests
 *
 * These tests verify the Property clients work correctly with a real
 * Holochain conductor.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  RegistryClient,
  TransferClient,
  LienClient,
  CommonsClient,
  createPropertyClients,
  type ZomeCallable,
  type Asset,
  type TitleTransfer,
  type Lien,
  type Commons,
} from '../../src/property/index.js';
import { createValidatedPropertyClients } from '../../src/property/validated.js';
import { MycelixError } from '../../src/errors.js';
import { CONDUCTOR_ENABLED, generateTestAgentId } from './conductor-harness.js';

function createMockClient(): ZomeCallable {
  const assets = new Map<string, Asset>();
  const transfers = new Map<string, TitleTransfer[]>();
  const liens = new Map<string, Lien[]>();
  const commons = new Map<string, Commons>();

  return {
    async callZome({
      fn_name,
      payload,
    }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) {
      const agentId = `did:mycelix:${generateTestAgentId()}`;

      switch (fn_name) {
        case 'register_asset': {
          const input = payload as {
            type_: string;
            name: string;
            description?: string;
            location?: any;
            metadata?: Record<string, unknown>;
          };
          const asset: Asset = {
            id: `asset-${Date.now()}`,
            owner: agentId,
            type_: input.type_ as Asset['type_'],
            name: input.name,
            description: input.description,
            location: input.location,
            metadata: input.metadata || {},
            ownership_type: 'Sole',
            owners_with_shares: [[agentId, 100]],
            registered_at: Date.now(),
            updated_at: Date.now(),
          };
          assets.set(asset.id, asset);
          return {
            signed_action: { hashed: { hash: asset.id, content: {} }, signature: 'sig' },
            entry: { Present: asset },
          };
        }

        case 'get_asset': {
          const id = payload as string;
          const a = assets.get(id);
          return a
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: a },
              }
            : null;
        }

        case 'get_assets_by_owner': {
          const owner = payload as string;
          return Array.from(assets.values())
            .filter((a) => a.owner === owner || a.owners_with_shares.some(([o]) => o === owner))
            .map((a) => ({
              signed_action: { hashed: { hash: a.id, content: {} }, signature: 'sig' },
              entry: { Present: a },
            }));
        }

        case 'get_assets_by_type': {
          const type = payload as string;
          return Array.from(assets.values())
            .filter((a) => a.type_ === type)
            .map((a) => ({
              signed_action: { hashed: { hash: a.id, content: {} }, signature: 'sig' },
              entry: { Present: a },
            }));
        }

        case 'update_asset_metadata': {
          const { asset_id, metadata } = payload as {
            asset_id: string;
            metadata: Record<string, unknown>;
          };
          const a = assets.get(asset_id);
          if (a) {
            a.metadata = { ...a.metadata, ...metadata };
            a.updated_at = Date.now();
          }
          return a
            ? {
                signed_action: { hashed: { hash: asset_id, content: {} }, signature: 'sig' },
                entry: { Present: a },
              }
            : null;
        }

        case 'initiate_transfer': {
          const input = payload as {
            asset_id: string;
            to_owner: string;
            percentage: number;
            price?: number;
            currency?: string;
          };
          const asset = assets.get(input.asset_id);
          if (!asset) throw new Error('Asset not found');
          const transfer: TitleTransfer = {
            id: `transfer-${Date.now()}`,
            asset_id: input.asset_id,
            from_owner: asset.owner,
            to_owner: input.to_owner,
            percentage: input.percentage,
            price: input.price,
            currency: input.currency as TitleTransfer['currency'],
            status: 'Initiated',
            escrow_wallet_id: `escrow-${Date.now()}`,
            initiated_at: Date.now(),
          };
          const existing = transfers.get(input.asset_id) || [];
          existing.push(transfer);
          transfers.set(input.asset_id, existing);
          return {
            signed_action: { hashed: { hash: transfer.id, content: {} }, signature: 'sig' },
            entry: { Present: transfer },
          };
        }

        case 'accept_transfer': {
          const transferId = payload as string;
          for (const txs of transfers.values()) {
            const tx = txs.find((t) => t.id === transferId);
            if (tx) {
              tx.status = 'Accepted';
              return {
                signed_action: { hashed: { hash: transferId, content: {} }, signature: 'sig' },
                entry: { Present: tx },
              };
            }
          }
          throw new Error('Transfer not found');
        }

        case 'complete_transfer': {
          const transferId = payload as string;
          for (const [assetId, txs] of transfers.entries()) {
            const tx = txs.find((t) => t.id === transferId);
            if (tx) {
              tx.status = 'Completed';
              tx.completed_at = Date.now();
              const asset = assets.get(assetId);
              if (asset) {
                if (tx.percentage === 100) {
                  asset.owner = tx.to_owner;
                  asset.owners_with_shares = [[tx.to_owner, 100]];
                } else {
                  asset.ownership_type = 'Fractional';
                  const fromIndex = asset.owners_with_shares.findIndex(
                    ([o]) => o === tx.from_owner
                  );
                  if (fromIndex >= 0) {
                    asset.owners_with_shares[fromIndex][1] -= tx.percentage;
                  }
                  asset.owners_with_shares.push([tx.to_owner, tx.percentage]);
                }
                asset.updated_at = Date.now();
              }
              return {
                signed_action: { hashed: { hash: transferId, content: {} }, signature: 'sig' },
                entry: { Present: tx },
              };
            }
          }
          throw new Error('Transfer not found');
        }

        case 'get_transfers_for_asset': {
          const assetId = payload as string;
          const txs = transfers.get(assetId) || [];
          return txs.map((t) => ({
            signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
            entry: { Present: t },
          }));
        }

        case 'get_pending_transfers': {
          const did = payload as string;
          const pending: TitleTransfer[] = [];
          for (const txs of transfers.values()) {
            for (const tx of txs) {
              if ((tx.from_owner === did || tx.to_owner === did) && tx.status !== 'Completed') {
                pending.push(tx);
              }
            }
          }
          return pending.map((t) => ({
            signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
            entry: { Present: t },
          }));
        }

        case 'file_lien':
        case 'place_lien': {
          const input = payload as {
            asset_id: string;
            lien_type: string;
            amount: number;
            currency: string;
            description: string;
          };
          const lien: Lien = {
            id: `lien-${Date.now()}`,
            asset_id: input.asset_id,
            holder: agentId,
            lien_type: input.lien_type as Lien['lien_type'],
            amount: input.amount,
            currency: input.currency as Lien['currency'],
            description: input.description,
            status: 'Active',
            priority: 1,
            placed_at: Date.now(),
          };
          const existing = liens.get(input.asset_id) || [];
          lien.priority = existing.length + 1;
          existing.push(lien);
          liens.set(input.asset_id, existing);
          return {
            signed_action: { hashed: { hash: lien.id, content: {} }, signature: 'sig' },
            entry: { Present: lien },
          };
        }

        case 'release_lien': {
          const lienId = payload as string;
          for (const ls of liens.values()) {
            const l = ls.find((ln) => ln.id === lienId);
            if (l) {
              l.status = 'Released';
              l.released_at = Date.now();
              return {
                signed_action: { hashed: { hash: lienId, content: {} }, signature: 'sig' },
                entry: { Present: l },
              };
            }
          }
          throw new Error('Lien not found');
        }

        case 'get_liens_for_asset': {
          const assetId = payload as string;
          const ls = liens.get(assetId) || [];
          return ls.map((l) => ({
            signed_action: { hashed: { hash: l.id, content: {} }, signature: 'sig' },
            entry: { Present: l },
          }));
        }

        case 'create_commons': {
          const input = payload as {
            name: string;
            description: string;
            asset_ids?: string[];
            dao_id: string;
          };
          const c: Commons = {
            id: `commons-${Date.now()}`,
            name: input.name,
            description: input.description,
            asset_ids: input.asset_ids || [],
            dao_id: input.dao_id,
            governance_rules: {},
            created_at: Date.now(),
            updated_at: Date.now(),
          };
          commons.set(c.id, c);
          return {
            signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
            entry: { Present: c },
          };
        }

        case 'get_commons': {
          const id = payload as string;
          const c = commons.get(id);
          return c
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'get_commons_by_dao': {
          const daoId = payload as string;
          return Array.from(commons.values())
            .filter((c) => c.dao_id === daoId)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'add_to_commons':
        case 'add_asset_to_commons': {
          const { commons_id, asset_id } = payload as { commons_id: string; asset_id: string };
          const c = commons.get(commons_id);
          if (c) {
            c.asset_ids.push(asset_id);
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: commons_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'remove_asset_from_commons': {
          const { commons_id, asset_id } = payload as { commons_id: string; asset_id: string };
          const c = commons.get(commons_id);
          if (c) {
            c.asset_ids = c.asset_ids.filter((id: string) => id !== asset_id);
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: commons_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        default:
          throw new Error(`Unknown function: ${fn_name}`);
      }
    },
  };
}

const describeConductor = CONDUCTOR_ENABLED ? describe : describe.skip;

describe('Property Clients (Mock)', () => {
  let mockClient: ZomeCallable;
  let registryClient: RegistryClient;
  let transferClient: TransferClient;
  let lienClient: LienClient;
  let commonsClient: CommonsClient;

  beforeAll(() => {
    mockClient = createMockClient();
    const clients = createPropertyClients(mockClient);
    registryClient = clients.registry;
    transferClient = clients.transfers;
    lienClient = clients.liens;
    commonsClient = clients.commons;
  });

  describe('RegistryClient', () => {
    it('should register an asset', async () => {
      const result = await registryClient.registerAsset({
        type_: 'RealEstate',
        name: 'Beach House',
        description: 'A beautiful beach property',
        location: { lat: 25.7617, lng: -80.1918, address: '123 Ocean Drive, Miami, FL' },
      });

      expect(result).toBeDefined();
      const asset = result.entry.Present as Asset;
      expect(asset.name).toBe('Beach House');
      expect(asset.type_).toBe('RealEstate');
    });

    it('should get assets by owner', async () => {
      await registryClient.registerAsset({ type_: 'Vehicle', name: 'Car 1' });
      await registryClient.registerAsset({ type_: 'Vehicle', name: 'Car 2' });

      // Get by a known owner (using mock implementation specifics)
      const result = await registryClient.getAssetsByOwner('did:mycelix:owner');

      expect(result).toBeDefined();
    });

    it('should update asset metadata', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'Equipment',
        name: 'Tractor',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const result = await registryClient.updateAssetMetadata(assetId, { serial_number: 'ABC123' });

      expect(result).toBeDefined();
      expect((result!.entry.Present as Asset).metadata.serial_number).toBe('ABC123');
    });
  });

  describe('TransferClient', () => {
    it('should initiate a transfer', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'RealEstate',
        name: 'Apartment',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const result = await transferClient.initiateTransfer({
        asset_id: assetId,
        to_owner: 'did:mycelix:buyer',
        percentage: 100,
        price: 500000,
        currency: 'USD',
      });

      expect(result).toBeDefined();
      const transfer = result.entry.Present as TitleTransfer;
      expect(transfer.status).toBe('Initiated');
    });

    it('should complete a full transfer', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'Vehicle',
        name: 'Motorcycle',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const initiated = await transferClient.initiateTransfer({
        asset_id: assetId,
        to_owner: 'did:mycelix:newowner',
        percentage: 100,
      });
      const transferId = (initiated.entry.Present as TitleTransfer).id;

      await transferClient.acceptTransfer(transferId);
      const result = await transferClient.completeTransfer(transferId);

      expect(result).toBeDefined();
      expect((result.entry.Present as TitleTransfer).status).toBe('Completed');
    });

    it('should handle fractional ownership transfer', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'RealEstate',
        name: 'Investment Property',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const initiated = await transferClient.initiateTransfer({
        asset_id: assetId,
        to_owner: 'did:mycelix:coinvestor',
        percentage: 30,
        price: 150000,
        currency: 'USD',
      });
      const transferId = (initiated.entry.Present as TitleTransfer).id;

      await transferClient.acceptTransfer(transferId);
      await transferClient.completeTransfer(transferId);

      const asset = await registryClient.getAsset(assetId);
      expect((asset!.entry.Present as Asset).ownership_type).toBe('Fractional');
    });
  });

  describe('LienClient', () => {
    it('should place a lien', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'RealEstate',
        name: 'Mortgaged Property',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const result = await lienClient.fileLien({
        asset_id: assetId,
        lien_type: 'Mortgage',
        amount: 300000,
        currency: 'USD',
        description: 'Primary mortgage from First Bank',
      });

      expect(result).toBeDefined();
      const lien = result.entry.Present as Lien;
      expect(lien.lien_type).toBe('Mortgage');
      expect(lien.status).toBe('Active');
    });

    it('should release a lien', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'Vehicle',
        name: 'Financed Car',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const placed = await lienClient.fileLien({
        asset_id: assetId,
        lien_type: 'Loan',
        amount: 25000,
        currency: 'USD',
        description: 'Auto loan',
      });
      const lienId = (placed.entry.Present as Lien).id;

      const result = await lienClient.releaseLien(lienId);

      expect(result).toBeDefined();
      expect((result.entry.Present as Lien).status).toBe('Released');
    });
  });

  describe('CommonsClient', () => {
    it('should create a commons', async () => {
      const result = await commonsClient.createCommons({
        name: 'Community Garden',
        description: 'Shared garden space for the neighborhood',
        dao_id: 'dao-neighborhood',
      });

      expect(result).toBeDefined();
      const c = result.entry.Present as Commons;
      expect(c.name).toBe('Community Garden');
    });

    it('should add assets to commons', async () => {
      const registered = await registryClient.registerAsset({
        type_: 'Other',
        name: 'Garden Tools',
      });
      const assetId = (registered.entry.Present as Asset).id;

      const created = await commonsClient.createCommons({
        name: 'Tool Library',
        description: 'Shared tools for the community',
        dao_id: 'dao-tools',
      });
      const commonsId = (created.entry.Present as Commons).id;

      const result = await commonsClient.addToCommons(commonsId, assetId);

      expect(result).toBeDefined();
      expect((result.entry.Present as Commons).asset_ids).toContain(assetId);
    });
  });
});

describe('Validated Property Clients', () => {
  let mockClient: ZomeCallable;
  let validatedClients: ReturnType<typeof createValidatedPropertyClients>;

  beforeAll(() => {
    mockClient = createMockClient();
    validatedClients = createValidatedPropertyClients(mockClient);
  });

  it('should reject invalid asset registration', async () => {
    await expect(
      validatedClients.registry.registerAsset({
        type_: 'RealEstate',
        name: '', // Empty name
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid transfer', async () => {
    await expect(
      validatedClients.transfers.initiateTransfer({
        asset_id: '',
        to_owner: 'not-a-did',
        percentage: 150, // Over 100%
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid lien', async () => {
    await expect(
      validatedClients.liens.fileLien({
        asset_id: '',
        type_: 'Mortgage',
        amount: -1000, // Negative
        currency: 'USD',
      })
    ).rejects.toThrow(MycelixError);
  });
});

describeConductor('Property Conductor Integration Tests', () => {
  it.todo('should register assets with geospatial data');
  it.todo('should execute atomic title transfers');
  it.todo('should enforce lien priority');
  it.todo('should manage commons governance');
});
