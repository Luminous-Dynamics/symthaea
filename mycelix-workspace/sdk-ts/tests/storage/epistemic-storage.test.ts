// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Epistemic Storage Tests
 *
 * Tests for the main EpistemicStorage implementation.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  EpistemicStorageImpl,
  createEpistemicStorage,
  StorageError,
  ImmutableError,
  AccessDeniedError,
  InvariantViolationError,
} from '../../src/storage/epistemic-storage.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../src/epistemic/types.js';
import type { EpistemicClassification } from '../../src/epistemic/types.js';
import type { AccessCapability, SchemaIdentity } from '../../src/storage/types.js';

describe('EpistemicStorage', () => {
  let storage: EpistemicStorageImpl;
  const testAgentId = 'agent:test-user';

  beforeEach(() => {
    storage = createEpistemicStorage({ agentId: testAgentId });
  });

  afterEach(() => {
    storage.dispose();
  });

  // Helper to create test schema
  const testSchema: SchemaIdentity = { id: 'test', version: '1.0.0' };

  // Helper to create owner capability
  function createOwnerCapability(key: string): AccessCapability {
    return {
      id: 'cap-' + Date.now(),
      resourceKeys: [key],
      permissions: ['read', 'write', 'delete', 'delegate', 'shred'],
      issuedBy: testAgentId,
      issuedTo: testAgentId,
      issuedAt: Date.now(),
      delegable: true,
      signature: 'mock-signature',
    };
  }

  describe('store()', () => {
    it('stores ephemeral data (M0)', async () => {
      const data = { session: 'abc123' };
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      };

      const receipt = await storage.store('session:123', data, classification, {
        schema: testSchema,
      });

      expect(receipt.key).toBe('session:123');
      expect(receipt.cid).toMatch(/^cid:/);
      expect(receipt.tier.backend).toBe('memory');
      expect(receipt.version).toBe(1);
    });

    it('stores temporal data (M1)', async () => {
      const data = { preference: 'dark-mode' };
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      const receipt = await storage.store('pref:theme', data, classification, {
        schema: testSchema,
      });

      expect(receipt.tier.backend).toBe('local');
      expect(receipt.shreddable).toBe(true); // N0 is encrypted
    });

    it('stores persistent data (M2)', async () => {
      const data = { profile: { name: 'Alice' } };
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const receipt = await storage.store('profile:alice', data, classification, {
        schema: testSchema,
      });

      expect(receipt.tier.backend).toBe('dht');
      expect(receipt.tier.contentAddressed).toBe(true);
      expect(receipt.shreddable).toBe(false); // N2 is public
    });

    it('requires schema identity', async () => {
      const data = { test: true };
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      };

      await expect(
        storage.store('test:key', data, classification, {} as any)
      ).rejects.toThrow();
    });

    it('rejects duplicate keys', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      };

      await storage.store('dup:key', { a: 1 }, classification, { schema: testSchema });

      await expect(
        storage.store('dup:key', { b: 2 }, classification, { schema: testSchema })
      ).rejects.toThrow(StorageError);
    });

    it('validates classification ranges', async () => {
      const invalidClassification = {
        empirical: 99 as EmpiricalLevel,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      };

      await expect(
        storage.store('test:key', { test: true }, invalidClassification, { schema: testSchema })
      ).rejects.toThrow();
    });
  });

  describe('retrieve()', () => {
    it('retrieves stored data', async () => {
      const data = { name: 'Alice', score: 100 };
      // Use N2 (public) for basic retrieval test to avoid capability requirement
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('game:alice', data, classification, { schema: testSchema });

      const result = await storage.retrieve('game:alice');

      expect(result).not.toBeNull();
      expect(result!.data).toEqual(data);
      expect(result!.metadata.classification).toEqual(classification);
    });

    it('returns null for non-existent keys', async () => {
      const result = await storage.retrieve('nonexistent');

      expect(result).toBeNull();
    });

    it('requires capability for N0 data', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('private:data', { secret: true }, classification, {
        schema: testSchema,
      });

      // Should throw without capability
      await expect(storage.retrieve('private:data')).rejects.toThrow(AccessDeniedError);

      // Should succeed with capability
      const cap = createOwnerCapability('private:data');
      const result = await storage.retrieve('private:data', cap);

      expect(result).not.toBeNull();
    });

    it('allows public access to N2 data', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('public:data', { public: true }, classification, {
        schema: testSchema,
      });

      // Should succeed without capability
      const result = await storage.retrieve('public:data');

      expect(result).not.toBeNull();
      expect(result!.data).toEqual({ public: true });
    });
  });

  describe('retrieveByCID()', () => {
    it('retrieves data by content identifier', async () => {
      const data = { content: 'addressed' };
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const receipt = await storage.store('cid:test', data, classification, {
        schema: testSchema,
      });

      const result = await storage.retrieveByCID(receipt.cid);

      expect(result).not.toBeNull();
      expect(result!.data).toEqual(data);
    });

    it('returns null for unknown CID', async () => {
      const result = await storage.retrieveByCID('cid:unknown');

      expect(result).toBeNull();
    });
  });

  describe('update()', () => {
    it('updates mutable data (E0-E1)', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network, // Public for easy access
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('mutable:data', { count: 1 }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('mutable:data');
      const receipt = await storage.update('mutable:data', { count: 2 }, cap);

      expect(receipt.version).toBe(2);

      const result = await storage.retrieve('mutable:data');
      expect(result!.data).toEqual({ count: 2 });
    });

    it('creates new version for append-only data (E2)', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const firstReceipt = await storage.store('append:data', { v: 1 }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('append:data');
      const secondReceipt = await storage.update('append:data', { v: 2 }, cap);

      expect(secondReceipt.version).toBe(2);
      expect(secondReceipt.previousCid).toBe(firstReceipt.cid);
    });

    it('throws ImmutableError for E3+ data', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('immutable:data', { fixed: true }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('immutable:data');

      await expect(
        storage.update('immutable:data', { fixed: false }, cap)
      ).rejects.toThrow(ImmutableError);
    });

    it('throws for non-existent key', async () => {
      const cap = createOwnerCapability('nonexistent');

      await expect(
        storage.update('nonexistent', { data: true }, cap)
      ).rejects.toThrow(StorageError);
    });
  });

  describe('delete()', () => {
    it('deletes mutable data', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('deletable:data', { temp: true }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('deletable:data');
      await storage.delete('deletable:data', cap);

      const result = await storage.retrieve('deletable:data');
      expect(result).toBeNull();
    });

    it('throws ImmutableError for E3+ data (without shred)', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('immutable:delete', { fixed: true }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('immutable:delete');

      await expect(storage.delete('immutable:delete', cap)).rejects.toThrow(ImmutableError);
    });
  });

  describe('reclassify() - INV-2', () => {
    it('allows upgrading classification', async () => {
      const initialClassification: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('reclassify:test', { data: true }, initialClassification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('reclassify:test');
      const reason = {
        reason: 'Upgrade to cryptographic verification',
        requestedBy: testAgentId,
        timestamp: Date.now(),
      };

      const receipt = await storage.reclassify(
        'reclassify:test',
        { empirical: EmpiricalLevel.E3_Cryptographic },
        reason,
        cap
      );

      expect(receipt.classification.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
      expect(receipt.classification.normative).toBe(NormativeLevel.N1_Communal);
    });

    it('rejects downgrading classification (INV-2)', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('downgrade:test', { data: true }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('downgrade:test');
      const reason = {
        reason: 'Attempted downgrade',
        requestedBy: testAgentId,
        timestamp: Date.now(),
      };

      await expect(
        storage.reclassify(
          'downgrade:test',
          { empirical: EmpiricalLevel.E1_Testimonial },
          reason,
          cap
        )
      ).rejects.toThrow(InvariantViolationError);
    });

    it('triggers migration when backend changes', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      await storage.store('migrate:test', { data: true }, classification, {
        schema: testSchema,
      });

      const cap = createOwnerCapability('migrate:test');
      const reason = {
        reason: 'Upgrade to persistent',
        requestedBy: testAgentId,
        timestamp: Date.now(),
      };

      const receipt = await storage.reclassify(
        'migrate:test',
        { materiality: MaterialityLevel.M2_Persistent },
        reason,
        cap
      );

      expect(receipt.tier.backend).toBe('dht');
      expect(receipt.version).toBe(2); // Migration increments version
    });
  });

  describe('getStorageInfo()', () => {
    it('returns storage info for existing key', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('info:test', { data: true }, classification, {
        schema: testSchema,
      });

      const info = await storage.getStorageInfo('info:test');

      expect(info).not.toBeNull();
      expect(info!.exists).toBe(true);
      expect(info!.classification).toEqual(classification);
      expect(info!.version).toBe(1);
    });

    it('returns exists=false for non-existent key', async () => {
      const info = await storage.getStorageInfo('nonexistent');

      expect(info).toEqual({ exists: false });
    });
  });

  describe('query()', () => {
    beforeEach(async () => {
      // Set up test data
      await storage.store('query:e0', { level: 'e0' }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema: testSchema });

      await storage.store('query:e2', { level: 'e2' }, {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema: testSchema });

      await storage.store('query:e3', { level: 'e3' }, {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema: testSchema });
    });

    it('filters by E-level range', async () => {
      const result = await storage.query({
        classification: {
          minEmpirical: EmpiricalLevel.E2_PrivateVerify,
        },
      });

      expect(result.items).toHaveLength(2);
      expect(result.items.map(i => (i.data as any).level)).toContain('e2');
      expect(result.items.map(i => (i.data as any).level)).toContain('e3');
    });

    it('filters by schema', async () => {
      await storage.store('query:other', { level: 'other' }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema: { id: 'other', version: '2.0.0' } });

      const result = await storage.query({
        schema: { id: 'test' },
      });

      expect(result.items).toHaveLength(3); // Only 'test' schema items
    });

    it('paginates results', async () => {
      const result = await storage.query({
        limit: 2,
        offset: 0,
      });

      expect(result.items).toHaveLength(2);
      expect(result.hasMore).toBe(true);
      expect(result.nextOffset).toBe(2);
    });

    it('sorts by storedAt', async () => {
      const result = await storage.query({
        orderBy: 'storedAt',
        orderDirection: 'asc',
      });

      expect(result.items.length).toBeGreaterThan(0);
      // First item should be oldest
      const times = result.items.map(i => i.metadata.storedAt);
      expect(times).toEqual([...times].sort((a, b) => a - b));
    });
  });

  describe('verify()', () => {
    it('verifies data integrity', async () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      await storage.store('verify:test', { important: true }, classification, {
        schema: testSchema,
      });

      const result = await storage.verify('verify:test');

      expect(result.verified).toBe(true);
      expect(result.cidValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('reports verification failure for non-existent key', async () => {
      const result = await storage.verify('nonexistent');

      expect(result.verified).toBe(false);
      expect(result.errors).toContain('Key not found');
    });
  });

  describe('exists()', () => {
    it('returns true for existing key', async () => {
      await storage.store('exists:test', { data: true }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema: testSchema });

      expect(await storage.exists('exists:test')).toBe(true);
    });

    it('returns false for non-existent key', async () => {
      expect(await storage.exists('nonexistent')).toBe(false);
    });
  });

  describe('getStats()', () => {
    it('returns storage statistics', async () => {
      await storage.store('stats:1', { a: 1 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema: testSchema });

      await storage.store('stats:2', { b: 2 }, {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema: testSchema });

      const stats = await storage.getStats();

      expect(stats.totalItems).toBe(2);
      expect(stats.itemsByMateriality[0]).toBe(1); // M0
      expect(stats.itemsByMateriality[1]).toBe(1); // M1
      expect(stats.itemsByEmpirical[0]).toBe(1); // E0
      expect(stats.itemsByEmpirical[2]).toBe(1); // E2
    });
  });
});
