/**
 * Services Tests
 *
 * Tests for gateway services including Holochain connection,
 * metrics, reputation, and WebSocket handling.
 */

import { describe, expect, it, beforeEach, vi } from 'vitest';
import {
  MockHolochainService,
  HolochainService,
} from '../src/services/holochain';

describe('HolochainService', () => {
  describe('MockHolochainService', () => {
    let service: MockHolochainService;

    beforeEach(async () => {
      service = new MockHolochainService();
      await service.connect();
    });

    it('connects successfully', () => {
      expect(service.isConnected()).toBe(true);
    });

    it('returns available hApps', () => {
      const happs = service.getAvailableHapps();

      expect(happs).toContain('identity');
      expect(happs).toContain('governance');
      expect(happs).toContain('health');
      expect(happs).toContain('climate');
      expect(happs).toContain('mutualaid');
      expect(happs).toContain('journalism');
    });

    describe('Identity hApp calls', () => {
      it('returns mock profile data', async () => {
        const profile = await service.callZome('identity', {
          zomeName: 'identity',
          fnName: 'get_profile',
          payload: null,
        });

        expect(profile).toMatchObject({
          did: 'did:mycelix:mock123',
          displayName: 'Test User',
        });
      });

      it('returns true for signature verification', async () => {
        const result = await service.callZome('identity', {
          zomeName: 'identity',
          fnName: 'verify_signature',
          payload: { did: 'did:mycelix:mock123', signature: 'sig' },
        });

        expect(result).toBe(true);
      });
    });

    describe('Governance hApp calls', () => {
      it('returns mock proposals', async () => {
        const proposals = await service.callZome('governance', {
          zomeName: 'proposals',
          fnName: 'get_proposals',
          payload: null,
        });

        expect(Array.isArray(proposals)).toBe(true);
        expect(proposals[0]).toMatchObject({
          id: 'prop-001',
          title: 'Community Fund Allocation',
        });
      });
    });

    describe('Health hApp calls', () => {
      it('returns mock health records', async () => {
        const records = await service.callZome('health', {
          zomeName: 'records',
          fnName: 'get_my_records',
          payload: null,
        });

        expect(Array.isArray(records)).toBe(true);
        expect(records[0]).toMatchObject({
          recordType: 'Observation',
          verified: true,
        });
      });
    });

    describe('Climate hApp calls', () => {
      it('returns mock carbon credits', async () => {
        const credits = await service.callZome('climate', {
          zomeName: 'credits',
          fnName: 'get_credits',
          payload: null,
        });

        expect(Array.isArray(credits)).toBe(true);
        expect(credits[0]).toMatchObject({
          vintage: 2024,
          verificationStandard: 'verra',
        });
      });
    });

    describe('MutualAid hApp calls', () => {
      it('returns mock aid requests', async () => {
        const requests = await service.callZome('mutualaid', {
          zomeName: 'requests',
          fnName: 'get_requests',
          payload: null,
        });

        expect(Array.isArray(requests)).toBe(true);
        expect(requests[0]).toMatchObject({
          requestType: 'Food assistance',
          status: 'Open',
        });
      });

      it('returns mock aid pools', async () => {
        const pools = await service.callZome('mutualaid', {
          zomeName: 'pools',
          fnName: 'get_pools',
          payload: null,
        });

        expect(Array.isArray(pools)).toBe(true);
        expect(pools[0]).toMatchObject({
          name: 'Richardson Community Fund',
          poolType: 'community',
        });
      });
    });

    describe('Journalism hApp calls', () => {
      it('returns mock articles with ENM classification', async () => {
        const articles = await service.callZome('journalism', {
          zomeName: 'articles',
          fnName: 'get_articles',
          payload: null,
        });

        expect(Array.isArray(articles)).toBe(true);
        expect(articles[0]).toMatchObject({
          title: 'Local Community Makes Progress',
          enmClassification: {
            empirical: 'Corroborated',
            normative: 'Factual',
            materiality: 'Local',
          },
        });
      });
    });

    describe('Unknown hApp/function handling', () => {
      it('returns empty object for unknown hApp', async () => {
        const result = await service.callZome('nonexistent', {
          zomeName: 'any',
          fnName: 'any',
          payload: null,
        });

        expect(result).toEqual({});
      });

      it('returns empty object for unknown function', async () => {
        const result = await service.callZome('identity', {
          zomeName: 'identity',
          fnName: 'nonexistent_function',
          payload: null,
        });

        expect(result).toEqual({});
      });
    });
  });

  describe('HolochainService (real)', () => {
    it('throws when calling without connection', async () => {
      const service = new HolochainService({
        adminUrl: 'ws://localhost:9000',
        appUrl: 'ws://localhost:9001',
        installedAppId: 'mycelix',
      });

      await expect(
        service.callZome('identity', {
          zomeName: 'identity',
          fnName: 'get_profile',
          payload: null,
        })
      ).rejects.toThrow('Not connected to Holochain conductor');
    });

    it('reports disconnected state before connect', () => {
      const service = new HolochainService({
        adminUrl: 'ws://localhost:9000',
        appUrl: 'ws://localhost:9001',
        installedAppId: 'mycelix',
      });

      expect(service.isConnected()).toBe(false);
    });

    it('returns empty hApps list before connect', () => {
      const service = new HolochainService({
        adminUrl: 'ws://localhost:9000',
        appUrl: 'ws://localhost:9001',
        installedAppId: 'mycelix',
      });

      expect(service.getAvailableHapps()).toEqual([]);
    });
  });
});
