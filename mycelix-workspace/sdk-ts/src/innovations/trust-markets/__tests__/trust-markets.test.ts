/**
 * Trust Market Protocol Tests
 *
 * Tests for prediction markets on trust claims, market mechanisms,
 * resolution, and calibration feedback integration.
 *
 * @module innovations/trust-markets/__tests__
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  TrustMarketService,
  type TrustMarket,
  type TrustClaim,
  type CreateTrustMarketInput,
  type SubmitTrustTradeInput,
  type TrustResolutionData,
} from '../index';

// ============================================================================
// MOCKS
// ============================================================================

const createMarketInput: CreateTrustMarketInput = {
  subjectDid: 'did:mycelix:alice123',
  claim: {
    type: 'reputation_maintenance',
    threshold: 0.7,
    contextHapps: ['governance', 'finance'],
    durationMs: 30 * 24 * 60 * 60 * 1000, // 30 days
    operator: 'gte',
  },
  mechanism: 'LMSR',
  liquidityParameter: 100,
  resolutionConfig: {
    resolutionType: 'automatic',
    dataSource: 'matl_reputation',
    minDataPoints: 10,
    oracleIds: [],
  },
  metadata: {
    description: 'Will Alice maintain reputation >= 0.7 for 30 days?',
    tags: ['reputation', 'governance'],
  },
};

// ============================================================================
// TESTS
// ============================================================================

describe('TrustMarketService', () => {
  let service: TrustMarketService;

  beforeEach(() => {
    service = new TrustMarketService();
  });

  describe('initialization', () => {
    it('should create a service instance', () => {
      expect(service).toBeInstanceOf(TrustMarketService);
    });

    it('should start with empty markets', () => {
      const markets = service.getActiveMarkets();
      expect(markets).toEqual([]);
    });

    it('should have zero stats initially', () => {
      const stats = service.getStats();
      expect(stats.totalMarkets).toBe(0);
      expect(stats.activeMarkets).toBe(0);
      expect(stats.resolvedMarkets).toBe(0);
      expect(stats.totalVolume).toBe(0);
    });
  });

  describe('createMarket', () => {
    it('should create a trust market', async () => {
      const market = await service.createMarket(createMarketInput);

      expect(market).toBeDefined();
      expect(market.id).toBeTruthy();
      expect(market.subjectDid).toBe('did:mycelix:alice123');
      expect(market.claim.type).toBe('reputation_maintenance');
      expect(market.claim.threshold).toBe(0.7);
      expect(market.mechanism.type).toBe('LMSR');
    });

    it('should track created market in active markets', async () => {
      const market = await service.createMarket(createMarketInput);
      const activeMarkets = service.getActiveMarkets();

      expect(activeMarkets.length).toBe(1);
      expect(activeMarkets[0].id).toBe(market.id);
    });

    it('should update stats after market creation', async () => {
      await service.createMarket(createMarketInput);
      const stats = service.getStats();

      expect(stats.totalMarkets).toBe(1);
      expect(stats.activeMarkets).toBe(1);
    });

    it('should support different claim types', async () => {
      const improvementInput: CreateTrustMarketInput = {
        ...createMarketInput,
        claim: {
          type: 'reputation_improvement',
          threshold: 0.1, // 10% improvement
          contextHapps: ['governance'],
          durationMs: 7 * 24 * 60 * 60 * 1000,
          operator: 'gte',
        },
      };

      const market = await service.createMarket(improvementInput);
      expect(market.claim.type).toBe('reputation_improvement');
    });

    it('should support different market mechanisms', async () => {
      const cdaInput: CreateTrustMarketInput = {
        ...createMarketInput,
        mechanism: 'CDA',
      };

      const market = await service.createMarket(cdaInput);
      expect(market.mechanism.type).toBe('CDA');
    });
  });

  describe('getMarket', () => {
    it('should retrieve existing market', async () => {
      const created = await service.createMarket(createMarketInput);
      const retrieved = service.getMarket(created.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(created.id);
    });

    it('should return undefined for non-existent market', () => {
      const market = service.getMarket('non-existent-id');
      expect(market).toBeUndefined();
    });
  });

  describe('submitTrade', () => {
    let market: TrustMarket;

    beforeEach(async () => {
      market = await service.createMarket(createMarketInput);
    });

    it('should accept a trade on existing market', async () => {
      const tradeInput: SubmitTrustTradeInput = {
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: {
          monetary: 50,
          reputation: 0.1,
          commitment: [],
        },
      };

      const result = await service.submitTrade(tradeInput);

      expect(result.success).toBe(true);
      expect(result.tradeId).toBeTruthy();
      expect(result.newShares).toBe(10);
    });

    it('should update market state after trade', async () => {
      const tradeInput: SubmitTrustTradeInput = {
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: {
          monetary: 50,
          reputation: 0.1,
          commitment: [],
        },
      };

      await service.submitTrade(tradeInput);
      const updatedMarket = service.getMarket(market.id);

      expect(updatedMarket?.state.totalVolume).toBeGreaterThan(0);
      expect(updatedMarket?.state.uniqueTraders).toBe(1);
    });

    it('should reject trade on non-existent market', async () => {
      const tradeInput: SubmitTrustTradeInput = {
        marketId: 'non-existent',
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: {
          monetary: 50,
          reputation: 0.1,
          commitment: [],
        },
      };

      await expect(service.submitTrade(tradeInput)).rejects.toThrow();
    });

    it('should track multiple traders', async () => {
      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: { monetary: 50, reputation: 0.1, commitment: [] },
      });

      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:carol789',
        outcome: 'no',
        shares: 5,
        stake: { monetary: 25, reputation: 0.05, commitment: [] },
      });

      const updatedMarket = service.getMarket(market.id);
      expect(updatedMarket?.state.uniqueTraders).toBe(2);
    });
  });

  describe('getPosition', () => {
    let market: TrustMarket;

    beforeEach(async () => {
      market = await service.createMarket(createMarketInput);
    });

    it('should return position after trading', async () => {
      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: { monetary: 50, reputation: 0.1, commitment: [] },
      });

      const position = service.getPosition(market.id, 'did:mycelix:bob456');

      expect(position).toBeDefined();
      expect(position?.shares.get('yes')).toBe(10);
    });

    it('should return undefined for non-participant', () => {
      const position = service.getPosition(market.id, 'did:mycelix:nonparticipant');
      expect(position).toBeUndefined();
    });
  });

  describe('resolveMarket', () => {
    let market: TrustMarket;

    beforeEach(async () => {
      market = await service.createMarket(createMarketInput);

      // Add some trades
      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 10,
        stake: { monetary: 50, reputation: 0.1, commitment: [] },
      });

      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:carol789',
        outcome: 'no',
        shares: 5,
        stake: { monetary: 25, reputation: 0.05, commitment: [] },
      });
    });

    it('should resolve market with outcome', async () => {
      const resolutionData: TrustResolutionData = {
        actualReputation: 0.75,
        dataPoints: 15,
        sources: ['matl_aggregation'],
        timestamp: Date.now(),
      };

      const resolution = await service.resolveMarket(market.id, 'yes', resolutionData);

      expect(resolution).toBeDefined();
      expect(resolution.outcome).toBe('yes');
      expect(resolution.payouts.size).toBeGreaterThan(0);
    });

    it('should update market to resolved state', async () => {
      const resolutionData: TrustResolutionData = {
        actualReputation: 0.75,
        dataPoints: 15,
        sources: ['matl_aggregation'],
        timestamp: Date.now(),
      };

      await service.resolveMarket(market.id, 'yes', resolutionData);
      const resolvedMarket = service.getMarket(market.id);

      expect(resolvedMarket?.resolvedAt).toBeDefined();
    });

    it('should remove from active markets after resolution', async () => {
      const resolutionData: TrustResolutionData = {
        actualReputation: 0.75,
        dataPoints: 15,
        sources: ['matl_aggregation'],
        timestamp: Date.now(),
      };

      await service.resolveMarket(market.id, 'yes', resolutionData);
      const activeMarkets = service.getActiveMarkets();

      expect(activeMarkets.find((m) => m.id === market.id)).toBeUndefined();
    });

    it('should update stats after resolution', async () => {
      const resolutionData: TrustResolutionData = {
        actualReputation: 0.75,
        dataPoints: 15,
        sources: ['matl_aggregation'],
        timestamp: Date.now(),
      };

      await service.resolveMarket(market.id, 'yes', resolutionData);
      const stats = service.getStats();

      expect(stats.resolvedMarkets).toBe(1);
    });
  });

  describe('getMarketsForSubject', () => {
    it('should return markets for specific subject', async () => {
      await service.createMarket(createMarketInput);
      await service.createMarket({
        ...createMarketInput,
        subjectDid: 'did:mycelix:different',
      });

      const markets = service.getMarketsForSubject('did:mycelix:alice123');

      expect(markets.length).toBe(1);
      expect(markets[0].subjectDid).toBe('did:mycelix:alice123');
    });

    it('should return empty array for subject without markets', () => {
      const markets = service.getMarketsForSubject('did:mycelix:nobody');
      expect(markets).toEqual([]);
    });
  });

  describe('market pricing', () => {
    let market: TrustMarket;

    beforeEach(async () => {
      market = await service.createMarket(createMarketInput);
    });

    it('should have initial prices', () => {
      expect(market.state.impliedProbabilities.get('yes')).toBeDefined();
      expect(market.state.impliedProbabilities.get('no')).toBeDefined();
    });

    it('should update prices after trades', async () => {
      const initialYesPrice = market.state.impliedProbabilities.get('yes');

      await service.submitTrade({
        marketId: market.id,
        traderId: 'did:mycelix:bob456',
        outcome: 'yes',
        shares: 100,
        stake: { monetary: 500, reputation: 0.5, commitment: [] },
      });

      const updatedMarket = service.getMarket(market.id);
      const newYesPrice = updatedMarket?.state.impliedProbabilities.get('yes');

      // Buying 'yes' shares should increase the 'yes' price
      expect(newYesPrice).toBeGreaterThan(initialYesPrice!);
    });
  });
});

describe('Trust Claim Types', () => {
  let service: TrustMarketService;

  beforeEach(() => {
    service = new TrustMarketService();
  });

  it('should support reputation_maintenance claims', async () => {
    const market = await service.createMarket({
      ...createMarketInput,
      claim: {
        type: 'reputation_maintenance',
        threshold: 0.7,
        contextHapps: ['governance'],
        durationMs: 86400000,
        operator: 'gte',
      },
    });
    expect(market.claim.type).toBe('reputation_maintenance');
  });

  it('should support reputation_improvement claims', async () => {
    const market = await service.createMarket({
      ...createMarketInput,
      claim: {
        type: 'reputation_improvement',
        threshold: 0.1,
        contextHapps: ['governance'],
        durationMs: 86400000,
        operator: 'gte',
      },
    });
    expect(market.claim.type).toBe('reputation_improvement');
  });

  it('should support byzantine_probability claims', async () => {
    const market = await service.createMarket({
      ...createMarketInput,
      claim: {
        type: 'byzantine_probability',
        threshold: 0.1,
        contextHapps: ['governance', 'finance'],
        durationMs: 86400000,
        operator: 'lt',
      },
    });
    expect(market.claim.type).toBe('byzantine_probability');
  });

  it('should support cross_happ_consistency claims', async () => {
    const market = await service.createMarket({
      ...createMarketInput,
      claim: {
        type: 'cross_happ_consistency',
        threshold: 0.05, // Max 5% deviation
        contextHapps: ['governance', 'finance', 'property'],
        durationMs: 86400000,
        operator: 'lte',
      },
    });
    expect(market.claim.type).toBe('cross_happ_consistency');
  });
});
