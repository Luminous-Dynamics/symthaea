// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Privacy-Preserving Cross-hApp Query Tests
 *
 * Tests for FHE-based private queries, aggregation, PSI,
 * threshold decryption, and differential privacy.
 *
 * @module innovations/private-queries/__tests__
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  PrivateQueryService,
  createPrivateQueryService,
  createStrictPrivateQueryService,
  createAnalyticsQueryService,
  type PrivateQuery,
  type SubmitAggregateQueryInput,
  type SubmitPSIQueryInput,
  type RequestThresholdDecryptionInput,
  type PrivateCorrelationInput,
} from '../index';

// ============================================================================
// TESTS
// ============================================================================

describe('PrivateQueryService', () => {
  let service: PrivateQueryService;

  beforeEach(() => {
    service = new PrivateQueryService();
  });

  describe('initialization', () => {
    it('should create a service instance', () => {
      expect(service).toBeInstanceOf(PrivateQueryService);
    });

    it('should not be initialized before calling initialize()', () => {
      expect(service.isInitialized()).toBe(false);
    });

    it('should initialize FHE infrastructure', async () => {
      await service.initialize();
      expect(service.isInitialized()).toBe(true);
    });

    it('should be idempotent for initialization', async () => {
      await service.initialize();
      await service.initialize(); // Should not throw
      expect(service.isInitialized()).toBe(true);
    });

    it('should accept custom privacy budget', () => {
      const customService = new PrivateQueryService({
        epsilon: 0.5,
        delta: 1e-8,
        accountingMethod: 'rdp',
      });
      expect(customService).toBeInstanceOf(PrivateQueryService);
    });
  });

  describe('submitAggregateQuery', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should submit an aggregate query', async () => {
      const input: SubmitAggregateQueryInput = {
        type: 'count',
        targetHapps: ['energy', 'health'],
        publicFilters: {
          dateRange: { start: Date.now() - 86400000, end: Date.now() },
        },
        privacyBudget: {
          epsilon: 0.1,
          delta: 1e-7,
          accountingMethod: 'rdp',
        },
      };

      const query = await service.submitAggregateQuery(input);

      expect(query).toBeDefined();
      expect(query.id).toBeTruthy();
      expect(query.type).toBe('count');
      expect(query.targetHapps).toEqual(['energy', 'health']);
      expect(query.state.status).toBe('pending');
    });

    it('should support different aggregation types', async () => {
      const sumQuery = await service.submitAggregateQuery({
        type: 'sum',
        targetHapps: ['finance'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'simple' },
      });
      expect(sumQuery.type).toBe('sum');

      const avgQuery = await service.submitAggregateQuery({
        type: 'average',
        targetHapps: ['finance'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'simple' },
      });
      expect(avgQuery.type).toBe('average');
    });

    it('should reject query exceeding privacy budget', async () => {
      // Submit a query consuming most of the budget
      await service.submitAggregateQuery({
        type: 'count',
        targetHapps: ['energy'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.9, delta: 1e-7, accountingMethod: 'rdp' },
      });

      // Try to submit another query exceeding remaining budget
      await expect(
        service.submitAggregateQuery({
          type: 'count',
          targetHapps: ['health'],
          publicFilters: {},
          privacyBudget: { epsilon: 0.2, delta: 1e-7, accountingMethod: 'rdp' },
        })
      ).rejects.toThrow(/privacy budget/i);
    });
  });

  describe('executeQuery', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should execute a submitted query', async () => {
      const query = await service.submitAggregateQuery({
        type: 'count',
        targetHapps: ['energy', 'health'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'rdp' },
      });

      const result = await service.executeQuery(query.id);

      expect(result).toBeDefined();
      expect(result.queryId).toBe(query.id);
      expect(result.queryType).toBe('count');
      expect(typeof result.value).toBe('object'); // number[]
      expect(result.confidenceInterval).toBeDefined();
    });

    it('should update query state after execution', async () => {
      const query = await service.submitAggregateQuery({
        type: 'count',
        targetHapps: ['energy'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'rdp' },
      });

      await service.executeQuery(query.id);
      const updatedQuery = service.getQuery(query.id);

      expect(updatedQuery?.state.status).toBe('completed');
      expect(updatedQuery?.state.decryptedResult).toBeDefined();
    });

    it('should reject execution of non-existent query', async () => {
      await expect(service.executeQuery('non-existent-id')).rejects.toThrow();
    });
  });

  describe('submitPSIQuery', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should submit a PSI query', async () => {
      const input: SubmitPSIQueryInput = {
        queryElement: 'user-identifier-hash',
        targetSet: 'energy-high-consumers',
        targetHapp: 'energy',
      };

      const query = await service.submitPSIQuery(input);

      expect(query).toBeDefined();
      expect(query.id).toBeTruthy();
      expect(query.queryElement).toBe('user-identifier-hash');
      expect(query.targetSet).toBe('energy-high-consumers');
    });

    it('should execute PSI query', async () => {
      const query = await service.submitPSIQuery({
        queryElement: 'user-hash-123',
        targetSet: 'high-consumers',
        targetHapp: 'energy',
      });

      const result = await service.executePSIQuery(query.id);

      expect(result).toBeDefined();
      expect(typeof result.isMember).toBe('boolean');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.privacyPreserved).toBe(true);
    });
  });

  describe('requestThresholdDecryption', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should create threshold decryption request', async () => {
      const input: RequestThresholdDecryptionInput = {
        encryptedDataId: 'encrypted-aggregate-result-123',
        threshold: 3,
        guardians: ['guardian-1', 'guardian-2', 'guardian-3', 'guardian-4', 'guardian-5'],
        justification: 'Regulatory audit requires access to aggregate statistics',
      };

      const request = await service.requestThresholdDecryption(input);

      expect(request).toBeDefined();
      expect(request.id).toBeTruthy();
      expect(request.threshold).toBe(3);
      expect(request.guardians.length).toBe(5);
      expect(request.status).toBe('pending');
    });

    it('should track guardian approvals', async () => {
      const request = await service.requestThresholdDecryption({
        encryptedDataId: 'encrypted-data-456',
        threshold: 2,
        guardians: ['guardian-a', 'guardian-b', 'guardian-c'],
        justification: 'Test decryption',
      });

      await service.submitGuardianApproval(request.id, 'guardian-a', true);

      const updatedRequest = service.getDecryptionRequest(request.id);
      expect(updatedRequest?.approvals.length).toBe(1);
      expect(updatedRequest?.approvals[0].guardianId).toBe('guardian-a');
      expect(updatedRequest?.approvals[0].approved).toBe(true);
    });

    it('should decrypt when threshold is met', async () => {
      const request = await service.requestThresholdDecryption({
        encryptedDataId: 'encrypted-data-789',
        threshold: 2,
        guardians: ['guardian-x', 'guardian-y', 'guardian-z'],
        justification: 'Test threshold decryption',
      });

      await service.submitGuardianApproval(request.id, 'guardian-x', true);
      await service.submitGuardianApproval(request.id, 'guardian-y', true);

      const updatedRequest = service.getDecryptionRequest(request.id);
      expect(updatedRequest?.status).toBe('decrypted');
    });

    it('should not decrypt if threshold not met', async () => {
      const request = await service.requestThresholdDecryption({
        encryptedDataId: 'encrypted-data-abc',
        threshold: 3,
        guardians: ['g1', 'g2', 'g3', 'g4'],
        justification: 'Test insufficient approvals',
      });

      await service.submitGuardianApproval(request.id, 'g1', true);
      await service.submitGuardianApproval(request.id, 'g2', false); // Rejected

      const updatedRequest = service.getDecryptionRequest(request.id);
      expect(updatedRequest?.status).toBe('pending');
    });
  });

  describe('computePrivateCorrelation', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should compute correlation without revealing data', async () => {
      const input: PrivateCorrelationInput = {
        happA: 'energy',
        metricA: 'consumption',
        happB: 'health',
        metricB: 'wellness_score',
        privacyBudget: {
          epsilon: 0.2,
          delta: 1e-7,
          accountingMethod: 'rdp',
        },
      };

      const result = await service.computePrivateCorrelation(input);

      expect(result).toBeDefined();
      expect(typeof result.correlation).toBe('number');
      expect(result.correlation).toBeGreaterThanOrEqual(-1);
      expect(result.correlation).toBeLessThanOrEqual(1);
      expect(result.confidenceInterval).toBeDefined();
      expect(result.interpretation).toBeTruthy();
    });

    it('should consume privacy budget', async () => {
      const statsBefore = service.getPrivacyStats();

      await service.computePrivateCorrelation({
        happA: 'energy',
        metricA: 'usage',
        happB: 'finance',
        metricB: 'spending',
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'rdp' },
      });

      const statsAfter = service.getPrivacyStats();
      expect(statsAfter.consumedEpsilon).toBeGreaterThan(statsBefore.consumedEpsilon);
    });
  });

  describe('getPrivacyStats', () => {
    beforeEach(async () => {
      await service.initialize();
    });

    it('should return privacy statistics', () => {
      const stats = service.getPrivacyStats();

      expect(stats).toBeDefined();
      expect(stats.totalBudget).toBeDefined();
      expect(stats.consumedEpsilon).toBeDefined();
      expect(stats.remainingBudget).toBeDefined();
      expect(stats.queriesExecuted).toBeDefined();
    });

    it('should track consumed epsilon', async () => {
      await service.submitAggregateQuery({
        type: 'count',
        targetHapps: ['energy'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.1, delta: 1e-7, accountingMethod: 'rdp' },
      });

      const stats = service.getPrivacyStats();
      expect(stats.consumedEpsilon).toBeGreaterThan(0);
    });
  });
});

describe('Factory Functions', () => {
  describe('createPrivateQueryService', () => {
    it('should create a basic service', () => {
      const service = createPrivateQueryService();
      expect(service).toBeInstanceOf(PrivateQueryService);
    });

    it('should accept custom privacy budget', () => {
      const service = createPrivateQueryService({
        epsilon: 2.0,
        delta: 1e-6,
        accountingMethod: 'simple',
      });
      expect(service).toBeInstanceOf(PrivateQueryService);
    });
  });

  describe('createStrictPrivateQueryService', () => {
    it('should create service with strict privacy settings', () => {
      const service = createStrictPrivateQueryService();
      expect(service).toBeInstanceOf(PrivateQueryService);
    });
  });

  describe('createAnalyticsQueryService', () => {
    it('should create service for analytics use cases', () => {
      const service = createAnalyticsQueryService();
      expect(service).toBeInstanceOf(PrivateQueryService);
    });
  });
});

describe('Privacy Budget Management', () => {
  it('should enforce total budget across operations', async () => {
    const service = new PrivateQueryService({
      epsilon: 0.5, // Small budget
      delta: 1e-7,
      accountingMethod: 'rdp',
    });
    await service.initialize();

    // First query consumes 0.3
    await service.submitAggregateQuery({
      type: 'count',
      targetHapps: ['energy'],
      publicFilters: {},
      privacyBudget: { epsilon: 0.3, delta: 1e-7, accountingMethod: 'rdp' },
    });

    // Second query should fail (0.3 + 0.3 > 0.5)
    await expect(
      service.submitAggregateQuery({
        type: 'count',
        targetHapps: ['health'],
        publicFilters: {},
        privacyBudget: { epsilon: 0.3, delta: 1e-7, accountingMethod: 'rdp' },
      })
    ).rejects.toThrow(/privacy budget/i);
  });
});
