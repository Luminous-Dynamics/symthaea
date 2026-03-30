// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Zome Conductor Integration Tests
 *
 * Tests the bridge zome against a real Holochain conductor.
 * These tests require a running conductor with the Mycelix hApp installed.
 *
 * Run with: CONDUCTOR_AVAILABLE=true npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import {
  getConductorConfig,
  isConductorAvailable,
  setupTestContext,
  teardownTestContext,
  generateTestAgentId,
  waitForSync,
  type TestContext,
} from './conductor-harness.js';

// Skip all tests if conductor is not available
const conductorAvailable = process.env.CONDUCTOR_AVAILABLE === 'true';

describe.skipIf(!conductorAvailable)('Bridge Zome - Conductor Integration', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    const config = getConductorConfig();
    const available = await isConductorAvailable(config);

    if (!available) {
      console.log('⚠️  Conductor not available, skipping tests');
      return;
    }

    ctx = await setupTestContext();
  });

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  });

  describe('Reputation Management', () => {
    it('should register a new reputation entry', async () => {
      const agentId = generateTestAgentId();

      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.5,
          source_happ: 'test_happ',
        },
      });

      expect(result).toBeDefined();
      await waitForSync();
    });

    it('should update reputation score', async () => {
      const agentId = generateTestAgentId();

      // Register first
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.5,
          source_happ: 'test_happ',
        },
      });

      await waitForSync();

      // Update
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'update_reputation',
        payload: {
          agent_id: agentId,
          delta: 0.1,
          reason: 'positive_interaction',
        },
      });

      expect(result).toBeDefined();
    });

    it('should query cross-hApp reputation', async () => {
      const agentId = generateTestAgentId();

      // Register in multiple "hApps" (simulated via different source_happ values)
      const happs = ['marketplace', 'governance', 'social'];

      for (const happ of happs) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'register_reputation',
          payload: {
            agent_id: agentId,
            initial_score: 0.7 + Math.random() * 0.2,
            source_happ: happ,
          },
        });
      }

      await waitForSync(1000);

      // Query aggregate
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'get_cross_happ_reputation',
        payload: {
          agent_id: agentId,
        },
      });

      expect(result).toBeDefined();
      expect(result.scores).toBeInstanceOf(Array);
      expect(result.aggregate).toBeGreaterThan(0);
    });
  });

  describe('Bridge Events', () => {
    it('should emit a bridge event', async () => {
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'test_event',
          source_happ: 'test_happ',
          payload: { test: true, timestamp: Date.now() },
        },
      });

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
    });

    it('should query events by type', async () => {
      // Emit some events
      for (let i = 0; i < 3; i++) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'emit_event',
          payload: {
            event_type: 'query_test_event',
            source_happ: 'test_happ',
            payload: { index: i },
          },
        });
      }

      await waitForSync();

      // Query
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'query_events',
        payload: {
          event_type: 'query_test_event',
          limit: 10,
        },
      });

      expect(result).toBeInstanceOf(Array);
      expect(result.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Trust Verification', () => {
    it('should verify agent trustworthiness', async () => {
      const agentId = generateTestAgentId();

      // Register with high reputation
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.9,
          source_happ: 'test_happ',
        },
      });

      await waitForSync();

      // Verify trustworthiness
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'is_trustworthy',
        payload: {
          agent_id: agentId,
          threshold: 0.7,
        },
      });

      expect(result).toBe(true);
    });

    it('should detect untrusted agents', async () => {
      const agentId = generateTestAgentId();

      // Register with low reputation
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.3,
          source_happ: 'test_happ',
        },
      });

      await waitForSync();

      // Verify trustworthiness
      const result = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'is_trustworthy',
        payload: {
          agent_id: agentId,
          threshold: 0.7,
        },
      });

      expect(result).toBe(false);
    });
  });
});
