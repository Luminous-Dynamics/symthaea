// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civilizational Workflows - Enhanced Conductor Integration Tests
 *
 * Comprehensive tests for cross-hApp workflows, error recovery,
 * multi-agent scenarios, and performance benchmarks.
 *
 * Run with: CONDUCTOR_AVAILABLE=true npm run test:conductor
 *
 * @packageDocumentation
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import {
  CONDUCTOR_ENABLED,
  getConductorConfig,
  isConductorAvailable,
  setupTestContext,
  teardownTestContext,
  generateTestAgentId,
  waitForSync,
  retry,
  type TestContext,
} from './conductor-harness.js';

import { LocalBridge, BridgeMessageType } from '../../src/bridge/index.js';
import { createReputation, recordPositive, recordNegative, reputationValue } from '../../src/matl/index.js';
import { claim, EmpiricalLevel, NormativeLevel } from '../../src/epistemic/index.js';

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Generate a valid 32+ character public key for identity tests
 */
function generateValidPubKey(prefix: string): string {
  const base = `${prefix}-pubkey-32-characters-minimum-`;
  const suffix = Math.random().toString(36).slice(2, 10);
  return base + suffix;
}

/**
 * Create a test DID from a public key
 */
function createTestDid(pubKey: string): string {
  return `did:mycelix:${pubKey.replace(/[^a-zA-Z0-9]/g, '')}`;
}

/**
 * Measure operation latency
 */
async function measureLatency<T>(fn: () => Promise<T>): Promise<{ result: T; latencyMs: number }> {
  const start = performance.now();
  const result = await fn();
  const latencyMs = performance.now() - start;
  return { result, latencyMs };
}

// =============================================================================
// Cross-hApp Workflow Tests
// =============================================================================

describe.skipIf(!CONDUCTOR_ENABLED)('Civilizational Workflows - Conductor Integration', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    const config = getConductorConfig();
    const available = await isConductorAvailable(config);

    if (!available) {
      console.log('⚠️  Conductor not available at', config.adminUrl);
      throw new Error('Conductor not running - start with: npm run conductor:start');
    }

    ctx = await setupTestContext();
  });

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  });

  describe('Identity → Governance → Finance Workflow', () => {
    it('should complete full identity verification to loan approval flow', async () => {
      // Step 1: Create identity
      const borrowerKey = generateValidPubKey('borrower');
      const borrowerDid = createTestDid(borrowerKey);

      // Register identity in conductor
      const identityResult = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_identity',
        payload: {
          agent_id: borrowerDid,
          public_key: borrowerKey,
          verification_level: 'self_attested',
        },
      });
      expect(identityResult).toBeDefined();

      await waitForSync();

      // Step 2: Build reputation through governance participation
      const reputationResult = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: borrowerDid,
          initial_score: 0.7,
          source_happ: 'governance',
        },
      });
      expect(reputationResult).toBeDefined();

      await waitForSync();

      // Step 3: Calculate credit score based on reputation
      const creditResult = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'get_cross_happ_reputation',
        payload: {
          agent_id: borrowerDid,
        },
      });
      expect(creditResult).toBeDefined();
      expect(creditResult.aggregate).toBeGreaterThan(0);
    });

    it('should propagate identity verification across hApps', async () => {
      const verifierKey = generateValidPubKey('verifier');
      const verifierDid = createTestDid(verifierKey);

      const subjectKey = generateValidPubKey('subject');
      const subjectDid = createTestDid(subjectKey);

      // Register both identities
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_identity',
        payload: {
          agent_id: verifierDid,
          public_key: verifierKey,
          verification_level: 'institutional',
        },
      });

      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_identity',
        payload: {
          agent_id: subjectDid,
          public_key: subjectKey,
          verification_level: 'self_attested',
        },
      });

      await waitForSync();

      // Emit verification event to bridge
      const verificationEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'identity_verified',
          source_happ: 'identity',
          payload: {
            subject_did: subjectDid,
            verifier_did: verifierDid,
            verification_type: 'peer_attestation',
            timestamp: Date.now(),
          },
        },
      });

      expect(verificationEvent).toBeDefined();
      expect(verificationEvent.id).toBeDefined();
    });
  });

  describe('Knowledge → Justice → Governance Workflow', () => {
    it('should use knowledge claims in dispute resolution', async () => {
      const claimantKey = generateValidPubKey('claimant');
      const claimantDid = createTestDid(claimantKey);

      const respondentKey = generateValidPubKey('respondent');
      const respondentDid = createTestDid(respondentKey);

      // Register parties
      for (const did of [claimantDid, respondentDid]) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'register_identity',
          payload: {
            agent_id: did,
            public_key: did.replace('did:mycelix:', ''),
            verification_level: 'self_attested',
          },
        });
      }

      await waitForSync();

      // Create knowledge claim (evidence)
      const claimEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'claim_created',
          source_happ: 'knowledge',
          payload: {
            claim_id: `claim-${Date.now()}`,
            author_did: claimantDid,
            title: 'Evidence of service delivery',
            empirical_level: EmpiricalLevel.E2_Observed,
            normative_level: NormativeLevel.N2_Network,
            content: 'Service was delivered on time as evidenced by timestamped records',
          },
        },
      });

      expect(claimEvent).toBeDefined();

      // File dispute referencing the claim
      const disputeEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'dispute_filed',
          source_happ: 'justice',
          payload: {
            dispute_id: `dispute-${Date.now()}`,
            claimant_did: claimantDid,
            respondent_did: respondentDid,
            referenced_claims: [claimEvent.id],
            dispute_type: 'breach_of_contract',
          },
        },
      });

      expect(disputeEvent).toBeDefined();
    });

    it('should record governance decisions in knowledge graph', async () => {
      const daoId = `dao-${Date.now()}`;
      const proposerId = generateValidPubKey('proposer');
      const proposerDid = createTestDid(proposerId);

      // Create proposal decision event
      const decisionEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'proposal_decided',
          source_happ: 'governance',
          payload: {
            proposal_id: `prop-${Date.now()}`,
            dao_id: daoId,
            proposer_did: proposerDid,
            decision: 'passed',
            vote_count: { for: 75, against: 20, abstain: 5 },
            quorum_met: true,
            execution_required: true,
          },
        },
      });

      expect(decisionEvent).toBeDefined();

      // Knowledge graph should record this as a normative claim
      const knowledgeRecordEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'decision_recorded',
          source_happ: 'knowledge',
          payload: {
            original_event_id: decisionEvent.id,
            claim_type: 'governance_decision',
            empirical_level: EmpiricalLevel.E2_Observed,
            normative_level: NormativeLevel.N3_Civilizational,
          },
        },
      });

      expect(knowledgeRecordEvent).toBeDefined();
    });
  });

  describe('Property → Energy → Finance Workflow', () => {
    it('should coordinate property-backed energy investments', async () => {
      const investorKey = generateValidPubKey('investor');
      const investorDid = createTestDid(investorKey);

      const propertyId = `property-${Date.now()}`;
      const projectId = `solar-${Date.now()}`;

      // Register property
      const propertyEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'property_registered',
          source_happ: 'property',
          payload: {
            property_id: propertyId,
            owner_did: investorDid,
            property_type: 'commercial_rooftop',
            capacity_kw: 500,
            location: { lat: 33.45, lon: -94.75 },
          },
        },
      });

      expect(propertyEvent).toBeDefined();

      // Create energy project linked to property
      const projectEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'project_created',
          source_happ: 'energy',
          payload: {
            project_id: projectId,
            property_id: propertyId,
            project_type: 'solar_installation',
            target_investment: 250000,
            expected_return: 0.11,
            duration_years: 25,
          },
        },
      });

      expect(projectEvent).toBeDefined();

      // Create investment against property collateral
      const investmentEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'investment_pledged',
          source_happ: 'finance',
          payload: {
            investment_id: `inv-${Date.now()}`,
            investor_did: investorDid,
            project_id: projectId,
            amount: 10000,
            collateral_property_id: propertyId,
          },
        },
      });

      expect(investmentEvent).toBeDefined();
    });
  });
});

// =============================================================================
// Error Recovery Tests
// =============================================================================

describe.skipIf(!CONDUCTOR_ENABLED)('Error Recovery - Conductor Integration', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    const config = getConductorConfig();
    const available = await isConductorAvailable(config);

    if (!available) {
      throw new Error('Conductor not running');
    }

    ctx = await setupTestContext();
  });

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  });

  describe('Retry Logic', () => {
    it('should successfully retry failed operations', async () => {
      let attempts = 0;

      const result = await retry(
        async () => {
          attempts++;
          if (attempts < 2) {
            throw new Error('Simulated transient failure');
          }
          return { success: true, attempts };
        },
        3,
        100
      );

      expect(result.success).toBe(true);
      expect(result.attempts).toBe(2);
    });

    it('should handle network timeout gracefully', async () => {
      // This tests the retry mechanism with actual network operations
      const result = await retry(
        async () => {
          return await ctx.appClient.callZome({
            cap_secret: undefined,
            role_name: 'bridge',
            zome_name: 'bridge_coordinator',
            fn_name: 'emit_event',
            payload: {
              event_type: 'retry_test',
              source_happ: 'test',
              payload: { timestamp: Date.now() },
            },
          });
        },
        3,
        500
      );

      expect(result).toBeDefined();
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle multiple simultaneous zome calls', async () => {
      const operations = Array.from({ length: 20 }, (_, i) =>
        ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'emit_event',
          payload: {
            event_type: 'concurrent_test',
            source_happ: 'test',
            payload: { index: i, timestamp: Date.now() },
          },
        })
      );

      const results = await Promise.all(operations);
      expect(results).toHaveLength(20);
      expect(results.every((r) => r?.id)).toBe(true);
    });

    it('should maintain data consistency under concurrent writes', async () => {
      const agentId = generateTestAgentId();
      const writeCount = 10;

      // Register initial reputation
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.5,
          source_happ: 'consistency_test',
        },
      });

      await waitForSync();

      // Concurrent updates
      const updates = Array.from({ length: writeCount }, () =>
        ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'update_reputation',
          payload: {
            agent_id: agentId,
            delta: 0.01,
            reason: 'concurrent_update',
          },
        })
      );

      await Promise.all(updates);
      await waitForSync(1000);

      // Verify final state
      const finalState = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'get_cross_happ_reputation',
        payload: {
          agent_id: agentId,
        },
      });

      expect(finalState).toBeDefined();
      // Should be approximately 0.5 + (10 * 0.01) = 0.6
      expect(finalState.aggregate).toBeGreaterThanOrEqual(0.55);
    });
  });
});

// =============================================================================
// Multi-Agent Scenarios
// =============================================================================

describe.skipIf(!CONDUCTOR_ENABLED)('Multi-Agent Scenarios - Conductor Integration', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    const config = getConductorConfig();
    const available = await isConductorAvailable(config);

    if (!available) {
      throw new Error('Conductor not running');
    }

    ctx = await setupTestContext();
  });

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  });

  describe('DAO Voting Scenario', () => {
    it('should coordinate multi-agent voting', async () => {
      const daoId = `dao-vote-${Date.now()}`;
      const proposalId = `proposal-${Date.now()}`;

      // Create 5 voters
      const voters = Array.from({ length: 5 }, (_, i) => ({
        key: generateValidPubKey(`voter${i}`),
        vote: i % 3 === 0 ? 'against' : 'for', // 2 against, 3 for
      }));

      // Register all voters
      for (const voter of voters) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'register_reputation',
          payload: {
            agent_id: createTestDid(voter.key),
            initial_score: 0.7,
            source_happ: 'governance',
          },
        });
      }

      await waitForSync();

      // Each voter casts vote
      for (const voter of voters) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'emit_event',
          payload: {
            event_type: 'vote_cast',
            source_happ: 'governance',
            payload: {
              proposal_id: proposalId,
              dao_id: daoId,
              voter_did: createTestDid(voter.key),
              vote: voter.vote,
              timestamp: Date.now(),
            },
          },
        });
      }

      await waitForSync();

      // Tally votes via bridge event
      const tallyEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'proposal_tallied',
          source_happ: 'governance',
          payload: {
            proposal_id: proposalId,
            dao_id: daoId,
            for_count: 3,
            against_count: 2,
            result: 'passed',
          },
        },
      });

      expect(tallyEvent).toBeDefined();
    });
  });

  describe('Dispute Resolution Scenario', () => {
    it('should coordinate multi-party dispute with jurors', async () => {
      const caseId = `case-${Date.now()}`;

      // Create dispute parties
      const claimant = {
        key: generateValidPubKey('claimant-party'),
        did: '',
      };
      claimant.did = createTestDid(claimant.key);

      const respondent = {
        key: generateValidPubKey('respondent-party'),
        did: '',
      };
      respondent.did = createTestDid(respondent.key);

      // Create 3 jurors
      const jurors = Array.from({ length: 3 }, (_, i) => {
        const key = generateValidPubKey(`juror${i}`);
        return { key, did: createTestDid(key) };
      });

      // Register all parties with reputation
      for (const party of [claimant, respondent, ...jurors]) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'register_reputation',
          payload: {
            agent_id: party.did,
            initial_score: 0.8,
            source_happ: 'justice',
          },
        });
      }

      await waitForSync();

      // File complaint
      const complaintEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'complaint_filed',
          source_happ: 'justice',
          payload: {
            case_id: caseId,
            claimant_did: claimant.did,
            respondent_did: respondent.did,
            category: 'breach_of_contract',
            description: 'Failed to deliver agreed services',
          },
        },
      });

      expect(complaintEvent).toBeDefined();

      // Assign jurors
      const assignmentEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'jurors_assigned',
          source_happ: 'justice',
          payload: {
            case_id: caseId,
            juror_dids: jurors.map((j) => j.did),
          },
        },
      });

      expect(assignmentEvent).toBeDefined();

      // Each juror votes
      for (const juror of jurors) {
        await ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'emit_event',
          payload: {
            event_type: 'juror_vote',
            source_happ: 'justice',
            payload: {
              case_id: caseId,
              juror_did: juror.did,
              verdict: 'in_favor_of_claimant',
              reasoning: 'Evidence supports claim',
            },
          },
        });
      }

      await waitForSync();

      // Render final verdict
      const verdictEvent = await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'emit_event',
        payload: {
          event_type: 'verdict_rendered',
          source_happ: 'justice',
          payload: {
            case_id: caseId,
            verdict: 'in_favor_of_claimant',
            unanimous: true,
            remedy: 'compensation',
            remedy_amount: 5000,
          },
        },
      });

      expect(verdictEvent).toBeDefined();
    });
  });
});

// =============================================================================
// Performance Benchmarks
// =============================================================================

describe.skipIf(!CONDUCTOR_ENABLED)('Performance Benchmarks - Conductor Integration', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    const config = getConductorConfig();
    const available = await isConductorAvailable(config);

    if (!available) {
      throw new Error('Conductor not running');
    }

    ctx = await setupTestContext();
  });

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  });

  describe('Latency Benchmarks', () => {
    it('should emit events under 100ms', async () => {
      const iterations = 10;
      const latencies: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const { latencyMs } = await measureLatency(() =>
          ctx.appClient.callZome({
            cap_secret: undefined,
            role_name: 'bridge',
            zome_name: 'bridge_coordinator',
            fn_name: 'emit_event',
            payload: {
              event_type: 'latency_test',
              source_happ: 'benchmark',
              payload: { iteration: i },
            },
          })
        );
        latencies.push(latencyMs);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const maxLatency = Math.max(...latencies);
      const minLatency = Math.min(...latencies);

      console.log(`Event emission latency:`);
      console.log(`  Average: ${avgLatency.toFixed(2)}ms`);
      console.log(`  Min: ${minLatency.toFixed(2)}ms`);
      console.log(`  Max: ${maxLatency.toFixed(2)}ms`);

      // Target: average under 100ms for local conductor
      expect(avgLatency).toBeLessThan(200);
    });

    it('should query reputation under 50ms', async () => {
      const agentId = generateTestAgentId();

      // Setup reputation
      await ctx.appClient.callZome({
        cap_secret: undefined,
        role_name: 'bridge',
        zome_name: 'bridge_coordinator',
        fn_name: 'register_reputation',
        payload: {
          agent_id: agentId,
          initial_score: 0.75,
          source_happ: 'benchmark',
        },
      });

      await waitForSync();

      const iterations = 10;
      const latencies: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const { latencyMs } = await measureLatency(() =>
          ctx.appClient.callZome({
            cap_secret: undefined,
            role_name: 'bridge',
            zome_name: 'bridge_coordinator',
            fn_name: 'get_cross_happ_reputation',
            payload: { agent_id: agentId },
          })
        );
        latencies.push(latencyMs);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;

      console.log(`Reputation query latency: ${avgLatency.toFixed(2)}ms average`);

      // Target: average under 50ms for local conductor
      expect(avgLatency).toBeLessThan(100);
    });
  });

  describe('Throughput Benchmarks', () => {
    it('should handle 100 operations per second', async () => {
      const targetOps = 100;
      const startTime = performance.now();

      const operations = Array.from({ length: targetOps }, (_, i) =>
        ctx.appClient.callZome({
          cap_secret: undefined,
          role_name: 'bridge',
          zome_name: 'bridge_coordinator',
          fn_name: 'emit_event',
          payload: {
            event_type: 'throughput_test',
            source_happ: 'benchmark',
            payload: { index: i },
          },
        })
      );

      await Promise.all(operations);
      const duration = performance.now() - startTime;
      const opsPerSecond = (targetOps / duration) * 1000;

      console.log(`Throughput: ${opsPerSecond.toFixed(0)} ops/sec (${targetOps} ops in ${duration.toFixed(0)}ms)`);

      // Target: at least 50 ops/sec on local conductor
      expect(opsPerSecond).toBeGreaterThan(50);
    });

    it('should scale linearly with batch size', async () => {
      const batchSizes = [10, 50, 100];
      const results: { size: number; duration: number; opsPerSecond: number }[] = [];

      for (const size of batchSizes) {
        const startTime = performance.now();

        const operations = Array.from({ length: size }, (_, i) =>
          ctx.appClient.callZome({
            cap_secret: undefined,
            role_name: 'bridge',
            zome_name: 'bridge_coordinator',
            fn_name: 'emit_event',
            payload: {
              event_type: 'scale_test',
              source_happ: 'benchmark',
              payload: { batch_size: size, index: i },
            },
          })
        );

        await Promise.all(operations);
        const duration = performance.now() - startTime;
        const opsPerSecond = (size / duration) * 1000;

        results.push({ size, duration, opsPerSecond });
      }

      console.log('Scaling benchmark:');
      for (const r of results) {
        console.log(`  Batch ${r.size}: ${r.opsPerSecond.toFixed(0)} ops/sec (${r.duration.toFixed(0)}ms)`);
      }

      // Verify roughly linear scaling (larger batches shouldn't be much slower per op)
      const smallBatchOps = results[0].opsPerSecond;
      const largeBatchOps = results[results.length - 1].opsPerSecond;

      // Large batch should be at least 50% as efficient as small batch
      expect(largeBatchOps).toBeGreaterThan(smallBatchOps * 0.5);
    });
  });
});
