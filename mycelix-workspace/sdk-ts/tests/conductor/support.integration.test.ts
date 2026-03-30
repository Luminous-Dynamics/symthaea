// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Support hApp Conductor Integration Tests
 *
 * These tests verify the Support SDK clients (knowledge, tickets, diagnostics)
 * work correctly with a real Holochain conductor. They require:
 *
 * 1. A running Holochain conductor with the Mycelix commons hApp installed
 * 2. HOLOCHAIN_CONDUCTOR_AVAILABLE=true environment variable
 *
 * Run with: npm run test:conductor
 *
 * Covers 5 scenarios:
 *   1. createArticle -> getArticle roundtrip
 *   2. createTicket -> updateTicket -> closeTicket lifecycle
 *   3. runDiagnostic with ticket link
 *   4. setPrivacyPreference -> getPrivacyPreference roundtrip
 *   5. publishCognitiveUpdate -> getCognitiveUpdatesByCategory
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  SupportClient,
  createSupportClient,
  KnowledgeClient,
  TicketsClient,
  DiagnosticsClient,
  type KnowledgeArticleInput,
  type SupportTicketInput,
  type DiagnosticResultInput,
  type PrivacyPreferenceInput,
  type CognitiveUpdateInput,
} from '../../src/clients/support/index.js';
import {
  type TestContext,
  setupTestContext,
  teardownTestContext,
  getConductorConfig,
  isConductorAvailable,
  waitForSync,
} from './conductor-harness.js';

// ============================================================================
// Conductor Availability Gate
// ============================================================================

const CONDUCTOR_AVAILABLE = process.env.HOLOCHAIN_CONDUCTOR_AVAILABLE === 'true';

// ============================================================================
// Test State
// ============================================================================

let ctx: TestContext | null = null;
let supportClient: SupportClient | null = null;
let agentPubKey: Uint8Array;

// Shared hashes across test scenarios (populated by earlier tests, consumed by later ones)
let createdArticleHash: Uint8Array | null = null;
let createdTicketHash: Uint8Array | null = null;
let updatedTicketHash: Uint8Array | null = null;
let diagnosticHash: Uint8Array | null = null;
let privacyPrefHash: Uint8Array | null = null;
let cognitiveUpdateHash: Uint8Array | null = null;

// ============================================================================
// Test Suite
// ============================================================================

describe.skipIf(!CONDUCTOR_AVAILABLE)('Support Conductor Integration Tests', () => {
  beforeAll(async () => {
    try {
      ctx = await setupTestContext();
      supportClient = SupportClient.fromClient(ctx.appClient);
      agentPubKey = ctx.appClient.myPubKey;
    } catch (error) {
      console.log(
        'Conductor not available, skipping Support integration tests:',
        error instanceof Error ? error.message : String(error),
      );
    }
  }, 60000);

  afterAll(async () => {
    if (ctx) {
      await teardownTestContext(ctx);
    }
  }, 30000);

  // ==========================================================================
  // Scenario 1: createArticle -> getArticle roundtrip
  // ==========================================================================

  describe('Knowledge Article Roundtrip', () => {
    it('should create a knowledge article', async () => {
      expect(supportClient).not.toBeNull();
      const knowledge = supportClient!.knowledge;

      const input: KnowledgeArticleInput = {
        title: 'How to diagnose DHT sync issues',
        content:
          'When your Holochain node stops syncing, check the following:\n' +
          '1. Verify network connectivity with `hc sandbox ping`\n' +
          '2. Check conductor logs for gossip errors\n' +
          '3. Restart the conductor if stale cache is suspected',
        category: 'Holochain',
        tags: ['dht', 'sync', 'troubleshooting', 'conductor'],
        author: agentPubKey,
        source: 'Community',
        difficultyLevel: 'Intermediate',
        upvotes: 0,
        verified: false,
        deprecated: false,
        deprecationReason: null,
        version: 1,
      };

      const article = await knowledge.createArticle(input);

      expect(article).toBeDefined();
      expect(article.actionHash).toBeDefined();
      expect(article.actionHash).toBeInstanceOf(Uint8Array);
      expect(article.title).toBe(input.title);
      expect(article.content).toBe(input.content);
      expect(article.category).toBe('Holochain');
      expect(article.tags).toEqual(['dht', 'sync', 'troubleshooting', 'conductor']);
      expect(article.source).toBe('Community');
      expect(article.difficultyLevel).toBe('Intermediate');
      expect(article.verified).toBe(false);
      expect(article.deprecated).toBe(false);
      expect(article.version).toBe(1);

      createdArticleHash = article.actionHash;
    });

    it('should retrieve the created article by hash', async () => {
      expect(supportClient).not.toBeNull();
      expect(createdArticleHash).not.toBeNull();

      await waitForSync();

      const article = await supportClient!.knowledge.getArticle(createdArticleHash!);

      expect(article).not.toBeNull();
      expect(article!.title).toBe('How to diagnose DHT sync issues');
      expect(article!.category).toBe('Holochain');
      expect(article!.tags).toContain('dht');
      expect(article!.tags).toContain('troubleshooting');
      expect(article!.source).toBe('Community');
      expect(article!.difficultyLevel).toBe('Intermediate');
      expect(article!.actionHash).toEqual(createdArticleHash);
    });

    it('should return null for a nonexistent article hash', async () => {
      expect(supportClient).not.toBeNull();

      const fakeHash = new Uint8Array(39).fill(0xff);
      const article = await supportClient!.knowledge.getArticle(fakeHash);

      expect(article).toBeNull();
    });
  });

  // ==========================================================================
  // Scenario 2: createTicket -> updateTicket -> closeTicket lifecycle
  // ==========================================================================

  describe('Ticket Lifecycle', () => {
    it('should create a support ticket', async () => {
      expect(supportClient).not.toBeNull();
      const tickets = supportClient!.tickets;
      const now = Date.now();

      const input: SupportTicketInput = {
        title: 'Node not syncing after power outage',
        description:
          'My Holochain node stopped syncing approximately 2 hours ago following ' +
          'a brief power outage. The conductor is running but no new entries appear.',
        category: 'Holochain',
        priority: 'High',
        status: 'Open',
        requester: agentPubKey,
        assignee: null,
        autonomyLevel: 'Advisory',
        systemInfo: 'NixOS 25.11, Holochain 0.6.0, 16GB RAM',
        isPreemptive: false,
        predictionConfidence: null,
        createdAt: now,
        updatedAt: now,
      };

      const ticketHash = await tickets.createTicket(input);

      expect(ticketHash).toBeDefined();
      expect(ticketHash).toBeInstanceOf(Uint8Array);
      expect(ticketHash.length).toBeGreaterThan(0);

      createdTicketHash = ticketHash;
    });

    it('should update the ticket status to InProgress', async () => {
      expect(supportClient).not.toBeNull();
      expect(createdTicketHash).not.toBeNull();

      await waitForSync();

      const now = Date.now();
      const newTicketHash = await supportClient!.tickets.updateTicket({
        originalHash: createdTicketHash!,
        updated: {
          title: 'Node not syncing after power outage',
          description:
            'My Holochain node stopped syncing approximately 2 hours ago following ' +
            'a brief power outage. The conductor is running but no new entries appear.\n\n' +
            'UPDATE: Diagnostics running, stale gossip cache suspected.',
          category: 'Holochain',
          priority: 'High',
          status: 'InProgress',
          requester: agentPubKey,
          assignee: agentPubKey,
          autonomyLevel: 'Advisory',
          systemInfo: 'NixOS 25.11, Holochain 0.6.0, 16GB RAM',
          isPreemptive: false,
          predictionConfidence: null,
          createdAt: now,
          updatedAt: now,
        },
      });

      expect(newTicketHash).toBeDefined();
      expect(newTicketHash).toBeInstanceOf(Uint8Array);

      updatedTicketHash = newTicketHash;
    });

    it('should close the ticket', async () => {
      expect(supportClient).not.toBeNull();
      // closeTicket takes the latest ticket hash
      const hashToClose = updatedTicketHash ?? createdTicketHash;
      expect(hashToClose).not.toBeNull();

      await waitForSync();

      // closeTicket returns void on success
      await expect(
        supportClient!.tickets.closeTicket(hashToClose!),
      ).resolves.toBeUndefined();
    });

    it('should retrieve the ticket after lifecycle', async () => {
      expect(supportClient).not.toBeNull();
      // Use the most recent hash from the update
      const hashToGet = updatedTicketHash ?? createdTicketHash;
      expect(hashToGet).not.toBeNull();

      await waitForSync();

      const ticket = await supportClient!.tickets.getTicket(hashToGet!);

      expect(ticket).not.toBeNull();
      expect(ticket.title).toBe('Node not syncing after power outage');
      expect(ticket.category).toBe('Holochain');
      expect(ticket.priority).toBe('High');
    });
  });

  // ==========================================================================
  // Scenario 3: runDiagnostic with ticket link
  // ==========================================================================

  describe('Diagnostic with Ticket Link', () => {
    let diagnosticTicketHash: Uint8Array | null = null;

    beforeAll(async () => {
      // Create a dedicated ticket for diagnostic linkage
      if (!supportClient) return;

      const now = Date.now();
      diagnosticTicketHash = await supportClient.tickets.createTicket({
        title: 'Conductor memory usage abnormally high',
        description: 'Memory usage exceeds 4GB after 24 hours of uptime',
        category: 'Software',
        priority: 'Medium',
        status: 'Open',
        requester: agentPubKey,
        assignee: null,
        autonomyLevel: 'SemiAutonomous',
        systemInfo: 'NixOS 25.11, 32GB RAM, Holochain 0.6.0',
        isPreemptive: false,
        predictionConfidence: null,
        createdAt: now,
        updatedAt: now,
      });

      await waitForSync();
    });

    it('should run a diagnostic linked to a ticket', async () => {
      expect(supportClient).not.toBeNull();
      expect(diagnosticTicketHash).not.toBeNull();

      const input: DiagnosticResultInput = {
        ticketHash: diagnosticTicketHash,
        diagnosticType: 'MemoryUsage',
        findings:
          'Conductor process using 4.2GB RSS. DHT cache size: 2.1GB. ' +
          'Working memory: 1.8GB. Gossip buffer: 0.3GB. ' +
          'Growth rate: ~150MB/hour, suggesting a slow memory leak in gossip handling.',
        severity: 'Warning',
        recommendations: [
          'Schedule conductor restart during low-traffic window',
          'Enable gossip cache eviction with max_cache_size=1GB',
          'Monitor memory after restart to confirm leak pattern',
        ],
        agent: agentPubKey,
        scrubbed: false,
        createdAt: Date.now(),
      };

      diagnosticHash = await supportClient!.diagnostics.runDiagnostic(input);

      expect(diagnosticHash).toBeDefined();
      expect(diagnosticHash).toBeInstanceOf(Uint8Array);
      expect(diagnosticHash!.length).toBeGreaterThan(0);
    });

    it('should retrieve the diagnostic result', async () => {
      expect(supportClient).not.toBeNull();
      expect(diagnosticHash).not.toBeNull();

      await waitForSync();

      const diagnostic = await supportClient!.diagnostics.getDiagnostic(diagnosticHash!);

      expect(diagnostic).not.toBeNull();
      expect(diagnostic.diagnosticType).toBe('MemoryUsage');
      expect(diagnostic.severity).toBe('Warning');
      expect(diagnostic.findings).toContain('4.2GB');
      expect(diagnostic.recommendations).toHaveLength(3);
      expect(diagnostic.recommendations[0]).toContain('conductor restart');
      expect(diagnostic.scrubbed).toBe(false);
      expect(diagnostic.ticketHash).toEqual(diagnosticTicketHash);
    });

    it('should list diagnostics for the linked ticket', async () => {
      expect(supportClient).not.toBeNull();
      expect(diagnosticTicketHash).not.toBeNull();

      await waitForSync();

      const diagnostics = await supportClient!.diagnostics.listDiagnostics(diagnosticTicketHash!);

      expect(Array.isArray(diagnostics)).toBe(true);
      expect(diagnostics.length).toBeGreaterThanOrEqual(1);

      const found = diagnostics.find(
        (d: any) => d.diagnosticType === 'MemoryUsage',
      );
      expect(found).toBeDefined();
    });
  });

  // ==========================================================================
  // Scenario 4: setPrivacyPreference -> getPrivacyPreference roundtrip
  // ==========================================================================

  describe('Privacy Preference Roundtrip', () => {
    it('should set a privacy preference', async () => {
      expect(supportClient).not.toBeNull();

      const input: PrivacyPreferenceInput = {
        agent: agentPubKey,
        sharingTier: 'Anonymized',
        allowedCategories: ['Network', 'Holochain', 'Software'],
        shareSystemInfo: false,
        shareResolutionPatterns: true,
        shareCognitiveUpdates: false,
        updatedAt: Date.now(),
      };

      privacyPrefHash = await supportClient!.diagnostics.setPrivacyPreference(input);

      expect(privacyPrefHash).toBeDefined();
      expect(privacyPrefHash).toBeInstanceOf(Uint8Array);
      expect(privacyPrefHash!.length).toBeGreaterThan(0);
    });

    it('should retrieve the privacy preference for the agent', async () => {
      expect(supportClient).not.toBeNull();

      await waitForSync();

      const pref = await supportClient!.diagnostics.getPrivacyPreference(agentPubKey);

      expect(pref).not.toBeNull();
      // The shape depends on whether the zome returns a raw record or mapped object.
      // Verify the core fields are present in either format.
      const prefData = pref.entry?.Present ?? pref;
      const sharingTier = prefData.sharingTier ?? prefData.sharing_tier;
      const allowedCategories = prefData.allowedCategories ?? prefData.allowed_categories;
      const shareSystemInfo = prefData.shareSystemInfo ?? prefData.share_system_info;
      const shareResolutionPatterns =
        prefData.shareResolutionPatterns ?? prefData.share_resolution_patterns;
      const shareCognitiveUpdates =
        prefData.shareCognitiveUpdates ?? prefData.share_cognitive_updates;

      expect(sharingTier).toBe('Anonymized');
      expect(allowedCategories).toContain('Network');
      expect(allowedCategories).toContain('Holochain');
      expect(allowedCategories).toContain('Software');
      expect(allowedCategories).toHaveLength(3);
      expect(shareSystemInfo).toBe(false);
      expect(shareResolutionPatterns).toBe(true);
      expect(shareCognitiveUpdates).toBe(false);
    });

    it('should update the privacy preference to Full tier', async () => {
      expect(supportClient).not.toBeNull();

      const updatedInput: PrivacyPreferenceInput = {
        agent: agentPubKey,
        sharingTier: 'Full',
        allowedCategories: [
          'Network', 'Hardware', 'Software', 'Holochain', 'Mycelix', 'Security', 'General',
        ],
        shareSystemInfo: true,
        shareResolutionPatterns: true,
        shareCognitiveUpdates: true,
        updatedAt: Date.now(),
      };

      const newHash = await supportClient!.diagnostics.setPrivacyPreference(updatedInput);

      expect(newHash).toBeDefined();
      expect(newHash).toBeInstanceOf(Uint8Array);

      await waitForSync();

      const pref = await supportClient!.diagnostics.getPrivacyPreference(agentPubKey);
      expect(pref).not.toBeNull();

      const prefData = pref.entry?.Present ?? pref;
      const sharingTier = prefData.sharingTier ?? prefData.sharing_tier;
      expect(sharingTier).toBe('Full');
    });
  });

  // ==========================================================================
  // Scenario 5: publishCognitiveUpdate -> getCognitiveUpdatesByCategory
  // ==========================================================================

  describe('Cognitive Update Publish and Query', () => {
    it('should publish a cognitive update with HDC encoding', async () => {
      expect(supportClient).not.toBeNull();

      // Simulate a 2048-byte HDC binary hypervector encoding
      const encoding = Array.from({ length: 2048 }, (_, i) => i % 256);

      const input: CognitiveUpdateInput = {
        category: 'Security',
        encoding,
        phi: 0.73,
        resolutionPattern: 'firewall_reconfiguration:port_scan_detected',
        sourceAgent: agentPubKey,
        createdAt: Date.now(),
      };

      cognitiveUpdateHash = await supportClient!.diagnostics.publishCognitiveUpdate(input);

      expect(cognitiveUpdateHash).toBeDefined();
      expect(cognitiveUpdateHash).toBeInstanceOf(Uint8Array);
      expect(cognitiveUpdateHash!.length).toBeGreaterThan(0);
    });

    it('should publish a second cognitive update in the same category', async () => {
      expect(supportClient).not.toBeNull();

      const encoding = Array.from({ length: 2048 }, (_, i) => (i * 7) % 256);

      const input: CognitiveUpdateInput = {
        category: 'Security',
        encoding,
        phi: 0.81,
        resolutionPattern: 'intrusion_detection:ssh_brute_force',
        sourceAgent: agentPubKey,
        createdAt: Date.now(),
      };

      const secondHash = await supportClient!.diagnostics.publishCognitiveUpdate(input);

      expect(secondHash).toBeDefined();
      expect(secondHash).toBeInstanceOf(Uint8Array);
    });

    it('should retrieve cognitive updates by category', async () => {
      expect(supportClient).not.toBeNull();

      await waitForSync();

      const updates = await supportClient!.diagnostics.getCognitiveUpdatesByCategory('Security');

      expect(Array.isArray(updates)).toBe(true);
      expect(updates.length).toBeGreaterThanOrEqual(2);

      // Verify that both resolution patterns are represented in the results
      const patterns = updates.map((u: any) => {
        return u.resolutionPattern ?? u.resolution_pattern;
      });
      expect(patterns).toContain('firewall_reconfiguration:port_scan_detected');
      expect(patterns).toContain('intrusion_detection:ssh_brute_force');
    });

    it('should return empty array for category with no updates', async () => {
      expect(supportClient).not.toBeNull();

      const updates = await supportClient!.diagnostics.getCognitiveUpdatesByCategory('Hardware');

      expect(Array.isArray(updates)).toBe(true);
      expect(updates).toHaveLength(0);
    });
  });

  // ==========================================================================
  // Client Factory Verification
  // ==========================================================================

  describe('SupportClient factory and sub-clients', () => {
    it('should expose knowledge, tickets, and diagnostics sub-clients', () => {
      expect(supportClient).not.toBeNull();

      expect(supportClient!.knowledge).toBeInstanceOf(KnowledgeClient);
      expect(supportClient!.tickets).toBeInstanceOf(TicketsClient);
      expect(supportClient!.diagnostics).toBeInstanceOf(DiagnosticsClient);
    });

    it('should create an equivalent client via createSupportClient factory', () => {
      expect(ctx).not.toBeNull();

      const factoryClient = createSupportClient(ctx!.appClient);

      expect(factoryClient).toBeInstanceOf(SupportClient);
      expect(factoryClient.knowledge).toBeInstanceOf(KnowledgeClient);
      expect(factoryClient.tickets).toBeInstanceOf(TicketsClient);
      expect(factoryClient.diagnostics).toBeInstanceOf(DiagnosticsClient);
    });

    it('should report connected status', async () => {
      expect(supportClient).not.toBeNull();

      const connected = await supportClient!.isConnected();
      expect(connected).toBe(true);
    });

    it('should expose the agent public key', () => {
      expect(supportClient).not.toBeNull();

      const pubKey = supportClient!.getAgentPubKey();
      expect(pubKey).toBeInstanceOf(Uint8Array);
      expect(pubKey).toEqual(agentPubKey);
    });
  });
});
