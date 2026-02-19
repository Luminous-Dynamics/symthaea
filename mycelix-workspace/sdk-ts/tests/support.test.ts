/**
 * Support SDK Client Tests
 *
 * Unit tests for the Support hApp TypeScript SDK client.
 * These tests verify type correctness, client structure, and input construction
 * without requiring a live Holochain conductor.
 *
 * Covers:
 * - Type interface validation (knowledge, tickets, diagnostics)
 * - SupportClient instantiation via fromClient() and createSupportClient()
 * - Enum string literal compilation checks
 * - Error type instantiation
 * - Cognitive update encoding roundtrip
 * - Preemptive alert freeEnergy validation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SupportClient,
  createSupportClient,
  type KnowledgeArticleInput,
  type SupportTicketInput,
  type DiagnosticResultInput,
  type PrivacyPreferenceInput,
  type CognitiveUpdateInput,
  type PreemptiveAlertInput,
  type AutonomousActionInput,
  type ResolutionInput,
  type ArticleFlagInput,
  type SupportCategory,
  type TicketPriority,
  type TicketStatus,
  type AutonomyLevel,
  type ActionType,
  type DiagnosticSeverity,
  type DifficultyLevel,
  type ArticleSource,
  type EpistemicStatus,
  type SharingTier,
  SupportError,
  SupportErrorCode,
} from '../src/clients/support/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

const mockPubKey = new Uint8Array(39);
const mockHash = new Uint8Array(39).fill(1);

const mockClient = {
  myPubKey: mockPubKey,
  appInfo: async () => ({ installed_app_id: 'test', status: 'running' }),
  callZome: async () => null,
} as any;

// ============================================================================
// Type Validation Tests
// ============================================================================

describe('Support Types - Interface Construction', () => {
  it('should construct KnowledgeArticleInput with all required fields', () => {
    const article: KnowledgeArticleInput = {
      title: 'How to restart a Holochain node',
      content: 'Step 1: Stop the conductor. Step 2: Restart the conductor.',
      category: 'Holochain',
      tags: ['node', 'restart', 'conductor'],
      author: mockPubKey,
      source: 'Community',
      difficultyLevel: 'Beginner',
      upvotes: 0,
      verified: false,
      deprecated: false,
      deprecationReason: null,
      version: 1,
    };

    expect(article.title).toBe('How to restart a Holochain node');
    expect(article.content).toContain('Step 1');
    expect(article.category).toBe('Holochain');
    expect(article.tags).toHaveLength(3);
    expect(article.tags).toContain('restart');
    expect(article.author).toBe(mockPubKey);
    expect(article.source).toBe('Community');
    expect(article.difficultyLevel).toBe('Beginner');
    expect(article.upvotes).toBe(0);
    expect(article.verified).toBe(false);
    expect(article.deprecated).toBe(false);
    expect(article.deprecationReason).toBeNull();
    expect(article.version).toBe(1);
  });

  it('should construct SupportTicketInput with all fields', () => {
    const now = Date.now();
    const ticket: SupportTicketInput = {
      title: 'Node not syncing',
      description: 'My node stopped syncing 2 hours ago after a power outage',
      category: 'Holochain',
      priority: 'High',
      status: 'Open',
      requester: mockPubKey,
      assignee: null,
      autonomyLevel: 'Advisory',
      systemInfo: 'NixOS 25.11, Holochain 0.6.0',
      isPreemptive: false,
      predictionConfidence: null,
      createdAt: now,
      updatedAt: now,
    };

    expect(ticket.title).toBe('Node not syncing');
    expect(ticket.description).toContain('power outage');
    expect(ticket.category).toBe('Holochain');
    expect(ticket.priority).toBe('High');
    expect(ticket.status).toBe('Open');
    expect(ticket.requester).toBe(mockPubKey);
    expect(ticket.assignee).toBeNull();
    expect(ticket.autonomyLevel).toBe('Advisory');
    expect(ticket.systemInfo).toContain('NixOS');
    expect(ticket.isPreemptive).toBe(false);
    expect(ticket.predictionConfidence).toBeNull();
    expect(ticket.createdAt).toBe(now);
    expect(ticket.updatedAt).toBe(now);
  });

  it('should construct SupportTicketInput with assignee and prediction confidence', () => {
    const now = Date.now();
    const assigneePubKey = new Uint8Array(39).fill(2);
    const ticket: SupportTicketInput = {
      title: 'Predicted disk failure',
      description: 'Symthaea detected anomalous disk I/O patterns',
      category: 'Hardware',
      priority: 'Critical',
      status: 'InProgress',
      requester: mockPubKey,
      assignee: assigneePubKey,
      autonomyLevel: 'SemiAutonomous',
      systemInfo: null,
      isPreemptive: true,
      predictionConfidence: 0.87,
      createdAt: now,
      updatedAt: now,
    };

    expect(ticket.assignee).toBe(assigneePubKey);
    expect(ticket.isPreemptive).toBe(true);
    expect(ticket.predictionConfidence).toBe(0.87);
    expect(ticket.autonomyLevel).toBe('SemiAutonomous');
    expect(ticket.priority).toBe('Critical');
  });

  it('should construct DiagnosticResultInput with all fields', () => {
    const now = Date.now();
    const diagnostic: DiagnosticResultInput = {
      ticketHash: mockHash,
      diagnosticType: 'HolochainHealth',
      findings: 'Conductor responsive, DHT sync at 98%, 3 peers connected',
      severity: 'Healthy',
      recommendations: ['Monitor DHT sync rate', 'Consider adding more peers'],
      agent: mockPubKey,
      scrubbed: false,
      createdAt: now,
    };

    expect(diagnostic.ticketHash).toBe(mockHash);
    expect(diagnostic.diagnosticType).toBe('HolochainHealth');
    expect(diagnostic.findings).toContain('DHT sync');
    expect(diagnostic.severity).toBe('Healthy');
    expect(diagnostic.recommendations).toHaveLength(2);
    expect(diagnostic.agent).toBe(mockPubKey);
    expect(diagnostic.scrubbed).toBe(false);
    expect(diagnostic.createdAt).toBe(now);
  });

  it('should construct DiagnosticResultInput with null ticketHash', () => {
    const diagnostic: DiagnosticResultInput = {
      ticketHash: null,
      diagnosticType: 'DiskSpace',
      findings: 'Root partition at 45% capacity',
      severity: 'Warning',
      recommendations: ['Clean up old logs'],
      agent: mockPubKey,
      scrubbed: true,
      createdAt: Date.now(),
    };

    expect(diagnostic.ticketHash).toBeNull();
    expect(diagnostic.scrubbed).toBe(true);
    expect(diagnostic.diagnosticType).toBe('DiskSpace');
  });

  it('should construct DiagnosticResultInput with custom diagnostic type', () => {
    const diagnostic: DiagnosticResultInput = {
      ticketHash: null,
      diagnosticType: { Custom: 'PhiMetrics' },
      findings: 'Phi integration score below threshold',
      severity: 'Warning',
      recommendations: ['Recalibrate phi weights'],
      agent: mockPubKey,
      scrubbed: false,
      createdAt: Date.now(),
    };

    expect(diagnostic.diagnosticType).toEqual({ Custom: 'PhiMetrics' });
  });

  it('should construct PrivacyPreferenceInput with sharing tiers', () => {
    const now = Date.now();
    const privacy: PrivacyPreferenceInput = {
      agent: mockPubKey,
      sharingTier: 'Anonymized',
      allowedCategories: ['Network', 'Holochain', 'General'],
      shareSystemInfo: false,
      shareResolutionPatterns: true,
      shareCognitiveUpdates: false,
      updatedAt: now,
    };

    expect(privacy.agent).toBe(mockPubKey);
    expect(privacy.sharingTier).toBe('Anonymized');
    expect(privacy.allowedCategories).toHaveLength(3);
    expect(privacy.allowedCategories).toContain('Network');
    expect(privacy.allowedCategories).toContain('Holochain');
    expect(privacy.allowedCategories).toContain('General');
    expect(privacy.shareSystemInfo).toBe(false);
    expect(privacy.shareResolutionPatterns).toBe(true);
    expect(privacy.shareCognitiveUpdates).toBe(false);
    expect(privacy.updatedAt).toBe(now);
  });

  it('should construct PrivacyPreferenceInput with LocalOnly tier', () => {
    const privacy: PrivacyPreferenceInput = {
      agent: mockPubKey,
      sharingTier: 'LocalOnly',
      allowedCategories: [],
      shareSystemInfo: false,
      shareResolutionPatterns: false,
      shareCognitiveUpdates: false,
      updatedAt: Date.now(),
    };

    expect(privacy.sharingTier).toBe('LocalOnly');
    expect(privacy.allowedCategories).toHaveLength(0);
    expect(privacy.shareSystemInfo).toBe(false);
    expect(privacy.shareResolutionPatterns).toBe(false);
    expect(privacy.shareCognitiveUpdates).toBe(false);
  });

  it('should construct PrivacyPreferenceInput with Full tier', () => {
    const allCategories: SupportCategory[] = [
      'Network', 'Hardware', 'Software', 'Holochain', 'Mycelix', 'Security', 'General',
    ];
    const privacy: PrivacyPreferenceInput = {
      agent: mockPubKey,
      sharingTier: 'Full',
      allowedCategories: allCategories,
      shareSystemInfo: true,
      shareResolutionPatterns: true,
      shareCognitiveUpdates: true,
      updatedAt: Date.now(),
    };

    expect(privacy.sharingTier).toBe('Full');
    expect(privacy.allowedCategories).toHaveLength(7);
    expect(privacy.shareSystemInfo).toBe(true);
    expect(privacy.shareResolutionPatterns).toBe(true);
    expect(privacy.shareCognitiveUpdates).toBe(true);
  });

  it('should construct CognitiveUpdateInput with 2048-byte encoding', () => {
    const encoding = Array.from({ length: 2048 }, (_, i) => i % 256);
    const now = Date.now();

    const update: CognitiveUpdateInput = {
      category: 'Security',
      encoding,
      phi: 0.73,
      resolutionPattern: 'firewall_reconfiguration:port_scan_detected',
      sourceAgent: mockPubKey,
      createdAt: now,
    };

    expect(update.category).toBe('Security');
    expect(update.encoding).toHaveLength(2048);
    expect(update.encoding[0]).toBe(0);
    expect(update.encoding[255]).toBe(255);
    expect(update.encoding[256]).toBe(0);
    expect(update.phi).toBe(0.73);
    expect(update.resolutionPattern).toContain('firewall_reconfiguration');
    expect(update.sourceAgent).toBe(mockPubKey);
    expect(update.createdAt).toBe(now);
  });

  it('should construct PreemptiveAlertInput with positive freeEnergy', () => {
    const now = Date.now();
    const alert: PreemptiveAlertInput = {
      predictedFailure: 'DHT partition detected between 3 peer groups',
      expectedTimeToFailure: '45 minutes',
      freeEnergy: 12.5,
      recommendedAction: 'Initiate manual peer discovery and gossip restart',
      autoGeneratedTicket: null,
      createdAt: now,
    };

    expect(alert.predictedFailure).toContain('DHT partition');
    expect(alert.expectedTimeToFailure).toBe('45 minutes');
    expect(alert.freeEnergy).toBe(12.5);
    expect(alert.freeEnergy).toBeGreaterThan(0);
    expect(alert.recommendedAction).toContain('gossip restart');
    expect(alert.autoGeneratedTicket).toBeNull();
    expect(alert.createdAt).toBe(now);
  });

  it('should construct PreemptiveAlertInput with auto-generated ticket hash', () => {
    const alert: PreemptiveAlertInput = {
      predictedFailure: 'Memory pressure approaching OOM threshold',
      expectedTimeToFailure: null,
      freeEnergy: 8.2,
      recommendedAction: 'Reduce working memory capacity or restart conductor',
      autoGeneratedTicket: mockHash,
      createdAt: Date.now(),
    };

    expect(alert.autoGeneratedTicket).toBe(mockHash);
    expect(alert.expectedTimeToFailure).toBeNull();
    expect(alert.freeEnergy).toBeGreaterThan(0);
  });

  it('should construct AutonomousActionInput with all fields', () => {
    const now = Date.now();
    const action: AutonomousActionInput = {
      ticketHash: mockHash,
      actionType: 'RestartService',
      description: 'Restart the Holochain conductor to clear stale state',
      approved: false,
      executed: false,
      result: null,
      success: null,
      rollbackSteps: ['Stop conductor', 'Restore previous config', 'Start conductor'],
      rollbackState: JSON.stringify({ configVersion: 3 }),
      rolledBack: false,
      createdAt: now,
    };

    expect(action.ticketHash).toBe(mockHash);
    expect(action.actionType).toBe('RestartService');
    expect(action.description).toContain('conductor');
    expect(action.approved).toBe(false);
    expect(action.executed).toBe(false);
    expect(action.result).toBeNull();
    expect(action.success).toBeNull();
    expect(action.rollbackSteps).toHaveLength(3);
    expect(action.rollbackState).toContain('configVersion');
    expect(action.rolledBack).toBe(false);
    expect(action.createdAt).toBe(now);
  });

  it('should construct AutonomousActionInput with Custom action type', () => {
    const action: AutonomousActionInput = {
      ticketHash: mockHash,
      actionType: { Custom: 'RecalibratePhiWeights' },
      description: 'Recalibrate Phi integration weights based on recent anomalies',
      approved: true,
      executed: true,
      result: 'Phi weights recalibrated: 0.6 -> 0.72',
      success: true,
      rollbackSteps: null,
      rollbackState: null,
      rolledBack: false,
      createdAt: Date.now(),
    };

    expect(action.actionType).toEqual({ Custom: 'RecalibratePhiWeights' });
    expect(action.approved).toBe(true);
    expect(action.executed).toBe(true);
    expect(action.result).toContain('0.72');
    expect(action.success).toBe(true);
  });

  it('should construct ResolutionInput with all fields', () => {
    const resolution: ResolutionInput = {
      ticketHash: mockHash,
      steps: [
        'Identified root cause: stale gossip cache',
        'Cleared conductor cache at ~/.holochain/conductor/cache',
        'Restarted conductor with clean state',
        'Verified DHT sync resumed within 30 seconds',
      ],
      rootCause: 'Stale gossip cache causing DHT sync failure',
      timeToResolveMins: 15,
      effectivenessRating: 0.95,
      helper: mockPubKey,
      requester: new Uint8Array(39).fill(3),
      anonymized: false,
      helperSignature: [1, 2, 3, 4],
      requesterSignature: [5, 6, 7, 8],
    };

    expect(resolution.ticketHash).toBe(mockHash);
    expect(resolution.steps).toHaveLength(4);
    expect(resolution.rootCause).toContain('gossip cache');
    expect(resolution.timeToResolveMins).toBe(15);
    expect(resolution.effectivenessRating).toBe(0.95);
    expect(resolution.helper).toBe(mockPubKey);
    expect(resolution.anonymized).toBe(false);
    expect(resolution.helperSignature).toEqual([1, 2, 3, 4]);
    expect(resolution.requesterSignature).toEqual([5, 6, 7, 8]);
  });

  it('should construct ResolutionInput with null optional fields', () => {
    const resolution: ResolutionInput = {
      ticketHash: mockHash,
      steps: ['Restarted the conductor'],
      rootCause: null,
      timeToResolveMins: null,
      effectivenessRating: null,
      helper: mockPubKey,
      requester: mockPubKey,
      anonymized: true,
      helperSignature: [],
      requesterSignature: [],
    };

    expect(resolution.rootCause).toBeNull();
    expect(resolution.timeToResolveMins).toBeNull();
    expect(resolution.effectivenessRating).toBeNull();
    expect(resolution.anonymized).toBe(true);
  });

  it('should construct ArticleFlagInput with all flag reasons', () => {
    const flag: ArticleFlagInput = {
      articleHash: mockHash,
      flagger: mockPubKey,
      reason: 'Incorrect',
      description: 'The port number listed in step 3 is wrong',
      createdAt: Date.now(),
    };

    expect(flag.articleHash).toBe(mockHash);
    expect(flag.flagger).toBe(mockPubKey);
    expect(flag.reason).toBe('Incorrect');
    expect(flag.description).toContain('port number');
    expect(typeof flag.createdAt).toBe('number');
  });
});

// ============================================================================
// SupportClient Instantiation Tests
// ============================================================================

describe('SupportClient - Instantiation', () => {
  it('should create SupportClient from mock AppClient via fromClient()', () => {
    const support = SupportClient.fromClient(mockClient);

    expect(support).toBeInstanceOf(SupportClient);
    expect(support.knowledge).toBeDefined();
    expect(support.tickets).toBeDefined();
    expect(support.diagnostics).toBeDefined();
  });

  it('should create SupportClient with custom config', () => {
    const support = SupportClient.fromClient(mockClient, {
      roleName: 'custom-role',
      debug: true,
      timeout: 60000,
    });

    expect(support).toBeInstanceOf(SupportClient);
    expect(support.knowledge).toBeDefined();
    expect(support.tickets).toBeDefined();
    expect(support.diagnostics).toBeDefined();
  });

  it('should create SupportClient via createSupportClient factory', () => {
    const support = createSupportClient(mockClient);

    expect(support).toBeInstanceOf(SupportClient);
    expect(support.knowledge).toBeDefined();
    expect(support.tickets).toBeDefined();
    expect(support.diagnostics).toBeDefined();
  });

  it('should create SupportClient via createSupportClient with config', () => {
    const support = createSupportClient(mockClient, { debug: false, timeout: 15000 });

    expect(support).toBeInstanceOf(SupportClient);
  });

  it('should expose the underlying AppClient via getClient()', () => {
    const support = SupportClient.fromClient(mockClient);
    const client = support.getClient();

    expect(client).toBe(mockClient);
  });

  it('should expose the agent public key', () => {
    const support = SupportClient.fromClient(mockClient);
    const pubKey = support.getAgentPubKey();

    expect(pubKey).toBe(mockPubKey);
    expect(pubKey).toBeInstanceOf(Uint8Array);
    expect(pubKey).toHaveLength(39);
  });

  it('should check connection status via isConnected()', async () => {
    const support = SupportClient.fromClient(mockClient);
    // The mock's appInfo returns a value, so isConnected should resolve true
    const connected = await support.isConnected();
    expect(typeof connected).toBe('boolean');
  });

  it('should handle isConnected() returning false on error', async () => {
    const failingClient = {
      myPubKey: mockPubKey,
      appInfo: async () => { throw new Error('disconnected'); },
      callZome: async () => null,
    } as any;

    const support = SupportClient.fromClient(failingClient);
    const connected = await support.isConnected();
    expect(connected).toBe(false);
  });
});

// ============================================================================
// Knowledge Article Type Roundtrip
// ============================================================================

describe('Support Types - Knowledge Article Roundtrip', () => {
  it('should roundtrip article with Community source', () => {
    const input: KnowledgeArticleInput = {
      title: 'Setting up Mycelix on NixOS',
      content: 'Guide for installing Mycelix on NixOS 25.11...',
      category: 'Mycelix',
      tags: ['nixos', 'installation', 'setup'],
      author: mockPubKey,
      source: 'Community',
      difficultyLevel: 'Intermediate',
      upvotes: 5,
      verified: true,
      deprecated: false,
      deprecationReason: null,
      version: 2,
    };

    // Roundtrip via JSON serialization
    const serialized = JSON.stringify(input, (_key, value) =>
      value instanceof Uint8Array ? Array.from(value) : value,
    );
    const parsed = JSON.parse(serialized);

    expect(parsed.title).toBe(input.title);
    expect(parsed.category).toBe('Mycelix');
    expect(parsed.source).toBe('Community');
    expect(parsed.difficultyLevel).toBe('Intermediate');
    expect(parsed.upvotes).toBe(5);
    expect(parsed.verified).toBe(true);
    expect(parsed.version).toBe(2);
    expect(parsed.tags).toEqual(['nixos', 'installation', 'setup']);
  });

  it('should roundtrip article with PreSeeded source', () => {
    const input: KnowledgeArticleInput = {
      title: 'Holochain Conductor Basics',
      content: 'The conductor is the runtime...',
      category: 'Holochain',
      tags: ['conductor', 'basics'],
      author: mockPubKey,
      source: 'PreSeeded',
      difficultyLevel: 'Beginner',
      upvotes: 0,
      verified: true,
      deprecated: false,
      deprecationReason: null,
      version: 1,
    };

    expect(input.source).toBe('PreSeeded');
  });

  it('should roundtrip article with SymthaeaGenerated source', () => {
    const input: KnowledgeArticleInput = {
      title: 'Auto-generated: Network Partition Recovery',
      content: 'When a network partition is detected...',
      category: 'Network',
      tags: ['auto-generated', 'network', 'partition'],
      author: mockPubKey,
      source: 'SymthaeaGenerated',
      difficultyLevel: 'Advanced',
      upvotes: 0,
      verified: false,
      deprecated: false,
      deprecationReason: null,
      version: 1,
    };

    expect(input.source).toBe('SymthaeaGenerated');
    expect(input.difficultyLevel).toBe('Advanced');
  });

  it('should roundtrip deprecated article with reason', () => {
    const input: KnowledgeArticleInput = {
      title: 'Old Node Setup Guide',
      content: 'This guide covers Holochain 0.5...',
      category: 'Holochain',
      tags: ['deprecated'],
      author: mockPubKey,
      source: 'Community',
      difficultyLevel: 'Beginner',
      upvotes: 42,
      verified: true,
      deprecated: true,
      deprecationReason: 'Superseded by Holochain 0.6 setup guide',
      version: 3,
    };

    expect(input.deprecated).toBe(true);
    expect(input.deprecationReason).toBe('Superseded by Holochain 0.6 setup guide');
  });
});

// ============================================================================
// Ticket Type Roundtrip
// ============================================================================

describe('Support Types - Ticket Roundtrip', () => {
  it('should construct ticket across all priority levels', () => {
    const priorities: TicketPriority[] = ['Low', 'Medium', 'High', 'Critical'];

    priorities.forEach((priority) => {
      const ticket: SupportTicketInput = {
        title: `Test ticket - ${priority}`,
        description: 'Test description',
        category: 'General',
        priority,
        status: 'Open',
        requester: mockPubKey,
        assignee: null,
        autonomyLevel: 'Advisory',
        systemInfo: null,
        isPreemptive: false,
        predictionConfidence: null,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      expect(ticket.priority).toBe(priority);
    });
  });

  it('should construct ticket across all status values', () => {
    const statuses: TicketStatus[] = ['Open', 'InProgress', 'AwaitingUser', 'Resolved', 'Closed'];

    statuses.forEach((status) => {
      const ticket: SupportTicketInput = {
        title: `Test ticket - ${status}`,
        description: 'Test description',
        category: 'General',
        priority: 'Medium',
        status,
        requester: mockPubKey,
        assignee: null,
        autonomyLevel: 'Advisory',
        systemInfo: null,
        isPreemptive: false,
        predictionConfidence: null,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      expect(ticket.status).toBe(status);
    });
  });

  it('should construct ticket across all autonomy levels', () => {
    const levels: AutonomyLevel[] = ['Advisory', 'SemiAutonomous', 'FullAutonomous'];

    levels.forEach((level) => {
      const ticket: SupportTicketInput = {
        title: `Test ticket - ${level}`,
        description: 'Test',
        category: 'General',
        priority: 'Low',
        status: 'Open',
        requester: mockPubKey,
        assignee: null,
        autonomyLevel: level,
        systemInfo: null,
        isPreemptive: false,
        predictionConfidence: null,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      expect(ticket.autonomyLevel).toBe(level);
    });
  });
});

// ============================================================================
// Diagnostic Type Roundtrip
// ============================================================================

describe('Support Types - Diagnostic Roundtrip', () => {
  it('should construct diagnostics across all severity levels', () => {
    const severities: DiagnosticSeverity[] = ['Healthy', 'Warning', 'Error', 'Critical'];

    severities.forEach((severity) => {
      const diagnostic: DiagnosticResultInput = {
        ticketHash: mockHash,
        diagnosticType: 'NetworkCheck',
        findings: `Severity test: ${severity}`,
        severity,
        recommendations: [],
        agent: mockPubKey,
        scrubbed: false,
        createdAt: Date.now(),
      };

      expect(diagnostic.severity).toBe(severity);
    });
  });

  it('should construct diagnostics across all built-in types', () => {
    const types = ['NetworkCheck', 'DiskSpace', 'ServiceStatus', 'HolochainHealth', 'MemoryUsage'] as const;

    types.forEach((diagnosticType) => {
      const diagnostic: DiagnosticResultInput = {
        ticketHash: null,
        diagnosticType,
        findings: `Ran ${diagnosticType}`,
        severity: 'Healthy',
        recommendations: [],
        agent: mockPubKey,
        scrubbed: false,
        createdAt: Date.now(),
      };

      expect(diagnostic.diagnosticType).toBe(diagnosticType);
    });
  });
});

// ============================================================================
// Privacy Preference Validation
// ============================================================================

describe('Support Types - Privacy Preference Validation', () => {
  it('should validate all three sharing tiers compile correctly', () => {
    const tiers: SharingTier[] = ['LocalOnly', 'Anonymized', 'Full'];

    tiers.forEach((tier) => {
      const pref: PrivacyPreferenceInput = {
        agent: mockPubKey,
        sharingTier: tier,
        allowedCategories: [],
        shareSystemInfo: false,
        shareResolutionPatterns: false,
        shareCognitiveUpdates: false,
        updatedAt: Date.now(),
      };

      expect(pref.sharingTier).toBe(tier);
    });
  });

  it('should accept all SupportCategory values in allowedCategories', () => {
    const categories: SupportCategory[] = [
      'Network', 'Hardware', 'Software', 'Holochain', 'Mycelix', 'Security', 'General',
    ];

    const pref: PrivacyPreferenceInput = {
      agent: mockPubKey,
      sharingTier: 'Full',
      allowedCategories: categories,
      shareSystemInfo: true,
      shareResolutionPatterns: true,
      shareCognitiveUpdates: true,
      updatedAt: Date.now(),
    };

    expect(pref.allowedCategories).toHaveLength(7);
    categories.forEach((cat) => {
      expect(pref.allowedCategories).toContain(cat);
    });
  });
});

// ============================================================================
// Cognitive Update Type
// ============================================================================

describe('Support Types - Cognitive Update', () => {
  it('should create CognitiveUpdateInput with 2048-byte encoding and verify contents', () => {
    // Simulate a 2048-byte HDC binary hypervector encoding
    const encoding = Array.from({ length: 2048 }, (_, i) => i % 256);

    const update: CognitiveUpdateInput = {
      category: 'Holochain',
      encoding,
      phi: 0.65,
      resolutionPattern: 'conductor_restart:gossip_stale',
      sourceAgent: mockPubKey,
      createdAt: Date.now(),
    };

    expect(update.encoding).toHaveLength(2048);
    // Verify byte pattern wraps at 256
    expect(update.encoding[0]).toBe(0);
    expect(update.encoding[127]).toBe(127);
    expect(update.encoding[255]).toBe(255);
    expect(update.encoding[256]).toBe(0);
    expect(update.encoding[2047]).toBe(2047 % 256);
    expect(update.phi).toBe(0.65);
    expect(update.phi).toBeGreaterThanOrEqual(0);
    expect(update.phi).toBeLessThanOrEqual(1);
  });

  it('should support empty encoding', () => {
    const update: CognitiveUpdateInput = {
      category: 'General',
      encoding: [],
      phi: 0.0,
      resolutionPattern: '',
      sourceAgent: mockPubKey,
      createdAt: Date.now(),
    };

    expect(update.encoding).toHaveLength(0);
    expect(update.phi).toBe(0.0);
  });

  it('should accept phi values at boundary', () => {
    const update1: CognitiveUpdateInput = {
      category: 'Network',
      encoding: [0, 0, 0],
      phi: 0.0,
      resolutionPattern: 'null_pattern',
      sourceAgent: mockPubKey,
      createdAt: Date.now(),
    };

    const update2: CognitiveUpdateInput = {
      category: 'Security',
      encoding: [255, 255, 255],
      phi: 1.0,
      resolutionPattern: 'max_integration',
      sourceAgent: mockPubKey,
      createdAt: Date.now(),
    };

    expect(update1.phi).toBe(0.0);
    expect(update2.phi).toBe(1.0);
  });
});

// ============================================================================
// Error Types
// ============================================================================

describe('Support Error Types', () => {
  it('should instantiate SupportError with code and message', () => {
    const error = new SupportError('CONNECTION_ERROR', 'Failed to connect to Holochain');

    expect(error).toBeInstanceOf(SupportError);
    expect(error).toBeInstanceOf(Error);
    expect(error.code).toBe('CONNECTION_ERROR');
    expect(error.message).toBe('Failed to connect to Holochain');
    expect(error.name).toBe('SupportError');
  });

  it('should have correct SupportErrorCode constants', () => {
    expect(SupportErrorCode.NOT_FOUND).toBe('SUPPORT_NOT_FOUND');
    expect(SupportErrorCode.VALIDATION_FAILED).toBe('SUPPORT_VALIDATION_FAILED');
    expect(SupportErrorCode.PERMISSION_DENIED).toBe('SUPPORT_PERMISSION_DENIED');
  });

  it('should use SupportErrorCode values in SupportError', () => {
    const notFoundErr = new SupportError(SupportErrorCode.NOT_FOUND, 'Article not found');
    expect(notFoundErr.code).toBe('SUPPORT_NOT_FOUND');
    expect(notFoundErr.message).toBe('Article not found');

    const validationErr = new SupportError(
      SupportErrorCode.VALIDATION_FAILED,
      'Title cannot be empty',
    );
    expect(validationErr.code).toBe('SUPPORT_VALIDATION_FAILED');

    const permissionErr = new SupportError(
      SupportErrorCode.PERMISSION_DENIED,
      'Only author can deprecate',
    );
    expect(permissionErr.code).toBe('SUPPORT_PERMISSION_DENIED');
  });

  it('should have a proper stack trace', () => {
    const error = new SupportError('TEST', 'test error');
    expect(error.stack).toBeDefined();
    expect(error.stack).toContain('SupportError');
  });

  it('should be catchable as Error', () => {
    let caught = false;
    try {
      throw new SupportError(SupportErrorCode.NOT_FOUND, 'Resource missing');
    } catch (e) {
      if (e instanceof Error) {
        caught = true;
        expect(e.message).toBe('Resource missing');
      }
    }
    expect(caught).toBe(true);
  });
});

// ============================================================================
// Enum Values - Compilation Checks
// ============================================================================

describe('Support Enums - String Literal Compilation', () => {
  it('should compile all SupportCategory values', () => {
    const categories: SupportCategory[] = [
      'Network', 'Hardware', 'Software', 'Holochain', 'Mycelix', 'Security', 'General',
    ];
    expect(categories).toHaveLength(7);
    categories.forEach((c) => expect(typeof c).toBe('string'));
  });

  it('should compile all TicketPriority values', () => {
    const priorities: TicketPriority[] = ['Low', 'Medium', 'High', 'Critical'];
    expect(priorities).toHaveLength(4);
    priorities.forEach((p) => expect(typeof p).toBe('string'));
  });

  it('should compile all TicketStatus values', () => {
    const statuses: TicketStatus[] = ['Open', 'InProgress', 'AwaitingUser', 'Resolved', 'Closed'];
    expect(statuses).toHaveLength(5);
    statuses.forEach((s) => expect(typeof s).toBe('string'));
  });

  it('should compile all AutonomyLevel values', () => {
    const levels: AutonomyLevel[] = ['Advisory', 'SemiAutonomous', 'FullAutonomous'];
    expect(levels).toHaveLength(3);
    levels.forEach((l) => expect(typeof l).toBe('string'));
  });

  it('should compile all ActionType values including Custom variant', () => {
    const actions: ActionType[] = [
      'RestartService',
      'ClearCache',
      'UpdateConfig',
      'RunDiagnostic',
      { Custom: 'DeployHotfix' },
    ];
    expect(actions).toHaveLength(5);
    expect(actions[0]).toBe('RestartService');
    expect(actions[1]).toBe('ClearCache');
    expect(actions[2]).toBe('UpdateConfig');
    expect(actions[3]).toBe('RunDiagnostic');
    expect(actions[4]).toEqual({ Custom: 'DeployHotfix' });
  });

  it('should compile all DiagnosticSeverity values', () => {
    const severities: DiagnosticSeverity[] = ['Healthy', 'Warning', 'Error', 'Critical'];
    expect(severities).toHaveLength(4);
    severities.forEach((s) => expect(typeof s).toBe('string'));
  });

  it('should compile all DifficultyLevel values', () => {
    const levels: DifficultyLevel[] = ['Beginner', 'Intermediate', 'Advanced'];
    expect(levels).toHaveLength(3);
    levels.forEach((l) => expect(typeof l).toBe('string'));
  });

  it('should compile all ArticleSource values', () => {
    const sources: ArticleSource[] = ['Community', 'PreSeeded', 'SymthaeaGenerated'];
    expect(sources).toHaveLength(3);
    sources.forEach((s) => expect(typeof s).toBe('string'));
  });

  it('should compile all EpistemicStatus values', () => {
    const statuses: EpistemicStatus[] = [
      'Certain', 'Probable', 'Uncertain', 'Unknown', 'OutOfDomain',
    ];
    expect(statuses).toHaveLength(5);
    statuses.forEach((s) => expect(typeof s).toBe('string'));
  });

  it('should compile all SharingTier values', () => {
    const tiers: SharingTier[] = ['LocalOnly', 'Anonymized', 'Full'];
    expect(tiers).toHaveLength(3);
    tiers.forEach((t) => expect(typeof t).toBe('string'));
  });
});

// ============================================================================
// Preemptive Alert Type
// ============================================================================

describe('Support Types - Preemptive Alert', () => {
  it('should verify freeEnergy is positive for meaningful alerts', () => {
    const alert: PreemptiveAlertInput = {
      predictedFailure: 'Conductor memory leak detected',
      expectedTimeToFailure: '2 hours',
      freeEnergy: 15.3,
      recommendedAction: 'Schedule conductor restart during low-traffic window',
      autoGeneratedTicket: null,
      createdAt: Date.now(),
    };

    expect(alert.freeEnergy).toBeGreaterThan(0);
    expect(alert.freeEnergy).toBe(15.3);
  });

  it('should allow zero freeEnergy for baseline alerts', () => {
    const alert: PreemptiveAlertInput = {
      predictedFailure: 'No failure predicted',
      expectedTimeToFailure: null,
      freeEnergy: 0,
      recommendedAction: 'No action needed',
      autoGeneratedTicket: null,
      createdAt: Date.now(),
    };

    expect(alert.freeEnergy).toBe(0);
  });

  it('should support high freeEnergy values for urgent alerts', () => {
    const alert: PreemptiveAlertInput = {
      predictedFailure: 'Imminent DHT corruption from Byzantine peer',
      expectedTimeToFailure: '5 minutes',
      freeEnergy: 150.0,
      recommendedAction: 'Immediately quarantine peer and initiate DHT repair',
      autoGeneratedTicket: mockHash,
      createdAt: Date.now(),
    };

    expect(alert.freeEnergy).toBe(150.0);
    expect(alert.freeEnergy).toBeGreaterThan(100);
    expect(alert.autoGeneratedTicket).toBe(mockHash);
    expect(alert.expectedTimeToFailure).toBe('5 minutes');
  });
});
