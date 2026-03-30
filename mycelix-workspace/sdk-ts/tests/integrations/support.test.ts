// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Community Support Integration Tests
 *
 * Tests for SupportClient — the domain-specific SDK client for the
 * community support zomes within the mycelix-commons cluster DNA.
 * Covers knowledge (articles, resolutions, reputation), tickets
 * (support tickets, escalation, preemptive alerts, satisfaction surveys),
 * and diagnostics (system checks, helpers, cognitive updates, privacy).
 *
 * All calls are dispatched through the commons_care role to the
 * support_knowledge, support_tickets, and support_diagnostics coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  SupportClient,
  createSupportClient,
  SUPPORT_ZOMES,
  type SupportCategory,
  type TicketPriority,
  type TicketStatus,
  type AutonomyLevel,
  type ActionType,
  type SharingTier,
  type DiagnosticType,
  type DiagnosticSeverity,
  type DifficultyLevel,
  type ArticleSource,
  type FlagReason,
  type EscalationLevel,
  type EpistemicStatus,
  type LinkReason,
  type KnowledgeArticle,
  type UpdateArticleInput,
  type DeprecateInput,
  type ArticleFlag,
  type Resolution,
  type LinkArticleInput,
  type SupportTicket,
  type UpdateTicketInput,
  type TicketComment,
  type AutonomousAction,
  type UndoAction,
  type PreemptiveAlert,
  type PromoteAlertInput,
  type EscalateInput,
  type SatisfactionSurvey,
  type DiagnosticResult,
  type PrivacyPreference,
  type HelperProfile,
  type UpdateAvailInput,
  type CognitiveUpdate,
} from '../../src/integrations/support/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

// ============================================================================
// Test helpers
// ============================================================================

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function makeArticle(overrides: Partial<KnowledgeArticle> = {}): KnowledgeArticle {
  return {
    title: 'How to Reset Your Holochain Node',
    content: 'Step 1: Stop the conductor. Step 2: Clear the database...',
    category: 'Holochain',
    tags: ['holochain', 'reset', 'troubleshooting'],
    author: fakeAgentKey(),
    source: 'Community',
    difficulty_level: 'Beginner',
    upvotes: 0,
    verified: false,
    deprecated: false,
    deprecation_reason: null,
    version: 1,
    ...overrides,
  };
}

function makeTicket(overrides: Partial<SupportTicket> = {}): SupportTicket {
  return {
    title: 'Network sync issue',
    description: 'My node is not syncing with peers',
    category: 'Network',
    priority: 'Medium',
    status: 'Open',
    requester: fakeAgentKey(),
    assignee: null,
    autonomy_level: 'Advisory',
    system_info: 'Holochain 0.6.0, NixOS 25.11',
    is_preemptive: false,
    prediction_confidence: null,
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeDiagnostic(overrides: Partial<DiagnosticResult> = {}): DiagnosticResult {
  return {
    ticket_hash: fakeActionHash(),
    diagnostic_type: 'NetworkCheck',
    findings: 'All peers reachable, latency within normal range',
    severity: 'Healthy',
    recommendations: [],
    agent: fakeAgentKey(),
    scrubbed: false,
    created_at: Date.now(),
    ...overrides,
  };
}

function makeHelperProfile(overrides: Partial<HelperProfile> = {}): HelperProfile {
  return {
    agent: fakeAgentKey(),
    expertise_categories: ['Network', 'Holochain'],
    max_concurrent: 3,
    difficulty_preference: 'Intermediate',
    available: true,
    created_at: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Constants
// ============================================================================

describe('Support Constants', () => {
  it('should export all 3 support zome names', () => {
    expect(SUPPORT_ZOMES).toEqual([
      'support_knowledge',
      'support_tickets',
      'support_diagnostics',
    ]);
    expect(SUPPORT_ZOMES).toHaveLength(3);
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Support Types', () => {
  describe('KnowledgeArticle', () => {
    it('should construct a valid article', () => {
      const article = makeArticle();
      expect(article.title).toBe('How to Reset Your Holochain Node');
      expect(article.category).toBe('Holochain');
      expect(article.tags).toHaveLength(3);
      expect(article.verified).toBe(false);
      expect(article.deprecated).toBe(false);
    });

    it('should accept all SupportCategory variants', () => {
      const categories: SupportCategory[] = [
        'Network', 'Hardware', 'Software', 'Holochain', 'Mycelix', 'Security', 'General',
      ];
      categories.forEach((c) => {
        const a = makeArticle({ category: c });
        expect(a.category).toBe(c);
      });
    });

    it('should accept all ArticleSource variants', () => {
      const sources: ArticleSource[] = ['Community', 'PreSeeded', 'SymthaeaGenerated'];
      sources.forEach((s) => {
        const a = makeArticle({ source: s });
        expect(a.source).toBe(s);
      });
    });

    it('should accept all DifficultyLevel variants', () => {
      const levels: DifficultyLevel[] = ['Beginner', 'Intermediate', 'Advanced'];
      levels.forEach((l) => {
        const a = makeArticle({ difficulty_level: l });
        expect(a.difficulty_level).toBe(l);
      });
    });
  });

  describe('SupportTicket', () => {
    it('should construct a valid ticket', () => {
      const ticket = makeTicket();
      expect(ticket.title).toBe('Network sync issue');
      expect(ticket.priority).toBe('Medium');
      expect(ticket.status).toBe('Open');
      expect(ticket.is_preemptive).toBe(false);
    });

    it('should accept all TicketPriority variants', () => {
      const priorities: TicketPriority[] = ['Low', 'Medium', 'High', 'Critical'];
      priorities.forEach((p) => {
        const t = makeTicket({ priority: p });
        expect(t.priority).toBe(p);
      });
    });

    it('should accept all TicketStatus variants', () => {
      const statuses: TicketStatus[] = ['Open', 'InProgress', 'AwaitingUser', 'Resolved', 'Closed'];
      statuses.forEach((s) => {
        const t = makeTicket({ status: s });
        expect(t.status).toBe(s);
      });
    });

    it('should accept all AutonomyLevel variants', () => {
      const levels: AutonomyLevel[] = ['Advisory', 'SemiAutonomous', 'FullAutonomous'];
      levels.forEach((l) => {
        const t = makeTicket({ autonomy_level: l });
        expect(t.autonomy_level).toBe(l);
      });
    });

    it('should handle nullable assignee and system_info', () => {
      const ticket = makeTicket({ assignee: null, system_info: null });
      expect(ticket.assignee).toBeNull();
      expect(ticket.system_info).toBeNull();
    });
  });

  describe('DiagnosticResult', () => {
    it('should construct a valid diagnostic', () => {
      const diag = makeDiagnostic();
      expect(diag.diagnostic_type).toBe('NetworkCheck');
      expect(diag.severity).toBe('Healthy');
      expect(diag.scrubbed).toBe(false);
    });

    it('should accept all DiagnosticSeverity variants', () => {
      const severities: DiagnosticSeverity[] = ['Healthy', 'Warning', 'Error', 'Critical'];
      severities.forEach((s) => {
        const d = makeDiagnostic({ severity: s });
        expect(d.severity).toBe(s);
      });
    });

    it('should handle nullable ticket_hash', () => {
      const diag = makeDiagnostic({ ticket_hash: null });
      expect(diag.ticket_hash).toBeNull();
    });
  });

  describe('HelperProfile', () => {
    it('should construct a valid helper profile', () => {
      const profile = makeHelperProfile();
      expect(profile.expertise_categories).toHaveLength(2);
      expect(profile.max_concurrent).toBe(3);
      expect(profile.available).toBe(true);
    });
  });

  describe('PreemptiveAlert', () => {
    it('should construct with nullable fields', () => {
      const alert: PreemptiveAlert = {
        predicted_failure: 'Disk space running low',
        expected_time_to_failure: '48 hours',
        free_energy: 0.85,
        recommended_action: 'Clear old logs',
        auto_generated_ticket: null,
        created_at: Date.now(),
      };
      expect(alert.free_energy).toBe(0.85);
      expect(alert.auto_generated_ticket).toBeNull();
    });
  });

  describe('SatisfactionSurvey', () => {
    it('should construct a valid survey', () => {
      const survey: SatisfactionSurvey = {
        ticket_hash: fakeActionHash(),
        respondent: fakeAgentKey(),
        rating: 5,
        comment: 'Excellent support',
        would_recommend: true,
        submitted_at: Date.now(),
      };
      expect(survey.rating).toBe(5);
      expect(survey.would_recommend).toBe(true);
    });
  });

  describe('CognitiveUpdate', () => {
    it('should construct a valid cognitive update', () => {
      const update: CognitiveUpdate = {
        category: 'Security',
        encoding: new Uint8Array(16384).fill(0x01),
        phi: 0.42,
        resolution_pattern: 'Firewall rule adjustment',
        source_agent: fakeAgentKey(),
        created_at: Date.now(),
      };
      expect(update.phi).toBe(0.42);
      expect(update.encoding).toHaveLength(16384);
    });
  });
});

// ============================================================================
// SupportClient — Factory
// ============================================================================

describe('SupportClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let support: SupportClient;

  beforeEach(() => {
    client = createMockClient();
    support = createSupportClient(client);
  });

  describe('factory', () => {
    it('should create via createSupportClient', () => {
      expect(support).toBeInstanceOf(SupportClient);
    });

    it('should also be constructable directly', () => {
      const direct = new SupportClient(client);
      expect(direct).toBeInstanceOf(SupportClient);
    });
  });

  // ==========================================================================
  // Knowledge zome
  // ==========================================================================

  describe('Knowledge (support_knowledge)', () => {
    describe('createArticle', () => {
      it('should call support_knowledge.create_article with correct params', async () => {
        const article = makeArticle();
        client.callZome.mockResolvedValue(fakeActionHash());

        await support.createArticle(article);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'create_article',
          payload: article,
        });
      });
    });

    describe('updateArticle', () => {
      it('should call support_knowledge.update_article with input', async () => {
        const input: UpdateArticleInput = {
          original_hash: fakeActionHash(),
          updated: makeArticle({ version: 2 }),
        };
        client.callZome.mockResolvedValue(fakeActionHash());

        await support.updateArticle(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'update_article',
          payload: input,
        });
      });
    });

    describe('deprecateArticle', () => {
      it('should call support_knowledge.deprecate_article with input', async () => {
        const input: DeprecateInput = {
          article_hash: fakeActionHash(),
          reason: 'Superseded by updated guide',
        };
        await support.deprecateArticle(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'deprecate_article',
          payload: input,
        });
      });
    });

    describe('getArticle', () => {
      it('should call support_knowledge.get_article with hash', async () => {
        const hash = fakeActionHash();
        client.callZome.mockResolvedValue(makeArticle());

        await support.getArticle(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'get_article',
          payload: hash,
        });
      });
    });

    describe('searchByCategory', () => {
      it('should call support_knowledge.search_by_category', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await support.searchByCategory('Network');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'search_by_category',
          payload: 'Network',
        });
        expect(result).toEqual([]);
      });
    });

    describe('searchByTag', () => {
      it('should call support_knowledge.search_by_tag', async () => {
        client.callZome.mockResolvedValue([]);

        await support.searchByTag('troubleshooting');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'search_by_tag',
          payload: 'troubleshooting',
        });
      });
    });

    describe('listRecentArticles', () => {
      it('should call support_knowledge.list_recent_articles with null', async () => {
        client.callZome.mockResolvedValue([makeArticle()]);

        const result = await support.listRecentArticles();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'list_recent_articles',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('upvoteArticle', () => {
      it('should call support_knowledge.upvote_article with hash', async () => {
        const hash = fakeActionHash();
        await support.upvoteArticle(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'upvote_article',
          payload: hash,
        });
      });
    });

    describe('verifyArticle', () => {
      it('should call support_knowledge.verify_article with hash', async () => {
        const hash = fakeActionHash();
        await support.verifyArticle(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'verify_article',
          payload: hash,
        });
      });
    });

    describe('createResolution', () => {
      it('should call support_knowledge.create_resolution with resolution data', async () => {
        const resolution: Resolution = {
          ticket_hash: fakeActionHash(),
          steps: ['Restart conductor', 'Clear cache', 'Reconnect'],
          root_cause: 'Stale peer connections',
          time_to_resolve_mins: 15,
          effectiveness_rating: 4.5,
          helper: fakeAgentKey(),
          requester: new Uint8Array(39).fill(0xbb),
          anonymized: false,
          helper_signature: new Uint8Array(64).fill(0x01),
          requester_signature: new Uint8Array(64).fill(0x02),
        };
        await support.createResolution(resolution);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'create_resolution',
          payload: resolution,
        });
      });
    });

    describe('flagArticle', () => {
      it('should call support_knowledge.flag_article with flag data', async () => {
        const flag: ArticleFlag = {
          article_hash: fakeActionHash(),
          flagger: fakeAgentKey(),
          reason: 'Outdated',
          description: 'Instructions reference Holochain 0.4',
          created_at: Date.now(),
        };
        await support.flagArticle(flag);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'flag_article',
          payload: flag,
        });
      });

      it('should accept all FlagReason variants', async () => {
        const reasons: FlagReason[] = ['Harmful', 'Incorrect', 'Outdated', 'Spam'];
        for (const reason of reasons) {
          const flag: ArticleFlag = {
            article_hash: fakeActionHash(),
            flagger: fakeAgentKey(),
            reason,
            description: `Flagged as ${reason}`,
            created_at: Date.now(),
          };
          await support.flagArticle(flag);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.reason).toBe(reason);
        }
      });
    });

    describe('getFlags', () => {
      it('should call support_knowledge.get_flags with article hash', async () => {
        client.callZome.mockResolvedValue([]);
        const result = await support.getFlags(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'get_flags',
          payload: fakeActionHash(),
        });
        expect(result).toEqual([]);
      });
    });

    describe('getAgentReputation', () => {
      it('should call support_knowledge.get_agent_reputation with agent key', async () => {
        const agent = fakeAgentKey();
        client.callZome.mockResolvedValue([]);

        await support.getAgentReputation(agent);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'get_agent_reputation',
          payload: agent,
        });
      });
    });

    describe('linkArticleToTicket', () => {
      it('should call support_knowledge.link_article_to_ticket with input', async () => {
        const input: LinkArticleInput = {
          article_hash: fakeActionHash(),
          ticket_hash: new Uint8Array(39).fill(0x01),
          link_reason: 'SuggestedFAQ',
        };
        await support.linkArticleToTicket(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'link_article_to_ticket',
          payload: input,
        });
      });

      it('should accept all LinkReason variants', async () => {
        const reasons: LinkReason[] = ['SuggestedFAQ', 'DuplicateResolution', 'RelatedKnowledge'];
        for (const reason of reasons) {
          const input: LinkArticleInput = {
            article_hash: fakeActionHash(),
            ticket_hash: fakeActionHash(),
            link_reason: reason,
          };
          await support.linkArticleToTicket(input);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.link_reason).toBe(reason);
        }
      });
    });

    describe('getSuggestedArticles', () => {
      it('should call support_knowledge.get_suggested_articles with ticket hash', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getSuggestedArticles(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'get_suggested_articles',
          payload: fakeActionHash(),
        });
      });
    });

    describe('getTicketsForArticle', () => {
      it('should call support_knowledge.get_tickets_for_article with article hash', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getTicketsForArticle(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_knowledge',
          fn_name: 'get_tickets_for_article',
          payload: fakeActionHash(),
        });
      });
    });
  });

  // ==========================================================================
  // Tickets zome
  // ==========================================================================

  describe('Tickets (support_tickets)', () => {
    describe('createTicket', () => {
      it('should call support_tickets.create_ticket with ticket data', async () => {
        const ticket = makeTicket();
        client.callZome.mockResolvedValue(fakeActionHash());

        await support.createTicket(ticket);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'create_ticket',
          payload: ticket,
        });
      });

      it('should pass preemptive ticket with confidence', async () => {
        const ticket = makeTicket({
          is_preemptive: true,
          prediction_confidence: 0.87,
          priority: 'High',
        });
        await support.createTicket(ticket);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.is_preemptive).toBe(true);
        expect(lastCall.payload.prediction_confidence).toBe(0.87);
      });
    });

    describe('updateTicket', () => {
      it('should call support_tickets.update_ticket with input', async () => {
        const input: UpdateTicketInput = {
          original_hash: fakeActionHash(),
          updated: makeTicket({ status: 'InProgress' }),
        };
        await support.updateTicket(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'update_ticket',
          payload: input,
        });
      });
    });

    describe('getTicket', () => {
      it('should call support_tickets.get_ticket with hash', async () => {
        const hash = fakeActionHash();
        client.callZome.mockResolvedValue(makeTicket());

        await support.getTicket(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'get_ticket',
          payload: hash,
        });
      });
    });

    describe('listTicketsByStatus', () => {
      it('should call support_tickets.list_tickets_by_status', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await support.listTicketsByStatus('Open');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'list_tickets_by_status',
          payload: 'Open',
        });
        expect(result).toEqual([]);
      });
    });

    describe('listMyTickets', () => {
      it('should call support_tickets.list_my_tickets with null', async () => {
        client.callZome.mockResolvedValue([makeTicket()]);

        const result = await support.listMyTickets();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'list_my_tickets',
          payload: null,
        });
        expect(result).toHaveLength(1);
      });
    });

    describe('closeTicket', () => {
      it('should call support_tickets.close_ticket with hash', async () => {
        const hash = fakeActionHash();
        await support.closeTicket(hash);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'close_ticket',
          payload: hash,
        });
      });
    });

    describe('addComment', () => {
      it('should call support_tickets.add_comment with comment data', async () => {
        const comment: TicketComment = {
          ticket_hash: fakeActionHash(),
          author: fakeAgentKey(),
          content: 'Have you tried restarting?',
          is_symthaea_response: false,
          confidence: null,
          epistemic_status: null,
        };
        await support.addComment(comment);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'add_comment',
          payload: comment,
        });
      });

      it('should handle Symthaea AI response with epistemic status', async () => {
        const comment: TicketComment = {
          ticket_hash: fakeActionHash(),
          author: fakeAgentKey(),
          content: 'Based on similar patterns, try clearing the conductor DB',
          is_symthaea_response: true,
          confidence: 0.82,
          epistemic_status: 'Probable',
        };
        await support.addComment(comment);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.is_symthaea_response).toBe(true);
        expect(lastCall.payload.confidence).toBe(0.82);
        expect(lastCall.payload.epistemic_status).toBe('Probable');
      });

      it('should accept all EpistemicStatus variants', async () => {
        const statuses: EpistemicStatus[] = ['Certain', 'Probable', 'Uncertain', 'Unknown', 'OutOfDomain'];
        for (const s of statuses) {
          const comment: TicketComment = {
            ticket_hash: fakeActionHash(),
            author: fakeAgentKey(),
            content: 'test',
            is_symthaea_response: true,
            confidence: 0.5,
            epistemic_status: s,
          };
          await support.addComment(comment);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.epistemic_status).toBe(s);
        }
      });
    });

    describe('getComments', () => {
      it('should call support_tickets.get_comments with ticket hash', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getComments(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'get_comments',
          payload: fakeActionHash(),
        });
      });
    });

    describe('proposeAction', () => {
      it('should call support_tickets.propose_action with action data', async () => {
        const action: AutonomousAction = {
          ticket_hash: fakeActionHash(),
          action_type: 'ClearCache',
          description: 'Clear Holochain conductor cache',
          approved: false,
          executed: false,
          result: null,
          success: null,
          rollback_steps: ['Restore backup'],
          rollback_state: null,
          rolled_back: false,
          created_at: Date.now(),
        };
        await support.proposeAction(action);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'propose_action',
          payload: action,
        });
      });

      it('should handle custom ActionType', async () => {
        const action: AutonomousAction = {
          ticket_hash: fakeActionHash(),
          action_type: { Custom: 'rebuild_indexes' },
          description: 'Rebuild DHT indexes',
          approved: false,
          executed: false,
          result: null,
          success: null,
          rollback_steps: null,
          rollback_state: null,
          rolled_back: false,
          created_at: Date.now(),
        };
        await support.proposeAction(action);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.action_type).toEqual({ Custom: 'rebuild_indexes' });
      });
    });

    describe('approveAction', () => {
      it('should call support_tickets.approve_action with hash', async () => {
        await support.approveAction(fakeActionHash());
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'approve_action',
          payload: fakeActionHash(),
        });
      });
    });

    describe('executeAction', () => {
      it('should call support_tickets.execute_action with hash', async () => {
        await support.executeAction(fakeActionHash());
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'execute_action',
          payload: fakeActionHash(),
        });
      });
    });

    describe('rollbackAction', () => {
      it('should call support_tickets.rollback_action with hash', async () => {
        await support.rollbackAction(fakeActionHash());
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'rollback_action',
          payload: fakeActionHash(),
        });
      });
    });

    describe('createUndo', () => {
      it('should call support_tickets.create_undo with undo data', async () => {
        const undo: UndoAction = {
          original_action_hash: fakeActionHash(),
          reason: 'Action caused unexpected side effects',
          rollback_result: 'Cache restored to previous state',
          created_at: Date.now(),
        };
        await support.createUndo(undo);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'create_undo',
          payload: undo,
        });
      });
    });

    describe('createPreemptiveAlert', () => {
      it('should call support_tickets.create_preemptive_alert with alert data', async () => {
        const alert: PreemptiveAlert = {
          predicted_failure: 'Disk space below 10%',
          expected_time_to_failure: '24 hours',
          free_energy: 0.92,
          recommended_action: 'Archive old DHT entries',
          auto_generated_ticket: null,
          created_at: Date.now(),
        };
        await support.createPreemptiveAlert(alert);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'create_preemptive_alert',
          payload: alert,
        });
      });
    });

    describe('listPreemptiveAlerts', () => {
      it('should call support_tickets.list_preemptive_alerts with null', async () => {
        client.callZome.mockResolvedValue([]);
        const result = await support.listPreemptiveAlerts();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'list_preemptive_alerts',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('promoteAlertToTicket', () => {
      it('should call support_tickets.promote_alert_to_ticket with input', async () => {
        const input: PromoteAlertInput = {
          alert_hash: fakeActionHash(),
          ticket: makeTicket({ priority: 'High', is_preemptive: true }),
        };
        await support.promoteAlertToTicket(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'promote_alert_to_ticket',
          payload: input,
        });
      });
    });

    describe('escalateTicket', () => {
      it('should call support_tickets.escalate_ticket with input', async () => {
        const input: EscalateInput = {
          ticket_hash: fakeActionHash(),
          from_level: 'Tier1',
          to_level: 'Tier2',
          reason: 'Issue requires specialized expertise',
        };
        await support.escalateTicket(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'escalate_ticket',
          payload: input,
        });
      });

      it('should accept all EscalationLevel variants', async () => {
        const levels: EscalationLevel[] = ['Tier1', 'Tier2', 'Management', 'Emergency'];
        for (const level of levels) {
          const input: EscalateInput = {
            ticket_hash: fakeActionHash(),
            from_level: 'Tier1',
            to_level: level,
            reason: `Escalate to ${level}`,
          };
          await support.escalateTicket(input);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.to_level).toBe(level);
        }
      });
    });

    describe('getEscalationHistory', () => {
      it('should call support_tickets.get_escalation_history with ticket hash', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getEscalationHistory(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'get_escalation_history',
          payload: fakeActionHash(),
        });
      });
    });

    describe('submitSatisfaction', () => {
      it('should call support_tickets.submit_satisfaction with survey data', async () => {
        const survey: SatisfactionSurvey = {
          ticket_hash: fakeActionHash(),
          respondent: fakeAgentKey(),
          rating: 4,
          comment: 'Quick resolution',
          would_recommend: true,
          submitted_at: Date.now(),
        };
        await support.submitSatisfaction(survey);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'submit_satisfaction',
          payload: survey,
        });
      });

      it('should handle null comment', async () => {
        const survey: SatisfactionSurvey = {
          ticket_hash: fakeActionHash(),
          respondent: fakeAgentKey(),
          rating: 3,
          comment: null,
          would_recommend: false,
          submitted_at: Date.now(),
        };
        await support.submitSatisfaction(survey);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.comment).toBeNull();
      });
    });

    describe('getTicketSatisfaction', () => {
      it('should call support_tickets.get_ticket_satisfaction with ticket hash', async () => {
        await support.getTicketSatisfaction(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_tickets',
          fn_name: 'get_ticket_satisfaction',
          payload: fakeActionHash(),
        });
      });
    });
  });

  // ==========================================================================
  // Diagnostics zome
  // ==========================================================================

  describe('Diagnostics (support_diagnostics)', () => {
    describe('runDiagnostic', () => {
      it('should call support_diagnostics.run_diagnostic with data', async () => {
        const diag = makeDiagnostic();
        client.callZome.mockResolvedValue(fakeActionHash());

        await support.runDiagnostic(diag);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'run_diagnostic',
          payload: diag,
        });
      });

      it('should handle custom DiagnosticType', async () => {
        const diag = makeDiagnostic({
          diagnostic_type: { Custom: 'conductor_health_deep' },
        });
        await support.runDiagnostic(diag);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.diagnostic_type).toEqual({ Custom: 'conductor_health_deep' });
      });
    });

    describe('getDiagnostic', () => {
      it('should call support_diagnostics.get_diagnostic with hash', async () => {
        await support.getDiagnostic(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_diagnostic',
          payload: fakeActionHash(),
        });
      });
    });

    describe('listDiagnostics', () => {
      it('should call support_diagnostics.list_diagnostics with null', async () => {
        client.callZome.mockResolvedValue([]);
        const result = await support.listDiagnostics();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'list_diagnostics',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('setPrivacyPreference', () => {
      it('should call support_diagnostics.set_privacy_preference with data', async () => {
        const pref: PrivacyPreference = {
          agent: fakeAgentKey(),
          sharing_tier: 'Anonymized',
          allowed_categories: ['Network', 'Holochain'],
          share_system_info: true,
          share_resolution_patterns: true,
          share_cognitive_updates: false,
          updated_at: Date.now(),
        };
        await support.setPrivacyPreference(pref);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'set_privacy_preference',
          payload: pref,
        });
      });

      it('should accept all SharingTier variants', async () => {
        const tiers: SharingTier[] = ['LocalOnly', 'Anonymized', 'Full'];
        for (const tier of tiers) {
          const pref: PrivacyPreference = {
            agent: fakeAgentKey(),
            sharing_tier: tier,
            allowed_categories: [],
            share_system_info: false,
            share_resolution_patterns: false,
            share_cognitive_updates: false,
            updated_at: Date.now(),
          };
          await support.setPrivacyPreference(pref);
          const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
          expect(lastCall.payload.sharing_tier).toBe(tier);
        }
      });
    });

    describe('getPrivacyPreference', () => {
      it('should call support_diagnostics.get_privacy_preference with agent', async () => {
        const agent = fakeAgentKey();
        await support.getPrivacyPreference(agent);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_privacy_preference',
          payload: agent,
        });
      });
    });

    describe('getShareableDiagnostics', () => {
      it('should call support_diagnostics.get_shareable_diagnostics with agent', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getShareableDiagnostics(fakeAgentKey());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_shareable_diagnostics',
          payload: fakeAgentKey(),
        });
      });
    });

    describe('registerHelper', () => {
      it('should call support_diagnostics.register_helper with profile', async () => {
        const profile = makeHelperProfile();
        await support.registerHelper(profile);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'register_helper',
          payload: profile,
        });
      });
    });

    describe('updateAvailability', () => {
      it('should call support_diagnostics.update_availability with input', async () => {
        const input: UpdateAvailInput = {
          helper_hash: fakeActionHash(),
          available: false,
        };
        await support.updateAvailability(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'update_availability',
          payload: input,
        });
      });
    });

    describe('getAvailableHelpers', () => {
      it('should call support_diagnostics.get_available_helpers with category', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getAvailableHelpers('Network');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_available_helpers',
          payload: 'Network',
        });
      });

      it('should handle null category (all helpers)', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getAvailableHelpers(null);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_available_helpers',
          payload: null,
        });
      });
    });

    describe('publishCognitiveUpdate', () => {
      it('should call support_diagnostics.publish_cognitive_update with data', async () => {
        const update: CognitiveUpdate = {
          category: 'Security',
          encoding: new Uint8Array(16384).fill(0x01),
          phi: 0.42,
          resolution_pattern: 'Firewall rule adjustment',
          source_agent: fakeAgentKey(),
          created_at: Date.now(),
        };
        await support.publishCognitiveUpdate(update);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'publish_cognitive_update',
          payload: update,
        });
      });
    });

    describe('getCognitiveUpdatesByCategory', () => {
      it('should call support_diagnostics.get_cognitive_updates_by_category', async () => {
        client.callZome.mockResolvedValue([]);
        await support.getCognitiveUpdatesByCategory('Holochain');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'get_cognitive_updates_by_category',
          payload: 'Holochain',
        });
      });
    });

    describe('absorbCognitiveUpdate', () => {
      it('should call support_diagnostics.absorb_cognitive_update with hash', async () => {
        await support.absorbCognitiveUpdate(fakeActionHash());

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'support_diagnostics',
          fn_name: 'absorb_cognitive_update',
          payload: fakeActionHash(),
        });
      });
    });
  });

  // ==========================================================================
  // Ticket Lifecycle (create -> comment -> escalate -> resolve -> satisfaction)
  // ==========================================================================

  describe('Ticket Lifecycle', () => {
    it('should support full ticket lifecycle through mock calls', async () => {
      const ticketHash = new Uint8Array(39).fill(0x01);
      const resolutionHash = new Uint8Array(39).fill(0x02);

      // Step 1: Create ticket
      client.callZome.mockResolvedValueOnce(ticketHash);
      await support.createTicket(makeTicket());

      // Step 2: Add a comment
      client.callZome.mockResolvedValueOnce(fakeActionHash());
      await support.addComment({
        ticket_hash: ticketHash,
        author: fakeAgentKey(),
        content: 'Looking into this',
        is_symthaea_response: false,
        confidence: null,
        epistemic_status: null,
      });

      // Step 3: Escalate
      client.callZome.mockResolvedValueOnce({});
      await support.escalateTicket({
        ticket_hash: ticketHash,
        from_level: 'Tier1',
        to_level: 'Tier2',
        reason: 'Complex issue',
      });

      // Step 4: Create resolution
      client.callZome.mockResolvedValueOnce(resolutionHash);
      await support.createResolution({
        ticket_hash: ticketHash,
        steps: ['Step 1', 'Step 2'],
        root_cause: 'Config mismatch',
        time_to_resolve_mins: 30,
        effectiveness_rating: 4.0,
        helper: fakeAgentKey(),
        requester: new Uint8Array(39).fill(0xbb),
        anonymized: false,
        helper_signature: new Uint8Array(64).fill(0x01),
        requester_signature: new Uint8Array(64).fill(0x02),
      });

      // Step 5: Close ticket
      client.callZome.mockResolvedValueOnce({});
      await support.closeTicket(ticketHash);

      // Step 6: Submit satisfaction
      client.callZome.mockResolvedValueOnce({});
      await support.submitSatisfaction({
        ticket_hash: ticketHash,
        respondent: fakeAgentKey(),
        rating: 5,
        comment: 'Great support!',
        would_recommend: true,
        submitted_at: Date.now(),
      });

      expect(client.callZome).toHaveBeenCalledTimes(6);

      const fnNames = client.callZome.mock.calls.map(
        (c: unknown[]) => (c[0] as { fn_name: string }).fn_name,
      );
      expect(fnNames).toEqual([
        'create_ticket',
        'add_comment',
        'escalate_ticket',
        'create_resolution',
        'close_ticket',
        'submit_satisfaction',
      ]);
    });
  });

  // ==========================================================================
  // Autonomous Action Lifecycle
  // ==========================================================================

  describe('Autonomous Action Lifecycle', () => {
    it('should support propose -> approve -> execute -> rollback flow', async () => {
      const actionHash = new Uint8Array(39).fill(0x01);

      // Propose
      client.callZome.mockResolvedValueOnce(actionHash);
      await support.proposeAction({
        ticket_hash: fakeActionHash(),
        action_type: 'RestartService',
        description: 'Restart conductor',
        approved: false,
        executed: false,
        result: null,
        success: null,
        rollback_steps: ['Stop service', 'Restore previous state'],
        rollback_state: null,
        rolled_back: false,
        created_at: Date.now(),
      });

      // Approve
      client.callZome.mockResolvedValueOnce({});
      await support.approveAction(actionHash);

      // Execute
      client.callZome.mockResolvedValueOnce({});
      await support.executeAction(actionHash);

      // Rollback
      client.callZome.mockResolvedValueOnce({});
      await support.rollbackAction(actionHash);

      expect(client.callZome).toHaveBeenCalledTimes(4);

      const fnNames = client.callZome.mock.calls.map(
        (c: unknown[]) => (c[0] as { fn_name: string }).fn_name,
      );
      expect(fnNames).toEqual([
        'propose_action',
        'approve_action',
        'execute_action',
        'rollback_action',
      ]);
    });
  });

  // ==========================================================================
  // Preemptive Alert Flow
  // ==========================================================================

  describe('Preemptive Alert Flow', () => {
    it('should support alert creation and promotion to ticket', async () => {
      const alertHash = new Uint8Array(39).fill(0x01);

      // Create alert
      client.callZome.mockResolvedValueOnce(alertHash);
      await support.createPreemptiveAlert({
        predicted_failure: 'Memory pressure detected',
        expected_time_to_failure: '6 hours',
        free_energy: 0.78,
        recommended_action: 'Increase swap or restart services',
        auto_generated_ticket: null,
        created_at: Date.now(),
      });

      // Promote to ticket
      client.callZome.mockResolvedValueOnce(fakeActionHash());
      await support.promoteAlertToTicket({
        alert_hash: alertHash,
        ticket: makeTicket({ priority: 'Critical', is_preemptive: true }),
      });

      expect(client.callZome).toHaveBeenCalledTimes(2);
    });
  });

  // ==========================================================================
  // Cross-domain dispatch pattern
  // ==========================================================================

  describe('Cross-domain dispatch to support', () => {
    it('should use commons_care role for all support_knowledge calls', async () => {
      await support.createArticle(makeArticle());
      await support.searchByCategory('Network');
      await support.listRecentArticles();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('support_knowledge');
      }
    });

    it('should use commons_care role for all support_tickets calls', async () => {
      await support.createTicket(makeTicket());
      await support.listMyTickets();
      await support.listPreemptiveAlerts();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('support_tickets');
      }
    });

    it('should use commons_care role for all support_diagnostics calls', async () => {
      await support.runDiagnostic(makeDiagnostic());
      await support.listDiagnostics();
      await support.getAvailableHelpers(null);

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('commons_care');
        expect((call[0] as { zome_name: string }).zome_name).toBe('support_diagnostics');
      }
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate errors for createArticle', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(support.createArticle(makeArticle())).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate errors for createTicket', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(support.createTicket(makeTicket())).rejects.toThrow('Validation failed');
    });

    it('should propagate errors for runDiagnostic', async () => {
      client.callZome.mockRejectedValue(new Error('Diagnostic timeout'));
      await expect(support.runDiagnostic(makeDiagnostic())).rejects.toThrow('Diagnostic timeout');
    });

    it('should propagate errors for escalateTicket', async () => {
      client.callZome.mockRejectedValue(new Error('Not authorized'));
      await expect(
        support.escalateTicket({
          ticket_hash: fakeActionHash(),
          from_level: 'Tier1',
          to_level: 'Management',
          reason: 'test',
        }),
      ).rejects.toThrow('Not authorized');
    });

    it('should propagate errors for publishCognitiveUpdate', async () => {
      client.callZome.mockRejectedValue(new Error('Rate limited'));
      const update: CognitiveUpdate = {
        category: 'General',
        encoding: new Uint8Array(16384),
        phi: 0.1,
        resolution_pattern: 'test',
        source_agent: fakeAgentKey(),
        created_at: Date.now(),
      };
      await expect(support.publishCognitiveUpdate(update)).rejects.toThrow('Rate limited');
    });
  });
});
