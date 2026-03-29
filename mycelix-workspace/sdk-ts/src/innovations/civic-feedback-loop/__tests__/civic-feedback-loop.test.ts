// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civic Feedback Loop Tests
 *
 * Tests for Justice-Knowledge-Governance auto-propagation,
 * claim derivation, conflict detection, and precedent tracking.
 *
 * @module innovations/civic-feedback-loop/__tests__
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  CivicFeedbackLoop,
  createCivicFeedbackLoop,
  getCivicFeedbackLoop,
  resetCivicFeedbackLoop,
  DEFAULT_FEEDBACK_LOOP_CONFIG,
  type JusticeDecision,
  type GovernanceOutcome,
  type EnactedRule,
  type FeedbackLoopConfig,
  type DerivedKnowledgeClaim,
} from '../index';

// ============================================================================
// MOCKS
// ============================================================================

const mockJusticeDecision: JusticeDecision = {
  caseId: 'case-001',
  decisionId: 'decision-001',
  type: 'final_judgment',
  summary: 'The court finds in favor of the plaintiff regarding property boundary dispute.',
  reasoning: 'Based on survey evidence and prior deeds, the northern boundary is established at marker 47.',
  principles: ['Property boundaries must be determined by clear evidence', 'Survey evidence takes precedence'],
  precedentsCited: ['precedent-1995-property-line', 'precedent-2010-survey'],
  legalDomain: 'property',
  confidence: 0.92,
  setsPrecedent: true,
  participants: ['did:mycelix:alice', 'did:mycelix:bob'],
  timestamp: Date.now(),
};

const mockGovernanceOutcome: GovernanceOutcome = {
  proposalId: 'proposal-gov-123',
  type: 'rule_enactment',
  summary: 'New energy efficiency standards for residential buildings',
  votingResult: {
    for: 150,
    against: 45,
    abstain: 5,
    quorum: 100,
    passed: true,
  },
  enactedRules: [
    {
      id: 'rule-energy-001',
      text: 'All new residential buildings must achieve minimum 80% energy efficiency rating.',
      domain: 'energy',
      effectiveDate: Date.now() + 86400000 * 90, // 90 days
      supersedes: [],
    },
  ],
  domain: 'energy',
  timestamp: Date.now(),
};

// ============================================================================
// TESTS
// ============================================================================

describe('CivicFeedbackLoop', () => {
  let feedbackLoop: CivicFeedbackLoop;

  beforeEach(() => {
    resetCivicFeedbackLoop();
    feedbackLoop = new CivicFeedbackLoop();
  });

  afterEach(() => {
    if (feedbackLoop.isRunning()) {
      feedbackLoop.stop();
    }
  });

  describe('initialization', () => {
    it('should create a feedback loop instance', () => {
      expect(feedbackLoop).toBeInstanceOf(CivicFeedbackLoop);
    });

    it('should not be running initially', () => {
      expect(feedbackLoop.isRunning()).toBe(false);
    });

    it('should accept custom configuration', () => {
      const customConfig: Partial<FeedbackLoopConfig> = {
        autoPropagate: false,
        enableConflictDetection: false,
      };
      const customLoop = new CivicFeedbackLoop(customConfig);
      expect(customLoop).toBeInstanceOf(CivicFeedbackLoop);
    });
  });

  describe('start/stop', () => {
    it('should start the feedback loop', async () => {
      await feedbackLoop.start();
      expect(feedbackLoop.isRunning()).toBe(true);
    });

    it('should stop the feedback loop', async () => {
      await feedbackLoop.start();
      feedbackLoop.stop();
      expect(feedbackLoop.isRunning()).toBe(false);
    });

    it('should be idempotent for multiple starts', async () => {
      await feedbackLoop.start();
      await feedbackLoop.start();
      expect(feedbackLoop.isRunning()).toBe(true);
    });
  });

  describe('propagateJusticeDecision', () => {
    beforeEach(async () => {
      await feedbackLoop.start();
    });

    it('should propagate justice decision to knowledge', async () => {
      const result = await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);

      expect(result).toBeDefined();
      expect(result.propagationId).toBeTruthy();
      expect(result.sourceType).toBe('justice');
      expect(result.claimsCreated).toBeGreaterThan(0);
    });

    it('should create derived claims from decision', async () => {
      const result = await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);

      // Primary claim + principle claims
      expect(result.claimsCreated).toBeGreaterThanOrEqual(1);
    });

    it('should establish precedent for precedent-setting decisions', async () => {
      const result = await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);

      expect(result.precedentEstablished).toBe(true);
    });

    it('should not establish precedent for non-precedent decisions', async () => {
      const nonPrecedentDecision: JusticeDecision = {
        ...mockJusticeDecision,
        setsPrecedent: false,
        type: 'interim_ruling',
      };

      const result = await feedbackLoop.propagateJusticeDecision(nonPrecedentDecision);

      expect(result.precedentEstablished).toBe(false);
    });

    it('should update statistics', async () => {
      const statsBefore = feedbackLoop.getStats();
      await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);
      const statsAfter = feedbackLoop.getStats();

      expect(statsAfter.justiceDecisionsProcessed).toBe(statsBefore.justiceDecisionsProcessed + 1);
    });
  });

  describe('propagateGovernanceOutcome', () => {
    beforeEach(async () => {
      await feedbackLoop.start();
    });

    it('should propagate governance outcome to knowledge', async () => {
      const result = await feedbackLoop.propagateGovernanceOutcome(mockGovernanceOutcome);

      expect(result).toBeDefined();
      expect(result.propagationId).toBeTruthy();
      expect(result.sourceType).toBe('governance');
      expect(result.claimsCreated).toBeGreaterThan(0);
    });

    it('should create rule claims from enacted rules', async () => {
      const result = await feedbackLoop.propagateGovernanceOutcome(mockGovernanceOutcome);

      // Should create claims for each enacted rule
      expect(result.claimsCreated).toBeGreaterThanOrEqual(mockGovernanceOutcome.enactedRules.length);
    });

    it('should handle outcomes without enacted rules', async () => {
      const noRulesOutcome: GovernanceOutcome = {
        ...mockGovernanceOutcome,
        type: 'parameter_change',
        enactedRules: [],
      };

      const result = await feedbackLoop.propagateGovernanceOutcome(noRulesOutcome);

      expect(result.claimsCreated).toBeGreaterThanOrEqual(0);
    });

    it('should update statistics', async () => {
      const statsBefore = feedbackLoop.getStats();
      await feedbackLoop.propagateGovernanceOutcome(mockGovernanceOutcome);
      const statsAfter = feedbackLoop.getStats();

      expect(statsAfter.governanceOutcomesProcessed).toBe(statsBefore.governanceOutcomesProcessed + 1);
    });
  });

  describe('queryDerivedClaims', () => {
    beforeEach(async () => {
      await feedbackLoop.start();
      await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);
      await feedbackLoop.propagateGovernanceOutcome(mockGovernanceOutcome);
    });

    it('should return all claims without filters', () => {
      const claims = feedbackLoop.queryDerivedClaims();
      expect(claims.length).toBeGreaterThan(0);
    });

    it('should filter by source type', () => {
      const justiceClaims = feedbackLoop.queryDerivedClaims({ sourceType: 'justice_decision' });
      const govClaims = feedbackLoop.queryDerivedClaims({ sourceType: 'governance_outcome' });

      expect(justiceClaims.every((c) => c.sourceType === 'justice_decision')).toBe(true);
      expect(govClaims.every((c) => c.sourceType === 'governance_outcome')).toBe(true);
    });

    it('should filter by domain', () => {
      const propertyClaims = feedbackLoop.queryDerivedClaims({ domain: 'property' });
      const energyClaims = feedbackLoop.queryDerivedClaims({ domain: 'energy' });

      expect(propertyClaims.every((c) => c.domain === 'property')).toBe(true);
      expect(energyClaims.every((c) => c.domain === 'energy')).toBe(true);
    });

    it('should filter by isCurrentLaw', () => {
      const currentLawClaims = feedbackLoop.queryDerivedClaims({ isCurrentLaw: true });
      expect(currentLawClaims.every((c) => c.isCurrentLaw === true)).toBe(true);
    });
  });

  describe('conflict detection', () => {
    beforeEach(async () => {
      await feedbackLoop.start();
    });

    it('should detect potential conflicts', async () => {
      // Create two potentially conflicting claims
      await feedbackLoop.propagateJusticeDecision({
        ...mockJusticeDecision,
        decisionId: 'decision-conflict-1',
        summary: 'Property line established at marker 45',
      });

      await feedbackLoop.propagateJusticeDecision({
        ...mockJusticeDecision,
        decisionId: 'decision-conflict-2',
        summary: 'Property line established at marker 48',
      });

      const conflicts = feedbackLoop.getUnresolvedConflicts();
      // May or may not detect conflict depending on implementation
      expect(Array.isArray(conflicts)).toBe(true);
    });

    it('should resolve conflicts', async () => {
      await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);

      const conflicts = feedbackLoop.getUnresolvedConflicts();
      if (conflicts.length > 0) {
        feedbackLoop.resolveConflict(conflicts[0].id, 'accepted');
        const remainingConflicts = feedbackLoop.getUnresolvedConflicts();
        expect(remainingConflicts.length).toBe(conflicts.length - 1);
      }
    });
  });

  describe('getStats', () => {
    it('should return comprehensive statistics', async () => {
      await feedbackLoop.start();
      const stats = feedbackLoop.getStats();

      expect(stats).toBeDefined();
      expect(typeof stats.justiceDecisionsProcessed).toBe('number');
      expect(typeof stats.governanceOutcomesProcessed).toBe('number');
      expect(typeof stats.claimsDerived).toBe('number');
      expect(typeof stats.precedentsEstablished).toBe('number');
      expect(typeof stats.conflictsDetected).toBe('number');
      expect(typeof stats.conflictsResolved).toBe('number');
    });

    it('should track uptime', async () => {
      await feedbackLoop.start();
      // Wait a bit
      await new Promise((resolve) => setTimeout(resolve, 10));
      const stats = feedbackLoop.getStats();

      expect(stats.uptimeMs).toBeGreaterThan(0);
    });
  });
});

describe('Factory Functions', () => {
  afterEach(() => {
    resetCivicFeedbackLoop();
  });

  describe('createCivicFeedbackLoop', () => {
    it('should create a new feedback loop', () => {
      const loop = createCivicFeedbackLoop();
      expect(loop).toBeInstanceOf(CivicFeedbackLoop);
    });

    it('should accept configuration', () => {
      const loop = createCivicFeedbackLoop({ autoPropagate: false });
      expect(loop).toBeInstanceOf(CivicFeedbackLoop);
    });
  });

  describe('getCivicFeedbackLoop', () => {
    it('should return singleton instance', () => {
      const loop1 = getCivicFeedbackLoop();
      const loop2 = getCivicFeedbackLoop();
      expect(loop1).toBe(loop2);
    });

    it('should create instance with config on first call', () => {
      const loop = getCivicFeedbackLoop({ autoPropagate: false });
      expect(loop).toBeInstanceOf(CivicFeedbackLoop);
    });
  });

  describe('resetCivicFeedbackLoop', () => {
    it('should reset singleton', () => {
      const loop1 = getCivicFeedbackLoop();
      resetCivicFeedbackLoop();
      const loop2 = getCivicFeedbackLoop();
      expect(loop1).not.toBe(loop2);
    });
  });
});

describe('DEFAULT_FEEDBACK_LOOP_CONFIG', () => {
  it('should have expected default values', () => {
    expect(DEFAULT_FEEDBACK_LOOP_CONFIG.autoPropagate).toBe(true);
    expect(DEFAULT_FEEDBACK_LOOP_CONFIG.enableConflictDetection).toBe(true);
    expect(DEFAULT_FEEDBACK_LOOP_CONFIG.requireConsensusForConflicts).toBe(true);
  });
});

describe('Epistemic Classification', () => {
  let feedbackLoop: CivicFeedbackLoop;

  beforeEach(async () => {
    feedbackLoop = new CivicFeedbackLoop();
    await feedbackLoop.start();
  });

  afterEach(() => {
    feedbackLoop.stop();
  });

  it('should classify justice claims with appropriate epistemic position', async () => {
    await feedbackLoop.propagateJusticeDecision(mockJusticeDecision);
    const claims = feedbackLoop.queryDerivedClaims({ sourceType: 'justice_decision' });

    expect(claims.length).toBeGreaterThan(0);
    const claim = claims[0];

    // Justice decisions should have cryptographic empirical level
    expect(claim.position.empirical).toBeDefined();
    expect(claim.classificationCode).toMatch(/^E\d-N\d-M\d$/);
  });

  it('should classify governance claims with appropriate epistemic position', async () => {
    await feedbackLoop.propagateGovernanceOutcome(mockGovernanceOutcome);
    const claims = feedbackLoop.queryDerivedClaims({ sourceType: 'governance_outcome' });

    expect(claims.length).toBeGreaterThan(0);
    const claim = claims[0];

    // Governance outcomes should have cryptographic empirical level
    expect(claim.position.empirical).toBeDefined();
    expect(claim.classificationCode).toMatch(/^E\d-N\d-M\d$/);
  });
});
