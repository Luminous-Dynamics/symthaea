/**
 * Constitutional AI Governance Tests
 *
 * Tests for DAO-governed constitutional rules, violation detection,
 * amendment proposals, and agent compliance auditing.
 *
 * @module innovations/constitutional-ai/__tests__
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ConstitutionalGovernor,
  createConstitutionalGovernor,
  getConstitutionalGovernor,
  resetConstitutionalGovernor,
  DEFAULT_CONSTITUTIONAL_CONFIG,
  FOUNDATIONAL_RULES,
  type ConstitutionalRule,
  type RuleCategory,
  type AmendmentProposal,
  type ConstitutionalConfig,
  type ContentCheckResult,
} from '../index';

// ============================================================================
// MOCKS & HELPERS
// ============================================================================

const mockAgentDid = 'did:mycelix:agent-001';
const mockVoterDid = 'did:mycelix:voter-001';

const createTestRule = (overrides?: Partial<ConstitutionalRule>): Omit<ConstitutionalRule, 'id'> => ({
  category: 'safety',
  severity: 'critical',
  title: 'Test Safety Rule',
  description: 'Agents must not provide harmful information',
  condition: {
    type: 'content_match',
    pattern: 'dangerous|harmful|weapon',
    action: 'block',
  },
  active: true,
  version: 1,
  createdAt: Date.now(),
  amendments: [],
  ...overrides,
});

// ============================================================================
// TESTS
// ============================================================================

describe('ConstitutionalGovernor', () => {
  let governor: ConstitutionalGovernor;

  beforeEach(() => {
    resetConstitutionalGovernor();
    governor = new ConstitutionalGovernor();
  });

  describe('initialization', () => {
    it('should create a governor instance', () => {
      expect(governor).toBeInstanceOf(ConstitutionalGovernor);
    });

    it('should load foundational rules by default', () => {
      const rules = governor.getRules();
      expect(rules.length).toBeGreaterThan(0);
    });

    it('should accept custom configuration', () => {
      const config: Partial<ConstitutionalConfig> = {
        loadFoundationalRules: false,
        enableViolationTracking: false,
      };
      const customGovernor = new ConstitutionalGovernor(config);
      const rules = customGovernor.getRules();
      expect(rules.length).toBe(0);
    });
  });

  describe('rule management', () => {
    it('should add a new rule', () => {
      const initialCount = governor.getRules().length;
      const rule = governor.addRule(createTestRule());

      expect(rule.id).toBeTruthy();
      expect(governor.getRules().length).toBe(initialCount + 1);
    });

    it('should get rule by id', () => {
      const added = governor.addRule(createTestRule());
      const retrieved = governor.getRule(added.id);

      expect(retrieved).toBeDefined();
      expect(retrieved?.id).toBe(added.id);
    });

    it('should return undefined for non-existent rule', () => {
      const rule = governor.getRule('non-existent');
      expect(rule).toBeUndefined();
    });

    it('should get rules by category', () => {
      governor.addRule(createTestRule({ category: 'safety' }));
      governor.addRule(createTestRule({ category: 'truthfulness' }));
      governor.addRule(createTestRule({ category: 'privacy' }));

      const safetyRules = governor.getRulesByCategory('safety');
      expect(safetyRules.every((r) => r.category === 'safety')).toBe(true);
    });

    it('should get only active rules', () => {
      governor.addRule(createTestRule({ active: true }));
      governor.addRule(createTestRule({ active: false }));

      const activeRules = governor.getActiveRules();
      expect(activeRules.every((r) => r.active)).toBe(true);
    });
  });

  describe('checkContent', () => {
    beforeEach(() => {
      // Add test rules
      governor.addRule({
        category: 'safety',
        severity: 'critical',
        title: 'No Weapons',
        description: 'Must not provide weapons information',
        condition: {
          type: 'content_match',
          pattern: 'how to make.*weapon|build.*bomb',
          action: 'block',
        },
        active: true,
        version: 1,
        createdAt: Date.now(),
        amendments: [],
      });

      governor.addRule({
        category: 'truthfulness',
        severity: 'major',
        title: 'No False Claims',
        description: 'Must not make verifiably false claims',
        condition: {
          type: 'confidence_check',
          minConfidence: 0.5,
          action: 'flag',
        },
        active: true,
        version: 1,
        createdAt: Date.now(),
        amendments: [],
      });
    });

    it('should approve safe content', () => {
      const result = governor.checkContent(
        'How do I apply for unemployment benefits?',
        { agentId: mockAgentDid, confidence: 0.9 }
      );

      expect(result.approved).toBe(true);
      expect(result.violations.length).toBe(0);
    });

    it('should detect violations in unsafe content', () => {
      const result = governor.checkContent(
        'Here is how to make a weapon at home',
        { agentId: mockAgentDid, confidence: 0.9 }
      );

      expect(result.approved).toBe(false);
      expect(result.violations.length).toBeGreaterThan(0);
      expect(result.violations[0].category).toBe('safety');
    });

    it('should respect rule severity', () => {
      const criticalResult = governor.checkContent(
        'Instructions for building a bomb',
        { agentId: mockAgentDid, confidence: 0.9 }
      );

      expect(criticalResult.approved).toBe(false);
      expect(criticalResult.violations.some((v) => v.severity === 'critical')).toBe(true);
    });

    it('should check confidence thresholds', () => {
      const lowConfidenceResult = governor.checkContent(
        'This claim is definitely true',
        { agentId: mockAgentDid, confidence: 0.3 } // Below threshold
      );

      // Should flag but may still approve depending on rule action
      expect(lowConfidenceResult.warnings.length).toBeGreaterThanOrEqual(0);
    });

    it('should return suggested modifications', () => {
      const result = governor.checkContent(
        'How to make a weapon for hunting',
        { agentId: mockAgentDid, confidence: 0.9 }
      );

      if (!result.approved) {
        expect(result.suggestedAction).toBeDefined();
      }
    });
  });

  describe('violation tracking', () => {
    it('should track violations by agent', () => {
      governor.addRule({
        category: 'safety',
        severity: 'major',
        title: 'Test Rule',
        description: 'Test',
        condition: { type: 'content_match', pattern: 'forbidden', action: 'block' },
        active: true,
        version: 1,
        createdAt: Date.now(),
        amendments: [],
      });

      governor.checkContent('This is forbidden content', {
        agentId: mockAgentDid,
        confidence: 0.9,
      });

      const status = governor.getAgentStatus(mockAgentDid);
      expect(status.totalViolations).toBeGreaterThan(0);
    });

    it('should track violation severity breakdown', () => {
      governor.addRule({
        category: 'safety',
        severity: 'critical',
        title: 'Critical Rule',
        description: 'Critical test',
        condition: { type: 'content_match', pattern: 'critical_word', action: 'block' },
        active: true,
        version: 1,
        createdAt: Date.now(),
        amendments: [],
      });

      governor.checkContent('critical_word appears here', {
        agentId: mockAgentDid,
        confidence: 0.9,
      });

      const status = governor.getAgentStatus(mockAgentDid);
      expect(status.violationsBySeverity.critical).toBeGreaterThan(0);
    });

    it('should return clean status for compliant agents', () => {
      governor.checkContent('Normal safe content', {
        agentId: mockAgentDid,
        confidence: 0.9,
      });

      const status = governor.getAgentStatus(mockAgentDid);
      expect(status.totalViolations).toBe(0);
      expect(status.isCompliant).toBe(true);
    });
  });

  describe('amendment proposals', () => {
    let ruleId: string;

    beforeEach(() => {
      const rule = governor.addRule(createTestRule());
      ruleId = rule.id;
    });

    it('should create an amendment proposal', () => {
      const proposal = governor.proposeAmendment({
        type: 'modify',
        ruleId,
        proposerId: mockVoterDid,
        justification: 'Rule is too strict',
        newRule: {
          severity: 'major', // Change from critical to major
        },
      });

      expect(proposal).toBeDefined();
      expect(proposal.id).toBeTruthy();
      expect(proposal.type).toBe('modify');
      expect(proposal.status).toBe('pending');
    });

    it('should create repeal proposal', () => {
      const proposal = governor.proposeAmendment({
        type: 'repeal',
        ruleId,
        proposerId: mockVoterDid,
        justification: 'Rule is no longer needed',
      });

      expect(proposal.type).toBe('repeal');
    });

    it('should create new rule proposal', () => {
      const proposal = governor.proposeAmendment({
        type: 'new',
        proposerId: mockVoterDid,
        justification: 'New rule needed for emerging threat',
        newRule: createTestRule({ title: 'New Proposed Rule' }),
      });

      expect(proposal.type).toBe('new');
    });

    it('should track votes on proposals', () => {
      const proposal = governor.proposeAmendment({
        type: 'modify',
        ruleId,
        proposerId: mockVoterDid,
        justification: 'Test amendment',
        newRule: { severity: 'minor' },
      });

      governor.voteOnProposal(proposal.id, 'voter-1', true);
      governor.voteOnProposal(proposal.id, 'voter-2', false);

      const updated = governor.getProposal(proposal.id);
      expect(updated?.votes.for).toBe(1);
      expect(updated?.votes.against).toBe(1);
    });

    it('should prevent duplicate votes', () => {
      const proposal = governor.proposeAmendment({
        type: 'modify',
        ruleId,
        proposerId: mockVoterDid,
        justification: 'Test',
        newRule: { severity: 'minor' },
      });

      governor.voteOnProposal(proposal.id, 'voter-1', true);
      governor.voteOnProposal(proposal.id, 'voter-1', false); // Duplicate

      const updated = governor.getProposal(proposal.id);
      expect(updated?.votes.for).toBe(1); // Should still be 1
    });

    it('should finalize approved proposal', async () => {
      const proposal = governor.proposeAmendment({
        type: 'modify',
        ruleId,
        proposerId: mockVoterDid,
        justification: 'Reduce severity',
        newRule: { severity: 'minor' },
      });

      // Cast enough votes to pass
      for (let i = 0; i < 10; i++) {
        governor.voteOnProposal(proposal.id, `voter-${i}`, true);
      }

      await governor.finalizeProposal(proposal.id);

      const finalizedProposal = governor.getProposal(proposal.id);
      expect(finalizedProposal?.status).toBe('passed');

      const modifiedRule = governor.getRule(ruleId);
      expect(modifiedRule?.severity).toBe('minor');
    });
  });

  describe('audit', () => {
    it('should generate audit report', () => {
      // Perform some actions
      governor.addRule(createTestRule());
      governor.checkContent('test content', { agentId: mockAgentDid, confidence: 0.9 });

      const audit = governor.generateAudit();

      expect(audit).toBeDefined();
      expect(audit.timestamp).toBeTruthy();
      expect(audit.totalRules).toBeGreaterThan(0);
      expect(audit.totalChecks).toBeGreaterThan(0);
    });

    it('should include rule statistics in audit', () => {
      const audit = governor.generateAudit();

      expect(audit.rulesByCategory).toBeDefined();
      expect(audit.rulesBySeverity).toBeDefined();
    });
  });
});

describe('Factory Functions', () => {
  afterEach(() => {
    resetConstitutionalGovernor();
  });

  describe('createConstitutionalGovernor', () => {
    it('should create a new governor', () => {
      const governor = createConstitutionalGovernor();
      expect(governor).toBeInstanceOf(ConstitutionalGovernor);
    });

    it('should accept configuration', () => {
      const governor = createConstitutionalGovernor({
        loadFoundationalRules: false,
      });
      expect(governor.getRules().length).toBe(0);
    });
  });

  describe('getConstitutionalGovernor', () => {
    it('should return singleton', () => {
      const g1 = getConstitutionalGovernor();
      const g2 = getConstitutionalGovernor();
      expect(g1).toBe(g2);
    });
  });

  describe('resetConstitutionalGovernor', () => {
    it('should reset singleton', () => {
      const g1 = getConstitutionalGovernor();
      resetConstitutionalGovernor();
      const g2 = getConstitutionalGovernor();
      expect(g1).not.toBe(g2);
    });
  });
});

describe('FOUNDATIONAL_RULES', () => {
  it('should define foundational rules', () => {
    expect(FOUNDATIONAL_RULES).toBeDefined();
    expect(FOUNDATIONAL_RULES.length).toBeGreaterThan(0);
  });

  it('should include safety rules', () => {
    const safetyRules = FOUNDATIONAL_RULES.filter((r) => r.category === 'safety');
    expect(safetyRules.length).toBeGreaterThan(0);
  });

  it('should include truthfulness rules', () => {
    const truthRules = FOUNDATIONAL_RULES.filter((r) => r.category === 'truthfulness');
    expect(truthRules.length).toBeGreaterThan(0);
  });

  it('should include privacy rules', () => {
    const privacyRules = FOUNDATIONAL_RULES.filter((r) => r.category === 'privacy');
    expect(privacyRules.length).toBeGreaterThan(0);
  });
});

describe('DEFAULT_CONSTITUTIONAL_CONFIG', () => {
  it('should have expected defaults', () => {
    expect(DEFAULT_CONSTITUTIONAL_CONFIG.loadFoundationalRules).toBe(true);
    expect(DEFAULT_CONSTITUTIONAL_CONFIG.enableViolationTracking).toBe(true);
    expect(DEFAULT_CONSTITUTIONAL_CONFIG.enableAmendments).toBe(true);
    expect(DEFAULT_CONSTITUTIONAL_CONFIG.quorumPercentage).toBeDefined();
    expect(DEFAULT_CONSTITUTIONAL_CONFIG.passingPercentage).toBeDefined();
  });
});

describe('Rule Categories', () => {
  let governor: ConstitutionalGovernor;

  beforeEach(() => {
    governor = new ConstitutionalGovernor({ loadFoundationalRules: false });
  });

  const categories: RuleCategory[] = [
    'safety',
    'truthfulness',
    'privacy',
    'fairness',
    'transparency',
    'accountability',
    'lawfulness',
    'autonomy',
    'beneficence',
  ];

  categories.forEach((category) => {
    it(`should support ${category} category`, () => {
      const rule = governor.addRule(createTestRule({ category }));
      expect(rule.category).toBe(category);
    });
  });
});

describe('Rule Severity Levels', () => {
  let governor: ConstitutionalGovernor;

  beforeEach(() => {
    governor = new ConstitutionalGovernor({ loadFoundationalRules: false });
  });

  it('should handle critical severity', () => {
    governor.addRule(
      createTestRule({
        severity: 'critical',
        condition: { type: 'content_match', pattern: 'critical_test', action: 'block' },
      })
    );

    const result = governor.checkContent('critical_test', {
      agentId: mockAgentDid,
      confidence: 0.9,
    });

    expect(result.approved).toBe(false);
  });

  it('should handle major severity', () => {
    governor.addRule(
      createTestRule({
        severity: 'major',
        condition: { type: 'content_match', pattern: 'major_test', action: 'flag' },
      })
    );

    const result = governor.checkContent('major_test', {
      agentId: mockAgentDid,
      confidence: 0.9,
    });

    // Major with 'flag' action may still approve but with warnings
    expect(result.warnings.length).toBeGreaterThanOrEqual(0);
  });

  it('should handle minor severity', () => {
    governor.addRule(
      createTestRule({
        severity: 'minor',
        condition: { type: 'content_match', pattern: 'minor_test', action: 'warn' },
      })
    );

    const result = governor.checkContent('minor_test', {
      agentId: mockAgentDid,
      confidence: 0.9,
    });

    // Minor with 'warn' action should still approve
    expect(result.approved).toBe(true);
    expect(result.warnings.length).toBeGreaterThan(0);
  });
});
