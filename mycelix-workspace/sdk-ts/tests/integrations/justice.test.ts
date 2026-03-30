// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Justice Integration Tests
 *
 * Tests for JusticeService - cases, mediation, arbitration, and enforcement
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  JusticeService,
  getJusticeService,
  resetJusticeService,
  type Case,
  type Evidence,
  type Decision,
  type MediatorProfile,
  type ArbitratorProfile,
  type RestorativeCircle,
} from '../../src/integrations/justice/index.js';

describe('Justice Integration', () => {
  let service: JusticeService;

  beforeEach(() => {
    resetJusticeService();
    service = new JusticeService();
  });

  describe('JusticeService', () => {
    describe('fileCase', () => {
      it('should create a new case', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:complainant',
          'did:mycelix:respondent',
          'Contract breach dispute',
          'Failure to deliver promised goods',
          'ContractDispute'
        );

        expect(caseRecord).toBeDefined();
        expect(caseRecord.id).toMatch(/^case-/);
        expect(caseRecord.complainantId).toBe('did:mycelix:complainant');
        expect(caseRecord.respondentId).toBe('did:mycelix:respondent');
        expect(caseRecord.category).toBe('ContractDispute');
        expect(caseRecord.status).toBe('Active');
        expect(caseRecord.phase).toBe('Filed');
      });

      it('should initialize with empty evidence', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Fraud case',
          'Fraudulent transaction',
          'ConductViolation'
        );

        expect(caseRecord.evidence).toEqual([]);
      });

      it('should initialize timeline with filing entry', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Misconduct',
          'Description',
          'ConductViolation'
        );

        expect(caseRecord.timeline.length).toBe(1);
        expect(caseRecord.timeline[0].action).toBe('Case filed');
        expect(caseRecord.timeline[0].actor).toBe('did:mycelix:c');
      });
    });

    describe('submitEvidence', () => {
      it('should add evidence to a case', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Contract breach',
          'Description',
          'ContractDispute'
        );

        const evidence = service.submitEvidence(
          caseRecord.id,
          'did:mycelix:c',
          'document',
          'Screenshot of conversation',
          'Evidence of breach',
          'ipfs://evidence123'
        );

        expect(evidence).toBeDefined();
        expect(evidence.id).toMatch(/^evidence-/);
        expect(evidence.type).toBe('document');
        expect(evidence.title).toBe('Screenshot of conversation');
        expect(evidence.verified).toBe(false);
      });

      it('should throw for non-existent case', () => {
        expect(() => {
          service.submitEvidence(
            'case-fake',
            'did:mycelix:submitter',
            'document',
            'Evidence',
            'Description',
            'hash123'
          );
        }).toThrow('Case not found');
      });

      it('should track evidence submission time', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'PropertyDispute'
        );

        const evidence = service.submitEvidence(
          caseRecord.id,
          'did:mycelix:c',
          'witness',
          'Witness statement',
          'Statement from witness',
          'hash456'
        );

        expect(evidence.submittedAt).toBeDefined();
        expect(evidence.submittedAt).toBeLessThanOrEqual(Date.now());
      });

      it('should add evidence to case record', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'ContractDispute'
        );

        service.submitEvidence(caseRecord.id, 'did:mycelix:c', 'Document', 'Doc 1', 'Desc', 'h1');
        service.submitEvidence(caseRecord.id, 'did:mycelix:r', 'Testimony', 'Statement', 'Desc', 'h2');

        const updated = service.getCase(caseRecord.id);
        expect(updated!.evidence.length).toBe(2);
      });
    });

    describe('registerMediator', () => {
      it('should register a new mediator', () => {
        const mediator = service.registerMediator('did:mycelix:mediator1', ['ContractDispute', 'PropertyDispute']);

        expect(mediator).toBeDefined();
        expect(mediator.did).toBe('did:mycelix:mediator1');
        expect(mediator.specializations).toContain('ContractDispute');
        expect(mediator.available).toBe(true);
        expect(mediator.casesMediated).toBe(0);
      });
    });

    describe('registerArbitrator', () => {
      it('should register a new arbitrator', () => {
        const arbitrator = service.registerArbitrator('did:mycelix:arb1', ['GovernanceDispute'], 2);

        expect(arbitrator).toBeDefined();
        expect(arbitrator.did).toBe('did:mycelix:arb1');
        expect(arbitrator.tier).toBe(2);
        expect(arbitrator.available).toBe(true);
      });

      it('should default to tier 1', () => {
        const arbitrator = service.registerArbitrator('did:mycelix:arb2', ['ContractDispute']);
        expect(arbitrator.tier).toBe(1);
      });
    });

    describe('initiateMediation', () => {
      it('should initiate mediation with available mediator', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Dispute',
          'Description',
          'ContractDispute'
        );

        service.registerMediator('did:mycelix:mediator', ['ContractDispute']);

        const updated = service.initiateMediation(caseRecord.id, 'did:mycelix:mediator');

        expect(updated.phase).toBe('Mediation');
        expect(updated.status).toBe('Active');
      });

      it('should throw for unavailable mediator', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Dispute',
          'Description',
          'ContractDispute'
        );

        expect(() => {
          service.initiateMediation(caseRecord.id, 'did:mycelix:unknown');
        }).toThrow('Mediator not available');
      });

      it('should throw for non-existent case', () => {
        service.registerMediator('did:mycelix:mediator', ['ContractDispute']);

        expect(() => {
          service.initiateMediation('case-fake', 'did:mycelix:mediator');
        }).toThrow('Case not found');
      });
    });

    describe('escalateToArbitration', () => {
      it('should escalate case to arbitration', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Complex dispute',
          'Description',
          'PropertyDispute'
        );

        service.registerArbitrator('did:mycelix:arb1', ['PropertyDispute']);
        service.registerArbitrator('did:mycelix:arb2', ['PropertyDispute']);

        const updated = service.escalateToArbitration(caseRecord.id, [
          'did:mycelix:arb1',
          'did:mycelix:arb2',
        ]);

        expect(updated.phase).toBe('Arbitration');
      });

      it('should throw for unavailable arbitrator', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Dispute',
          'Description',
          'ContractDispute'
        );

        expect(() => {
          service.escalateToArbitration(caseRecord.id, ['did:mycelix:unknown']);
        }).toThrow(/not available/);
      });
    });

    describe('renderDecision', () => {
      it('should render a decision for a case', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Contract breach',
          'Description',
          'ContractDispute'
        );

        service.registerArbitrator('did:mycelix:arb', ['ContractDispute']);
        service.escalateToArbitration(caseRecord.id, ['did:mycelix:arb']);

        const decision = service.renderDecision(
          caseRecord.id,
          'ForComplainant',
          'Evidence clearly shows breach of contract',
          [
            {
              type: 'compensation',
              targetId: 'did:mycelix:r',
              description: 'Pay damages',
              amount: 1000,
              completed: false,
            },
          ],
          ['did:mycelix:arb']
        );

        expect(decision).toBeDefined();
        expect(decision.id).toMatch(/^decision-/);
        expect(decision.outcome).toBe('ForComplainant');
        expect(decision.remedies.length).toBe(1);
        expect(decision.appealable).toBe(true);
      });

      it('should update case phase to enforcement', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'ContractDispute'
        );

        service.registerArbitrator('did:mycelix:arb', ['ContractDispute']);
        service.escalateToArbitration(caseRecord.id, ['did:mycelix:arb']);
        service.renderDecision(caseRecord.id, 'ForRespondent', 'Claim not substantiated', [], ['did:mycelix:arb']);

        const updated = service.getCase(caseRecord.id);
        expect(updated!.phase).toBe('Enforcement');
      });

      it('should throw for non-existent case', () => {
        expect(() => {
          service.renderDecision('case-fake', 'Dismissed', 'No merit', [], ['did:mycelix:arb']);
        }).toThrow('Case not found');
      });
    });

    describe('createRestorativeCircle', () => {
      it('should create a restorative circle', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:harmed',
          'did:mycelix:responsible',
          'Community harm',
          'Description',
          'ConductViolation'
        );

        const circle = service.createRestorativeCircle(caseRecord.id, 'did:mycelix:facilitator', [
          { did: 'did:mycelix:harmed', role: 'harmed' },
          { did: 'did:mycelix:responsible', role: 'responsible' },
          { did: 'did:mycelix:supporter', role: 'supporter' },
        ]);

        expect(circle).toBeDefined();
        expect(circle.id).toMatch(/^circle-/);
        expect(circle.status).toBe('forming');
        expect(circle.participants.length).toBe(3);
        expect(circle.participants.every((p) => !p.consented)).toBe(true);
      });
    });

    describe('recordConsent', () => {
      it('should record participant consent', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:harmed',
          'did:mycelix:responsible',
          'Case',
          'Description',
          'ConductViolation'
        );

        const circle = service.createRestorativeCircle(caseRecord.id, 'did:mycelix:facilitator', [
          { did: 'did:mycelix:harmed', role: 'harmed' },
          { did: 'did:mycelix:responsible', role: 'responsible' },
        ]);

        const updated = service.recordConsent(circle.id, 'did:mycelix:harmed');

        const participant = updated.participants.find((p) => p.did === 'did:mycelix:harmed');
        expect(participant!.consented).toBe(true);
      });

      it('should activate circle when all consent', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:harmed',
          'did:mycelix:responsible',
          'Case',
          'Description',
          'ConductViolation'
        );

        const circle = service.createRestorativeCircle(caseRecord.id, 'did:mycelix:facilitator', [
          { did: 'did:mycelix:harmed', role: 'harmed' },
          { did: 'did:mycelix:responsible', role: 'responsible' },
        ]);

        service.recordConsent(circle.id, 'did:mycelix:harmed');
        const activated = service.recordConsent(circle.id, 'did:mycelix:responsible');

        expect(activated.status).toBe('active');
      });
    });

    describe('getCase', () => {
      it('should retrieve an existing case', () => {
        const created = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'ContractDispute'
        );

        const retrieved = service.getCase(created.id);

        expect(retrieved).toBeDefined();
        expect(retrieved!.id).toBe(created.id);
      });

      it('should return undefined for non-existent case', () => {
        const result = service.getCase('case-fake');
        expect(result).toBeUndefined();
      });
    });

    describe('getDecision', () => {
      it('should retrieve decision for a case', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'ContractDispute'
        );

        service.registerArbitrator('did:mycelix:arb', ['ContractDispute']);
        service.escalateToArbitration(caseRecord.id, ['did:mycelix:arb']);
        service.renderDecision(caseRecord.id, 'SplitDecision', 'Both parties share responsibility', [], ['did:mycelix:arb']);

        const decision = service.getDecision(caseRecord.id);

        expect(decision).toBeDefined();
        expect(decision!.outcome).toBe('SplitDecision');
      });

      it('should return undefined for case without decision', () => {
        const caseRecord = service.fileCase(
          'did:mycelix:c',
          'did:mycelix:r',
          'Case',
          'Description',
          'ContractDispute'
        );

        const decision = service.getDecision(caseRecord.id);
        expect(decision).toBeUndefined();
      });
    });

    describe('getAvailableMediators', () => {
      it('should return available mediators for a category', () => {
        service.registerMediator('did:mycelix:m1', ['ContractDispute', 'PropertyDispute']);
        service.registerMediator('did:mycelix:m2', ['GovernanceDispute']);
        service.registerMediator('did:mycelix:m3', ['ContractDispute']);

        const mediators = service.getAvailableMediators('ContractDispute');

        expect(mediators.length).toBe(2);
        expect(mediators.every((m) => m.specializations.includes('ContractDispute'))).toBe(true);
      });

      it('should return empty array when no mediators available', () => {
        const mediators = service.getAvailableMediators('GovernanceDispute');
        expect(mediators).toEqual([]);
      });
    });
  });

  describe('getJusticeService', () => {
    it('should return singleton instance', () => {
      const service1 = getJusticeService();
      const service2 = getJusticeService();

      expect(service1).toBe(service2);
    });
  });
});
