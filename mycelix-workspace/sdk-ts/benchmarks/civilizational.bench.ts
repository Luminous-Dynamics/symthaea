/**
 * @mycelix/sdk Civilizational OS Benchmarks
 *
 * Performance benchmarks for the 8 Civilizational hApp integrations:
 * Identity, Governance, Finance, Property, Energy, Media, Justice, Knowledge
 */

import { bench, describe } from 'vitest';
import { IdentityService, resetIdentityService } from '../src/integrations/identity/index.js';
import { GovernanceService, resetGovernanceService } from '../src/integrations/governance/index.js';
import { FinanceService, resetFinanceService } from '../src/integrations/finance/index.js';
import { PropertyService, resetPropertyService } from '../src/integrations/property/index.js';
import { EnergyService, resetEnergyService } from '../src/integrations/energy/index.js';
import { MediaService, resetMediaService } from '../src/integrations/media/index.js';
import { JusticeService, resetJusticeService } from '../src/integrations/justice/index.js';
import { KnowledgeService, resetKnowledgeService } from '../src/integrations/knowledge/index.js';
import { EmpiricalLevel, NormativeLevel } from '../src/epistemic/index.js';
import { LocalBridge } from '../src/bridge/index.js';
import * as matl from '../src/matl/index.js';

// ============================================================================
// Identity Service Benchmarks
// ============================================================================

describe('Identity Service Performance', () => {
  const identity = new IdentityService();

  bench('createIdentity', () => {
    identity.createIdentity(`pubkey-${Math.random().toString(36)}`);
  });

  bench('issueCredential', () => {
    const issuer = identity.createIdentity('issuer-key-benchmark');
    const subject = identity.createIdentity('subject-key-benchmark');
    identity.issueCredential(issuer.did, subject.did, 'TestCredential', { test: true });
  });

  bench('attestTrust', () => {
    const attester = identity.createIdentity(`attester-${Math.random()}`);
    const subject = identity.createIdentity(`subject-${Math.random()}`);
    identity.attestTrust(attester.did, subject.did);
  });

  bench('verifyCredential', () => {
    const issuer = identity.createIdentity('verify-issuer');
    const subject = identity.createIdentity('verify-subject');
    const cred = identity.issueCredential(issuer.did, subject.did, 'VerifyTest', {});
    return () => identity.verifyCredential(cred.id);
  });

  bench('getProfile', () => {
    identity.getProfile('did:mycelix:benchmark');
  });
});

// ============================================================================
// Governance Service Benchmarks
// ============================================================================

describe('Governance Service Performance', () => {
  const governance = new GovernanceService();

  bench('registerMember', () => {
    governance.registerMember({
      did: `did:mycelix:member-${Math.random()}`,
      daoId: 'benchmark-dao',
      role: 'member',
      votingPower: 100,
      delegatedPower: 0,
      joinedAt: Date.now(),
    });
  });

  bench('createProposal', () => {
    governance.registerMember({
      did: 'proposer-did',
      daoId: 'benchmark-dao',
      role: 'member',
      votingPower: 100,
      delegatedPower: 0,
      joinedAt: Date.now(),
    });
    governance.createProposal({
      title: 'Benchmark Proposal',
      description: 'Performance test',
      proposerId: 'proposer-did',
      daoId: 'benchmark-dao',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });
  });

  bench('castVote', () => {
    governance.registerMember({
      did: `voter-${Math.random()}`,
      daoId: 'vote-dao',
      role: 'member',
      votingPower: 100,
      delegatedPower: 0,
      joinedAt: Date.now(),
    });
    const proposal = governance.createProposal({
      title: 'Vote Test',
      description: 'Test',
      proposerId: 'proposer-did',
      daoId: 'vote-dao',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });
    return () =>
      governance.castVote({
        proposalId: proposal.id,
        voterId: `voter-${Math.random()}`,
        choice: 'approve',
        weight: 100,
      });
  });
});

// ============================================================================
// Finance Service Benchmarks
// ============================================================================

describe('Finance Service Performance', () => {
  const finance = new FinanceService();

  bench('createWallet', () => {
    finance.createWallet(`owner-${Math.random()}`, 'personal');
  });

  bench('transfer', () => {
    const from = finance.createWallet('from-owner', 'personal');
    const to = finance.createWallet('to-owner', 'personal');
    from.balances.set('MCX', 1000);
    return () => finance.transfer(from.id, to.id, 10, 'MCX');
  });

  bench('requestLoan', () => {
    finance.requestLoan({
      borrowerId: 'borrower-did',
      amount: 1000,
      currency: 'MCX',
      purpose: 'Benchmark test',
      termMonths: 12,
      interestRate: 5,
    });
  });

  bench('calculateCreditLimit', () => {
    const rep = matl.createReputation('credit-test');
    finance.calculateCreditLimit({
      reputation: rep,
      transactionHistory: 50,
      communityBacking: 500,
      collateralValue: 1000,
    });
  });
});

// ============================================================================
// Property Service Benchmarks
// ============================================================================

describe('Property Service Performance', () => {
  const property = new PropertyService();

  bench('registerAsset', () => {
    property.registerAsset(
      'equipment',
      'Benchmark Asset',
      'Performance test asset',
      `owner-${Math.random()}`,
      'sole'
    );
  });

  bench('proposeTransfer', () => {
    const asset = property.registerAsset('vehicle', 'Transfer Test', 'Test', 'owner-transfer', 'sole');
    return () => property.proposeTransfer(asset.id, 'owner-transfer', 'buyer-transfer', 100);
  });

  bench('joinCommons', () => {
    property.joinCommons(`commons-${Math.random()}`, `member-${Math.random()}`, ['read', 'harvest']);
  });

  bench('recordUsage', () => {
    property.recordUsage('asset-benchmark', 'user-benchmark', 'Benchmark usage');
  });

  bench('verifyOwnership', async () => {
    const asset = property.registerAsset('digital', 'Verify Test', 'Test', 'owner-verify', 'sole');
    return () => property.verifyOwnership(asset.id, 'owner-verify');
  });
});

// ============================================================================
// Energy Service Benchmarks
// ============================================================================

describe('Energy Service Performance', () => {
  const energy = new EnergyService();

  bench('registerParticipant', () => {
    energy.registerParticipant(`producer-${Math.random()}`, 'producer', ['solar'], 100);
  });

  bench('submitReading', () => {
    const producer = energy.registerParticipant('reading-producer', 'producer', ['solar'], 100);
    return () => energy.submitReading(producer.id, 90, 10, 'solar');
  });

  bench('tradeEnergy', () => {
    const producer = energy.registerParticipant('trade-producer', 'producer', ['solar'], 100);
    const consumer = energy.registerParticipant('trade-consumer', 'consumer', [], 0);
    energy.submitReading(producer.id, 100, 0, 'solar');
    return () => energy.tradeEnergy(producer.id, consumer.id, 50, 'solar', 0.10);
  });

  bench('requestGridBalance', () => {
    const gridOp = energy.registerParticipant('grid-op', 'grid_operator', ['grid'], 10000);
    return () => energy.requestGridBalance(gridOp.id, 'demand', 500, 0.15, 'high');
  });

  bench('getGridStats', () => {
    energy.getGridStats();
  });
});

// ============================================================================
// Media Service Benchmarks
// ============================================================================

describe('Media Service Performance', () => {
  const media = new MediaService();

  bench('publishContent', () => {
    media.publishContent(
      `creator-${Math.random()}`,
      'Article',
      'Benchmark Article',
      'Performance test content',
      `hash-${Math.random()}`,
      'ipfs://benchmark'
    );
  });

  bench('recordView', () => {
    const content = media.publishContent('view-creator', 'Article', 'View Test', 'Test', 'hash', 'ipfs://view');
    return () => media.recordView(content.id, 'viewer-did');
  });

  bench('grantLicense', () => {
    const content = media.publishContent('license-creator', 'Article', 'License Test', 'Test', 'hash2', 'ipfs://license');
    return () => media.grantLicense(content.id, 'licensee-did', 'CCBY');
  });

  bench('searchContent', () => {
    for (let i = 0; i < 50; i++) {
      media.publishContent('search-creator', 'Article', `Article ${i}`, `Content ${i}`, `hash-${i}`, `ipfs://${i}`, 'CCBY', ['test']);
    }
    return () => media.searchContent('Article', ['test']);
  });
});

// ============================================================================
// Justice Service Benchmarks
// ============================================================================

describe('Justice Service Performance', () => {
  const justice = new JusticeService();

  bench('fileCase', () => {
    justice.fileCase(
      `complainant-${Math.random()}`,
      `respondent-${Math.random()}`,
      'Benchmark Case',
      'Performance test dispute',
      'contract'
    );
  });

  bench('submitEvidence', () => {
    const caseRecord = justice.fileCase('c', 'r', 'Evidence Case', 'Test', 'property');
    return () =>
      justice.submitEvidence(caseRecord.id, 'submitter', 'document', 'Evidence Title', 'Evidence content', 'hash123');
  });

  bench('registerMediator', () => {
    justice.registerMediator(`mediator-${Math.random()}`, ['contract', 'property']);
  });

  bench('createRestorativeCircle', () => {
    const caseRecord = justice.fileCase('harmed', 'responsible', 'Circle Case', 'Test', 'interpersonal');
    return () =>
      justice.createRestorativeCircle(caseRecord.id, 'facilitator', [
        { did: 'harmed', role: 'harmed' },
        { did: 'responsible', role: 'responsible' },
      ]);
  });
});

// ============================================================================
// Knowledge Service Benchmarks
// ============================================================================

describe('Knowledge Service Performance', () => {
  const knowledge = new KnowledgeService();

  bench('submitClaim', () => {
    knowledge.submitClaim(
      `author-${Math.random()}`,
      'Benchmark Claim',
      'Performance test claim content',
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N2_Regional,
      ['benchmark', 'test']
    );
  });

  bench('submitEvidence', () => {
    const claim = knowledge.submitClaim(
      'evidence-author',
      'Evidence Claim',
      'Claim for evidence',
      EmpiricalLevel.E1_Testimonial,
      NormativeLevel.N1_Local
    );
    return () => knowledge.submitEvidence(claim.id, 'submitter', 'empirical', 'Evidence content', 'https://source.com');
  });

  bench('endorseClaim', () => {
    const claim = knowledge.submitClaim(
      'endorse-author',
      'Endorse Claim',
      'Claim for endorsement',
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N1_Local
    );
    return () => knowledge.endorseClaim(claim.id, `endorser-${Math.random()}`);
  });

  bench('search', () => {
    for (let i = 0; i < 50; i++) {
      knowledge.submitClaim(
        'search-author',
        `Claim ${i}`,
        `Content about ${i}`,
        EmpiricalLevel.E1_Testimonial,
        NormativeLevel.N1_Local,
        ['searchable']
      );
    }
    return () => knowledge.search('Claim', ['searchable']);
  });
});

// ============================================================================
// Cross-hApp Bridge Benchmarks
// ============================================================================

describe('Cross-hApp Bridge Performance', () => {
  bench('LocalBridge setup with 8 hApps', () => {
    const bridge = new LocalBridge();
    ['identity', 'governance', 'finance', 'property', 'energy', 'media', 'justice', 'knowledge'].forEach((happ) =>
      bridge.registerHapp(happ)
    );
  });

  bench('reputation propagation across 8 hApps', () => {
    const bridge = new LocalBridge();
    const happs = ['identity', 'governance', 'finance', 'property', 'energy', 'media', 'justice', 'knowledge'];
    happs.forEach((happ) => bridge.registerHapp(happ));

    return () => {
      const agent = 'multi-happ-agent';
      happs.forEach((happ) => {
        let rep = matl.createReputation(agent);
        rep = matl.recordPositive(rep);
        bridge.setReputation(happ, agent, rep);
      });
      bridge.getAggregateReputation(agent);
    };
  });

  bench('getCrossHappReputation with 8 domains', () => {
    const bridge = new LocalBridge();
    const happs = ['identity', 'governance', 'finance', 'property', 'energy', 'media', 'justice', 'knowledge'];
    happs.forEach((happ) => bridge.registerHapp(happ));

    const agent = 'cross-happ-agent';
    happs.forEach((happ) => {
      let rep = matl.createReputation(agent);
      rep = matl.recordPositive(rep);
      bridge.setReputation(happ, agent, rep);
    });

    return () => bridge.getCrossHappReputation(agent);
  });
});

// ============================================================================
// End-to-End Workflow Benchmarks
// ============================================================================

describe('End-to-End Workflow Performance', () => {
  bench('complete identity + credential + verification flow', () => {
    const identity = new IdentityService();

    // Create identities
    const issuer = identity.createIdentity('e2e-issuer-key');
    const holder = identity.createIdentity('e2e-holder-key');

    // Build issuer trust
    for (let i = 0; i < 5; i++) {
      const peer = identity.createIdentity(`peer-${i}-key`);
      identity.attestTrust(peer.did, issuer.did);
    }

    // Issue credential
    const cred = identity.issueCredential(issuer.did, holder.did, 'E2ECredential', { test: true });

    // Verify
    identity.verifyCredential(cred.id);
  });

  bench('complete proposal + voting + finalization flow', () => {
    resetGovernanceService();
    const governance = new GovernanceService();

    // Register members
    for (let i = 0; i < 5; i++) {
      governance.registerMember({
        did: `voter-${i}`,
        daoId: 'e2e-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });
    }

    // Create proposal
    const proposal = governance.createProposal({
      title: 'E2E Proposal',
      description: 'Complete flow test',
      proposerId: 'voter-0',
      daoId: 'e2e-dao',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    // Cast votes
    for (let i = 0; i < 5; i++) {
      governance.castVote({
        proposalId: proposal.id,
        voterId: `voter-${i}`,
        choice: i < 4 ? 'approve' : 'reject',
        weight: 100,
      });
    }

    // Finalize
    governance.finalizeProposal(proposal.id);
  });

  bench('complete energy production + trade + settlement flow', () => {
    resetEnergyService();
    resetFinanceService();
    const energy = new EnergyService();
    const finance = new FinanceService();

    // Setup participants
    const producer = energy.registerParticipant('e2e-producer', 'producer', ['solar'], 100);
    const consumer = energy.registerParticipant('e2e-consumer', 'consumer', [], 0);

    const producerWallet = finance.createWallet('e2e-producer', 'business');
    const consumerWallet = finance.createWallet('e2e-consumer', 'personal');
    consumerWallet.balances.set('MCX', 1000);

    // Produce energy
    energy.submitReading(producer.id, 100, 0, 'solar');

    // Trade
    const trade = energy.tradeEnergy(producer.id, consumer.id, 50, 'solar', 0.10);

    // Settle financially
    finance.transfer(consumerWallet.id, producerWallet.id, 5, 'MCX', 'Energy purchase');
  });
});
