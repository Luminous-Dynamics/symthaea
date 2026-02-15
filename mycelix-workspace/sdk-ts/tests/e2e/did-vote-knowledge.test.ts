/**
 * End-to-End Vertical Test: DID → Vote → Knowledge
 *
 * Tests the complete flow from identity creation through governance
 * participation to knowledge recording. This validates the core
 * civilizational stack integration.
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Core services
import { getIdentityService, resetIdentityService } from '../../src/integrations/identity/index.js';
import { getGovernanceService, resetGovernanceService } from '../../src/integrations/governance/index.js';
import { getKnowledgeService, resetKnowledgeService } from '../../src/integrations/knowledge/index.js';
import { getJusticeService, resetJusticeService } from '../../src/integrations/justice/index.js';
import { getFinanceService, resetFinanceService } from '../../src/integrations/finance/index.js';

// Bridge for cross-hApp coordination
import {
  getCrossHappBridge,
  resetCrossHappBridge,
  type HappId,
} from '../../src/bridge/cross-happ.js';

// Epistemic classification
import { EmpiricalLevel, NormativeLevel } from '../../src/epistemic/index.js';

// MATL for reputation
import { type ReputationScore, reputationValue } from '../../src/matl/index.js';

describe('E2E Vertical: DID → Vote → Knowledge', () => {
  beforeEach(() => {
    resetIdentityService();
    resetGovernanceService();
    resetKnowledgeService();
    resetJusticeService();
    resetFinanceService();
    resetCrossHappBridge();
  });

  describe('Complete Governance Lifecycle', () => {
    it('should complete the full identity → vote → knowledge flow', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const knowledge = getKnowledgeService();
      const bridge = getCrossHappBridge();

      // =========================================================
      // PHASE 1: Identity Creation and Verification
      // =========================================================

      // Create the proposer's identity
      const proposerProfile = identity.createIdentity('proposer-pubkey-32-chars-minimum');
      expect(proposerProfile.did).toMatch(/^did:mycelix:/);
      expect(proposerProfile.verificationLevel).toBe('self_attested');

      // Create voter identities
      const voter1 = identity.createIdentity('voter1-pubkey-32-characters-long');
      const voter2 = identity.createIdentity('voter2-pubkey-32-characters-long');
      const voter3 = identity.createIdentity('voter3-pubkey-32-characters-long');

      // Peer attestation for proposer (3 attestations = peer_verified)
      identity.attestTrust(voter1.did, proposerProfile.did);
      identity.attestTrust(voter2.did, proposerProfile.did);
      identity.attestTrust(voter3.did, proposerProfile.did);

      // Verify proposer is now peer verified
      const verifiedProposer = identity.getProfile(proposerProfile.did);
      expect(verifiedProposer?.verificationLevel).toBe('peer_verified');

      // =========================================================
      // PHASE 2: DAO Registration and Membership
      // =========================================================

      // Register all members in the DAO
      governance.registerMember({
        did: proposerProfile.did,
        daoId: 'civilizational-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: voter1.did,
        daoId: 'civilizational-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: voter2.did,
        daoId: 'civilizational-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: voter3.did,
        daoId: 'civilizational-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      // =========================================================
      // PHASE 3: Proposal Creation
      // =========================================================

      const proposal = governance.createProposal({
        title: 'Adopt Renewable Energy Standard',
        description: 'Require all community buildings to use 100% renewable energy by 2030',
        proposerId: proposerProfile.did,
        daoId: 'civilizational-dao',
        votingPeriodHours: 168, // 1 week
        quorumPercentage: 0.5,
      });

      expect(proposal.status).toBe('active');
      expect(proposal.title).toBe('Adopt Renewable Energy Standard');

      // =========================================================
      // PHASE 4: Voting
      // =========================================================

      // Proposer votes approve
      governance.castVote({
        proposalId: proposal.id,
        voterId: proposerProfile.did,
        choice: 'approve',
        weight: 100,
      });

      // Voter 1 votes approve
      governance.castVote({
        proposalId: proposal.id,
        voterId: voter1.did,
        choice: 'approve',
        weight: 100,
      });

      // Voter 2 votes approve
      governance.castVote({
        proposalId: proposal.id,
        voterId: voter2.did,
        choice: 'approve',
        weight: 100,
      });

      // Voter 3 votes reject
      governance.castVote({
        proposalId: proposal.id,
        voterId: voter3.did,
        choice: 'reject',
        weight: 100,
      });

      // =========================================================
      // PHASE 5: Finalization
      // =========================================================

      const result = governance.finalizeProposal(proposal.id);

      expect(result.passed).toBe(true);
      expect(result.quorumMet).toBe(true);
      expect(result.approvalPercentage).toBe(0.75); // 3/4 approved

      // =========================================================
      // PHASE 6: Record Decision in Knowledge Graph
      // =========================================================

      // Submit the governance decision as a normative claim
      const decisionClaim = knowledge.submitClaim(
        proposerProfile.did,
        `DAO Decision: ${proposal.title}`,
        `The Civilizational DAO voted to ${result.passed ? 'APPROVE' : 'REJECT'} the proposal: "${proposal.description}". Vote: ${result.approvalPercentage * 100}% approval, quorum ${result.quorumMet ? 'met' : 'not met'}.`,
        EmpiricalLevel.E2_Observed, // The vote occurred
        NormativeLevel.N3_Civilizational, // Community-wide norm
        ['governance', 'energy', 'climate', 'renewable']
      );

      expect(decisionClaim.status).toBe('proposed');
      expect(decisionClaim.classification.normative).toBe(NormativeLevel.N3_Civilizational);

      // Add evidence linking to the proposal (no sourceUrl for internal record)
      knowledge.submitEvidence(
        decisionClaim.id,
        proposerProfile.did,
        'empirical',
        `Proposal ID: ${proposal.id} - Voting record: ${JSON.stringify(result)}`
      );

      // Voters endorse the claim to verify it
      knowledge.endorseClaim(decisionClaim.id, voter1.did);
      knowledge.endorseClaim(decisionClaim.id, voter2.did);
      knowledge.endorseClaim(decisionClaim.id, voter3.did);

      // Additional endorsements for verification
      for (let i = 0; i < 2; i++) {
        const endorser = identity.createIdentity(`endorser-${i}-padding-to-32-characters`);
        knowledge.endorseClaim(decisionClaim.id, endorser.did);
      }

      // Claim should now be verified
      const verifiedClaim = knowledge.getClaim(decisionClaim.id);
      expect(verifiedClaim?.status).toBe('verified');

      // =========================================================
      // PHASE 7: Cross-hApp Bridge Coordination
      // =========================================================

      // Update reputations via bridge
      const positiveRep: ReputationScore = {
        agentId: proposerProfile.did,
        positiveCount: 10,
        negativeCount: 1,
        lastUpdate: Date.now(),
      };

      bridge.updateReputation(proposerProfile.did, 'governance', positiveRep);
      bridge.updateReputation(proposerProfile.did, 'knowledge', positiveRep);

      // Query cross-hApp reputation
      const aggregatedRep = bridge.queryReputation(
        proposerProfile.did,
        ['identity', 'governance', 'knowledge']
      );

      // Verify the complete cycle was recorded
      expect(verifiedClaim?.title).toContain('DAO Decision');
      expect(verifiedClaim?.content).toContain('APPROVE');
    });

    it('should handle rejected proposal flow', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const knowledge = getKnowledgeService();

      // Create identities
      const proposer = identity.createIdentity('proposer-key-32-characters-long-a');
      const voter1 = identity.createIdentity('voter1-key-32-characters-long-ab');
      const voter2 = identity.createIdentity('voter2-key-32-characters-long-bc');
      const voter3 = identity.createIdentity('voter3-key-32-characters-long-cd');

      // Register DAO members
      [proposer, voter1, voter2, voter3].forEach((p) => {
        governance.registerMember({
          did: p.did,
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
      });

      // Create controversial proposal
      const proposal = governance.createProposal({
        title: 'Controversial Change',
        description: 'A proposal that will be rejected',
        proposerId: proposer.did,
        daoId: 'test-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      // Majority rejects
      governance.castVote({ proposalId: proposal.id, voterId: proposer.did, choice: 'approve', weight: 100 });
      governance.castVote({ proposalId: proposal.id, voterId: voter1.did, choice: 'reject', weight: 100 });
      governance.castVote({ proposalId: proposal.id, voterId: voter2.did, choice: 'reject', weight: 100 });
      governance.castVote({ proposalId: proposal.id, voterId: voter3.did, choice: 'reject', weight: 100 });

      const result = governance.finalizeProposal(proposal.id);

      expect(result.passed).toBe(false);
      expect(result.approvalPercentage).toBe(0.25);

      // Record rejection in knowledge graph
      const rejectionClaim = knowledge.submitClaim(
        proposer.did,
        `DAO Decision: ${proposal.title} REJECTED`,
        `Proposal rejected with ${(1 - result.approvalPercentage) * 100}% opposition.`,
        EmpiricalLevel.E2_Observed,
        NormativeLevel.N2_Network,
        ['governance', 'decision', 'rejected']
      );

      expect(rejectionClaim.status).toBe('proposed');
    });
  });

  describe('Weighted Voting', () => {
    it('should support weighted voting in governance', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();

      // Create identities with varying voting power
      const expert = identity.createIdentity('expert-key-32-characters-long-ab');
      const member1 = identity.createIdentity('member1-key-32-characters-long-bc');
      const member2 = identity.createIdentity('member2-key-32-characters-long-cd');

      // Register members with different voting power
      governance.registerMember({
        did: expert.did,
        daoId: 'expert-dao',
        role: 'member',
        votingPower: 400, // Expert has 4x voting power
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: member1.did,
        daoId: 'expert-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: member2.did,
        daoId: 'expert-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      // Create proposal
      const proposal = governance.createProposal({
        title: 'Technical Standard Update',
        description: 'Update to the latest protocol version',
        proposerId: expert.did,
        daoId: 'expert-dao',
        votingPeriodHours: 48,
        quorumPercentage: 0.5,
      });

      // Expert votes approve with high weight
      governance.castVote({
        proposalId: proposal.id,
        voterId: expert.did,
        choice: 'approve',
        weight: 400,
      });

      // Both regular members vote reject
      governance.castVote({
        proposalId: proposal.id,
        voterId: member1.did,
        choice: 'reject',
        weight: 100,
      });

      governance.castVote({
        proposalId: proposal.id,
        voterId: member2.did,
        choice: 'reject',
        weight: 100,
      });

      const result = governance.finalizeProposal(proposal.id);

      // Expert's 400 votes > 200 reject votes (66% approval)
      expect(result.passed).toBe(true);
      expect(result.quorumMet).toBe(true);
    });
  });

  describe('Knowledge-Informed Governance', () => {
    it('should use verified knowledge claims to inform voting', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const knowledge = getKnowledgeService();

      // Create researcher identity
      const researcher = identity.createIdentity('researcher-key-32-characters-valid');

      // First, establish a verified knowledge claim
      const researchClaim = knowledge.submitClaim(
        researcher.did,
        'Solar Efficiency Study',
        'Solar panel efficiency has increased 15% in the last 5 years, making community solar economically viable.',
        EmpiricalLevel.E3_Measured,
        NormativeLevel.N1_Personal,
        ['energy', 'solar', 'research', 'economics']
      );

      // Add evidence
      knowledge.submitEvidence(
        researchClaim.id,
        researcher.did,
        'empirical',
        'Efficiency Data',
        'https://data.example.com/solar-efficiency-study-2024'
      );

      // Get endorsements
      for (let i = 0; i < 5; i++) {
        const endorser = identity.createIdentity(`endorser-${i}-padding-to-32-characters`);
        knowledge.endorseClaim(researchClaim.id, endorser.did);
      }

      const verifiedResearch = knowledge.getClaim(researchClaim.id);
      expect(verifiedResearch?.status).toBe('verified');

      // Now create governance proposal citing verified research
      const proposer = identity.createIdentity('proposer-key-32-characters-long-a');

      governance.registerMember({
        did: proposer.did,
        daoId: 'solar-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Community Solar Investment',
        description: `Based on verified research (${researchClaim.id}): "${verifiedResearch?.title}". Proposal to invest in community solar based on economic viability.`,
        proposerId: proposer.did,
        daoId: 'solar-dao',
        votingPeriodHours: 72,
        quorumPercentage: 0.6,
      });

      expect(proposal.description).toContain(researchClaim.id);
      expect(proposal.description).toContain('Solar Efficiency Study');
    });
  });

  describe('Dispute Resolution Integration', () => {
    it('should handle disputed governance decisions through justice system', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const justice = getJusticeService();

      // Create identities
      const winner = identity.createIdentity('winner-key-32-characters-long-ab');
      const loser = identity.createIdentity('loser-key-32-characters-long-abc');
      const neutral = identity.createIdentity('neutral-key-32-characters-long-a');

      // Register DAO members
      [winner, loser, neutral].forEach((p) => {
        governance.registerMember({
          did: p.did,
          daoId: 'dispute-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
      });

      // Create and pass proposal
      const proposal = governance.createProposal({
        title: 'Contested Decision',
        description: 'A decision that will be disputed',
        proposerId: winner.did,
        daoId: 'dispute-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      governance.castVote({ proposalId: proposal.id, voterId: winner.did, choice: 'approve', weight: 100 });
      governance.castVote({ proposalId: proposal.id, voterId: neutral.did, choice: 'approve', weight: 100 });
      governance.castVote({ proposalId: proposal.id, voterId: loser.did, choice: 'reject', weight: 100 });

      const result = governance.finalizeProposal(proposal.id);
      expect(result.passed).toBe(true);

      // Loser disputes the decision
      const dispute = justice.fileCase(
        loser.did,
        winner.did,
        'Governance Process Violation',
        `Disputing proposal ${proposal.id}: Alleges procedural irregularity in voting`,
        'GovernanceDispute'
      );

      expect(dispute.status).toBe('Active');
      expect(dispute.category).toBe('GovernanceDispute');

      // Submit evidence
      justice.submitEvidence(
        dispute.id,
        loser.did,
        'document',
        'Voting Record',
        'Evidence of procedural violation',
        'hash123'
      );

      // Register mediator
      justice.registerMediator('mediator-did-32-chars-minimum-ok', ['GovernanceDispute']);

      // Initiate mediation
      justice.initiateMediation(dispute.id, 'mediator-did-32-chars-minimum-ok');

      const updatedCase = justice.getCase(dispute.id);
      expect(updatedCase?.phase).toBe('Mediation');
    });
  });

  describe('Multi-DAO Coordination', () => {
    it('should coordinate decisions across multiple DAOs via knowledge graph', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const knowledge = getKnowledgeService();

      // Create cross-DAO member
      const crossDaoMember = identity.createIdentity('crossdao-member-32-characters-valid');

      // Register in multiple DAOs
      ['energy-dao', 'finance-dao', 'governance-dao'].forEach((daoId) => {
        governance.registerMember({
          did: crossDaoMember.did,
          daoId,
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
      });

      // Energy DAO decision
      const energyProposal = governance.createProposal({
        title: 'Solar Funding Request',
        description: 'Request 1M MCX for solar installation',
        proposerId: crossDaoMember.did,
        daoId: 'energy-dao',
        votingPeriodHours: 48,
        quorumPercentage: 0.5,
      });

      governance.castVote({
        proposalId: energyProposal.id,
        voterId: crossDaoMember.did,
        choice: 'approve',
        weight: 100,
      });

      const energyResult = governance.finalizeProposal(energyProposal.id);

      // Record in knowledge graph
      const energyClaim = knowledge.submitClaim(
        crossDaoMember.did,
        `Energy DAO Decision: ${energyProposal.title}`,
        `Energy DAO ${energyResult.passed ? 'approved' : 'rejected'} solar funding request.`,
        EmpiricalLevel.E2_Observed,
        NormativeLevel.N2_Network,
        ['energy-dao', 'solar', 'funding']
      );

      // Finance DAO can now see and reference this decision
      const financeProposal = governance.createProposal({
        title: 'Approve Energy DAO Funding',
        description: `Referencing Energy DAO decision (${energyClaim.id}): Approve treasury transfer of 1M MCX`,
        proposerId: crossDaoMember.did,
        daoId: 'finance-dao',
        votingPeriodHours: 48,
        quorumPercentage: 0.6,
      });

      expect(financeProposal.description).toContain(energyClaim.id);

      governance.castVote({
        proposalId: financeProposal.id,
        voterId: crossDaoMember.did,
        choice: 'approve',
        weight: 100,
      });

      const financeResult = governance.finalizeProposal(financeProposal.id);

      // Record finance decision
      const financeClaim = knowledge.submitClaim(
        crossDaoMember.did,
        `Finance DAO Decision: ${financeProposal.title}`,
        `Finance DAO ${financeResult.passed ? 'approved' : 'rejected'} the funding transfer. References: ${energyClaim.id}`,
        EmpiricalLevel.E2_Observed,
        NormativeLevel.N2_Network,
        ['finance-dao', 'treasury', 'transfer', 'solar']
      );

      // Search knowledge graph for related decisions
      const solarDecisions = knowledge.search('solar');
      expect(solarDecisions.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Complete Civilizational Stack', () => {
    it('should demonstrate full stack integration: Identity → Governance → Knowledge → Finance', async () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();
      const knowledge = getKnowledgeService();
      const finance = getFinanceService();
      const bridge = getCrossHappBridge();

      // =========================================================
      // STEP 1: Identity Foundation
      // =========================================================
      const communityLeader = identity.createIdentity('community-leader-32-characters-valid');
      const treasurer = identity.createIdentity('treasurer-key-32-characters-long');
      const members = [
        identity.createIdentity('member1-key-32-characters-long-a'),
        identity.createIdentity('member2-key-32-characters-long-b'),
        identity.createIdentity('member3-key-32-characters-long-c'),
      ];

      // Peer verification for leader
      members.forEach((m) => identity.attestTrust(m.did, communityLeader.did));
      expect(identity.getProfile(communityLeader.did)?.verificationLevel).toBe('peer_verified');

      // =========================================================
      // STEP 2: DAO Setup
      // =========================================================
      const allMembers = [communityLeader, treasurer, ...members];
      allMembers.forEach((m) => {
        governance.registerMember({
          did: m.did,
          daoId: 'community-dao',
          role: m === treasurer ? 'treasurer' : 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
      });

      // =========================================================
      // STEP 3: Create Proposal
      // =========================================================
      const fundingProposal = governance.createProposal({
        title: 'Community Solar Project Funding',
        description: 'Allocate 500,000 MCX from treasury for solar installation',
        proposerId: communityLeader.did,
        daoId: 'community-dao',
        votingPeriodHours: 168,
        quorumPercentage: 0.6,
      });

      // =========================================================
      // STEP 4: Voting Phase
      // =========================================================
      allMembers.forEach((m) => {
        governance.castVote({
          proposalId: fundingProposal.id,
          voterId: m.did,
          choice: 'approve',
          weight: 100,
        });
      });

      const votingResult = governance.finalizeProposal(fundingProposal.id);
      expect(votingResult.passed).toBe(true);
      expect(votingResult.approvalPercentage).toBe(1.0);

      // =========================================================
      // STEP 5: Record in Knowledge Graph
      // =========================================================
      const governanceDecision = knowledge.submitClaim(
        communityLeader.did,
        `Community Decision: ${fundingProposal.title}`,
        `APPROVED with ${votingResult.approvalPercentage * 100}% support. Funding of 500,000 MCX allocated for community solar project.`,
        EmpiricalLevel.E2_Observed,
        NormativeLevel.N3_Civilizational,
        ['governance', 'solar', 'funding', 'community', 'approved']
      );

      // Link evidence (no sourceUrl needed for internal governance records)
      knowledge.submitEvidence(
        governanceDecision.id,
        communityLeader.did,
        'empirical',
        `Voting Record - Proposal ${fundingProposal.id}: 100% approval, all 5 members voted`
      );

      // =========================================================
      // STEP 6: Financial Execution
      // =========================================================
      const daoTreasury = finance.createWallet('community-dao', 'cooperative');
      daoTreasury.balances.set('MCX', 1000000); // 1M MCX

      const solarProjectWallet = finance.createWallet('solar-project', 'project');

      const fundingTransfer = finance.transfer(
        daoTreasury.id,
        solarProjectWallet.id,
        500000,
        'MCX',
        `Governance-approved transfer: ${fundingProposal.id}`
      );

      expect(fundingTransfer.status).toBe('confirmed');
      expect(finance.getBalance(solarProjectWallet.id, 'MCX')).toBe(500000);
      expect(finance.getBalance(daoTreasury.id, 'MCX')).toBe(500000);

      // =========================================================
      // STEP 7: Record Financial Transaction in Knowledge Graph
      // =========================================================
      const financialRecord = knowledge.submitClaim(
        treasurer.did,
        'Treasury Transfer Executed',
        `500,000 MCX transferred to Solar Project wallet. Reference: Governance decision ${governanceDecision.id}`,
        EmpiricalLevel.E3_Measured,
        NormativeLevel.N2_Network,
        ['finance', 'treasury', 'transfer', 'solar', 'executed']
      );

      // =========================================================
      // STEP 8: Cross-hApp Bridge Coordination
      // =========================================================
      // Update reputations for all participants
      allMembers.forEach((m) => {
        const rep: ReputationScore = {
          agentId: m.did,
          positiveCount: 5,
          negativeCount: 0,
          lastUpdate: Date.now(),
        };
        bridge.updateReputation(m.did, 'governance', rep);
        bridge.updateReputation(m.did, 'identity', rep);
      });

      // Broadcast the decision
      await bridge.broadcastDecision(
        'governance',
        fundingProposal.id,
        'approved',
        allMembers.map((m) => m.did)
      );

      // =========================================================
      // VERIFICATION: Complete audit trail exists
      // =========================================================
      // 1. Identity trail
      const leaderProfile = identity.getProfile(communityLeader.did);
      expect(leaderProfile?.verificationLevel).toBe('peer_verified');

      // 2. Governance trail - status is 'passed' after finalization
      const finalProposal = governance.getProposal(fundingProposal.id);
      expect(finalProposal?.status).toBe('passed');

      // 3. Knowledge trail
      const decisionsClaims = knowledge.search('solar');
      expect(decisionsClaims.length).toBeGreaterThanOrEqual(1);

      // 4. Financial trail
      const treasuryHistory = finance.getTransactionHistory(daoTreasury.id);
      expect(treasuryHistory.length).toBe(1);
      expect(treasuryHistory[0].amount).toBe(500000);

      // 5. Bridge trail
      const messageHistory = bridge.getMessageHistory();
      expect(messageHistory.some((m) => m.type === 'decision_broadcast')).toBe(true);
    });
  });
});
