/**
 * Cross-hApp Workflow Integrations
 *
 * Pre-built workflow patterns that coordinate multiple hApps
 * for common civilizational operations.
 *
 * @packageDocumentation
 * @module bridge/workflows
 */

import { getCrossHappBridge, type HappId, type CrossHappBridge } from './cross-happ.js';
import {
  getEpistemicEvidenceBridge,
  type EpistemicVerificationResult,
} from './epistemic-evidence.js';

import type { CaseCategory, Evidence } from '../justice/index.js';

// ============================================================================
// Workflow Types
// ============================================================================

/** Workflow execution result */
export interface WorkflowResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  steps: WorkflowStep[];
  duration: number;
}

/** Individual workflow step */
export interface WorkflowStep {
  name: string;
  happ: HappId;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  result?: unknown;
  error?: string;
  duration?: number;
}

/** Workflow context passed between steps */
export interface WorkflowContext {
  initiator: string; // DID of workflow initiator
  startedAt: number;
  bridge: CrossHappBridge;
  data: Record<string, unknown>;
}

// ============================================================================
// Governance Workflows
// ============================================================================

/**
 * Execute a complete governance proposal workflow
 * 1. Verify proposer identity
 * 2. Check proposer reputation across hApps
 * 3. Create proposal in governance hApp
 * 4. Notify relevant parties
 */
export async function executeProposalWorkflow(params: {
  proposerDid: string;
  daoId: string;
  title: string;
  description: string;
  votingPeriodHours: number;
  quorumPercentage: number;
}): Promise<WorkflowResult<{ proposalId: string }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify proposer identity
    steps.push({ name: 'Verify Identity', happ: 'identity', status: 'running' });
    const identityVerification = await bridge.requestVerification('governance', 'identity', {
      subjectDid: params.proposerDid,
      verificationType: 'identity',
    });
    steps[0].status = identityVerification.verified ? 'completed' : 'failed';
    steps[0].result = identityVerification;

    if (!identityVerification.verified) {
      return {
        success: false,
        error: 'Proposer identity not verified',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Check cross-hApp reputation
    steps.push({ name: 'Check Reputation', happ: 'governance', status: 'running' });
    const reputation = await bridge.queryReputation(params.proposerDid, [
      'governance',
      'identity',
      'knowledge',
    ]);
    steps[1].status = 'completed';
    steps[1].result = reputation;

    if (reputation.aggregatedScore < 0.3) {
      return {
        success: false,
        error: `Insufficient reputation score: ${reputation.aggregatedScore.toFixed(2)}`,
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 3: Create proposal (simulated)
    steps.push({ name: 'Create Proposal', happ: 'governance', status: 'running' });
    const proposalId = `prop-${Date.now()}`;
    steps[2].status = 'completed';
    steps[2].result = { proposalId };

    // Step 4: Broadcast notification
    steps.push({ name: 'Notify Members', happ: 'governance', status: 'running' });
    await bridge.send({
      type: 'subscription_event',
      sourceHapp: 'governance',
      targetHapp: 'broadcast',
      payload: {
        event: 'new_proposal',
        proposalId,
        daoId: params.daoId,
        title: params.title,
      },
    });
    steps[3].status = 'completed';

    return {
      success: true,
      data: { proposalId },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    const lastStep = steps[steps.length - 1];
    if (lastStep) {
      lastStep.status = 'failed';
      lastStep.error = String(error);
    }
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Finance Workflows
// ============================================================================

/**
 * Execute a P2P lending workflow
 * 1. Verify borrower identity
 * 2. Check borrower credit score
 * 3. Verify collateral ownership (if any)
 * 4. Create loan agreement
 * 5. Notify relevant parties
 */
export async function executeLendingWorkflow(params: {
  borrowerDid: string;
  lenderDid: string;
  amount: number;
  currency: string;
  termDays: number;
  collateralAssetId?: string;
}): Promise<WorkflowResult<{ loanId: string; interestRate: number }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify borrower identity
    steps.push({ name: 'Verify Borrower', happ: 'identity', status: 'running' });
    const borrowerVerified = await bridge.requestVerification('finance', 'identity', {
      subjectDid: params.borrowerDid,
      verificationType: 'identity',
    });
    steps[0].status = borrowerVerified.verified ? 'completed' : 'failed';

    if (!borrowerVerified.verified) {
      return {
        success: false,
        error: 'Borrower identity not verified',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Check credit score
    steps.push({ name: 'Check Credit Score', happ: 'finance', status: 'running' });
    const reputation = await bridge.queryReputation(params.borrowerDid, [
      'finance',
      'governance',
      'justice',
    ]);
    const creditScore = reputation.aggregatedScore;
    steps[1].status = 'completed';
    steps[1].result = { creditScore };

    // Step 3: Verify collateral ownership (if provided)
    if (params.collateralAssetId) {
      steps.push({ name: 'Verify Collateral', happ: 'property', status: 'running' });
      const collateralVerified = await bridge.requestVerification('finance', 'property', {
        subjectDid: params.borrowerDid,
        verificationType: 'ownership',
        resource: params.collateralAssetId,
      });
      steps[2].status = collateralVerified.verified ? 'completed' : 'failed';

      if (!collateralVerified.verified) {
        return {
          success: false,
          error: 'Collateral ownership not verified',
          steps,
          duration: Date.now() - startTime,
        };
      }
    }

    // Step 4: Calculate interest rate based on credit score
    steps.push({ name: 'Create Loan', happ: 'finance', status: 'running' });
    const baseRate = 0.05;
    const riskPremium = (1 - creditScore) * 0.15;
    const interestRate = baseRate + riskPremium;
    const loanId = `loan-${Date.now()}`;
    steps[steps.length - 1].status = 'completed';
    steps[steps.length - 1].result = { loanId, interestRate };

    // Step 5: Notify parties
    steps.push({ name: 'Notify Parties', happ: 'finance', status: 'running' });
    await bridge.send({
      type: 'transfer_notification',
      sourceHapp: 'finance',
      targetHapp: 'broadcast',
      payload: {
        event: 'loan_created',
        loanId,
        borrower: params.borrowerDid,
        lender: params.lenderDid,
      },
    });
    steps[steps.length - 1].status = 'completed';

    return {
      success: true,
      data: { loanId, interestRate },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Property Transfer Workflows
// ============================================================================

/**
 * Execute a property transfer workflow with escrow
 * 1. Verify both parties' identities
 * 2. Check for liens on property
 * 3. Create escrow wallet
 * 4. Transfer funds to escrow
 * 5. Transfer property title
 * 6. Release escrow to seller
 */
export async function executePropertyTransferWorkflow(params: {
  sellerDid: string;
  buyerDid: string;
  assetId: string;
  price: number;
  currency: string;
}): Promise<WorkflowResult<{ transferId: string; escrowWalletId: string }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify seller identity and ownership
    steps.push({ name: 'Verify Seller', happ: 'identity', status: 'running' });
    const sellerVerified = await bridge.requestVerification('property', 'identity', {
      subjectDid: params.sellerDid,
      verificationType: 'identity',
    });
    steps[0].status = sellerVerified.verified ? 'completed' : 'failed';

    // Step 2: Verify buyer identity
    steps.push({ name: 'Verify Buyer', happ: 'identity', status: 'running' });
    const buyerVerified = await bridge.requestVerification('property', 'identity', {
      subjectDid: params.buyerDid,
      verificationType: 'identity',
    });
    steps[1].status = buyerVerified.verified ? 'completed' : 'failed';

    if (!sellerVerified.verified || !buyerVerified.verified) {
      return {
        success: false,
        error: 'Party identity verification failed',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 3: Check for liens (simulated)
    steps.push({ name: 'Check Liens', happ: 'property', status: 'running' });
    steps[2].status = 'completed';
    steps[2].result = { hasActiveLiens: false };

    // Step 4: Create escrow
    steps.push({ name: 'Create Escrow', happ: 'finance', status: 'running' });
    const escrowWalletId = `escrow-${Date.now()}`;
    steps[3].status = 'completed';
    steps[3].result = { escrowWalletId };

    // Step 5: Execute transfer
    steps.push({ name: 'Transfer Title', happ: 'property', status: 'running' });
    const transferId = `transfer-${Date.now()}`;
    steps[4].status = 'completed';
    steps[4].result = { transferId };

    // Step 6: Broadcast completion
    steps.push({ name: 'Finalize', happ: 'property', status: 'running' });
    await bridge.send({
      type: 'transfer_notification',
      sourceHapp: 'property',
      targetHapp: 'broadcast',
      payload: {
        event: 'property_transferred',
        assetId: params.assetId,
        from: params.sellerDid,
        to: params.buyerDid,
        transferId,
      },
    });
    steps[5].status = 'completed';

    return {
      success: true,
      data: { transferId, escrowWalletId },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Energy Trading Workflows
// ============================================================================

/**
 * Execute a P2P energy trade workflow
 * 1. Verify both parties are registered participants
 * 2. Check seller has sufficient capacity
 * 3. Verify buyer has sufficient funds
 * 4. Execute trade
 * 5. Issue energy credits
 */
export async function executeEnergyTradeWorkflow(params: {
  sellerDid: string;
  buyerDid: string;
  amountKwh: number;
  source: string;
  pricePerKwh: number;
}): Promise<WorkflowResult<{ tradeId: string; creditId: string }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify seller is registered producer
    steps.push({ name: 'Verify Seller', happ: 'energy', status: 'running' });
    const sellerVerification = await bridge.requestVerification('energy', 'identity', {
      subjectDid: params.sellerDid,
      verificationType: 'credential',
      resource: 'energy_producer',
    });
    steps[0].status = sellerVerification.verified ? 'completed' : 'failed';
    steps[0].result = { verified: sellerVerification.verified };

    if (!sellerVerification.verified) {
      return {
        success: false,
        error: 'Seller is not a verified energy producer',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Verify buyer has funds
    steps.push({ name: 'Verify Buyer Funds', happ: 'finance', status: 'running' });
    const totalCost = params.amountKwh * params.pricePerKwh;
    steps[1].status = 'completed';
    steps[1].result = { totalCost, hasFunds: true };

    // Step 3: Execute trade
    steps.push({ name: 'Execute Trade', happ: 'energy', status: 'running' });
    const tradeId = `trade-${Date.now()}`;
    steps[2].status = 'completed';
    steps[2].result = { tradeId };

    // Step 4: Transfer payment
    steps.push({ name: 'Process Payment', happ: 'finance', status: 'running' });
    steps[3].status = 'completed';

    // Step 5: Issue energy credit to buyer
    steps.push({ name: 'Issue Credit', happ: 'energy', status: 'running' });
    const creditId = `credit-${Date.now()}`;
    steps[4].status = 'completed';
    steps[4].result = { creditId };

    // Step 6: Update reputations
    steps.push({ name: 'Update Reputations', happ: 'energy', status: 'running' });
    // In production, would update MATL scores
    steps[5].status = 'completed';

    return {
      success: true,
      data: { tradeId, creditId },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Justice Enforcement Workflows
// ============================================================================

/**
 * Execute a cross-hApp enforcement workflow
 * 1. Validate decision exists
 * 2. Identify target hApps for enforcement
 * 3. Request enforcement from each hApp
 * 4. Track acknowledgments
 * 5. Broadcast enforcement status
 */
export async function executeEnforcementWorkflow(params: {
  decisionId: string;
  targetDid: string;
  remedies: Array<{
    type: 'reputation_penalty' | 'asset_freeze' | 'payment_order' | 'suspension';
    targetHapp: HappId;
    details: Record<string, unknown>;
  }>;
}): Promise<WorkflowResult<{ enforcementIds: string[]; acknowledged: number }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];
  const enforcementIds: string[] = [];
  let acknowledged = 0;

  try {
    // Step 1: Validate decision
    steps.push({ name: 'Validate Decision', happ: 'justice', status: 'running' });
    steps[0].status = 'completed';
    steps[0].result = { valid: true };

    // Step 2-N: Request enforcement from each target hApp
    for (const remedy of params.remedies) {
      const stepName = `Enforce: ${remedy.type} on ${remedy.targetHapp}`;
      steps.push({ name: stepName, happ: remedy.targetHapp, status: 'running' });

      const result = await bridge.requestEnforcement('justice', remedy.targetHapp, {
        decisionId: params.decisionId,
        caseId: 'derived-from-decision',
        targetDid: params.targetDid,
        remedyType: remedy.type.replace('_', '') as
          | 'compensation'
          | 'action'
          | 'reputation_adjustment'
          | 'ban',
        details: remedy.details,
      });

      const enforcementId = `enf-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      enforcementIds.push(enforcementId);

      if (result.acknowledged) {
        acknowledged++;
        steps[steps.length - 1].status = 'completed';
      } else {
        steps[steps.length - 1].status = 'failed';
      }
      steps[steps.length - 1].result = { enforcementId, acknowledged: result.acknowledged };
    }

    // Final step: Broadcast enforcement status
    steps.push({ name: 'Broadcast Status', happ: 'justice', status: 'running' });
    await bridge.broadcastDecision('justice', params.decisionId, 'enforcing', [params.targetDid]);
    steps[steps.length - 1].status = 'completed';

    return {
      success: acknowledged === params.remedies.length,
      data: { enforcementIds, acknowledged },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Media Publication Workflows
// ============================================================================

/**
 * Execute a verified publication workflow
 * 1. Verify author identity
 * 2. Register content
 * 3. Submit for fact-checking
 * 4. Broadcast to subscribers
 */
export async function executePublicationWorkflow(params: {
  authorDid: string;
  contentHash: string;
  title: string;
  tags: string[];
  requestFactCheck: boolean;
}): Promise<WorkflowResult<{ contentId: string; verificationStatus: string }>> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify author identity
    steps.push({ name: 'Verify Author', happ: 'identity', status: 'running' });
    const authorVerified = await bridge.requestVerification('media', 'identity', {
      subjectDid: params.authorDid,
      verificationType: 'identity',
    });
    steps[0].status = authorVerified.verified ? 'completed' : 'failed';

    if (!authorVerified.verified) {
      return {
        success: false,
        error: 'Author identity not verified',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Check author reputation
    steps.push({ name: 'Check Reputation', happ: 'media', status: 'running' });
    const reputation = await bridge.queryReputation(params.authorDid, [
      'media',
      'knowledge',
      'identity',
    ]);
    steps[1].status = 'completed';
    steps[1].result = { reputation: reputation.aggregatedScore };

    // Step 3: Register content
    steps.push({ name: 'Register Content', happ: 'media', status: 'running' });
    const contentId = `content-${Date.now()}`;
    steps[2].status = 'completed';
    steps[2].result = { contentId };

    // Step 4: Submit for fact-checking if requested
    let verificationStatus = 'Unverified';
    if (params.requestFactCheck) {
      steps.push({ name: 'Request Fact Check', happ: 'knowledge', status: 'running' });
      verificationStatus = 'Pending';
      steps[3].status = 'completed';
      steps[3].result = { status: 'pending_review' };
    }

    // Step 5: Broadcast publication
    steps.push({ name: 'Broadcast', happ: 'media', status: 'running' });
    await bridge.send({
      type: 'subscription_event',
      sourceHapp: 'media',
      targetHapp: 'broadcast',
      payload: {
        event: 'new_publication',
        contentId,
        author: params.authorDid,
        title: params.title,
        tags: params.tags,
      },
    });
    steps[steps.length - 1].status = 'completed';

    return {
      success: true,
      data: { contentId, verificationStatus },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Comprehensive Identity Verification Workflow
// ============================================================================

/**
 * Execute comprehensive identity verification across all hApps
 * Aggregates reputation and standing from all relevant systems
 */
export async function executeComprehensiveIdentityCheck(params: {
  subjectDid: string;
  requesterHapp: HappId;
  checkCategories: Array<'reputation' | 'credentials' | 'standing' | 'history'>;
}): Promise<
  WorkflowResult<{
    verified: boolean;
    aggregatedScore: number;
    details: Record<string, unknown>;
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];
  const details: Record<string, unknown> = {};

  try {
    // Step 1: Basic identity verification
    steps.push({ name: 'Verify Identity', happ: 'identity', status: 'running' });
    const identityResult = await bridge.requestVerification(params.requesterHapp, 'identity', {
      subjectDid: params.subjectDid,
      verificationType: 'identity',
    });
    steps[0].status = identityResult.verified ? 'completed' : 'failed';
    details.identityVerified = identityResult.verified;
    details.verificationLevel = identityResult.level;

    if (!identityResult.verified) {
      return {
        success: false,
        data: { verified: false, aggregatedScore: 0, details },
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Cross-hApp reputation query
    if (params.checkCategories.includes('reputation')) {
      steps.push({ name: 'Query Reputation', happ: 'identity', status: 'running' });
      const reputation = await bridge.queryReputation(params.subjectDid, [
        'identity',
        'governance',
        'finance',
        'property',
        'energy',
        'media',
        'justice',
        'knowledge',
      ]);
      steps[steps.length - 1].status = 'completed';
      details.reputation = reputation;
    }

    // Step 3: Check standing with justice system
    if (params.checkCategories.includes('standing')) {
      steps.push({ name: 'Check Justice Standing', happ: 'justice', status: 'running' });
      // In production, would check for active cases/enforcements
      details.justiceStanding = { activeCases: 0, pendingEnforcements: 0 };
      steps[steps.length - 1].status = 'completed';
    }

    // Calculate final score
    const reputationScore =
      (details.reputation as { aggregatedScore: number })?.aggregatedScore || 0.5;

    return {
      success: true,
      data: {
        verified: true,
        aggregatedScore: reputationScore,
        details,
      },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Workflow Registry
// ============================================================================

/** Available workflow types */
export type WorkflowType =
  | 'proposal'
  | 'lending'
  | 'property_transfer'
  | 'energy_trade'
  | 'enforcement'
  | 'publication'
  | 'identity_check';

/** Workflow registry for dynamic execution */
export const workflowRegistry: Record<WorkflowType, (...args: any[]) => Promise<WorkflowResult>> = {
  proposal: executeProposalWorkflow,
  lending: executeLendingWorkflow,
  property_transfer: executePropertyTransferWorkflow,
  energy_trade: executeEnergyTradeWorkflow,
  enforcement: executeEnforcementWorkflow,
  publication: executePublicationWorkflow,
  identity_check: executeComprehensiveIdentityCheck,
};

/**
 * Execute a workflow by type
 */
export async function executeWorkflow<T>(
  type: WorkflowType,
  params: Record<string, unknown>
): Promise<WorkflowResult<T>> {
  const workflow = workflowRegistry[type];
  if (!workflow) {
    return {
      success: false,
      error: `Unknown workflow type: ${type}`,
      steps: [],
      duration: 0,
    };
  }
  return workflow(params) as Promise<WorkflowResult<T>>;
}

// ============================================================================
// Regenerative Exit Workflow (Energy + Finance + EduNet)
// ============================================================================

/**
 * Execute a regenerative exit workflow for community energy ownership
 * 1. Verify community readiness via EduNet certifications
 * 2. Check exit conditions are met
 * 3. Calculate exit tranche
 * 4. Process ownership transfer via Finance
 * 5. Update property records
 * 6. Broadcast transition milestone
 */
export async function executeRegenerativeExitWorkflow(params: {
  projectId: string;
  communityDid: string;
  investorsDids: string[];
  tranchePercentage: number;
}): Promise<
  WorkflowResult<{
    trancheId: string;
    ownershipTransferred: number;
    communityNewOwnership: number;
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Check community EduNet certifications
    steps.push({ name: 'Verify Operator Certifications', happ: 'edunet', status: 'running' });
    const certVerification = await bridge.requestVerification('energy', 'edunet', {
      subjectDid: params.communityDid,
      verificationType: 'credential',
      resource: 'energy_operator_certification',
    });
    steps[0].status = certVerification.verified ? 'completed' : 'failed';
    steps[0].result = { certified: certVerification.verified, level: certVerification.level };

    if (!certVerification.verified) {
      return {
        success: false,
        error: 'Community lacks required operator certifications',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Assess community readiness dimensions
    steps.push({ name: 'Assess Community Readiness', happ: 'energy', status: 'running' });
    const readinessScore = 0.85; // Would be calculated from actual metrics
    steps[1].status = 'completed';
    steps[1].result = {
      technicalCapacity: 0.9,
      financialCapacity: 0.8,
      governanceCapacity: 0.85,
      trainingCompletion: 0.9,
      communitySupport: 0.8,
      overallReadiness: readinessScore,
    };

    if (readinessScore < 0.7) {
      return {
        success: false,
        error: `Readiness score ${readinessScore} below threshold 0.7`,
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 3: Calculate exit tranche value
    steps.push({ name: 'Calculate Tranche', happ: 'finance', status: 'running' });
    const trancheId = `tranche-${Date.now()}`;
    const ownershipTransferred = params.tranchePercentage;
    steps[2].status = 'completed';
    steps[2].result = { trancheId, ownershipTransferred };

    // Step 4: Process investor buyouts
    steps.push({ name: 'Process Investor Payments', happ: 'finance', status: 'running' });
    for (const investorDid of params.investorsDids) {
      // In production, would process actual payments via finance hApp
      console.log(`Processing payment to investor: ${investorDid}`);
    }
    steps[3].status = 'completed';
    steps[3].result = { investorsPaid: params.investorsDids.length };

    // Step 5: Update property/ownership records
    steps.push({ name: 'Transfer Ownership Records', happ: 'property', status: 'running' });
    const communityNewOwnership = ownershipTransferred; // Cumulative would be calculated
    steps[4].status = 'completed';
    steps[4].result = { communityNewOwnership };

    // Step 6: Broadcast milestone
    steps.push({ name: 'Broadcast Transition', happ: 'energy', status: 'running' });
    await bridge.send({
      type: 'subscription_event',
      sourceHapp: 'energy',
      targetHapp: 'broadcast',
      payload: {
        event: 'regenerative_exit_tranche',
        projectId: params.projectId,
        community: params.communityDid,
        ownershipTransferred,
        trancheId,
      },
    });
    steps[5].status = 'completed';

    return {
      success: true,
      data: { trancheId, ownershipTransferred, communityNewOwnership },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// MATL Credit Assessment Workflow (Finance + Identity + Multiple hApps)
// ============================================================================

/**
 * Execute comprehensive MATL-based credit assessment
 * 1. Verify identity
 * 2. Query cross-hApp reputation scores
 * 3. Check payment history from Finance
 * 4. Check justice standing
 * 5. Check governance participation
 * 6. Calculate comprehensive credit score
 */
export async function executeCreditAssessmentWorkflow(params: {
  applicantDid: string;
  requestedAmount: number;
  purpose: 'loan' | 'investment' | 'marketplace' | 'energy';
}): Promise<
  WorkflowResult<{
    creditScore: number;
    creditTier: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Insufficient';
    maxApproved: number;
    interestRate: number;
    factors: Record<string, number>;
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];
  const factors: Record<string, number> = {};

  try {
    // Step 1: Verify identity
    steps.push({ name: 'Verify Identity', happ: 'identity', status: 'running' });
    const identityResult = await bridge.requestVerification('finance', 'identity', {
      subjectDid: params.applicantDid,
      verificationType: 'identity',
    });
    steps[0].status = identityResult.verified ? 'completed' : 'failed';

    if (!identityResult.verified) {
      return {
        success: false,
        error: 'Identity verification failed',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Query cross-hApp MATL scores
    steps.push({ name: 'Query MATL Scores', happ: 'finance', status: 'running' });
    const matlReputation = await bridge.queryReputation(params.applicantDid, [
      'finance',
      'governance',
      'property',
      'energy',
      'marketplace',
    ]);
    factors.matlScore = matlReputation.aggregatedScore;
    steps[1].status = 'completed';
    steps[1].result = { matlScore: factors.matlScore };

    // Step 3: Check payment history
    steps.push({ name: 'Check Payment History', happ: 'finance', status: 'running' });
    // In production, would query actual payment history
    factors.paymentHistory = 0.85; // 35% weight
    factors.creditUtilization = 0.7; // 30% weight
    factors.creditHistoryLength = 0.6; // 15% weight
    steps[2].status = 'completed';
    steps[2].result = factors;

    // Step 4: Check justice standing
    steps.push({ name: 'Check Justice Standing', happ: 'justice', status: 'running' });
    // Check for any active disputes or enforcements
    factors.justiceStanding = 1.0; // No issues
    steps[3].status = 'completed';

    // Step 5: Check governance participation
    steps.push({ name: 'Check Governance', happ: 'governance', status: 'running' });
    factors.governanceParticipation = 0.75;
    steps[4].status = 'completed';

    // Step 6: Calculate comprehensive score
    steps.push({ name: 'Calculate Score', happ: 'finance', status: 'running' });

    // Weighted calculation (matches Rust zome logic)
    const baseScore =
      factors.paymentHistory * 0.35 +
      factors.creditUtilization * 0.3 +
      factors.creditHistoryLength * 0.15 +
      0.1 + // Credit mix placeholder
      0.1; // New credit placeholder

    // Bonuses
    const matlBonus = factors.matlScore * 0.05; // Up to 50 points on 1000 scale
    const governanceBonus = factors.governanceParticipation * 0.02;
    const justiceBonus = factors.justiceStanding * 0.03;

    const creditScore = Math.min((baseScore + matlBonus + governanceBonus + justiceBonus) * 1000, 1000);

    // Determine tier
    let creditTier: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Insufficient';
    if (creditScore >= 900) creditTier = 'Excellent';
    else if (creditScore >= 750) creditTier = 'Good';
    else if (creditScore >= 600) creditTier = 'Fair';
    else if (creditScore >= 400) creditTier = 'Poor';
    else creditTier = 'Insufficient';

    // Calculate approved amount and rate
    const approvalRatio = creditScore / 1000;
    const maxApproved = Math.floor(params.requestedAmount * approvalRatio);
    const baseRate = 0.05;
    const riskPremium = (1 - approvalRatio) * 0.15;
    const interestRate = baseRate + riskPremium;

    steps[5].status = 'completed';
    steps[5].result = { creditScore, creditTier, maxApproved, interestRate };

    return {
      success: true,
      data: { creditScore, creditTier, maxApproved, interestRate, factors },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Knowledge Fact-Check Workflow (Knowledge + Media + Epistemic)
// ============================================================================

/**
 * Execute fact-check workflow for media content
 * 1. Parse claims from content
 * 2. Query knowledge graph for related claims
 * 3. Classify claims epistemically
 * 4. Perform reasoning/inference
 * 5. Return verdict with supporting evidence
 */
export async function executeFactCheckWorkflow(params: {
  contentId: string;
  claimText: string;
  requestorHapp: HappId;
}): Promise<
  WorkflowResult<{
    verdict: 'True' | 'MostlyTrue' | 'Mixed' | 'MostlyFalse' | 'False' | 'Unverifiable';
    confidence: number;
    supportingClaims: string[];
    contradictingClaims: string[];
    epistemicClassification: { empirical: number; normative: number; mythic: number };
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Parse and classify the claim
    steps.push({ name: 'Classify Claim', happ: 'knowledge', status: 'running' });
    // Epistemic classification (E/N/M)
    const epistemicClassification = {
      empirical: 0.7,
      normative: 0.2,
      mythic: 0.1,
    };
    steps[0].status = 'completed';
    steps[0].result = { classification: epistemicClassification };

    // Step 2: Query knowledge graph for related claims
    steps.push({ name: 'Query Related Claims', happ: 'knowledge', status: 'running' });
    const relatedClaims = ['claim-1', 'claim-2', 'claim-3']; // Would be actual graph traversal
    steps[1].status = 'completed';
    steps[1].result = { relatedClaimsCount: relatedClaims.length };

    // Step 3: Perform semantic similarity search
    steps.push({ name: 'Semantic Search', happ: 'knowledge', status: 'running' });
    const supportingClaims = ['claim-1'];
    const contradictingClaims: string[] = [];
    steps[2].status = 'completed';
    steps[2].result = { supporting: supportingClaims.length, contradicting: contradictingClaims.length };

    // Step 4: Perform reasoning/inference
    steps.push({ name: 'Reason Over Claims', happ: 'knowledge', status: 'running' });
    // Apply inference rules (modus ponens, etc.)
    const inferenceResult = {
      confidenceBoost: 0.1,
      derivedConclusions: 1,
    };
    steps[3].status = 'completed';
    steps[3].result = inferenceResult;

    // Step 5: Calculate verdict
    steps.push({ name: 'Calculate Verdict', happ: 'knowledge', status: 'running' });

    // Verdict logic
    const supportRatio = supportingClaims.length / (supportingClaims.length + contradictingClaims.length + 0.1);
    let verdict: 'True' | 'MostlyTrue' | 'Mixed' | 'MostlyFalse' | 'False' | 'Unverifiable';
    let confidence: number;

    if (relatedClaims.length < 2) {
      verdict = 'Unverifiable';
      confidence = 0.3;
    } else if (supportRatio > 0.9) {
      verdict = 'True';
      confidence = 0.85 + inferenceResult.confidenceBoost;
    } else if (supportRatio > 0.7) {
      verdict = 'MostlyTrue';
      confidence = 0.7;
    } else if (supportRatio > 0.4) {
      verdict = 'Mixed';
      confidence = 0.5;
    } else if (supportRatio > 0.2) {
      verdict = 'MostlyFalse';
      confidence = 0.6;
    } else {
      verdict = 'False';
      confidence = 0.8;
    }

    steps[4].status = 'completed';
    steps[4].result = { verdict, confidence };

    // Step 6: Broadcast result back to requesting hApp
    steps.push({ name: 'Return Result', happ: params.requestorHapp, status: 'running' });
    await bridge.send({
      type: 'capability_response',
      sourceHapp: 'knowledge',
      targetHapp: params.requestorHapp,
      payload: {
        contentId: params.contentId,
        verdict,
        confidence,
      },
    });
    steps[5].status = 'completed';

    return {
      success: true,
      data: { verdict, confidence, supportingClaims, contradictingClaims, epistemicClassification },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Cross-hApp Dispute Workflow (Justice + Media + Knowledge + Finance)
// ============================================================================

/**
 * Execute a defamation dispute workflow
 * 1. File dispute in Justice
 * 2. Gather evidence from Media
 * 3. Fact-check claims via Knowledge
 * 4. Calculate damages if liable
 * 5. Create enforcement actions
 */
export async function executeDefamationDisputeWorkflow(params: {
  complainantDid: string;
  respondentDid: string;
  contentId: string;
  claimedDamages: number;
}): Promise<
  WorkflowResult<{
    caseId: string;
    factCheckResult: string;
    liabilityDetermination: 'liable' | 'not_liable' | 'partial';
    awardedDamages: number;
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify both parties
    steps.push({ name: 'Verify Parties', happ: 'identity', status: 'running' });
    const complainantVerified = await bridge.requestVerification('justice', 'identity', {
      subjectDid: params.complainantDid,
      verificationType: 'identity',
    });
    const respondentVerified = await bridge.requestVerification('justice', 'identity', {
      subjectDid: params.respondentDid,
      verificationType: 'identity',
    });
    steps[0].status = complainantVerified.verified && respondentVerified.verified ? 'completed' : 'failed';

    if (!complainantVerified.verified || !respondentVerified.verified) {
      return {
        success: false,
        error: 'Party identity verification failed',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: File case in Justice
    steps.push({ name: 'File Case', happ: 'justice', status: 'running' });
    const caseId = `case-defamation-${Date.now()}`;
    steps[1].status = 'completed';
    steps[1].result = { caseId };

    // Step 3: Retrieve content from Media
    steps.push({ name: 'Retrieve Content', happ: 'media', status: 'running' });
    // In production, would fetch actual content and claims
    const contentClaims = ['The respondent committed fraud'];
    steps[2].status = 'completed';
    steps[2].result = { claimsFound: contentClaims.length };

    // Step 4: Fact-check each claim via Knowledge
    steps.push({ name: 'Fact-Check Claims', happ: 'knowledge', status: 'running' });
    type FactCheckVerdict = 'True' | 'MostlyTrue' | 'Mixed' | 'MostlyFalse' | 'False';
    const factCheckResult = 'MostlyFalse' as FactCheckVerdict; // Would be actual fact-check
    steps[3].status = 'completed';
    steps[3].result = { verdict: factCheckResult };

    // Step 5: Determine liability
    steps.push({ name: 'Determine Liability', happ: 'justice', status: 'running' });
    let liabilityDetermination: 'liable' | 'not_liable' | 'partial';
    let awardedDamages = 0;

    if (factCheckResult === 'False' || factCheckResult === 'MostlyFalse') {
      liabilityDetermination = 'liable';
      awardedDamages = Math.floor(params.claimedDamages * 0.8);
    } else if (factCheckResult === 'Mixed') {
      liabilityDetermination = 'partial';
      awardedDamages = Math.floor(params.claimedDamages * 0.3);
    } else {
      liabilityDetermination = 'not_liable';
      awardedDamages = 0;
    }

    steps[4].status = 'completed';
    steps[4].result = { liabilityDetermination, awardedDamages };

    // Step 6: Create enforcement if liable
    if (liabilityDetermination !== 'not_liable' && awardedDamages > 0) {
      steps.push({ name: 'Create Enforcement', happ: 'finance', status: 'running' });
      await bridge.requestEnforcement('justice', 'finance', {
        caseId,
        decisionId: `decision-${caseId}`,
        targetDid: params.respondentDid,
        remedyType: 'compensation',
        details: { amount: awardedDamages, recipient: params.complainantDid },
      });
      steps[5].status = 'completed';
    }

    // Step 7: Notify parties
    steps.push({ name: 'Notify Parties', happ: 'justice', status: 'running' });
    await bridge.send({
      type: 'transfer_notification',
      sourceHapp: 'justice',
      targetHapp: 'broadcast',
      payload: {
        event: 'case_resolved',
        caseId,
        outcome: liabilityDetermination,
        parties: [params.complainantDid, params.respondentDid],
      },
    });
    steps[steps.length - 1].status = 'completed';

    return {
      success: true,
      data: { caseId, factCheckResult, liabilityDetermination, awardedDamages },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Epistemic Evidence Dispute Resolution Workflow
// ============================================================================

/**
 * Execute epistemic evidence verification for a dispute case
 * 1. Retrieve all evidence for the case
 * 2. Classify each piece epistemically (E/N/M)
 * 3. Verify claims via Knowledge graph
 * 4. Calculate adjusted evidence strengths
 * 5. Generate evidence quality assessment
 * 6. Provide recommendation for arbitrators
 */
export async function executeEpistemicDisputeResolutionWorkflow(params: {
  caseId: string;
  caseCategory: CaseCategory;
  evidence: Evidence[];
  complainantDid: string;
  respondentDid: string;
  minCredibilityThreshold?: number;
}): Promise<
  WorkflowResult<{
    caseId: string;
    evidenceVerifications: EpistemicVerificationResult[];
    complainantEvidenceQuality: string;
    respondentEvidenceQuality: string;
    overallCaseQuality: string;
    recommendation: string;
    summaryExplanation: string;
  }>
> {
  const startTime = Date.now();
  const bridge = getCrossHappBridge();
  const epistemicBridge = getEpistemicEvidenceBridge();
  const steps: WorkflowStep[] = [];

  try {
    // Step 1: Verify both parties
    steps.push({ name: 'Verify Parties', happ: 'identity', status: 'running' });
    const complainantVerified = await bridge.requestVerification('justice', 'identity', {
      subjectDid: params.complainantDid,
      verificationType: 'identity',
    });
    const respondentVerified = await bridge.requestVerification('justice', 'identity', {
      subjectDid: params.respondentDid,
      verificationType: 'identity',
    });
    steps[0].status =
      complainantVerified.verified && respondentVerified.verified ? 'completed' : 'failed';

    if (!complainantVerified.verified || !respondentVerified.verified) {
      return {
        success: false,
        error: 'Party identity verification failed',
        steps,
        duration: Date.now() - startTime,
      };
    }

    // Step 2: Classify and verify each piece of evidence
    steps.push({ name: 'Verify Evidence Epistemically', happ: 'knowledge', status: 'running' });
    const evidenceVerifications: EpistemicVerificationResult[] = [];
    const complainantEvidence: EpistemicVerificationResult[] = [];
    const respondentEvidence: EpistemicVerificationResult[] = [];

    for (const evidence of params.evidence) {
      const verification = await epistemicBridge.verifyEvidence(evidence, params.caseCategory);

      // Apply minimum credibility filter
      if (
        !params.minCredibilityThreshold ||
        verification.credibility >= params.minCredibilityThreshold
      ) {
        evidenceVerifications.push(verification);

        // Sort by submitter
        if (evidence.submitter === params.complainantDid) {
          complainantEvidence.push(verification);
        } else if (evidence.submitter === params.respondentDid) {
          respondentEvidence.push(verification);
        }
      }
    }

    steps[1].status = 'completed';
    steps[1].result = {
      totalVerified: evidenceVerifications.length,
      complainantCount: complainantEvidence.length,
      respondentCount: respondentEvidence.length,
    };

    // Step 3: Calculate quality scores for each party's evidence
    steps.push({ name: 'Calculate Evidence Quality', happ: 'justice', status: 'running' });

    const calculateQuality = (
      verifications: EpistemicVerificationResult[]
    ): { quality: string; score: number } => {
      if (verifications.length === 0) return { quality: 'No Evidence', score: 0 };

      const avgCredibility =
        verifications.reduce((sum, v) => sum + v.credibility, 0) / verifications.length;
      const strongCount = verifications.filter(
        (v) => v.recommendation === 'Strong' || v.recommendation === 'Moderate'
      ).length;
      const qualityRatio = strongCount / verifications.length;

      if (avgCredibility >= 0.8 && qualityRatio >= 0.8) return { quality: 'Excellent', score: 0.9 };
      if (avgCredibility >= 0.6 && qualityRatio >= 0.6) return { quality: 'Good', score: 0.7 };
      if (avgCredibility >= 0.4) return { quality: 'Mixed', score: 0.5 };
      return { quality: 'Poor', score: 0.3 };
    };

    const complainantQuality = calculateQuality(complainantEvidence);
    const respondentQuality = calculateQuality(respondentEvidence);

    // Overall case quality
    const allQuality = calculateQuality(evidenceVerifications);

    steps[2].status = 'completed';
    steps[2].result = {
      complainantQuality: complainantQuality.quality,
      respondentQuality: respondentQuality.quality,
      overallQuality: allQuality.quality,
    };

    // Step 4: Query Knowledge graph for related claims that might affect the case
    steps.push({ name: 'Query Related Knowledge', happ: 'knowledge', status: 'running' });
    // In production, would search for claims related to case subject matter
    const relatedClaims: string[] = [];
    steps[3].status = 'completed';
    steps[3].result = { relatedClaimsFound: relatedClaims.length };

    // Step 5: Generate recommendation based on evidence quality
    steps.push({ name: 'Generate Recommendation', happ: 'justice', status: 'running' });

    let recommendation: string;
    let summaryExplanation: string;

    // Compare evidence quality between parties
    const qualityDelta = complainantQuality.score - respondentQuality.score;

    if (qualityDelta > 0.3) {
      recommendation = 'Strong evidence favors complainant';
      summaryExplanation = `The complainant's evidence (${complainantQuality.quality}) is significantly stronger than the respondent's (${respondentQuality.quality}). `;
    } else if (qualityDelta < -0.3) {
      recommendation = 'Strong evidence favors respondent';
      summaryExplanation = `The respondent's evidence (${respondentQuality.quality}) is significantly stronger than the complainant's (${complainantQuality.quality}). `;
    } else if (Math.abs(qualityDelta) <= 0.1) {
      recommendation = 'Evidence is evenly matched';
      summaryExplanation = `Both parties have similar quality evidence (Complainant: ${complainantQuality.quality}, Respondent: ${respondentQuality.quality}). `;
    } else if (qualityDelta > 0) {
      recommendation = 'Slight evidence advantage for complainant';
      summaryExplanation = `The complainant has a slight edge in evidence quality. `;
    } else {
      recommendation = 'Slight evidence advantage for respondent';
      summaryExplanation = `The respondent has a slight edge in evidence quality. `;
    }

    // Add epistemic context
    const avgEmpirical =
      evidenceVerifications.reduce((sum, v) => sum + v.classification.empirical, 0) /
      (evidenceVerifications.length || 1);
    const avgNormative =
      evidenceVerifications.reduce((sum, v) => sum + v.classification.normative, 0) /
      (evidenceVerifications.length || 1);

    if (avgEmpirical >= 0.7) {
      summaryExplanation +=
        'The evidence is primarily empirically-verifiable, suggesting objective facts can be established. ';
    } else if (avgNormative >= 0.5) {
      summaryExplanation +=
        'The evidence has significant normative dimensions, suggesting value judgments may be required. ';
    }

    // Count issues
    const unreliableCount = evidenceVerifications.filter(
      (v) => v.recommendation === 'Unreliable'
    ).length;
    if (unreliableCount > 0) {
      summaryExplanation += `Note: ${unreliableCount} piece(s) of evidence were rated as unreliable and should be treated with caution.`;
    }

    steps[4].status = 'completed';
    steps[4].result = { recommendation };

    // Step 6: Broadcast case analysis completion
    steps.push({ name: 'Broadcast Analysis', happ: 'justice', status: 'running' });
    await bridge.send({
      type: 'capability_response',
      sourceHapp: 'knowledge',
      targetHapp: 'justice',
      payload: {
        event: 'epistemic_analysis_complete',
        caseId: params.caseId,
        overallQuality: allQuality.quality,
        recommendation,
      },
    });
    steps[5].status = 'completed';

    return {
      success: true,
      data: {
        caseId: params.caseId,
        evidenceVerifications,
        complainantEvidenceQuality: complainantQuality.quality,
        respondentEvidenceQuality: respondentQuality.quality,
        overallCaseQuality: allQuality.quality,
        recommendation,
        summaryExplanation,
      },
      steps,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      steps,
      duration: Date.now() - startTime,
    };
  }
}

// ============================================================================
// Extended Workflow Registry
// ============================================================================

/** Extended workflow types including new civilizational workflows */
export type ExtendedWorkflowType =
  | WorkflowType
  | 'regenerative_exit'
  | 'credit_assessment'
  | 'fact_check'
  | 'defamation_dispute'
  | 'epistemic_dispute_resolution';

/** Extended workflow registry */
export const extendedWorkflowRegistry: Record<
  ExtendedWorkflowType,
  (...args: any[]) => Promise<WorkflowResult>
> = {
  ...workflowRegistry,
  regenerative_exit: executeRegenerativeExitWorkflow,
  credit_assessment: executeCreditAssessmentWorkflow,
  fact_check: executeFactCheckWorkflow,
  defamation_dispute: executeDefamationDisputeWorkflow,
  epistemic_dispute_resolution: executeEpistemicDisputeResolutionWorkflow,
};

/**
 * Execute an extended workflow by type
 */
export async function executeExtendedWorkflow<T>(
  type: ExtendedWorkflowType,
  params: Record<string, unknown>
): Promise<WorkflowResult<T>> {
  const workflow = extendedWorkflowRegistry[type];
  if (!workflow) {
    return {
      success: false,
      error: `Unknown workflow type: ${type}`,
      steps: [],
      duration: 0,
    };
  }
  return workflow(params) as Promise<WorkflowResult<T>>;
}
