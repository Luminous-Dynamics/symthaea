// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civilizational Workflows Example
 *
 * Demonstrates the pre-built cross-hApp workflows that coordinate
 * multiple Civilizational OS hApps for common operations:
 *
 * 1. Community Onboarding - New member joins the ecosystem
 * 2. Governance Proposal - Democratic decision making
 * 3. Credit Application - MATL-based lending
 * 4. Property Transfer - Escrowed ownership change
 * 5. Energy Trade - P2P renewable energy exchange
 * 6. Content Verification - Fact-checking with knowledge graph
 * 7. Justice Enforcement - Cross-hApp penalty execution
 *
 * Each workflow coordinates 4-7 hApps automatically, handling
 * verification, validation, and state management.
 */

import {
  executeProposalWorkflow,
  executeCreditApplicationWorkflow,
  executePropertyTransferWorkflow,
  executeEnergyTradeWorkflow,
  executeJusticeEnforcementWorkflow,
  executeContentVerificationWorkflow,
  executeCommunityOnboardingWorkflow,
  type WorkflowResult,
} from '../src/bridge/workflows.js';

// ============================================================================
// Helper: Print Workflow Results
// ============================================================================

function printWorkflowResult<T>(name: string, result: WorkflowResult<T>): void {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`📋 ${name}`);
  console.log('='.repeat(60));

  console.log(`Status: ${result.success ? '✅ SUCCESS' : '❌ FAILED'}`);
  console.log(`Duration: ${result.duration}ms`);

  if (result.error) {
    console.log(`Error: ${result.error}`);
  }

  console.log('\nWorkflow Steps:');
  result.steps.forEach((step, i) => {
    const icon =
      step.status === 'completed'
        ? '✓'
        : step.status === 'failed'
          ? '✗'
          : step.status === 'skipped'
            ? '○'
            : '…';
    console.log(`  ${i + 1}. [${icon}] ${step.name} (${step.happ})`);
    if (step.result && Object.keys(step.result).length > 0) {
      console.log(`      └─ ${JSON.stringify(step.result)}`);
    }
    if (step.error) {
      console.log(`      └─ Error: ${step.error}`);
    }
  });

  if (result.data) {
    console.log('\nOutput Data:');
    console.log(`  ${JSON.stringify(result.data, null, 2).replace(/\n/g, '\n  ')}`);
  }
}

// ============================================================================
// Example 1: Community Onboarding
// ============================================================================

async function exampleCommunityOnboarding(): Promise<void> {
  console.log('\n🌟 EXAMPLE 1: COMMUNITY ONBOARDING');
  console.log('A new member joins the Mycelix ecosystem');
  console.log('This workflow coordinates: Identity → Finance → Governance');

  const result = await executeCommunityOnboardingWorkflow({
    agentPubKey: 'uhCAk_new_member_public_key_xyz123',
    displayName: 'Alice Chen',
    communityDid: 'did:mycelix:solar-coop-sf',
    initialStake: 100, // Initial stake in community token
    skills: ['solar-installation', 'electrical-engineering'],
    referrerDid: 'did:mycelix:existing-member-bob',
  });

  printWorkflowResult('Community Onboarding', result);
}

// ============================================================================
// Example 2: Governance Proposal
// ============================================================================

async function exampleGovernanceProposal(): Promise<void> {
  console.log('\n🗳️ EXAMPLE 2: GOVERNANCE PROPOSAL');
  console.log('Creating a community proposal with identity verification');
  console.log('This workflow coordinates: Identity → Governance → Knowledge');

  const result = await executeProposalWorkflow({
    proposerDid: 'did:mycelix:alice-verified',
    daoId: 'solar-coop-sf',
    title: 'Expand Community Solar Array by 50kW',
    description: `
      Proposal to install additional solar panels on the community center roof.
      - Cost: 15,000 MCX from treasury
      - Expected ROI: 18% annually
      - Environmental impact: 75 tons CO2 offset/year
    `,
    votingPeriodHours: 168, // 1 week
    quorumPercentage: 0.51,
  });

  printWorkflowResult('Governance Proposal', result);
}

// ============================================================================
// Example 3: Credit Application
// ============================================================================

async function exampleCreditApplication(): Promise<void> {
  console.log('\n💳 EXAMPLE 3: CREDIT APPLICATION');
  console.log('MATL-based credit with collateral verification');
  console.log('This workflow coordinates: Identity → Property → Finance → Knowledge');

  const result = await executeCreditApplicationWorkflow({
    applicantDid: 'did:mycelix:borrower-charlie',
    amount: 10000,
    currency: 'MCX',
    purpose: 'Purchase solar installation equipment for community project',
    termMonths: 24,
    collateralAssetId: 'asset:property:warehouse-lot-7',
  });

  printWorkflowResult('Credit Application', result);
}

// ============================================================================
// Example 4: Property Transfer with Escrow
// ============================================================================

async function examplePropertyTransfer(): Promise<void> {
  console.log('\n🏠 EXAMPLE 4: PROPERTY TRANSFER');
  console.log('Escrowed property transfer with lien checks');
  console.log('This workflow coordinates: Property → Finance → Identity → Justice');

  const result = await executePropertyTransferWorkflow({
    assetId: 'asset:property:solar-farm-lot-12',
    fromOwner: 'did:mycelix:seller-dana',
    toOwner: 'did:mycelix:buyer-evan',
    considerationAmount: 50000,
    currency: 'MCX',
    escrowDurationDays: 30,
  });

  printWorkflowResult('Property Transfer', result);
}

// ============================================================================
// Example 5: P2P Energy Trade
// ============================================================================

async function exampleEnergyTrade(): Promise<void> {
  console.log('\n⚡ EXAMPLE 5: P2P ENERGY TRADE');
  console.log('Renewable energy trade with automatic REC issuance');
  console.log('This workflow coordinates: Energy → Finance → Identity → Knowledge');

  const result = await executeEnergyTradeWorkflow({
    sellerDid: 'did:mycelix:solar-producer-frank',
    buyerDid: 'did:mycelix:consumer-grace',
    amountKwh: 500,
    pricePerKwh: 0.12,
    source: 'solar',
    deliveryWindow: {
      start: Date.now(),
      end: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
    },
  });

  printWorkflowResult('P2P Energy Trade', result);
}

// ============================================================================
// Example 6: Content Verification
// ============================================================================

async function exampleContentVerification(): Promise<void> {
  console.log('\n📰 EXAMPLE 6: CONTENT VERIFICATION');
  console.log('Fact-checking content through the knowledge graph');
  console.log('This workflow coordinates: Media → Knowledge → Identity → Governance');

  const result = await executeContentVerificationWorkflow({
    creatorDid: 'did:mycelix:journalist-henry',
    contentType: 'article',
    title: 'Community Solar Reduces Grid Costs by 15%',
    contentHash: 'QmXyz123abc456def789...',
    claims: [
      'claim:energy-savings-15-percent',
      'claim:grid-stability-improved',
      'claim:community-investment-roi',
    ],
    storageUri: 'ipfs://QmContent...',
  });

  printWorkflowResult('Content Verification', result);
}

// ============================================================================
// Example 7: Justice Enforcement
// ============================================================================

async function exampleJusticeEnforcement(): Promise<void> {
  console.log('\n⚖️ EXAMPLE 7: JUSTICE ENFORCEMENT');
  console.log('Cross-hApp enforcement of a judgment');
  console.log('This workflow coordinates: Justice → Finance → Governance → Identity');

  const result = await executeJusticeEnforcementWorkflow({
    caseId: 'case:contract-dispute-2024-001',
    judgmentId: 'judgment:case-001-final',
    targetDid: 'did:mycelix:respondent-iris',
    penalties: [
      {
        type: 'financial',
        happId: 'finance',
        amount: 500,
        currency: 'MCX',
        description: 'Damages for breach of contract',
      },
      {
        type: 'reputation',
        happId: 'governance',
        adjustment: -0.15,
        description: 'Voting power reduction for 6 months',
      },
    ],
    notifyParties: ['did:mycelix:complainant-jack'],
  });

  printWorkflowResult('Justice Enforcement', result);
}

// ============================================================================
// Main: Run All Examples
// ============================================================================

async function main(): Promise<void> {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║     MYCELIX CIVILIZATIONAL WORKFLOWS DEMONSTRATION       ║');
  console.log('║                                                          ║');
  console.log('║  Pre-built workflows that coordinate multiple hApps      ║');
  console.log('║  for seamless civilizational operations                  ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  try {
    // Run all examples
    await exampleCommunityOnboarding();
    await exampleGovernanceProposal();
    await exampleCreditApplication();
    await examplePropertyTransfer();
    await exampleEnergyTrade();
    await exampleContentVerification();
    await exampleJusticeEnforcement();

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 WORKFLOW SUMMARY');
    console.log('='.repeat(60));
    console.log(`
The Mycelix Civilizational OS provides these pre-built workflows:

┌─────────────────────────────────────────────────────────────┐
│ Workflow                  │ hApps Coordinated              │
├─────────────────────────────────────────────────────────────┤
│ Community Onboarding      │ Identity, Finance, Governance  │
│ Governance Proposal       │ Identity, Governance, Knowledge│
│ Credit Application        │ Identity, Property, Finance    │
│ Property Transfer         │ Property, Finance, Identity    │
│ P2P Energy Trade          │ Energy, Finance, Identity      │
│ Content Verification      │ Media, Knowledge, Identity     │
│ Justice Enforcement       │ Justice, Finance, Governance   │
└─────────────────────────────────────────────────────────────┘

Each workflow automatically:
• Verifies identities and permissions
• Checks cross-hApp reputation via MATL
• Validates inputs using Zod schemas
• Tracks step-by-step progress
• Handles errors gracefully
• Emits real-time signals

For custom workflows, use the CrossHappBridge directly:
  import { getCrossHappBridge } from '@mycelix/sdk';
  const bridge = getCrossHappBridge();
  await bridge.requestVerification(...);
  await bridge.queryReputation(...);
`);

    console.log('\n✅ All workflow examples completed successfully!');
  } catch (error) {
    console.error('\n❌ Example failed:', error);
    process.exit(1);
  }
}

// Run
main();
