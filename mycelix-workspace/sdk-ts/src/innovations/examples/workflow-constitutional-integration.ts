// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Self-Healing Workflows + Constitutional AI Integration Example
 *
 * Demonstrates how workflow steps are governed by constitutional rules,
 * with automatic compensation when violations are detected.
 *
 * @module innovations/examples/workflow-constitutional-integration
 */

import { getConstitutionalGovernor } from '../constitutional-ai/index.js';
import {
  createWorkflow,
  createOrchestrator,
  type WorkflowContext,
} from '../self-healing-workflows/index.js';

/**
 * Mock hApp responses
 */
const mockResponses = {
  identity: { verified: true, did: 'did:mycelix:applicant-001', name: 'Jane Doe' },
  eligibility: { eligible: true, score: 0.89, reasoning: 'Below FPL threshold' },
  benefits: { approved: true, type: 'SNAP', amount: 234 },
};

/**
 * Example: Constitutionally-Governed Workflow
 */
export async function workflowConstitutionalExample() {
  console.log('=== Self-Healing Workflows + Constitutional AI ===\n');

  // Get constitutional governor
  const governor = await getConstitutionalGovernor();

  // Create orchestrator
  const orchestrator = createOrchestrator({
    maxConcurrentWorkflows: 10,
    enableObservability: true,
  });

  // Build workflow with constitutional checks
  const benefitWorkflow = createWorkflow('benefit-application', '1.0.0')
    .description('Process benefit application with constitutional governance')
    .timeout(60000)
    .compensateOnFailure(true)
    .step(
      'verify-identity',
      'identity',
      async (_input: unknown, ctx: WorkflowContext) => {
        ctx.log('Verifying applicant identity...');
        const result = mockResponses.identity;

        // Constitutional check
        const check = governor.checkContent(
          'workflow-identity',
          `Identity verified: ${result.name}`,
          { domain: 'benefits' }
        );

        if (!check.allowed) {
          throw new Error(`Constitutional violation: ${check.reason}`);
        }

        return result;
      },
      {
        critical: true,
        compensate: async (_i, _o, ctx) => ctx.log('COMPENSATE: Revoking verification'),
      }
    )
    .step(
      'check-eligibility',
      'governance',
      async (input: unknown, ctx: WorkflowContext) => {
        ctx.log('Checking eligibility...');
        const prev = input as typeof mockResponses.identity;
        return { applicantDid: prev.did, ...mockResponses.eligibility };
      },
      {
        dependsOn: ['verify-identity'],
        compensate: async (_i, _o, ctx) => ctx.log('COMPENSATE: Clearing eligibility'),
      }
    )
    .step(
      'provision-benefit',
      'finance',
      async (input: unknown, ctx: WorkflowContext) => {
        ctx.log('Provisioning benefit...');
        const prev = input as { eligible: boolean; applicantDid: string };

        if (!prev.eligible) {
          return { approved: false, reason: 'Not eligible' };
        }

        const check = governor.checkContent(
          'workflow-benefits',
          `Benefit approved: $${mockResponses.benefits.amount}/month`,
          { domain: 'benefits' }
        );

        if (!check.allowed) {
          throw new Error('Benefit provision blocked');
        }

        return { ...mockResponses.benefits, applicantDid: prev.applicantDid };
      },
      {
        dependsOn: ['check-eligibility'],
        compensate: async (_i, o, ctx) => {
          const output = o as { approved?: boolean } | undefined;
          if (output?.approved) ctx.log('COMPENSATE: Revoking benefit');
        },
      }
    )
    .build();

  console.log(`Workflow: ${benefitWorkflow.name} v${benefitWorkflow.version}`);
  console.log(`Steps: ${benefitWorkflow.steps.map(s => s.name).join(' → ')}`);
  console.log(`Compensation: ${benefitWorkflow.compensateOnFailure ? 'enabled' : 'disabled'}\n`);

  // Execute workflow
  console.log('Executing workflow...\n');
  const result = await orchestrator.execute(
    benefitWorkflow,
    {
      initiatorDid: 'did:mycelix:caseworker-001',
      input: { applicantId: 'applicant-001' },
      metadata: {},
    }
  );

  console.log(`\n=== Workflow Result ===`);
  console.log(`Status: ${result.status}`);
  console.log(`Duration: ${result.completedAt! - result.startedAt}ms`);

  // Constitutional audit
  const periodStart = Date.now() - 3600000; // Last hour
  const audit = governor.generateAudit(periodStart);
  console.log(`\n=== Constitutional Audit ===`);
  console.log(`Agents audited: ${audit.agentsAudited}`);
  console.log(`Interactions checked: ${audit.interactionsChecked}`);
  console.log(`Compliance rate: ${(audit.complianceRate * 100).toFixed(1)}%`);

  return { workflow: benefitWorkflow, result, audit };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  workflowConstitutionalExample().catch(console.error);
}
