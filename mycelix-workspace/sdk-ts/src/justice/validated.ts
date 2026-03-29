// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Justice Validated Clients
 *
 * Provides input-validated versions of all Justice clients.
 * All inputs are validated using Zod schemas before being sent to the conductor.
 *
 * @packageDocumentation
 * @module justice/validated
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  type ZomeCallable,
  type HolochainRecord,
  type Case,
  type FileCaseInput,
  type Evidence,
  type SubmitEvidenceInput,
  type Decision,
  type DecisionOutcome,
  type Remedy,
  type MediatorProfile,
  type ArbitratorProfile,
  type RegisterMediatorInput,
  type RegisterArbitratorInput,
  type Enforcement,
  type RequestEnforcementInput,
  type EnforcementStatus,
  type CaseCategory,
} from './index.js';

// ============================================================================
// Validation Schemas
// ============================================================================

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID (start with "did:")' });

const caseCategorySchema = z.enum([
  'ContractDispute',
  'ConductViolation',
  'PropertyDispute',
  'FinancialDispute',
  'GovernanceDispute',
  'IdentityDispute',
  'IPDispute',
  'Other',
]);

const evidenceTypeSchema = z.enum([
  'Document',
  'Testimony',
  'Transaction',
  'Screenshot',
  'Witness',
  'Other',
]);

const decisionOutcomeSchema = z.enum([
  'ForComplainant',
  'ForRespondent',
  'SplitDecision',
  'Dismissed',
  'Settled',
]);

const enforcementTypeSchema = z.enum([
  'ReputationPenalty',
  'TemporarySuspension',
  'PermanentBan',
  'AssetFreeze',
  'PaymentOrder',
  'RequiredAction',
]);

const enforcementStatusSchema = z.enum([
  'Requested',
  'Acknowledged',
  'Executed',
  'Rejected',
  'Appealed',
]);

const arbitratorTierSchema = z.enum(['Panel', 'Lead', 'Appeals']);

const fileCaseInputSchema = z.object({
  title: z.string().min(1, 'Title is required').max(200, 'Title too long'),
  description: z.string().min(10, 'Description must be at least 10 characters'),
  category: caseCategorySchema,
  respondent: didSchema,
});

const submitEvidenceInputSchema = z.object({
  case_id: z.string().min(1, 'Case ID is required'),
  type_: evidenceTypeSchema,
  title: z.string().min(1, 'Title is required'),
  description: z.string().min(1, 'Description is required'),
  content_hash: z.string().min(32, 'Content hash must be at least 32 characters'),
  storage_ref: z.string().optional(),
});

const remedySchema = z.object({
  type_: z.enum(['Compensation', 'Action', 'Apology', 'ReputationAdjustment', 'Ban']),
  target: didSchema,
  description: z.string().min(1),
  amount: z.number().min(0).optional(),
  deadline: z.number().positive().optional(),
  completed: z.boolean(),
});

const registerMediatorInputSchema = z.object({
  specializations: z.array(caseCategorySchema).min(1, 'At least one specialization required'),
});

const registerArbitratorInputSchema = z.object({
  specializations: z.array(caseCategorySchema).min(1, 'At least one specialization required'),
  tier: arbitratorTierSchema,
});

const requestEnforcementInputSchema = z.object({
  decision_id: z.string().min(1),
  target_happ: z.string().min(1),
  target_did: didSchema,
  action_type: enforcementTypeSchema,
  details: z.string().min(1),
  amount: z.number().min(0).optional(),
  deadline: z.number().positive().optional(),
});

// ============================================================================
// Validation Utility
// ============================================================================

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

// ============================================================================
// Validated Cases Client
// ============================================================================

/**
 * Validated Cases Client - All inputs are validated before zome calls
 */
export class ValidatedCasesClient {
  private client: CasesClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new CasesClient(zomeClient);
  }

  async fileCase(input: FileCaseInput): Promise<HolochainRecord<Case>> {
    validateOrThrow(fileCaseInputSchema, input, 'fileCase input');
    return this.client.fileCase(input);
  }

  async getCase(caseId: string): Promise<HolochainRecord<Case> | null> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    return this.client.getCase(caseId);
  }

  async getCasesByComplainant(complainantDid: string): Promise<HolochainRecord<Case>[]> {
    validateOrThrow(didSchema, complainantDid, 'complainantDid');
    return this.client.getCasesByComplainant(complainantDid);
  }

  async getCasesByRespondent(respondentDid: string): Promise<HolochainRecord<Case>[]> {
    validateOrThrow(didSchema, respondentDid, 'respondentDid');
    return this.client.getCasesByRespondent(respondentDid);
  }

  async getCasesByParty(did: string): Promise<HolochainRecord<Case>[]> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getCasesByParty(did);
  }

  async withdrawCase(caseId: string, reason: string): Promise<HolochainRecord<Case>> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    validateOrThrow(z.string().min(1, 'Reason is required'), reason, 'reason');
    return this.client.withdrawCase(caseId, reason);
  }

  async respondToCase(caseId: string, response: string): Promise<HolochainRecord<Case>> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    validateOrThrow(
      z.string().min(10, 'Response must be at least 10 characters'),
      response,
      'response'
    );
    return this.client.respondToCase(caseId, response);
  }
}

// ============================================================================
// Validated Evidence Client
// ============================================================================

/**
 * Validated Evidence Client
 */
export class ValidatedEvidenceClient {
  private client: EvidenceClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new EvidenceClient(zomeClient);
  }

  async submitEvidence(input: SubmitEvidenceInput): Promise<HolochainRecord<Evidence>> {
    validateOrThrow(submitEvidenceInputSchema, input, 'submitEvidence input');
    return this.client.submitEvidence(input);
  }

  async getEvidenceForCase(caseId: string): Promise<HolochainRecord<Evidence>[]> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    return this.client.getEvidenceForCase(caseId);
  }

  async verifyEvidence(evidenceId: string): Promise<HolochainRecord<Evidence>> {
    validateOrThrow(z.string().min(1), evidenceId, 'evidenceId');
    return this.client.verifyEvidence(evidenceId);
  }

  async challengeEvidence(evidenceId: string, reason: string): Promise<void> {
    validateOrThrow(z.string().min(1), evidenceId, 'evidenceId');
    validateOrThrow(
      z.string().min(10, 'Challenge reason must be at least 10 characters'),
      reason,
      'reason'
    );
    return this.client.challengeEvidence(evidenceId, reason);
  }
}

// ============================================================================
// Validated Arbitration Client
// ============================================================================

/**
 * Validated Arbitration Client
 */
export class ValidatedArbitrationClient {
  private client: ArbitrationClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new ArbitrationClient(zomeClient);
  }

  async registerAsMediator(
    input: RegisterMediatorInput
  ): Promise<HolochainRecord<MediatorProfile>> {
    validateOrThrow(registerMediatorInputSchema, input, 'registerAsMediator input');
    return this.client.registerAsMediator(input);
  }

  async registerAsArbitrator(
    input: RegisterArbitratorInput
  ): Promise<HolochainRecord<ArbitratorProfile>> {
    validateOrThrow(registerArbitratorInputSchema, input, 'registerAsArbitrator input');
    return this.client.registerAsArbitrator(input);
  }

  async getAvailableMediators(category: CaseCategory): Promise<HolochainRecord<MediatorProfile>[]> {
    validateOrThrow(caseCategorySchema, category, 'category');
    return this.client.getAvailableMediators(category);
  }

  async getAvailableArbitrators(
    category: CaseCategory
  ): Promise<HolochainRecord<ArbitratorProfile>[]> {
    validateOrThrow(caseCategorySchema, category, 'category');
    return this.client.getAvailableArbitrators(category);
  }

  async assignMediator(caseId: string, mediatorDid: string): Promise<HolochainRecord<Case>> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    validateOrThrow(didSchema, mediatorDid, 'mediatorDid');
    return this.client.assignMediator(caseId, mediatorDid);
  }

  async escalateToArbitration(
    caseId: string,
    arbitratorDids: string[]
  ): Promise<HolochainRecord<Case>> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    validateOrThrow(
      z.array(didSchema).min(1, 'At least one arbitrator required'),
      arbitratorDids,
      'arbitratorDids'
    );
    return this.client.escalateToArbitration(caseId, arbitratorDids);
  }

  async renderDecision(
    caseId: string,
    outcome: DecisionOutcome,
    reasoning: string,
    remedies: Remedy[]
  ): Promise<HolochainRecord<Decision>> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    validateOrThrow(decisionOutcomeSchema, outcome, 'outcome');
    validateOrThrow(
      z.string().min(20, 'Reasoning must be at least 20 characters'),
      reasoning,
      'reasoning'
    );
    validateOrThrow(z.array(remedySchema), remedies, 'remedies');
    return this.client.renderDecision(caseId, outcome, reasoning, remedies);
  }

  async getDecision(caseId: string): Promise<HolochainRecord<Decision> | null> {
    validateOrThrow(z.string().min(1), caseId, 'caseId');
    return this.client.getDecision(caseId);
  }

  async fileAppeal(decisionId: string, grounds: string): Promise<HolochainRecord<Case>> {
    validateOrThrow(z.string().min(1), decisionId, 'decisionId');
    validateOrThrow(
      z.string().min(20, 'Appeal grounds must be at least 20 characters'),
      grounds,
      'grounds'
    );
    return this.client.fileAppeal(decisionId, grounds);
  }
}

// ============================================================================
// Validated Enforcement Client
// ============================================================================

/**
 * Validated Enforcement Client
 */
export class ValidatedEnforcementClient {
  private client: EnforcementClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new EnforcementClient(zomeClient);
  }

  async requestEnforcement(input: RequestEnforcementInput): Promise<HolochainRecord<Enforcement>> {
    validateOrThrow(requestEnforcementInputSchema, input, 'requestEnforcement input');
    return this.client.requestEnforcement(input);
  }

  async getPendingEnforcements(targetHapp?: string): Promise<HolochainRecord<Enforcement>[]> {
    if (targetHapp !== undefined) {
      validateOrThrow(z.string().min(1), targetHapp, 'targetHapp');
    }
    return this.client.getPendingEnforcements(targetHapp);
  }

  async acknowledgeEnforcement(
    enforcementId: string,
    status: EnforcementStatus
  ): Promise<HolochainRecord<Enforcement>> {
    validateOrThrow(z.string().min(1), enforcementId, 'enforcementId');
    validateOrThrow(enforcementStatusSchema, status, 'status');
    return this.client.acknowledgeEnforcement(enforcementId, status);
  }

  async getEnforcementsForDecision(decisionId: string): Promise<HolochainRecord<Enforcement>[]> {
    validateOrThrow(z.string().min(1), decisionId, 'decisionId');
    return this.client.getEnforcementsForDecision(decisionId);
  }

  async markRemedyCompleted(
    decisionId: string,
    remedyIndex: number
  ): Promise<HolochainRecord<Decision>> {
    validateOrThrow(z.string().min(1), decisionId, 'decisionId');
    validateOrThrow(z.number().min(0), remedyIndex, 'remedyIndex');
    return this.client.markRemedyCompleted(decisionId, remedyIndex);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all validated Justice hApp clients
 */
export function createValidatedJusticeClients(client: ZomeCallable) {
  return {
    cases: new ValidatedCasesClient(client),
    evidence: new ValidatedEvidenceClient(client),
    arbitration: new ValidatedArbitrationClient(client),
    enforcement: new ValidatedEnforcementClient(client),
  };
}
