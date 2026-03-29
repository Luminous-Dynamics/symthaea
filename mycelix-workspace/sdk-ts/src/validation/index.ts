// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Validation Module
 *
 * Zod schemas for validating all bridge zome payloads.
 * Provides runtime type validation before sending to Holochain.
 *
 * @packageDocumentation
 * @module validation
 */

import { z } from 'zod';

// ============================================================================
// Common Schemas
// ============================================================================

/** Valid DID format */
export const didSchema = z.string()
  .startsWith('did:mycelix:')
  .min(20)
  .describe('Mycelix DID identifier');

/** Valid hApp ID */
export const happIdSchema = z.string()
  .min(1)
  .max(128)
  .regex(/^[a-z0-9-]+$/)
  .describe('hApp identifier');

/** MATL score (0.0 - 1.0) */
export const matlScoreSchema = z.number()
  .min(0)
  .max(1)
  .describe('MATL trust score (0.0-1.0)');

/** Positive amount */
export const amountSchema = z.number()
  .positive()
  .describe('Positive numeric amount');

/** Timestamp (microseconds since epoch) */
export const timestampSchema = z.number()
  .int()
  .nonnegative()
  .describe('Timestamp in microseconds');

/** Location coordinates */
export const locationSchema = z.object({
  lat: z.number().min(-90).max(90),
  lng: z.number().min(-180).max(180),
  radius_km: z.number().positive().optional(),
}).describe('Geographic location');

// ============================================================================
// Identity Bridge Schemas
// ============================================================================

export const registerHappInputSchema = z.object({
  happ_name: z.string().min(1).max(128),
  capabilities: z.array(z.string()).min(1),
}).describe('Register hApp with identity bridge');

export const queryIdentityInputSchema = z.object({
  did: didSchema,
  source_happ: happIdSchema,
  requested_fields: z.array(z.string()).optional().default([]),
}).describe('Query identity from another hApp');

export const reportReputationInputSchema = z.object({
  did: didSchema,
  source_happ: happIdSchema,
  score: matlScoreSchema,
  interactions: z.number().int().nonnegative(),
}).describe('Report reputation for a DID');

export const bridgeEventTypeSchema = z.enum([
  'IdentityCreated',
  'IdentityUpdated',
  'CredentialIssued',
  'CredentialRevoked',
  'TrustAttested',
  'RecoveryInitiated',
]);

export const broadcastIdentityEventInputSchema = z.object({
  event_type: bridgeEventTypeSchema,
  did: didSchema.optional(),
  payload: z.string(),
}).describe('Broadcast identity event');

// ============================================================================
// Finance Bridge Schemas
// ============================================================================

export const creditPurposeSchema = z.enum([
  'LoanApplication',
  'TrustVerification',
  'MarketplaceTransaction',
  'PropertyPurchase',
  'EnergyInvestment',
]);

export const queryCreditInputSchema = z.object({
  did: didSchema,
  purpose: creditPurposeSchema,
  amount_requested: amountSchema.optional(),
}).describe('Query credit score for a DID');

export const processPaymentInputSchema = z.object({
  payee_did: didSchema,
  amount: amountSchema,
  currency: z.string().min(1).max(16).default('MCX'),
  target_happ: happIdSchema,
  reference_id: z.string().min(1).max(256),
}).describe('Process cross-hApp payment');

export const assetTypeSchema = z.enum([
  'RealEstate',
  'Vehicle',
  'Cryptocurrency',
  'EnergyAsset',
  'Equipment',
]);

export const registerCollateralInputSchema = z.object({
  asset_type: assetTypeSchema,
  asset_id: z.string().min(1),
  valuation: amountSchema,
}).describe('Register collateral from another hApp');

export const financeBridgeEventTypeSchema = z.enum([
  'PaymentProcessed',
  'LoanApproved',
  'LoanDefaulted',
  'CollateralPledged',
  'CollateralReleased',
  'CreditScoreUpdated',
]);

// ============================================================================
// Property Bridge Schemas
// ============================================================================

export const verificationTypeSchema = z.enum(['Current', 'Historical', 'Encumbrances']);

export const verifyOwnershipInputSchema = z.object({
  asset_id: z.string().min(1),
  verification_type: verificationTypeSchema,
}).describe('Verify ownership of an asset');

export const pledgeCollateralInputSchema = z.object({
  asset_id: z.string().min(1),
  pledge_to_happ: happIdSchema,
  pledge_amount: amountSchema,
  loan_reference: z.string().optional(),
}).describe('Pledge collateral for a loan');

export const propertyBridgeEventTypeSchema = z.enum([
  'OwnershipTransferred',
  'CollateralPledged',
  'CollateralReleased',
  'EncumbranceAdded',
  'EncumbranceRemoved',
  'PropertyRegistered',
]);

// ============================================================================
// Energy Bridge Schemas
// ============================================================================

export const energySourceSchema = z.enum([
  'solar',
  'wind',
  'hydro',
  'battery',
  'grid',
  'other_renewable',
]);

export const queryAvailableEnergyInputSchema = z.object({
  source_happ: happIdSchema,
  location: locationSchema.optional(),
  energy_sources: z.array(energySourceSchema).optional(),
  min_amount: amountSchema.optional(),
}).describe('Query available energy in the network');

export const requestEnergyPurchaseInputSchema = z.object({
  seller_did: didSchema,
  amount_kwh: amountSchema,
  max_price_per_kwh: amountSchema.optional(),
  source_happ: happIdSchema,
}).describe('Request energy purchase from a seller');

export const reportProductionInputSchema = z.object({
  participant_id: z.string().min(1),
  production_kwh: z.number().nonnegative(),
  consumption_kwh: z.number().nonnegative(),
  energy_source: energySourceSchema,
  timestamp: timestampSchema,
}).describe('Report energy production');

export const energyBridgeEventTypeSchema = z.enum([
  'EnergyListed',
  'EnergyPurchased',
  'SettlementCompleted',
  'CarbonCreditIssued',
  'GridBalanceRequest',
  'GridBalanceFulfilled',
]);

// ============================================================================
// Media Bridge Schemas
// ============================================================================

export const contentTypeSchema = z.enum([
  'Article',
  'Opinion',
  'Investigation',
  'Review',
  'Analysis',
  'Interview',
  'Report',
  'Editorial',
  'Other',
]);

export const licenseTypeSchema = z.enum([
  'CC0',
  'CCBY',
  'CCBYSA',
  'CCBYNC',
  'CCBYNCSA',
  'AllRightsReserved',
  'Custom',
]);

export const queryContentInputSchema = z.object({
  source_happ: happIdSchema,
  creator_did: didSchema.optional(),
  content_type: contentTypeSchema.optional(),
  tags: z.array(z.string()).optional(),
  license_types: z.array(licenseTypeSchema).optional(),
  limit: z.number().int().positive().max(100).optional(),
}).describe('Query content from the media network');

export const requestLicenseInputSchema = z.object({
  content_id: z.string().min(1),
  licensee_did: didSchema,
  license_type: licenseTypeSchema,
  purpose: z.string().min(1).max(1024),
  duration_days: z.number().int().positive().optional(),
}).describe('Request a license for content');

export const distributeRoyaltiesInputSchema = z.object({
  content_id: z.string().min(1),
  total_amount: amountSchema,
  currency: z.string().min(1).max(16).default('MCX'),
  reason: z.enum(['view', 'license', 'tip', 'subscription']),
}).describe('Distribute royalties to content creators');

export const verifyLicenseInputSchema = z.object({
  content_id: z.string().min(1),
  licensee_did: didSchema,
  source_happ: happIdSchema,
}).describe('Verify a license is valid');

export const mediaBridgeEventTypeSchema = z.enum([
  'ContentPublished',
  'ContentLicensed',
  'RoyaltyDistributed',
  'ContentFlagged',
  'ContentRemoved',
  'CreatorVerified',
]);

// ============================================================================
// Governance Bridge Schemas
// ============================================================================

export const governanceQueryTypeSchema = z.enum([
  'ProposalStatus',
  'VotingPower',
  'DelegationChain',
  'ExecutionHistory',
]);

export const queryGovernanceInputSchema = z.object({
  query_type: governanceQueryTypeSchema,
  query_params: z.string(),
  source_happ: happIdSchema,
}).describe('Query governance information');

export const requestExecutionInputSchema = z.object({
  proposal_hash: z.string().min(1),
  target_happ: happIdSchema,
  action: z.string().min(1).max(256),
  payload: z.string(),
}).describe('Request execution of a passed proposal');

export const governanceBridgeEventTypeSchema = z.enum([
  'ProposalCreated',
  'ProposalPassed',
  'ProposalRejected',
  'VoteCast',
  'ExecutionRequested',
  'ExecutionCompleted',
  'DelegationChanged',
]);

export const broadcastGovernanceEventInputSchema = z.object({
  event_type: governanceBridgeEventTypeSchema,
  proposal_hash: z.string().optional(),
  voter_did: didSchema.optional(),
  payload: z.string(),
}).describe('Broadcast governance event');

// ============================================================================
// Justice Bridge Schemas
// ============================================================================

export const crossHappDisputeTypeSchema = z.enum([
  'ContractBreach',
  'Fraud',
  'PropertyDispute',
  'Defamation',
  'CopyrightViolation',
  'TokenDispute',
  'GovernanceViolation',
  'Other',
]);

export const enforcementTypeSchema = z.enum([
  'ReputationPenalty',
  'TemporarySuspension',
  'PermanentBan',
  'AssetFreeze',
  'PaymentOrder',
  'RequiredAction',
]);

export const fileCrossHappDisputeInputSchema = z.object({
  dispute_type: crossHappDisputeTypeSchema,
  respondent_did: didSchema,
  related_happs: z.array(happIdSchema),
  title: z.string().min(1).max(256),
  description: z.string().min(1).max(4096),
  evidence_hashes: z.array(z.string()),
}).describe('File a cross-hApp dispute');

export const requestEnforcementInputSchema = z.object({
  dispute_id: z.string().min(1),
  target_happ: happIdSchema,
  target_did: didSchema,
  action_type: enforcementTypeSchema,
  details: z.string().min(1).max(1024),
  amount: amountSchema.optional(),
  deadline: timestampSchema.optional(),
}).describe('Request enforcement action on another hApp');

export const disputeHistoryQuerySchema = z.object({
  did: didSchema,
  role: z.enum(['complainant', 'respondent', 'both']),
  limit: z.number().int().positive().max(100).optional(),
}).describe('Query dispute history for a DID');

export const justiceBridgeEventTypeSchema = z.enum([
  'DisputeFiled',
  'DisputeResolved',
  'EnforcementRequested',
  'EnforcementExecuted',
  'AppealFiled',
  'MediationInitiated',
]);

// ============================================================================
// Knowledge Bridge Schemas
// ============================================================================

export const knowledgeQueryTypeSchema = z.enum([
  'VerifyClaim',
  'ClaimsBySubject',
  'EpistemicScore',
  'GraphTraversal',
  'FactCheck',
]);

export const queryKnowledgeInputSchema = z.object({
  query_type: knowledgeQueryTypeSchema,
  query_params: z.string(),
  source_happ: happIdSchema,
}).describe('Query knowledge from the knowledge graph');

export const factCheckInputSchema = z.object({
  claim_text: z.string().min(1).max(4096),
  source_happ: happIdSchema,
}).describe('Fact-check a claim against the knowledge graph');

export const registerExternalClaimInputSchema = z.object({
  claim_hash: z.string().min(1),
  source_happ: happIdSchema,
  title: z.string().min(1).max(256),
  empirical: matlScoreSchema,
  normative: matlScoreSchema,
  mythic: matlScoreSchema,
  author_did: didSchema,
}).describe('Register an external claim from another hApp');

export const knowledgeBridgeEventTypeSchema = z.enum([
  'ClaimCreated',
  'ClaimVerified',
  'ClaimDisputed',
  'EvidenceAdded',
  'SynthesisCompleted',
  'FactCheckResult',
]);

export const broadcastKnowledgeEventInputSchema = z.object({
  event_type: knowledgeBridgeEventTypeSchema,
  claim_hash: z.string().optional(),
  subject: z.string().optional(),
  payload: z.string(),
}).describe('Broadcast knowledge event');

// ============================================================================
// Validation Utilities
// ============================================================================

/**
 * Validate input and throw on error with detailed message
 */
export function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map(i => `${i.path.join('.')}: ${i.message}`);
    throw new Error(`Validation failed: ${errors.join('; ')}`);
  }
  return result.data;
}

/**
 * Validate input and return result object (safe version, doesn't throw)
 */
export function validateSafe<T>(schema: z.ZodSchema<T>, data: unknown): { success: true; data: T } | { success: false; errors: string[] } {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map(i => `${i.path.join('.')}: ${i.message}`);
    return { success: false, errors };
  }
  return { success: true, data: result.data };
}

/**
 * Create a validated zome call wrapper
 */
export function withValidation<TInput, TOutput>(
  inputSchema: z.ZodSchema<TInput>,
  fn: (input: TInput) => Promise<TOutput>
): (input: unknown) => Promise<TOutput> {
  return async (input: unknown) => {
    const validated = validateOrThrow(inputSchema, input);
    return fn(validated);
  };
}

// ============================================================================
// Schema Collections for Integration Clients
// ============================================================================

export const IdentitySchemas = {
  RegisterHappInput: registerHappInputSchema,
  QueryIdentityInput: queryIdentityInputSchema,
  ReportReputationInput: reportReputationInputSchema,
  BroadcastEventInput: broadcastIdentityEventInputSchema,
};

export const FinanceSchemas = {
  QueryCreditInput: queryCreditInputSchema,
  ProcessPaymentInput: processPaymentInputSchema,
  RegisterCollateralInput: registerCollateralInputSchema,
};

export const PropertySchemas = {
  VerifyOwnershipInput: verifyOwnershipInputSchema,
  PledgeCollateralInput: pledgeCollateralInputSchema,
};

export const EnergySchemas = {
  QueryAvailableEnergyInput: queryAvailableEnergyInputSchema,
  RequestEnergyPurchaseInput: requestEnergyPurchaseInputSchema,
  ReportProductionInput: reportProductionInputSchema,
};

export const MediaSchemas = {
  QueryContentInput: queryContentInputSchema,
  RequestLicenseInput: requestLicenseInputSchema,
  DistributeRoyaltiesInput: distributeRoyaltiesInputSchema,
  VerifyLicenseInput: verifyLicenseInputSchema,
};

export const GovernanceSchemas = {
  QueryGovernanceInput: queryGovernanceInputSchema,
  RequestExecutionInput: requestExecutionInputSchema,
  BroadcastEventInput: broadcastGovernanceEventInputSchema,
};

export const JusticeSchemas = {
  FileCrossHappDisputeInput: fileCrossHappDisputeInputSchema,
  RequestEnforcementInput: requestEnforcementInputSchema,
  DisputeHistoryQuery: disputeHistoryQuerySchema,
};

export const KnowledgeSchemas = {
  QueryKnowledgeInput: queryKnowledgeInputSchema,
  FactCheckInput: factCheckInputSchema,
  RegisterExternalClaimInput: registerExternalClaimInputSchema,
  BroadcastEventInput: broadcastKnowledgeEventInputSchema,
};

// Re-export zod for convenience
export { z } from 'zod';
