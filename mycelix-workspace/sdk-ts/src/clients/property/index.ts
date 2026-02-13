/**
 * Mycelix Property Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-property hApp.
 * Provides property registration, title transfers, disputes, and commons management.
 *
 * @module @mycelix/sdk/clients/property
 */

import type { AppClient } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
  entry: { Present: T };
}

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Property Types
// ============================================================================

export type PropertyType =
  | 'Land'
  | 'Building'
  | 'Vehicle'
  | 'Equipment'
  | 'IntellectualProperty'
  | 'Other';

export interface GeoLocation {
  latitude: number;
  longitude: number;
}

export interface Address {
  street?: string;
  city?: string;
  state?: string;
  country: string;
  postal_code?: string;
}

export interface PropertyMetadata {
  parcel_id?: string;
  zoning?: string;
  assessed_value?: number;
  [key: string]: unknown;
}

export interface CoOwner {
  did: string;
  share_percentage: number;
  acquired_at: number;
}

export interface Property {
  id: string;
  property_type: PropertyType;
  title: string;
  description: string;
  owner_did: string;
  co_owners: CoOwner[];
  geolocation?: GeoLocation;
  address?: Address;
  metadata: PropertyMetadata;
  registered: number;
  last_transfer?: number;
}

export interface RegisterPropertyInput {
  property_type: PropertyType;
  title: string;
  description: string;
  owner_did: string;
  co_owners: CoOwner[];
  geolocation?: GeoLocation;
  address?: Address;
  metadata: PropertyMetadata;
}

export interface LocationSearchInput {
  latitude: number;
  longitude: number;
  radius_km: number;
}

export interface UpdateMetadataInput {
  property_id: string;
  requester_did: string;
  metadata: PropertyMetadata;
}

// ============================================================================
// Encumbrance Types
// ============================================================================

export type EncumbranceType =
  | 'Mortgage'
  | 'Lien'
  | 'Easement'
  | 'Restriction'
  | 'Other';

export interface Encumbrance {
  encumbrance_type: EncumbranceType;
  holder_did: string;
  amount?: number;
  registered: number;
  expires?: number;
}

export interface AddEncumbranceInput {
  property_id: string;
  encumbrance_type: EncumbranceType;
  holder_did: string;
  amount?: number;
  expires?: number;
}

export interface RemoveEncumbranceInput {
  property_id: string;
  encumbrance_index: number;
}

// ============================================================================
// Title Deed Types
// ============================================================================

export type DeedType =
  | 'Original'
  | 'Transfer'
  | 'Inheritance'
  | 'CourtOrder';

export interface TitleDeed {
  id: string;
  property_id: string;
  owner_did: string;
  deed_type: DeedType;
  issued: number;
  previous_deed_id?: string;
  encumbrances: Encumbrance[];
}

export interface TransferOwnershipInput {
  property_id: string;
  from_did: string;
  to_did: string;
  transfer_type: string;
  transfer_id?: string;
}

export interface TransferOwnershipResult {
  property_action_hash: string;
  new_deed_id: string;
  deed_action_hash: string;
  previous_deed_id: string;
  encumbrances_carried: number;
}

export interface OwnershipRecord {
  deed_id: string;
  owner_did: string;
  deed_type: DeedType;
  issued: number;
  previous_deed_id?: string;
}

export interface VerifyOwnershipInput {
  property_id: string;
  did: string;
}

export interface AddCoOwnerInput {
  property_id: string;
  requester_did: string;
  co_owner: CoOwner;
}

export interface RemoveCoOwnerInput {
  property_id: string;
  requester_did: string;
  co_owner_did: string;
}

// ============================================================================
// Transfer Types
// ============================================================================

export type TransferType =
  | 'Sale'
  | 'Inheritance'
  | 'Gift'
  | 'CourtOrder'
  | 'Exchange'
  | 'Other';

export type TransferStatus =
  | 'Initiated'
  | 'AwaitingAcceptance'
  | 'InEscrow'
  | 'ConditionsPending'
  | 'Completed'
  | 'Cancelled'
  | 'Disputed';

export type ConditionType =
  | 'Inspection'
  | 'Financing'
  | 'TitleSearch'
  | 'Survey'
  | 'Appraisal'
  | 'Custom';

export interface TransferCondition {
  condition_type: ConditionType;
  description: string;
  satisfied: boolean;
  verified_by?: string;
}

export interface Transfer {
  id: string;
  property_id: string;
  from_did: string;
  to_did: string;
  transfer_type: TransferType;
  price?: number;
  currency?: string;
  conditions: TransferCondition[];
  status: TransferStatus;
  initiated: number;
  completed?: number;
}

export interface InitiateTransferInput {
  property_id: string;
  from_did: string;
  to_did: string;
  transfer_type: TransferType;
  price?: number;
  currency?: string;
  conditions: TransferCondition[];
}

export interface Escrow {
  id: string;
  transfer_id: string;
  escrow_agent_did?: string;
  amount: number;
  currency: string;
  funded: boolean;
  release_conditions: string[];
  created: number;
  released?: number;
}

export interface CreateEscrowInput {
  transfer_id: string;
  escrow_agent_did?: string;
  amount: number;
  currency: string;
  release_conditions: string[];
}

export interface SatisfyConditionInput {
  transfer_id: string;
  condition_index: number;
  verifier_did: string;
}

export interface CancelTransferInput {
  transfer_id: string;
  requester_did: string;
}

export interface AcceptTransferInput {
  transfer_id: string;
  requester_did: string;
}

export interface DisputeTransferInput {
  transfer_id: string;
  requester_did: string;
}

export interface AddConditionInput {
  transfer_id: string;
  condition_type: ConditionType;
  description: string;
}

// ============================================================================
// Dispute Types
// ============================================================================

export type DisputeType =
  | 'Boundary'
  | 'Ownership'
  | 'Encumbrance'
  | 'AccessRights'
  | 'Damage'
  | 'Other';

export type DisputeStatus =
  | 'Filed'
  | 'UnderReview'
  | 'Mediation'
  | 'Arbitration'
  | 'Resolved'
  | 'Dismissed';

export type ClaimBasis =
  | 'Inheritance'
  | 'Purchase'
  | 'AdversePossession'
  | 'Gift'
  | 'CourtJudgment';

export type ClaimStatus =
  | 'Pending'
  | 'UnderReview'
  | 'Approved'
  | 'Rejected'
  | 'Withdrawn';

export interface PropertyDispute {
  id: string;
  property_id: string;
  dispute_type: DisputeType;
  claimant_did: string;
  respondent_did: string;
  description: string;
  evidence_ids: string[];
  status: DisputeStatus;
  justice_case_id?: string;
  filed: number;
  resolved?: number;
}

export interface FileDisputeInput {
  property_id: string;
  dispute_type: DisputeType;
  claimant_did: string;
  respondent_did: string;
  description: string;
  evidence_ids: string[];
}

export interface OwnershipClaim {
  id: string;
  property_id: string;
  claimant_did: string;
  claim_basis: ClaimBasis;
  supporting_documents: string[];
  status: ClaimStatus;
  filed: number;
}

export interface FileClaimInput {
  property_id: string;
  claimant_did: string;
  claim_basis: ClaimBasis;
  supporting_documents: string[];
}

export interface EscalateInput {
  dispute_id: string;
  justice_case_id: string;
}

export interface ResolveDisputeInput {
  dispute_id: string;
  dismissed: boolean;
}

export interface UpdateDisputeStatusInput {
  dispute_id: string;
  new_status: DisputeStatus;
}

export interface UpdateClaimStatusInput {
  claim_id: string;
  new_status: ClaimStatus;
}

export interface AddEvidenceInput {
  dispute_id: string;
  evidence_id: string;
  submitter_did: string;
}

export interface AddDocumentInput {
  claim_id: string;
  document_id: string;
  submitter_did: string;
}

// ============================================================================
// Commons Types
// ============================================================================

export type ResourceType =
  | 'Land'
  | 'Water'
  | 'Forest'
  | 'Infrastructure'
  | 'Digital'
  | 'Other';

export type RightType =
  | 'Access'
  | 'Extraction'
  | 'Management'
  | 'Exclusion'
  | 'Alienation';

export interface GovernanceRules {
  decision_threshold: number;
  voting_period_days: number;
  quorum_percentage: number;
  [key: string]: unknown;
}

export interface CommonResource {
  id: string;
  name: string;
  description: string;
  resource_type: ResourceType;
  property_id?: string;
  stewards: string[];
  governance_rules: GovernanceRules;
  created: number;
}

export interface CreateResourceInput {
  name: string;
  description: string;
  resource_type: ResourceType;
  property_id?: string;
  stewards: string[];
  governance_rules: GovernanceRules;
}

export interface UsageRight {
  id: string;
  resource_id: string;
  holder_did: string;
  right_type: RightType;
  quota?: number;
  granted: number;
  expires?: number;
  active: boolean;
}

export interface GrantRightInput {
  resource_id: string;
  holder_did: string;
  right_type: RightType;
  quota?: number;
  expires?: number;
}

export interface UsageLog {
  id: string;
  resource_id: string;
  user_did: string;
  usage_type: string;
  quantity: number;
  unit: string;
  timestamp: number;
}

export interface LogUsageInput {
  resource_id: string;
  user_did: string;
  usage_type: string;
  quantity: number;
  unit: string;
}

export interface CheckQuotaInput {
  resource_id: string;
  user_did: string;
  requested_amount: number;
}

export interface RevokeRightInput {
  right_id: string;
  revoker_did: string;
}

export interface AddStewardInput {
  resource_id: string;
  new_steward_did: string;
  added_by_did: string;
}

export interface RemoveStewardInput {
  resource_id: string;
  steward_did: string;
  removed_by_did: string;
}

export interface UpdateGovernanceInput {
  resource_id: string;
  steward_did: string;
  new_rules: GovernanceRules;
}

export interface UserUsageInput {
  resource_id: string;
  user_did: string;
}

export interface UpdateQuotaInput {
  right_id: string;
  steward_did: string;
  new_quota?: number;
}

// ============================================================================
// Registry Client
// ============================================================================

const PROPERTY_ROLE = 'commons';
const REGISTRY_ZOME = 'property_registry';

/**
 * Registry Client - Property registration and management
 *
 * Covers 15 zome functions from registry coordinator
 */
export class RegistryClient {
  constructor(private readonly client: ZomeCallable) {}

  async registerProperty(input: RegisterPropertyInput): Promise<HolochainRecord<Property>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'register_property',
      payload: input,
    }) as Promise<HolochainRecord<Property>>;
  }

  async getProperty(propertyId: string): Promise<HolochainRecord<Property> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_property',
      payload: propertyId,
    }) as Promise<HolochainRecord<Property> | null>;
  }

  async getOwnerProperties(did: string): Promise<HolochainRecord<Property>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_owner_properties',
      payload: did,
    }) as Promise<HolochainRecord<Property>[]>;
  }

  async searchByLocation(input: LocationSearchInput): Promise<HolochainRecord<Property>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'search_by_location',
      payload: input,
    }) as Promise<HolochainRecord<Property>[]>;
  }

  async getTitleDeed(propertyId: string): Promise<HolochainRecord<TitleDeed> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_title_deed',
      payload: propertyId,
    }) as Promise<HolochainRecord<TitleDeed> | null>;
  }

  async getPropertyDeeds(propertyId: string): Promise<HolochainRecord<TitleDeed>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_property_deeds',
      payload: propertyId,
    }) as Promise<HolochainRecord<TitleDeed>[]>;
  }

  async updatePropertyMetadata(input: UpdateMetadataInput): Promise<HolochainRecord<Property>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'update_property_metadata',
      payload: input,
    }) as Promise<HolochainRecord<Property>>;
  }

  async addEncumbrance(input: AddEncumbranceInput): Promise<HolochainRecord<TitleDeed>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'add_encumbrance',
      payload: input,
    }) as Promise<HolochainRecord<TitleDeed>>;
  }

  async removeEncumbrance(input: RemoveEncumbranceInput): Promise<HolochainRecord<TitleDeed>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'remove_encumbrance',
      payload: input,
    }) as Promise<HolochainRecord<TitleDeed>>;
  }

  async getEncumbrances(propertyId: string): Promise<Encumbrance[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_encumbrances',
      payload: propertyId,
    }) as Promise<Encumbrance[]>;
  }

  async hasClearTitle(propertyId: string): Promise<boolean> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'has_clear_title',
      payload: propertyId,
    }) as Promise<boolean>;
  }

  async getPropertiesByType(propertyType: PropertyType): Promise<HolochainRecord<Property>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_properties_by_type',
      payload: propertyType,
    }) as Promise<HolochainRecord<Property>[]>;
  }

  async addCoOwner(input: AddCoOwnerInput): Promise<HolochainRecord<Property>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'add_co_owner',
      payload: input,
    }) as Promise<HolochainRecord<Property>>;
  }

  async removeCoOwner(input: RemoveCoOwnerInput): Promise<HolochainRecord<Property>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'remove_co_owner',
      payload: input,
    }) as Promise<HolochainRecord<Property>>;
  }

  async transferOwnership(input: TransferOwnershipInput): Promise<TransferOwnershipResult> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'transfer_ownership',
      payload: input,
    }) as Promise<TransferOwnershipResult>;
  }

  async getOwnershipHistory(propertyId: string): Promise<OwnershipRecord[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_ownership_history',
      payload: propertyId,
    }) as Promise<OwnershipRecord[]>;
  }

  async verifyOwnership(input: VerifyOwnershipInput): Promise<boolean> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'verify_ownership',
      payload: input,
    }) as Promise<boolean>;
  }
}

// ============================================================================
// Transfer Client
// ============================================================================

const TRANSFER_ZOME = 'property_transfer';

/**
 * Transfer Client - Property title transfers
 *
 * Covers 18 zome functions from transfer coordinator
 */
export class TransfersClient {
  constructor(private readonly client: ZomeCallable) {}

  async initiateTransfer(input: InitiateTransferInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'initiate_transfer',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async getTransfer(transferId: string): Promise<HolochainRecord<Transfer> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<Transfer> | null>;
  }

  async getSellerTransfers(did: string): Promise<HolochainRecord<Transfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_seller_transfers',
      payload: did,
    }) as Promise<HolochainRecord<Transfer>[]>;
  }

  async getBuyerTransfers(did: string): Promise<HolochainRecord<Transfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_buyer_transfers',
      payload: did,
    }) as Promise<HolochainRecord<Transfer>[]>;
  }

  async getPropertyTransfers(propertyId: string): Promise<HolochainRecord<Transfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_property_transfers',
      payload: propertyId,
    }) as Promise<HolochainRecord<Transfer>[]>;
  }

  async getTransfersByStatus(status: TransferStatus): Promise<HolochainRecord<Transfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_transfers_by_status',
      payload: status,
    }) as Promise<HolochainRecord<Transfer>[]>;
  }

  async acceptTransfer(input: AcceptTransferInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'accept_transfer',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async cancelTransfer(input: CancelTransferInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'cancel_transfer',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async completeTransfer(transferId: string): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'complete_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async disputeTransfer(input: DisputeTransferInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'dispute_transfer',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async createEscrow(input: CreateEscrowInput): Promise<HolochainRecord<Escrow>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'create_escrow',
      payload: input,
    }) as Promise<HolochainRecord<Escrow>>;
  }

  async getTransferEscrow(transferId: string): Promise<HolochainRecord<Escrow> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_transfer_escrow',
      payload: transferId,
    }) as Promise<HolochainRecord<Escrow> | null>;
  }

  async fundEscrow(escrowId: string): Promise<HolochainRecord<Escrow>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'fund_escrow',
      payload: escrowId,
    }) as Promise<HolochainRecord<Escrow>>;
  }

  async releaseEscrow(escrowId: string): Promise<HolochainRecord<Escrow>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'release_escrow',
      payload: escrowId,
    }) as Promise<HolochainRecord<Escrow>>;
  }

  async addCondition(input: AddConditionInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'add_condition',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }

  async satisfyCondition(input: SatisfyConditionInput): Promise<HolochainRecord<Transfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'satisfy_condition',
      payload: input,
    }) as Promise<HolochainRecord<Transfer>>;
  }
}

// ============================================================================
// Disputes Client
// ============================================================================

const DISPUTES_ZOME = 'property_disputes';

/**
 * Disputes Client - Property dispute management
 *
 * Covers 14 zome functions from disputes coordinator
 */
export class DisputesClient {
  constructor(private readonly client: ZomeCallable) {}

  async fileDispute(input: FileDisputeInput): Promise<HolochainRecord<PropertyDispute>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'file_dispute',
      payload: input,
    }) as Promise<HolochainRecord<PropertyDispute>>;
  }

  async getDispute(disputeId: string): Promise<HolochainRecord<PropertyDispute> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_dispute',
      payload: disputeId,
    }) as Promise<HolochainRecord<PropertyDispute> | null>;
  }

  async getPropertyDisputes(propertyId: string): Promise<HolochainRecord<PropertyDispute>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_property_disputes',
      payload: propertyId,
    }) as Promise<HolochainRecord<PropertyDispute>[]>;
  }

  async getClaimantDisputes(claimantDid: string): Promise<HolochainRecord<PropertyDispute>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_claimant_disputes',
      payload: claimantDid,
    }) as Promise<HolochainRecord<PropertyDispute>[]>;
  }

  async getRespondentDisputes(respondentDid: string): Promise<HolochainRecord<PropertyDispute>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_respondent_disputes',
      payload: respondentDid,
    }) as Promise<HolochainRecord<PropertyDispute>[]>;
  }

  async getDisputesByStatus(status: DisputeStatus): Promise<HolochainRecord<PropertyDispute>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_disputes_by_status',
      payload: status,
    }) as Promise<HolochainRecord<PropertyDispute>[]>;
  }

  async updateDisputeStatus(input: UpdateDisputeStatusInput): Promise<HolochainRecord<PropertyDispute>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'update_dispute_status',
      payload: input,
    }) as Promise<HolochainRecord<PropertyDispute>>;
  }

  async escalateToJustice(input: EscalateInput): Promise<HolochainRecord<PropertyDispute>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'escalate_to_justice',
      payload: input,
    }) as Promise<HolochainRecord<PropertyDispute>>;
  }

  async resolveDispute(input: ResolveDisputeInput): Promise<HolochainRecord<PropertyDispute>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'resolve_dispute',
      payload: input,
    }) as Promise<HolochainRecord<PropertyDispute>>;
  }

  async addDisputeEvidence(input: AddEvidenceInput): Promise<HolochainRecord<PropertyDispute>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'add_dispute_evidence',
      payload: input,
    }) as Promise<HolochainRecord<PropertyDispute>>;
  }

  async fileOwnershipClaim(input: FileClaimInput): Promise<HolochainRecord<OwnershipClaim>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'file_ownership_claim',
      payload: input,
    }) as Promise<HolochainRecord<OwnershipClaim>>;
  }

  async getOwnershipClaim(claimId: string): Promise<HolochainRecord<OwnershipClaim> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_ownership_claim',
      payload: claimId,
    }) as Promise<HolochainRecord<OwnershipClaim> | null>;
  }

  async getPropertyClaims(propertyId: string): Promise<HolochainRecord<OwnershipClaim>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'get_property_claims',
      payload: propertyId,
    }) as Promise<HolochainRecord<OwnershipClaim>[]>;
  }

  async updateClaimStatus(input: UpdateClaimStatusInput): Promise<HolochainRecord<OwnershipClaim>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'update_claim_status',
      payload: input,
    }) as Promise<HolochainRecord<OwnershipClaim>>;
  }

  async addClaimDocument(input: AddDocumentInput): Promise<HolochainRecord<OwnershipClaim>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: DISPUTES_ZOME,
      fn_name: 'add_claim_document',
      payload: input,
    }) as Promise<HolochainRecord<OwnershipClaim>>;
  }
}

// ============================================================================
// Commons Client
// ============================================================================

const COMMONS_ZOME = 'property_commons';

/**
 * Commons Client - Common resource management
 *
 * Covers 16 zome functions from commons coordinator
 */
export class CommonsClient {
  constructor(private readonly client: ZomeCallable) {}

  async createCommonResource(input: CreateResourceInput): Promise<HolochainRecord<CommonResource>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'create_common_resource',
      payload: input,
    }) as Promise<HolochainRecord<CommonResource>>;
  }

  async getResource(resourceId: string): Promise<HolochainRecord<CommonResource> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_resource',
      payload: resourceId,
    }) as Promise<HolochainRecord<CommonResource> | null>;
  }

  async getStewardResources(stewardDid: string): Promise<HolochainRecord<CommonResource>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_steward_resources',
      payload: stewardDid,
    }) as Promise<HolochainRecord<CommonResource>[]>;
  }

  async getResourcesByType(resourceType: ResourceType): Promise<HolochainRecord<CommonResource>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_resources_by_type',
      payload: resourceType,
    }) as Promise<HolochainRecord<CommonResource>[]>;
  }

  async addSteward(input: AddStewardInput): Promise<HolochainRecord<CommonResource>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'add_steward',
      payload: input,
    }) as Promise<HolochainRecord<CommonResource>>;
  }

  async removeSteward(input: RemoveStewardInput): Promise<HolochainRecord<CommonResource>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'remove_steward',
      payload: input,
    }) as Promise<HolochainRecord<CommonResource>>;
  }

  async updateGovernanceRules(input: UpdateGovernanceInput): Promise<HolochainRecord<CommonResource>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'update_governance_rules',
      payload: input,
    }) as Promise<HolochainRecord<CommonResource>>;
  }

  async grantUsageRight(input: GrantRightInput): Promise<HolochainRecord<UsageRight>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'grant_usage_right',
      payload: input,
    }) as Promise<HolochainRecord<UsageRight>>;
  }

  async getUsageRight(rightId: string): Promise<HolochainRecord<UsageRight> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_usage_right',
      payload: rightId,
    }) as Promise<HolochainRecord<UsageRight> | null>;
  }

  async getResourceRights(resourceId: string): Promise<HolochainRecord<UsageRight>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_resource_rights',
      payload: resourceId,
    }) as Promise<HolochainRecord<UsageRight>[]>;
  }

  async getHolderRights(holderDid: string): Promise<HolochainRecord<UsageRight>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_holder_rights',
      payload: holderDid,
    }) as Promise<HolochainRecord<UsageRight>[]>;
  }

  async revokeUsageRight(input: RevokeRightInput): Promise<HolochainRecord<UsageRight>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'revoke_usage_right',
      payload: input,
    }) as Promise<HolochainRecord<UsageRight>>;
  }

  async updateRightQuota(input: UpdateQuotaInput): Promise<HolochainRecord<UsageRight>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'update_right_quota',
      payload: input,
    }) as Promise<HolochainRecord<UsageRight>>;
  }

  async logUsage(input: LogUsageInput): Promise<HolochainRecord<UsageLog>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'log_usage',
      payload: input,
    }) as Promise<HolochainRecord<UsageLog>>;
  }

  async getResourceUsage(resourceId: string): Promise<HolochainRecord<UsageLog>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_resource_usage',
      payload: resourceId,
    }) as Promise<HolochainRecord<UsageLog>[]>;
  }

  async getUserUsage(input: UserUsageInput): Promise<number> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_user_usage',
      payload: input,
    }) as Promise<number>;
  }

  async checkUsageQuota(input: CheckQuotaInput): Promise<boolean> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'check_usage_quota',
      payload: input,
    }) as Promise<boolean>;
  }
}

// ============================================================================
// Unified Property Client
// ============================================================================

/**
 * Unified Property Client for mycelix-property hApp
 *
 * Provides access to all property functionality through a single interface:
 * - registry: Property registration and management (17 functions)
 * - transfers: Title transfer workflow (16 functions)
 * - disputes: Property dispute resolution (15 functions)
 * - commons: Common resource management (17 functions)
 *
 * Total: 65 functions covered
 *
 * @example
 * ```typescript
 * import { PropertyClient } from '@mycelix/sdk/clients/property';
 *
 * const property = new PropertyClient(appClient, 'mycelix-property');
 *
 * // Register a property
 * const result = await property.registry.registerProperty({
 *   property_type: 'Land',
 *   title: 'My Land',
 *   description: 'A parcel of land',
 *   owner_did: 'did:mycelix:abc123',
 *   co_owners: [],
 *   metadata: { parcel_id: 'P001' },
 * });
 *
 * // Initiate transfer
 * const transfer = await property.transfers.initiateTransfer({
 *   property_id: result.entry.Present.id,
 *   from_did: 'did:mycelix:abc123',
 *   to_did: 'did:mycelix:xyz789',
 *   transfer_type: 'Sale',
 *   price: 100000,
 *   currency: 'USD',
 *   conditions: [],
 * });
 * ```
 */
export class PropertyClient {
  /** Property registration and management */
  readonly registry: RegistryClient;

  /** Title transfer workflow */
  readonly transfers: TransfersClient;

  /** Property dispute resolution */
  readonly disputes: DisputesClient;

  /** Common resource management */
  readonly commons: CommonsClient;

  constructor(
    client: AppClient | ZomeCallable,
    _appId: string = 'property'
  ) {
    const callable = client as ZomeCallable;
    this.registry = new RegistryClient(callable);
    this.transfers = new TransfersClient(callable);
    this.disputes = new DisputesClient(callable);
    this.commons = new CommonsClient(callable);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Property hApp clients
 */
export function createPropertyClients(client: ZomeCallable) {
  return {
    registry: new RegistryClient(client),
    transfers: new TransfersClient(client),
    disputes: new DisputesClient(client),
    commons: new CommonsClient(client),
  };
}

export default PropertyClient;
