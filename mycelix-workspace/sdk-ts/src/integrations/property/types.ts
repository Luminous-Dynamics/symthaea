// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Property SDK Types
 *
 * Type definitions for the Property hApp SDK, covering asset registry,
 * ownership transfers, liens, commons, and fractional ownership.
 *
 * @module @mycelix/sdk/integrations/property/types
 */

// ============================================================================
// Asset Types
// ============================================================================

/** Asset type categories (matches Rust PropertyType in property-registry integrity) */
export type AssetType =
  | 'Land'
  | 'Building'
  | 'Unit'
  | 'Equipment'
  | 'Intellectual'
  | 'Digital'
  | 'Other';

/** Ownership type */
export type OwnershipType =
  | 'Sole'
  | 'Joint'
  | 'Fractional'
  | 'Trust'
  | 'Corporate'
  | 'Commons';

/** Asset status */
export type AssetStatus =
  | 'Active'
  | 'Encumbered'
  | 'Disputed'
  | 'Archived'
  | 'PendingTransfer';

/** A registered asset */
export interface Asset {
  /** Unique asset identifier */
  id: string;
  /** Asset type */
  asset_type: AssetType;
  /** Asset name */
  name: string;
  /** Description */
  description: string;
  /** Geospatial location (optional) */
  location?: GeoLocation;
  /** Current status */
  status: AssetStatus;
  /** Metadata (JSON string) */
  metadata?: string;
  /** Creation timestamp */
  created_at: number;
  /** Last update timestamp */
  updated_at: number;
}

/** Input for registering an asset */
export interface RegisterAssetInput {
  /** Unique asset identifier */
  id: string;
  /** Asset type */
  asset_type: AssetType;
  /** Asset name */
  name: string;
  /** Description */
  description: string;
  /** Geospatial location (optional) */
  location?: GeoLocation;
  /** Initial valuation (optional) */
  valuation?: number;
  /** Valuation currency */
  valuation_currency?: string;
  /** Initial owner DID */
  owner_did: string;
  /** Metadata (JSON string) */
  metadata?: string;
}

/** Geographic location (matches Rust GeoLocation in property-registry integrity) */
export interface GeoLocation {
  /** Latitude */
  latitude: number;
  /** Longitude */
  longitude: number;
  /** Polygon boundaries [[lat, lon], ...] */
  boundaries?: number[][];
  /** Area in square meters */
  area_sqm?: number;
}

/** Street address (matches Rust Address in property-registry integrity) */
export interface Address {
  /** Street address */
  street: string;
  /** City */
  city: string;
  /** Region/State */
  region: string;
  /** Country */
  country: string;
  /** Postal code */
  postal_code: string;
}

// ============================================================================
// Ownership Types
// ============================================================================

/** Ownership record */
export interface Ownership {
  /** Unique ownership record ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Owner's DID */
  owner_did: string;
  /** Ownership percentage (0-100) */
  percentage: number;
  /** Ownership type */
  ownership_type: OwnershipType;
  /** Acquisition timestamp */
  acquired_at: number;
  /** Transfer reference (if acquired via transfer) */
  transfer_id?: string;
}

/** Ownership share summary */
export interface OwnershipShare {
  /** Owner's DID */
  owner_did: string;
  /** Percentage owned */
  percentage: number;
  /** Role in ownership */
  role?: OwnershipRole;
  /** Acquisition timestamp */
  acquired_at: number;
}

/** Role in ownership arrangement */
export type OwnershipRole =
  | 'Owner'
  | 'Steward'
  | 'Beneficiary'
  | 'Trustee'
  | 'Manager';

// ============================================================================
// Valuation Types
// ============================================================================

/** Valuation method */
export type ValuationMethod =
  | 'Market'
  | 'Income'
  | 'Cost'
  | 'SelfDeclared'
  | 'Appraisal';

/** Asset valuation record */
export interface Valuation {
  /** Unique valuation ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Valuation amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Valuation method */
  method: ValuationMethod;
  /** Appraiser DID (if applicable) */
  appraiser_did?: string;
  /** Valuation date */
  valued_at: number;
  /** Expiry date (optional) */
  expires_at?: number;
  /** Notes */
  notes?: string;
}

/** Input for recording valuation */
export interface RecordValuationInput {
  /** Asset ID */
  asset_id: string;
  /** Valuation amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Valuation method */
  method: ValuationMethod;
  /** Appraiser DID (if applicable) */
  appraiser_did?: string;
  /** Notes */
  notes?: string;
}

// ============================================================================
// Transfer Types
// ============================================================================

/** Transfer status */
export type TransferStatus =
  | 'Proposed'
  | 'Pending'
  | 'Accepted'
  | 'Completed'
  | 'Cancelled'
  | 'Disputed';

/** Ownership transfer record */
export interface Transfer {
  /** Unique transfer ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** From owner DID */
  from_owner: string;
  /** To owner DID */
  to_owner: string;
  /** Percentage being transferred */
  percentage: number;
  /** Consideration amount (optional) */
  consideration?: number;
  /** Currency for consideration */
  currency?: string;
  /** Transfer status */
  status: TransferStatus;
  /** Escrow ID (if using escrow) */
  escrow_id?: string;
  /** Required approvals (for joint ownership) */
  required_approvals: number;
  /** Current approvals */
  approvals: string[];
  /** Initiated timestamp */
  initiated_at: number;
  /** Completed timestamp */
  completed_at?: number;
}

/** Input for initiating transfer */
export interface InitiateTransferInput {
  /** Asset ID */
  asset_id: string;
  /** To owner DID */
  to_owner: string;
  /** Percentage to transfer */
  percentage: number;
  /** Consideration amount (optional) */
  consideration?: number;
  /** Currency for consideration */
  currency?: string;
  /** Use escrow for payment */
  use_escrow?: boolean;
}

// ============================================================================
// Lien Types
// ============================================================================

/** Lien type */
export type LienType =
  | 'Mortgage'
  | 'Tax'
  | 'Mechanic'
  | 'Judgment'
  | 'SecurityInterest'
  | 'Other';

/** Lien status */
export type LienStatus = 'Active' | 'Satisfied' | 'Released' | 'Foreclosed';

/** A lien on an asset */
export interface Lien {
  /** Unique lien ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Lien holder DID */
  holder_did: string;
  /** Lien type */
  lien_type: LienType;
  /** Amount secured */
  amount: number;
  /** Currency */
  currency: string;
  /** Priority (lower = higher priority) */
  priority: number;
  /** Status */
  status: LienStatus;
  /** Related loan ID (optional) */
  loan_id?: string;
  /** Filed timestamp */
  filed_at: number;
  /** Satisfied timestamp */
  satisfied_at?: number;
  /** Released timestamp */
  released_at?: number;
  /** Description */
  description?: string;
}

/** Input for placing a lien */
export interface PlaceLienInput {
  /** Asset ID */
  asset_id: string;
  /** Lien type */
  lien_type: LienType;
  /** Amount secured */
  amount: number;
  /** Currency */
  currency: string;
  /** Related loan ID (optional) */
  loan_id?: string;
  /** Description */
  description?: string;
}

// ============================================================================
// Commons Types
// ============================================================================

/** Commons resource type */
export type CommonsType =
  | 'Land'
  | 'Water'
  | 'Forest'
  | 'Infrastructure'
  | 'Digital'
  | 'Knowledge'
  | 'Other';

/** Commons access level */
export type CommonsAccessLevel = 'Read' | 'Use' | 'Contribute' | 'Manage' | 'Admin';

/** A commons resource */
export interface Commons {
  /** Unique commons ID */
  id: string;
  /** Commons name */
  name: string;
  /** Description */
  description: string;
  /** Commons type */
  commons_type: CommonsType;
  /** Managing DAO ID */
  dao_id?: string;
  /** Governance rules (JSON) */
  governance_rules: string;
  /** Geographic boundary (for physical commons) */
  boundary?: GeoLocation[];
  /** Members */
  members: CommonsMember[];
  /** Associated assets */
  asset_ids: string[];
  /** Creation timestamp */
  created_at: number;
  /** Creator DID */
  created_by: string;
}

/** Input for creating commons */
export interface CreateCommonsInput {
  /** Unique commons ID */
  id: string;
  /** Commons name */
  name: string;
  /** Description */
  description: string;
  /** Commons type */
  commons_type: CommonsType;
  /** Managing DAO ID (optional) */
  dao_id?: string;
  /** Governance rules (JSON) */
  governance_rules: string;
  /** Geographic boundary (optional) */
  boundary?: GeoLocation[];
  /** Creator DID */
  created_by: string;
}

/** Commons member */
export interface CommonsMember {
  /** Member DID */
  did: string;
  /** Access level */
  access_level: CommonsAccessLevel;
  /** Contribution credits */
  contribution_credits: number;
  /** Joined timestamp */
  joined_at: number;
}

/** Usage record for commons */
export interface CommonsUsage {
  /** Unique usage ID */
  id: string;
  /** Commons ID */
  commons_id: string;
  /** User DID */
  user_did: string;
  /** Start time */
  start_time: number;
  /** End time (optional) */
  end_time?: number;
  /** Purpose */
  purpose: string;
  /** Was usage approved */
  approved: boolean;
  /** Approver DID (if required) */
  approved_by?: string;
}

// ============================================================================
// Fractional Ownership Types
// ============================================================================

/** Share class for fractional ownership */
export type ShareClass = 'Common' | 'Preferred' | 'Voting' | 'NonVoting';

/** Fractional share structure */
export interface FractionalShare {
  /** Unique share ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Total shares issued */
  total_shares: number;
  /** Share class */
  share_class: ShareClass;
  /** Shareholders */
  shareholders: Shareholder[];
  /** Created timestamp */
  created_at: number;
}

/** Shareholder record */
export interface Shareholder {
  /** Shareholder DID */
  did: string;
  /** Number of shares */
  shares: number;
  /** Acquisition timestamp */
  acquired_at: number;
}

/** Input for fractionalizing an asset */
export interface FractionalizeInput {
  /** Asset ID */
  asset_id: string;
  /** Total shares to issue */
  total_shares: number;
  /** Share class */
  share_class: ShareClass;
}

// ============================================================================
// Cross-hApp Types
// ============================================================================

/** Collateral pledge for finance integration */
export interface CollateralPledge {
  /** Unique pledge ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Owner DID */
  owner_did: string;
  /** Pledged to hApp */
  pledge_to_happ: string;
  /** Pledge amount */
  pledge_amount: number;
  /** Loan reference (optional) */
  loan_reference?: string;
  /** Status */
  status: 'Active' | 'Released' | 'Foreclosed';
  /** Pledged timestamp */
  pledged_at: number;
  /** Released timestamp */
  released_at?: number;
}

/** Input for pledging collateral */
export interface PledgeCollateralInput {
  /** Asset ID */
  asset_id: string;
  /** Pledge to hApp */
  pledge_to_happ: string;
  /** Pledge amount */
  pledge_amount: number;
  /** Loan reference (optional) */
  loan_reference?: string;
}

/** Ownership verification result */
export interface OwnershipVerification {
  /** Asset ID */
  asset_id: string;
  /** Owner DID */
  owner_did: string;
  /** Verified ownership */
  verified: boolean;
  /** Percentage owned */
  percentage: number;
  /** Active liens count */
  active_liens: number;
  /** Total lien amount */
  total_lien_amount: number;
  /** Verification timestamp */
  verified_at: number;
}

/** Property event types */
export type PropertyEventType =
  | 'AssetRegistered'
  | 'OwnershipTransferred'
  | 'LienPlaced'
  | 'LienReleased'
  | 'CollateralPledged'
  | 'CollateralReleased'
  | 'CommonsCreated'
  | 'MemberJoined';

/** Property event */
export interface PropertyEvent {
  /** Event ID */
  id: string;
  /** Event type */
  event_type: PropertyEventType;
  /** Asset ID (if applicable) */
  asset_id?: string;
  /** DID involved */
  did?: string;
  /** Event payload (JSON) */
  payload: string;
  /** Source hApp */
  source_happ: string;
  /** Timestamp */
  timestamp: number;
}

// ============================================================================
// Error Types
// ============================================================================

/** Property SDK error codes */
export type PropertySdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'ASSET_NOT_FOUND'
  | 'OWNERSHIP_NOT_FOUND'
  | 'INSUFFICIENT_OWNERSHIP'
  | 'TRANSFER_NOT_APPROVED'
  | 'LIEN_EXISTS'
  | 'COMMONS_NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'SERIALIZATION_ERROR'
  | 'UNKNOWN_ERROR';

/**
 * Property SDK Error
 */
export class PropertySdkError extends Error {
  constructor(
    public readonly code: PropertySdkErrorCode,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'PropertySdkError';
  }
}
