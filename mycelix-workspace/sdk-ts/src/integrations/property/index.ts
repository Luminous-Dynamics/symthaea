// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Property Integration
 *
 * hApp-specific adapter for Mycelix-Property providing:
 * - Decentralized asset registry with ownership tracking
 * - Fractional ownership and cooperative stewardship
 * - Transfer management with reputation verification
 * - Cross-hApp property coordination via Bridge
 * - Commons management for shared resources
 * - Lien and encumbrance tracking
 * - Collateral pledging for finance integration
 *
 * @packageDocumentation
 * @module integrations/property
 */

// ============================================================================
// New Holochain Zome Clients (Recommended)
// ============================================================================

export { MycelixPropertyClient } from './client';
export type {
  PropertyClientConfig,
  PropertyConnectionOptions,
} from './client';

// Zome clients
export {
  RegistryClient,
  TransferClient,
  LienClient,
  CommonsClient,
} from './zomes';
export type {
  RegistryClientConfig,
  TransferClientConfig,
  LienClientConfig,
  CommonsClientConfig,
} from './zomes';

// Types
export type {
  // Asset types
  AssetStatus,
  RegisterAssetInput,
  GeoLocation,
  // Ownership types
  Ownership,
  OwnershipType,
  OwnershipRole,
  // Valuation types
  Valuation,
  ValuationMethod,
  RecordValuationInput,
  // Transfer types
  Transfer,
  InitiateTransferInput,
  // Lien types
  Lien,
  LienType,
  LienStatus,
  PlaceLienInput,
  // Commons types
  Commons,
  CommonsType,
  CommonsAccessLevel,
  CommonsMember,
  CommonsUsage,
  CreateCommonsInput,
  // Fractional types
  FractionalShare,
  ShareClass,
  Shareholder,
  FractionalizeInput,
  // Cross-hApp types
  OwnershipVerification,
  PropertyEvent,
  PropertyEventType,
  // Error types
  PropertySdkError,
  PropertySdkErrorCode,
} from './types';

// ============================================================================
// Legacy Mock Services (for testing/development)
// ============================================================================


import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  createReputation,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Bridge Zome Types (matching Rust property_bridge zome)
// ============================================================================

/** Ownership type for bridge queries */
export type BridgeOwnershipType = 'Sole' | 'Joint' | 'Fractional' | 'Corporate' | 'Trust' | 'Commons';

/** Encumbrance type */
export type EncumbranceType = 'Mortgage' | 'Lien' | 'Easement' | 'Lease' | 'CollateralPledge';

/** Ownership query */
export interface OwnershipQuery {
  id: string;
  asset_id: string;
  source_happ: string;
  verification_type: 'Current' | 'Historical' | 'Encumbrances';
  queried_at: number;
}

/** Ownership result */
export interface OwnershipResult {
  id: string;
  query_id: string;
  asset_id: string;
  owners: BridgeOwnershipShare[];
  ownership_type: BridgeOwnershipType;
  encumbrances: Encumbrance[];
  verified: boolean;
  as_of: number;
}

/** Bridge ownership share */
export interface BridgeOwnershipShare {
  did: string;
  percentage: number;
  role: 'Owner' | 'Steward' | 'Beneficiary' | 'Trustee';
  since: number;
}

/** Encumbrance record */
export interface Encumbrance {
  encumbrance_type: EncumbranceType;
  holder_did: string;
  amount?: number;
  expires_at?: number;
}

/** Collateral pledge for finance */
export interface CollateralPledge {
  id: string;
  asset_id: string;
  owner_did: string;
  pledge_to_happ: string;
  pledge_amount: number;
  loan_reference?: string;
  status: 'Active' | 'Released' | 'Foreclosed';
  pledged_at: number;
  released_at?: number;
}

/** Property bridge event types */
export type PropertyBridgeEventType =
  | 'OwnershipTransferred'
  | 'CollateralPledged'
  | 'CollateralReleased'
  | 'EncumbranceAdded'
  | 'EncumbranceRemoved'
  | 'PropertyRegistered';

/** Property bridge event */
export interface PropertyBridgeEvent {
  id: string;
  event_type: PropertyBridgeEventType;
  asset_id?: string;
  owner_did?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Verify ownership input */
export interface VerifyOwnershipInput {
  asset_id: string;
  verification_type: OwnershipQuery['verification_type'];
}

/** Pledge collateral input */
export interface PledgeCollateralInput {
  asset_id: string;
  pledge_to_happ: string;
  pledge_amount: number;
  loan_reference?: string;
}

// ============================================================================
// Property-Specific Types
// ============================================================================

/** Asset type categories (legacy mock — see types.ts for Holochain-aligned AssetType) */
export type AssetType = 'real_estate' | 'equipment' | 'digital' | 'vehicle' | 'commons' | 'intellectual';

/** Ownership model */
export type OwnershipModel = 'sole' | 'joint' | 'fractional' | 'cooperative' | 'commons';

/** Transfer status */
export type TransferStatus = 'proposed' | 'pending_approval' | 'approved' | 'completed' | 'rejected';

/** Asset record */
export interface Asset {
  id: string;
  type: AssetType;
  name: string;
  description: string;
  ownershipModel: OwnershipModel;
  owners: OwnershipShare[];
  valuation?: number;
  location?: string;
  metadata: Record<string, unknown>;
  registeredAt: number;
  updatedAt: number;
}

/** Ownership share */
export interface OwnershipShare {
  ownerId: string;
  percentage: number;
  acquiredAt: number;
  role?: 'steward' | 'beneficiary' | 'trustee';
}

/** Transfer proposal */
export interface TransferProposal {
  id: string;
  assetId: string;
  fromOwnerId: string;
  toOwnerId: string;
  sharePercentage: number;
  consideration?: number;
  status: TransferStatus;
  approvals: string[];
  requiredApprovals: number;
  createdAt: number;
  completedAt?: number;
}

/** Commons membership */
export interface CommonsMembership {
  commonsId: string;
  memberId: string;
  accessRights: string[];
  contributionCredits: number;
  reputation: ReputationScore;
  joinedAt: number;
}

/** Usage record for commons */
export interface UsageRecord {
  id: string;
  assetId: string;
  userId: string;
  startTime: number;
  endTime?: number;
  purpose: string;
  approved: boolean;
}

// ============================================================================
// Property Service
// ============================================================================

/**
 * Property service for asset registry and transfer management
 */
export class PropertyService {
  private assets = new Map<string, Asset>();
  private transfers = new Map<string, TransferProposal>();
  private memberships = new Map<string, CommonsMembership[]>();
  private usageRecords: UsageRecord[] = [];
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('property');
  }

  /**
   * Register a new asset
   */
  registerAsset(
    type: AssetType,
    name: string,
    description: string,
    primaryOwnerId: string,
    ownershipModel: OwnershipModel = 'sole',
    metadata: Record<string, unknown> = {}
  ): Asset {
    const id = `asset-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const asset: Asset = {
      id,
      type,
      name,
      description,
      ownershipModel,
      owners: [
        {
          ownerId: primaryOwnerId,
          percentage: 100,
          acquiredAt: Date.now(),
          role: ownershipModel === 'cooperative' ? 'steward' : undefined,
        },
      ],
      metadata,
      registeredAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.assets.set(id, asset);
    return asset;
  }

  /**
   * Propose a transfer of ownership
   */
  proposeTransfer(
    assetId: string,
    fromOwnerId: string,
    toOwnerId: string,
    sharePercentage: number,
    consideration?: number
  ): TransferProposal {
    // Validate transfer share percentage must be positive (> 0)
    if (sharePercentage <= 0) {
      throw new Error('Transfer share percentage must be greater than 0');
    }
    if (sharePercentage > 100) {
      throw new Error('Transfer share percentage cannot exceed 100%');
    }

    const asset = this.assets.get(assetId);
    if (!asset) throw new Error('Asset not found');

    const ownerShare = asset.owners.find((o) => o.ownerId === fromOwnerId);
    if (!ownerShare || ownerShare.percentage < sharePercentage) {
      throw new Error('Insufficient ownership share');
    }

    const id = `transfer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // Determine required approvals based on ownership model
    let requiredApprovals = 1;
    if (asset.ownershipModel === 'joint' || asset.ownershipModel === 'cooperative') {
      requiredApprovals = Math.ceil(asset.owners.length * 0.5);
    }

    const proposal: TransferProposal = {
      id,
      assetId,
      fromOwnerId,
      toOwnerId,
      sharePercentage,
      consideration,
      status: requiredApprovals > 1 ? 'pending_approval' : 'approved',
      approvals: [fromOwnerId],
      requiredApprovals,
      createdAt: Date.now(),
    };

    this.transfers.set(id, proposal);
    return proposal;
  }

  /**
   * Approve a transfer proposal
   */
  approveTransfer(transferId: string, approverId: string): TransferProposal {
    const transfer = this.transfers.get(transferId);
    if (!transfer) throw new Error('Transfer not found');

    const asset = this.assets.get(transfer.assetId);
    if (!asset) throw new Error('Asset not found');

    // Verify approver is an owner
    if (!asset.owners.some((o) => o.ownerId === approverId)) {
      throw new Error('Approver is not an owner');
    }

    if (!transfer.approvals.includes(approverId)) {
      transfer.approvals.push(approverId);
    }

    if (transfer.approvals.length >= transfer.requiredApprovals) {
      transfer.status = 'approved';
    }

    return transfer;
  }

  /**
   * Execute an approved transfer
   */
  executeTransfer(transferId: string): Asset {
    const transfer = this.transfers.get(transferId);
    if (!transfer) throw new Error('Transfer not found');
    if (transfer.status !== 'approved') throw new Error('Transfer not approved');

    const asset = this.assets.get(transfer.assetId)!;

    // Update from owner
    const fromOwner = asset.owners.find((o) => o.ownerId === transfer.fromOwnerId)!;
    fromOwner.percentage -= transfer.sharePercentage;

    // Remove if zero
    if (fromOwner.percentage <= 0) {
      asset.owners = asset.owners.filter((o) => o.ownerId !== transfer.fromOwnerId);
    }

    // Add to new owner
    const existingTo = asset.owners.find((o) => o.ownerId === transfer.toOwnerId);
    if (existingTo) {
      existingTo.percentage += transfer.sharePercentage;
    } else {
      asset.owners.push({
        ownerId: transfer.toOwnerId,
        percentage: transfer.sharePercentage,
        acquiredAt: Date.now(),
      });
    }

    asset.updatedAt = Date.now();
    transfer.status = 'completed';
    transfer.completedAt = Date.now();

    return asset;
  }

  /**
   * Join a commons
   */
  joinCommons(commonsId: string, memberId: string, accessRights: string[] = ['read']): CommonsMembership {
    const membership: CommonsMembership = {
      commonsId,
      memberId,
      accessRights,
      contributionCredits: 0,
      reputation: createReputation(memberId),
      joinedAt: Date.now(),
    };

    const existing = this.memberships.get(commonsId) || [];
    existing.push(membership);
    this.memberships.set(commonsId, existing);

    return membership;
  }

  /**
   * Record usage of a commons asset
   */
  recordUsage(assetId: string, userId: string, purpose: string): UsageRecord {
    const record: UsageRecord = {
      id: `usage-${Date.now()}`,
      assetId,
      userId,
      startTime: Date.now(),
      purpose,
      approved: true,
    };
    this.usageRecords.push(record);
    return record;
  }

  /**
   * Get asset by ID
   */
  getAsset(assetId: string): Asset | undefined {
    return this.assets.get(assetId);
  }

  /**
   * Get assets owned by a DID
   */
  getOwnedAssets(ownerId: string): Asset[] {
    return Array.from(this.assets.values()).filter((asset) =>
      asset.owners.some((o) => o.ownerId === ownerId)
    );
  }

  /**
   * Verify ownership for cross-hApp coordination
   */
  async verifyOwnership(assetId: string, claimantId: string): Promise<{ verified: boolean; share: number }> {
    const asset = this.assets.get(assetId);
    if (!asset) return { verified: false, share: 0 };

    const ownership = asset.owners.find((o) => o.ownerId === claimantId);
    return {
      verified: !!ownership,
      share: ownership?.percentage || 0,
    };
  }
}

// Singleton
let instance: PropertyService | null = null;

export function getPropertyService(): PropertyService {
  if (!instance) instance = new PropertyService();
  return instance;
}

export function resetPropertyService(): void {
  instance = null;
}

// ============================================================================
// Property Bridge Client (Holochain Zome Calls)
// ============================================================================

const PROPERTY_ROLE = 'commons';
const BRIDGE_ZOME = 'commons_bridge';

/**
 * Property Bridge Client - Direct Holochain zome calls for cross-hApp property
 */
export class PropertyBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Verify ownership of an asset
   */
  async verifyOwnership(input: VerifyOwnershipInput): Promise<OwnershipResult> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_ownership',
      payload: input,
    });
  }

  /**
   * Pledge collateral for a loan
   */
  async pledgeCollateral(input: PledgeCollateralInput): Promise<CollateralPledge> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'pledge_collateral',
      payload: input,
    });
  }

  /**
   * Release collateral after loan repayment
   */
  async releaseCollateral(pledgeId: string): Promise<CollateralPledge> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'release_collateral',
      payload: pledgeId,
    });
  }

  /**
   * Get properties by owner
   */
  async getPropertiesByOwner(ownerDid: string): Promise<OwnershipResult[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_properties_by_owner',
      payload: ownerDid,
    });
  }

  /**
   * Get active collateral pledges for a hApp
   */
  async getActiveCollateral(happId: string): Promise<CollateralPledge[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_active_collateral',
      payload: happId,
    });
  }

  /**
   * Broadcast a property event
   */
  async broadcastPropertyEvent(eventType: PropertyBridgeEventType, assetId?: string, ownerDid?: string, payload: string = '{}'): Promise<PropertyBridgeEvent> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_property_event',
      payload: { event_type: eventType, asset_id: assetId, owner_did: ownerDid, payload },
    });
  }

  /**
   * Get recent property events
   */
  async getRecentEvents(limit?: number): Promise<PropertyBridgeEvent[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }
}

// Bridge client singleton
let bridgeInstance: PropertyBridgeClient | null = null;

export function getPropertyBridgeClient(client: MycelixClient): PropertyBridgeClient {
  if (!bridgeInstance) bridgeInstance = new PropertyBridgeClient(client);
  return bridgeInstance;
}

export function resetPropertyBridgeClient(): void {
  bridgeInstance = null;
}
