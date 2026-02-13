/**
 * Mycelix Property Module
 *
 * TypeScript client for the mycelix-property hApp.
 * Provides asset registry, title transfer, and commons management.
 *
 * @module @mycelix/sdk/property
 */

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
// Asset Types
// ============================================================================

/** Asset type */
export type AssetType =
  | 'RealEstate'
  | 'Vehicle'
  | 'Equipment'
  | 'Intellectual'
  | 'Digital'
  | 'Other';

/** Ownership type */
export type OwnershipType = 'Sole' | 'Joint' | 'Fractional' | 'Commons';

/** Asset status */
export type AssetStatus = 'Active' | 'Encumbered' | 'Disputed' | 'Archived';

/** An asset */
export interface Asset {
  /** Asset ID */
  id: string;
  /** Asset type */
  type_: AssetType;
  /** Name */
  name: string;
  /** Description */
  description: string;
  /** Ownership type */
  ownership_type: OwnershipType;
  /** Current owners with percentages */
  owners: OwnershipShare[];
  /** Geospatial data (if applicable) */
  location?: GeoLocation;
  /** Current valuation */
  valuation?: number;
  /** Currency of valuation */
  valuation_currency?: string;
  /** Status */
  status: AssetStatus;
  /** Metadata (JSON) */
  metadata?: string;
  /** Creation timestamp */
  created_at: number;
  /** Last update timestamp */
  updated_at: number;
}

/** Ownership share */
export interface OwnershipShare {
  /** Owner DID */
  owner: string;
  /** Percentage (0-100) */
  percentage: number;
  /** Acquired timestamp */
  acquired_at: number;
}

/** Geographic location */
export interface GeoLocation {
  /** Latitude */
  lat: number;
  /** Longitude */
  lng: number;
  /** Address (optional) */
  address?: string;
  /** Geohash */
  geohash?: string;
}

/** Input for registering an asset */
export interface RegisterAssetInput {
  type_: AssetType;
  name: string;
  description: string;
  ownership_type: OwnershipType;
  location?: GeoLocation;
  valuation?: number;
  valuation_currency?: string;
  metadata?: string;
}

// ============================================================================
// Transfer Types
// ============================================================================

/** Transfer status */
export type TransferStatus = 'Pending' | 'Completed' | 'Cancelled' | 'Disputed';

/** Title transfer */
export interface TitleTransfer {
  /** Transfer ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** From owner DID */
  from_owner: string;
  /** To owner DID */
  to_owner: string;
  /** Percentage being transferred */
  percentage: number;
  /** Consideration (price) */
  consideration?: number;
  /** Currency */
  currency?: string;
  /** Status */
  status: TransferStatus;
  /** Escrow ID (if using escrow) */
  escrow_id?: string;
  /** Initiated timestamp */
  initiated_at: number;
  /** Completed timestamp */
  completed_at?: number;
}

/** Input for initiating transfer */
export interface InitiateTransferInput {
  asset_id: string;
  to_owner: string;
  percentage: number;
  consideration?: number;
  currency?: string;
  use_escrow?: boolean;
}

// ============================================================================
// Lien Types
// ============================================================================

/** Lien type */
export type LienType = 'Mortgage' | 'Tax' | 'Mechanic' | 'Judgment' | 'Other';

/** Lien status */
export type LienStatus = 'Active' | 'Satisfied' | 'Released';

/** A lien on an asset */
export interface Lien {
  /** Lien ID */
  id: string;
  /** Asset ID */
  asset_id: string;
  /** Lien holder DID */
  holder: string;
  /** Lien type */
  type_: LienType;
  /** Amount */
  amount: number;
  /** Currency */
  currency: string;
  /** Priority (lower = higher priority) */
  priority: number;
  /** Status */
  status: LienStatus;
  /** Filed timestamp */
  filed_at: number;
  /** Satisfied timestamp */
  satisfied_at?: number;
}

/** Input for filing a lien */
export interface FileLienInput {
  asset_id: string;
  type_: LienType;
  amount: number;
  currency: string;
}

// ============================================================================
// Commons Types
// ============================================================================

/** Commons resource type */
export type CommonsType = 'Land' | 'Water' | 'Forest' | 'Infrastructure' | 'Digital' | 'Other';

/** A commons resource */
export interface Commons {
  /** Commons ID */
  id: string;
  /** Name */
  name: string;
  /** Description */
  description: string;
  /** Type */
  type_: CommonsType;
  /** Managing DAO ID */
  managing_dao: string;
  /** Geographic boundary */
  boundary?: GeoLocation[];
  /** Rules (JSON) */
  rules: string;
  /** Members */
  members: CommonsMember[];
  /** Creation timestamp */
  created_at: number;
}

/** Commons member */
export interface CommonsMember {
  /** Member DID */
  did: string;
  /** Access level */
  access_level: 'Read' | 'Use' | 'Manage' | 'Admin';
  /** Joined timestamp */
  joined_at: number;
}

/** Input for creating commons */
export interface CreateCommonsInput {
  name: string;
  description: string;
  type_: CommonsType;
  managing_dao: string;
  boundary?: GeoLocation[];
  rules: string;
}

// ============================================================================
// Registry Client
// ============================================================================

const PROPERTY_ROLE = 'commons';
const REGISTRY_ZOME = 'property_registry';

/**
 * Registry Client - Asset registration and management
 */
export class RegistryClient {
  constructor(private readonly client: ZomeCallable) {}

  async registerAsset(input: RegisterAssetInput): Promise<HolochainRecord<Asset>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'register_asset',
      payload: input,
    }) as Promise<HolochainRecord<Asset>>;
  }

  async getAsset(assetId: string): Promise<HolochainRecord<Asset> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_asset',
      payload: assetId,
    }) as Promise<HolochainRecord<Asset> | null>;
  }

  async getAssetsByOwner(ownerDid: string): Promise<HolochainRecord<Asset>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_assets_by_owner',
      payload: ownerDid,
    }) as Promise<HolochainRecord<Asset>[]>;
  }

  async getAssetsByType(assetType: AssetType): Promise<HolochainRecord<Asset>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_assets_by_type',
      payload: assetType,
    }) as Promise<HolochainRecord<Asset>[]>;
  }

  async updateAsset(
    assetId: string,
    updates: Partial<RegisterAssetInput>
  ): Promise<HolochainRecord<Asset>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'update_asset',
      payload: { asset_id: assetId, ...updates },
    }) as Promise<HolochainRecord<Asset>>;
  }

  async getAssetHistory(assetId: string): Promise<HolochainRecord<Asset>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'get_asset_history',
      payload: assetId,
    }) as Promise<HolochainRecord<Asset>[]>;
  }

  async updateAssetMetadata(
    assetId: string,
    metadata: Record<string, unknown>
  ): Promise<HolochainRecord<Asset>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: REGISTRY_ZOME,
      fn_name: 'update_asset_metadata',
      payload: { asset_id: assetId, metadata },
    }) as Promise<HolochainRecord<Asset>>;
  }
}

// ============================================================================
// Transfer Client
// ============================================================================

const TRANSFER_ZOME = 'property_transfer';

/**
 * Transfer Client - Title transfers
 */
export class TransferClient {
  constructor(private readonly client: ZomeCallable) {}

  async initiateTransfer(input: InitiateTransferInput): Promise<HolochainRecord<TitleTransfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'initiate_transfer',
      payload: input,
    }) as Promise<HolochainRecord<TitleTransfer>>;
  }

  async acceptTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'accept_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<TitleTransfer>>;
  }

  async cancelTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'cancel_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<TitleTransfer>>;
  }

  async completeTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'complete_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<TitleTransfer>>;
  }

  async getTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_transfer',
      payload: transferId,
    }) as Promise<HolochainRecord<TitleTransfer> | null>;
  }

  async getTransfersForAsset(assetId: string): Promise<HolochainRecord<TitleTransfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_transfers_for_asset',
      payload: assetId,
    }) as Promise<HolochainRecord<TitleTransfer>[]>;
  }

  async getPendingTransfers(did: string): Promise<HolochainRecord<TitleTransfer>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: TRANSFER_ZOME,
      fn_name: 'get_pending_transfers',
      payload: did,
    }) as Promise<HolochainRecord<TitleTransfer>[]>;
  }
}

// ============================================================================
// Lien Client
// ============================================================================

const LIENS_ZOME = 'property_liens';

/**
 * Lien Client - Manage liens on assets
 */
export class LienClient {
  constructor(private readonly client: ZomeCallable) {}

  async fileLien(input: FileLienInput): Promise<HolochainRecord<Lien>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: LIENS_ZOME,
      fn_name: 'file_lien',
      payload: input,
    }) as Promise<HolochainRecord<Lien>>;
  }

  async getLien(lienId: string): Promise<HolochainRecord<Lien> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: LIENS_ZOME,
      fn_name: 'get_lien',
      payload: lienId,
    }) as Promise<HolochainRecord<Lien> | null>;
  }

  async getLiensForAsset(assetId: string): Promise<HolochainRecord<Lien>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: LIENS_ZOME,
      fn_name: 'get_liens_for_asset',
      payload: assetId,
    }) as Promise<HolochainRecord<Lien>[]>;
  }

  async satisfyLien(lienId: string): Promise<HolochainRecord<Lien>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: LIENS_ZOME,
      fn_name: 'satisfy_lien',
      payload: lienId,
    }) as Promise<HolochainRecord<Lien>>;
  }

  async releaseLien(lienId: string): Promise<HolochainRecord<Lien>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: LIENS_ZOME,
      fn_name: 'release_lien',
      payload: lienId,
    }) as Promise<HolochainRecord<Lien>>;
  }
}

// ============================================================================
// Commons Client
// ============================================================================

const COMMONS_ZOME = 'property_commons';

/**
 * Commons Client - Shared resource management
 */
export class CommonsClient {
  constructor(private readonly client: ZomeCallable) {}

  async createCommons(input: CreateCommonsInput): Promise<HolochainRecord<Commons>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'create_commons',
      payload: input,
    }) as Promise<HolochainRecord<Commons>>;
  }

  async getCommons(commonsId: string): Promise<HolochainRecord<Commons> | null> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_commons',
      payload: commonsId,
    }) as Promise<HolochainRecord<Commons> | null>;
  }

  async getCommonsByDAO(daoId: string): Promise<HolochainRecord<Commons>[]> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'get_commons_by_dao',
      payload: daoId,
    }) as Promise<HolochainRecord<Commons>[]>;
  }

  async joinCommons(commonsId: string): Promise<HolochainRecord<Commons>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'join_commons',
      payload: commonsId,
    }) as Promise<HolochainRecord<Commons>>;
  }

  async updateRules(commonsId: string, rules: string): Promise<HolochainRecord<Commons>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'update_rules',
      payload: { commons_id: commonsId, rules },
    }) as Promise<HolochainRecord<Commons>>;
  }

  async addToCommons(commonsId: string, assetId: string): Promise<HolochainRecord<Commons>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'add_asset_to_commons',
      payload: { commons_id: commonsId, asset_id: assetId },
    }) as Promise<HolochainRecord<Commons>>;
  }

  async removeFromCommons(commonsId: string, assetId: string): Promise<HolochainRecord<Commons>> {
    return this.client.callZome({
      role_name: PROPERTY_ROLE,
      zome_name: COMMONS_ZOME,
      fn_name: 'remove_asset_from_commons',
      payload: { commons_id: commonsId, asset_id: assetId },
    }) as Promise<HolochainRecord<Commons>>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createPropertyClients(client: ZomeCallable) {
  return {
    registry: new RegistryClient(client),
    transfers: new TransferClient(client),
    liens: new LienClient(client),
    commons: new CommonsClient(client),
  };
}

export default {
  RegistryClient,
  TransferClient,
  LienClient,
  CommonsClient,
  createPropertyClients,
};
