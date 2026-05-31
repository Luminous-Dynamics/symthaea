// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Client
 *
 * Handles cross-hApp integration including:
 * - Anticipatory Repair Loop (Property → Knowledge → Fabrication)
 * - Marketplace integration
 * - Supply Chain integration
 * - Event emission and subscription
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  RepairPrediction,
  RepairWorkflow,
  RepairWorkflowStatus,
  RepairAction,
  MarketplaceListing,
  ListingType,
  SupplyChainLink,
  GeoLocation,
  License,
  SafetyClass,
} from '../types';

export interface CreateRepairPredictionInput {
  propertyAssetHash: ActionHash;
  assetModel: string;
  predictedFailureComponent: string;
  failureProbability: number;
  estimatedFailureDate: number;
  confidenceIntervalDays: number;
  sensorDataSummary: string;
}

export interface UpdateWorkflowInput {
  workflowHash: ActionHash;
  status: RepairWorkflowStatus;
  designHash?: ActionHash;
  printerHash?: ActionHash;
  hearthFundingHash?: ActionHash;
  printJobHash?: ActionHash;
}

export interface FabricationEvent {
  eventType: string;
  designId?: ActionHash;
  payload: string;
  sourceHapp: string;
  timestamp: number;
}

export interface MarketplaceListingInput {
  designHash: ActionHash;
  price?: number;
  listingType: ListingType;
}

export interface AvailabilityResult {
  available: boolean;
  suppliers: Array<{
    did: string;
    quantity: number;
    price?: number;
    leadTimeDays?: number;
  }>;
  localOptions: number;
}

export interface ComplianceResult {
  compliant: boolean;
  license: License;
  restrictions: string[];
  attributionRequired: boolean;
}

export class BridgeClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'bridge'
  ) {}

  // =========================================================================
  // ANTICIPATORY REPAIR LOOP
  // =========================================================================

  /**
   * Create a repair prediction from digital twin data
   *
   * Part of the Anticipatory Repair Loop:
   * Property hApp → Fabrication → Knowledge → Printer → Installed
   */
  async createRepairPrediction(input: CreateRepairPredictionInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_repair_prediction',
      payload: input,
    });
  }

  /**
   * Create a repair workflow from a prediction
   */
  async createRepairWorkflow(predictionHash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_repair_workflow',
      payload: predictionHash,
    });
  }

  /**
   * Update repair workflow status
   */
  async updateRepairWorkflow(input: UpdateWorkflowInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_repair_workflow',
      payload: input,
    });
  }

  /**
   * Get active repair workflows
   */
  async getActiveWorkflows(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_active_workflows',
      payload: null,
    });
  }

  /**
   * Get repair predictions for an asset
   */
  async getAssetPredictions(assetHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_asset_predictions',
      payload: assetHash,
    });
  }

  // =========================================================================
  // EVENT SYSTEM
  // =========================================================================

  /**
   * Emit a fabrication event
   */
  async emitEvent(
    eventType: string,
    designId?: ActionHash,
    payload: string = '{}'
  ): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'emit_fabrication_event',
      payload: { event_type: eventType, design_id: designId, payload },
    });
  }

  /**
   * Get recent fabrication events
   */
  async getRecentEvents(since?: number): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_recent_events',
      payload: since || null,
    });
  }

  // =========================================================================
  // MARKETPLACE INTEGRATION
  // =========================================================================

  /**
   * List a design on the marketplace
   */
  async listOnMarketplace(input: MarketplaceListingInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'list_design_on_marketplace',
      payload: input,
    });
  }

  /**
   * Get marketplace status for a design
   */
  async getMarketplaceStatus(designHash: ActionHash): Promise<Record | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_marketplace_status',
      payload: designHash,
    });
  }

  /**
   * Handle design purchase callback
   */
  async onDesignPurchased(designHash: ActionHash, buyerPubKey: Uint8Array): Promise<void> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'on_design_purchased',
      payload: { design: designHash, buyer: buyerPubKey },
    });
  }

  // =========================================================================
  // SUPPLY CHAIN INTEGRATION
  // =========================================================================

  /**
   * Link material to a supplier
   */
  async linkMaterialToSupplier(
    materialHash: ActionHash,
    supplierDid: string,
    supplyChainItemHash?: ActionHash
  ): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'link_material_to_supplier',
      payload: {
        material_hash: materialHash,
        supplier_did: supplierDid,
        supplychain_item_hash: supplyChainItemHash,
      },
    });
  }

  /**
   * Query material availability from supply chain
   */
  async queryMaterialAvailability(
    materialHash: ActionHash,
    location: GeoLocation
  ): Promise<AvailabilityResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'query_material_availability',
      payload: { material: materialHash, location },
    });
  }

  /**
   * Handle material order callback
   */
  async onMaterialOrderPlaced(materialHash: ActionHash, orderHash: ActionHash): Promise<void> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'on_material_order_placed',
      payload: { material: materialHash, order_hash: orderHash },
    });
  }

  // =========================================================================
  // KNOWLEDGE INTEGRATION
  // =========================================================================

  /**
   * Request design verification through Knowledge hApp markets
   */
  async requestVerificationMarket(
    designHash: ActionHash,
    targetSafetyClass: SafetyClass
  ): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'request_design_verification_market',
      payload: { design: designHash, target: targetSafetyClass },
    });
  }

  /**
   * Sync epistemic scores from Knowledge hApp
   */
  async syncEpistemicScores(designHash: ActionHash): Promise<{
    empirical: number;
    normative: number;
    mythic: number;
    overallConfidence: number;
  }> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'sync_epistemic_scores',
      payload: designHash,
    });
  }

  // =========================================================================
  // PROPERTY INTEGRATION (IP/Licensing)
  // =========================================================================

  /**
   * Register design IP
   */
  async registerDesignIp(designHash: ActionHash, license: License): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'register_design_ip',
      payload: { design: designHash, license },
    });
  }

  /**
   * Check license compliance for usage
   */
  async checkLicenseCompliance(
    designHash: ActionHash,
    usageType: 'personal' | 'commercial' | 'derivative' | 'distribution'
  ): Promise<ComplianceResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'check_license_compliance',
      payload: { design: designHash, usage: usageType },
    });
  }
}
