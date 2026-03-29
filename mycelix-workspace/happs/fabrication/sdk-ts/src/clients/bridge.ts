// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Client
 *
 * Handles cross-hApp integration including:
 * - Anticipatory Repair Loop (Property -> Fabrication -> Printer)
 * - Marketplace integration
 * - Supply Chain integration
 * - Event emission and audit trail
 */

import type { AppClient, ActionHash, AgentPubKey, Record } from '@holochain/client';
import type {
  RepairWorkflowStatus,
  ListingType,
} from '../types';

export interface PaginationInput {
  offset: number;
  limit: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface CreateRepairPredictionInput {
  property_asset_hash: ActionHash;
  asset_model: string;
  predicted_failure_component: string;
  failure_probability: number;
  estimated_failure_date: number;
  confidence_interval_days: number;
  sensor_data_summary: string;
}

export interface UpdateWorkflowInput {
  workflow_hash: ActionHash;
  status: RepairWorkflowStatus;
  design_hash?: ActionHash;
  printer_hash?: ActionHash;
  hearth_funding_hash?: ActionHash;
  print_job_hash?: ActionHash;
}

export interface EmitEventInput {
  event_type: string;
  design_id?: ActionHash;
  payload: string;
}

export interface MarketplaceListingInput {
  design_hash: ActionHash;
  price: number;
  listing_type: ListingType;
}

export interface AuditTrailFilter {
  domain?: string;
  agent?: AgentPubKey;
  after?: number;
  before?: number;
  limit: number;
  pagination?: PaginationInput;
}

export class BridgeClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'bridge_coordinator'
  ) {}

  // =========================================================================
  // ANTICIPATORY REPAIR LOOP
  // =========================================================================

  /**
   * Create a repair prediction from digital twin data
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
  async getActiveWorkflows(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_active_workflows',
      payload: { pagination: pagination ?? null },
    });
  }

  // =========================================================================
  // EVENT SYSTEM
  // =========================================================================

  /**
   * Emit a fabrication event
   */
  async emitEvent(input: EmitEventInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'emit_fabrication_event',
      payload: input,
    });
  }

  /**
   * Get recent fabrication events
   */
  async getRecentEvents(
    since?: number,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_recent_events',
      payload: {
        since: since ?? null,
        pagination: pagination ?? { offset: 0, limit: 100 },
      },
    });
  }

  /**
   * Get audit trail entries
   */
  async getAuditTrail(filter: AuditTrailFilter): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_audit_trail',
      payload: filter,
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

  // =========================================================================
  // SUPPLY CHAIN INTEGRATION
  // =========================================================================

  /**
   * Link material to a supplier
   */
  async linkMaterialToSupplier(
    materialHash: ActionHash,
    supplierDid: string,
    supplychainItemHash?: ActionHash
  ): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'link_material_to_supplier',
      payload: {
        material_hash: materialHash,
        supplier_did: supplierDid,
        supplychain_item_hash: supplychainItemHash ?? null,
      },
    });
  }
}
