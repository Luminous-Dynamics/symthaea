// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Printers Client
 *
 * Handles printer registry operations including:
 * - Registration and management
 * - Discovery and matching (geohash + Haversine)
 * - Availability status
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  PrinterType,
  PrinterCapabilities,
  MaterialType,
  GeoLocation,
  AvailabilityStatus,
  PrinterMatch,
  RegisterPrinterInput,
  PrinterRates,
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

export interface UpdatePrinterInput {
  original_action_hash: ActionHash;
  name?: string;
  location?: GeoLocation;
  capabilities?: PrinterCapabilities;
  materials_available?: MaterialType[];
  rates?: PrinterRates;
}

export interface PrinterRequirements {
  build_volume_min?: { x: number; y: number; z: number };
  materials?: MaterialType[];
  features?: string[];
  max_layer_height?: number;
  heated_bed?: boolean;
  enclosure?: boolean;
}

export interface FindNearbyInput {
  location: GeoLocation;
  radius_km: number;
  pagination?: PaginationInput;
}

export interface FindByCapabilityInput {
  requirements: PrinterRequirements;
  pagination?: PaginationInput;
}

export interface MatchDesignInput {
  design_hash: ActionHash;
  location?: GeoLocation;
  limit?: number;
}

export interface CheckCompatibilityInput {
  printer_hash: ActionHash;
  design_hash: ActionHash;
}

export interface CompatibilityResult {
  compatible: boolean;
  score: number;
  issues: string[];
  recommendations: string[];
}

export interface UpdateAvailabilityInput {
  printer_hash: ActionHash;
  status: AvailabilityStatus;
  message?: string;
  eta_available?: number;
  current_job?: ActionHash;
}

export class PrintersClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'printers_coordinator'
  ) {}

  /**
   * Register a new printer
   */
  async register(input: RegisterPrinterInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'register_printer',
      payload: input,
    });
  }

  /**
   * Get a printer by hash
   */
  async get(hash: ActionHash): Promise<Record | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_printer',
      payload: hash,
    });
  }

  /**
   * Update printer details
   */
  async update(input: UpdatePrinterInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_printer',
      payload: input,
    });
  }

  /**
   * Deactivate a printer
   */
  async deactivate(hash: ActionHash): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'deactivate_printer',
      payload: hash,
    });
  }

  /**
   * Get my registered printers
   */
  async getMine(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_printers',
      payload: { pagination: pagination ?? null },
    });
  }

  /**
   * Find printers near a location (geohash prefix + Haversine refinement)
   */
  async findNearby(input: FindNearbyInput): Promise<PaginatedResponse<PrinterMatch>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'find_printers_nearby',
      payload: input,
    });
  }

  /**
   * Find printers by capability requirements
   */
  async findByCapability(input: FindByCapabilityInput): Promise<PaginatedResponse<PrinterMatch>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'find_printers_by_capability',
      payload: input,
    });
  }

  /**
   * Get all currently available printers
   */
  async getAvailable(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_available_printers',
      payload: { pagination: pagination ?? null },
    });
  }

  /**
   * Match a design to compatible printers
   */
  async matchDesign(input: MatchDesignInput): Promise<PrinterMatch[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'match_design_to_printers',
      payload: input,
    });
  }

  /**
   * Check if a specific printer is compatible with a design
   */
  async checkCompatibility(input: CheckCompatibilityInput): Promise<CompatibilityResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'check_printer_compatibility',
      payload: input,
    });
  }

  /**
   * Update printer availability status
   */
  async updateAvailability(input: UpdateAvailabilityInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_availability',
      payload: input,
    });
  }

  /**
   * Get print queue for a printer
   */
  async getQueue(printerHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_printer_queue',
      payload: printerHash,
    });
  }
}
