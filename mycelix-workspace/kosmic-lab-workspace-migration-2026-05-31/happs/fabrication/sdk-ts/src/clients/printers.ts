// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Printers Client
 *
 * Handles printer registry operations including:
 * - Registration and management
 * - Discovery and matching
 * - Availability status
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  Printer,
  PrinterType,
  PrinterCapabilities,
  MaterialType,
  GeoLocation,
  AvailabilityStatus,
  PrinterMatch,
  RegisterPrinterInput,
} from '../types';

export interface PrinterRequirements {
  buildVolumeMin?: { x: number; y: number; z: number };
  materials?: MaterialType[];
  features?: string[];
  maxLayerHeight?: number;
  heatedBed?: boolean;
  enclosure?: boolean;
}

export interface CompatibilityResult {
  compatible: boolean;
  score: number;
  issues: string[];
  recommendations: string[];
}

export class PrintersClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'printers'
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
  async update(input: {
    printerHash: ActionHash;
    name?: string;
    capabilities?: PrinterCapabilities;
    materialsAvailable?: MaterialType[];
    location?: GeoLocation;
  }): Promise<Record> {
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
  async getMine(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_printers',
      payload: null,
    });
  }

  /**
   * Find printers near a location
   */
  async findNearby(location: GeoLocation, radiusKm: number = 50): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'find_printers_nearby',
      payload: { location, radius_km: radiusKm },
    });
  }

  /**
   * Find printers by capability requirements
   */
  async findByCapability(requirements: PrinterRequirements): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'find_printers_by_capability',
      payload: requirements,
    });
  }

  /**
   * Get all currently available printers
   */
  async getAvailable(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_available_printers',
      payload: null,
    });
  }

  /**
   * Match a design to compatible printers
   */
  async matchDesign(designHash: ActionHash): Promise<PrinterMatch[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'match_design_to_printers',
      payload: designHash,
    });
  }

  /**
   * Check if a specific printer is compatible with a design
   */
  async checkCompatibility(
    printerHash: ActionHash,
    designHash: ActionHash
  ): Promise<CompatibilityResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'check_printer_compatibility',
      payload: { printer: printerHash, design: designHash },
    });
  }

  /**
   * Update printer availability status
   */
  async updateAvailability(hash: ActionHash, status: AvailabilityStatus): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_availability',
      payload: { hash, status },
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
