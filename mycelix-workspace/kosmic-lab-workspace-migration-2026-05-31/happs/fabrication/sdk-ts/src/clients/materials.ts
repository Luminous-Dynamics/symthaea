// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Materials Client
 *
 * Handles material specifications and sourcing:
 * - Material registry
 * - Certification tracking
 * - Supply chain integration
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  Material,
  MaterialType,
  MaterialProperties,
  Certification,
  CertificationType,
  GeoLocation,
} from '../types';

export interface CreateMaterialInput {
  name: string;
  materialType: MaterialType;
  properties: MaterialProperties;
  certifications: Certification[];
  safetyDataSheet?: string;
}

export interface SupplierInfo {
  supplierDid: string;
  name: string;
  location?: GeoLocation;
  price?: number;
  leadTimeDays?: number;
  minOrderQuantity?: number;
}

export interface AvailabilityInfo {
  available: boolean;
  suppliers: SupplierInfo[];
  estimatedLeadTime?: number;
  closestSupplierDistance?: number;
}

export class MaterialsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'materials'
  ) {}

  /**
   * Create a new material specification
   */
  async create(input: CreateMaterialInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_material',
      payload: input,
    });
  }

  /**
   * Get material by hash
   */
  async get(hash: ActionHash): Promise<Record | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_material',
      payload: hash,
    });
  }

  /**
   * Update material specification
   */
  async update(input: {
    materialHash: ActionHash;
    properties?: MaterialProperties;
    certifications?: Certification[];
    safetyDataSheet?: string;
  }): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_material',
      payload: input,
    });
  }

  /**
   * Get materials by type
   */
  async getByType(materialType: MaterialType): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_materials_by_type',
      payload: materialType,
    });
  }

  /**
   * Find materials compatible with a design
   */
  async findCompatible(designHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'find_compatible_materials',
      payload: designHash,
    });
  }

  /**
   * Get food-safe materials only
   */
  async getFoodSafe(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_food_safe_materials',
      payload: null,
    });
  }

  /**
   * Get materials by certification type
   */
  async getByCertification(certType: CertificationType): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_materials_by_certification',
      payload: certType,
    });
  }

  /**
   * Get suppliers for a material (via Supply Chain bridge)
   */
  async getSuppliers(materialHash: ActionHash): Promise<SupplierInfo[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_material_suppliers',
      payload: materialHash,
    });
  }

  /**
   * Check material availability at a location
   */
  async checkAvailability(
    materialHash: ActionHash,
    location: GeoLocation
  ): Promise<AvailabilityInfo> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'check_material_availability',
      payload: { material: materialHash, location },
    });
  }

  /**
   * Get all materials
   */
  async getAll(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_all_materials',
      payload: null,
    });
  }

  /**
   * Search materials by name or properties
   */
  async search(query: {
    name?: string;
    minTensileStrength?: number;
    foodSafe?: boolean;
    uvResistant?: boolean;
  }): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'search_materials',
      payload: query,
    });
  }
}
