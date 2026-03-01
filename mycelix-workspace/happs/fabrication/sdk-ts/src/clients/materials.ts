/**
 * Materials Client
 *
 * Handles material specifications:
 * - Material registry (create, get)
 * - Type-based discovery
 * - Food-safe material filtering
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  MaterialType,
  MaterialProperties,
  Certification,
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

export interface CreateMaterialInput {
  name: string;
  material_type: MaterialType;
  properties: MaterialProperties;
  certifications: Certification[];
  safety_data_sheet?: string;
}

export class MaterialsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'materials_coordinator'
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
   * Get materials by type
   */
  async getByType(
    materialType: MaterialType,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_materials_by_type',
      payload: { material_type: materialType, pagination: pagination ?? null },
    });
  }

  /**
   * Get food-safe materials only
   */
  async getFoodSafe(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_food_safe_materials',
      payload: { pagination: pagination ?? null },
    });
  }
}
