/**
 * Symthaea Client
 *
 * Handles HDC (Hyperdimensional Computing) operations including:
 * - Natural language to intent vector encoding
 * - Lateral binding of semantic concepts
 * - Semantic similarity search (standard + LSH)
 * - Parametric design generation
 * - Local optimization
 * - Repair prediction from sensor data
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  SemanticBinding,
  BindingRole,
  IntentResult,
  SearchResult,
  RepairPredictionResult,
  SensorReading,
  EnergyType,
  PrinterCapabilities,
  MaterialType,
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

export interface CreateIntentInput {
  description: string;
  language?: string;
}

export interface LateralBindInput {
  base_intent_hash: ActionHash;
  modifier_descriptions: string[];
}

export interface SemanticSearchInput {
  intent_hash: ActionHash;
  threshold?: number;
  limit?: number;
}

export interface GenerateVariantInput {
  base_design_hash: ActionHash;
  intent_modifiers: SemanticBinding[];
  material_constraints: MaterialType[];
  printer_constraints?: PrinterCapabilities;
}

export interface OptimizeLocalInput {
  design_hash: ActionHash;
  local_materials: ActionHash[];
  local_printers: ActionHash[];
  energy_preference: EnergyType;
}

export interface PredictRepairInput {
  property_asset_hash: ActionHash;
  sensor_history: SensorReading[];
  usage_hours: number;
}

export class SymthaeaClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'symthaea_coordinator'
  ) {}

  // =========================================================================
  // INTENT VECTOR GENERATION
  // =========================================================================

  /**
   * Generate HDC hypervector from natural language description
   *
   * The description is parsed into semantic bindings and encoded into a
   * 16,384-dimensional bipolar hypervector using HDC operations.
   */
  async generateIntentVector(input: CreateIntentInput): Promise<IntentResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'generate_intent_vector',
      payload: input,
    });
  }

  /**
   * Get all my intent vectors
   */
  async getMyIntents(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_intents',
      payload: { pagination: pagination ?? null },
    });
  }

  // =========================================================================
  // LATERAL BINDING (Vector Composition)
  // =========================================================================

  /**
   * Combine base intent with modifiers using lateral binding
   */
  async lateralBind(input: LateralBindInput): Promise<IntentResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'lateral_bind',
      payload: input,
    });
  }

  // =========================================================================
  // SEMANTIC SEARCH
  // =========================================================================

  /**
   * Find designs by semantic similarity in HDC space
   */
  async semanticSearch(input: SemanticSearchInput): Promise<SearchResult[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'semantic_search',
      payload: input,
    });
  }

  /**
   * Quick search by description (generates intent then searches)
   */
  async searchByDescription(
    description: string,
    threshold: number = 0.7,
    limit: number = 10
  ): Promise<SearchResult[]> {
    const intent = await this.generateIntentVector({ description });
    return this.semanticSearch({
      intent_hash: intent.record.signed_action.hashed.hash,
      threshold,
      limit,
    });
  }

  // =========================================================================
  // PARAMETRIC GENERATION
  // =========================================================================

  /**
   * Generate a parametric variant from intent + constraints
   */
  async generateParametricVariant(input: GenerateVariantInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'generate_parametric_variant',
      payload: input,
    });
  }

  // =========================================================================
  // LOCAL OPTIMIZATION
  // =========================================================================

  /**
   * Optimize design for local conditions
   */
  async optimizeForLocal(input: OptimizeLocalInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'optimize_for_local',
      payload: input,
    });
  }

  /**
   * Get optimization history for a design
   */
  async getDesignOptimizations(
    designHash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_optimizations',
      payload: { hash: designHash, pagination: pagination ?? null },
    });
  }

  // =========================================================================
  // REPAIR PREDICTION (Anticipatory Repair Loop)
  // =========================================================================

  /**
   * Predict repair needs from digital twin sensor data
   */
  async predictRepairNeeds(input: PredictRepairInput): Promise<RepairPredictionResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'predict_repair_needs',
      payload: input,
    });
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  createBinding(concept: string, role: BindingRole, weight: number = 1.0): SemanticBinding {
    return { concept, role, weight };
  }

  createModifiers(options: {
    size?: string;
    material?: string;
    property?: string;
    function?: string;
  }): SemanticBinding[] {
    const bindings: SemanticBinding[] = [];
    if (options.size) bindings.push(this.createBinding(options.size, 'Dimensional', 0.9));
    if (options.material) bindings.push(this.createBinding(options.material, 'Material', 0.8));
    if (options.property) bindings.push(this.createBinding(options.property, 'Modifier', 0.8));
    if (options.function) bindings.push(this.createBinding(options.function, 'Functional', 0.9));
    return bindings;
  }

  parseDescription(description: string): SemanticBinding[] {
    const bindings: SemanticBinding[] = [];
    const lower = description.toLowerCase();

    const objects = ['bracket', 'mount', 'holder', 'clip', 'adapter', 'enclosure', 'gear', 'hinge', 'knob', 'handle', 'hook', 'stand', 'cover', 'case', 'container', 'box'];
    for (const obj of objects) {
      if (lower.includes(obj)) bindings.push({ concept: obj, role: 'Base', weight: 1.0 });
    }

    const dimMatch = lower.match(/(\d+)\s*(mm|cm|inch|m\d)/gi);
    if (dimMatch) {
      for (const dim of dimMatch) bindings.push({ concept: dim, role: 'Dimensional', weight: 0.9 });
    }

    const materials = ['pla', 'petg', 'abs', 'tpu', 'nylon', 'food-safe', 'food safe'];
    for (const mat of materials) {
      if (lower.includes(mat)) bindings.push({ concept: mat, role: 'Material', weight: 0.8 });
    }

    const properties = ['weatherproof', 'waterproof', 'uv-resistant', 'heat-resistant', 'heavy-duty', 'lightweight', 'flexible', 'rigid', 'strong'];
    for (const prop of properties) {
      if (lower.includes(prop)) bindings.push({ concept: prop, role: 'Modifier', weight: 0.8 });
    }

    const functions = ['load-bearing', 'decorative', 'structural', 'replacement', 'repair', 'custom'];
    for (const func of functions) {
      if (lower.includes(func)) bindings.push({ concept: func, role: 'Functional', weight: 0.9 });
    }

    return bindings;
  }
}
