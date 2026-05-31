// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Symthaea Client
 *
 * Handles HDC (Hyperdimensional Computing) operations including:
 * - Natural language to intent vector encoding
 * - Lateral binding of semantic concepts
 * - Semantic similarity search
 * - Parametric design generation
 * - Local optimization
 * - Repair prediction from sensor data
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  HdcHypervector,
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

export interface CreateIntentInput {
  description: string;
  language?: string;
}

export interface LateralBindInput {
  baseIntentHash: ActionHash;
  modifierDescriptions: string[];
}

export interface SemanticSearchInput {
  intentHash: ActionHash;
  threshold?: number;
  limit?: number;
}

export interface GenerateVariantInput {
  baseDesignHash: ActionHash;
  intentModifiers: SemanticBinding[];
  materialConstraints: MaterialType[];
  printerConstraints?: PrinterCapabilities;
}

export interface OptimizeLocalInput {
  designHash: ActionHash;
  localMaterials: ActionHash[];
  localPrinters: ActionHash[];
  energyPreference: EnergyType;
}

export interface PredictRepairInput {
  propertyAssetHash: ActionHash;
  sensorHistory: SensorReading[];
  usageHours: number;
}

export class SymthaeaClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'symthaea'
  ) {}

  // =========================================================================
  // INTENT VECTOR GENERATION
  // =========================================================================

  /**
   * Generate HDC hypervector from natural language description
   *
   * Example: "I need a bracket for a 12mm pipe that's weatherproof"
   *
   * The description is parsed into semantic bindings:
   * - Base: bracket
   * - Dimensional: 12mm
   * - Modifier: weatherproof
   *
   * These are then encoded into a 10,000-dimensional bipolar hypervector
   * using HDC operations (bundling, binding, permutation).
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
  async getMyIntents(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_intents',
      payload: null,
    });
  }

  // =========================================================================
  // LATERAL BINDING (Vector Composition)
  // =========================================================================

  /**
   * Combine base intent with modifiers using lateral binding
   *
   * HDC Operation: bracket_vector ⊛ 12mm_vector ⊛ weatherproof_vector
   *
   * Lateral binding creates a new hypervector that is similar to all
   * component vectors but distinct from each. This enables semantic
   * composition while preserving searchability.
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
   *
   * Uses cosine similarity between hypervectors to find designs
   * that are semantically similar to the query intent, even if
   * they use different words or terminology.
   *
   * @param input Search parameters including intent hash and threshold
   * @returns Array of matching designs with similarity scores
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
    // First generate intent
    const intent = await this.generateIntentVector({ description });

    // Then search
    return this.semanticSearch({
      intentHash: intent.record.signed_action.hashed.hash,
      threshold,
      limit,
    });
  }

  // =========================================================================
  // PARAMETRIC GENERATION
  // =========================================================================

  /**
   * Generate a parametric variant from intent + constraints
   *
   * Takes a base design and modifies its parameters based on:
   * - Semantic intent modifiers (HDC bindings)
   * - Material constraints
   * - Printer capability constraints
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
   *
   * Adjusts design parameters to optimize for:
   * - Locally available materials
   * - Local printer capabilities
   * - Energy preference (renewable sources)
   *
   * This supports the metabolic economy by prioritizing local resources.
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
  async getDesignOptimizations(designHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_optimizations',
      payload: designHash,
    });
  }

  // =========================================================================
  // REPAIR PREDICTION (Anticipatory Repair Loop)
  // =========================================================================

  /**
   * Predict repair needs from digital twin sensor data
   *
   * Part of the Anticipatory Repair Loop:
   * 1. Property hApp provides digital twin sensor data
   * 2. Symthaea analyzes degradation patterns
   * 3. Predicts likely failure component and timing
   * 4. Recommends action (monitor, order, print replacement)
   * 5. Searches for matching repair designs
   *
   * This enables parts to arrive BEFORE failure occurs.
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

  /**
   * Create a semantic binding
   */
  createBinding(concept: string, role: BindingRole, weight: number = 1.0): SemanticBinding {
    return { concept, role, weight };
  }

  /**
   * Create common modifier bindings
   */
  createModifiers(options: {
    size?: string;
    material?: string;
    property?: string;
    function?: string;
  }): SemanticBinding[] {
    const bindings: SemanticBinding[] = [];

    if (options.size) {
      bindings.push(this.createBinding(options.size, 'Dimensional', 0.9));
    }
    if (options.material) {
      bindings.push(this.createBinding(options.material, 'Material', 0.8));
    }
    if (options.property) {
      bindings.push(this.createBinding(options.property, 'Modifier', 0.8));
    }
    if (options.function) {
      bindings.push(this.createBinding(options.function, 'Functional', 0.9));
    }

    return bindings;
  }

  /**
   * Parse description into semantic bindings (client-side preview)
   *
   * Note: The actual parsing happens on-chain in the zome.
   * This method provides a preview for UI purposes.
   */
  parseDescription(description: string): SemanticBinding[] {
    const bindings: SemanticBinding[] = [];
    const lower = description.toLowerCase();

    // Object types
    const objects = [
      'bracket',
      'mount',
      'holder',
      'clip',
      'adapter',
      'enclosure',
      'gear',
      'hinge',
      'knob',
      'handle',
      'hook',
      'stand',
      'cover',
      'case',
      'container',
      'box',
    ];
    for (const obj of objects) {
      if (lower.includes(obj)) {
        bindings.push({ concept: obj, role: 'Base', weight: 1.0 });
      }
    }

    // Dimensions
    const dimMatch = lower.match(/(\d+)\s*(mm|cm|inch|m\d)/gi);
    if (dimMatch) {
      for (const dim of dimMatch) {
        bindings.push({ concept: dim, role: 'Dimensional', weight: 0.9 });
      }
    }

    // Materials
    const materials = ['pla', 'petg', 'abs', 'tpu', 'nylon', 'food-safe', 'food safe'];
    for (const mat of materials) {
      if (lower.includes(mat)) {
        bindings.push({ concept: mat, role: 'Material', weight: 0.8 });
      }
    }

    // Properties
    const properties = [
      'weatherproof',
      'waterproof',
      'uv-resistant',
      'heat-resistant',
      'heavy-duty',
      'lightweight',
      'flexible',
      'rigid',
      'strong',
    ];
    for (const prop of properties) {
      if (lower.includes(prop)) {
        bindings.push({ concept: prop, role: 'Modifier', weight: 0.8 });
      }
    }

    // Functions
    const functions = ['load-bearing', 'decorative', 'structural', 'replacement', 'repair', 'custom'];
    for (const func of functions) {
      if (lower.includes(func)) {
        bindings.push({ concept: func, role: 'Functional', weight: 0.9 });
      }
    }

    return bindings;
  }
}
