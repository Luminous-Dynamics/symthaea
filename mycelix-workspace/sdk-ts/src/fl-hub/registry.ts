// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Model Registry
 *
 * Manages the lifecycle of federated learning models including
 * registration, versioning, checkpoints, and deployment.
 */

import type {
  ModelId,
  ModelMetadata,
  ModelCheckpoint,
  ModelDeployment,
  ModelArchitecture,
  TrainingMetrics,
  SessionId,
} from './types.js';
import type { AggregationMethod } from '../fl/index.js';
import type { AgentId, HappId } from '../utils/index.js';

// =============================================================================
// Model Registry
// =============================================================================

export interface ModelRegistryConfig {
  maxCheckpointsPerModel: number;
  enableVersioning: boolean;
  compressCheckpoints: boolean;
}

const DEFAULT_REGISTRY_CONFIG: ModelRegistryConfig = {
  maxCheckpointsPerModel: 10,
  enableVersioning: true,
  compressCheckpoints: true,
};

/**
 * Registry for managing federated learning models
 */
export class ModelRegistry {
  private config: ModelRegistryConfig;
  private models: Map<ModelId, ModelMetadata> = new Map();
  private checkpoints: Map<ModelId, ModelCheckpoint[]> = new Map();
  private deployments: Map<string, ModelDeployment> = new Map();

  constructor(config: Partial<ModelRegistryConfig> = {}) {
    this.config = { ...DEFAULT_REGISTRY_CONFIG, ...config };
  }

  // ===========================================================================
  // Model Management
  // ===========================================================================

  /**
   * Register a new model
   */
  registerModel(input: RegisterModelInput): ModelMetadata {
    const modelId = this.generateModelId();
    const metadata: ModelMetadata = {
      modelId,
      name: input.name,
      description: input.description ?? '',
      version: input.version ?? '1.0.0',
      architecture: input.architecture,
      inputShape: input.inputShape,
      outputShape: input.outputShape,
      parameterCount: input.parameterCount,
      createdAt: Date.now(),
      createdBy: input.createdBy,
      happId: input.happId,
      tags: input.tags ?? [],
    };

    this.models.set(modelId, metadata);
    this.checkpoints.set(modelId, []);

    return metadata;
  }

  /**
   * Get model by ID
   */
  getModel(modelId: ModelId): ModelMetadata | null {
    return this.models.get(modelId) ?? null;
  }

  /**
   * List all models
   */
  listModels(filter?: ModelFilter): ModelMetadata[] {
    let models = Array.from(this.models.values());

    if (filter?.architecture) {
      models = models.filter((m) => m.architecture === filter.architecture);
    }
    if (filter?.happId) {
      models = models.filter((m) => m.happId === filter.happId);
    }
    if (filter?.createdBy) {
      models = models.filter((m) => m.createdBy === filter.createdBy);
    }
    if (filter?.tags && filter.tags.length > 0) {
      models = models.filter((m) =>
        filter.tags!.some((tag) => m.tags.includes(tag))
      );
    }

    return models;
  }

  /**
   * Update model metadata
   */
  updateModel(modelId: ModelId, updates: Partial<ModelMetadata>): boolean {
    const model = this.models.get(modelId);
    if (!model) return false;

    // Don't allow updating immutable fields
    const { modelId: _id, createdAt: _ca, createdBy: _cb, ...allowedUpdates } = updates;
    this.models.set(modelId, { ...model, ...allowedUpdates });
    return true;
  }

  /**
   * Delete a model and all its checkpoints
   */
  deleteModel(modelId: ModelId): boolean {
    if (!this.models.has(modelId)) return false;

    this.models.delete(modelId);
    this.checkpoints.delete(modelId);

    // Remove associated deployments
    for (const [id, deployment] of this.deployments) {
      if (deployment.modelId === modelId) {
        this.deployments.delete(id);
      }
    }

    return true;
  }

  // ===========================================================================
  // Checkpoint Management
  // ===========================================================================

  /**
   * Save a model checkpoint
   */
  saveCheckpoint(input: SaveCheckpointInput): ModelCheckpoint {
    const modelId = input.modelId;
    if (!this.models.has(modelId)) {
      throw new Error(`Model ${modelId} not found`);
    }

    const checkpoint: ModelCheckpoint = {
      checkpointId: this.generateCheckpointId(),
      modelId,
      sessionId: input.sessionId,
      round: input.round,
      parameters: input.parameters,
      metrics: input.metrics,
      timestamp: Date.now(),
      participantCount: input.participantCount,
      aggregationMethod: input.aggregationMethod,
    };

    const modelCheckpoints = this.checkpoints.get(modelId) ?? [];
    modelCheckpoints.push(checkpoint);

    // Enforce max checkpoints limit
    while (modelCheckpoints.length > this.config.maxCheckpointsPerModel) {
      modelCheckpoints.shift(); // Remove oldest
    }

    this.checkpoints.set(modelId, modelCheckpoints);
    return checkpoint;
  }

  /**
   * Get a specific checkpoint
   */
  getCheckpoint(checkpointId: string): ModelCheckpoint | null {
    for (const checkpoints of this.checkpoints.values()) {
      const found = checkpoints.find((c) => c.checkpointId === checkpointId);
      if (found) return found;
    }
    return null;
  }

  /**
   * List checkpoints for a model
   */
  listCheckpoints(modelId: ModelId): ModelCheckpoint[] {
    return this.checkpoints.get(modelId) ?? [];
  }

  /**
   * Get the latest checkpoint for a model
   */
  getLatestCheckpoint(modelId: ModelId): ModelCheckpoint | null {
    const checkpoints = this.checkpoints.get(modelId);
    if (!checkpoints || checkpoints.length === 0) return null;
    return checkpoints[checkpoints.length - 1];
  }

  /**
   * Get the best checkpoint by metric
   */
  getBestCheckpoint(
    modelId: ModelId,
    metric: keyof TrainingMetrics,
    minimize: boolean = true
  ): ModelCheckpoint | null {
    const checkpoints = this.checkpoints.get(modelId);
    if (!checkpoints || checkpoints.length === 0) return null;

    return checkpoints.reduce((best, current) => {
      const bestValue = best.metrics[metric] as number | undefined;
      const currentValue = current.metrics[metric] as number | undefined;

      if (bestValue === undefined) return current;
      if (currentValue === undefined) return best;

      if (minimize) {
        return currentValue < bestValue ? current : best;
      } else {
        return currentValue > bestValue ? current : best;
      }
    });
  }

  // ===========================================================================
  // Deployment Management
  // ===========================================================================

  /**
   * Deploy a model checkpoint
   */
  deployModel(input: DeployModelInput): ModelDeployment {
    const checkpoint = this.getCheckpoint(input.checkpointId);
    if (!checkpoint) {
      throw new Error(`Checkpoint ${input.checkpointId} not found`);
    }

    const deployment: ModelDeployment = {
      deploymentId: this.generateDeploymentId(),
      modelId: checkpoint.modelId,
      checkpointId: input.checkpointId,
      deployedAt: Date.now(),
      endpoint: input.endpoint,
      status: 'active',
    };

    // Deactivate previous deployments of this model
    for (const [id, existing] of this.deployments) {
      if (existing.modelId === checkpoint.modelId && existing.status === 'active') {
        this.deployments.set(id, { ...existing, status: 'inactive' });
      }
    }

    this.deployments.set(deployment.deploymentId, deployment);
    return deployment;
  }

  /**
   * Get active deployment for a model
   */
  getActiveDeployment(modelId: ModelId): ModelDeployment | null {
    for (const deployment of this.deployments.values()) {
      if (deployment.modelId === modelId && deployment.status === 'active') {
        return deployment;
      }
    }
    return null;
  }

  /**
   * List all deployments
   */
  listDeployments(modelId?: ModelId): ModelDeployment[] {
    const deployments = Array.from(this.deployments.values());
    if (modelId) {
      return deployments.filter((d) => d.modelId === modelId);
    }
    return deployments;
  }

  /**
   * Update deployment status
   */
  updateDeploymentStatus(
    deploymentId: string,
    status: ModelDeployment['status']
  ): boolean {
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) return false;

    this.deployments.set(deploymentId, { ...deployment, status });
    return true;
  }

  // ===========================================================================
  // Statistics
  // ===========================================================================

  /**
   * Get registry statistics
   */
  getStats(): RegistryStats {
    let totalCheckpoints = 0;
    let totalParameters = 0;

    for (const checkpoints of this.checkpoints.values()) {
      totalCheckpoints += checkpoints.length;
    }

    for (const model of this.models.values()) {
      totalParameters += model.parameterCount;
    }

    return {
      totalModels: this.models.size,
      totalCheckpoints,
      totalDeployments: this.deployments.size,
      activeDeployments: Array.from(this.deployments.values()).filter(
        (d) => d.status === 'active'
      ).length,
      totalParameters,
      modelsByArchitecture: this.countByArchitecture(),
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private generateModelId(): ModelId {
    return `model-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private generateCheckpointId(): string {
    return `ckpt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private generateDeploymentId(): string {
    return `deploy-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private countByArchitecture(): Record<ModelArchitecture, number> {
    const counts: Record<ModelArchitecture, number> = {
      linear: 0,
      mlp: 0,
      cnn: 0,
      rnn: 0,
      lstm: 0,
      transformer: 0,
      custom: 0,
    };

    for (const model of this.models.values()) {
      counts[model.architecture]++;
    }

    return counts;
  }
}

// =============================================================================
// Input Types
// =============================================================================

export interface RegisterModelInput {
  name: string;
  description?: string;
  version?: string;
  architecture: ModelArchitecture;
  inputShape: number[];
  outputShape: number[];
  parameterCount: number;
  createdBy: AgentId;
  happId?: HappId;
  tags?: string[];
}

export interface SaveCheckpointInput {
  modelId: ModelId;
  sessionId: SessionId;
  round: number;
  parameters: Uint8Array;
  metrics: TrainingMetrics;
  participantCount: number;
  aggregationMethod: AggregationMethod;
}

export interface DeployModelInput {
  checkpointId: string;
  endpoint?: string;
}

export interface ModelFilter {
  architecture?: ModelArchitecture;
  happId?: HappId;
  createdBy?: AgentId;
  tags?: string[];
}

export interface RegistryStats {
  totalModels: number;
  totalCheckpoints: number;
  totalDeployments: number;
  activeDeployments: number;
  totalParameters: number;
  modelsByArchitecture: Record<ModelArchitecture, number>;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new model registry
 */
export function createModelRegistry(
  config?: Partial<ModelRegistryConfig>
): ModelRegistry {
  return new ModelRegistry(config);
}
