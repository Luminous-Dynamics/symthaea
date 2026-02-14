/**
 * Federated Learning Hub Tests
 *
 * Tests for ModelRegistry, PrivacyManager, and FLHubCoordinator
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ModelRegistry,
  createModelRegistry,
  PrivacyManager,
  createPrivacyManager,
  createDevPrivacyBudget,
  createStrictPrivacyBudget,
  FLHubCoordinator,
  createFLHub,
  createFLHubWithRegistry,
  DEFAULT_FL_HUB_CONFIG,
  type SessionId,
  type ModelId,
  type ModelArchitecture,
  type SessionStatus,
  type ParticipantStatus,
} from '../src/fl-hub/index.js';
import { AggregationMethod } from '../src/fl/index.js';

// =============================================================================
// Type Export Tests
// =============================================================================

describe('FL Hub Type Exports', () => {
  it('should export type aliases', () => {
    const sessionId: SessionId = 'session-1';
    const modelId: ModelId = 'model-1';
    const architecture: ModelArchitecture = 'mlp';
    const sessionStatus: SessionStatus = 'training';
    const participantStatus: ParticipantStatus = 'submitted';

    expect(sessionId).toBe('session-1');
    expect(modelId).toBe('model-1');
    expect(architecture).toBe('mlp');
    expect(sessionStatus).toBe('training');
    expect(participantStatus).toBe('submitted');
  });

  it('should export DEFAULT_FL_HUB_CONFIG', () => {
    expect(DEFAULT_FL_HUB_CONFIG).toBeDefined();
    expect(typeof DEFAULT_FL_HUB_CONFIG).toBe('object');
  });
});

// =============================================================================
// ModelRegistry Tests
// =============================================================================

describe('ModelRegistry', () => {
  let registry: ModelRegistry;

  beforeEach(() => {
    registry = new ModelRegistry();
  });

  describe('construction', () => {
    it('should create with default config', () => {
      const reg = new ModelRegistry();
      expect(reg).toBeInstanceOf(ModelRegistry);
    });

    it('should create with custom config', () => {
      const reg = new ModelRegistry({
        maxCheckpointsPerModel: 20,
        enableVersioning: false,
      });
      expect(reg).toBeInstanceOf(ModelRegistry);
    });
  });

  describe('registerModel', () => {
    it('should register a new model', () => {
      const model = registry.registerModel({
        name: 'test-model',
        architecture: 'mlp',
        inputShape: [256],
        outputShape: [2],
        parameterCount: 10000,
        createdBy: 'agent-1',
      });

      expect(model).toBeDefined();
      expect(model.modelId).toBeDefined();
      expect(model.name).toBe('test-model');
      expect(model.architecture).toBe('mlp');
      expect(model.parameterCount).toBe(10000);
    });

    it('should register with optional fields', () => {
      const model = registry.registerModel({
        name: 'full-model',
        description: 'A test model',
        version: '2.0.0',
        architecture: 'transformer',
        inputShape: [512],
        outputShape: [10],
        parameterCount: 100000,
        createdBy: 'agent-1',
        happId: 'happ-1',
        tags: ['test', 'experimental'],
      });

      expect(model.description).toBe('A test model');
      expect(model.version).toBe('2.0.0');
      expect(model.happId).toBe('happ-1');
      expect(model.tags).toEqual(['test', 'experimental']);
    });

    it('should set createdAt timestamp', () => {
      const before = Date.now();
      const model = registry.registerModel({
        name: 'timestamp-test',
        architecture: 'linear',
        inputShape: [10],
        outputShape: [1],
        parameterCount: 100,
        createdBy: 'agent-1',
      });
      const after = Date.now();

      expect(model.createdAt).toBeGreaterThanOrEqual(before);
      expect(model.createdAt).toBeLessThanOrEqual(after);
    });
  });

  describe('getModel', () => {
    it('should retrieve registered model', () => {
      const registered = registry.registerModel({
        name: 'retrieval-test',
        architecture: 'cnn',
        inputShape: [28, 28, 1],
        outputShape: [10],
        parameterCount: 50000,
        createdBy: 'agent-1',
      });

      const retrieved = registry.getModel(registered.modelId);

      expect(retrieved).toEqual(registered);
    });

    it('should return null for unknown model', () => {
      const retrieved = registry.getModel('unknown-id');
      expect(retrieved).toBeNull();
    });
  });

  describe('listModels', () => {
    beforeEach(() => {
      registry.registerModel({
        name: 'model-1',
        architecture: 'mlp',
        inputShape: [100],
        outputShape: [5],
        parameterCount: 5000,
        createdBy: 'agent-1',
        happId: 'happ-1',
        tags: ['production'],
      });

      registry.registerModel({
        name: 'model-2',
        architecture: 'cnn',
        inputShape: [28, 28],
        outputShape: [10],
        parameterCount: 20000,
        createdBy: 'agent-2',
        happId: 'happ-2',
        tags: ['experimental'],
      });

      registry.registerModel({
        name: 'model-3',
        architecture: 'mlp',
        inputShape: [50],
        outputShape: [3],
        parameterCount: 3000,
        createdBy: 'agent-1',
        happId: 'happ-1',
        tags: ['production', 'stable'],
      });
    });

    it('should list all models without filter', () => {
      const models = registry.listModels();
      expect(models).toHaveLength(3);
    });

    it('should filter by architecture', () => {
      const mlpModels = registry.listModels({ architecture: 'mlp' });
      expect(mlpModels).toHaveLength(2);
      mlpModels.forEach((m) => expect(m.architecture).toBe('mlp'));
    });

    it('should filter by happId', () => {
      const happ1Models = registry.listModels({ happId: 'happ-1' });
      expect(happ1Models).toHaveLength(2);
    });

    it('should filter by createdBy', () => {
      const agent1Models = registry.listModels({ createdBy: 'agent-1' });
      expect(agent1Models).toHaveLength(2);
    });

    it('should filter by tags', () => {
      const prodModels = registry.listModels({ tags: ['production'] });
      expect(prodModels).toHaveLength(2);
    });
  });

  describe('updateModel', () => {
    it('should update model metadata', () => {
      const model = registry.registerModel({
        name: 'update-test',
        architecture: 'linear',
        inputShape: [10],
        outputShape: [1],
        parameterCount: 100,
        createdBy: 'agent-1',
      });

      const result = registry.updateModel(model.modelId, {
        name: 'updated-name',
        description: 'Updated description',
        tags: ['updated'],
      });

      expect(result).toBe(true);

      const updated = registry.getModel(model.modelId);
      expect(updated?.name).toBe('updated-name');
      expect(updated?.description).toBe('Updated description');
      expect(updated?.tags).toEqual(['updated']);
    });

    it('should return false for unknown model', () => {
      const result = registry.updateModel('unknown-id', { name: 'new' });
      expect(result).toBe(false);
    });
  });

  describe('deleteModel', () => {
    it('should delete model', () => {
      const model = registry.registerModel({
        name: 'delete-test',
        architecture: 'mlp',
        inputShape: [10],
        outputShape: [5],
        parameterCount: 1000,
        createdBy: 'agent-1',
      });

      const result = registry.deleteModel(model.modelId);
      expect(result).toBe(true);

      const retrieved = registry.getModel(model.modelId);
      expect(retrieved).toBeNull();
    });

    it('should return false for unknown model', () => {
      const result = registry.deleteModel('unknown-id');
      expect(result).toBe(false);
    });
  });

  describe('getStats', () => {
    it('should return registry statistics', () => {
      registry.registerModel({
        name: 'stats-test',
        architecture: 'mlp',
        inputShape: [10],
        outputShape: [5],
        parameterCount: 1000,
        createdBy: 'agent-1',
      });

      const stats = registry.getStats();

      expect(stats).toBeDefined();
      expect(stats.totalModels).toBeGreaterThanOrEqual(1);
    });
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('createModelRegistry', () => {
    it('should create a ModelRegistry', () => {
      const registry = createModelRegistry();
      expect(registry).toBeInstanceOf(ModelRegistry);
    });

    it('should create with custom config', () => {
      const registry = createModelRegistry({ maxCheckpointsPerModel: 5 });
      expect(registry).toBeInstanceOf(ModelRegistry);
    });
  });

  describe('createPrivacyManager', () => {
    it('should create a PrivacyManager', () => {
      const manager = createPrivacyManager({
        epsilon: 5,
        delta: 1e-5,
        accountingMethod: 'rdp',
      });
      expect(manager).toBeInstanceOf(PrivacyManager);
    });
  });

  describe('createDevPrivacyBudget', () => {
    it('should create a development privacy budget', () => {
      const budget = createDevPrivacyBudget();

      expect(budget).toBeDefined();
      expect(budget.epsilon).toBeGreaterThan(0);
      expect(budget.delta).toBeGreaterThan(0);
    });
  });

  describe('createStrictPrivacyBudget', () => {
    it('should create a strict privacy budget', () => {
      const budget = createStrictPrivacyBudget();

      expect(budget).toBeDefined();
      expect(budget.epsilon).toBeGreaterThan(0);
      expect(budget.delta).toBeGreaterThan(0);
      // Strict budget should have lower epsilon
      const devBudget = createDevPrivacyBudget();
      expect(budget.epsilon).toBeLessThan(devBudget.epsilon);
    });
  });

  describe('createFLHub', () => {
    it('should create an FLHubCoordinator', () => {
      const hub = createFLHub();
      expect(hub).toBeInstanceOf(FLHubCoordinator);
    });

    it('should create with custom config', () => {
      const hub = createFLHub({
        maxConcurrentSessions: 5,
        defaultRoundTimeout: 30000,
      });
      expect(hub).toBeInstanceOf(FLHubCoordinator);
    });
  });

  describe('createFLHubWithRegistry', () => {
    it('should create a hub with existing registry', () => {
      const registry = new ModelRegistry();
      const hub = createFLHubWithRegistry(registry);

      expect(hub).toBeInstanceOf(FLHubCoordinator);
    });
  });
});

// =============================================================================
// PrivacyManager Tests
// =============================================================================

describe('PrivacyManager', () => {
  let manager: PrivacyManager;

  beforeEach(() => {
    manager = new PrivacyManager({
      epsilon: 5,
      delta: 1e-5,
      accountingMethod: 'rdp',
    });
  });

  describe('construction', () => {
    it('should create with budget', () => {
      expect(manager).toBeInstanceOf(PrivacyManager);
    });
  });

  describe('getRemainingBudget', () => {
    it('should return remaining epsilon budget', () => {
      const remaining = manager.getRemainingBudget();

      expect(remaining).toBe(5);
    });
  });

  describe('hasBudgetForRound', () => {
    it('should have sufficient budget initially', () => {
      const sufficient = manager.hasBudgetForRound(1);
      expect(sufficient).toBe(true);
    });

    it('should return false for excessive budget request', () => {
      const sufficient = manager.hasBudgetForRound(100);
      expect(sufficient).toBe(false);
    });
  });

  describe('calculatePerRoundBudget', () => {
    it('should calculate per-round budget', () => {
      const perRound = manager.calculatePerRoundBudget(10);

      expect(perRound).toBeGreaterThan(0);
      expect(perRound).toBeLessThan(5);
    });

    it('should reserve fraction of budget', () => {
      const perRound = manager.calculatePerRoundBudget(10, 0.2);

      // Should be (5 * 0.8) / 10 = 0.4
      expect(perRound).toBeCloseTo(0.4, 1);
    });
  });

  describe('getState', () => {
    it('should return privacy state', () => {
      const state = manager.getState();

      expect(state).toBeDefined();
      expect(state.totalBudget).toBeDefined();
      expect(state.remainingEpsilon).toBe(5);
      expect(state.usedEpsilon).toBe(0);
    });
  });

  describe('allocateRoundBudget', () => {
    it('should allocate budget for a round', () => {
      const allocation = manager.allocateRoundBudget(1, 0.5);

      expect(allocation).toBeDefined();
      expect(allocation.round).toBe(1);
      expect(allocation.epsilonUsed).toBe(0.5);
      expect(allocation.noiseScale).toBeGreaterThan(0);

      // Check budget was deducted
      expect(manager.getRemainingBudget()).toBe(4.5);
    });

    it('should throw when exceeding budget', () => {
      expect(() => manager.allocateRoundBudget(1, 100)).toThrow();
    });
  });

  describe('clipGradients', () => {
    it('should not clip gradients within bound', () => {
      const gradients = [0.1, 0.2, 0.3];
      const clipped = manager.clipGradients(gradients, 1.0);

      expect(clipped).toEqual(gradients);
    });

    it('should clip gradients exceeding bound', () => {
      const gradients = [3, 4]; // norm = 5
      const clipped = manager.clipGradients(gradients, 2.5);

      // Should scale by 2.5/5 = 0.5
      expect(clipped[0]).toBeCloseTo(1.5, 5);
      expect(clipped[1]).toBeCloseTo(2, 5);
    });
  });
});

// =============================================================================
// FLHubCoordinator Tests
// =============================================================================

describe('FLHubCoordinator', () => {
  let hub: FLHubCoordinator;
  let registry: ModelRegistry;
  let modelId: ModelId;

  beforeEach(() => {
    registry = new ModelRegistry();
    hub = new FLHubCoordinator({}, registry);

    // Register a model for testing
    const model = registry.registerModel({
      name: 'test-model',
      architecture: 'mlp',
      inputShape: [100],
      outputShape: [10],
      parameterCount: 10000,
      createdBy: 'coordinator',
    });
    modelId = model.modelId;
  });

  describe('construction', () => {
    it('should create with default config', () => {
      const h = new FLHubCoordinator();
      expect(h).toBeInstanceOf(FLHubCoordinator);
    });

    it('should create with custom config and registry', () => {
      const h = new FLHubCoordinator({ maxConcurrentSessions: 3 }, registry);
      expect(h).toBeInstanceOf(FLHubCoordinator);
    });
  });

  describe('getRegistry', () => {
    it('should return the model registry', () => {
      const reg = hub.getRegistry();
      expect(reg).toBeInstanceOf(ModelRegistry);
    });
  });

  describe('createSession', () => {
    it('should create a training session', () => {
      const session = hub.createSession(
        {
          name: 'Test Session',
          modelId,
          totalRounds: 10,
          minParticipants: 3,
          maxParticipants: 10,
          roundTimeout: 60000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      expect(session).toBeDefined();
      expect(session.sessionId).toBeDefined();
      expect(session.config.name).toBe('Test Session');
      expect(session.status).toBe('created');
    });

    it('should create session with privacy budget', () => {
      const session = hub.createSession(
        {
          name: 'Private Session',
          modelId,
          totalRounds: 10,
          minParticipants: 5,
          maxParticipants: 20,
          roundTimeout: 60000,
          aggregationMethod: AggregationMethod.FedAvg,
          privacyBudget: {
            epsilon: 5,
            delta: 1e-5,
            accountingMethod: 'rdp',
          },
        },
        'coordinator'
      );

      expect(session).toBeDefined();
      expect(session.sessionId).toBeDefined();
    });
  });

  describe('getSession', () => {
    it('should retrieve session by ID', () => {
      const created = hub.createSession(
        {
          name: 'Retrieval Test',
          modelId,
          totalRounds: 5,
          minParticipants: 2,
          maxParticipants: 5,
          roundTimeout: 30000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      const retrieved = hub.getSession(created.sessionId);
      expect(retrieved).toEqual(created);
    });

    it('should return null for unknown session', () => {
      const retrieved = hub.getSession('unknown-id');
      expect(retrieved).toBeNull();
    });
  });

  describe('listSessions', () => {
    it('should list all sessions', () => {
      hub.createSession(
        {
          name: 'Session 1',
          modelId,
          totalRounds: 5,
          minParticipants: 2,
          maxParticipants: 5,
          roundTimeout: 30000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      const sessions = hub.listSessions();
      expect(sessions.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('session lifecycle', () => {
    it('should start recruiting', () => {
      const session = hub.createSession(
        {
          name: 'Lifecycle Test',
          modelId,
          totalRounds: 5,
          minParticipants: 2,
          maxParticipants: 5,
          roundTimeout: 30000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      hub.startRecruiting(session.sessionId);

      const updated = hub.getSession(session.sessionId);
      expect(updated?.status).toBe('recruiting');
    });

    it('should allow participants to join', () => {
      const session = hub.createSession(
        {
          name: 'Join Test',
          modelId,
          totalRounds: 5,
          minParticipants: 2,
          maxParticipants: 5,
          roundTimeout: 30000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      hub.startRecruiting(session.sessionId);
      const result = hub.joinSession(session.sessionId, 'participant-1', 0.85);

      expect(result).toBe(true);
    });

    it('should reject join for unknown session', () => {
      const result = hub.joinSession('unknown-session', 'participant-1', 0.85);
      expect(result).toBe(false);
    });

    it('should cancel session', () => {
      const session = hub.createSession(
        {
          name: 'Cancel Test',
          modelId,
          totalRounds: 5,
          minParticipants: 2,
          maxParticipants: 5,
          roundTimeout: 30000,
          aggregationMethod: AggregationMethod.FedAvg,
        },
        'coordinator'
      );

      const result = hub.cancelSession(session.sessionId, 'Testing cancellation');

      expect(result).toBe(true);
      const updated = hub.getSession(session.sessionId);
      expect(updated?.status).toBe('cancelled');
    });
  });

  describe('getStats', () => {
    it('should return hub statistics', () => {
      const stats = hub.getStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalSessions).toBe('number');
      expect(typeof stats.activeSessions).toBe('number');
    });
  });
});
