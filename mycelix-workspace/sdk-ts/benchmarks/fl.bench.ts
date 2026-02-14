/**
 * Federated Learning Module Benchmarks
 *
 * Performance tests for FL aggregation algorithms.
 */

import { bench, describe } from 'vitest';
import {
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  trustWeightedAggregation,
  FLCoordinator,
  serializeGradients,
  deserializeGradients,
  type GradientUpdate,
  type Participant,
} from '../src/fl/index.js';
import { createReputation, recordPositive } from '../src/matl/index.js';

// Helper to create realistic gradient updates
function createGradientUpdate(
  id: string,
  modelDim: number,
  isHonest = true
): GradientUpdate {
  const gradients = new Float32Array(modelDim);
  for (let i = 0; i < modelDim; i++) {
    if (isHonest) {
      // Honest: small gradients centered around 0
      gradients[i] = (Math.random() - 0.5) * 0.1;
    } else {
      // Byzantine: large random gradients
      gradients[i] = (Math.random() - 0.5) * 10;
    }
  }
  return {
    participantId: id,
    modelVersion: 1,
    gradients,
    batchSize: 32,
    loss: isHonest ? 0.3 + Math.random() * 0.2 : Math.random(),
  };
}

describe('FL: FedAvg Aggregation', () => {
  bench('fedAvg (10 participants, 100 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 100));
    }
    fedAvg(updates);
  });

  bench('fedAvg (100 participants, 100 params)', () => {
    const updates = [];
    for (let i = 0; i < 100; i++) {
      updates.push(createGradientUpdate(`p${i}`, 100));
    }
    fedAvg(updates);
  });

  bench('fedAvg (10 participants, 10000 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 10000));
    }
    fedAvg(updates);
  });

  bench('fedAvg (10 participants, 100000 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 100000));
    }
    fedAvg(updates);
  });
});

describe('FL: Byzantine-Resistant Aggregation', () => {
  bench('trimmedMean (10 participants, 1000 params, 20% trim)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i >= 8)); // 2 Byzantine
    }
    trimmedMean(updates, 0.2);
  });

  bench('coordinateMedian (10 participants, 1000 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i >= 8));
    }
    coordinateMedian(updates);
  });

  bench('krum (10 participants, 1000 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i >= 8));
    }
    krum(updates);
  });

  bench('krum (20 participants, 1000 params, 20% Byzantine)', () => {
    const updates = [];
    for (let i = 0; i < 20; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i >= 16)); // 4 Byzantine
    }
    krum(updates);
  });
});

describe('FL: Trust-Weighted Aggregation', () => {
  function createParticipantMap(count: number): Record<string, Participant> {
    const participants: Record<string, Participant> = {};
    for (let i = 0; i < count; i++) {
      let rep = createReputation(`p${i}`);
      // Vary reputation - some trusted, some not
      const positives = i < count * 0.8 ? 20 : 2;
      const negatives = i < count * 0.8 ? 2 : 10;
      for (let j = 0; j < positives; j++) {
        rep = recordPositive(rep);
      }
      participants[`p${i}`] = { id: `p${i}`, reputation: rep };
    }
    return participants;
  }

  bench('trustWeightedAggregation (10 participants, 1000 params)', () => {
    const updates = [];
    for (let i = 0; i < 10; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i < 8));
    }
    const participants = createParticipantMap(10);
    trustWeightedAggregation(updates, participants, 0.3);
  });

  bench('trustWeightedAggregation (50 participants, 1000 params)', () => {
    const updates = [];
    for (let i = 0; i < 50; i++) {
      updates.push(createGradientUpdate(`p${i}`, 1000, i < 40));
    }
    const participants = createParticipantMap(50);
    trustWeightedAggregation(updates, participants, 0.3);
  });
});

describe('FL: Coordinator Operations', () => {
  bench('register 100 participants', () => {
    const coordinator = new FLCoordinator();
    for (let i = 0; i < 100; i++) {
      coordinator.registerParticipant(`participant_${i}`);
    }
  });

  bench('full round (10 participants, 1000 params)', () => {
    const coordinator = new FLCoordinator({ minParticipants: 10 });
    for (let i = 0; i < 10; i++) {
      coordinator.registerParticipant(`p${i}`);
    }
    coordinator.startRound();
    for (let i = 0; i < 10; i++) {
      coordinator.submitUpdate(createGradientUpdate(`p${i}`, 1000));
    }
    coordinator.aggregateRound();
  });

  bench('5 consecutive rounds (10 participants)', () => {
    const coordinator = new FLCoordinator({ minParticipants: 10 });
    for (let i = 0; i < 10; i++) {
      coordinator.registerParticipant(`p${i}`);
    }
    for (let round = 0; round < 5; round++) {
      coordinator.startRound();
      for (let i = 0; i < 10; i++) {
        coordinator.submitUpdate(createGradientUpdate(`p${i}`, 1000));
      }
      coordinator.aggregateRound();
    }
  });
});

describe('FL: Serialization', () => {
  bench('serializeGradients (1000 params)', () => {
    const gradients = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) {
      gradients[i] = Math.random();
    }
    serializeGradients(gradients);
  });

  bench('serializeGradients (100000 params)', () => {
    const gradients = new Float32Array(100000);
    for (let i = 0; i < 100000; i++) {
      gradients[i] = Math.random();
    }
    serializeGradients(gradients);
  });

  bench('deserializeGradients (1000 params)', () => {
    const gradients = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) {
      gradients[i] = Math.random();
    }
    const serialized = serializeGradients(gradients);
    deserializeGradients(serialized);
  });

  bench('roundtrip serialize/deserialize (10000 params)', () => {
    const original = new Float32Array(10000);
    for (let i = 0; i < 10000; i++) {
      original[i] = Math.random();
    }
    const serialized = serializeGradients(original);
    deserializeGradients(serialized);
  });
});

describe('FL: Realistic Scenarios', () => {
  bench('healthcare FL round (5 hospitals, 50000 params)', () => {
    // Simulate healthcare FL scenario
    const coordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: 'trust_weighted',
      trustThreshold: 0.5,
    });

    // Register hospitals
    for (let i = 0; i < 5; i++) {
      coordinator.registerParticipant(`hospital_${i}`);
    }

    // One round
    coordinator.startRound();
    for (let i = 0; i < 5; i++) {
      const update = createGradientUpdate(`hospital_${i}`, 50000);
      update.batchSize = 1000 + i * 500; // Different dataset sizes
      coordinator.submitUpdate(update);
    }
    coordinator.aggregateRound();
  });

  bench('cross-device FL (100 devices, 1000 params, with dropouts)', () => {
    const coordinator = new FLCoordinator({
      minParticipants: 50, // Only need 50% participation
    });

    // Register all devices
    for (let i = 0; i < 100; i++) {
      coordinator.registerParticipant(`device_${i}`);
    }

    coordinator.startRound();

    // Only 60 devices respond (40% dropout)
    for (let i = 0; i < 60; i++) {
      coordinator.submitUpdate(createGradientUpdate(`device_${i}`, 1000));
    }

    coordinator.aggregateRound();
  });
});
