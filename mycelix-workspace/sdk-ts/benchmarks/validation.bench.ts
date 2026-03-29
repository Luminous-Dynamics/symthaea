// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Validation Module Benchmarks
 *
 * Performance tests for SDK validation utilities:
 * - Type guards
 * - Schema validation
 * - Input validation chains
 */

import { bench, describe } from 'vitest';
import { Validator, validate, assert, assertDefined } from '../src/errors.js';
import {
  isEpistemicClaim,
  isEpistemicClassification,
  isReputationScore,
  isCompositeScore,
  isProofOfGradientQuality,
  isGradientUpdate,
  isAggregatedGradient,
  isFLConfig,
  isFLRound,
  isHealthCheckResult,
  isMycelixConfig,
  isBridgeMessage,
  isSecureMessage,
  isConnectionState,
  isTrustCheckResult,
} from '../src/utils/validation.js';
import { createClaim, EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../src/epistemic/index.js';
import { createPoGQ, createReputation, calculateComposite } from '../src/matl/index.js';
import { createReputationQuery } from '../src/bridge/index.js';

// ============================================================================
// Type Guard Benchmarks
// ============================================================================

describe('Validation: Epistemic Type Guards', () => {
  const validClaim = createClaim({
    subject: 'Test claim',
    classification: {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N1_Local,
      materiality: MaterialityLevel.M1_Temporal,
    },
    confidence: 0.8,
  });

  const invalidClaim = { subject: 'Invalid', foo: 'bar' };

  bench('isEpistemicClaim() - valid', () => {
    isEpistemicClaim(validClaim);
  });

  bench('isEpistemicClaim() - invalid object', () => {
    isEpistemicClaim(invalidClaim);
  });

  bench('isEpistemicClaim() - null', () => {
    isEpistemicClaim(null);
  });

  bench('isEpistemicClaim() - undefined', () => {
    isEpistemicClaim(undefined);
  });

  bench('isEpistemicClassification() - valid', () => {
    isEpistemicClassification({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N1_Local,
      materiality: MaterialityLevel.M1_Temporal,
    });
  });

  bench('isEpistemicClassification() - invalid', () => {
    isEpistemicClassification({ empirical: 10 });
  });

  bench('isEpistemicClaim() (100 checks)', () => {
    for (let i = 0; i < 100; i++) {
      isEpistemicClaim(i % 2 === 0 ? validClaim : invalidClaim);
    }
  });
});

describe('Validation: MATL Type Guards', () => {
  const validPogq = createPoGQ(0.85, 0.9, 0.1);
  const validRep = createReputation('agent-1');
  const validComposite = calculateComposite(validPogq, validRep);

  bench('isProofOfGradientQuality() - valid', () => {
    isProofOfGradientQuality(validPogq);
  });

  bench('isProofOfGradientQuality() - invalid', () => {
    isProofOfGradientQuality({ quality: 0.5 });
  });

  bench('isReputationScore() - valid', () => {
    isReputationScore(validRep);
  });

  bench('isReputationScore() - invalid', () => {
    isReputationScore({ score: 0.8 });
  });

  bench('isCompositeScore() - valid', () => {
    isCompositeScore(validComposite);
  });

  bench('isCompositeScore() - invalid', () => {
    isCompositeScore({ value: 0.7 });
  });

  bench('MATL type guards (100 mixed checks)', () => {
    for (let i = 0; i < 100; i++) {
      if (i % 3 === 0) isProofOfGradientQuality(validPogq);
      else if (i % 3 === 1) isReputationScore(validRep);
      else isCompositeScore(validComposite);
    }
  });
});

describe('Validation: FL Type Guards', () => {
  const validGradientUpdate = {
    participantId: 'p-1',
    modelVersion: 1,
    gradients: new Float32Array([0.1, 0.2, 0.3]),
    metadata: {
      batchSize: 32,
      loss: 0.5,
      timestamp: Date.now(),
    },
  };

  const validAggregated = {
    gradients: new Float32Array([0.1, 0.2, 0.3]),
    contributorCount: 10,
    aggregationMethod: 'fedAvg',
    metadata: {
      roundId: 1,
      timestamp: Date.now(),
    },
  };

  bench('isGradientUpdate() - valid', () => {
    isGradientUpdate(validGradientUpdate);
  });

  bench('isGradientUpdate() - invalid', () => {
    isGradientUpdate({ participant: 'p-1' });
  });

  bench('isAggregatedGradient() - valid', () => {
    isAggregatedGradient(validAggregated);
  });

  bench('isAggregatedGradient() - invalid', () => {
    isAggregatedGradient({ gradients: [] });
  });

  bench('isFLConfig() - valid', () => {
    isFLConfig({
      minParticipants: 3,
      roundTimeoutMs: 30000,
      aggregationMethod: 'fedAvg',
    });
  });

  bench('isFLRound() - valid', () => {
    isFLRound({
      id: 'round-1',
      status: 'active',
      startTime: Date.now(),
      participants: ['p1', 'p2'],
      updates: [],
    });
  });
});

describe('Validation: Bridge Type Guards', () => {
  const validMessage = createReputationQuery('test-happ', 'agent-1');

  bench('isBridgeMessage() - valid', () => {
    isBridgeMessage(validMessage);
  });

  bench('isBridgeMessage() - invalid', () => {
    isBridgeMessage({ type: 'unknown' });
  });

  bench('isBridgeMessage() - null', () => {
    isBridgeMessage(null);
  });

  bench('isSecureMessage() - valid', () => {
    isSecureMessage({
      id: 'msg-1',
      payload: 'encrypted-data',
      signature: 'sig-123',
      timestamp: Date.now(),
    });
  });

  bench('isSecureMessage() - invalid', () => {
    isSecureMessage({ id: 'msg-1' });
  });
});

describe('Validation: Utility Type Guards', () => {
  bench('isHealthCheckResult() - valid', () => {
    isHealthCheckResult({
      name: 'test',
      status: 'healthy',
      latencyMs: 10,
    });
  });

  bench('isConnectionState() - valid', () => {
    isConnectionState('connected');
  });

  bench('isConnectionState() - invalid', () => {
    isConnectionState('unknown-state');
  });

  bench('isTrustCheckResult() - valid', () => {
    isTrustCheckResult({
      agentId: 'agent-1',
      trusted: true,
      score: 0.85,
      reason: 'high_reputation',
    });
  });

  bench('isMycelixConfig() - valid', () => {
    isMycelixConfig({
      client: { appId: 'test' },
      matl: { byzantineThreshold: 0.34 },
      fl: { minParticipants: 3 },
    });
  });
});

// ============================================================================
// Validator Chain Benchmarks
// ============================================================================

describe('Validation: Validator Chain', () => {
  bench('validate() chain - simple', () => {
    validate()
      .notEmpty('name', 'John')
      .range('age', 25, 0, 150)
      .throwIfInvalid();
  });

  bench('validate() chain - multiple fields', () => {
    validate()
      .notEmpty('name', 'John Doe')
      .notEmpty('email', 'john@example.com')
      .range('age', 30, 0, 150)
      .range('score', 0.85, 0, 1)
      .custom('status', 'active', (v) => ['active', 'inactive'].includes(v as string))
      .throwIfInvalid();
  });

  bench('validate() chain - failing early', () => {
    try {
      validate()
        .notEmpty('name', '')
        .range('age', 25, 0, 150)
        .throwIfInvalid();
    } catch (e) {
      // Expected
    }
  });

  bench('validate() with custom validators', () => {
    validate()
      .custom('email', 'test@example.com', (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v as string))
      .custom('phone', '+1234567890', (v) => /^\+?\d{10,15}$/.test(v as string))
      .custom('url', 'https://example.com', (v) => {
        try {
          new URL(v as string);
          return true;
        } catch {
          return false;
        }
      })
      .throwIfInvalid();
  });

  bench('Validator class - 10 fields', () => {
    const validator = new Validator();
    for (let i = 0; i < 10; i++) {
      validator.notEmpty(`field${i}`, `value${i}`);
    }
    validator.throwIfInvalid();
  });

  bench('validate() pattern matching', () => {
    validate()
      .pattern('id', 'abc-123-def', /^[a-z]+-\d+-[a-z]+$/)
      .pattern('hex', 'ff00ff', /^[0-9a-f]+$/i)
      .pattern('uuid', '550e8400-e29b-41d4-a716-446655440000', /^[0-9a-f-]{36}$/i)
      .throwIfInvalid();
  });
});

// ============================================================================
// Assert Benchmarks
// ============================================================================

describe('Validation: Assertions', () => {
  bench('assert() - passing', () => {
    assert(true, 'Should pass');
  });

  bench('assert() - failing', () => {
    try {
      assert(false, 'Should fail');
    } catch (e) {
      // Expected
    }
  });

  bench('assertDefined() - defined value', () => {
    assertDefined('value', 'test');
  });

  bench('assertDefined() - undefined', () => {
    try {
      assertDefined(undefined, 'test');
    } catch (e) {
      // Expected
    }
  });

  bench('assertDefined() (100 checks)', () => {
    for (let i = 0; i < 100; i++) {
      try {
        assertDefined(i % 10 === 0 ? undefined : i, 'value');
      } catch (e) {
        // Expected for some
      }
    }
  });
});

// ============================================================================
// Realistic Scenarios
// ============================================================================

describe('Validation: Realistic Scenarios', () => {
  bench('API input validation', () => {
    const input = {
      userId: 'user-12345',
      action: 'transfer',
      amount: 100.50,
      recipient: 'user-67890',
      memo: 'Payment for services',
    };

    validate()
      .notEmpty('userId', input.userId)
      .pattern('userId', input.userId, /^user-\d+$/)
      .notEmpty('action', input.action)
      .custom('action', input.action, (v) => ['transfer', 'deposit', 'withdraw'].includes(v as string))
      .range('amount', input.amount, 0.01, 10000)
      .notEmpty('recipient', input.recipient)
      .throwIfInvalid();
  });

  bench('FL participant validation', () => {
    const participant = {
      id: 'participant-1',
      modelVersion: 5,
      gradients: new Float32Array(1000),
      trustScore: 0.87,
      metadata: { batchSize: 32, loss: 0.15 },
    };

    validate()
      .notEmpty('id', participant.id)
      .range('modelVersion', participant.modelVersion, 1, 1000)
      .custom('gradients', participant.gradients, (v) => v instanceof Float32Array)
      .range('trustScore', participant.trustScore, 0, 1)
      .throwIfInvalid();

    isGradientUpdate(participant);
  });

  bench('claim verification pipeline', () => {
    const claim = createClaim({
      subject: 'Product X is certified organic',
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Local,
        materiality: MaterialityLevel.M2_Persistent,
      },
      confidence: 0.92,
    });

    // Validate claim structure
    if (isEpistemicClaim(claim)) {
      // Validate classification
      isEpistemicClassification(claim.classification);

      // Validate confidence
      validate()
        .range('confidence', claim.confidence, 0, 1)
        .throwIfInvalid();
    }
  });

  bench('bridge message validation pipeline', () => {
    const messages = [
      createReputationQuery('happ-1', 'agent-1'),
      createReputationQuery('happ-2', 'agent-2'),
      createReputationQuery('happ-3', 'agent-3'),
    ];

    for (const msg of messages) {
      if (isBridgeMessage(msg)) {
        validate()
          .notEmpty('sourceHapp', msg.sourceHapp)
          .custom('timestamp', msg.timestamp, (v) => (v as number) <= Date.now())
          .throwIfInvalid();
      }
    }
  });

  bench('composite trust validation (10 checks)', () => {
    const pogq = createPoGQ(0.85, 0.9, 0.1);
    const rep = createReputation('agent-1');
    const composite = calculateComposite(pogq, rep);

    for (let i = 0; i < 10; i++) {
      if (isProofOfGradientQuality(pogq)) {
        validate()
          .range('quality', pogq.quality, 0, 1)
          .range('consistency', pogq.consistency, 0, 1)
          .range('entropy', pogq.entropy, 0, 1)
          .throwIfInvalid();
      }

      if (isCompositeScore(composite)) {
        validate()
          .range('score', composite.score, 0, 1)
          .throwIfInvalid();
      }
    }
  });
});
