/**
 * MATL Module Benchmarks
 *
 * Performance tests for the Mycelix Adaptive Trust Layer.
 * Uses Vitest's bench API for accurate measurements.
 */

import { bench, describe } from 'vitest';
import {
  createPoGQ,
  compositeScore,
  isByzantine,
  createReputation,
  reputationValue,
  recordPositive,
  recordNegative,
  calculateComposite,
  isTrustworthy,
  createAdaptiveThreshold,
  observe,
  getThreshold,
  isAnomalous,
} from '../src/matl/index.js';

describe('MATL: PoGQ Operations', () => {
  bench('createPoGQ', () => {
    createPoGQ(0.85, 0.9, 0.1);
  });

  bench('compositeScore', () => {
    const pogq = createPoGQ(0.85, 0.9, 0.1);
    compositeScore(pogq, 0.75);
  });

  bench('isByzantine', () => {
    const pogq = createPoGQ(0.85, 0.9, 0.1);
    isByzantine(pogq);
  });

  bench('isByzantine (batch 1000)', () => {
    const pogq = createPoGQ(0.85, 0.9, 0.1);
    for (let i = 0; i < 1000; i++) {
      isByzantine(pogq);
    }
  });
});

describe('MATL: Reputation Operations', () => {
  bench('createReputation', () => {
    createReputation('agent_' + Math.random().toString(36).slice(2));
  });

  bench('reputationValue', () => {
    const rep = createReputation('agent_001');
    reputationValue(rep);
  });

  bench('recordPositive', () => {
    let rep = createReputation('agent_001');
    rep = recordPositive(rep);
  });

  bench('recordNegative', () => {
    let rep = createReputation('agent_001');
    rep = recordNegative(rep);
  });

  bench('100 positive interactions', () => {
    let rep = createReputation('agent_001');
    for (let i = 0; i < 100; i++) {
      rep = recordPositive(rep);
    }
  });

  bench('mixed interactions (50/50)', () => {
    let rep = createReputation('agent_001');
    for (let i = 0; i < 100; i++) {
      rep = i % 2 === 0 ? recordPositive(rep) : recordNegative(rep);
    }
  });
});

describe('MATL: Composite Score', () => {
  bench('calculateComposite', () => {
    const pogq = createPoGQ(0.9, 0.8, 0.1);
    let rep = createReputation('agent_001');
    rep = recordPositive(rep);
    rep = recordPositive(rep);
    calculateComposite(pogq, rep);
  });

  bench('isTrustworthy', () => {
    const pogq = createPoGQ(0.9, 0.8, 0.1);
    let rep = createReputation('agent_001');
    rep = recordPositive(rep);
    const composite = calculateComposite(pogq, rep);
    isTrustworthy(composite);
  });

  bench('full trust verification pipeline', () => {
    // Complete workflow: create measurements, calculate composite, check trust
    const pogq = createPoGQ(0.9, 0.8, 0.1);
    let rep = createReputation('agent_' + Math.random());
    for (let i = 0; i < 10; i++) {
      rep = recordPositive(rep);
    }
    const composite = calculateComposite(pogq, rep);
    isTrustworthy(composite);
  });
});

describe('MATL: Adaptive Threshold', () => {
  bench('createAdaptiveThreshold', () => {
    createAdaptiveThreshold('node_' + Math.random().toString(36).slice(2));
  });

  bench('observe (single)', () => {
    let at = createAdaptiveThreshold('node_001');
    at = observe(at, 0.8);
  });

  bench('observe (100 observations)', () => {
    let at = createAdaptiveThreshold('node_001', 200);
    for (let i = 0; i < 100; i++) {
      at = observe(at, 0.7 + Math.random() * 0.2);
    }
  });

  bench('getThreshold (after 100 observations)', () => {
    let at = createAdaptiveThreshold('node_001', 100);
    for (let i = 0; i < 100; i++) {
      at = observe(at, 0.8);
    }
    getThreshold(at);
  });

  bench('isAnomalous', () => {
    let at = createAdaptiveThreshold('node_001', 100);
    for (let i = 0; i < 50; i++) {
      at = observe(at, 0.8);
    }
    isAnomalous(at, 0.3);
  });

  bench('anomaly detection pipeline (100 checks)', () => {
    let at = createAdaptiveThreshold('node_001', 100);
    for (let i = 0; i < 50; i++) {
      at = observe(at, 0.8);
    }
    for (let i = 0; i < 100; i++) {
      isAnomalous(at, Math.random());
    }
  });
});

describe('MATL: Realistic Scenarios', () => {
  bench('marketplace transaction verification', () => {
    // Simulate verifying a seller before transaction
    const sellerPogq = createPoGQ(0.92, 0.88, 0.08);
    let sellerRep = createReputation('seller_001');
    for (let i = 0; i < 47; i++) {
      sellerRep = recordPositive(sellerRep);
    }
    for (let i = 0; i < 3; i++) {
      sellerRep = recordNegative(sellerRep);
    }
    const composite = calculateComposite(sellerPogq, sellerRep);
    isTrustworthy(composite, 0.6);
  });

  bench('FL participant validation (10 participants)', () => {
    // Validate 10 FL participants before aggregation
    const participants = [];
    for (let i = 0; i < 10; i++) {
      const pogq = createPoGQ(0.85 + Math.random() * 0.1, 0.8 + Math.random() * 0.1, Math.random() * 0.2);
      let rep = createReputation(`participant_${i}`);
      for (let j = 0; j < 20; j++) {
        rep = Math.random() > 0.2 ? recordPositive(rep) : recordNegative(rep);
      }
      const composite = calculateComposite(pogq, rep);
      participants.push({ composite, trusted: isTrustworthy(composite) });
    }
  });

  bench('Byzantine detection (1000 nodes)', () => {
    // Check 1000 nodes for Byzantine behavior
    const results = [];
    for (let i = 0; i < 1000; i++) {
      // 90% honest, 10% Byzantine
      const isActuallyByzantine = Math.random() < 0.1;
      const pogq = isActuallyByzantine
        ? createPoGQ(0.1 + Math.random() * 0.2, 0.2 + Math.random() * 0.2, 0.7 + Math.random() * 0.3)
        : createPoGQ(0.85 + Math.random() * 0.1, 0.8 + Math.random() * 0.1, Math.random() * 0.2);
      results.push(isByzantine(pogq));
    }
  });
});
