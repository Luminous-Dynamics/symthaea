// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Classification Router Tests
 *
 * Tests for E/N/M → storage tier routing logic.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  StorageRouter,
  createRouter,
  DEFAULT_ROUTER_CONFIG,
  type RouterConfig,
} from '../../src/storage/router.js';
import { MutabilityMode, AccessControlMode } from '../../src/storage/types.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../src/epistemic/types.js';
import type { EpistemicClassification } from '../../src/epistemic/types.js';

describe('StorageRouter', () => {
  let router: StorageRouter;

  beforeEach(() => {
    router = new StorageRouter();
  });

  describe('route()', () => {
    describe('M-level → Backend mapping', () => {
      it('routes M0 to memory backend', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M0_Ephemeral,
        };

        const tier = router.route(classification);

        expect(tier.backend).toBe('memory');
        expect(tier.ttl).toBeDefined();
        expect(tier.replication).toBe(0);
      });

      it('routes M1 to local backend', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        };

        const tier = router.route(classification);

        expect(tier.backend).toBe('local');
        expect(tier.ttl).toBeDefined();
        expect(tier.replication).toBe(1);
      });

      it('routes M2 to DHT backend', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };

        const tier = router.route(classification);

        expect(tier.backend).toBe('dht');
        expect(tier.ttl).toBeUndefined();
      });

      it('routes M3 to IPFS by default (enableIpfs: true)', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        };

        const tier = router.route(classification);

        // IPFS is content-addressed and immutable, ideal for M3
        expect(tier.backend).toBe('ipfs');
        expect(tier.ttl).toBeUndefined();
      });

      it('routes M3 to DHT when IPFS disabled', () => {
        const noIpfsRouter = createRouter({ enableIpfs: false });
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        };

        const tier = noIpfsRouter.route(classification);

        expect(tier.backend).toBe('dht');
        expect(tier.ttl).toBeUndefined();
      });

      it('routes M3 to Filecoin when enabled', () => {
        const filecoinRouter = createRouter({ enableFilecoin: true });
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        };

        const tier = filecoinRouter.route(classification);

        expect(tier.backend).toBe('filecoin');
      });
    });

    describe('E-level → Mutability mapping', () => {
      it('E0-E1 uses MUTABLE_CRDT', () => {
        const e0: EpistemicClassification = {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        };
        const e1: EpistemicClassification = {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        };

        expect(router.route(e0).mutability).toBe(MutabilityMode.MUTABLE_CRDT);
        expect(router.route(e1).mutability).toBe(MutabilityMode.MUTABLE_CRDT);
      });

      it('E2 uses APPEND_ONLY', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N1_Communal,
          materiality: MaterialityLevel.M2_Persistent,
        };

        expect(router.route(classification).mutability).toBe(MutabilityMode.APPEND_ONLY);
      });

      it('E3-E4 uses IMMUTABLE', () => {
        const e3: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };
        const e4: EpistemicClassification = {
          empirical: EmpiricalLevel.E4_Consensus,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        };

        expect(router.route(e3).mutability).toBe(MutabilityMode.IMMUTABLE);
        expect(router.route(e4).mutability).toBe(MutabilityMode.IMMUTABLE);
      });
    });

    describe('N-level → Access Control mapping', () => {
      it('N0 uses OWNER access control', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        };

        expect(router.route(classification).accessControl).toBe(AccessControlMode.OWNER);
      });

      it('N1 uses CAPBAC access control', () => {
        const classification: EpistemicClassification = {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N1_Communal,
          materiality: MaterialityLevel.M1_Temporal,
        };

        expect(router.route(classification).accessControl).toBe(AccessControlMode.CAPBAC);
      });

      it('N2-N3 uses PUBLIC access control', () => {
        const n2: EpistemicClassification = {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };
        const n3: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        };

        expect(router.route(n2).accessControl).toBe(AccessControlMode.PUBLIC);
        expect(router.route(n3).accessControl).toBe(AccessControlMode.PUBLIC);
      });
    });

    describe('Encryption requirements', () => {
      it('N0-N1 requires encryption', () => {
        const n0: EpistemicClassification = {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        };
        const n1: EpistemicClassification = {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N1_Communal,
          materiality: MaterialityLevel.M1_Temporal,
        };

        expect(router.route(n0).encrypted).toBe(true);
        expect(router.route(n1).encrypted).toBe(true);
      });

      it('N2-N3 does not require encryption', () => {
        const n2: EpistemicClassification = {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };

        expect(router.route(n2).encrypted).toBe(false);
      });
    });

    describe('Content addressing', () => {
      it('E0-E2 is not content-addressed', () => {
        const e2: EpistemicClassification = {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };

        expect(router.route(e2).contentAddressed).toBe(false);
      });

      it('E3-E4 is content-addressed', () => {
        const e3: EpistemicClassification = {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        };

        expect(router.route(e3).contentAddressed).toBe(true);
      });
    });
  });

  describe('getReplication()', () => {
    it('M0 has 0 replication', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      };

      expect(router.getReplication(classification, 100)).toBe(0);
    });

    it('M1 has 1 replication', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      expect(router.getReplication(classification, 100)).toBe(1);
    });

    it('M2 N0 has 3 replication', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.getReplication(classification, 100)).toBe(3);
    });

    it('M2 N1 has 5 replication', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.getReplication(classification, 100)).toBe(5);
    });

    it('M3 uses logarithmic replication', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M3_Immutable,
      };

      // log2(100) ≈ 7, but min is 5
      expect(router.getReplication(classification, 100)).toBe(7);

      // log2(1000000) ≈ 20
      expect(router.getReplication(classification, 1000000)).toBe(20);

      // Small network uses minimum
      expect(router.getReplication(classification, 10)).toBe(5);
    });
  });

  describe('validateTransition() - INV-2 Monotonic Classification', () => {
    it('allows upgrading E-level', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      const result = router.validateTransition(from, to);

      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('rejects downgrading E-level', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      };

      const result = router.validateTransition(from, to);

      expect(result.valid).toBe(false);
      expect(result.error).toContain('E-level');
    });

    it('rejects downgrading N-level', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const result = router.validateTransition(from, to);

      expect(result.valid).toBe(false);
      expect(result.error).toContain('N-level');
    });

    it('rejects downgrading M-level', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M1_Temporal,
      };

      const result = router.validateTransition(from, to);

      expect(result.valid).toBe(false);
      expect(result.error).toContain('M-level');
    });

    it('allows staying at same level', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const result = router.validateTransition(classification, classification);

      expect(result.valid).toBe(true);
    });
  });

  describe('requiresMigration()', () => {
    it('requires migration when backend changes', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.requiresMigration(from, to)).toBe(true);
    });

    it('requires migration when encryption changes', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M2_Persistent,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.requiresMigration(from, to)).toBe(true);
    });

    it('requires migration when mutability changes', () => {
      const from: EpistemicClassification = {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };
      const to: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.requiresMigration(from, to)).toBe(true);
    });

    it('does not require migration for same tier', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      expect(router.requiresMigration(classification, classification)).toBe(false);
    });
  });

  describe('getBackends()', () => {
    it('returns primary and additional backends', () => {
      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      };

      const backends = router.getBackends(classification);

      expect(backends).toContain('dht');
      // Should also include IPFS for E3+ N2+ content-addressed data
      expect(backends.length).toBeGreaterThan(1);
    });
  });

  describe('createRouter()', () => {
    it('creates router with custom config', () => {
      const customRouter = createRouter({
        m0DefaultTtl: 30000,
        m1DefaultTtl: 86400000,
      });

      const tier = customRouter.route({
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      });

      expect(tier.ttl).toBe(30000);
    });

    it('creates router with custom network size provider', () => {
      const customRouter = createRouter({}, () => 1000000);

      const classification: EpistemicClassification = {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M3_Immutable,
      };

      // log2(1000000) ≈ 20
      expect(customRouter.getReplication(classification, 1000000)).toBe(20);
    });
  });
});
