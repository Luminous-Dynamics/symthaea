/**
 * UESS Classification Router
 *
 * Maps E/N/M classification to storage decisions.
 * @see docs/architecture/uess/UESS-01-CORE.md §4
 */

import {
  type StorageTier,
  type StorageBackend,
  MutabilityMode,
  AccessControlMode,
} from './types.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../epistemic/types.js';

import type { EpistemicClassification } from '../epistemic/types.js';

// =============================================================================
// Router Configuration
// =============================================================================

/**
 * Configuration for the classification router
 */
export interface RouterConfig {
  /** Default TTL for M0 in ms (default: 1 hour) */
  m0DefaultTtl: number;

  /** Default TTL for M1 in ms (default: 7 days) */
  m1DefaultTtl: number;

  /** Default DHT replication factor */
  dhtDefaultReplication: number;

  /** Minimum replication for M3 */
  m3MinReplication: number;

  /** Maximum replication for M3 */
  m3MaxReplication: number;

  /** Enable IPFS for N2+ content-addressed data */
  enableIpfs: boolean;

  /** Enable Filecoin for M3 archival */
  enableFilecoin: boolean;
}

/**
 * Default router configuration
 */
export const DEFAULT_ROUTER_CONFIG: RouterConfig = {
  m0DefaultTtl: 60 * 60 * 1000, // 1 hour
  m1DefaultTtl: 7 * 24 * 60 * 60 * 1000, // 7 days
  dhtDefaultReplication: 10,
  m3MinReplication: 5,
  m3MaxReplication: 30,
  enableIpfs: true,
  enableFilecoin: false, // Requires Filecoin integration
};

// =============================================================================
// Classification Router Interface
// =============================================================================

/**
 * Classification Router - Maps E/N/M to storage decisions
 */
export interface ClassificationRouter {
  /**
   * Determine storage tier for classification
   */
  route(classification: EpistemicClassification): StorageTier;

  /**
   * Check if reclassification requires migration
   */
  requiresMigration(
    from: EpistemicClassification,
    to: EpistemicClassification
  ): boolean;

  /**
   * Get all backends that should hold this data
   */
  getBackends(classification: EpistemicClassification): StorageBackend[];

  /**
   * Calculate replication factor
   */
  getReplication(
    classification: EpistemicClassification,
    networkSize: number
  ): number;

  /**
   * Validate classification transition (must be monotonically increasing)
   */
  validateTransition(
    from: EpistemicClassification,
    to: EpistemicClassification
  ): { valid: boolean; error?: string };
}

// =============================================================================
// Storage Router Implementation
// =============================================================================

/**
 * Default implementation of the classification router
 */
export class StorageRouter implements ClassificationRouter {
  constructor(
    private readonly config: RouterConfig = DEFAULT_ROUTER_CONFIG,
    private readonly networkSizeProvider: () => number = () => 100
  ) {}

  /**
   * Route classification to storage tier
   */
  route(classification: EpistemicClassification): StorageTier {
    const { empirical, normative, materiality } = classification;

    // Determine backend from M-level
    const backend = this.getBackendForMateriality(materiality, normative);
    const additionalBackends = this.getAdditionalBackends(materiality, normative, empirical);

    // Determine mutability from E-level
    const mutability = this.getMutabilityForEmpirical(empirical);

    // Determine access control from N-level
    const accessControl = this.getAccessControlForNormative(normative);

    // Calculate replication
    const networkSize = this.networkSizeProvider();
    const replication = this.getReplication(classification, networkSize);

    // Encryption required for N0/N1 (INV-5)
    const encrypted = normative < NormativeLevel.N2_Network;

    // Content-addressed for E3+ (CID-based)
    const contentAddressed = empirical >= EmpiricalLevel.E3_Cryptographic;

    // TTL for M0/M1
    const ttl = this.getTTL(materiality);

    return {
      backend,
      additionalBackends,
      replication,
      mutability,
      accessControl,
      ttl,
      encrypted,
      contentAddressed,
    };
  }

  /**
   * Get primary backend based on M-level
   */
  private getBackendForMateriality(
    materiality: MaterialityLevel,
    _normative: NormativeLevel
  ): StorageBackend {
    switch (materiality) {
      case MaterialityLevel.M0_Ephemeral:
        return 'memory';

      case MaterialityLevel.M1_Temporal:
        return 'local';

      case MaterialityLevel.M2_Persistent:
        // DHT is primary for all M2 data
        return 'dht';

      case MaterialityLevel.M3_Immutable:
        // Filecoin > IPFS > DHT for immutable data
        // IPFS is content-addressed and immutable by nature, ideal for M3
        if (this.config.enableFilecoin) return 'filecoin';
        if (this.config.enableIpfs) return 'ipfs';
        return 'dht';

      default:
        return 'local';
    }
  }

  /**
   * Get additional backends for redundancy
   */
  private getAdditionalBackends(
    materiality: MaterialityLevel,
    normative: NormativeLevel,
    empirical: EmpiricalLevel
  ): StorageBackend[] {
    const backends: StorageBackend[] = [];

    if (materiality === MaterialityLevel.M3_Immutable) {
      // M3 uses DHT index + IPFS for retrieval
      backends.push('dht');
      if (this.config.enableIpfs && normative >= NormativeLevel.N2_Network) {
        backends.push('ipfs');
      }
    } else if (materiality === MaterialityLevel.M2_Persistent) {
      // M2 with N2+ can add IPFS for content-addressing (E3+)
      if (
        normative >= NormativeLevel.N2_Network &&
        empirical >= EmpiricalLevel.E3_Cryptographic &&
        this.config.enableIpfs
      ) {
        backends.push('ipfs');
      }
      // Local backup for N0/N1
      if (normative < NormativeLevel.N2_Network) {
        backends.push('local');
      }
    }

    return backends;
  }

  /**
   * Calculate replication factor
   */
  getReplication(
    classification: EpistemicClassification,
    networkSize: number
  ): number {
    const { materiality, normative } = classification;

    switch (materiality) {
      case MaterialityLevel.M0_Ephemeral:
        return 0; // No replication

      case MaterialityLevel.M1_Temporal:
        return 1; // Local only

      case MaterialityLevel.M2_Persistent:
        switch (normative) {
          case NormativeLevel.N0_Personal:
            return 3; // Personal data, moderate replication
          case NormativeLevel.N1_Communal:
            return 5; // Community data, higher replication
          default:
            return this.config.dhtDefaultReplication;
        }

      case MaterialityLevel.M3_Immutable:
        // Logarithmic replication for M3
        return this.calculateLogarithmicReplication(networkSize);

      default:
        return 1;
    }
  }

  /**
   * Logarithmic Replication for M3
   *
   * Formula: Replication = ceil(log₂(NetworkSize))
   *
   * Ensures:
   * - Small networks: Adequate redundancy (log₂(100) = 7)
   * - Large networks: No bloat (log₂(1M) = 20)
   */
  private calculateLogarithmicReplication(networkSize: number): number {
    const { m3MinReplication, m3MaxReplication } = this.config;

    if (networkSize <= 0) return m3MinReplication;

    const logReplication = Math.ceil(Math.log2(networkSize));

    return Math.max(m3MinReplication, Math.min(m3MaxReplication, logReplication));
  }

  /**
   * Get mutability mode from E-level
   */
  private getMutabilityForEmpirical(empirical: EmpiricalLevel): MutabilityMode {
    if (empirical <= EmpiricalLevel.E1_Testimonial) {
      return MutabilityMode.MUTABLE_CRDT;
    }
    if (empirical === EmpiricalLevel.E2_PrivateVerify) {
      return MutabilityMode.APPEND_ONLY;
    }
    return MutabilityMode.IMMUTABLE;
  }

  /**
   * Get access control mode from N-level
   */
  private getAccessControlForNormative(normative: NormativeLevel): AccessControlMode {
    switch (normative) {
      case NormativeLevel.N0_Personal:
        return AccessControlMode.OWNER;
      case NormativeLevel.N1_Communal:
        return AccessControlMode.CAPBAC;
      default:
        return AccessControlMode.PUBLIC;
    }
  }

  /**
   * Get TTL for M-level
   */
  private getTTL(materiality: MaterialityLevel): number | undefined {
    switch (materiality) {
      case MaterialityLevel.M0_Ephemeral:
        return this.config.m0DefaultTtl;
      case MaterialityLevel.M1_Temporal:
        return this.config.m1DefaultTtl;
      default:
        return undefined; // No expiry
    }
  }

  /**
   * Check if reclassification requires migration
   */
  requiresMigration(
    from: EpistemicClassification,
    to: EpistemicClassification
  ): boolean {
    const fromTier = this.route(from);
    const toTier = this.route(to);

    // Migration required if backend changes
    if (fromTier.backend !== toTier.backend) return true;

    // Or if encryption status changes
    if (fromTier.encrypted !== toTier.encrypted) return true;

    // Or if mutability mode changes (more restrictive)
    if (fromTier.mutability !== toTier.mutability) return true;

    // Or if content-addressing changes
    if (fromTier.contentAddressed !== toTier.contentAddressed) return true;

    return false;
  }

  /**
   * Get all backends for classification
   */
  getBackends(classification: EpistemicClassification): StorageBackend[] {
    const tier = this.route(classification);
    return [tier.backend, ...tier.additionalBackends];
  }

  /**
   * Validate classification transition (INV-2: monotonically increasing)
   */
  validateTransition(
    from: EpistemicClassification,
    to: EpistemicClassification
  ): { valid: boolean; error?: string } {
    // E-level must be >= (cannot downgrade verification)
    if (to.empirical < from.empirical) {
      return {
        valid: false,
        error: `Cannot downgrade E-level from E${from.empirical} to E${to.empirical}`,
      };
    }

    // N-level must be >= (cannot narrow consensus scope)
    if (to.normative < from.normative) {
      return {
        valid: false,
        error: `Cannot downgrade N-level from N${from.normative} to N${to.normative}`,
      };
    }

    // M-level must be >= (cannot reduce durability)
    if (to.materiality < from.materiality) {
      return {
        valid: false,
        error: `Cannot downgrade M-level from M${from.materiality} to M${to.materiality}`,
      };
    }

    return { valid: true };
  }
}

/**
 * Global router instance with default configuration
 */
export const defaultRouter = new StorageRouter();

/**
 * Create a router with custom configuration
 */
export function createRouter(
  config?: Partial<RouterConfig>,
  networkSizeProvider?: () => number
): StorageRouter {
  return new StorageRouter(
    { ...DEFAULT_ROUTER_CONFIG, ...config },
    networkSizeProvider
  );
}
