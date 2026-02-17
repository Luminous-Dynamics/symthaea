/**
 * @mycelix/sdk SupplyChain Integration
 *
 * hApp-specific adapter for Mycelix-SupplyChain providing:
 * - Complete provenance tracking from origin to destination
 * - Multi-evidence checkpoint verification (IoT, GPS, blockchain, etc.)
 * - Chain integrity scoring and weak link detection
 * - Handler reputation management
 * - Cross-hApp trust integration via Bridge
 *
 * @packageDocumentation
 * @module integrations/supplychain
 * @see {@link SupplyChainProvenanceService} - Main service class
 * @see {@link getSupplyChainService} - Singleton accessor
 *
 * @example Recording a checkpoint
 * ```typescript
 * import { getSupplyChainService } from '@mycelix/sdk/integrations/supplychain';
 *
 * const supplychain = getSupplyChainService();
 *
 * const checkpoint = supplychain.recordCheckpoint({
 *   productId: 'organic-coffee-001',
 *   location: 'Processing Facility, Colombia',
 *   coordinates: { lat: 6.25, lng: -75.56 },
 *   handler: 'processor-cafe-co',
 *   action: 'processed',
 *   evidence: [
 *     { type: 'iot_sensor', data: { temp: 22, humidity: 65 }, timestamp: Date.now(), verified: true },
 *     { type: 'photo', data: { url: 'batch.jpg' }, timestamp: Date.now(), verified: true },
 *   ],
 * });
 * ```
 *
 * @example Verifying a provenance chain
 * ```typescript
 * const verification = supplychain.verifyChain('organic-coffee-001');
 *
 * if (verification.verified) {
 *   console.log('Full chain verified');
 *   console.log(`Integrity: ${(verification.integrityScore * 100).toFixed(1)}%`);
 * } else {
 *   console.log('Weak links:', verification.weakLinks);
 *   console.log('Recommendations:', verification.recommendations);
 * }
 * ```
 */

import { LocalBridge, createReputationQuery } from '../../bridge/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';


// ============================================================================
// SupplyChain-Specific Types
// ============================================================================

/**
 * Evidence types for provenance verification
 *
 * @remarks
 * Evidence types ranked by verification strength (highest to lowest):
 * 1. `blockchain` - Immutable, cryptographically secured
 * 2. `third_party_verification` - Independent audit
 * 3. `iot_sensor` - Automated, tamper-resistant
 * 4. `rfid` - Hardware-based tracking
 * 5. `gps` - Location verification
 * 6. `photo` - Visual evidence
 * 7. `signature` - Human attestation
 * 8. `manual_entry` - Lowest confidence
 */
export type EvidenceType =
  | 'manual_entry'
  | 'signature'
  | 'photo'
  | 'gps'
  | 'iot_sensor'
  | 'rfid'
  | 'blockchain'
  | 'third_party_verification';

/** Checkpoint evidence */
export interface CheckpointEvidence {
  type: EvidenceType;
  data: Record<string, unknown>;
  timestamp: number;
  verified: boolean;
}

/** Supply chain checkpoint */
export interface Checkpoint {
  id: string;
  productId: string;
  batchId?: string;
  location: string;
  coordinates?: { lat: number; lng: number };
  handler: string;
  handlerDid: string;
  action: 'received' | 'processed' | 'shipped' | 'delivered' | 'inspected';
  timestamp: number;
  evidence: CheckpointEvidence[];
  claim: EpistemicClaim;
  previousCheckpointId?: string;
}

/** Product provenance chain */
export interface ProvenanceChain {
  productId: string;
  productName: string;
  origin: Checkpoint;
  checkpoints: Checkpoint[];
  currentLocation: string;
  chainIntegrity: number;
  totalDistance: number;
  totalTime: number;
}

/** Handler profile */
export interface HandlerProfile {
  handlerId: string;
  handlerDid: string;
  organization: string;
  role: 'producer' | 'processor' | 'distributor' | 'retailer' | 'inspector';
  reputation: ReputationScore;
  trustScore: number;
  checkpointsRecorded: number;
  verificationRate: number;
  lastActive: number;
}

/** Chain verification result */
export interface ChainVerification {
  productId: string;
  verified: boolean;
  integrityScore: number;
  weakLinks: string[];
  recommendations: string[];
  timeline: Array<{
    checkpoint: string;
    status: 'verified' | 'suspicious' | 'unverified';
    reason?: string;
  }>;
}

// ============================================================================
// SupplyChain Provenance Service
// ============================================================================

/**
 * SupplyChainProvenanceService - Provenance tracking and trust for supply chains
 *
 * @example
 * ```typescript
 * const supplychain = new SupplyChainProvenanceService();
 *
 * // Record a checkpoint
 * const checkpoint = supplychain.recordCheckpoint({
 *   productId: 'product-123',
 *   location: 'Warehouse A',
 *   handler: 'handler-456',
 *   action: 'received',
 *   evidence: [{
 *     type: 'iot_sensor',
 *     data: { temperature: 4.5, humidity: 45 },
 *     timestamp: Date.now(),
 *     verified: true,
 *   }],
 * });
 *
 * // Verify the full provenance chain
 * const verification = supplychain.verifyChain('product-123');
 * ```
 */
export class SupplyChainProvenanceService {
  private checkpoints: Map<string, Checkpoint> = new Map();
  private handlerReputations: Map<string, ReputationScore> = new Map();
  private productChains: Map<string, string[]> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('supplychain');
  }

  /**
   * Record a provenance checkpoint with evidence
   *
   * @param params - Checkpoint parameters
   * @param params.productId - Product being tracked
   * @param params.batchId - Optional batch identifier
   * @param params.location - Human-readable location description
   * @param params.coordinates - Optional GPS coordinates
   * @param params.handler - Entity handling the product
   * @param params.action - Action performed (received, processed, shipped, delivered, inspected)
   * @param params.evidence - Array of evidence items supporting the checkpoint
   * @returns Created checkpoint with epistemic claim
   *
   * @remarks
   * - Checkpoints are automatically linked to form a provenance chain
   * - The empirical level of the claim is determined by evidence quality
   * - Handler reputation is updated with each checkpoint recorded
   *
   * @example
   * ```typescript
   * const checkpoint = service.recordCheckpoint({
   *   productId: 'product-001',
   *   batchId: 'BATCH-2024-001',
   *   location: 'Distribution Center, Miami',
   *   coordinates: { lat: 25.76, lng: -80.19 },
   *   handler: 'distributor-east',
   *   action: 'received',
   *   evidence: [
   *     { type: 'rfid', data: { tag: 'ABC123' }, timestamp: Date.now(), verified: true },
   *     { type: 'iot_sensor', data: { temp: 4.5 }, timestamp: Date.now(), verified: true },
   *   ],
   * });
   * ```
   */
  recordCheckpoint(params: {
    productId: string;
    batchId?: string;
    location: string;
    coordinates?: { lat: number; lng: number };
    handler: string;
    action: 'received' | 'processed' | 'shipped' | 'delivered' | 'inspected';
    evidence: CheckpointEvidence[];
  }): Checkpoint {
    const { productId, batchId, location, coordinates, handler, action, evidence } =
      params;

    // Get previous checkpoint
    const existingChain = this.productChains.get(productId) || [];
    const previousCheckpointId =
      existingChain.length > 0 ? existingChain[existingChain.length - 1] : undefined;

    // Determine empirical level based on evidence quality
    const empiricalLevel = this.determineEmpiricalLevel(evidence);

    // Create epistemic claim
    const checkpointClaim = claim(`Product ${productId} ${action} at ${location}`)
      .withEmpirical(empiricalLevel)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .build();

    // Create checkpoint
    const checkpoint: Checkpoint = {
      id: `cp-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      productId,
      batchId,
      location,
      coordinates,
      handler,
      handlerDid: `did:mycelix:${handler}`,
      action,
      timestamp: Date.now(),
      evidence,
      claim: checkpointClaim,
      previousCheckpointId,
    };

    // Store checkpoint
    this.checkpoints.set(checkpoint.id, checkpoint);

    // Update product chain
    existingChain.push(checkpoint.id);
    this.productChains.set(productId, existingChain);

    // Update handler reputation
    let handlerRep = this.handlerReputations.get(handler) || createReputation(handler);
    handlerRep = recordPositive(handlerRep);
    this.handlerReputations.set(handler, handlerRep);

    // Store reputation in bridge for cross-hApp queries
    this.bridge.setReputation('supplychain', handler, handlerRep);

    return checkpoint;
  }

  /**
   * Get the complete provenance chain for a product
   *
   * @param productId - Product identifier to trace
   * @returns Full provenance chain or null if product not found
   *
   * @remarks
   * The chain includes:
   * - Origin checkpoint (first recorded)
   * - All intermediate checkpoints in order
   * - Current location (latest checkpoint)
   * - Chain integrity score (0-1)
   * - Total distance traveled (km, if coordinates available)
   * - Total time in chain (hours)
   *
   * @example
   * ```typescript
   * const chain = service.getProvenanceChain('product-001');
   *
   * if (chain) {
   *   console.log(`Origin: ${chain.origin.location}`);
   *   console.log(`Current: ${chain.currentLocation}`);
   *   console.log(`Checkpoints: ${chain.checkpoints.length}`);
   *   console.log(`Distance: ${chain.totalDistance.toFixed(0)}km`);
   *   console.log(`Integrity: ${(chain.chainIntegrity * 100).toFixed(1)}%`);
   * }
   * ```
   */
  getProvenanceChain(productId: string): ProvenanceChain | null {
    const checkpointIds = this.productChains.get(productId);
    if (!checkpointIds || checkpointIds.length === 0) {
      return null;
    }

    const checkpoints = checkpointIds
      .map((id) => this.checkpoints.get(id))
      .filter((cp): cp is Checkpoint => cp !== undefined);

    if (checkpoints.length === 0) {
      return null;
    }

    const origin = checkpoints[0];
    const current = checkpoints[checkpoints.length - 1];

    return {
      productId,
      productName: `Product ${productId}`,
      origin,
      checkpoints,
      currentLocation: current.location,
      chainIntegrity: this.calculateChainIntegrity(checkpoints),
      totalDistance: this.calculateTotalDistance(checkpoints),
      totalTime: (current.timestamp - origin.timestamp) / (1000 * 60 * 60),
    };
  }

  /**
   * Verify the integrity of a complete provenance chain
   *
   * @param productId - Product identifier to verify
   * @returns Comprehensive verification result with timeline and recommendations
   *
   * @remarks
   * Verification checks for each checkpoint:
   * 1. Evidence quality (verified vs unverified)
   * 2. Time gaps between checkpoints (>72 hours flagged as suspicious)
   * 3. Handler reputation (low trust handlers flagged)
   *
   * A chain is considered `verified` when:
   * - Chain integrity >= 0.8
   * - No weak links detected
   *
   * @example
   * ```typescript
   * const verification = service.verifyChain('product-001');
   *
   * console.log(`Verified: ${verification.verified}`);
   * console.log(`Integrity: ${(verification.integrityScore * 100).toFixed(1)}%`);
   *
   * if (verification.weakLinks.length > 0) {
   *   console.log('Weak links found:', verification.weakLinks);
   * }
   *
   * verification.timeline.forEach(t => {
   *   console.log(`${t.checkpoint}: ${t.status} ${t.reason || ''}`);
   * });
   * ```
   */
  verifyChain(productId: string): ChainVerification {
    const chain = this.getProvenanceChain(productId);
    if (!chain) {
      return {
        productId,
        verified: false,
        integrityScore: 0,
        weakLinks: [],
        recommendations: ['Product not found in supply chain'],
        timeline: [],
      };
    }

    const weakLinks: string[] = [];
    const recommendations: string[] = [];
    const timeline: ChainVerification['timeline'] = [];

    for (let i = 0; i < chain.checkpoints.length; i++) {
      const cp = chain.checkpoints[i];
      const prevCp = i > 0 ? chain.checkpoints[i - 1] : null;

      let status: 'verified' | 'suspicious' | 'unverified' = 'unverified';
      let reason: string | undefined;

      // Check evidence quality
      const hasStrongEvidence = cp.evidence.some(
        (e) =>
          e.verified &&
          ['iot_sensor', 'blockchain', 'third_party_verification'].includes(e.type)
      );

      if (hasStrongEvidence) {
        status = 'verified';
      } else if (cp.evidence.some((e) => e.verified)) {
        status = 'verified';
      } else {
        status = 'unverified';
        reason = 'No verified evidence';
        weakLinks.push(cp.id);
      }

      // Check for time gaps
      if (prevCp) {
        const timeDiff = (cp.timestamp - prevCp.timestamp) / (1000 * 60 * 60);
        if (timeDiff > 72) {
          status = 'suspicious';
          reason = `Large time gap (${timeDiff.toFixed(1)} hours)`;
          if (!weakLinks.includes(cp.id)) {
            weakLinks.push(cp.id);
          }
        }
      }

      // Check handler reputation
      const handlerRep = this.handlerReputations.get(cp.handler);
      if (handlerRep && reputationValue(handlerRep) < 0.5) {
        status = 'suspicious';
        reason = 'Handler has low trust score';
        if (!weakLinks.includes(cp.id)) {
          weakLinks.push(cp.id);
        }
      }

      timeline.push({ checkpoint: cp.id, status, reason });
    }

    if (weakLinks.length > 0) {
      recommendations.push(
        `${weakLinks.length} checkpoint(s) have weak verification`
      );
    }

    if (chain.chainIntegrity < 0.9) {
      recommendations.push('Chain integrity below 90% - investigate gaps');
    }

    const verifiedCount = timeline.filter((t) => t.status === 'verified').length;
    if (verifiedCount === timeline.length) {
      recommendations.push('Full chain verified - product provenance confirmed');
    }

    return {
      productId,
      verified: chain.chainIntegrity >= 0.8 && weakLinks.length === 0,
      integrityScore: chain.chainIntegrity,
      weakLinks,
      recommendations,
      timeline,
    };
  }

  /**
   * Get handler profile
   */
  getHandlerProfile(handlerId: string): HandlerProfile {
    const reputation = this.handlerReputations.get(handlerId) || createReputation(handlerId);

    let checkpointsRecorded = 0;
    let verifiedCheckpoints = 0;

    for (const cp of this.checkpoints.values()) {
      if (cp.handler === handlerId) {
        checkpointsRecorded++;
        if (cp.evidence.some((e) => e.verified)) {
          verifiedCheckpoints++;
        }
      }
    }

    return {
      handlerId,
      handlerDid: `did:mycelix:${handlerId}`,
      organization: 'Unknown',
      role: 'distributor',
      reputation,
      trustScore: reputationValue(reputation),
      checkpointsRecorded,
      verificationRate:
        checkpointsRecorded > 0 ? verifiedCheckpoints / checkpointsRecorded : 0,
      lastActive: Date.now(),
    };
  }

  /**
   * Query handler reputation from other hApps
   */
  queryExternalReputation(handlerId: string): void {
    const query = createReputationQuery('supplychain', handlerId);
    this.bridge.send('marketplace', query);
    this.bridge.send('mail', query);
  }

  /**
   * Check if handler is trusted
   */
  isHandlerTrusted(handlerId: string, minimumScore = 0.5): boolean {
    const reputation = this.handlerReputations.get(handlerId);
    if (!reputation) return false;
    return reputationValue(reputation) >= minimumScore;
  }

  // Private helpers

  private determineEmpiricalLevel(evidence: CheckpointEvidence[]): EmpiricalLevel {
    const hasBlockchain = evidence.some(
      (e) => e.verified && e.type === 'blockchain'
    );
    const hasIoT = evidence.some((e) => e.verified && e.type === 'iot_sensor');
    const hasThirdParty = evidence.some(
      (e) => e.verified && e.type === 'third_party_verification'
    );
    const hasPhoto = evidence.some((e) => e.verified && e.type === 'photo');
    const hasGPS = evidence.some((e) => e.verified && e.type === 'gps');

    if (hasBlockchain) return EmpiricalLevel.E4_Consensus;
    if (hasIoT && hasGPS && hasPhoto) return EmpiricalLevel.E3_Cryptographic;
    if ((hasIoT || hasGPS) && (hasPhoto || hasThirdParty))
      return EmpiricalLevel.E2_PrivateVerify;
    if (evidence.some((e) => e.verified)) return EmpiricalLevel.E1_Testimonial;
    return EmpiricalLevel.E0_Unverified;
  }

  private calculateChainIntegrity(checkpoints: Checkpoint[]): number {
    if (checkpoints.length === 0) return 0;

    let integrityScore = 0;
    for (const cp of checkpoints) {
      const evidenceScore = cp.evidence.some((e) => e.verified) ? 1 : 0.5;
      const cryptoBonus = cp.evidence.some((e) =>
        ['iot_sensor', 'blockchain', 'rfid'].includes(e.type)
      )
        ? 0.2
        : 0;
      integrityScore += evidenceScore + cryptoBonus;
    }

    return Math.min(1, integrityScore / checkpoints.length);
  }

  private calculateTotalDistance(checkpoints: Checkpoint[]): number {
    let total = 0;
    for (let i = 1; i < checkpoints.length; i++) {
      const prev = checkpoints[i - 1];
      const curr = checkpoints[i];
      if (prev.coordinates && curr.coordinates) {
        total += this.haversineDistance(
          prev.coordinates.lat,
          prev.coordinates.lng,
          curr.coordinates.lat,
          curr.coordinates.lng
        );
      }
    }
    return total;
  }

  private haversineDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
};

// Default service instance
let defaultService: SupplyChainProvenanceService | null = null;

/**
 * Get the default supply chain provenance service instance
 */
export function getSupplyChainService(): SupplyChainProvenanceService {
  if (!defaultService) {
    defaultService = new SupplyChainProvenanceService();
  }
  return defaultService;
}
