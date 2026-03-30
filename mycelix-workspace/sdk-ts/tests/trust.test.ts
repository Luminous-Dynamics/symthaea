// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Module Tests
 *
 * Tests for the trust pipeline, K-Vector operations, proof statements,
 * domain translation, and the full pipeline flow.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  // K-Vector
  createUniformKVector,
  kvectorFromArray,
  kvectorToArray,
  computeTrustScore,
  isVerified,
  KVectorDimension,
  KVECTOR_WEIGHTS,
  KVECTOR_DIMENSION_NAMES,

  // Trust Delta
  TrustDirection,
  applyTrustDelta,

  // Epistemic
  classificationCode,

  // Provenance
  DerivationType,

  // Consensus
  PhiConsensusStatus,

  // Pipeline
  TrustPipeline,
  createTrustPipeline,
  createAgentOutput,
  createAgentVote,
  DEFAULT_PIPELINE_CONFIG,
  TrustPipelineError,
  TrustPipelineErrorCode,

  // Proofs
  ProofStatementType,
  trustAbove,
  trustInRange,
  dimensionAbove,
  multipleDimensionsAbove,
  provenanceQuality,
  evaluateStatement,
  evaluateProvenanceStatement,
  generateSimulatedProof,
  verifySimulatedProof,
  aggregateProofs,

  // Domains
  balancedRelevance,
  relevanceToArray,
  relevanceSimilarity,
  DOMAIN_FINANCIAL,
  DOMAIN_CODE_REVIEW,
  DOMAIN_SOCIAL,
  DOMAIN_GOVERNANCE,
  ALL_DOMAINS,
  DomainRegistry,
  defaultDomainRegistry,
  computeDomainTrust,
  translateTrust,
  analyzeDomainCompatibility,
} from '../src/trust/index.js';

// =============================================================================
// K-Vector Tests
// =============================================================================

describe('KVector', () => {
  describe('createUniformKVector', () => {
    it('should create a vector with all dimensions equal', () => {
      const kv = createUniformKVector(0.7);
      expect(kv.k_r).toBe(0.7);
      expect(kv.k_a).toBe(0.7);
      expect(kv.k_i).toBe(0.7);
      expect(kv.k_p).toBe(0.7);
      expect(kv.k_m).toBe(0.7);
      expect(kv.k_s).toBe(0.7);
      expect(kv.k_h).toBe(0.7);
      expect(kv.k_topo).toBe(0.7);
      expect(kv.k_v).toBe(0.7);
    });

    it('should clamp values above 1', () => {
      const kv = createUniformKVector(1.5);
      expect(kv.k_r).toBe(1);
    });

    it('should clamp values below 0', () => {
      const kv = createUniformKVector(-0.5);
      expect(kv.k_r).toBe(0);
    });
  });

  describe('kvectorFromArray / kvectorToArray', () => {
    it('should round-trip correctly', () => {
      const arr: [number, number, number, number, number, number, number, number, number] =
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
      const kv = kvectorFromArray(arr);
      const result = kvectorToArray(kv);
      expect(result).toEqual(arr);
    });

    it('should clamp out-of-range values in fromArray', () => {
      const kv = kvectorFromArray([1.5, -0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
      expect(kv.k_r).toBe(1);
      expect(kv.k_a).toBe(0);
    });
  });

  describe('computeTrustScore', () => {
    it('should compute weighted average', () => {
      const kv = createUniformKVector(0.8);
      const score = computeTrustScore(kv);
      expect(score).toBeCloseTo(0.8, 5);
    });

    it('should return 0 for zero vector', () => {
      const kv = createUniformKVector(0);
      expect(computeTrustScore(kv)).toBe(0);
    });

    it('should weight dimensions differently', () => {
      const kv = kvectorFromArray([1, 0, 0, 0, 0, 0, 0, 0, 0]);
      const score = computeTrustScore(kv);
      // Only reputation dimension has value, weight is 0.15
      expect(score).toBeCloseTo(0.15, 2);
    });
  });

  describe('isVerified', () => {
    it('should return true when k_v >= threshold', () => {
      const kv = createUniformKVector(0.7);
      expect(isVerified(kv)).toBe(true);
    });

    it('should return false when k_v < threshold', () => {
      const kv = createUniformKVector(0.3);
      expect(isVerified(kv)).toBe(false);
    });

    it('should accept custom threshold', () => {
      const kv = createUniformKVector(0.8);
      expect(isVerified(kv, 0.9)).toBe(false);
      expect(isVerified(kv, 0.7)).toBe(true);
    });
  });

  describe('constants', () => {
    it('should have 9 dimension names', () => {
      expect(Object.keys(KVECTOR_DIMENSION_NAMES)).toHaveLength(9);
    });

    it('should have weights summing to 1', () => {
      const sum = Object.values(KVECTOR_WEIGHTS).reduce((s, w) => s + w, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });
  });
});

// =============================================================================
// Trust Delta Tests
// =============================================================================

describe('TrustDelta', () => {
  describe('applyTrustDelta', () => {
    it('should increase dimensions with positive delta', () => {
      const kv = createUniformKVector(0.5);
      const delta = {
        reputationDelta: 0.1,
        performanceDelta: 0.05,
        integrityDelta: 0.02,
        historicalDelta: 0.01,
        direction: TrustDirection.Increase,
        reason: 'test',
      };
      const result = applyTrustDelta(kv, delta);
      expect(result.k_r).toBeCloseTo(0.6);
      expect(result.k_p).toBeCloseTo(0.55);
      expect(result.k_i).toBeCloseTo(0.52);
      expect(result.k_h).toBeCloseTo(0.51);
    });

    it('should not change activity, membership, stake, topology, or verification', () => {
      const kv = createUniformKVector(0.5);
      const delta = {
        reputationDelta: 0.1,
        performanceDelta: 0.1,
        integrityDelta: 0.1,
        historicalDelta: 0.1,
        direction: TrustDirection.Increase,
        reason: 'test',
      };
      const result = applyTrustDelta(kv, delta);
      expect(result.k_a).toBe(0.5);
      expect(result.k_m).toBe(0.5);
      expect(result.k_s).toBe(0.5);
      expect(result.k_topo).toBe(0.5);
      expect(result.k_v).toBe(0.5);
    });

    it('should clamp to [0, 1] range', () => {
      const kv = createUniformKVector(0.95);
      const delta = {
        reputationDelta: 0.2,
        performanceDelta: 0,
        integrityDelta: 0,
        historicalDelta: 0,
        direction: TrustDirection.Increase,
        reason: 'test',
      };
      const result = applyTrustDelta(kv, delta);
      expect(result.k_r).toBe(1);
    });
  });
});

// =============================================================================
// Classification Code Tests
// =============================================================================

describe('classificationCode', () => {
  it('should format E-N-M-H code', () => {
    const code = classificationCode({
      empirical: 3,
      normative: 2,
      materiality: 1,
      harmonic: 0,
    });
    expect(code).toBe('E3-N2-M1-H0');
  });
});

// =============================================================================
// Proof Statement Tests
// =============================================================================

describe('Proof Statements', () => {
  describe('builders', () => {
    it('should build trustAbove statement', () => {
      const stmt = trustAbove(0.7);
      expect(stmt.type).toBe(ProofStatementType.TrustExceedsThreshold);
      expect(stmt.threshold).toBe(0.7);
    });

    it('should clamp trustAbove to [0,1]', () => {
      expect(trustAbove(1.5).threshold).toBe(1);
      expect(trustAbove(-0.5).threshold).toBe(0);
    });

    it('should build trustInRange statement', () => {
      const stmt = trustInRange(0.3, 0.8);
      expect(stmt.type).toBe(ProofStatementType.TrustInRange);
      expect(stmt.minTrust).toBe(0.3);
      expect(stmt.maxTrust).toBe(0.8);
    });

    it('should build dimensionAbove statement', () => {
      const stmt = dimensionAbove(KVectorDimension.Integrity, 0.6);
      expect(stmt.type).toBe(ProofStatementType.DimensionExceedsThreshold);
      expect(stmt.dimension).toBe(2);
      expect(stmt.threshold).toBe(0.6);
    });

    it('should build multipleDimensionsAbove statement', () => {
      const stmt = multipleDimensionsAbove([0, 2, 3], 0.5);
      expect(stmt.type).toBe(ProofStatementType.MultipleDimensionsExceed);
      expect(stmt.dimensions).toEqual([0, 2, 3]);
    });

    it('should build provenanceQuality statement', () => {
      const stmt = provenanceQuality(2, 5, 0.8);
      expect(stmt.type).toBe(ProofStatementType.ProvenanceQuality);
      expect(stmt.minEpistemicLevel).toBe(2);
      expect(stmt.maxChainDepth).toBe(5);
      expect(stmt.minConfidence).toBe(0.8);
    });
  });

  describe('evaluateStatement', () => {
    const highKV = createUniformKVector(0.8);
    const lowKV = createUniformKVector(0.2);

    it('should evaluate TrustExceedsThreshold correctly', () => {
      expect(evaluateStatement(highKV, trustAbove(0.5))).toBe(true);
      expect(evaluateStatement(lowKV, trustAbove(0.5))).toBe(false);
    });

    it('should evaluate TrustInRange correctly', () => {
      expect(evaluateStatement(highKV, trustInRange(0.5, 0.9))).toBe(true);
      expect(evaluateStatement(lowKV, trustInRange(0.5, 0.9))).toBe(false);
    });

    it('should evaluate DimensionExceedsThreshold', () => {
      const kv = kvectorFromArray([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
      expect(evaluateStatement(kv, dimensionAbove(0, 0.5))).toBe(true);
      expect(evaluateStatement(kv, dimensionAbove(1, 0.5))).toBe(false);
    });

    it('should evaluate MultipleDimensionsExceed', () => {
      const kv = kvectorFromArray([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
      expect(evaluateStatement(kv, multipleDimensionsAbove([0, 1, 2], 0.5))).toBe(true);
      expect(evaluateStatement(kv, multipleDimensionsAbove([0, 3], 0.5))).toBe(false);
    });

    it('should return false for KVectorMatchesCommitment', () => {
      expect(evaluateStatement(highKV, {
        type: ProofStatementType.KVectorMatchesCommitment,
        commitmentHash: 'abc',
      })).toBe(false);
    });
  });

  describe('evaluateProvenanceStatement', () => {
    const chain = {
      rootId: 'r1',
      nodes: [{ nodeId: 'r1', contentHash: 'h', creator: 'a', timestamp: 1, epistemicLevel: 3, epistemicFloor: 2, confidence: 0.9, derivationType: DerivationType.Original, parentIds: [] }],
      depth: 3,
      minEpistemicLevel: 2,
      confidence: 0.9,
      hasVerifiedRoots: true,
      derivationSteps: 2,
    };

    it('should evaluate ProvenanceEpistemicFloor', () => {
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceEpistemicFloor,
        minLevel: 2,
      })).toBe(true);
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceEpistemicFloor,
        minLevel: 4,
      })).toBe(false);
    });

    it('should evaluate ProvenanceChainDepth', () => {
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceChainDepth,
        maxDepth: 5,
      })).toBe(true);
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceChainDepth,
        maxDepth: 1,
      })).toBe(false);
    });

    it('should evaluate ProvenanceConfidence', () => {
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceConfidence,
        minConfidence: 0.8,
      })).toBe(true);
    });

    it('should evaluate ProvenanceHasVerifiedRoots', () => {
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceHasVerifiedRoots,
      })).toBe(true);
    });

    it('should evaluate ProvenanceChainValid', () => {
      expect(evaluateProvenanceStatement(chain, {
        type: ProofStatementType.ProvenanceChainValid,
      })).toBe(true);
    });

    it('should evaluate ProvenanceQuality', () => {
      expect(evaluateProvenanceStatement(chain, provenanceQuality(2, 5, 0.8))).toBe(true);
      expect(evaluateProvenanceStatement(chain, provenanceQuality(4, 5, 0.8))).toBe(false);
    });
  });

  describe('generateSimulatedProof / verifySimulatedProof', () => {
    it('should generate a valid simulated proof', () => {
      const kv = createUniformKVector(0.8);
      const proof = generateSimulatedProof(kv, trustAbove(0.5), {
        agentId: 'agent-1',
        timestamp: Date.now(),
        simulationMode: true,
      });

      expect(proof.proofId).toBeDefined();
      expect(proof.isSimulation).toBe(true);
      expect(proof.commitment.hash).toBeDefined();
      expect(proof.statement.type).toBe(ProofStatementType.TrustExceedsThreshold);
    });

    it('should verify simulated proof', () => {
      const kv = createUniformKVector(0.8);
      const proof = generateSimulatedProof(kv, trustAbove(0.5), {
        agentId: 'agent-1',
        timestamp: Date.now(),
        simulationMode: true,
      });
      const result = verifySimulatedProof(proof);
      expect(result.isValid).toBe(true);
      expect(result.verifiedStatement).toBeDefined();
    });

    it('should reject non-simulation proofs', () => {
      const kv = createUniformKVector(0.8);
      const proof = generateSimulatedProof(kv, trustAbove(0.5), {
        agentId: 'agent-1',
        timestamp: Date.now(),
        simulationMode: true,
      });
      proof.isSimulation = false;
      const result = verifySimulatedProof(proof);
      expect(result.isValid).toBe(false);
    });
  });

  describe('aggregateProofs', () => {
    it('should aggregate multiple valid proofs', () => {
      const kv = createUniformKVector(0.8);
      const config = { agentId: 'a', timestamp: Date.now(), simulationMode: true };
      const proofs = [
        generateSimulatedProof(kv, trustAbove(0.5), config),
        generateSimulatedProof(kv, trustInRange(0.5, 0.9), config),
      ];
      const result = aggregateProofs(proofs);
      expect(result.allValid).toBe(true);
      expect(result.proofs).toHaveLength(2);
    });
  });
});

// =============================================================================
// Domain Tests
// =============================================================================

describe('Trust Domains', () => {
  describe('DomainRelevance', () => {
    it('should create balanced relevance', () => {
      const rel = balancedRelevance();
      const arr = relevanceToArray(rel);
      expect(arr).toHaveLength(9);
      expect(arr.every(v => v === 1.0)).toBe(true);
    });

    it('should compute similarity of 1.0 for identical profiles', () => {
      const rel = balancedRelevance();
      expect(relevanceSimilarity(rel, rel)).toBeCloseTo(1.0);
    });

    it('should compute similarity < 1 for different profiles', () => {
      const sim = relevanceSimilarity(
        DOMAIN_FINANCIAL.dimensionRelevance,
        DOMAIN_SOCIAL.dimensionRelevance
      );
      expect(sim).toBeGreaterThan(0);
      expect(sim).toBeLessThan(1);
    });
  });

  describe('predefined domains', () => {
    it('should have 6 predefined domains', () => {
      expect(ALL_DOMAINS).toHaveLength(6);
    });

    it('should have financial domain requiring verification', () => {
      expect(DOMAIN_FINANCIAL.requiresVerification).toBe(true);
    });

    it('should have code review domain not requiring verification', () => {
      expect(DOMAIN_CODE_REVIEW.requiresVerification).toBe(false);
    });
  });

  describe('computeDomainTrust', () => {
    it('should compute domain-weighted trust', () => {
      const kv = createUniformKVector(0.8);
      const trust = computeDomainTrust(kv, DOMAIN_FINANCIAL);
      expect(trust).toBeGreaterThan(0);
      expect(trust).toBeLessThanOrEqual(1);
    });

    it('should give higher trust for aligned dimensions', () => {
      // Financial domain cares about integrity and stake
      const aligned = kvectorFromArray([0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]);
      const misaligned = kvectorFromArray([0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1]);
      expect(computeDomainTrust(aligned, DOMAIN_FINANCIAL))
        .toBeGreaterThan(computeDomainTrust(misaligned, DOMAIN_FINANCIAL));
    });
  });

  describe('translateTrust', () => {
    it('should translate between domains', () => {
      const kv = createUniformKVector(0.8);
      const result = translateTrust(kv, DOMAIN_CODE_REVIEW, DOMAIN_FINANCIAL);
      expect(result.sourceDomainId).toBe('code_review');
      expect(result.targetDomainId).toBe('financial');
      expect(result.translationConfidence).toBeGreaterThan(0);
      expect(result.dimensionTranslations).toHaveLength(9);
    });

    it('should flag unverified agents for verification-requiring domains', () => {
      const kv = kvectorFromArray([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.3]);
      const result = translateTrust(kv, DOMAIN_SOCIAL, DOMAIN_FINANCIAL);
      expect(result.warnings.some(w => w.includes('verification'))).toBe(true);
    });
  });

  describe('DomainRegistry', () => {
    it('should list all default domains', () => {
      const domains = defaultDomainRegistry.listDomains();
      expect(domains).toContain('financial');
      expect(domains).toContain('code_review');
      expect(domains).toContain('social');
    });

    it('should translate between domains by ID', () => {
      const kv = createUniformKVector(0.7);
      const result = defaultDomainRegistry.translate(kv, 'social', 'governance');
      expect(result).toBeDefined();
      expect(result!.targetDomainId).toBe('governance');
    });

    it('should return undefined for unknown domains', () => {
      const kv = createUniformKVector(0.7);
      const result = defaultDomainRegistry.translate(kv, 'social', 'nonexistent');
      expect(result).toBeUndefined();
    });

    it('should register custom domains', () => {
      const registry = new DomainRegistry();
      registry.register({
        domainId: 'custom',
        name: 'Custom',
        description: 'Test domain',
        dimensionRelevance: balancedRelevance(),
        minTrustThreshold: 0.5,
        requiresVerification: false,
      });
      expect(registry.get('custom')).toBeDefined();
    });

    it('should compute similarity matrix', () => {
      const matrix = defaultDomainRegistry.similarityMatrix();
      expect(matrix.size).toBeGreaterThan(0);
      // Self-similarity should be 1.0
      const financialRow = matrix.get('financial');
      expect(financialRow?.get('financial')).toBeCloseTo(1.0);
    });
  });

  describe('analyzeDomainCompatibility', () => {
    it('should analyze compatibility between domains', () => {
      const result = analyzeDomainCompatibility(DOMAIN_CODE_REVIEW, DOMAIN_FINANCIAL);
      expect(result.source).toBe('code_review');
      expect(result.target).toBe('financial');
      expect(result.compatibility).toBeGreaterThan(0);
    });

    it('should identify strong and weak transfers', () => {
      const result = analyzeDomainCompatibility(DOMAIN_SOCIAL, DOMAIN_FINANCIAL);
      expect(result.weakTransfers.length + result.strongTransfers.length).toBeGreaterThan(0);
    });

    it('should recommend verification when target requires it', () => {
      const result = analyzeDomainCompatibility(DOMAIN_SOCIAL, DOMAIN_FINANCIAL);
      expect(result.recommendations.some(r => r.includes('verification'))).toBe(true);
    });
  });
});

// =============================================================================
// Trust Pipeline Tests
// =============================================================================

describe('TrustPipeline', () => {
  let pipeline: TrustPipeline;

  beforeEach(() => {
    pipeline = createTrustPipeline();
  });

  describe('agent management', () => {
    it('should register an agent with default K-Vector', () => {
      pipeline.registerAgent('agent-1');
      const agent = pipeline.getAgent('agent-1');
      expect(agent).toBeDefined();
      expect(agent!.agentId).toBe('agent-1');
      expect(agent!.kvector.k_r).toBe(0.5);
    });

    it('should register an agent with custom K-Vector', () => {
      const kv = createUniformKVector(0.8);
      pipeline.registerAgent('agent-1', kv);
      expect(pipeline.getAgent('agent-1')!.kvector.k_r).toBe(0.8);
    });

    it('should return undefined for unknown agent', () => {
      expect(pipeline.getAgent('nonexistent')).toBeUndefined();
    });

    it('should update agent K-Vector', () => {
      pipeline.registerAgent('agent-1');
      pipeline.updateAgentKVector('agent-1', createUniformKVector(0.9));
      expect(pipeline.getAgent('agent-1')!.kvector.k_r).toBe(0.9);
    });
  });

  describe('registerOutput', () => {
    it('should register output with provenance', () => {
      pipeline.registerAgent('agent-1');
      const output = createAgentOutput('agent-1', 'test content');
      const registered = pipeline.registerOutput(output, 'agent-1');
      expect(registered.provenance.nodeId).toBeDefined();
      expect(registered.provenance.creator).toBe('agent-1');
      expect(registered.agentId).toBe('agent-1');
    });
  });

  describe('runConsensus', () => {
    it('should return insufficient votes when below minimum', () => {
      pipeline.registerAgent('agent-1');
      const output = createAgentOutput('agent-1', 'test');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const result = pipeline.runConsensus(registered, [
        createAgentVote('voter-1', 0.8),
      ]);
      expect(result.consensusReached).toBe(false);
      expect(result.consensus.status).toBe(PhiConsensusStatus.InsufficientVotes);
    });

    it('should reach consensus with aligned votes', () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('voter-1');
      pipeline.registerAgent('voter-2');
      pipeline.registerAgent('voter-3');
      const output = createAgentOutput('agent-1', 'quality content');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const result = pipeline.runConsensus(registered, [
        createAgentVote('voter-1', 0.9, 0.9),
        createAgentVote('voter-2', 0.85, 0.8),
        createAgentVote('voter-3', 0.88, 0.85),
      ]);
      expect(result.consensusReached).toBe(true);
      expect(result.effectiveValue).toBeGreaterThan(0.5);
    });

    it('should not reach consensus with divergent votes', () => {
      pipeline.registerAgent('agent-1');
      const output = createAgentOutput('agent-1', 'controversial');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const result = pipeline.runConsensus(registered, [
        createAgentVote('v1', 0.95),
        createAgentVote('v2', 0.1),
        createAgentVote('v3', 0.5),
      ]);
      expect(result.consensus.dissent).toBeGreaterThan(0);
    });
  });

  describe('processConsensus', () => {
    it('should update agent trust after consensus', () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('v1');
      pipeline.registerAgent('v2');
      const output = createAgentOutput('agent-1', 'good output');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const consensus = pipeline.runConsensus(registered, [
        createAgentVote('v1', 0.9, 0.9),
        createAgentVote('v2', 0.85, 0.8),
      ]);
      const update = pipeline.processConsensus(consensus);
      expect(update.agentId).toBe('agent-1');
      expect(update.newTrustScore).toBeDefined();
    });

    it('should throw for unknown agent', () => {
      pipeline.registerAgent('agent-1');
      const output = createAgentOutput('agent-1', 'test');
      const registered = pipeline.registerOutput(output, 'unknown-agent');
      const consensus = pipeline.runConsensus(registered, [
        createAgentVote('v1', 0.9),
        createAgentVote('v2', 0.8),
      ]);
      // The registered output has agentId 'unknown-agent'
      expect(() => pipeline.processConsensus(consensus)).toThrow(TrustPipelineError);
    });
  });

  describe('generateAttestation', () => {
    it('should generate attestation with proof', () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('v1');
      pipeline.registerAgent('v2');
      const output = createAgentOutput('agent-1', 'attested output');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const consensus = pipeline.runConsensus(registered, [
        createAgentVote('v1', 0.9, 0.9),
        createAgentVote('v2', 0.85, 0.8),
      ]);
      const update = pipeline.processConsensus(consensus);
      const attestation = pipeline.generateAttestation(update);
      expect(attestation.proofId).toBeDefined();
      expect(attestation.kvectorCommitment).toBeDefined();
      expect(attestation.isVerified).toBe(true);
    });
  });

  describe('translateForDomain', () => {
    it('should throw for unknown target domain', () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('v1');
      pipeline.registerAgent('v2');
      const output = createAgentOutput('agent-1', 'test');
      const registered = pipeline.registerOutput(output, 'agent-1');
      const consensus = pipeline.runConsensus(registered, [
        createAgentVote('v1', 0.9, 0.9),
        createAgentVote('v2', 0.85, 0.8),
      ]);
      const update = pipeline.processConsensus(consensus);
      const attestation = pipeline.generateAttestation(update);
      expect(() => pipeline.translateForDomain(attestation, 'nonexistent')).toThrow(TrustPipelineError);
    });
  });

  describe('runFullPipeline', () => {
    it('should run complete pipeline successfully', async () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('v1');
      pipeline.registerAgent('v2');

      const output = createAgentOutput('agent-1', 'pipeline test', { empirical: 3 });
      const votes = [
        createAgentVote('v1', 0.9, 0.9),
        createAgentVote('v2', 0.85, 0.8),
      ];

      const result = await pipeline.runFullPipeline(output, 'agent-1', votes);
      expect(result.success).toBe(true);
      expect(result.attestation).toBeDefined();
      expect(result.executionTimeMs).toBeGreaterThanOrEqual(0);
    });

    it('should include translation when target domain specified', async () => {
      pipeline.registerAgent('agent-1');
      pipeline.registerAgent('v1');
      pipeline.registerAgent('v2');

      const output = createAgentOutput('agent-1', 'translate test');
      const result = await pipeline.runFullPipeline(output, 'agent-1', [
        createAgentVote('v1', 0.9, 0.9),
        createAgentVote('v2', 0.85, 0.8),
      ], { targetDomainId: 'financial' });

      expect(result.success).toBe(true);
      expect(result.translatedAttestation).toBeDefined();
    });

    it('should return error for unregistered agent', async () => {
      const output = createAgentOutput('unknown', 'test');
      const result = await pipeline.runFullPipeline(output, 'unknown', [
        createAgentVote('v1', 0.9),
        createAgentVote('v2', 0.8),
      ]);
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('helper factories', () => {
    it('should create agent output', () => {
      const output = createAgentOutput('agent-1', 'test content', { empirical: 3 });
      expect(output.agentId).toBe('agent-1');
      expect(output.content).toEqual({ type: 'text', value: 'test content' });
      expect(output.classification.empirical).toBe(3);
    });

    it('should create agent vote with clamped values', () => {
      const vote = createAgentVote('agent-1', 1.5, 2.0);
      expect(vote.value).toBe(1);
      expect(vote.confidence).toBe(1);
    });

    it('should create agent vote with defaults', () => {
      const vote = createAgentVote('agent-1', 0.7);
      expect(vote.confidence).toBe(0.8);
      expect(vote.epistemicLevel).toBe(2);
    });
  });

  describe('config', () => {
    it('should use default config', () => {
      expect(DEFAULT_PIPELINE_CONFIG.minParticipants).toBe(2);
      expect(DEFAULT_PIPELINE_CONFIG.trackProvenance).toBe(true);
    });

    it('should accept custom config', () => {
      const custom = createTrustPipeline({ minParticipants: 5 });
      custom.registerAgent('a');
      const output = createAgentOutput('a', 'test');
      const registered = custom.registerOutput(output, 'a');
      // Only 2 votes, but need 5
      const result = custom.runConsensus(registered, [
        createAgentVote('v1', 0.9),
        createAgentVote('v2', 0.8),
      ]);
      expect(result.consensus.status).toBe(PhiConsensusStatus.InsufficientVotes);
    });
  });
});
