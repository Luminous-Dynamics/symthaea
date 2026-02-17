/**
 * @mycelix/sdk Trust Pipeline
 *
 * Orchestration layer connecting provenance, consensus, trust updates,
 * ZK attestation, and cross-domain translation.
 *
 * Pipeline Flow:
 * ```
 * Agent Output
 *     |
 *     v
 * 1. REGISTER OUTPUT (provenance tracking)
 *     |
 *     v
 * 2. PHI-WEIGHTED CONSENSUS (collect votes, weight by trust)
 *     |
 *     v
 * 3. TRUST UPDATE (compute delta, update K-Vector)
 *     |
 *     v
 * 4. GENERATE ATTESTATION (ZK proof)
 *     |
 *     v
 * 5. CROSS-DOMAIN TRANSLATION (optional)
 * ```
 *
 * @packageDocumentation
 * @module trust/pipeline
 */

import {
  type DomainRegistry,
  translateAttestation,
  defaultDomainRegistry,
  DOMAIN_CODE_REVIEW,
} from './domains.js';
// Note: TrustDomain removed as unused
import {
  type ProofStatement,
  type KVectorCommitment,
  type ProverConfig,
  generateSimulatedProof,
  trustAbove,
} from './proofs.js';
import {
  type KVector,
  type AgentOutput,
  type AgentVote,
  DerivationType,
  type ProvenanceNode,
  type ProvenanceChain,
  type PhiConsensusResult,
  PhiConsensusStatus,
  type RegisteredOutput,
  type ConsensusOutcome,
  type TrustUpdate,
  type TrustAttestation,
  type TrustDelta,
  TrustDirection,
  type PipelineConfig,
  type PipelineResult,
  type TranslatedAttestation,
  TrustPipelineError,
  TrustPipelineErrorCode,
  DEFAULT_PIPELINE_CONFIG,
  computeTrustScore,
  applyTrustDelta,
  createUniformKVector,
} from './types.js';
// Note: TrustProof, verifySimulatedProof removed as unused

// ============================================================================
// Agent Registry
// ============================================================================

/**
 * Registered agent with K-Vector
 */
export interface RegisteredAgent {
  /** Agent ID */
  agentId: string;
  /** Current K-Vector */
  kvector: KVector;
  /** Registration timestamp */
  registeredAt: number;
  /** Last activity timestamp */
  lastActivity: number;
}

// ============================================================================
// Provenance Registry
// ============================================================================

/**
 * Registry for provenance tracking
 */
class ProvenanceRegistry {
  private nodes: Map<string, ProvenanceNode> = new Map();
  private childIndex: Map<string, Set<string>> = new Map();

  /**
   * Register a provenance node
   */
  register(node: ProvenanceNode): void {
    this.nodes.set(node.nodeId, node);

    // Index parent-child relationships
    for (const parentId of node.parentIds) {
      if (!this.childIndex.has(parentId)) {
        this.childIndex.set(parentId, new Set());
      }
      this.childIndex.get(parentId)!.add(node.nodeId);
    }
  }

  /**
   * Get a node by ID
   */
  get(nodeId: string): ProvenanceNode | undefined {
    return this.nodes.get(nodeId);
  }

  /**
   * Build a provenance chain from a node
   */
  buildChain(nodeId: string): ProvenanceChain | undefined {
    const node = this.nodes.get(nodeId);
    if (!node) return undefined;

    const nodes: ProvenanceNode[] = [];
    const visited = new Set<string>();
    const queue = [nodeId];
    let minEpistemicLevel = node.epistemicLevel;
    let hasVerifiedRoots = false;
    let derivationSteps = 0;

    while (queue.length > 0) {
      const currentId = queue.shift()!;
      if (visited.has(currentId)) continue;
      visited.add(currentId);

      const currentNode = this.nodes.get(currentId);
      if (!currentNode) continue;

      nodes.push(currentNode);
      minEpistemicLevel = Math.min(minEpistemicLevel, currentNode.epistemicLevel);

      if (currentNode.parentIds.length === 0) {
        // Root node
        hasVerifiedRoots = hasVerifiedRoots || currentNode.epistemicLevel >= 3;
      } else {
        derivationSteps++;
        queue.push(...currentNode.parentIds);
      }
    }

    // Compute chain confidence
    const confidence = nodes.length > 0
      ? nodes.reduce((sum, n) => sum + n.confidence, 0) / nodes.length
      : 0;

    return {
      rootId: nodeId,
      nodes,
      depth: nodes.length,
      minEpistemicLevel,
      confidence,
      hasVerifiedRoots,
      derivationSteps,
    };
  }
}

// ============================================================================
// Trust Pipeline
// ============================================================================

/**
 * Trust Pipeline - orchestrates the full trust flow
 */
export class TrustPipeline {
  private config: PipelineConfig;
  private agents: Map<string, RegisteredAgent> = new Map();
  private provenanceRegistry: ProvenanceRegistry = new ProvenanceRegistry();
  private domainRegistry: DomainRegistry;
  private commitments: Map<string, KVectorCommitment> = new Map();
  private currentTimestamp: number = Date.now();

  constructor(config: Partial<PipelineConfig> = {}) {
    this.config = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    this.domainRegistry = defaultDomainRegistry;
  }

  /**
   * Set current timestamp (for testing)
   */
  setTimestamp(timestamp: number): void {
    this.currentTimestamp = timestamp;
  }

  /**
   * Get current timestamp
   */
  getTimestamp(): number {
    return this.currentTimestamp;
  }

  /**
   * Register an agent
   */
  registerAgent(agentId: string, initialKVector?: KVector): void {
    const kvector = initialKVector ?? createUniformKVector(0.5);
    this.agents.set(agentId, {
      agentId,
      kvector,
      registeredAt: this.currentTimestamp,
      lastActivity: this.currentTimestamp,
    });
  }

  /**
   * Get an agent by ID
   */
  getAgent(agentId: string): RegisteredAgent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Update an agent's K-Vector
   */
  updateAgentKVector(agentId: string, kvector: KVector): void {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.kvector = kvector;
      agent.lastActivity = this.currentTimestamp;
    }
  }

  // ==========================================================================
  // Stage 1: Register Output
  // ==========================================================================

  /**
   * Register an agent output with provenance tracking
   */
  registerOutput(
    output: AgentOutput,
    agentId: string,
    parents: ProvenanceNode[] = [],
    derivation: DerivationType = DerivationType.Original
  ): RegisteredOutput {
    // Create provenance node
    const nodeId = `prov_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const contentHash = this.hashContent(output);

    // Compute epistemic floor
    const parentFloor = parents.length > 0
      ? Math.min(...parents.map(p => p.epistemicLevel))
      : output.classification.empirical;

    const epistemicFloor = Math.min(output.classification.empirical, parentFloor);

    const provenance: ProvenanceNode = {
      nodeId,
      contentHash,
      creator: agentId,
      timestamp: this.currentTimestamp,
      epistemicLevel: output.classification.empirical,
      epistemicFloor,
      confidence: output.classificationConfidence,
      derivationType: derivation,
      parentIds: parents.map(p => p.nodeId),
    };

    // Register in provenance registry
    if (this.config.trackProvenance) {
      for (const parent of parents) {
        this.provenanceRegistry.register(parent);
      }
      this.provenanceRegistry.register(provenance);
    }

    return {
      output,
      provenance,
      parents,
      registeredAt: this.currentTimestamp,
      agentId,
    };
  }

  // ==========================================================================
  // Stage 2: Phi-Weighted Consensus
  // ==========================================================================

  /**
   * Run phi-weighted consensus on a registered output
   */
  runConsensus(
    registered: RegisteredOutput,
    votes: AgentVote[]
  ): ConsensusOutcome {
    if (votes.length < this.config.minParticipants) {
      return {
        output: registered,
        consensus: this.createEmptyConsensus(PhiConsensusStatus.InsufficientVotes),
        votes,
        consensusReached: false,
        effectiveValue: 0,
      };
    }

    // Compute weighted consensus
    let totalWeight = 0;
    let weightedSum = 0;
    let maxPhi = 0;

    for (const vote of votes) {
      const agent = this.agents.get(vote.agentId);
      const trustWeight = agent ? computeTrustScore(agent.kvector) : 0.5;
      const weight = trustWeight * vote.confidence;

      weightedSum += vote.value * weight;
      totalWeight += weight;
      maxPhi = Math.max(maxPhi, vote.confidence);
    }

    const consensusValue = totalWeight > 0 ? weightedSum / totalWeight : 0;

    // Compute dissent
    let dissent = 0;
    for (const vote of votes) {
      dissent += Math.abs(vote.value - consensusValue);
    }
    dissent = votes.length > 0 ? dissent / votes.length : 0;

    const consensusReached = consensusValue >= this.config.phiConsensusThreshold && dissent < 0.3;
    const status = consensusReached
      ? (maxPhi >= 0.7 ? PhiConsensusStatus.Reached : PhiConsensusStatus.ReachedLowCoherence)
      : (dissent >= 0.5 ? PhiConsensusStatus.Deadlock : PhiConsensusStatus.Pending);

    const consensus: PhiConsensusResult = {
      consensusValue,
      confidence: maxPhi,
      participantCount: votes.length,
      totalTrustWeight: totalWeight,
      dissent,
      consensusReached,
      status,
      populationPhi: maxPhi * (1 - dissent),
      harmonicConfidence: consensusValue * (1 - dissent),
    };

    return {
      output: registered,
      consensus,
      votes,
      consensusReached,
      effectiveValue: consensusValue,
    };
  }

  // ==========================================================================
  // Stage 3: Trust Update
  // ==========================================================================

  /**
   * Process consensus result and update agent trust
   */
  processConsensus(outcome: ConsensusOutcome): TrustUpdate {
    const agentId = outcome.output.agentId;
    const agent = this.agents.get(agentId);

    if (!agent) {
      throw new TrustPipelineError(
        TrustPipelineErrorCode.AgentNotFound,
        `Agent not found: ${agentId}`
      );
    }

    const previousKVector = { ...agent.kvector };
    const trustDelta = this.computeTrustDelta(outcome);
    const updatedKVector = applyTrustDelta(previousKVector, trustDelta);

    // Update agent
    agent.kvector = updatedKVector;
    agent.lastActivity = this.currentTimestamp;

    const newTrustScore = computeTrustScore(updatedKVector);

    return {
      outcome,
      agentId,
      previousKVector,
      updatedKVector,
      trustDelta,
      newTrustScore,
    };
  }

  /**
   * Compute trust delta from consensus outcome
   */
  private computeTrustDelta(outcome: ConsensusOutcome): TrustDelta {
    const value = outcome.effectiveValue;
    const reached = outcome.consensusReached;

    // Determine direction
    let direction: TrustDirection;
    if (value >= this.config.minConsensusForTrust && reached) {
      direction = TrustDirection.Increase;
    } else if (value < 0.3) {
      direction = TrustDirection.Decrease;
    } else {
      direction = TrustDirection.Unchanged;
    }

    // Epistemic multiplier
    const epistemicMultiplier = [0.3, 0.7, 1.0, 1.2, 1.5][outcome.output.output.classification.empirical] ?? 1.0;

    // Base delta
    let baseDelta: number;
    switch (direction) {
      case TrustDirection.Increase:
        baseDelta = (value - this.config.minConsensusForTrust) * 0.2;
        break;
      case TrustDirection.Decrease:
        baseDelta = -(this.config.minConsensusForTrust - value) * 0.1;
        break;
      default:
        baseDelta = 0;
    }

    const adjustedDelta = Math.max(
      -this.config.maxTrustDelta,
      Math.min(this.config.maxTrustDelta, baseDelta * epistemicMultiplier)
    );

    const reason = direction === TrustDirection.Increase
      ? `Consensus reached (${Math.round(value * 100)}%) on E${outcome.output.output.classification.empirical} output`
      : direction === TrustDirection.Decrease
        ? `Low consensus (${Math.round(value * 100)}%) on output`
        : 'Consensus inconclusive';

    return {
      reputationDelta: adjustedDelta * 0.4,
      performanceDelta: adjustedDelta * 0.3,
      integrityDelta: adjustedDelta * 0.2,
      historicalDelta: adjustedDelta * 0.1,
      direction,
      reason,
    };
  }

  // ==========================================================================
  // Stage 4: Generate Attestation
  // ==========================================================================

  /**
   * Generate ZK attestation for trust update
   */
  generateAttestation(
    update: TrustUpdate,
    statement?: ProofStatement
  ): TrustAttestation {
    const proofStatement = statement ?? trustAbove(0.5);

    const proverConfig: ProverConfig = {
      agentId: update.agentId,
      timestamp: this.currentTimestamp,
      simulationMode: true,
    };

    const proof = generateSimulatedProof(
      update.updatedKVector,
      proofStatement,
      proverConfig
    );

    // Store commitment
    this.commitments.set(proof.commitment.hash, proof.commitment);

    // Build provenance chain
    const provenanceChain = this.config.trackProvenance
      ? this.provenanceRegistry.buildChain(update.outcome.output.provenance.nodeId)
      : undefined;

    return {
      update,
      proofId: proof.proofId,
      kvectorCommitment: proof.commitment.hash,
      provenanceChain,
      attestedAt: this.currentTimestamp,
      isVerified: true,
    };
  }

  // ==========================================================================
  // Stage 5: Cross-Domain Translation
  // ==========================================================================

  /**
   * Translate attestation for a target domain
   */
  translateForDomain(
    attestation: TrustAttestation,
    targetDomainId: string,
    sourceDomainId?: string
  ): TranslatedAttestation {
    const sourceDomain = sourceDomainId
      ? this.domainRegistry.get(sourceDomainId) ?? DOMAIN_CODE_REVIEW
      : DOMAIN_CODE_REVIEW;

    const targetDomain = this.domainRegistry.get(targetDomainId);
    if (!targetDomain) {
      throw new TrustPipelineError(
        TrustPipelineErrorCode.DomainNotFound,
        `Domain not found: ${targetDomainId}`
      );
    }

    return translateAttestation(attestation, sourceDomain, targetDomain);
  }

  // ==========================================================================
  // Full Pipeline
  // ==========================================================================

  /**
   * Run the complete pipeline from output to attestation
   */
  async runFullPipeline(
    output: AgentOutput,
    agentId: string,
    votes: AgentVote[],
    options: {
      parents?: ProvenanceNode[];
      derivation?: DerivationType;
      proofStatement?: ProofStatement;
      targetDomainId?: string;
      sourceDomainId?: string;
    } = {}
  ): Promise<PipelineResult> {
    const startTime = Date.now();

    try {
      // Stage 1: Register
      const registered = this.registerOutput(
        output,
        agentId,
        options.parents ?? [],
        options.derivation ?? DerivationType.Original
      );

      // Stage 2: Consensus
      const consensus = this.runConsensus(registered, votes);

      // Stage 3: Trust update
      const update = this.processConsensus(consensus);

      // Stage 4: Attestation
      const attestation = this.generateAttestation(update, options.proofStatement);

      // Stage 5: Translation (optional)
      let translatedAttestation: TranslatedAttestation | undefined;
      if (options.targetDomainId) {
        translatedAttestation = this.translateForDomain(
          attestation,
          options.targetDomainId,
          options.sourceDomainId
        );
      }

      return {
        success: true,
        attestation,
        translatedAttestation,
        executionTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        executionTimeMs: Date.now() - startTime,
      };
    }
  }

  // ==========================================================================
  // Verification
  // ==========================================================================

  /**
   * Verify an attestation
   */
  verifyAttestation(attestation: TrustAttestation): boolean {
    // Check commitment exists
    const commitment = this.commitments.get(attestation.kvectorCommitment);
    if (!commitment) {
      return false;
    }

    return attestation.isVerified;
  }

  /**
   * Verify a translated attestation
   */
  verifyTranslatedAttestation(attestation: TranslatedAttestation): boolean {
    return this.verifyAttestation(attestation.original);
  }

  // ==========================================================================
  // Helpers
  // ==========================================================================

  /**
   * Create empty consensus result
   */
  private createEmptyConsensus(status: PhiConsensusStatus): PhiConsensusResult {
    return {
      consensusValue: 0,
      confidence: 0,
      participantCount: 0,
      totalTrustWeight: 0,
      dissent: 0,
      consensusReached: false,
      status,
      populationPhi: 0,
      harmonicConfidence: 0,
    };
  }

  /**
   * Hash content for provenance
   */
  private hashContent(output: AgentOutput): string {
    // Simple hash for now - in production use SHA3-256
    const str = JSON.stringify(output);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new trust pipeline with default configuration
 */
export function createTrustPipeline(config?: Partial<PipelineConfig>): TrustPipeline {
  return new TrustPipeline(config);
}

/**
 * Create an agent output
 */
export function createAgentOutput(
  agentId: string,
  content: string,
  classification: {
    empirical?: number;
    normative?: number;
    materiality?: number;
    harmonic?: number;
  } = {}
): AgentOutput {
  return {
    outputId: `output_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    agentId,
    content: { type: 'text', value: content },
    classification: {
      empirical: classification.empirical ?? 2,
      normative: classification.normative ?? 1,
      materiality: classification.materiality ?? 1,
      harmonic: classification.harmonic ?? 1,
    },
    classificationConfidence: 0.8,
    timestamp: Date.now(),
    hasProof: false,
    contextReferences: [],
  };
}

/**
 * Create an agent vote
 */
export function createAgentVote(
  agentId: string,
  value: number,
  confidence = 0.8
): AgentVote {
  return {
    agentId,
    value: Math.max(0, Math.min(1, value)),
    confidence: Math.max(0, Math.min(1, confidence)),
    epistemicLevel: 2,
    timestamp: Date.now(),
  };
}
