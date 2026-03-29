// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * HDC Genetics Integration Module
 *
 * TypeScript SDK bindings for the hdc_genetics Holochain zome.
 * Enables privacy-preserving genetic similarity searches using
 * Hyperdimensional Computing (HDC).
 *
 * @module @mycelix/sdk/integrations/genetics
 */

import type { ActionHash, AgentPubKey } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

/**
 * Types of genetic data that can be encoded
 */
export type GeneticEncodingType =
  | 'DnaSequence'
  | 'SnpPanel'
  | 'HlaTyping'
  | 'Pharmacogenomics'
  | 'DiseaseRisk'
  | 'Ancestry'
  | { GenePanel: string };

/**
 * Similarity metrics for comparing hypervectors
 */
export type SimilarityMetric = 'Cosine' | 'Hamming' | 'Jaccard';

/**
 * Purpose of a genetic similarity query (for audit trail)
 */
export type QueryPurpose =
  | 'OrganDonorMatching'
  | 'HlaMatching'
  | 'PharmacogenomicPrediction'
  | 'DiseaseRiskAssessment'
  | 'ClinicalTrialMatching'
  | { Research: string }
  | { Other: string };

/**
 * Metadata about the genetic data source
 */
export interface GeneticSourceMetadata {
  sourceSystem: string;
  testDate?: number;
  sequencingMethod?: string;
  qualityScore?: number;
  consentHash?: ActionHash;
}

/**
 * A hypervector representing encoded genetic data
 */
export interface GeneticHypervector {
  vectorId: string;
  patientHash: ActionHash;
  data: Uint8Array;
  encodingType: GeneticEncodingType;
  kmerLength: number;
  kmerCount: number;
  createdAt: number;
  sourceMetadata: GeneticSourceMetadata;
}

/**
 * Result of a similarity query
 */
export interface GeneticSimilarityResult {
  resultId: string;
  queryVectorHash: ActionHash;
  targetVectorHash: ActionHash;
  similarityScore: number;
  similarityMetric: SimilarityMetric;
  queryPurpose: QueryPurpose;
  queriedAt: number;
  queriedBy: AgentPubKey;
}

/**
 * Result from batch similarity search
 */
export interface SimilaritySearchResult {
  vectorHash: ActionHash;
  patientHash: ActionHash;
  similarityScore: number;
}

/**
 * A bundled (aggregated) hypervector from multiple sources
 */
export interface BundledGeneticVector {
  bundleId: string;
  patientHash: ActionHash;
  data: Uint8Array;
  sourceVectorHashes: ActionHash[];
  weights?: number[];
  bundledAt: number;
}

/**
 * Codebook for consistent k-mer encoding
 */
export interface GeneticCodebook {
  codebookId: string;
  kmerLength: number;
  seed: Uint8Array;
  version: number;
  createdAt: number;
  description?: string;
}

// ============================================================================
// Input Types
// ============================================================================

export interface EncodeDnaSequenceInput {
  patientHash: ActionHash;
  sequence: string;
  kmerLength?: number;
  sourceMetadata: GeneticSourceMetadata;
}

export interface EncodeSnpPanelInput {
  patientHash: ActionHash;
  snps: Array<[string, string]>; // [rsID, allele]
  sourceMetadata: GeneticSourceMetadata;
}

export interface EncodeHlaTypingInput {
  patientHash: ActionHash;
  hlaTypes: string[]; // e.g., ["A*02:01", "B*07:02"]
  sourceMetadata: GeneticSourceMetadata;
}

export interface SimilarityQueryInput {
  queryVectorHash: ActionHash;
  targetVectorHash: ActionHash;
  metric?: SimilarityMetric;
  purpose: QueryPurpose;
}

export interface BatchSimilaritySearchInput {
  queryVectorHash: ActionHash;
  encodingType: GeneticEncodingType;
  minSimilarity: number;
  limit: number;
  purpose: QueryPurpose;
}

export interface BundleVectorsInput {
  patientHash: ActionHash;
  vectorHashes: ActionHash[];
  weights?: number[];
}

// ============================================================================
// Zome Callable Interface
// ============================================================================

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Client Classes
// ============================================================================

const HEALTH_ROLE = 'health';
const GENETICS_ZOME = 'hdc_genetics';

/**
 * Client for encoding genetic data as hypervectors
 */
export class GeneticEncodingClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Encode a DNA sequence as a hypervector
   *
   * Uses k-mer encoding to create a privacy-preserving representation
   * that can be compared without exposing the raw sequence.
   */
  async encodeDnaSequence(input: EncodeDnaSequenceInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'encode_dna_sequence',
      payload: {
        patient_hash: input.patientHash,
        sequence: input.sequence,
        kmer_length: input.kmerLength,
        source_metadata: {
          source_system: input.sourceMetadata.sourceSystem,
          test_date: input.sourceMetadata.testDate,
          sequencing_method: input.sourceMetadata.sequencingMethod,
          quality_score: input.sourceMetadata.qualityScore,
          consent_hash: input.sourceMetadata.consentHash,
        },
      },
    }) as Promise<ActionHash>;
  }

  /**
   * Encode a SNP panel as a hypervector
   *
   * SNPs are represented as rsID:allele pairs (e.g., rs1234:A)
   */
  async encodeSnpPanel(input: EncodeSnpPanelInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'encode_snp_panel',
      payload: {
        patient_hash: input.patientHash,
        snps: input.snps,
        source_metadata: {
          source_system: input.sourceMetadata.sourceSystem,
          test_date: input.sourceMetadata.testDate,
          sequencing_method: input.sourceMetadata.sequencingMethod,
          quality_score: input.sourceMetadata.qualityScore,
          consent_hash: input.sourceMetadata.consentHash,
        },
      },
    }) as Promise<ActionHash>;
  }

  /**
   * Encode HLA typing as a hypervector
   *
   * HLA types are critical for transplant matching.
   */
  async encodeHlaTyping(input: EncodeHlaTypingInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'encode_hla_typing',
      payload: {
        patient_hash: input.patientHash,
        hla_types: input.hlaTypes,
        source_metadata: {
          source_system: input.sourceMetadata.sourceSystem,
          test_date: input.sourceMetadata.testDate,
          sequencing_method: input.sourceMetadata.sequencingMethod,
          quality_score: input.sourceMetadata.qualityScore,
          consent_hash: input.sourceMetadata.consentHash,
        },
      },
    }) as Promise<ActionHash>;
  }

  /**
   * Get all genetic vectors for a patient
   */
  async getPatientVectors(patientHash: ActionHash): Promise<GeneticHypervector[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'get_patient_genetic_vectors',
      payload: patientHash,
    }) as Promise<GeneticHypervector[]>;
  }
}

/**
 * Client for genetic similarity queries
 */
export class GeneticSimilarityClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Calculate similarity between two genetic hypervectors
   *
   * Returns a similarity score between 0 and 1.
   */
  async calculateSimilarity(input: SimilarityQueryInput): Promise<GeneticSimilarityResult> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'calculate_similarity',
      payload: {
        query_vector_hash: input.queryVectorHash,
        target_vector_hash: input.targetVectorHash,
        metric: input.metric,
        purpose: input.purpose,
      },
    }) as Promise<GeneticSimilarityResult>;
  }

  /**
   * Search for similar genetic profiles
   *
   * Returns profiles above the minimum similarity threshold,
   * sorted by similarity score (descending).
   */
  async searchSimilar(input: BatchSimilaritySearchInput): Promise<SimilaritySearchResult[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'search_similar_genetics',
      payload: {
        query_vector_hash: input.queryVectorHash,
        encoding_type: input.encodingType,
        min_similarity: input.minSimilarity,
        limit: input.limit,
        purpose: input.purpose,
      },
    }) as Promise<SimilaritySearchResult[]>;
  }

  /**
   * Find potential HLA matches for transplant
   *
   * Specialized search for organ/tissue transplant matching.
   */
  async findHlaMatches(
    patientVectorHash: ActionHash,
    minSimilarity: number = 0.8,
    limit: number = 10
  ): Promise<SimilaritySearchResult[]> {
    return this.searchSimilar({
      queryVectorHash: patientVectorHash,
      encodingType: 'HlaTyping',
      minSimilarity,
      limit,
      purpose: 'HlaMatching',
    });
  }

  /**
   * Find patients with similar pharmacogenomic profiles
   *
   * Useful for drug response prediction based on similar genetics.
   */
  async findPharmacogenomicMatches(
    patientVectorHash: ActionHash,
    minSimilarity: number = 0.7,
    limit: number = 20
  ): Promise<SimilaritySearchResult[]> {
    return this.searchSimilar({
      queryVectorHash: patientVectorHash,
      encodingType: 'Pharmacogenomics',
      minSimilarity,
      limit,
      purpose: 'PharmacogenomicPrediction',
    });
  }

  /**
   * Find patients for clinical trial matching
   *
   * Identifies genetically similar candidates for research studies.
   */
  async findTrialCandidates(
    referenceVectorHash: ActionHash,
    encodingType: GeneticEncodingType,
    minSimilarity: number = 0.6,
    limit: number = 50
  ): Promise<SimilaritySearchResult[]> {
    return this.searchSimilar({
      queryVectorHash: referenceVectorHash,
      encodingType,
      minSimilarity,
      limit,
      purpose: 'ClinicalTrialMatching',
    });
  }
}

/**
 * Client for vector bundling operations
 */
export class GeneticBundlingClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Bundle multiple genetic vectors into one
   *
   * Useful for combining multiple genetic data sources
   * (e.g., SNP panel + HLA typing) into a single representation.
   */
  async bundleVectors(input: BundleVectorsInput): Promise<ActionHash> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'bundle_genetic_vectors',
      payload: {
        patient_hash: input.patientHash,
        vector_hashes: input.vectorHashes,
        weights: input.weights,
      },
    }) as Promise<ActionHash>;
  }
}

/**
 * Client for codebook management
 */
export class GeneticCodebookClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Initialize the default codebook
   *
   * Called automatically if not present, but can be called
   * explicitly to ensure codebook exists before encoding.
   */
  async initCodebook(): Promise<ActionHash> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: GENETICS_ZOME,
      fn_name: 'init_codebook',
      payload: null,
    }) as Promise<ActionHash>;
  }
}

// ============================================================================
// Unified Genetics Bridge
// ============================================================================

/**
 * Unified interface for HDC Genetics operations
 */
export class GeneticsBridge {
  readonly encoding: GeneticEncodingClient;
  readonly similarity: GeneticSimilarityClient;
  readonly bundling: GeneticBundlingClient;
  readonly codebook: GeneticCodebookClient;

  constructor(client: ZomeCallable) {
    this.encoding = new GeneticEncodingClient(client);
    this.similarity = new GeneticSimilarityClient(client);
    this.bundling = new GeneticBundlingClient(client);
    this.codebook = new GeneticCodebookClient(client);
  }

  /**
   * Complete workflow: Encode DNA and find similar patients
   */
  async encodeAndFindSimilar(
    patientHash: ActionHash,
    sequence: string,
    sourceMetadata: GeneticSourceMetadata,
    minSimilarity: number = 0.7,
    limit: number = 10
  ): Promise<{
    vectorHash: ActionHash;
    similarPatients: SimilaritySearchResult[];
  }> {
    // Encode the sequence
    const vectorHash = await this.encoding.encodeDnaSequence({
      patientHash,
      sequence,
      sourceMetadata,
    });

    // Search for similar patients
    const similarPatients = await this.similarity.searchSimilar({
      queryVectorHash: vectorHash,
      encodingType: 'DnaSequence',
      minSimilarity,
      limit,
      purpose: { Research: 'similarity_search' },
    });

    return { vectorHash, similarPatients };
  }

  /**
   * Complete workflow: HLA typing and donor matching
   */
  async findOrganDonorMatches(
    recipientPatientHash: ActionHash,
    recipientHlaTypes: string[],
    sourceMetadata: GeneticSourceMetadata,
    minSimilarity: number = 0.85
  ): Promise<{
    recipientVectorHash: ActionHash;
    potentialDonors: SimilaritySearchResult[];
  }> {
    // Encode recipient HLA
    const recipientVectorHash = await this.encoding.encodeHlaTyping({
      patientHash: recipientPatientHash,
      hlaTypes: recipientHlaTypes,
      sourceMetadata,
    });

    // Find potential donors
    const potentialDonors = await this.similarity.findHlaMatches(
      recipientVectorHash,
      minSimilarity,
      20
    );

    return { recipientVectorHash, potentialDonors };
  }

  /**
   * Get genetic profile summary for a patient
   */
  async getPatientGeneticProfile(patientHash: ActionHash): Promise<{
    vectors: GeneticHypervector[];
    byType: Record<string, GeneticHypervector[]>;
    totalEncodings: number;
  }> {
    const vectors = await this.encoding.getPatientVectors(patientHash);

    const byType: Record<string, GeneticHypervector[]> = {};
    for (const vector of vectors) {
      const typeKey = typeof vector.encodingType === 'string'
        ? vector.encodingType
        : 'GenePanel';
      if (!byType[typeKey]) {
        byType[typeKey] = [];
      }
      byType[typeKey].push(vector);
    }

    return {
      vectors,
      byType,
      totalEncodings: vectors.length,
    };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Genetics bridge instance
 */
export function createGeneticsBridge(client: ZomeCallable): GeneticsBridge {
  return new GeneticsBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Interpret similarity score for HLA matching
 */
export function interpretHlaSimilarity(score: number): {
  matchQuality: 'Excellent' | 'Good' | 'Moderate' | 'Poor' | 'Unsuitable';
  description: string;
  recommendation: string;
} {
  if (score >= 0.95) {
    return {
      matchQuality: 'Excellent',
      description: 'Near-perfect HLA match',
      recommendation: 'Highly suitable for transplant',
    };
  } else if (score >= 0.85) {
    return {
      matchQuality: 'Good',
      description: 'Strong HLA compatibility',
      recommendation: 'Suitable for transplant with standard immunosuppression',
    };
  } else if (score >= 0.70) {
    return {
      matchQuality: 'Moderate',
      description: 'Partial HLA match',
      recommendation: 'May require enhanced immunosuppression',
    };
  } else if (score >= 0.50) {
    return {
      matchQuality: 'Poor',
      description: 'Significant HLA mismatch',
      recommendation: 'High risk of rejection, consider alternatives',
    };
  } else {
    return {
      matchQuality: 'Unsuitable',
      description: 'Major HLA incompatibility',
      recommendation: 'Not recommended for transplant',
    };
  }
}

/**
 * Calculate confidence level based on k-mer count
 */
export function calculateEncodingConfidence(kmerCount: number): {
  confidence: 'High' | 'Medium' | 'Low';
  percentage: number;
  description: string;
} {
  // More k-mers generally means better encoding fidelity
  if (kmerCount >= 10000) {
    return {
      confidence: 'High',
      percentage: 95,
      description: 'Large sequence with comprehensive k-mer coverage',
    };
  } else if (kmerCount >= 1000) {
    return {
      confidence: 'Medium',
      percentage: 75,
      description: 'Moderate sequence with good k-mer representation',
    };
  } else {
    return {
      confidence: 'Low',
      percentage: 50,
      description: 'Small sequence, similarity scores may have higher variance',
    };
  }
}

// ============================================================================
// Exports
// ============================================================================

export default {
  GeneticsBridge,
  GeneticEncodingClient,
  GeneticSimilarityClient,
  GeneticBundlingClient,
  GeneticCodebookClient,
  createGeneticsBridge,
  interpretHlaSimilarity,
  calculateEncodingConfidence,
};
