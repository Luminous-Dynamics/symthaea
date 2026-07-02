// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge SDK - AI-Assisted Analysis Module
 *
 * Provides intelligent tools for claim analysis:
 * - Auto-classification into E-N-M dimensions
 * - Evidence extraction from text
 * - Contradiction detection
 * - Similarity analysis and clustering
 *
 * @example
 * ```typescript
 * import {
 *   EpistemicClassifier,
 *   EvidenceExtractor,
 *   ContradictionDetector,
 *   SimilarityAnalyzer,
 * } from '@mycelix/knowledge-sdk/ai';
 *
 * // Classify a claim
 * const classifier = new EpistemicClassifier();
 * const classification = await classifier.classify("The Earth is 4.5 billion years old");
 *
 * // Extract evidence from text
 * const extractor = new EvidenceExtractor();
 * const evidence = await extractor.extract(articleText);
 *
 * // Check for contradictions
 * const detector = new ContradictionDetector();
 * const result = await detector.detect(claim1, claim2);
 *
 * // Analyze similarity
 * const analyzer = new SimilarityAnalyzer();
 * const similarity = await analyzer.compare(claim1, claim2);
 * ```
 */

// Epistemic Classifier
export { EpistemicClassifier } from './classifier';
export type {
  EpistemicClassification,
  ClassificationResult,
  ClassifierOptions,
} from './classifier';

// Evidence Extractor
export { EvidenceExtractor } from './evidence-extractor';
export type {
  ExtractedEvidence,
  SourceReference,
  StatisticClaim,
  QuotedStatement,
  TemporalReference,
  CausalClaim,
  ExtractionMetadata,
  ExtractorOptions,
} from './evidence-extractor';

// Contradiction Detector
export { ContradictionDetector } from './contradiction-detector';
export type {
  ContradictionResult,
  ContradictionType,
  ConflictDetail,
  DetectorOptions,
} from './contradiction-detector';

// Similarity Analyzer
export { SimilarityAnalyzer } from './similarity-analyzer';
export type {
  SimilarityResult,
  ClusterResult,
  SimilarityMatrix,
  AnalyzerOptions,
} from './similarity-analyzer';

// ============================================================================
// Convenience Functions
// ============================================================================

import { EpistemicClassifier, ClassificationResult } from './classifier';
import { EvidenceExtractor, ExtractedEvidence } from './evidence-extractor';
import { ContradictionDetector, ContradictionResult } from './contradiction-detector';
import { SimilarityAnalyzer, SimilarityResult } from './similarity-analyzer';

// Singleton instances for quick usage
let _classifier: EpistemicClassifier | null = null;
let _extractor: EvidenceExtractor | null = null;
let _detector: ContradictionDetector | null = null;
let _analyzer: SimilarityAnalyzer | null = null;

/**
 * Quick classification of a claim (uses shared instance)
 */
export async function classifyClaim(text: string): Promise<ClassificationResult> {
  if (!_classifier) _classifier = new EpistemicClassifier();
  return _classifier.classify(text);
}

/**
 * Quick evidence extraction (uses shared instance)
 */
export async function extractEvidence(text: string): Promise<ExtractedEvidence> {
  if (!_extractor) _extractor = new EvidenceExtractor();
  return _extractor.extract(text);
}

/**
 * Quick contradiction check (uses shared instance)
 */
export async function checkContradiction(claim1: string, claim2: string): Promise<ContradictionResult> {
  if (!_detector) _detector = new ContradictionDetector();
  return _detector.detect(claim1, claim2);
}

/**
 * Quick similarity comparison (uses shared instance)
 */
export async function compareClaims(claim1: string, claim2: string): Promise<SimilarityResult> {
  if (!_analyzer) _analyzer = new SimilarityAnalyzer();
  return _analyzer.compare(claim1, claim2);
}

/**
 * Comprehensive claim analysis
 */
export interface ComprehensiveAnalysis {
  classification: ClassificationResult;
  evidence: ExtractedEvidence;
  similarityToCorpus?: Array<{ claim: string; similarity: number }>;
}

/**
 * Perform comprehensive analysis on a claim
 */
export async function analyzeClaim(
  text: string,
  corpus?: string[]
): Promise<ComprehensiveAnalysis> {
  const [classification, evidence] = await Promise.all([
    classifyClaim(text),
    extractEvidence(text),
  ]);

  let similarityToCorpus: Array<{ claim: string; similarity: number }> | undefined;

  if (corpus && corpus.length > 0) {
    if (!_analyzer) _analyzer = new SimilarityAnalyzer();
    similarityToCorpus = await _analyzer.findMostSimilar(text, corpus, 5);
  }

  return {
    classification,
    evidence,
    similarityToCorpus,
  };
}

/**
 * Check consistency of a set of claims
 */
export async function checkConsistency(claims: string[]): Promise<{
  totalClaims: number;
  contradictionCount: number;
  consistencyScore: number;
  contradictions: Array<{
    claim1: string;
    claim2: string;
    type: string;
    confidence: number;
  }>;
}> {
  if (!_detector) _detector = new ContradictionDetector();

  const summary = await _detector.summarizeContradictions(claims);

  return {
    totalClaims: summary.totalClaims,
    contradictionCount: summary.contradictionCount,
    consistencyScore: summary.consistency,
    contradictions: summary.contradictionPairs.map(p => ({
      claim1: p.claim1,
      claim2: p.claim2,
      type: p.result.type || 'unknown',
      confidence: p.result.confidence,
    })),
  };
}
