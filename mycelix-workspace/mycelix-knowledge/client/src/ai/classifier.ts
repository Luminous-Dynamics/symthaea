// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI-Assisted Epistemic Classifier
 *
 * Automatically classifies claims into Empirical-Normative-Mythic dimensions
 * using linguistic analysis and pattern matching.
 *
 * @example
 * ```typescript
 * const classifier = new EpistemicClassifier();
 * const result = await classifier.classify("The Earth is approximately 4.5 billion years old");
 * console.log(result); // { empirical: 0.95, normative: 0.02, mythic: 0.03 }
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface EpistemicClassification {
  empirical: number;  // 0-1: verifiable through observation
  normative: number;  // 0-1: value-based, ethical
  mythic: number;     // 0-1: meaning-making, narrative
}

export interface ClassificationResult extends EpistemicClassification {
  confidence: number;
  reasoning: string[];
  dominantType: 'empirical' | 'normative' | 'mythic' | 'mixed';
  suggestedTags: string[];
}

export interface ClassifierOptions {
  /** Minimum confidence threshold */
  minConfidence?: number;
  /** Include reasoning explanations */
  includeReasoning?: boolean;
  /** Language of the text */
  language?: string;
}

// ============================================================================
// Linguistic Patterns
// ============================================================================

const EMPIRICAL_PATTERNS = {
  keywords: [
    'measure', 'observe', 'experiment', 'data', 'evidence', 'fact', 'study',
    'research', 'statistic', 'percent', 'rate', 'quantity', 'number',
    'test', 'verify', 'demonstrate', 'prove', 'show', 'indicate',
    'temperature', 'distance', 'weight', 'speed', 'size', 'amount',
  ],
  phrases: [
    /according to (?:research|studies|data)/i,
    /scientists (?:have found|discovered|showed)/i,
    /\d+(?:\.\d+)?%/,  // percentages
    /\d+ (?:million|billion|thousand)/i,  // quantities
    /(?:in|during) \d{4}/,  // specific dates
    /(?:studies|research) show(?:s)?/i,
    /evidence (?:suggests|indicates|shows)/i,
    /measured at/i,
    /the (?:average|mean|median)/i,
  ],
  negativeIndicators: [
    'should', 'ought', 'must', 'believe', 'feel', 'hope', 'wish',
    'sacred', 'divine', 'spiritual', 'eternal', 'soul',
  ],
};

const NORMATIVE_PATTERNS = {
  keywords: [
    'should', 'ought', 'must', 'right', 'wrong', 'good', 'bad', 'evil',
    'moral', 'ethical', 'just', 'unjust', 'fair', 'unfair', 'virtue',
    'value', 'duty', 'obligation', 'responsibility', 'deserve',
    'appropriate', 'inappropriate', 'acceptable', 'unacceptable',
  ],
  phrases: [
    /(?:is|are) (?:right|wrong|good|bad|evil)/i,
    /should (?:be|have|not)/i,
    /ought to/i,
    /(?:it is|we) have a (?:duty|responsibility|obligation)/i,
    /morally (?:acceptable|wrong|right)/i,
    /(?:fair|unfair) (?:to|that)/i,
    /(?:virtuous|vicious) (?:to|behavior)/i,
    /ethical(?:ly)? (?:required|forbidden|permitted)/i,
  ],
  negativeIndicators: [
    'measured', 'observed', 'data shows', 'experiment',
    'sacred', 'divine', 'creation', 'destiny',
  ],
};

const MYTHIC_PATTERNS = {
  keywords: [
    'sacred', 'divine', 'spiritual', 'soul', 'eternal', 'destiny',
    'meaning', 'purpose', 'transcendent', 'holy', 'blessed', 'cosmic',
    'creation', 'myth', 'legend', 'archetype', 'symbol', 'ritual',
    'faith', 'believe', 'revelation', 'prophecy', 'miracle',
  ],
  phrases: [
    /(?:the|a) (?:sacred|divine|holy)/i,
    /meaning of (?:life|existence)/i,
    /spiritual (?:truth|reality|dimension)/i,
    /(?:our|the) destiny/i,
    /eternal (?:truth|life|soul)/i,
    /(?:it is|was) written/i,
    /(?:the|a) creation (?:story|myth)/i,
    /transcendent (?:reality|truth)/i,
    /cosmic (?:order|harmony|purpose)/i,
  ],
  negativeIndicators: [
    'data', 'research', 'study', 'experiment', 'measure',
    'policy', 'law', 'regulation', 'should',
  ],
};

// ============================================================================
// Classifier Implementation
// ============================================================================

export class EpistemicClassifier {
  private options: Required<ClassifierOptions>;

  constructor(options: ClassifierOptions = {}) {
    this.options = {
      minConfidence: options.minConfidence ?? 0.6,
      includeReasoning: options.includeReasoning ?? true,
      language: options.language ?? 'en',
    };
  }

  /**
   * Classify a claim into E-N-M dimensions
   */
  async classify(text: string): Promise<ClassificationResult> {
    const normalized = this.normalizeText(text);
    const reasoning: string[] = [];

    // Calculate scores for each dimension
    const empiricalScore = this.calculateDimensionScore(normalized, EMPIRICAL_PATTERNS, reasoning, 'Empirical');
    const normativeScore = this.calculateDimensionScore(normalized, NORMATIVE_PATTERNS, reasoning, 'Normative');
    const mythicScore = this.calculateDimensionScore(normalized, MYTHIC_PATTERNS, reasoning, 'Mythic');

    // Normalize scores to sum to 1
    const total = empiricalScore + normativeScore + mythicScore;
    const classification: EpistemicClassification = total > 0
      ? {
          empirical: empiricalScore / total,
          normative: normativeScore / total,
          mythic: mythicScore / total,
        }
      : { empirical: 0.33, normative: 0.33, mythic: 0.34 };

    // Calculate confidence
    const confidence = this.calculateConfidence(classification, total);

    // Determine dominant type
    const dominantType = this.getDominantType(classification);

    // Generate suggested tags
    const suggestedTags = this.generateTags(text, classification, dominantType);

    return {
      ...classification,
      confidence,
      reasoning: this.options.includeReasoning ? reasoning : [],
      dominantType,
      suggestedTags,
    };
  }

  /**
   * Batch classify multiple claims
   */
  async classifyBatch(texts: string[]): Promise<ClassificationResult[]> {
    return Promise.all(texts.map(text => this.classify(text)));
  }

  /**
   * Get classification explanation in natural language
   */
  explain(result: ClassificationResult): string {
    const { empirical, normative, mythic, dominantType, confidence } = result;

    const percentages = {
      empirical: Math.round(empirical * 100),
      normative: Math.round(normative * 100),
      mythic: Math.round(mythic * 100),
    };

    let explanation = `This claim is primarily ${dominantType} `;
    explanation += `(E: ${percentages.empirical}%, N: ${percentages.normative}%, M: ${percentages.mythic}%). `;

    if (dominantType === 'empirical') {
      explanation += 'It makes factual assertions that can be verified through observation or measurement.';
    } else if (dominantType === 'normative') {
      explanation += 'It expresses value judgments or makes claims about what should or ought to be.';
    } else if (dominantType === 'mythic') {
      explanation += 'It relates to meaning-making, spiritual significance, or transcendent truths.';
    } else {
      explanation += 'It contains a balanced mix of empirical facts, value judgments, and meaning-making elements.';
    }

    explanation += ` Classification confidence: ${Math.round(confidence * 100)}%.`;

    return explanation;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private normalizeText(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private calculateDimensionScore(
    text: string,
    patterns: typeof EMPIRICAL_PATTERNS,
    reasoning: string[],
    dimension: string
  ): number {
    let score = 0;

    // Check keywords
    const keywordMatches = patterns.keywords.filter(kw => text.includes(kw));
    if (keywordMatches.length > 0) {
      score += keywordMatches.length * 0.1;
      if (this.options.includeReasoning && keywordMatches.length > 0) {
        reasoning.push(`${dimension}: Found keywords: ${keywordMatches.slice(0, 3).join(', ')}`);
      }
    }

    // Check phrases
    const phraseMatches = patterns.phrases.filter(p => p.test(text));
    if (phraseMatches.length > 0) {
      score += phraseMatches.length * 0.2;
      if (this.options.includeReasoning) {
        reasoning.push(`${dimension}: Matched ${phraseMatches.length} characteristic phrases`);
      }
    }

    // Apply negative indicators
    const negativeMatches = patterns.negativeIndicators.filter(ni => text.includes(ni));
    if (negativeMatches.length > 0) {
      score *= Math.max(0.3, 1 - negativeMatches.length * 0.15);
    }

    return Math.min(1, score);
  }

  private calculateConfidence(classification: EpistemicClassification, rawTotal: number): number {
    // Higher confidence when:
    // 1. One dimension clearly dominates
    // 2. Total raw score was higher (more pattern matches)
    const values = Object.values(classification);
    const max = Math.max(...values);
    const variance = values.reduce((acc, v) => acc + Math.pow(v - 0.33, 2), 0) / 3;

    const dominanceScore = max > 0.5 ? (max - 0.33) / 0.67 : 0;
    const strengthScore = Math.min(1, rawTotal / 0.5);

    return Math.min(0.99, (dominanceScore * 0.6 + strengthScore * 0.4));
  }

  private getDominantType(classification: EpistemicClassification): ClassificationResult['dominantType'] {
    const { empirical, normative, mythic } = classification;
    const threshold = 0.4;

    if (empirical >= threshold && empirical > normative && empirical > mythic) {
      return 'empirical';
    }
    if (normative >= threshold && normative > empirical && normative > mythic) {
      return 'normative';
    }
    if (mythic >= threshold && mythic > empirical && mythic > normative) {
      return 'mythic';
    }
    return 'mixed';
  }

  private generateTags(
    text: string,
    classification: EpistemicClassification,
    dominantType: ClassificationResult['dominantType']
  ): string[] {
    const tags: string[] = [];

    // Add dimension tag
    tags.push(dominantType);

    // Add specific tags based on content
    const normalized = text.toLowerCase();

    if (/scien\w+|research|study/i.test(text)) tags.push('science');
    if (/politic\w+|government|policy/i.test(text)) tags.push('politics');
    if (/econom\w+|market|financial/i.test(text)) tags.push('economics');
    if (/health|medical|disease/i.test(text)) tags.push('health');
    if (/environment|climate|nature/i.test(text)) tags.push('environment');
    if (/technology|digital|computer/i.test(text)) tags.push('technology');
    if (/education|school|learning/i.test(text)) tags.push('education');
    if (/history|historical|ancient/i.test(text)) tags.push('history');
    if (/religion|spiritual|faith/i.test(text)) tags.push('religion');
    if (/philosophy|philosophical/i.test(text)) tags.push('philosophy');

    return [...new Set(tags)].slice(0, 5);
  }
}

export default EpistemicClassifier;
