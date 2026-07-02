// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI-Assisted Contradiction Detector
 *
 * Detects logical contradictions and inconsistencies between claims.
 * Uses semantic analysis and logical inference patterns.
 *
 * @example
 * ```typescript
 * const detector = new ContradictionDetector();
 * const result = await detector.detect(claim1, claim2);
 * console.log(result.contradicts); // true/false
 * console.log(result.reason); // "Direct negation detected"
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface ContradictionResult {
  /** Whether a contradiction was detected */
  contradicts: boolean;
  /** Confidence in the detection (0-1) */
  confidence: number;
  /** Type of contradiction detected */
  type: ContradictionType | null;
  /** Human-readable explanation */
  reason: string;
  /** Specific conflicting elements */
  conflicts: ConflictDetail[];
  /** Suggestions for resolution */
  resolutionHints: string[];
}

export type ContradictionType =
  | 'direct_negation'         // "X is true" vs "X is false"
  | 'mutually_exclusive'      // "X is A" vs "X is B" where A and B are exclusive
  | 'quantitative_conflict'   // Different numbers for the same claim
  | 'temporal_conflict'       // Conflicting timeline/dates
  | 'scope_conflict'          // "All X" vs "Some X not"
  | 'causal_conflict'         // "A causes B" vs "A doesn't cause B"
  | 'definitional_conflict';  // Different definitions of the same term

export interface ConflictDetail {
  /** Element from first claim */
  element1: string;
  /** Element from second claim */
  element2: string;
  /** Why these conflict */
  explanation: string;
}

export interface DetectorOptions {
  /** Minimum confidence threshold */
  minConfidence?: number;
  /** Include resolution hints */
  includeHints?: boolean;
  /** Strict mode - lower tolerance */
  strictMode?: boolean;
}

// ============================================================================
// Patterns for Contradiction Detection
// ============================================================================

const NEGATION_WORDS = new Set([
  'not', "n't", 'no', 'never', 'none', 'nothing', 'nowhere',
  'neither', 'nobody', 'hardly', 'scarcely', 'rarely',
  'deny', 'denies', 'denied', 'reject', 'rejects', 'rejected',
  'false', 'untrue', 'incorrect', 'wrong', 'inaccurate',
]);

const ABSOLUTIST_WORDS = {
  universal: ['all', 'every', 'always', 'everyone', 'everywhere', 'entire', 'complete'],
  existential: ['some', 'sometimes', 'someone', 'somewhere', 'few', 'partial'],
  impossible: ['never', 'none', 'nobody', 'nothing', 'nowhere', 'impossible'],
  certain: ['must', 'definitely', 'certainly', 'absolutely', 'guaranteed'],
};

const CAUSAL_VERBS = [
  'cause', 'causes', 'caused',
  'lead', 'leads', 'led',
  'result', 'results', 'resulted',
  'produce', 'produces', 'produced',
  'create', 'creates', 'created',
  'trigger', 'triggers', 'triggered',
];

const COMPARATIVE_PAIRS: [string[], string[]][] = [
  [['more', 'higher', 'greater', 'increase', 'increased', 'rise', 'rose'], ['less', 'lower', 'fewer', 'decrease', 'decreased', 'fall', 'fell']],
  [['better', 'improved', 'superior'], ['worse', 'worsened', 'inferior']],
  [['true', 'correct', 'right', 'accurate'], ['false', 'incorrect', 'wrong', 'inaccurate']],
  [['beneficial', 'helpful', 'positive'], ['harmful', 'detrimental', 'negative']],
  [['safe', 'secure'], ['dangerous', 'risky', 'hazardous']],
];

// ============================================================================
// Contradiction Detector Implementation
// ============================================================================

export class ContradictionDetector {
  private options: Required<DetectorOptions>;

  constructor(options: DetectorOptions = {}) {
    this.options = {
      minConfidence: options.minConfidence ?? 0.6,
      includeHints: options.includeHints ?? true,
      strictMode: options.strictMode ?? false,
    };
  }

  /**
   * Detect contradictions between two claims
   */
  async detect(claim1: string, claim2: string): Promise<ContradictionResult> {
    const tokens1 = this.tokenize(claim1);
    const tokens2 = this.tokenize(claim2);

    const normalized1 = this.normalize(claim1);
    const normalized2 = this.normalize(claim2);

    // Check various contradiction types
    const checks = [
      this.checkDirectNegation(normalized1, normalized2, tokens1, tokens2),
      this.checkMutuallyExclusive(tokens1, tokens2),
      this.checkQuantitativeConflict(claim1, claim2),
      this.checkTemporalConflict(claim1, claim2),
      this.checkScopeConflict(tokens1, tokens2),
      this.checkCausalConflict(normalized1, normalized2),
    ];

    // Find the highest confidence contradiction
    let bestResult: ContradictionResult | null = null;

    for (const result of checks) {
      if (result && result.confidence >= this.options.minConfidence) {
        if (!bestResult || result.confidence > bestResult.confidence) {
          bestResult = result;
        }
      }
    }

    if (bestResult) {
      return bestResult;
    }

    // No contradiction detected
    return {
      contradicts: false,
      confidence: this.calculateNonContradictionConfidence(normalized1, normalized2),
      type: null,
      reason: 'No contradiction detected between these claims.',
      conflicts: [],
      resolutionHints: [],
    };
  }

  /**
   * Check multiple claims for contradictions
   */
  async detectInSet(claims: string[]): Promise<Map<string, ContradictionResult[]>> {
    const results = new Map<string, ContradictionResult[]>();

    for (let i = 0; i < claims.length; i++) {
      const contradictions: ContradictionResult[] = [];

      for (let j = 0; j < claims.length; j++) {
        if (i !== j) {
          const result = await this.detect(claims[i], claims[j]);
          if (result.contradicts) {
            contradictions.push(result);
          }
        }
      }

      results.set(claims[i], contradictions);
    }

    return results;
  }

  /**
   * Get a summary of contradictions in a set of claims
   */
  async summarizeContradictions(claims: string[]): Promise<{
    totalClaims: number;
    contradictionCount: number;
    contradictionPairs: Array<{ claim1: string; claim2: string; result: ContradictionResult }>;
    consistency: number;
  }> {
    const pairs: Array<{ claim1: string; claim2: string; result: ContradictionResult }> = [];

    for (let i = 0; i < claims.length; i++) {
      for (let j = i + 1; j < claims.length; j++) {
        const result = await this.detect(claims[i], claims[j]);
        if (result.contradicts) {
          pairs.push({ claim1: claims[i], claim2: claims[j], result });
        }
      }
    }

    const maxPossiblePairs = (claims.length * (claims.length - 1)) / 2;
    const consistency = 1 - (pairs.length / maxPossiblePairs);

    return {
      totalClaims: claims.length,
      contradictionCount: pairs.length,
      contradictionPairs: pairs,
      consistency,
    };
  }

  // ============================================================================
  // Private Detection Methods
  // ============================================================================

  private checkDirectNegation(
    norm1: string,
    norm2: string,
    tokens1: string[],
    tokens2: string[]
  ): ContradictionResult | null {
    // Check if one claim is a negation of the other
    const negations1 = tokens1.filter(t => NEGATION_WORDS.has(t));
    const negations2 = tokens2.filter(t => NEGATION_WORDS.has(t));

    // If one has negation and the other doesn't, and they share core content
    const negationDiff = (negations1.length % 2) !== (negations2.length % 2);

    if (negationDiff) {
      // Check for shared core content
      const shared = this.findSharedContent(norm1, norm2);

      if (shared.similarity > 0.6) {
        return {
          contradicts: true,
          confidence: 0.75 + (shared.similarity - 0.6) * 0.5,
          type: 'direct_negation',
          reason: 'One claim appears to directly negate the other.',
          conflicts: [{
            element1: shared.overlapping[0] || norm1.slice(0, 50),
            element2: negationDiff ? `NOT ${shared.overlapping[0] || norm2.slice(0, 50)}` : norm2.slice(0, 50),
            explanation: 'Direct negation of the same proposition',
          }],
          resolutionHints: this.options.includeHints ? [
            'Verify which claim has more supporting evidence',
            'Check if the claims refer to different contexts or time periods',
            'Consider whether both could be partially true in different scopes',
          ] : [],
        };
      }
    }

    return null;
  }

  private checkMutuallyExclusive(tokens1: string[], tokens2: string[]): ContradictionResult | null {
    for (const [group1, group2] of COMPARATIVE_PAIRS) {
      const has1inG1 = tokens1.some(t => group1.includes(t));
      const has1inG2 = tokens1.some(t => group2.includes(t));
      const has2inG1 = tokens2.some(t => group1.includes(t));
      const has2inG2 = tokens2.some(t => group2.includes(t));

      // Check if claims use opposite terms
      if ((has1inG1 && has2inG2) || (has1inG2 && has2inG1)) {
        // Verify they're about the same subject
        const sharedSubject = this.findSharedSubject(tokens1, tokens2);

        if (sharedSubject) {
          return {
            contradicts: true,
            confidence: 0.7,
            type: 'mutually_exclusive',
            reason: 'Claims use mutually exclusive terms about the same subject.',
            conflicts: [{
              element1: group1.find(t => tokens1.includes(t)) || group2.find(t => tokens1.includes(t)) || '',
              element2: group2.find(t => tokens2.includes(t)) || group1.find(t => tokens2.includes(t)) || '',
              explanation: `These terms are mutually exclusive regarding "${sharedSubject}"`,
            }],
            resolutionHints: this.options.includeHints ? [
              'Check if claims refer to different aspects of the subject',
              'Verify the specific context of each claim',
              'Look for qualifying conditions that might resolve the conflict',
            ] : [],
          };
        }
      }
    }

    return null;
  }

  private checkQuantitativeConflict(claim1: string, claim2: string): ContradictionResult | null {
    // Extract numbers from both claims
    const numbers1 = this.extractNumbers(claim1);
    const numbers2 = this.extractNumbers(claim2);

    if (numbers1.length === 0 || numbers2.length === 0) return null;

    // Check if they're about the same thing but have different values
    const subjects = this.findSharedContent(claim1, claim2);

    if (subjects.similarity > 0.4) {
      // Compare numbers
      for (const n1 of numbers1) {
        for (const n2 of numbers2) {
          // Check if numbers are significantly different
          const ratio = Math.max(n1.value, n2.value) / Math.min(n1.value, n2.value);

          if (ratio > 1.5 && n1.context === n2.context) {
            return {
              contradicts: true,
              confidence: 0.65 + Math.min(0.25, (ratio - 1.5) * 0.1),
              type: 'quantitative_conflict',
              reason: 'Claims contain conflicting numerical values for the same measurement.',
              conflicts: [{
                element1: `${n1.value}${n1.unit || ''}`,
                element2: `${n2.value}${n2.unit || ''}`,
                explanation: `${ratio.toFixed(1)}x difference in the same metric`,
              }],
              resolutionHints: this.options.includeHints ? [
                'Check the sources of each numerical claim',
                'Verify the measurement methodology',
                'Consider if numbers refer to different time periods or populations',
              ] : [],
            };
          }
        }
      }
    }

    return null;
  }

  private checkTemporalConflict(claim1: string, claim2: string): ContradictionResult | null {
    const dates1 = this.extractDates(claim1);
    const dates2 = this.extractDates(claim2);

    if (dates1.length === 0 || dates2.length === 0) return null;

    // Check for timeline conflicts
    const subjects = this.findSharedContent(claim1, claim2);

    if (subjects.similarity > 0.5) {
      for (const d1 of dates1) {
        for (const d2 of dates2) {
          // Check for significant date differences on the same event
          const yearDiff = Math.abs((d1.year || 0) - (d2.year || 0));

          if (yearDiff > 5) {
            return {
              contradicts: true,
              confidence: 0.6,
              type: 'temporal_conflict',
              reason: 'Claims contain conflicting dates for the same event.',
              conflicts: [{
                element1: d1.original,
                element2: d2.original,
                explanation: `${yearDiff} year difference`,
              }],
              resolutionHints: this.options.includeHints ? [
                'Verify the primary sources for each date',
                'Check if dates refer to different phases of an event',
                'Consider calendar system differences',
              ] : [],
            };
          }
        }
      }
    }

    return null;
  }

  private checkScopeConflict(tokens1: string[], tokens2: string[]): ContradictionResult | null {
    // Check for universal vs existential conflicts
    const universal1 = ABSOLUTIST_WORDS.universal.some(w => tokens1.includes(w));
    const impossible2 = ABSOLUTIST_WORDS.impossible.some(w => tokens2.includes(w));
    const universal2 = ABSOLUTIST_WORDS.universal.some(w => tokens2.includes(w));
    const impossible1 = ABSOLUTIST_WORDS.impossible.some(w => tokens1.includes(w));

    if ((universal1 && impossible2) || (universal2 && impossible1)) {
      return {
        contradicts: true,
        confidence: 0.75,
        type: 'scope_conflict',
        reason: 'One claim makes a universal assertion that the other denies.',
        conflicts: [{
          element1: universal1 ? 'Universal claim (all/every/always)' : 'Absolute denial (never/none)',
          element2: impossible2 ? 'Absolute denial (never/none)' : 'Universal claim (all/every/always)',
          explanation: 'Universal and absolute denial cannot both be true',
        }],
        resolutionHints: this.options.includeHints ? [
          'Look for exceptions that might invalidate the universal claim',
          'Consider if the scope of each claim is actually different',
          'Check for implicit qualifiers in context',
        ] : [],
      };
    }

    return null;
  }

  private checkCausalConflict(norm1: string, norm2: string): ContradictionResult | null {
    // Extract causal relationships
    const causal1 = this.extractCausalRelation(norm1);
    const causal2 = this.extractCausalRelation(norm2);

    if (!causal1 || !causal2) return null;

    // Check if same cause-effect pair with opposite direction
    const samePair = (
      (this.similar(causal1.cause, causal2.cause) && this.similar(causal1.effect, causal2.effect)) ||
      (this.similar(causal1.cause, causal2.effect) && this.similar(causal1.effect, causal2.cause))
    );

    if (samePair && causal1.negated !== causal2.negated) {
      return {
        contradicts: true,
        confidence: 0.7,
        type: 'causal_conflict',
        reason: 'Claims make contradictory causal assertions.',
        conflicts: [{
          element1: `${causal1.cause} ${causal1.negated ? 'does NOT cause' : 'causes'} ${causal1.effect}`,
          element2: `${causal2.cause} ${causal2.negated ? 'does NOT cause' : 'causes'} ${causal2.effect}`,
          explanation: 'Contradictory causal claims about the same relationship',
        }],
        resolutionHints: this.options.includeHints ? [
          'Review the evidence for each causal claim',
          'Consider confounding variables',
          'Check if correlation is being confused with causation',
        ] : [],
      };
    }

    return null;
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1);
  }

  private normalize(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private findSharedContent(norm1: string, norm2: string): { similarity: number; overlapping: string[] } {
    const words1 = new Set(norm1.split(' ').filter(w => w.length > 3));
    const words2 = new Set(norm2.split(' ').filter(w => w.length > 3));

    const overlapping = [...words1].filter(w => words2.has(w));
    const similarity = overlapping.length / Math.max(words1.size, words2.size);

    return { similarity, overlapping };
  }

  private findSharedSubject(tokens1: string[], tokens2: string[]): string | null {
    // Simple heuristic: find shared nouns (longer words that aren't common)
    const commonWords = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'are', 'was', 'is']);
    const shared = tokens1.filter(t => tokens2.includes(t) && t.length > 3 && !commonWords.has(t));
    return shared[0] || null;
  }

  private extractNumbers(text: string): Array<{ value: number; unit: string; context: string }> {
    const results: Array<{ value: number; unit: string; context: string }> = [];
    const pattern = /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|%|km|kg|m)?/gi;

    let match;
    while ((match = pattern.exec(text)) !== null) {
      let value = parseFloat(match[1].replace(/,/g, ''));
      const unit = match[2]?.toLowerCase() || '';

      // Convert to base units
      if (unit === 'million') value *= 1e6;
      if (unit === 'billion') value *= 1e9;
      if (unit === 'thousand') value *= 1e3;

      results.push({
        value,
        unit,
        context: text.slice(Math.max(0, match.index - 20), match.index + 20),
      });
    }

    return results;
  }

  private extractDates(text: string): Array<{ year?: number; original: string }> {
    const results: Array<{ year?: number; original: string }> = [];

    // Year pattern
    const yearPattern = /\b(1[89]\d{2}|2[01]\d{2})\b/g;
    let match;
    while ((match = yearPattern.exec(text)) !== null) {
      results.push({ year: parseInt(match[1]), original: match[0] });
    }

    return results;
  }

  private extractCausalRelation(text: string): { cause: string; effect: string; negated: boolean } | null {
    for (const verb of CAUSAL_VERBS) {
      const pattern = new RegExp(`(.{5,30})\\s+(?:does not |doesn't |did not |didn't )?(${verb})\\s+(.{5,30})`, 'i');
      const match = text.match(pattern);

      if (match) {
        return {
          cause: match[1].trim(),
          effect: match[3].trim(),
          negated: !!match[0].match(/does not|doesn't|did not|didn't/i),
        };
      }
    }

    return null;
  }

  private similar(s1: string, s2: string): boolean {
    const t1 = this.tokenize(s1);
    const t2 = this.tokenize(s2);
    const shared = t1.filter(t => t2.includes(t));
    return shared.length / Math.max(t1.length, t2.length) > 0.5;
  }

  private calculateNonContradictionConfidence(norm1: string, norm2: string): number {
    const shared = this.findSharedContent(norm1, norm2);
    // Less shared content = more confident they don't contradict
    return 0.5 + (1 - shared.similarity) * 0.4;
  }
}

export default ContradictionDetector;
