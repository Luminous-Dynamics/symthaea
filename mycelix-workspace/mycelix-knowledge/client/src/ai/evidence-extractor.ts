// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI-Assisted Evidence Extractor
 *
 * Extracts structured evidence from unstructured text,
 * identifying sources, statistics, and supporting claims.
 *
 * @example
 * ```typescript
 * const extractor = new EvidenceExtractor();
 * const evidence = await extractor.extract(articleText);
 * console.log(evidence.sources); // ['Nature', 'WHO']
 * console.log(evidence.statistics); // ['45% increase', '3.2 million cases']
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface ExtractedEvidence {
  /** Named sources mentioned in text */
  sources: SourceReference[];
  /** Statistical claims found */
  statistics: StatisticClaim[];
  /** Quoted statements */
  quotes: QuotedStatement[];
  /** Date/time references */
  temporalReferences: TemporalReference[];
  /** Causal claims */
  causalClaims: CausalClaim[];
  /** Overall evidence quality score */
  qualityScore: number;
  /** Extraction metadata */
  metadata: ExtractionMetadata;
}

export interface SourceReference {
  name: string;
  type: 'publication' | 'organization' | 'person' | 'study' | 'unknown';
  context: string;
  confidence: number;
  position: { start: number; end: number };
}

export interface StatisticClaim {
  value: string;
  unit?: string;
  context: string;
  type: 'percentage' | 'count' | 'ratio' | 'currency' | 'measurement' | 'other';
  confidence: number;
  position: { start: number; end: number };
}

export interface QuotedStatement {
  text: string;
  attribution?: string;
  context: string;
  position: { start: number; end: number };
}

export interface TemporalReference {
  text: string;
  type: 'specific_date' | 'relative' | 'range' | 'duration';
  normalized?: string;
  position: { start: number; end: number };
}

export interface CausalClaim {
  cause: string;
  effect: string;
  strength: 'strong' | 'moderate' | 'weak' | 'correlational';
  context: string;
  position: { start: number; end: number };
}

export interface ExtractionMetadata {
  textLength: number;
  processingTimeMs: number;
  totalEvidenceItems: number;
  coveragePercent: number;
}

export interface ExtractorOptions {
  /** Extract sources */
  extractSources?: boolean;
  /** Extract statistics */
  extractStatistics?: boolean;
  /** Extract quotes */
  extractQuotes?: boolean;
  /** Extract temporal references */
  extractTemporal?: boolean;
  /** Extract causal claims */
  extractCausal?: boolean;
  /** Minimum confidence threshold */
  minConfidence?: number;
}

// ============================================================================
// Patterns
// ============================================================================

const SOURCE_PATTERNS = {
  publications: [
    /(?:published in|according to|reported by|in) (?:the )?([A-Z][A-Za-z\s&]+(?:Journal|Review|Times|Post|News|Report))/g,
    /(?:a )?(?:study|paper|article|report) (?:in|by|from) ([A-Z][A-Za-z\s&]+)/g,
    /([A-Z][A-Za-z]+ (?:University|Institute|Center|Centre))/g,
  ],
  organizations: [
    /((?:World|International|National|American|European|United Nations) [A-Za-z\s]+)/g,
    /(?:the )?([A-Z]{2,6})\b/g,  // Acronyms like WHO, NASA, FBI
    /((?:Department|Ministry|Agency) of [A-Za-z\s]+)/g,
  ],
  people: [
    /(?:Dr\.|Prof\.|Professor|Mr\.|Mrs\.|Ms\.) ([A-Z][a-z]+ [A-Z][a-z]+)/g,
    /([A-Z][a-z]+ [A-Z][a-z]+)(?:,? (?:a|an|the) (?:researcher|scientist|expert|professor|doctor|analyst))/g,
  ],
};

const STATISTIC_PATTERNS = [
  { pattern: /(\d+(?:\.\d+)?)\s*%/g, type: 'percentage' as const },
  { pattern: /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|trillion)/gi, type: 'count' as const },
  { pattern: /\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|trillion)?/gi, type: 'currency' as const },
  { pattern: /(\d+(?:\.\d+)?)\s*(?:to|times|x)\s*(\d+(?:\.\d+)?)/g, type: 'ratio' as const },
  { pattern: /(\d+(?:\.\d+)?)\s*(km|m|cm|mm|kg|g|mg|L|mL|°C|°F)/g, type: 'measurement' as const },
  { pattern: /(?:approximately|about|around|nearly|roughly|over|under|less than|more than) (\d+(?:,\d{3})*(?:\.\d+)?)/gi, type: 'count' as const },
];

const TEMPORAL_PATTERNS = [
  { pattern: /(?:in|during|since|by) (\d{4})/g, type: 'specific_date' as const },
  { pattern: /(\d{1,2}\/\d{1,2}\/\d{2,4})/g, type: 'specific_date' as const },
  { pattern: /(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:st|nd|rd|th)?,? \d{4}/gi, type: 'specific_date' as const },
  { pattern: /(last|next|this) (week|month|year|decade|century)/gi, type: 'relative' as const },
  { pattern: /(\d+) (days?|weeks?|months?|years?) (?:ago|from now|later|earlier)/gi, type: 'relative' as const },
  { pattern: /(?:from|between) (\d{4})(?: to| and| - )(\d{4})/gi, type: 'range' as const },
  { pattern: /(?:for|over|during) (\d+) (days?|weeks?|months?|years?)/gi, type: 'duration' as const },
];

const CAUSAL_PATTERNS = [
  { pattern: /(.{10,50}) (?:caused|causes|caused by|resulting in|leads? to|led to) (.{10,50})/gi, strength: 'strong' as const },
  { pattern: /(.{10,50}) (?:contributes? to|affects?|impacts?|influences?) (.{10,50})/gi, strength: 'moderate' as const },
  { pattern: /(.{10,50}) (?:may cause|might lead to|could result in|possibly affects?) (.{10,50})/gi, strength: 'weak' as const },
  { pattern: /(.{10,50}) (?:is (?:correlated|associated|linked) with) (.{10,50})/gi, strength: 'correlational' as const },
];

const QUOTE_PATTERN = /"([^"]{10,500})"/g;

// ============================================================================
// Evidence Extractor Implementation
// ============================================================================

export class EvidenceExtractor {
  private options: Required<ExtractorOptions>;

  constructor(options: ExtractorOptions = {}) {
    this.options = {
      extractSources: options.extractSources ?? true,
      extractStatistics: options.extractStatistics ?? true,
      extractQuotes: options.extractQuotes ?? true,
      extractTemporal: options.extractTemporal ?? true,
      extractCausal: options.extractCausal ?? true,
      minConfidence: options.minConfidence ?? 0.5,
    };
  }

  /**
   * Extract all evidence from text
   */
  async extract(text: string): Promise<ExtractedEvidence> {
    const startTime = Date.now();

    const sources = this.options.extractSources ? this.extractSources(text) : [];
    const statistics = this.options.extractStatistics ? this.extractStatistics(text) : [];
    const quotes = this.options.extractQuotes ? this.extractQuotes(text) : [];
    const temporalReferences = this.options.extractTemporal ? this.extractTemporalReferences(text) : [];
    const causalClaims = this.options.extractCausal ? this.extractCausalClaims(text) : [];

    const totalItems = sources.length + statistics.length + quotes.length +
                      temporalReferences.length + causalClaims.length;

    // Calculate quality score
    const qualityScore = this.calculateQualityScore(sources, statistics, quotes);

    // Calculate coverage
    const coveredPositions = new Set<number>();
    [...sources, ...statistics, ...quotes].forEach(item => {
      if ('position' in item) {
        for (let i = item.position.start; i < item.position.end; i++) {
          coveredPositions.add(i);
        }
      }
    });

    return {
      sources,
      statistics,
      quotes,
      temporalReferences,
      causalClaims,
      qualityScore,
      metadata: {
        textLength: text.length,
        processingTimeMs: Date.now() - startTime,
        totalEvidenceItems: totalItems,
        coveragePercent: (coveredPositions.size / text.length) * 100,
      },
    };
  }

  /**
   * Extract only sources from text
   */
  extractSources(text: string): SourceReference[] {
    const sources: SourceReference[] = [];

    // Extract publications
    for (const pattern of SOURCE_PATTERNS.publications) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        sources.push({
          name: match[1].trim(),
          type: 'publication',
          context: this.getContext(text, match.index, 50),
          confidence: 0.8,
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    // Extract organizations
    for (const pattern of SOURCE_PATTERNS.organizations) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        // Skip very short acronyms
        if (match[1].length < 3) continue;

        sources.push({
          name: match[1].trim(),
          type: 'organization',
          context: this.getContext(text, match.index, 50),
          confidence: 0.7,
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    // Extract people
    for (const pattern of SOURCE_PATTERNS.people) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        sources.push({
          name: match[1].trim(),
          type: 'person',
          context: this.getContext(text, match.index, 50),
          confidence: 0.75,
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    return this.deduplicateAndFilter(sources);
  }

  /**
   * Extract statistics from text
   */
  extractStatistics(text: string): StatisticClaim[] {
    const statistics: StatisticClaim[] = [];

    for (const { pattern, type } of STATISTIC_PATTERNS) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        statistics.push({
          value: match[0],
          unit: match[2] || undefined,
          context: this.getContext(text, match.index, 50),
          type,
          confidence: 0.85,
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    return statistics;
  }

  /**
   * Extract quoted statements
   */
  extractQuotes(text: string): QuotedStatement[] {
    const quotes: QuotedStatement[] = [];
    let match;

    QUOTE_PATTERN.lastIndex = 0;
    while ((match = QUOTE_PATTERN.exec(text)) !== null) {
      const quoteText = match[1];

      // Look for attribution before or after the quote
      const attribution = this.findAttribution(text, match.index, match[0].length);

      quotes.push({
        text: quoteText,
        attribution,
        context: this.getContext(text, match.index, 30),
        position: { start: match.index, end: match.index + match[0].length },
      });
    }

    return quotes;
  }

  /**
   * Extract temporal references
   */
  extractTemporalReferences(text: string): TemporalReference[] {
    const references: TemporalReference[] = [];

    for (const { pattern, type } of TEMPORAL_PATTERNS) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        references.push({
          text: match[0],
          type,
          normalized: this.normalizeDate(match[0], type),
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    return references;
  }

  /**
   * Extract causal claims
   */
  extractCausalClaims(text: string): CausalClaim[] {
    const claims: CausalClaim[] = [];

    for (const { pattern, strength } of CAUSAL_PATTERNS) {
      let match;
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text)) !== null) {
        claims.push({
          cause: match[1].trim(),
          effect: match[2].trim(),
          strength,
          context: match[0],
          position: { start: match.index, end: match.index + match[0].length },
        });
      }
    }

    return claims;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private getContext(text: string, position: number, radius: number): string {
    const start = Math.max(0, position - radius);
    const end = Math.min(text.length, position + radius);
    return text.slice(start, end).trim();
  }

  private findAttribution(text: string, quoteStart: number, quoteLength: number): string | undefined {
    // Look for "said X" or "X said" patterns near the quote
    const searchRadius = 100;
    const before = text.slice(Math.max(0, quoteStart - searchRadius), quoteStart);
    const after = text.slice(quoteStart + quoteLength, quoteStart + quoteLength + searchRadius);

    const saidPattern = /([A-Z][a-z]+ [A-Z][a-z]+|(?:Dr\.|Prof\.) [A-Z][a-z]+) (?:said|stated|noted|claimed|argued)/;
    const saidByPattern = /(?:said|stated|noted|claimed|argued) ([A-Z][a-z]+ [A-Z][a-z]+|(?:Dr\.|Prof\.) [A-Z][a-z]+)/;

    const beforeMatch = before.match(saidPattern);
    const afterMatch = after.match(saidByPattern);

    return beforeMatch?.[1] || afterMatch?.[1] || undefined;
  }

  private normalizeDate(text: string, type: TemporalReference['type']): string | undefined {
    if (type === 'specific_date') {
      const yearMatch = text.match(/\d{4}/);
      if (yearMatch) return yearMatch[0];
    }
    return undefined;
  }

  private deduplicateAndFilter<T extends { name?: string; confidence: number }>(items: T[]): T[] {
    const seen = new Set<string>();
    return items.filter(item => {
      const key = ('name' in item && item.name) ? item.name.toLowerCase() : JSON.stringify(item);
      if (seen.has(key) || item.confidence < this.options.minConfidence) return false;
      seen.add(key);
      return true;
    });
  }

  private calculateQualityScore(
    sources: SourceReference[],
    statistics: StatisticClaim[],
    quotes: QuotedStatement[]
  ): number {
    let score = 0;

    // Sources contribute to quality
    score += Math.min(0.4, sources.length * 0.1);

    // Statistics contribute
    score += Math.min(0.3, statistics.length * 0.075);

    // Quotes contribute
    score += Math.min(0.2, quotes.length * 0.1);

    // Bonus for diverse source types
    const sourceTypes = new Set(sources.map(s => s.type));
    score += Math.min(0.1, sourceTypes.size * 0.025);

    return Math.min(1, score);
  }
}

export default EvidenceExtractor;
