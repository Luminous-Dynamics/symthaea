// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI-Assisted Similarity Analyzer
 *
 * Analyzes semantic similarity between claims using
 * TF-IDF, cosine similarity, and n-gram analysis.
 *
 * @example
 * ```typescript
 * const analyzer = new SimilarityAnalyzer();
 * const score = await analyzer.compare(claim1, claim2);
 * console.log(score); // 0.85 (very similar)
 *
 * const clusters = await analyzer.cluster(claims, { threshold: 0.7 });
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface SimilarityResult {
  /** Overall similarity score (0-1) */
  score: number;
  /** Lexical similarity */
  lexical: number;
  /** Semantic similarity (based on topic overlap) */
  semantic: number;
  /** Structural similarity */
  structural: number;
  /** Shared key terms */
  sharedTerms: string[];
  /** Explanation */
  explanation: string;
}

export interface ClusterResult {
  /** Cluster ID */
  id: number;
  /** Claims in this cluster */
  claims: string[];
  /** Centroid/representative claim */
  centroid: string;
  /** Average intra-cluster similarity */
  cohesion: number;
  /** Common topics */
  topics: string[];
}

export interface SimilarityMatrix {
  /** The claims */
  claims: string[];
  /** NxN similarity matrix */
  matrix: number[][];
  /** Top similar pairs */
  topPairs: Array<{ claim1: number; claim2: number; similarity: number }>;
}

export interface AnalyzerOptions {
  /** Use stemming */
  useStemming?: boolean;
  /** Remove stopwords */
  removeStopwords?: boolean;
  /** N-gram size for analysis */
  ngramSize?: number;
  /** Weight for lexical vs semantic similarity */
  lexicalWeight?: number;
}

// ============================================================================
// Constants
// ============================================================================

const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'it', 'its', 'this', 'that', 'these',
  'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
  'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
  'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
]);

const STEM_RULES: Array<[RegExp, string]> = [
  [/ies$/, 'y'],
  [/ves$/, 'f'],
  [/(s|x|ch|sh)es$/, '$1'],
  [/s$/, ''],
  [/ing$/, ''],
  [/ed$/, ''],
  [/ly$/, ''],
  [/ment$/, ''],
  [/ness$/, ''],
  [/tion$/, 't'],
  [/ation$/, ''],
  [/ity$/, ''],
];

// ============================================================================
// Similarity Analyzer Implementation
// ============================================================================

export class SimilarityAnalyzer {
  private options: Required<AnalyzerOptions>;
  private idfCache: Map<string, number> = new Map();
  private documentFrequencies: Map<string, number> = new Map();
  private totalDocuments: number = 0;

  constructor(options: AnalyzerOptions = {}) {
    this.options = {
      useStemming: options.useStemming ?? true,
      removeStopwords: options.removeStopwords ?? true,
      ngramSize: options.ngramSize ?? 2,
      lexicalWeight: options.lexicalWeight ?? 0.4,
    };
  }

  /**
   * Compare two claims for similarity
   */
  async compare(claim1: string, claim2: string): Promise<SimilarityResult> {
    const tokens1 = this.tokenize(claim1);
    const tokens2 = this.tokenize(claim2);

    // Lexical similarity (Jaccard)
    const lexical = this.jaccardSimilarity(tokens1, tokens2);

    // N-gram similarity
    const ngrams1 = this.getNgrams(tokens1, this.options.ngramSize);
    const ngrams2 = this.getNgrams(tokens2, this.options.ngramSize);
    const ngramSim = this.jaccardSimilarity(ngrams1, ngrams2);

    // TF-IDF cosine similarity
    const tfidf1 = this.getTfIdf(tokens1, [tokens1, tokens2]);
    const tfidf2 = this.getTfIdf(tokens2, [tokens1, tokens2]);
    const semantic = this.cosineSimilarity(tfidf1, tfidf2);

    // Structural similarity (length, sentence structure)
    const structural = this.structuralSimilarity(claim1, claim2);

    // Combined score
    const score = (
      this.options.lexicalWeight * lexical +
      (1 - this.options.lexicalWeight) * 0.5 * semantic +
      (1 - this.options.lexicalWeight) * 0.3 * ngramSim +
      (1 - this.options.lexicalWeight) * 0.2 * structural
    );

    // Find shared terms
    const sharedTerms = tokens1.filter(t => tokens2.includes(t))
      .filter(t => t.length > 3)
      .slice(0, 10);

    return {
      score,
      lexical,
      semantic,
      structural,
      sharedTerms,
      explanation: this.generateExplanation(score, lexical, semantic, sharedTerms),
    };
  }

  /**
   * Build a similarity matrix for multiple claims
   */
  async buildMatrix(claims: string[]): Promise<SimilarityMatrix> {
    const n = claims.length;
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
    const topPairs: Array<{ claim1: number; claim2: number; similarity: number }> = [];

    // Calculate all pairwise similarities
    for (let i = 0; i < n; i++) {
      matrix[i][i] = 1.0;
      for (let j = i + 1; j < n; j++) {
        const result = await this.compare(claims[i], claims[j]);
        matrix[i][j] = result.score;
        matrix[j][i] = result.score;

        topPairs.push({ claim1: i, claim2: j, similarity: result.score });
      }
    }

    // Sort top pairs
    topPairs.sort((a, b) => b.similarity - a.similarity);

    return { claims, matrix, topPairs: topPairs.slice(0, 20) };
  }

  /**
   * Cluster similar claims together
   */
  async cluster(claims: string[], options: { threshold?: number; maxClusters?: number } = {}): Promise<ClusterResult[]> {
    const threshold = options.threshold ?? 0.6;
    const maxClusters = options.maxClusters ?? Math.ceil(claims.length / 3);

    // Build similarity matrix
    const { matrix } = await this.buildMatrix(claims);

    // Simple hierarchical clustering
    const clusters: Set<number>[] = claims.map((_, i) => new Set([i]));
    const clusterMap = claims.map((_, i) => i);

    let merged = true;
    while (merged && clusters.filter(c => c.size > 0).length > maxClusters) {
      merged = false;

      // Find most similar pair of clusters
      let bestI = -1, bestJ = -1, bestSim = threshold;

      for (let i = 0; i < clusters.length; i++) {
        if (clusters[i].size === 0) continue;
        for (let j = i + 1; j < clusters.length; j++) {
          if (clusters[j].size === 0) continue;

          const sim = this.clusterSimilarity(clusters[i], clusters[j], matrix);
          if (sim > bestSim) {
            bestSim = sim;
            bestI = i;
            bestJ = j;
          }
        }
      }

      // Merge best pair
      if (bestI >= 0 && bestJ >= 0) {
        for (const idx of clusters[bestJ]) {
          clusters[bestI].add(idx);
          clusterMap[idx] = bestI;
        }
        clusters[bestJ].clear();
        merged = true;
      }
    }

    // Build result
    const results: ClusterResult[] = [];
    let clusterId = 0;

    for (const cluster of clusters) {
      if (cluster.size === 0) continue;

      const claimIndices = Array.from(cluster);
      const clusterClaims = claimIndices.map(i => claims[i]);

      // Find centroid (claim most similar to all others)
      let centroidIdx = claimIndices[0];
      let maxAvgSim = 0;

      for (const i of claimIndices) {
        const avgSim = claimIndices
          .filter(j => j !== i)
          .reduce((sum, j) => sum + matrix[i][j], 0) / (claimIndices.length - 1 || 1);
        if (avgSim > maxAvgSim) {
          maxAvgSim = avgSim;
          centroidIdx = i;
        }
      }

      // Calculate cohesion
      let cohesion = 1;
      if (claimIndices.length > 1) {
        let total = 0, count = 0;
        for (let i = 0; i < claimIndices.length; i++) {
          for (let j = i + 1; j < claimIndices.length; j++) {
            total += matrix[claimIndices[i]][claimIndices[j]];
            count++;
          }
        }
        cohesion = count > 0 ? total / count : 1;
      }

      // Extract topics
      const topics = this.extractTopics(clusterClaims);

      results.push({
        id: clusterId++,
        claims: clusterClaims,
        centroid: claims[centroidIdx],
        cohesion,
        topics,
      });
    }

    return results.sort((a, b) => b.claims.length - a.claims.length);
  }

  /**
   * Find the most similar claim from a set
   */
  async findMostSimilar(query: string, candidates: string[], topK: number = 5): Promise<Array<{ claim: string; similarity: number }>> {
    const results: Array<{ claim: string; similarity: number }> = [];

    for (const candidate of candidates) {
      const result = await this.compare(query, candidate);
      results.push({ claim: candidate, similarity: result.score });
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  /**
   * Train the analyzer on a corpus (builds IDF values)
   */
  train(documents: string[]): void {
    this.documentFrequencies.clear();
    this.idfCache.clear();
    this.totalDocuments = documents.length;

    for (const doc of documents) {
      const tokens = new Set(this.tokenize(doc));
      for (const token of tokens) {
        this.documentFrequencies.set(
          token,
          (this.documentFrequencies.get(token) || 0) + 1
        );
      }
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private tokenize(text: string): string[] {
    let tokens = text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1);

    if (this.options.removeStopwords) {
      tokens = tokens.filter(t => !STOPWORDS.has(t));
    }

    if (this.options.useStemming) {
      tokens = tokens.map(t => this.stem(t));
    }

    return tokens;
  }

  private stem(word: string): string {
    for (const [pattern, replacement] of STEM_RULES) {
      if (pattern.test(word)) {
        return word.replace(pattern, replacement);
      }
    }
    return word;
  }

  private getNgrams(tokens: string[], n: number): string[] {
    const ngrams: string[] = [];
    for (let i = 0; i <= tokens.length - n; i++) {
      ngrams.push(tokens.slice(i, i + n).join(' '));
    }
    return ngrams;
  }

  private jaccardSimilarity(set1: string[], set2: string[]): number {
    const s1 = new Set(set1);
    const s2 = new Set(set2);

    const intersection = new Set([...s1].filter(x => s2.has(x)));
    const union = new Set([...s1, ...s2]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private getTfIdf(tokens: string[], corpus: string[][]): Map<string, number> {
    const tf = new Map<string, number>();
    const tfidf = new Map<string, number>();

    // Calculate TF
    for (const token of tokens) {
      tf.set(token, (tf.get(token) || 0) + 1);
    }

    // Calculate TF-IDF
    for (const [token, count] of tf) {
      const idf = this.getIdf(token, corpus);
      tfidf.set(token, (count / tokens.length) * idf);
    }

    return tfidf;
  }

  private getIdf(term: string, corpus: string[][]): number {
    if (this.totalDocuments > 0 && this.documentFrequencies.has(term)) {
      // Use pre-computed IDF
      if (!this.idfCache.has(term)) {
        const df = this.documentFrequencies.get(term) || 1;
        this.idfCache.set(term, Math.log(this.totalDocuments / df));
      }
      return this.idfCache.get(term)!;
    }

    // Compute on the fly
    const df = corpus.filter(doc => doc.includes(term)).length || 1;
    return Math.log(corpus.length / df);
  }

  private cosineSimilarity(vec1: Map<string, number>, vec2: Map<string, number>): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    const allKeys = new Set([...vec1.keys(), ...vec2.keys()]);

    for (const key of allKeys) {
      const v1 = vec1.get(key) || 0;
      const v2 = vec2.get(key) || 0;
      dotProduct += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denominator > 0 ? dotProduct / denominator : 0;
  }

  private structuralSimilarity(text1: string, text2: string): number {
    // Compare length
    const len1 = text1.length;
    const len2 = text2.length;
    const lengthSim = Math.min(len1, len2) / Math.max(len1, len2);

    // Compare sentence count
    const sent1 = text1.split(/[.!?]+/).length;
    const sent2 = text2.split(/[.!?]+/).length;
    const sentSim = Math.min(sent1, sent2) / Math.max(sent1, sent2);

    // Compare word count
    const words1 = text1.split(/\s+/).length;
    const words2 = text2.split(/\s+/).length;
    const wordSim = Math.min(words1, words2) / Math.max(words1, words2);

    return (lengthSim + sentSim + wordSim) / 3;
  }

  private clusterSimilarity(cluster1: Set<number>, cluster2: Set<number>, matrix: number[][]): number {
    // Average linkage
    let total = 0;
    let count = 0;

    for (const i of cluster1) {
      for (const j of cluster2) {
        total += matrix[i][j];
        count++;
      }
    }

    return count > 0 ? total / count : 0;
  }

  private extractTopics(claims: string[]): string[] {
    const termFreq = new Map<string, number>();

    for (const claim of claims) {
      const tokens = this.tokenize(claim);
      for (const token of tokens) {
        if (token.length > 3) {
          termFreq.set(token, (termFreq.get(token) || 0) + 1);
        }
      }
    }

    return [...termFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([term]) => term);
  }

  private generateExplanation(score: number, lexical: number, semantic: number, sharedTerms: string[]): string {
    let explanation = '';

    if (score >= 0.9) {
      explanation = 'These claims are nearly identical or paraphrases of each other.';
    } else if (score >= 0.7) {
      explanation = 'These claims are very similar and likely discuss the same topic.';
    } else if (score >= 0.5) {
      explanation = 'These claims have moderate overlap and may be related.';
    } else if (score >= 0.3) {
      explanation = 'These claims have some connection but discuss different aspects.';
    } else {
      explanation = 'These claims appear to be about different topics.';
    }

    if (sharedTerms.length > 0) {
      explanation += ` Key shared terms: ${sharedTerms.slice(0, 3).join(', ')}.`;
    }

    return explanation;
  }
}

export default SimilarityAnalyzer;
