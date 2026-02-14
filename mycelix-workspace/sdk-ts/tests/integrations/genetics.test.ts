/**
 * Genetics Integration Tests
 *
 * Tests for GeneticsBridge, GeneticEncodingClient, GeneticSimilarityClient,
 * GeneticBundlingClient, GeneticCodebookClient, and utility functions
 * (interpretHlaSimilarity, calculateEncodingConfidence).
 *
 * Uses a mock ZomeCallable for conductor interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  GeneticsBridge,
  GeneticEncodingClient,
  GeneticSimilarityClient,
  GeneticBundlingClient,
  GeneticCodebookClient,
  createGeneticsBridge,
  interpretHlaSimilarity,
  calculateEncodingConfidence,
  type ZomeCallable,
  type GeneticHypervector,
  type GeneticSimilarityResult,
  type SimilaritySearchResult,
} from '../../src/integrations/genetics/index.js';

function mockClient(): ZomeCallable {
  return {
    callZome: vi.fn(),
  };
}

const fakeHash = new Uint8Array(39);

describe('Genetics Integration', () => {
  describe('GeneticEncodingClient', () => {
    let client: ZomeCallable;
    let encoding: GeneticEncodingClient;

    beforeEach(() => {
      client = mockClient();
      encoding = new GeneticEncodingClient(client);
    });

    it('should encode a DNA sequence', async () => {
      vi.mocked(client.callZome).mockResolvedValue(fakeHash);

      const result = await encoding.encodeDnaSequence({
        patientHash: fakeHash,
        sequence: 'ATCGATCG',
        sourceMetadata: { sourceSystem: 'lab-1' },
      });

      expect(result).toBe(fakeHash);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'health',
          zome_name: 'hdc_genetics',
          fn_name: 'encode_dna_sequence',
        }),
      );
    });

    it('should encode a SNP panel', async () => {
      vi.mocked(client.callZome).mockResolvedValue(fakeHash);

      const result = await encoding.encodeSnpPanel({
        patientHash: fakeHash,
        snps: [
          ['rs1234', 'A'],
          ['rs5678', 'G'],
        ],
        sourceMetadata: { sourceSystem: 'snp-array' },
      });

      expect(result).toBe(fakeHash);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'encode_snp_panel' }),
      );
    });

    it('should encode HLA typing', async () => {
      vi.mocked(client.callZome).mockResolvedValue(fakeHash);

      const result = await encoding.encodeHlaTyping({
        patientHash: fakeHash,
        hlaTypes: ['A*02:01', 'B*07:02'],
        sourceMetadata: { sourceSystem: 'hla-lab' },
      });

      expect(result).toBe(fakeHash);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'encode_hla_typing' }),
      );
    });

    it('should get patient vectors', async () => {
      const vectors: GeneticHypervector[] = [
        {
          vectorId: 'v-1',
          patientHash: fakeHash,
          data: new Uint8Array(128),
          encodingType: 'DnaSequence',
          kmerLength: 3,
          kmerCount: 100,
          createdAt: Date.now(),
          sourceMetadata: { sourceSystem: 'lab' },
        },
      ];
      vi.mocked(client.callZome).mockResolvedValue(vectors);

      const result = await encoding.getPatientVectors(fakeHash);

      expect(result).toHaveLength(1);
      expect(result[0].vectorId).toBe('v-1');
    });
  });

  describe('GeneticSimilarityClient', () => {
    let client: ZomeCallable;
    let similarity: GeneticSimilarityClient;

    beforeEach(() => {
      client = mockClient();
      similarity = new GeneticSimilarityClient(client);
    });

    it('should calculate similarity between two vectors', async () => {
      const mockResult: GeneticSimilarityResult = {
        resultId: 'sim-1',
        queryVectorHash: fakeHash,
        targetVectorHash: fakeHash,
        similarityScore: 0.92,
        similarityMetric: 'Cosine',
        queryPurpose: 'HlaMatching',
        queriedAt: Date.now(),
        queriedBy: fakeHash,
      };
      vi.mocked(client.callZome).mockResolvedValue(mockResult);

      const result = await similarity.calculateSimilarity({
        queryVectorHash: fakeHash,
        targetVectorHash: fakeHash,
        purpose: 'HlaMatching',
      });

      expect(result.similarityScore).toBe(0.92);
    });

    it('should search for similar genetic profiles', async () => {
      const results: SimilaritySearchResult[] = [
        { vectorHash: fakeHash, patientHash: fakeHash, similarityScore: 0.95 },
        { vectorHash: fakeHash, patientHash: fakeHash, similarityScore: 0.88 },
      ];
      vi.mocked(client.callZome).mockResolvedValue(results);

      const found = await similarity.searchSimilar({
        queryVectorHash: fakeHash,
        encodingType: 'DnaSequence',
        minSimilarity: 0.8,
        limit: 10,
        purpose: { Research: 'test' },
      });

      expect(found).toHaveLength(2);
      expect(found[0].similarityScore).toBe(0.95);
    });

    it('should find HLA matches with defaults', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      await similarity.findHlaMatches(fakeHash);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'search_similar_genetics',
          payload: expect.objectContaining({
            encoding_type: 'HlaTyping',
            min_similarity: 0.8,
            limit: 10,
          }),
        }),
      );
    });

    it('should find pharmacogenomic matches', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      await similarity.findPharmacogenomicMatches(fakeHash);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            encoding_type: 'Pharmacogenomics',
            min_similarity: 0.7,
          }),
        }),
      );
    });

    it('should find clinical trial candidates', async () => {
      vi.mocked(client.callZome).mockResolvedValue([]);

      await similarity.findTrialCandidates(fakeHash, 'SnpPanel', 0.65, 25);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            encoding_type: 'SnpPanel',
            min_similarity: 0.65,
            limit: 25,
            purpose: 'ClinicalTrialMatching',
          }),
        }),
      );
    });
  });

  describe('GeneticBundlingClient', () => {
    it('should bundle multiple vectors', async () => {
      const client = mockClient();
      vi.mocked(client.callZome).mockResolvedValue(fakeHash);
      const bundling = new GeneticBundlingClient(client);

      const result = await bundling.bundleVectors({
        patientHash: fakeHash,
        vectorHashes: [fakeHash, fakeHash],
        weights: [0.6, 0.4],
      });

      expect(result).toBe(fakeHash);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'bundle_genetic_vectors' }),
      );
    });
  });

  describe('GeneticCodebookClient', () => {
    it('should initialize codebook', async () => {
      const client = mockClient();
      vi.mocked(client.callZome).mockResolvedValue(fakeHash);
      const codebook = new GeneticCodebookClient(client);

      const result = await codebook.initCodebook();

      expect(result).toBe(fakeHash);
      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ fn_name: 'init_codebook' }),
      );
    });
  });

  describe('GeneticsBridge', () => {
    let client: ZomeCallable;
    let bridge: GeneticsBridge;

    beforeEach(() => {
      client = mockClient();
      bridge = new GeneticsBridge(client);
    });

    it('should expose all sub-clients', () => {
      expect(bridge.encoding).toBeInstanceOf(GeneticEncodingClient);
      expect(bridge.similarity).toBeInstanceOf(GeneticSimilarityClient);
      expect(bridge.bundling).toBeInstanceOf(GeneticBundlingClient);
      expect(bridge.codebook).toBeInstanceOf(GeneticCodebookClient);
    });

    it('should encode and find similar in one workflow', async () => {
      vi.mocked(client.callZome)
        .mockResolvedValueOnce(fakeHash) // encodeDnaSequence
        .mockResolvedValueOnce([
          // searchSimilar
          { vectorHash: fakeHash, patientHash: fakeHash, similarityScore: 0.91 },
        ]);

      const result = await bridge.encodeAndFindSimilar(
        fakeHash,
        'ATCGATCG',
        { sourceSystem: 'lab' },
        0.7,
        10,
      );

      expect(result.vectorHash).toBe(fakeHash);
      expect(result.similarPatients).toHaveLength(1);
    });

    it('should find organ donor matches workflow', async () => {
      vi.mocked(client.callZome)
        .mockResolvedValueOnce(fakeHash) // encodeHlaTyping
        .mockResolvedValueOnce([
          // searchSimilar
          { vectorHash: fakeHash, patientHash: fakeHash, similarityScore: 0.95 },
        ]);

      const result = await bridge.findOrganDonorMatches(
        fakeHash,
        ['A*02:01', 'B*07:02'],
        { sourceSystem: 'hla-lab' },
        0.85,
      );

      expect(result.recipientVectorHash).toBe(fakeHash);
      expect(result.potentialDonors).toHaveLength(1);
    });

    it('should get patient genetic profile', async () => {
      const vectors: GeneticHypervector[] = [
        {
          vectorId: 'v-1',
          patientHash: fakeHash,
          data: new Uint8Array(128),
          encodingType: 'DnaSequence',
          kmerLength: 3,
          kmerCount: 100,
          createdAt: Date.now(),
          sourceMetadata: { sourceSystem: 'lab' },
        },
        {
          vectorId: 'v-2',
          patientHash: fakeHash,
          data: new Uint8Array(128),
          encodingType: 'HlaTyping',
          kmerLength: 5,
          kmerCount: 50,
          createdAt: Date.now(),
          sourceMetadata: { sourceSystem: 'hla' },
        },
      ];
      vi.mocked(client.callZome).mockResolvedValue(vectors);

      const profile = await bridge.getPatientGeneticProfile(fakeHash);

      expect(profile.totalEncodings).toBe(2);
      expect(profile.byType['DnaSequence']).toHaveLength(1);
      expect(profile.byType['HlaTyping']).toHaveLength(1);
    });
  });

  describe('interpretHlaSimilarity (utility)', () => {
    it('should return Excellent for score >= 0.95', () => {
      const result = interpretHlaSimilarity(0.98);
      expect(result.matchQuality).toBe('Excellent');
    });

    it('should return Good for score >= 0.85', () => {
      const result = interpretHlaSimilarity(0.87);
      expect(result.matchQuality).toBe('Good');
    });

    it('should return Moderate for score >= 0.70', () => {
      const result = interpretHlaSimilarity(0.72);
      expect(result.matchQuality).toBe('Moderate');
    });

    it('should return Poor for score >= 0.50', () => {
      const result = interpretHlaSimilarity(0.55);
      expect(result.matchQuality).toBe('Poor');
    });

    it('should return Unsuitable for score < 0.50', () => {
      const result = interpretHlaSimilarity(0.30);
      expect(result.matchQuality).toBe('Unsuitable');
    });
  });

  describe('calculateEncodingConfidence (utility)', () => {
    it('should return High confidence for kmerCount >= 10000', () => {
      const result = calculateEncodingConfidence(15000);
      expect(result.confidence).toBe('High');
      expect(result.percentage).toBe(95);
    });

    it('should return Medium confidence for kmerCount >= 1000', () => {
      const result = calculateEncodingConfidence(5000);
      expect(result.confidence).toBe('Medium');
      expect(result.percentage).toBe(75);
    });

    it('should return Low confidence for kmerCount < 1000', () => {
      const result = calculateEncodingConfidence(500);
      expect(result.confidence).toBe('Low');
      expect(result.percentage).toBe(50);
    });
  });

  describe('createGeneticsBridge (factory)', () => {
    it('should create a GeneticsBridge instance', () => {
      const bridge = createGeneticsBridge(mockClient());
      expect(bridge).toBeInstanceOf(GeneticsBridge);
    });
  });
});
