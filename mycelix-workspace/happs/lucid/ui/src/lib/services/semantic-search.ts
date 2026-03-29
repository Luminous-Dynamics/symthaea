// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Semantic Search Service
 *
 * Hybrid search architecture with Symthaea integration:
 * - SYMTHAEA (Primary): 16,384-dimensional HDC embeddings via Tauri backend
 * - FALLBACK: transformers.js for when Symthaea is unavailable
 *
 * The service automatically initializes Symthaea and falls back to
 * transformers.js if Symthaea fails to initialize.
 */

import { get, writable } from 'svelte/store';
import { get as idbGet, set as idbSet, del as idbDel } from 'idb-keyval';
import type { Thought } from '@mycelix/lucid-client';
import { HNSWIndex, getHDCIndex, getTransformersIndex } from './hnsw-index';

// ============================================================================
// TYPES
// ============================================================================

export interface SearchResult {
	thought: Thought;
	score: number;
	matchType: 'semantic' | 'keyword' | 'hybrid';
	highlights?: string[];
}

export interface EmbeddingCache {
	thoughtId: string;
	embedding: number[];
	updatedAt: number;
}

export interface SearchOptions {
	limit?: number;
	threshold?: number;
	includeKeyword?: boolean;
	searchCollective?: boolean;
}

export interface CollectiveSearchResult {
	beliefHash: string;
	content: string;
	resonanceScore: number;
	consensusType?: string;
	source: 'collective';
}

/** E/N/M/H epistemic classification from Symthaea */
export interface EpistemicClassification {
	empirical: number; // E0-E4
	normative: number; // N0-N3
	materiality: number; // M0-M3
	harmonic: number; // H0-H4
}

/** Analyzed thought from Symthaea */
export interface AnalyzedThought {
	content: string;
	thought_type: string;
	epistemic: EpistemicClassification;
	embedding: number[];
	confidence: number;
	phi: number;
	coherence: number;
	tags: string[];
	domain: string | null;
	safe: boolean;
}

/** Symthaea status */
export interface SymthaeaStatus {
	ready: boolean;
	initializing: boolean;
	dimension: number;
}

// ============================================================================
// STATE
// ============================================================================

export const searchReady = writable(false);
export const searchLoading = writable(false);
export const embeddingProgress = writable(0);
export const symthaeaReady = writable(false);
export const symthaeaStatus = writable<SymthaeaStatus | null>(null);

let pipeline: any = null;
let embeddingCache: Map<string, number[]> = new Map();
let useSymthaea = false;
let hnswIndex: HNSWIndex | null = null;
let useHNSW = false;

// Dimensions
const TRANSFORMERS_DIM = 384; // all-MiniLM-L6-v2 dimension
const SYMTHAEA_DIM = 16384; // HDC dimension

const CACHE_KEY_TRANSFORMERS = 'lucid-embeddings-v1';
const CACHE_KEY_SYMTHAEA = 'lucid-embeddings-hdc-v1';

// ============================================================================
// TAURI INVOCATIONS
// ============================================================================

async function isTauri(): Promise<boolean> {
	return typeof window !== 'undefined' && '__TAURI__' in window;
}

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
	if (!(await isTauri())) {
		throw new Error('Not running in Tauri');
	}
	const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
	return tauriInvoke<T>(command, args);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the semantic search system
 *
 * Attempts to initialize Symthaea first (16,384D HDC embeddings).
 * Falls back to transformers.js (384D) if Symthaea is unavailable.
 */
export async function initializeSearch(): Promise<boolean> {
	try {
		searchLoading.set(true);

		// Try Symthaea first (Tauri backend)
		if (await isTauri()) {
			try {
				const status = await invoke<SymthaeaStatus>('get_symthaea_status');
				symthaeaStatus.set(status);

				if (status.ready) {
					console.log(`Symthaea ready: ${status.dimension}D HDC embeddings`);
					useSymthaea = true;
					symthaeaReady.set(true);

					// Load Symthaea cache
					const cached = await idbGet(CACHE_KEY_SYMTHAEA);
					if (cached) {
						embeddingCache = new Map(Object.entries(cached));
						console.log(`Loaded ${embeddingCache.size} cached HDC embeddings`);
					}

					searchReady.set(true);
					searchLoading.set(false);
					return true;
				} else if (!status.initializing) {
					// Try to initialize Symthaea
					console.log('Initializing Symthaea...');
					const initResult = await invoke<{ success: boolean; message: string; dimension: number }>(
						'initialize_symthaea'
					);

					if (initResult.success) {
						console.log(`Symthaea initialized: ${initResult.dimension}D HDC embeddings`);
						useSymthaea = true;
						symthaeaReady.set(true);
						symthaeaStatus.set({ ready: true, initializing: false, dimension: initResult.dimension });

						searchReady.set(true);
						searchLoading.set(false);
						return true;
					}
				}
			} catch (e) {
				console.warn('Symthaea unavailable, falling back to transformers.js:', e);
			}
		}

		// Fallback: transformers.js
		console.log('Using transformers.js fallback');
		useSymthaea = false;

		// Load cached embeddings from IndexedDB
		const cached = await idbGet(CACHE_KEY_TRANSFORMERS);
		if (cached) {
			embeddingCache = new Map(Object.entries(cached));
			console.log(`Loaded ${embeddingCache.size} cached transformer embeddings`);
		}

		// Dynamically import transformers.js
		const { pipeline: createPipeline } = await import('@xenova/transformers');

		// Load the embedding model
		pipeline = await createPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
			progress_callback: (progress: any) => {
				if (progress.status === 'progress') {
					embeddingProgress.set(Math.round(progress.progress));
				}
			},
		});

		searchReady.set(true);
		searchLoading.set(false);
		console.log('Semantic search initialized (transformers.js)');
		return true;
	} catch (error) {
		console.error('Failed to initialize semantic search:', error);
		searchLoading.set(false);
		return false;
	}
}

/**
 * Listen for Symthaea ready event from Tauri
 */
export async function listenForSymthaeaReady(): Promise<void> {
	if (!(await isTauri())) return;

	try {
		const { listen } = await import('@tauri-apps/api/event');
		await listen<{ success: boolean; dimension: number }>('symthaea-ready', (event) => {
			if (event.payload.success) {
				console.log(`Symthaea ready event: ${event.payload.dimension}D`);
				useSymthaea = true;
				symthaeaReady.set(true);
				symthaeaStatus.set({
					ready: true,
					initializing: false,
					dimension: event.payload.dimension,
				});

				// Clear old transformer cache and use new HDC cache
				embeddingCache.clear();
			}
		});
	} catch (e) {
		console.warn('Failed to listen for Symthaea ready event:', e);
	}
}

// ============================================================================
// EMBEDDING FUNCTIONS
// ============================================================================

/**
 * Generate embedding for text
 *
 * Uses Symthaea (16,384D) if available, otherwise transformers.js (384D).
 */
export async function embed(text: string): Promise<number[]> {
	if (useSymthaea) {
		return invoke<number[]>('embed_text', { text });
	}

	if (!pipeline) {
		throw new Error('Search not initialized');
	}

	const output = await pipeline(text, { pooling: 'mean', normalize: true });
	return Array.from(output.data);
}

/**
 * Generate and cache embedding for a thought
 */
export async function embedThought(thought: Thought): Promise<number[]> {
	// Check cache first
	const cached = embeddingCache.get(thought.id);
	if (cached) {
		return cached;
	}

	// Generate new embedding
	const text = prepareThoughtText(thought);
	const embedding = await embed(text);

	// Cache it
	embeddingCache.set(thought.id, embedding);

	// Persist to IndexedDB (debounced)
	debouncedSaveCache();

	return embedding;
}

/**
 * Prepare thought text for embedding
 */
function prepareThoughtText(thought: Thought): string {
	const parts = [thought.content];

	// Include type for context
	if (thought.thought_type) {
		parts.unshift(`[${thought.thought_type}]`);
	}

	// Include tags
	if (thought.tags?.length) {
		parts.push(`Tags: ${thought.tags.join(', ')}`);
	}

	return parts.join(' ');
}

/**
 * Batch embed all thoughts and build HNSW index
 */
export async function embedAllThoughts(thoughts: Thought[]): Promise<void> {
	searchLoading.set(true);
	embeddingProgress.set(0);

	const total = thoughts.length;
	let processed = 0;

	// Initialize HNSW index
	hnswIndex = useSymthaea ? getHDCIndex() : getTransformersIndex();

	// Try to load existing HNSW index
	const indexLoaded = await hnswIndex.load();
	if (indexLoaded) {
		console.log(`Loaded HNSW index with ${hnswIndex.size} nodes`);
	}

	// Batch processing for Symthaea
	if (useSymthaea) {
		const batchSize = 50;
		for (let i = 0; i < thoughts.length; i += batchSize) {
			const batch = thoughts.slice(i, i + batchSize);
			const texts = batch.map((t) => prepareThoughtText(t));
			const embeddings = await invoke<number[][]>('batch_embed', { texts });

			for (let j = 0; j < batch.length; j++) {
				const thought = batch[j];
				const embedding = embeddings[j];
				embeddingCache.set(thought.id, embedding);

				// Add to HNSW index
				if (!hnswIndex.has(thought.id)) {
					hnswIndex.add(thought.id, embedding);
				}
			}

			processed += batch.length;
			embeddingProgress.set(Math.round((processed / total) * 100));
		}
	} else {
		// Sequential for transformers.js
		for (const thought of thoughts) {
			if (!embeddingCache.has(thought.id)) {
				const embedding = await embedThought(thought);

				// Add to HNSW index
				if (!hnswIndex.has(thought.id)) {
					hnswIndex.add(thought.id, embedding);
				}
			}
			processed++;
			embeddingProgress.set(Math.round((processed / total) * 100));
		}
	}

	// Save caches
	await saveCache();
	await hnswIndex.save();

	// Enable HNSW search if we have enough nodes
	useHNSW = hnswIndex.size >= 10;
	if (useHNSW) {
		const stats = hnswIndex.getStats();
		console.log(`HNSW index ready: ${stats.nodeCount} nodes, ${stats.memoryEstimateMB}MB`);
	}

	searchLoading.set(false);
}

// ============================================================================
// SYMTHAEA-SPECIFIC FUNCTIONS
// ============================================================================

/**
 * Analyze thought content through Symthaea
 *
 * Returns full analysis including epistemic classification,
 * HDC embedding, and confidence scores.
 */
export async function analyzeThought(content: string): Promise<AnalyzedThought> {
	if (!useSymthaea) {
		throw new Error('Symthaea not available');
	}

	return invoke<AnalyzedThought>('analyze_thought', { content });
}

/**
 * Check coherence across multiple thoughts via Symthaea
 */
export async function checkCoherence(thoughtContents: string[]): Promise<{
	overall: number;
	logical: number;
	temporal: number;
	epistemic: number;
	harmonic: number;
	contradictions: Array<{
		thought_a: string;
		thought_b: string;
		description: string;
		severity: number;
	}>;
}> {
	if (!useSymthaea) {
		// Fallback: simple coherence estimation
		return {
			overall: 0.7,
			logical: 0.7,
			temporal: 1.0,
			epistemic: 0.7,
			harmonic: 0.7,
			contradictions: [],
		};
	}

	return invoke('check_coherence', { thoughtContents });
}

/**
 * Classify text epistemically
 */
export async function classifyEpistemic(
	content: string,
	confidence: number = 0.5
): Promise<EpistemicClassification> {
	if (!useSymthaea) {
		// Fallback: default classification
		return {
			empirical: 0,
			normative: 0,
			materiality: 0,
			harmonic: 1,
		};
	}

	return invoke('classify_epistemic', { content, confidence });
}

/**
 * Get current embedding dimension
 */
export function getEmbeddingDimension(): number {
	return useSymthaea ? SYMTHAEA_DIM : TRANSFORMERS_DIM;
}

/**
 * Check if using Symthaea backend
 */
export function isUsingSymthaea(): boolean {
	return useSymthaea;
}

// ============================================================================
// LOCAL SEARCH
// ============================================================================

/**
 * Search thoughts by semantic similarity
 *
 * Uses HNSW index for O(log n) approximate nearest neighbor search
 * when available, falling back to O(n) linear search otherwise.
 */
export async function searchLocal(
	query: string,
	thoughts: Thought[],
	options: SearchOptions = {}
): Promise<SearchResult[]> {
	const { limit = 10, threshold = 0.3, includeKeyword = true } = options;

	if (!get(searchReady)) {
		// Fallback to keyword search
		return keywordSearch(query, thoughts, limit);
	}

	const queryEmbedding = await embed(query);

	// Use HNSW index for fast approximate search if available
	if (useHNSW && hnswIndex && hnswIndex.size >= 10) {
		return searchWithHNSW(query, queryEmbedding, thoughts, options);
	}

	// Fallback to linear search
	return searchLinear(query, queryEmbedding, thoughts, options);
}

/**
 * HNSW-accelerated search (O(log n))
 */
async function searchWithHNSW(
	query: string,
	queryEmbedding: number[],
	thoughts: Thought[],
	options: SearchOptions
): Promise<SearchResult[]> {
	const { limit = 10, threshold = 0.3, includeKeyword = true } = options;

	// Get approximate nearest neighbors from HNSW
	const candidates = hnswIndex!.search(queryEmbedding, limit * 2);

	// Build a map for quick thought lookup
	const thoughtMap = new Map(thoughts.map((t) => [t.id, t]));

	const results: SearchResult[] = [];

	for (const candidate of candidates) {
		const thought = thoughtMap.get(candidate.id);
		if (!thought) continue;

		// Distance to similarity (HNSW returns 1 - cosine similarity)
		let semanticScore = 1 - candidate.distance;

		let score = semanticScore;
		let matchType: SearchResult['matchType'] = 'semantic';

		if (includeKeyword) {
			const keywordScore = calculateKeywordScore(query, thought);
			score = semanticScore * 0.7 + keywordScore * 0.3;
			matchType = 'hybrid';
		}

		if (score >= threshold) {
			results.push({
				thought,
				score,
				matchType,
				highlights: findHighlights(query, thought.content),
			});
		}
	}

	// Sort by score descending
	results.sort((a, b) => b.score - a.score);

	return results.slice(0, limit);
}

/**
 * Linear search fallback (O(n))
 */
async function searchLinear(
	query: string,
	queryEmbedding: number[],
	thoughts: Thought[],
	options: SearchOptions
): Promise<SearchResult[]> {
	const { limit = 10, threshold = 0.3, includeKeyword = true } = options;
	const results: SearchResult[] = [];

	for (const thought of thoughts) {
		let score = 0;
		let matchType: SearchResult['matchType'] = 'semantic';

		// Semantic similarity
		const thoughtEmbedding = await embedThought(thought);
		const semanticScore = cosineSimilarity(queryEmbedding, thoughtEmbedding);

		if (includeKeyword) {
			// Hybrid: combine semantic + keyword
			const keywordScore = calculateKeywordScore(query, thought);
			score = semanticScore * 0.7 + keywordScore * 0.3;
			matchType = 'hybrid';
		} else {
			score = semanticScore;
		}

		if (score >= threshold) {
			results.push({
				thought,
				score,
				matchType,
				highlights: findHighlights(query, thought.content),
			});
		}
	}

	// Sort by score descending
	results.sort((a, b) => b.score - a.score);

	return results.slice(0, limit);
}

/**
 * Keyword-based search (fallback)
 */
function keywordSearch(query: string, thoughts: Thought[], limit: number): SearchResult[] {
	const results: SearchResult[] = [];

	for (const thought of thoughts) {
		const score = calculateKeywordScore(query, thought);
		if (score > 0) {
			results.push({
				thought,
				score,
				matchType: 'keyword',
				highlights: findHighlights(query, thought.content),
			});
		}
	}

	results.sort((a, b) => b.score - a.score);
	return results.slice(0, limit);
}

/**
 * Calculate keyword match score
 */
function calculateKeywordScore(query: string, thought: Thought): number {
	const queryTerms = query.toLowerCase().split(/\s+/);
	const content = thought.content.toLowerCase();
	const tags = (thought.tags || []).map((t) => t.toLowerCase());

	let matches = 0;
	let tagMatches = 0;

	for (const term of queryTerms) {
		if (content.includes(term)) matches++;
		if (tags.some((tag) => tag.includes(term))) tagMatches++;
	}

	// Weight tag matches higher
	return (matches / queryTerms.length) * 0.7 + (tagMatches / queryTerms.length) * 0.3;
}

/**
 * Find text highlights
 */
function findHighlights(query: string, content: string): string[] {
	const terms = query.toLowerCase().split(/\s+/);
	const highlights: string[] = [];
	const contentLower = content.toLowerCase();

	for (const term of terms) {
		const index = contentLower.indexOf(term);
		if (index !== -1) {
			const start = Math.max(0, index - 30);
			const end = Math.min(content.length, index + term.length + 30);
			let highlight = content.slice(start, end);
			if (start > 0) highlight = '...' + highlight;
			if (end < content.length) highlight = highlight + '...';
			highlights.push(highlight);
		}
	}

	return highlights.slice(0, 3);
}

// ============================================================================
// SYMTHAEA INTEGRATION (Collective Search)
// ============================================================================

/**
 * Search for resonant beliefs in the collective via Symthaea
 */
export async function searchCollective(
	query: string,
	options: SearchOptions = {}
): Promise<CollectiveSearchResult[]> {
	// Collective search requires future Holochain DHT integration
	console.log('Collective search not yet connected to Symthaea DHT');
	return [];
}

/**
 * Find thoughts that resonate with a collective pattern
 *
 * This fetches the pattern's embedding from Symthaea and compares it
 * against local thought embeddings to find resonant content.
 */
export async function findResonance(patternId: string, thoughts: Thought[]): Promise<SearchResult[]> {
	if (!get(searchReady) || thoughts.length === 0) {
		return [];
	}

	try {
		let patternEmbedding: number[] | null = null;

		// Try to get pattern embedding from Symthaea via Tauri
		if (useSymthaea) {
			try {
				patternEmbedding = await invoke<number[]>('get_pattern_embedding', { patternId });
			} catch (e) {
				console.warn('Failed to get pattern embedding from Symthaea:', e);
			}
		}

		// If no pattern embedding available, try to synthesize from pattern description
		if (!patternEmbedding) {
			// Fallback: embed the pattern ID as a query (patterns often have descriptive IDs)
			patternEmbedding = await embed(patternId);
		}

		if (!patternEmbedding || patternEmbedding.length === 0) {
			return [];
		}

		// Compare pattern embedding against all thoughts
		const results: SearchResult[] = [];

		for (const thought of thoughts) {
			const thoughtEmbedding = await embedThought(thought);
			const score = await computeSimilarity(patternEmbedding, thoughtEmbedding);

			// Resonance threshold - patterns need higher similarity for meaningful resonance
			const RESONANCE_THRESHOLD = 0.6;

			if (score >= RESONANCE_THRESHOLD) {
				results.push({
					thought,
					score,
					matchType: 'semantic',
					highlights: extractResonanceHighlights(thought.content, patternId),
				});
			}
		}

		// Sort by resonance score (highest first)
		results.sort((a, b) => b.score - a.score);

		return results;
	} catch (error) {
		console.error('Error finding resonance:', error);
		return [];
	}
}

/**
 * Extract highlights showing why a thought resonates with a pattern
 */
function extractResonanceHighlights(content: string, patternId: string): string[] {
	const highlights: string[] = [];
	const words = patternId.toLowerCase().split(/[\s_-]+/);

	// Find sentences containing pattern-related words
	const sentences = content.split(/[.!?]+/).filter(s => s.trim());
	for (const sentence of sentences) {
		const lowerSentence = sentence.toLowerCase();
		if (words.some(word => word.length > 3 && lowerSentence.includes(word))) {
			highlights.push(sentence.trim());
		}
	}

	return highlights.slice(0, 3); // Limit to 3 highlights
}

// ============================================================================
// SIMILARITY
// ============================================================================

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
	if (a.length !== b.length) return 0;

	let dotProduct = 0;
	let normA = 0;
	let normB = 0;

	for (let i = 0; i < a.length; i++) {
		dotProduct += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}

	const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
	return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Compute similarity via Tauri backend (more efficient for large vectors)
 */
export async function computeSimilarity(embeddingA: number[], embeddingB: number[]): Promise<number> {
	if (useSymthaea && embeddingA.length === SYMTHAEA_DIM) {
		return invoke<number>('compute_similarity', {
			embeddingA,
			embeddingB,
		});
	}
	return cosineSimilarity(embeddingA, embeddingB);
}

/**
 * Find similar thoughts to a given thought
 */
export async function findSimilar(
	thought: Thought,
	allThoughts: Thought[],
	limit: number = 5
): Promise<SearchResult[]> {
	if (!get(searchReady)) return [];

	const sourceEmbedding = await embedThought(thought);
	const results: SearchResult[] = [];

	for (const candidate of allThoughts) {
		if (candidate.id === thought.id) continue;

		const candidateEmbedding = await embedThought(candidate);
		const score = cosineSimilarity(sourceEmbedding, candidateEmbedding);

		if (score > 0.5) {
			results.push({
				thought: candidate,
				score,
				matchType: 'semantic',
			});
		}
	}

	results.sort((a, b) => b.score - a.score);
	return results.slice(0, limit);
}

// ============================================================================
// CACHE MANAGEMENT
// ============================================================================

let saveTimeout: ReturnType<typeof setTimeout> | null = null;

function debouncedSaveCache() {
	if (saveTimeout) clearTimeout(saveTimeout);
	saveTimeout = setTimeout(saveCache, 5000);
}

async function saveCache() {
	try {
		const cacheObject: Record<string, number[]> = {};
		for (const [key, value] of embeddingCache) {
			cacheObject[key] = value;
		}
		const cacheKey = useSymthaea ? CACHE_KEY_SYMTHAEA : CACHE_KEY_TRANSFORMERS;
		await idbSet(cacheKey, cacheObject);
	} catch (error) {
		console.error('Failed to save embedding cache:', error);
	}
}

/**
 * Clear the embedding cache
 */
export async function clearCache(): Promise<void> {
	embeddingCache.clear();
	await idbDel(CACHE_KEY_TRANSFORMERS);
	await idbDel(CACHE_KEY_SYMTHAEA);
}

/**
 * Remove a thought from the cache and HNSW index
 */
export function removeFromCache(thoughtId: string): void {
	embeddingCache.delete(thoughtId);
	if (hnswIndex) {
		hnswIndex.remove(thoughtId);
	}
	debouncedSaveCache();
}

/**
 * Get cache statistics
 */
export function getCacheStats(): {
	size: number;
	thoughtIds: string[];
	dimension: number;
	backend: 'symthaea' | 'transformers';
	hnswEnabled: boolean;
	hnswStats?: {
		nodeCount: number;
		maxLayer: number;
		avgConnections: number;
		memoryEstimateMB: number;
	};
} {
	return {
		size: embeddingCache.size,
		thoughtIds: Array.from(embeddingCache.keys()),
		dimension: getEmbeddingDimension(),
		backend: useSymthaea ? 'symthaea' : 'transformers',
		hnswEnabled: useHNSW,
		hnswStats: hnswIndex ? hnswIndex.getStats() : undefined,
	};
}

/**
 * Add a single thought to the HNSW index
 */
export async function addToIndex(thought: Thought): Promise<void> {
	if (!hnswIndex) {
		hnswIndex = useSymthaea ? getHDCIndex() : getTransformersIndex();
	}

	const embedding = await embedThought(thought);
	if (!hnswIndex.has(thought.id)) {
		hnswIndex.add(thought.id, embedding);
	}
}

/**
 * Rebuild the HNSW index from scratch
 */
export async function rebuildIndex(thoughts: Thought[]): Promise<void> {
	// Clear and rebuild
	hnswIndex = useSymthaea ? getHDCIndex() : getTransformersIndex();
	await hnswIndex.clear();

	for (const thought of thoughts) {
		const embedding = embeddingCache.get(thought.id);
		if (embedding) {
			hnswIndex.add(thought.id, embedding);
		}
	}

	await hnswIndex.save();
	useHNSW = hnswIndex.size >= 10;
}
