<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { thoughts, selectedThought } from '$stores/thoughts';
	import {
		searchLocal,
		findSimilar,
		searchReady,
		searchLoading,
		embeddingProgress,
		embedAllThoughts,
		getCacheStats,
		type SearchResult,
	} from '$services/semantic-search';

	let query = $state('');
	let results: SearchResult[] = $state([]);
	let searching = $state(false);
	let showSimilar = $state(false);
	let similarResults: SearchResult[] = $state([]);

	let debounceTimer: ReturnType<typeof setTimeout>;

	function handleInput() {
		clearTimeout(debounceTimer);
		if (query.trim().length < 2) {
			results = [];
			return;
		}
		debounceTimer = setTimeout(doSearch, 300);
	}

	async function doSearch() {
		if (!query.trim()) {
			results = [];
			return;
		}

		searching = true;
		try {
			results = await searchLocal(query, $thoughts, {
				limit: 15,
				threshold: 0.25,
				includeKeyword: true,
			});
		} catch (error) {
			console.error('Search failed:', error);
			results = [];
		}
		searching = false;
	}

	async function findSimilarThoughts() {
		if (!$selectedThought) return;
		showSimilar = true;
		similarResults = await findSimilar($selectedThought, $thoughts, 5);
	}

	function selectResult(result: SearchResult) {
		selectedThought.set(result.thought);
	}

	function getScoreColor(score: number): string {
		if (score >= 0.7) return 'text-green-400';
		if (score >= 0.5) return 'text-yellow-400';
		return 'text-gray-400';
	}

	function getMatchTypeLabel(type: string): string {
		switch (type) {
			case 'semantic':
				return 'Meaning';
			case 'keyword':
				return 'Keyword';
			case 'hybrid':
				return 'Hybrid';
			default:
				return type;
		}
	}

	async function indexAll() {
		await embedAllThoughts($thoughts);
	}

	const cacheStats = $derived(getCacheStats());
</script>

<div class="semantic-search">
	<div class="search-header">
		<h3>Semantic Search</h3>
		{#if !$searchReady}
			<div class="status loading">
				{#if $searchLoading}
					<span>Loading model... {$embeddingProgress}%</span>
				{:else}
					<span>Initializing...</span>
				{/if}
			</div>
		{:else}
			<div class="status ready">
				<span class="dot"></span>
				<span>Ready</span>
			</div>
		{/if}
	</div>

	<div class="search-input-container">
		<input
			type="text"
			bind:value={query}
			oninput={handleInput}
			placeholder="Search by meaning..."
			class="search-input"
			disabled={!$searchReady}
		/>
		{#if searching}
			<div class="search-spinner"></div>
		{/if}
	</div>

	{#if results.length > 0}
		<div class="results">
			<div class="results-header">
				<span>{results.length} results</span>
			</div>
			<div class="results-list">
				{#each results as result}
					<button class="result-item" onclick={() => selectResult(result)}>
						<div class="result-content">
							<span class="result-type">{result.thought.thought_type}</span>
							<p class="result-text">{result.thought.content.slice(0, 120)}...</p>
							{#if result.highlights?.length}
								<div class="highlights">
									{#each result.highlights.slice(0, 2) as highlight}
										<span class="highlight">...{highlight}...</span>
									{/each}
								</div>
							{/if}
						</div>
						<div class="result-meta">
							<span class="score {getScoreColor(result.score)}">
								{Math.round(result.score * 100)}%
							</span>
							<span class="match-type">{getMatchTypeLabel(result.matchType)}</span>
						</div>
					</button>
				{/each}
			</div>
		</div>
	{:else if query.trim().length >= 2 && !searching}
		<div class="no-results">
			<p>No matching thoughts found</p>
			<p class="hint">Try different keywords or phrases</p>
		</div>
	{/if}

	<!-- Similar thoughts section -->
	{#if $selectedThought}
		<div class="similar-section">
			<button class="similar-btn" onclick={findSimilarThoughts}>
				Find Similar to Selected
			</button>

			{#if showSimilar && similarResults.length > 0}
				<div class="similar-results">
					<h4>Similar Thoughts</h4>
					{#each similarResults as result}
						<button class="similar-item" onclick={() => selectResult(result)}>
							<span class="similar-score {getScoreColor(result.score)}">
								{Math.round(result.score * 100)}%
							</span>
							<span class="similar-text">{result.thought.content.slice(0, 80)}...</span>
						</button>
					{/each}
				</div>
			{:else if showSimilar}
				<p class="no-similar">No similar thoughts found</p>
			{/if}
		</div>
	{/if}

	<!-- Index management -->
	<div class="index-section">
		<div class="index-stats">
			<span>{cacheStats.size} thoughts indexed</span>
			<span class="separator">•</span>
			<span>{$thoughts.length} total</span>
		</div>
		{#if cacheStats.size < $thoughts.length}
			<button class="index-btn" onclick={indexAll} disabled={$searchLoading}>
				{#if $searchLoading}
					Indexing... {$embeddingProgress}%
				{:else}
					Index All Thoughts
				{/if}
			</button>
		{/if}
	</div>
</div>

<style>
	.semantic-search {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding-top: 24px;
	}

	.search-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	h3 {
		margin: 0;
		font-size: 1rem;
		color: #e5e5e5;
	}

	.status {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 0.75rem;
	}

	.status.loading {
		color: #f59e0b;
	}

	.status.ready {
		color: #10b981;
	}

	.status .dot {
		width: 6px;
		height: 6px;
		background: #10b981;
		border-radius: 50%;
	}

	.search-input-container {
		position: relative;
	}

	.search-input {
		width: 100%;
		padding: 12px 16px;
		background: #252540;
		border: 1px solid #3a3a5e;
		border-radius: 8px;
		color: #e5e5e5;
		font-size: 0.9rem;
	}

	.search-input:focus {
		outline: none;
		border-color: #7c3aed;
	}

	.search-input:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.search-spinner {
		position: absolute;
		right: 12px;
		top: 50%;
		transform: translateY(-50%);
		width: 16px;
		height: 16px;
		border: 2px solid #3a3a5e;
		border-top-color: #7c3aed;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: translateY(-50%) rotate(360deg);
		}
	}

	.results {
		background: #1e1e2e;
		border: 1px solid #2a2a4e;
		border-radius: 8px;
		overflow: hidden;
	}

	.results-header {
		padding: 8px 12px;
		background: #252540;
		font-size: 0.75rem;
		color: #888;
	}

	.results-list {
		max-height: 400px;
		overflow-y: auto;
	}

	.result-item {
		display: flex;
		width: 100%;
		text-align: left;
		padding: 12px;
		background: transparent;
		border: none;
		border-bottom: 1px solid #2a2a4e;
		color: #e5e5e5;
		cursor: pointer;
		transition: background 0.2s;
	}

	.result-item:hover {
		background: #252540;
	}

	.result-item:last-child {
		border-bottom: none;
	}

	.result-content {
		flex: 1;
		min-width: 0;
	}

	.result-type {
		font-size: 0.7rem;
		color: #7c3aed;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.result-text {
		margin: 4px 0;
		font-size: 0.85rem;
		line-height: 1.4;
	}

	.highlights {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin-top: 6px;
	}

	.highlight {
		font-size: 0.75rem;
		color: #888;
		background: #252540;
		padding: 2px 6px;
		border-radius: 4px;
	}

	.result-meta {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 4px;
		margin-left: 12px;
	}

	.score {
		font-size: 0.85rem;
		font-weight: 600;
	}

	.match-type {
		font-size: 0.65rem;
		color: #666;
		text-transform: uppercase;
	}

	.no-results {
		text-align: center;
		padding: 24px;
		color: #666;
	}

	.no-results .hint {
		font-size: 0.8rem;
		margin-top: 4px;
	}

	.similar-section {
		margin-top: 8px;
	}

	.similar-btn {
		width: 100%;
		padding: 10px;
		background: #252540;
		border: 1px solid #3a3a5e;
		border-radius: 8px;
		color: #e5e5e5;
		font-size: 0.85rem;
		cursor: pointer;
		transition: all 0.2s;
	}

	.similar-btn:hover {
		border-color: #7c3aed;
	}

	.similar-results {
		margin-top: 12px;
	}

	.similar-results h4 {
		margin: 0 0 8px;
		font-size: 0.8rem;
		color: #888;
	}

	.similar-item {
		display: flex;
		width: 100%;
		text-align: left;
		padding: 8px;
		background: #1e1e2e;
		border: 1px solid #2a2a4e;
		border-radius: 6px;
		color: #e5e5e5;
		cursor: pointer;
		margin-bottom: 6px;
		gap: 8px;
		align-items: flex-start;
	}

	.similar-item:hover {
		border-color: #7c3aed;
	}

	.similar-score {
		font-size: 0.75rem;
		font-weight: 600;
		flex-shrink: 0;
	}

	.similar-text {
		font-size: 0.8rem;
		line-height: 1.4;
	}

	.no-similar {
		text-align: center;
		padding: 12px;
		color: #666;
		font-size: 0.85rem;
	}

	.index-section {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px;
		background: #1e1e2e;
		border: 1px solid #2a2a4e;
		border-radius: 8px;
		font-size: 0.75rem;
	}

	.index-stats {
		color: #888;
	}

	.separator {
		margin: 0 6px;
	}

	.index-btn {
		padding: 6px 12px;
		background: #7c3aed;
		border: none;
		border-radius: 6px;
		color: white;
		font-size: 0.75rem;
		cursor: pointer;
	}

	.index-btn:hover:not(:disabled) {
		background: #6d28d9;
	}

	.index-btn:disabled {
		opacity: 0.7;
		cursor: not-allowed;
	}

	.text-green-400 {
		color: #4ade80;
	}
	.text-yellow-400 {
		color: #facc15;
	}
	.text-gray-400 {
		color: #9ca3af;
	}
</style>
