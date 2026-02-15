<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { isConnected, demoMode } from '$stores/holochain';
	import { thoughts, selectedThought, filteredThoughts } from '$stores/thoughts';

	// Components
	import ConnectionStatus from '$components/ConnectionStatus.svelte';
	import ThoughtForm from '$components/ThoughtForm.svelte';
	import ThoughtList from '$components/ThoughtList.svelte';
	import KnowledgeGraph from '$components/KnowledgeGraph.svelte';
	import GraphControls from '$components/GraphControls.svelte';
	import CommandPalette from '$components/CommandPalette.svelte';
	import KeyboardShortcuts from '$components/KeyboardShortcuts.svelte';
	import SemanticSearch from '$components/SemanticSearch.svelte';
	import AmbientCapture from '$components/AmbientCapture.svelte';
	import ReviewPanel from '$components/ReviewPanel.svelte';
	import CoherencePanel from '$components/CoherencePanel.svelte';
	import CognitiveAugmentation from '$components/CognitiveAugmentation.svelte';
	import TemporalConsciousness from '$components/TemporalConsciousness.svelte';
	import ImportExport from '$components/ImportExport.svelte';
	import CollectiveSensemakingPanel from '$components/CollectiveSensemakingPanel.svelte';
	import ConsciousnessDashboard from '$components/ConsciousnessDashboard.svelte';

	type ViewMode = 'list' | 'graph' | 'split';
	type PanelMode = 'none' | 'review' | 'coherence' | 'augment' | 'temporal' | 'search' | 'collective' | 'consciousness';

	let viewMode: ViewMode = $state('list');
	let activePanel: PanelMode = $state('none');

	// Graph settings
	let showLabels = $state(true);
	let clusterByDomain = $state(false);
	let linkStrength = $state(0.5);

	// Modals
	let showCommandPalette = $state(false);
	let showShortcuts = $state(false);
	let showImportExport = $state(false);

	function handleCommand(event: CustomEvent<{ type: string; value?: any }>) {
		const { type, value } = event.detail;

		switch (type) {
			case 'new-thought':
				const textarea = document.querySelector('.thought-form textarea') as HTMLTextAreaElement;
				textarea?.focus();
				break;
			case 'focus-search':
				activePanel = 'search';
				break;
			case 'view':
				viewMode = value;
				break;
			case 'show-shortcuts':
				showShortcuts = true;
				break;
			case 'export':
				showImportExport = true;
				break;
			case 'panel':
				activePanel = value;
				break;
		}
	}

	function handleGlobalKeydown(event: KeyboardEvent) {
		const target = event.target as HTMLElement;
		if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
			return;
		}

		// View switching
		if (event.key === '1') viewMode = 'list';
		else if (event.key === '2') viewMode = 'graph';
		else if (event.key === '3') viewMode = 'split';

		// Panel switching
		if (event.key === 'r' && event.ctrlKey) {
			event.preventDefault();
			activePanel = activePanel === 'review' ? 'none' : 'review';
		}
		if (event.key === 's' && event.ctrlKey && event.shiftKey) {
			event.preventDefault();
			activePanel = activePanel === 'search' ? 'none' : 'search';
		}

		// Show shortcuts
		if (event.key === '?') {
			event.preventDefault();
			showShortcuts = true;
		}

		// Navigation
		if (event.key === 'j' && $filteredThoughts.length > 0) {
			const currentIndex = $filteredThoughts.findIndex((t) => t.id === $selectedThought?.id);
			const nextIndex = Math.min(currentIndex + 1, $filteredThoughts.length - 1);
			selectedThought.set($filteredThoughts[nextIndex]);
		}
		if (event.key === 'k' && $filteredThoughts.length > 0) {
			const currentIndex = $filteredThoughts.findIndex((t) => t.id === $selectedThought?.id);
			const prevIndex = Math.max(currentIndex - 1, 0);
			selectedThought.set($filteredThoughts[prevIndex]);
		}

		// Escape to deselect/close
		if (event.key === 'Escape') {
			selectedThought.set(null);
			showShortcuts = false;
			activePanel = 'none';
		}
	}

	onMount(() => {
		window.addEventListener('keydown', handleGlobalKeydown);
	});

	onDestroy(() => {
		window.removeEventListener('keydown', handleGlobalKeydown);
	});
</script>

<div class="app">
	<header>
		<div class="brand">
			<h1>LUCID</h1>
			<span class="tagline">Living Unified Consciousness for Insight & Discovery</span>
			{#if $demoMode}
				<span class="demo-badge">DEMO</span>
			{/if}
		</div>

		{#if $isConnected}
			<div class="controls">
				<!-- View toggle -->
				<div class="view-toggle">
					<button
						class="view-btn"
						class:active={viewMode === 'list'}
						onclick={() => (viewMode = 'list')}
						title="List view (1)"
					>
						<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<line x1="8" y1="6" x2="21" y2="6"></line>
							<line x1="8" y1="12" x2="21" y2="12"></line>
							<line x1="8" y1="18" x2="21" y2="18"></line>
							<line x1="3" y1="6" x2="3.01" y2="6"></line>
							<line x1="3" y1="12" x2="3.01" y2="12"></line>
							<line x1="3" y1="18" x2="3.01" y2="18"></line>
						</svg>
					</button>
					<button
						class="view-btn"
						class:active={viewMode === 'graph'}
						onclick={() => (viewMode = 'graph')}
						title="Graph view (2)"
					>
						<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<circle cx="18" cy="5" r="3"></circle>
							<circle cx="6" cy="12" r="3"></circle>
							<circle cx="18" cy="19" r="3"></circle>
							<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
							<line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
						</svg>
					</button>
					<button
						class="view-btn"
						class:active={viewMode === 'split'}
						onclick={() => (viewMode = 'split')}
						title="Split view (3)"
					>
						<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
							<line x1="12" y1="3" x2="12" y2="21"></line>
						</svg>
					</button>
				</div>

				<!-- Panel toggle -->
				<div class="panel-toggle">
					<button
						class="panel-btn"
						class:active={activePanel === 'search'}
						onclick={() => (activePanel = activePanel === 'search' ? 'none' : 'search')}
						title="Semantic Search"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<circle cx="11" cy="11" r="8"></circle>
							<line x1="21" y1="21" x2="16.65" y2="16.65"></line>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'review'}
						onclick={() => (activePanel = activePanel === 'review' ? 'none' : 'review')}
						title="Review Session"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<path d="M12 2L2 7l10 5 10-5-10-5z"></path>
							<path d="M2 17l10 5 10-5"></path>
							<path d="M2 12l10 5 10-5"></path>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'coherence'}
						onclick={() => (activePanel = activePanel === 'coherence' ? 'none' : 'coherence')}
						title="Coherence Analysis"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<circle cx="12" cy="12" r="10"></circle>
							<path d="M12 16v-4M12 8h.01"></path>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'augment'}
						onclick={() => (activePanel = activePanel === 'augment' ? 'none' : 'augment')}
						title="Cognitive Augmentation"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
							<path d="M9 12l2 2 4-4"></path>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'temporal'}
						onclick={() => (activePanel = activePanel === 'temporal' ? 'none' : 'temporal')}
						title="Temporal Consciousness"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<circle cx="12" cy="12" r="10"></circle>
							<polyline points="12,6 12,12 16,14"></polyline>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'collective'}
						onclick={() => (activePanel = activePanel === 'collective' ? 'none' : 'collective')}
						title="Collective Sensemaking"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<circle cx="12" cy="5" r="3"></circle>
							<circle cx="5" cy="19" r="3"></circle>
							<circle cx="19" cy="19" r="3"></circle>
							<line x1="12" y1="8" x2="5" y2="16"></line>
							<line x1="12" y1="8" x2="19" y2="16"></line>
							<line x1="5" y1="19" x2="19" y2="19"></line>
						</svg>
					</button>
					<button
						class="panel-btn"
						class:active={activePanel === 'consciousness'}
						onclick={() => (activePanel = activePanel === 'consciousness' ? 'none' : 'consciousness')}
						title="Consciousness Dashboard"
					>
						<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
							<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z"></path>
							<line x1="9" y1="21" x2="15" y2="21"></line>
						</svg>
					</button>
				</div>

				<!-- Import/Export -->
				<button
					class="icon-btn"
					onclick={() => (showImportExport = true)}
					title="Import/Export"
				>
					<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
						<polyline points="7,10 12,15 17,10"></polyline>
						<line x1="12" y1="15" x2="12" y2="3"></line>
					</svg>
				</button>
			</div>
		{/if}

		<ConnectionStatus />
	</header>

	<main class:with-panel={activePanel !== 'none'}>
		{#if $isConnected}
			<div class="main-content">
				<section class="input-section">
					<ThoughtForm />
				</section>

				{#if viewMode === 'list'}
					<section class="thoughts-section">
						<ThoughtList />
					</section>
				{:else if viewMode === 'graph'}
					<section class="graph-section">
						<GraphControls bind:showLabels bind:clusterByDomain bind:linkStrength />
						<div class="graph-container">
							<KnowledgeGraph />
						</div>
					</section>
				{:else if viewMode === 'split'}
					<section class="split-section">
						<div class="split-left">
							<ThoughtList compact />
						</div>
						<div class="split-right">
							<KnowledgeGraph />
						</div>
					</section>
				{/if}
			</div>

			<!-- Side Panel -->
			{#if activePanel !== 'none'}
				<aside class="side-panel">
					<button class="close-panel" onclick={() => (activePanel = 'none')}>×</button>

					{#if activePanel === 'search'}
						<SemanticSearch />
					{:else if activePanel === 'review'}
						<ReviewPanel />
					{:else if activePanel === 'coherence'}
						<CoherencePanel />
					{:else if activePanel === 'augment'}
						{#if $selectedThought}
							<CognitiveAugmentation thought={$selectedThought} recentThoughts={$thoughts.slice(-10)} />
						{:else}
							<div class="panel-empty">
								<p>Select a thought to analyze</p>
							</div>
						{/if}
					{:else if activePanel === 'temporal'}
						<TemporalConsciousness thoughts={$thoughts} />
					{:else if activePanel === 'collective'}
						<CollectiveSensemakingPanel />
					{:else if activePanel === 'consciousness'}
						<ConsciousnessDashboard />
					{/if}
				</aside>
			{/if}

			<!-- Floating Ambient Capture Button -->
			<AmbientCapture />

			<!-- Selected thought detail (for graph views) -->
			{#if $selectedThought && viewMode !== 'list'}
				<aside class="thought-detail">
					<h3>{$selectedThought.thought_type}</h3>
					<p>{$selectedThought.content}</p>
					<div class="detail-meta">
						<span>Confidence: {Math.round($selectedThought.confidence * 100)}%</span>
						{#if $selectedThought.tags?.length}
							<span>Tags: {$selectedThought.tags.join(', ')}</span>
						{/if}
					</div>
					<button class="close-btn" onclick={() => selectedThought.set(null)}>×</button>
				</aside>
			{/if}
		{:else}
			<div class="welcome">
				<div class="logo">
					<svg viewBox="0 0 100 100" width="120" height="120">
						<defs>
							<linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="100%">
								<stop offset="0%" style="stop-color:#7c3aed" />
								<stop offset="100%" style="stop-color:#3b82f6" />
							</linearGradient>
						</defs>
						<circle cx="50" cy="50" r="45" fill="none" stroke="url(#glow)" stroke-width="2" opacity="0.3" />
						<circle cx="50" cy="50" r="35" fill="none" stroke="url(#glow)" stroke-width="2" opacity="0.5" />
						<circle cx="50" cy="50" r="25" fill="none" stroke="url(#glow)" stroke-width="2" opacity="0.7" />
						<circle cx="50" cy="50" r="8" fill="url(#glow)" />
					</svg>
				</div>
				<h2>Personal Knowledge Graph</h2>
				<p>
					LUCID is your decentralized personal knowledge management system built on Holochain.
					Connect to your local conductor to start capturing and organizing your thoughts.
				</p>
				<a class="demo-btn" href="?demo=true">Try Demo Mode</a>

				<div class="features">
					<div class="feature">
						<span class="icon">🧠</span>
						<h3>E/N/M/H Classification</h3>
						<p>Classify knowledge by Empirical, Normative, Materiality, and Harmonic dimensions</p>
					</div>
					<div class="feature">
						<span class="icon">🔍</span>
						<h3>Semantic Search</h3>
						<p>Find thoughts by meaning with local AI embeddings</p>
					</div>
					<div class="feature">
						<span class="icon">🎙️</span>
						<h3>Ambient Capture</h3>
						<p>Voice-activated thought capture detects insights automatically</p>
					</div>
					<div class="feature">
						<span class="icon">🌐</span>
						<h3>Collective Sensemaking</h3>
						<p>Share anonymized beliefs and discover emergent consensus</p>
					</div>
				</div>
			</div>
		{/if}
	</main>

	<footer>
		<span>LUCID v0.2.0</span>
		<span>•</span>
		<span>Mycelix Ecosystem</span>
		<span>•</span>
		<span>{$thoughts.length} thoughts</span>
	</footer>
</div>

<!-- Global modals -->
<CommandPalette bind:isOpen={showCommandPalette} on:command={handleCommand} />
<KeyboardShortcuts bind:isOpen={showShortcuts} />
{#if showImportExport}
	<ImportExport on:close={() => (showImportExport = false)} />
{/if}

<style>
	.app {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 16px 24px;
		background: #1a1a2e;
		border-bottom: 1px solid #2a2a4e;
		gap: 16px;
	}

	.brand {
		display: flex;
		align-items: baseline;
		gap: 12px;
	}

	h1 {
		margin: 0;
		font-size: 1.5rem;
		font-weight: 700;
		background: linear-gradient(135deg, #7c3aed, #3b82f6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}

	.tagline {
		font-size: 0.8rem;
		color: #666;
	}

	.demo-badge {
		font-size: 0.65rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		padding: 2px 8px;
		border-radius: 4px;
		background: linear-gradient(135deg, #f59e0b, #ef4444);
		color: #fff;
		vertical-align: middle;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.view-toggle,
	.panel-toggle {
		display: flex;
		gap: 4px;
		padding: 4px;
		background: #252540;
		border-radius: 8px;
	}

	.view-btn,
	.panel-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		background: transparent;
		border: none;
		border-radius: 6px;
		color: #666;
		cursor: pointer;
		transition: all 0.2s;
	}

	.panel-btn {
		width: 32px;
		height: 32px;
	}

	.view-btn:hover,
	.panel-btn:hover {
		color: #aaa;
		background: #3a3a5e;
	}

	.view-btn.active,
	.panel-btn.active {
		color: #fff;
		background: #7c3aed;
	}

	.icon-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		background: transparent;
		border: 1px solid #3a3a5e;
		border-radius: 8px;
		color: #888;
		cursor: pointer;
		transition: all 0.2s;
	}

	.icon-btn:hover {
		color: #fff;
		border-color: #7c3aed;
	}

	main {
		flex: 1;
		padding: 24px;
		max-width: 1600px;
		width: 100%;
		margin: 0 auto;
		display: flex;
		gap: 24px;
	}

	main.with-panel {
		padding-right: 400px;
	}

	.main-content {
		flex: 1;
		min-width: 0;
	}

	.input-section {
		margin-bottom: 24px;
	}

	.graph-section {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.graph-container {
		height: 600px;
	}

	.split-section {
		display: grid;
		grid-template-columns: 350px 1fr;
		gap: 24px;
		height: calc(100vh - 280px);
		min-height: 500px;
	}

	.split-left {
		overflow-y: auto;
	}

	.split-right {
		min-height: 400px;
	}

	.side-panel {
		position: fixed;
		top: 73px;
		right: 0;
		bottom: 53px;
		width: 380px;
		background: #1a1a2e;
		border-left: 1px solid #2a2a4e;
		padding: 16px;
		overflow-y: auto;
	}

	.close-panel {
		position: absolute;
		top: 12px;
		right: 12px;
		width: 28px;
		height: 28px;
		background: #252540;
		border: none;
		border-radius: 6px;
		color: #888;
		font-size: 1.2rem;
		cursor: pointer;
		z-index: 10;
	}

	.close-panel:hover {
		color: #fff;
		background: #3a3a5e;
	}

	.panel-empty {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 200px;
		color: #666;
		text-align: center;
	}

	.thought-detail {
		position: fixed;
		bottom: 80px;
		right: 24px;
		width: 320px;
		background: #1e1e2e;
		border: 1px solid #2a2a4e;
		border-radius: 12px;
		padding: 16px;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
	}

	.thought-detail h3 {
		margin: 0 0 8px;
		color: #7c3aed;
		font-size: 0.9rem;
	}

	.thought-detail p {
		margin: 0 0 12px;
		font-size: 0.9rem;
		line-height: 1.5;
	}

	.detail-meta {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 0.75rem;
		color: #666;
	}

	.close-btn {
		position: absolute;
		top: 8px;
		right: 8px;
		width: 24px;
		height: 24px;
		background: none;
		border: none;
		color: #666;
		font-size: 1.2rem;
		cursor: pointer;
	}

	.close-btn:hover {
		color: #fff;
	}

	.welcome {
		text-align: center;
		padding: 60px 20px;
		max-width: 800px;
		margin: 0 auto;
	}

	.logo {
		margin-bottom: 24px;
	}

	.logo svg {
		animation: pulse 3s ease-in-out infinite;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.7;
		}
	}

	.welcome h2 {
		font-size: 1.8rem;
		margin: 0 0 16px;
		color: #fff;
	}

	.welcome > p {
		color: #888;
		font-size: 1.1rem;
		line-height: 1.6;
		margin-bottom: 20px;
	}

	.demo-btn {
		display: inline-block;
		padding: 10px 24px;
		margin-bottom: 32px;
		background: linear-gradient(135deg, #7c3aed, #3b82f6);
		color: #fff;
		border-radius: 8px;
		font-size: 0.95rem;
		font-weight: 500;
		text-decoration: none;
		transition: opacity 0.2s;
	}

	.demo-btn:hover {
		opacity: 0.85;
		text-decoration: none;
	}

	.features {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 24px;
		text-align: left;
	}

	.feature {
		background: #1a1a2e;
		border: 1px solid #2a2a4e;
		border-radius: 12px;
		padding: 20px;
	}

	.feature .icon {
		font-size: 2rem;
		display: block;
		margin-bottom: 12px;
	}

	.feature h3 {
		margin: 0 0 8px;
		font-size: 1rem;
		color: #fff;
	}

	.feature p {
		margin: 0;
		font-size: 0.85rem;
		color: #888;
		line-height: 1.5;
	}

	footer {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 12px;
		padding: 16px;
		background: #1a1a2e;
		border-top: 1px solid #2a2a4e;
		font-size: 0.8rem;
		color: #555;
	}

	@media (max-width: 1200px) {
		main.with-panel {
			padding-right: 24px;
		}

		.side-panel {
			position: fixed;
			top: 0;
			right: 0;
			bottom: 0;
			z-index: 100;
		}
	}

	@media (max-width: 900px) {
		.split-section {
			grid-template-columns: 1fr;
		}

		.split-left {
			max-height: 300px;
		}
	}

	@media (max-width: 600px) {
		header {
			flex-direction: column;
			gap: 12px;
		}

		.tagline {
			display: none;
		}

		main {
			padding: 16px;
		}

		.thought-detail {
			left: 16px;
			right: 16px;
			width: auto;
		}

		.side-panel {
			width: 100%;
		}
	}
</style>
