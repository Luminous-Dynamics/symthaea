<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { initializeHolochain } from '$stores/holochain';
	import { loadThoughts } from '$stores/thoughts';
	import { initializeSearch } from '$services/semantic-search';
	import { initializePluginSystem } from '$services/plugin-system';

	let { children } = $props();

	onMount(async () => {
		// Initialize plugin system first
		await initializePluginSystem();

		// Initialize Holochain connection (or demo mode)
		const connected = await initializeHolochain();

		if (connected) {
			// Load thoughts (demo data or from conductor)
			await loadThoughts();
		}

		// Initialize semantic search (loads embeddings model)
		await initializeSearch();
	});
</script>

<div class="app-shell">
	{@render children()}
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		height: 100%;
		background: #0f0f1a;
		color: #e5e5e5;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
	}

	:global(*) {
		box-sizing: border-box;
	}

	:global(a) {
		color: #7c3aed;
		text-decoration: none;
	}

	:global(a:hover) {
		text-decoration: underline;
	}

	.app-shell {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}
</style>
