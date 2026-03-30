<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script>
	import { onMount } from 'svelte';
	import { SpaceScene } from '$lib/three/SpaceScene.js';
	import { TimeController } from '$lib/three/TimeController.js';
	import { CelestrakClient, CATEGORIES } from '$lib/three/CelestrakClient.js';
	import { screenConjunctions, screenConjunctionsProgressive } from '$lib/three/ConjunctionScreener.js';
	import { HolochainBridge } from '$lib/three/HolochainBridge.js';

	// ── Reactive state (Svelte 5) ────────────────────────────────────

	/** @type {HTMLDivElement|undefined} */
	let canvasContainer = $state();

	/** @type {SpaceScene|null} */
	let scene = $state(null);

	let simTimeStr = $state('--');
	let warp = $state(1);
	let paused = $state(false);
	let filterType = $state('All');
	let searchQuery = $state('');

	/** @type {{ norad_id: number, name: string, type: string }[]} */
	let objectList = $state([]);

	/** @type {number|null} */
	let selectedId = $state(null);

	/** @type {{ id: string, primary: string, secondary: string, pc: number, tca: string, risk: string }[]} */
	let conjunctionWarnings = $state([]);

	/** @type {string} */
	let dataSource = $state('loading');

	/** @type {string} */
	let category = $state('stations');

	/** @type {boolean} */
	let isLoading = $state(false);

	/** @type {string} */
	let errorMessage = $state('');

	/** @type {number} */
	let satCount = $state(0);

	// ── Holochain bridge state ──────────────────────────────────────

	/** @type {HolochainBridge} */
	let bridge = new HolochainBridge();

	/** @type {string} */
	let bridgeStatus = $state('offline');

	/** @type {boolean} */
	let isDeepScreening = $state(false);

	/** @type {string} */
	let screeningProgress = $state('');

	// ── Derived state ────────────────────────────────────────────────

	let filteredObjects = $derived.by(() => {
		let list = objectList;
		if (filterType !== 'All') {
			list = list.filter((o) => o.type === filterType.toLowerCase().replace(' ', '_'));
		}
		if (searchQuery.trim()) {
			const q = searchQuery.trim().toLowerCase();
			list = list.filter(
				(o) => o.name.toLowerCase().includes(q) || String(o.norad_id).includes(q)
			);
		}
		return list;
	});

	// ── Mock data (offline fallback) ─────────────────────────────────

	function generateCircularOrbit(altKm, incDeg, raanDeg, points = 180) {
		const R = 6371 + altKm;
		const inc = (incDeg * Math.PI) / 180;
		const raan = (raanDeg * Math.PI) / 180;
		const positions = [];

		for (let i = 0; i < points; i++) {
			const nu = (i / points) * Math.PI * 2;
			const xFinal = R * Math.cos(nu) * Math.cos(raan) - R * Math.sin(nu) * Math.sin(raan) * Math.cos(inc);
			const yFinal = R * Math.cos(nu) * Math.sin(raan) + R * Math.sin(nu) * Math.cos(raan) * Math.cos(inc);
			const zFinal = R * Math.sin(nu) * Math.sin(inc);
			positions.push([xFinal, yFinal, zFinal]);
		}
		return positions;
	}

	const MOCK_SATELLITES = [
		{ norad_id: 25544, name: 'ISS (ZARYA)', type: 'payload', altKm: 420, incDeg: 51.6, raanDeg: 0 },
		{ norad_id: 20580, name: 'HUBBLE', type: 'payload', altKm: 540, incDeg: 28.5, raanDeg: 80 },
		{ norad_id: 41335, name: 'COSMOS 2251 DEB', type: 'debris', altKm: 780, incDeg: 74.0, raanDeg: 145 },
		{ norad_id: 37348, name: 'FENGYUN 1C DEB', type: 'debris', altKm: 850, incDeg: 99.0, raanDeg: 210 },
		{ norad_id: 28884, name: 'SL-16 R/B', type: 'rocket_body', altKm: 610, incDeg: 82.5, raanDeg: 300 },
	];

	const MOCK_CONJUNCTIONS = [
		{
			id: 'CDM-2026-001', primary: 'ISS (ZARYA)', secondary: 'FENGYUN 1C DEB',
			pc: 2.3e-4, tca: new Date(Date.now() + 3 * 3600000).toISOString(),
			risk: 'High', pos1Km: [6791, 0, 0], pos2Km: [6800, 150, 80],
		},
		{
			id: 'CDM-2026-002', primary: 'HUBBLE', secondary: 'COSMOS 2251 DEB',
			pc: 5.1e-6, tca: new Date(Date.now() + 12 * 3600000).toISOString(),
			risk: 'Medium', pos1Km: [0, 6911, 300], pos2Km: [100, 6920, 350],
		},
		{
			id: 'CDM-2026-003', primary: 'SL-16 R/B', secondary: 'FENGYUN 1C DEB',
			pc: 8.7e-8, tca: new Date(Date.now() + 48 * 3600000).toISOString(),
			risk: 'Low', pos1Km: [4000, 4000, 3500], pos2Km: [4050, 4020, 3550],
		},
	];

	function loadMockData(spaceScene) {
		for (const sat of MOCK_SATELLITES) {
			const positions = generateCircularOrbit(sat.altKm, sat.incDeg, sat.raanDeg);
			spaceScene.addOrbit(sat.norad_id, sat.name, sat.type, positions);
		}

		for (const conj of MOCK_CONJUNCTIONS) {
			spaceScene.addConjunction({
				id: conj.id,
				pos1Km: conj.pos1Km,
				pos2Km: conj.pos2Km,
				pc: conj.pc,
				tca: new Date(conj.tca),
			});
		}

		conjunctionWarnings = MOCK_CONJUNCTIONS.map((c) => ({
			id: c.id, primary: c.primary, secondary: c.secondary,
			pc: c.pc, tca: c.tca, risk: c.risk,
		})).sort((a, b) => {
			const riskOrder = { Emergency: 0, High: 1, Medium: 2, Low: 3, Negligible: 4 };
			return (riskOrder[a.risk] ?? 5) - (riskOrder[b.risk] ?? 5);
		});

		dataSource = 'mock';
		satCount = MOCK_SATELLITES.length;
	}

	// ── Live CelesTrak loading ──────────────────────────────────────

	async function loadLiveData(spaceScene, cat) {
		isLoading = true;
		errorMessage = '';

		try {
			const client = new CelestrakClient();
			const satellites = await client.fetchCategory(cat);
			const now = new Date();

			// Clear existing orbits
			for (const obj of objectList) {
				spaceScene.removeOrbit(obj.norad_id);
			}

			// Add live satellites with SGP4-propagated trails
			let added = 0;
			for (const sat of satellites) {
				const trail = CelestrakClient.generateOrbitTrail(sat.satrec, now);
				if (trail.length > 10) {
					spaceScene.addOrbit(sat.norad_id, sat.name, sat.type, trail);
					added++;
				}
			}

			objectList = spaceScene.orbits.listSatellites();
			dataSource = 'live';
			satCount = added;

			// Screen for conjunctions among the loaded satellites
			try {
				const screened = screenConjunctions(satellites, {
					windowHours: 24,
					thresholdKm: 50,
					maxSatellites: 80,
					maxResults: 15,
				});

				// Clear existing conjunction visuals
				for (const c of conjunctionWarnings) {
					spaceScene.conjunctionRenderer.removeConjunction(c.id);
				}

				// Add screened conjunctions to 3D scene
				for (const c of screened) {
					spaceScene.addConjunction({
						id: c.id,
						pos1Km: c.pos1Km,
						pos2Km: c.pos2Km,
						pc: c.pc,
						tca: c.tca,
					});
				}

				// Update sidebar
				conjunctionWarnings = screened.map((c) => ({
					id: c.id,
					primary: c.primaryName,
					secondary: c.secondaryName,
					pc: c.pc,
					tca: c.tca.toISOString(),
					risk: c.risk,
				})).sort((a, b) => {
					const riskOrder = { Emergency: 0, High: 1, Medium: 2, Low: 3, Negligible: 4 };
					return (riskOrder[a.risk] ?? 5) - (riskOrder[b.risk] ?? 5);
				});
			} catch (screenErr) {
				console.warn('Conjunction screening failed:', screenErr);
			}
		} catch (err) {
			errorMessage = `CelesTrak fetch failed: ${err.message}. Using mock data.`;
			loadMockData(spaceScene);
		} finally {
			isLoading = false;
		}
	}

	// ── Lifecycle ────────────────────────────────────────────────────

	onMount(() => {
		if (!canvasContainer) return;

		const spaceScene = new SpaceScene(canvasContainer);
		scene = spaceScene;

		// Try to load live data; fall back to mock on failure
		loadLiveData(spaceScene, category).then(() => {
			objectList = spaceScene.orbits.listSatellites();
		});

		// Attempt Holochain conductor connection (non-blocking)
		bridge.onStatusChange((status) => { bridgeStatus = status; });
		bridge.connect().then((ok) => {
			bridgeStatus = ok ? 'connected' : 'offline';
		});

		// Selection callback
		spaceScene.onSatelliteSelect((id, _name) => {
			selectedId = id;
		});

		// Time display update
		const timeUnsub = spaceScene.time.onTimeUpdate((time, _w) => {
			simTimeStr = time.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
		});

		simTimeStr =
			spaceScene.getSimulatedTime().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

		return () => {
			timeUnsub();
			spaceScene.dispose();
			bridge.disconnect();
			scene = null;
		};
	});

	// ── Event handlers ───────────────────────────────────────────────

	function handleWarp(mult) {
		warp = mult;
		scene?.setTimeWarp(mult);
	}

	function handleTogglePause() {
		paused = !paused;
		if (paused) {
			scene?.time.pause();
		} else {
			scene?.time.resume();
		}
	}

	function handleSelectObject(norad_id) {
		selectedId = norad_id;
		scene?.orbits.selectSatellite(norad_id);
	}

	function handleCategoryChange(cat) {
		category = cat;
		if (scene) {
			loadLiveData(scene, cat).then(() => {
				objectList = scene.orbits.listSatellites();
			});
		}
	}

	function handleRefresh() {
		if (scene) {
			loadLiveData(scene, category).then(() => {
				objectList = scene.orbits.listSatellites();
			});
		}
	}

	async function handleDeepScreen() {
		if (!scene || isDeepScreening) return;
		isDeepScreening = true;
		screeningProgress = 'Fetching active catalog...';

		try {
			const client = new CelestrakClient();
			const allSats = await client.fetchCategory('active');
			screeningProgress = `${allSats.length} satellites loaded. Screening...`;

			const screened = screenConjunctionsProgressive(allSats, {
				windowHours: 24,
				thresholdKm: 50,
				maxResults: 30,
				bandWidthKm: 200,
				maxPerBand: 150,
				onProgress: (pct, msg) => {
					screeningProgress = `${Math.round(pct)}% — ${msg}`;
				},
			});

			// Clear existing conjunction visuals
			for (const c of conjunctionWarnings) {
				scene.conjunctionRenderer.removeConjunction(c.id);
			}

			// Add new conjunctions to 3D scene
			for (const c of screened) {
				scene.addConjunction({
					id: c.id,
					pos1Km: c.pos1Km,
					pos2Km: c.pos2Km,
					pc: c.pc,
					tca: c.tca,
				});
			}

			// Update sidebar
			conjunctionWarnings = screened.map((c) => ({
				id: c.id,
				primary: c.primaryName,
				secondary: c.secondaryName,
				pc: c.pc,
				tca: c.tca.toISOString(),
				risk: c.risk,
				missDistanceKm: c.missDistanceKm,
			})).sort((a, b) => {
				const riskOrder = { Emergency: 0, High: 1, Medium: 2, Low: 3, Negligible: 4 };
				return (riskOrder[a.risk] ?? 5) - (riskOrder[b.risk] ?? 5);
			});

			screeningProgress = `Done: ${screened.length} conjunctions from ${allSats.length} satellites`;
		} catch (err) {
			screeningProgress = `Deep screening failed: ${err.message}`;
		} finally {
			isDeepScreening = false;
		}
	}

	async function handleSubmitToDht(conjId) {
		const conj = conjunctionWarnings.find((c) => c.id === conjId);
		if (!conj) return;

		// Build a ScreenedConjunction-like object for the bridge
		const screened = {
			id: conj.id,
			primaryName: conj.primary,
			secondaryName: conj.secondary,
			primaryId: parseInt(conj.id.split('-')[1]) || 0,
			secondaryId: parseInt(conj.id.split('-')[2]) || 0,
			missDistanceKm: conj.missDistanceKm || 0,
			pc: conj.pc,
			tca: new Date(conj.tca),
			risk: conj.risk,
		};

		const result = await bridge.submitConjunction(screened);
		if (result.submitted) {
			bridgeStatus = 'submitted';
		} else if (result.queued) {
			bridgeStatus = `queued (${bridge.queueLength})`;
		}
	}

	async function handleSubmitAllToDht() {
		const mediumOrHigher = conjunctionWarnings.filter((c) =>
			['Emergency', 'High', 'Medium'].includes(c.risk)
		);
		if (mediumOrHigher.length === 0) return;

		const batch = mediumOrHigher.map((c) => ({
			id: c.id,
			primaryName: c.primary,
			secondaryName: c.secondary,
			primaryId: parseInt(c.id.split('-')[1]) || 0,
			secondaryId: parseInt(c.id.split('-')[2]) || 0,
			missDistanceKm: c.missDistanceKm || 0,
			pc: c.pc,
			tca: new Date(c.tca),
			risk: c.risk,
		}));

		const result = await bridge.submitBatch(batch);
		bridgeStatus = `${result.submitted} submitted, ${result.queued} queued`;
	}

	function riskColor(risk) {
		const colors = {
			Emergency: '#dc2626', High: '#ef4444', Medium: '#f59e0b',
			Low: '#22c55e', Negligible: '#6b7280',
		};
		return colors[risk] || '#6b7280';
	}

	function formatPc(pc) {
		if (pc < 1e-10) return '< 1e-10';
		return pc.toExponential(1);
	}

	function typeColor(type) {
		const colors = { payload: '#4488ff', debris: '#888888', rocket_body: '#ff8800' };
		return colors[type] || '#aaaaaa';
	}
</script>

<div class="orbit-viewer">
	<!-- Left panel: Conjunction warnings + data source -->
	<aside class="panel panel-left">
		<h3>Data Source</h3>
		<div class="source-info">
			{#if isLoading}
				<div class="loading-indicator">
					<span class="spinner"></span> Loading {CATEGORIES[category] ?? category}...
				</div>
			{:else}
				<span class="source-badge" class:live={dataSource === 'live'} class:mock={dataSource === 'mock'}>
					{dataSource === 'live' ? 'LIVE' : 'MOCK'}
				</span>
				<span class="source-count">{satCount} objects</span>
			{/if}
		</div>

		{#if errorMessage}
			<div class="error-toast">{errorMessage}</div>
		{/if}

		<div class="category-row">
			{#each Object.entries(CATEGORIES) as [key, label]}
				<button
					class="cat-btn"
					class:active={category === key}
					onclick={() => handleCategoryChange(key)}
					disabled={isLoading}
				>
					{label}
				</button>
			{/each}
		</div>

		<div class="action-row">
			<button class="refresh-btn" onclick={handleRefresh} disabled={isLoading}>
				Refresh
			</button>
			<button class="refresh-btn deep-btn" onclick={handleDeepScreen} disabled={isLoading || isDeepScreening}>
				{isDeepScreening ? 'Screening...' : 'Deep Screen'}
			</button>
		</div>

		{#if screeningProgress}
			<div class="screening-status">{screeningProgress}</div>
		{/if}

		<h3>Conjunction Warnings</h3>

		<div class="dht-row">
			<span class="bridge-status" class:connected={bridgeStatus === 'connected'}>
				DHT: {bridgeStatus}
			</span>
			{#if conjunctionWarnings.some((c) => ['Emergency', 'High', 'Medium'].includes(c.risk))}
				<button class="dht-btn" onclick={handleSubmitAllToDht} disabled={bridgeStatus === 'offline'}>
					Submit to DHT
				</button>
			{/if}
		</div>
		{#each conjunctionWarnings as c (c.id)}
			<div class="conj-card">
				<div class="conj-header">
					<span class="conj-risk" style="background: {riskColor(c.risk)}">{c.risk}</span>
					<span class="conj-id">{c.id}</span>
				</div>
				<div class="conj-pair">
					{c.primary} / {c.secondary}
				</div>
				<div class="conj-detail">
					<span>Pc: <strong class="mono">{formatPc(c.pc)}</strong></span>
					<span>TCA: {new Date(c.tca).toLocaleTimeString()}</span>
				</div>
				{#if bridge.getStatus(c.id) === 'submitted'}
					<div class="conj-submitted">On DHT</div>
				{:else if bridge.getStatus(c.id) === 'queued'}
					<div class="conj-queued">Queued</div>
				{:else if ['Emergency', 'High', 'Medium'].includes(c.risk)}
					<button class="conj-submit-btn" onclick={() => handleSubmitToDht(c.id)}
						disabled={bridgeStatus === 'offline'}>
						Submit
					</button>
				{/if}
			</div>
		{/each}
	</aside>

	<!-- 3D canvas -->
	<div class="canvas-area" bind:this={canvasContainer}></div>

	<!-- Right panel: Object list -->
	<aside class="panel panel-right">
		<h3>Tracked Objects</h3>
		<input
			class="search-input"
			type="text"
			placeholder="Search by name or NORAD ID..."
			bind:value={searchQuery}
		/>
		<div class="filter-row">
			{#each ['All', 'Payload', 'Debris', 'Rocket Body'] as ft}
				<button
					class="filter-btn"
					class:active={filterType === ft}
					onclick={() => (filterType = ft)}
				>
					{ft}
				</button>
			{/each}
		</div>
		<div class="object-list">
			{#each filteredObjects as obj (obj.norad_id)}
				<button
					class="object-item"
					class:selected={selectedId === obj.norad_id}
					onclick={() => handleSelectObject(obj.norad_id)}
				>
					<span class="obj-dot" style="background: {typeColor(obj.type)}"></span>
					<span class="obj-name">{obj.name}</span>
					<span class="obj-id mono">{obj.norad_id}</span>
				</button>
			{/each}
		</div>
	</aside>

	<!-- Bottom bar: Time controls -->
	<footer class="time-bar">
		<button class="time-btn play-btn" onclick={handleTogglePause}>
			{paused ? '\u25B6' : '\u275A\u275A'}
		</button>
		<div class="warp-group">
			{#each [1, 10, 100, 1000] as mult}
				<button
					class="time-btn warp-btn"
					class:active={warp === mult}
					onclick={() => handleWarp(mult)}
				>
					{mult}x
				</button>
			{/each}
		</div>
		<div class="sim-time mono">{simTimeStr}</div>
	</footer>
</div>

<style>
	/* ── Layout ──────────────────────────────────────────────────── */

	:global(body) {
		margin: 0;
		overflow: hidden;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
		background: #000510;
		color: #e2e8f0;
	}

	.orbit-viewer {
		position: fixed;
		inset: 0;
		display: grid;
		grid-template-columns: 260px 1fr 280px;
		grid-template-rows: 1fr 56px;
	}

	.canvas-area {
		grid-column: 2;
		grid-row: 1;
		overflow: hidden;
	}

	/* ── Panels ─────────────────────────────────────────────────── */

	.panel {
		background: rgba(15, 23, 42, 0.92);
		border: 1px solid #1e293b;
		padding: 0.75rem;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.panel-left {
		grid-column: 1;
		grid-row: 1;
		border-right: 1px solid #334155;
	}

	.panel-right {
		grid-column: 3;
		grid-row: 1;
		border-left: 1px solid #334155;
	}

	.panel h3 {
		margin: 0;
		font-size: 0.85rem;
		color: #94a3b8;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		padding-bottom: 0.4rem;
		border-bottom: 1px solid #1e293b;
	}

	/* ── Data source ────────────────────────────────────────────── */

	.source-info {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.75rem;
	}

	.source-badge {
		padding: 0.1rem 0.4rem;
		border-radius: 0.2rem;
		font-size: 0.65rem;
		font-weight: 700;
		text-transform: uppercase;
	}

	.source-badge.live {
		background: #22c55e;
		color: #000;
	}

	.source-badge.mock {
		background: #f59e0b;
		color: #000;
	}

	.source-count {
		color: #94a3b8;
	}

	.loading-indicator {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		color: #60a5fa;
		font-size: 0.75rem;
	}

	.spinner {
		display: inline-block;
		width: 14px;
		height: 14px;
		border: 2px solid #334155;
		border-top-color: #60a5fa;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.error-toast {
		background: rgba(220, 38, 38, 0.2);
		border: 1px solid #dc2626;
		border-radius: 0.3rem;
		padding: 0.4rem;
		font-size: 0.7rem;
		color: #fca5a5;
	}

	.category-row {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}

	.cat-btn {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.25rem;
		color: #94a3b8;
		padding: 0.3rem 0.5rem;
		font-size: 0.7rem;
		cursor: pointer;
		text-align: left;
	}

	.cat-btn:hover:not(:disabled) {
		background: #334155;
	}

	.cat-btn.active {
		background: #334155;
		color: #f1f5f9;
		border-color: #60a5fa;
	}

	.cat-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.refresh-btn {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.25rem;
		color: #60a5fa;
		padding: 0.35rem 0.5rem;
		font-size: 0.72rem;
		cursor: pointer;
		font-weight: 600;
	}

	.refresh-btn:hover:not(:disabled) {
		background: #334155;
	}

	.refresh-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.action-row {
		display: flex;
		gap: 0.3rem;
	}

	.action-row .refresh-btn {
		flex: 1;
	}

	.deep-btn {
		background: #1e293b !important;
		border-color: #f59e0b !important;
		color: #f59e0b !important;
	}

	.deep-btn:hover:not(:disabled) {
		background: #334155 !important;
	}

	.screening-status {
		font-size: 0.68rem;
		color: #60a5fa;
		padding: 0.2rem 0.4rem;
		background: rgba(96, 165, 250, 0.1);
		border-radius: 0.2rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.dht-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.3rem;
	}

	.bridge-status {
		font-size: 0.65rem;
		color: #64748b;
	}

	.bridge-status.connected {
		color: #22c55e;
	}

	.dht-btn {
		background: #1e293b;
		border: 1px solid #22c55e;
		border-radius: 0.25rem;
		color: #22c55e;
		padding: 0.2rem 0.5rem;
		font-size: 0.65rem;
		cursor: pointer;
		font-weight: 600;
	}

	.dht-btn:hover:not(:disabled) {
		background: rgba(34, 197, 94, 0.1);
	}

	.dht-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	/* ── Conjunction cards ───────────────────────────────────────── */

	.conj-card {
		background: rgba(30, 41, 59, 0.7);
		border: 1px solid #334155;
		border-radius: 0.4rem;
		padding: 0.5rem;
	}

	.conj-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		margin-bottom: 0.3rem;
	}

	.conj-risk {
		padding: 0.1rem 0.4rem;
		border-radius: 0.2rem;
		font-size: 0.65rem;
		font-weight: 700;
		color: white;
		text-transform: uppercase;
	}

	.conj-id {
		font-size: 0.65rem;
		color: #64748b;
		font-family: monospace;
	}

	.conj-pair {
		font-size: 0.78rem;
		color: #cbd5e1;
		margin-bottom: 0.2rem;
	}

	.conj-detail {
		display: flex;
		justify-content: space-between;
		font-size: 0.7rem;
		color: #94a3b8;
	}

	.conj-detail strong {
		color: #f1f5f9;
	}

	.conj-submit-btn {
		background: transparent;
		border: 1px solid #334155;
		border-radius: 0.2rem;
		color: #60a5fa;
		padding: 0.15rem 0.4rem;
		font-size: 0.6rem;
		cursor: pointer;
		margin-top: 0.2rem;
	}

	.conj-submit-btn:hover:not(:disabled) {
		background: rgba(96, 165, 250, 0.1);
		border-color: #60a5fa;
	}

	.conj-submit-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.conj-submitted {
		font-size: 0.6rem;
		color: #22c55e;
		margin-top: 0.2rem;
	}

	.conj-queued {
		font-size: 0.6rem;
		color: #f59e0b;
		margin-top: 0.2rem;
	}

	/* ── Object list ────────────────────────────────────────────── */

	.search-input {
		width: 100%;
		box-sizing: border-box;
		background: #0f172a;
		border: 1px solid #334155;
		border-radius: 0.3rem;
		color: #e2e8f0;
		padding: 0.4rem 0.5rem;
		font-size: 0.78rem;
		outline: none;
	}

	.search-input:focus {
		border-color: #60a5fa;
	}

	.filter-row {
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
	}

	.filter-btn {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.25rem;
		color: #94a3b8;
		padding: 0.2rem 0.45rem;
		font-size: 0.68rem;
		cursor: pointer;
	}

	.filter-btn.active {
		background: #334155;
		color: #f1f5f9;
		border-color: #60a5fa;
	}

	.object-list {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		overflow-y: auto;
		flex: 1;
	}

	.object-item {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		background: transparent;
		border: 1px solid transparent;
		border-radius: 0.25rem;
		padding: 0.35rem 0.4rem;
		text-align: left;
		cursor: pointer;
		color: #cbd5e1;
		font-size: 0.75rem;
		width: 100%;
	}

	.object-item:hover {
		background: rgba(51, 65, 85, 0.5);
	}

	.object-item.selected {
		background: rgba(96, 165, 250, 0.15);
		border-color: #60a5fa;
	}

	.obj-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.obj-name {
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.obj-id {
		color: #64748b;
		font-size: 0.68rem;
	}

	/* ── Time bar ───────────────────────────────────────────────── */

	.time-bar {
		grid-column: 1 / -1;
		grid-row: 2;
		background: rgba(15, 23, 42, 0.95);
		border-top: 1px solid #334155;
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0 1rem;
	}

	.time-btn {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.3rem;
		color: #cbd5e1;
		padding: 0.3rem 0.6rem;
		font-size: 0.8rem;
		cursor: pointer;
		min-width: 36px;
		text-align: center;
	}

	.time-btn:hover {
		background: #334155;
	}

	.play-btn {
		font-size: 1rem;
		min-width: 40px;
	}

	.warp-group {
		display: flex;
		gap: 0.25rem;
	}

	.warp-btn.active {
		background: #2563eb;
		border-color: #3b82f6;
		color: white;
	}

	.sim-time {
		margin-left: auto;
		font-size: 0.85rem;
		color: #60a5fa;
		letter-spacing: 0.03em;
	}

	.mono {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
	}
</style>
