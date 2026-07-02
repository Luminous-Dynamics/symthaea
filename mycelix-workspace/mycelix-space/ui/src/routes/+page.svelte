<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script>
	import { onMount, onDestroy } from 'svelte';
	import { trackedObjects as mockObjects, conjunctions as mockConj, bounties as mockBounties, networkStats as mockStats } from '$lib/mock-data.js';
	import { init, destroy, getStatus, getObjects, getConjunctions, getBounties, getStats, getError } from '$lib/store.js';

	let status = $state('disconnected');
	let objects = $state(mockObjects);
	let conjunctions = $state(mockConj);
	let bounties_ = $state(mockBounties);
	let networkStats = $state(mockStats);
	let error = $state(null);

	onMount(async () => {
		await init();
		// After init, read reactive state from store
		const interval = setInterval(() => {
			status = getStatus();
			objects = getObjects();
			conjunctions = getConjunctions();
			bounties_ = getBounties();
			networkStats = getStats();
			error = getError();
		}, 1000);

		return () => clearInterval(interval);
	});

	onDestroy(() => {
		destroy();
	});

	function riskColor(risk) {
		const colors = {
			Emergency: '#dc2626',
			High: '#ef4444',
			Medium: '#f59e0b',
			Low: '#22c55e',
			Negligible: '#6b7280',
		};
		return colors[risk] || '#6b7280';
	}

	function statusColor(status) {
		const colors = {
			Open: '#3b82f6',
			Claimed: '#f59e0b',
			InProgress: '#8b5cf6',
			PendingVerification: '#06b6d4',
			Completed: '#22c55e',
			Expired: '#6b7280',
			Cancelled: '#ef4444',
		};
		return colors[status] || '#6b7280';
	}

	function formatPc(pc) {
		if (pc < 1e-10) return '< 1e-10';
		return pc.toExponential(1);
	}

	function formatDate(iso) {
		return new Date(iso).toLocaleString('en-US', {
			month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false
		});
	}

	function timeUntil(iso) {
		const ms = new Date(iso) - Date.now();
		if (ms < 0) return 'Past';
		const hours = Math.floor(ms / 3600000);
		if (hours < 1) return `${Math.floor(ms / 60000)}m`;
		if (hours < 48) return `${hours}h`;
		return `${Math.floor(hours / 24)}d`;
	}
</script>

<main>
	<header>
		<h1>Mycelix Space</h1>
		<p class="subtitle">Decentralized Space Situational Awareness</p>
		{#if status === 'connected'}
			<span class="badge live">LIVE</span>
		{:else if status === 'connecting'}
			<span class="badge connecting">CONNECTING...</span>
		{:else}
			<span class="badge demo">DEMO MODE</span>
		{/if}
		{#if error}
			<p class="error-msg">{error}</p>
		{/if}
	</header>

	<!-- Stats bar -->
	<section class="stats-bar">
		<div class="stat">
			<span class="stat-value">{networkStats.total_objects}</span>
			<span class="stat-label">Tracked Objects</span>
		</div>
		<div class="stat">
			<span class="stat-value">{networkStats.active_conjunctions}</span>
			<span class="stat-label">Active Conjunctions</span>
		</div>
		<div class="stat">
			<span class="stat-value">{networkStats.active_bounties}</span>
			<span class="stat-label">Cleanup Bounties</span>
		</div>
		<div class="stat">
			<span class="stat-value">{networkStats.sensors_online}</span>
			<span class="stat-label">Sensors Online</span>
		</div>
		<div class="stat">
			<span class="stat-value">{networkStats.agents_connected}</span>
			<span class="stat-label">Network Agents</span>
		</div>
	</section>

	<!-- Active Conjunctions -->
	<section class="panel">
		<h2>Active Conjunctions</h2>
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th>Primary</th>
						<th>Secondary</th>
						<th>TCA</th>
						<th>Miss (km)</th>
						<th>Pc</th>
						<th>Risk</th>
					</tr>
				</thead>
				<tbody>
					{#each conjunctions as c}
						<tr>
							<td>{c.primary.name}</td>
							<td>{c.secondary.name}</td>
							<td>
								<span class="mono">{formatDate(c.tca)}</span>
								<span class="time-badge">T-{timeUntil(c.tca)}</span>
							</td>
							<td class="mono">{c.miss_km.toFixed(2)}</td>
							<td class="mono">{formatPc(c.pc)}</td>
							<td>
								<span class="risk-badge" style="background: {riskColor(c.risk)}">
									{c.risk}
								</span>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>

	<!-- Two-column layout: Objects + Bounties -->
	<div class="two-col">
		<!-- Tracked Objects -->
		<section class="panel">
			<h2>Tracked Objects</h2>
			<div class="table-wrap">
				<table>
					<thead>
						<tr>
							<th>NORAD</th>
							<th>Name</th>
							<th>Type</th>
							<th>Alt (km)</th>
							<th>Inc</th>
						</tr>
					</thead>
					<tbody>
						{#each objects as obj}
							<tr>
								<td class="mono">{obj.norad_id}</td>
								<td>{obj.name}</td>
								<td>
									<span class="type-badge type-{obj.type.toLowerCase()}">{obj.type}</span>
								</td>
								<td class="mono">{obj.altitude_km}</td>
								<td class="mono">{obj.inclination.toFixed(1)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>

		<!-- Bounties -->
		<section class="panel">
			<h2>Kessler Cleanup Market</h2>
			<div class="bounty-grid">
				{#each bounties_ as b}
					<div class="bounty-card">
						<div class="bounty-header">
							<span class="bounty-target">{b.debris_name}</span>
							<span class="status-badge" style="background: {statusColor(b.status)}">
								{b.status}
							</span>
						</div>
						<div class="bounty-id">{b.bounty_id}</div>
						<div class="bounty-amount">
							${b.total_contributed.toLocaleString()}
							<span class="bounty-currency">{b.currency}</span>
						</div>
						<div class="bounty-meta">
							{b.contributors} contributors
						</div>
						<div class="bounty-reason">{b.justification}</div>
					</div>
				{/each}
			</div>
		</section>
	</div>

	<footer>
		<p>Mycelix Space &mdash; Decentralized SSA on Holochain | <a href="https://mycelix.net">mycelix.net</a></p>
	</footer>
</main>

<style>
	:global(body) {
		margin: 0;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
		background: #0f172a;
		color: #e2e8f0;
	}

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1rem;
	}

	header {
		text-align: center;
		padding: 2rem 0 1rem;
	}

	h1 {
		font-size: 2rem;
		margin: 0;
		color: #f8fafc;
	}

	.subtitle {
		color: #94a3b8;
		margin: 0.25rem 0 0.5rem;
	}

	.badge {
		padding: 0.25rem 0.75rem;
		border-radius: 1rem;
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.1em;
		color: white;
	}

	.badge.demo { background: #7c3aed; }
	.badge.live { background: #22c55e; }
	.badge.connecting { background: #f59e0b; }

	.error-msg {
		color: #ef4444;
		font-size: 0.8rem;
		margin: 0.5rem 0 0;
	}

	h2 {
		font-size: 1.1rem;
		color: #cbd5e1;
		margin: 0 0 0.75rem;
		border-bottom: 1px solid #1e293b;
		padding-bottom: 0.5rem;
	}

	/* Stats bar */
	.stats-bar {
		display: flex;
		gap: 1rem;
		justify-content: center;
		flex-wrap: wrap;
		margin: 1.5rem 0;
	}

	.stat {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.5rem;
		padding: 0.75rem 1.25rem;
		text-align: center;
		min-width: 120px;
	}

	.stat-value {
		display: block;
		font-size: 1.5rem;
		font-weight: 700;
		color: #60a5fa;
	}

	.stat-label {
		font-size: 0.75rem;
		color: #94a3b8;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	/* Panels */
	.panel {
		background: #1e293b;
		border: 1px solid #334155;
		border-radius: 0.5rem;
		padding: 1rem;
		margin-bottom: 1rem;
	}

	/* Tables */
	.table-wrap {
		overflow-x: auto;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
	}

	th {
		text-align: left;
		padding: 0.5rem;
		color: #94a3b8;
		font-weight: 600;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		border-bottom: 1px solid #334155;
	}

	td {
		padding: 0.5rem;
		border-bottom: 1px solid #1e293b;
	}

	tr:hover {
		background: #334155;
	}

	.mono {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.8rem;
	}

	/* Badges */
	.risk-badge, .status-badge {
		padding: 0.15rem 0.5rem;
		border-radius: 0.25rem;
		font-size: 0.7rem;
		font-weight: 600;
		color: white;
	}

	.time-badge {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 0.1rem 0.35rem;
		background: #334155;
		border-radius: 0.25rem;
		font-size: 0.7rem;
		color: #94a3b8;
	}

	.type-badge {
		padding: 0.1rem 0.4rem;
		border-radius: 0.2rem;
		font-size: 0.7rem;
		font-weight: 500;
	}

	.type-payload { background: #1d4ed8; color: white; }
	.type-debris { background: #dc2626; color: white; }
	.type-rocketbody { background: #d97706; color: white; }

	/* Two column layout */
	.two-col {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}

	@media (max-width: 768px) {
		.two-col { grid-template-columns: 1fr; }
	}

	/* Bounty cards */
	.bounty-grid {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.bounty-card {
		background: #0f172a;
		border: 1px solid #334155;
		border-radius: 0.5rem;
		padding: 0.75rem;
	}

	.bounty-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.25rem;
	}

	.bounty-target {
		font-weight: 600;
		color: #f8fafc;
	}

	.bounty-id {
		font-family: monospace;
		font-size: 0.7rem;
		color: #64748b;
		margin-bottom: 0.5rem;
	}

	.bounty-amount {
		font-size: 1.25rem;
		font-weight: 700;
		color: #22c55e;
	}

	.bounty-currency {
		font-size: 0.75rem;
		color: #94a3b8;
		font-weight: 400;
	}

	.bounty-meta {
		font-size: 0.75rem;
		color: #94a3b8;
		margin: 0.25rem 0;
	}

	.bounty-reason {
		font-size: 0.75rem;
		color: #64748b;
		margin-top: 0.25rem;
		line-height: 1.4;
	}

	/* Footer */
	footer {
		text-align: center;
		padding: 2rem 0;
		color: #475569;
		font-size: 0.8rem;
	}

	footer a {
		color: #60a5fa;
		text-decoration: none;
	}

	footer a:hover {
		text-decoration: underline;
	}
</style>
