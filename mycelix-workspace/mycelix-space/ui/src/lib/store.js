// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Reactive store for Mycelix Space dashboard.
 *
 * Provides a unified interface: live Holochain data when connected,
 * mock data otherwise. Uses Svelte 5 $state runes for reactivity.
 */

import { SpaceClient } from './holochain-client.js';
import { trackedObjects, conjunctions, bounties, networkStats } from './mock-data.js';

/** @type {SpaceClient} */
let client = null;

/** Reactive state */
let _status = $state('disconnected');
let _objects = $state(trackedObjects);
let _conjunctions = $state(conjunctions);
let _bounties = $state(bounties);
let _stats = $state(networkStats);
let _error = $state(null);

/** Polling interval ID */
let pollInterval = null;

/**
 * Initialize the store. Attempts conductor connection, uses mock on failure.
 * @param {string} [wsUrl] - WebSocket URL for the conductor
 */
export async function init(wsUrl = undefined) {
	client = new SpaceClient(wsUrl);

	client.onStatus((status) => {
		_status = status;
	});

	client.onSignal((signal) => {
		// On any signal, refresh data
		if (signal?.signal_type === 'Conjunction') {
			refreshConjunctions();
		} else if (signal?.signal_type === 'Bounty') {
			refreshBounties();
		}
	});

	const connected = await client.init();
	_status = client.status;

	if (connected) {
		await refreshAll();
		// Poll every 30s for updates
		pollInterval = setInterval(refreshAll, 30000);
	}

	return connected;
}

/** Refresh all data from the conductor */
async function refreshAll() {
	try {
		await Promise.all([
			refreshConjunctions(),
			refreshBounties(),
		]);
		_error = null;
	} catch (err) {
		_error = err.message;
	}
}

async function refreshConjunctions() {
	if (!client?.isLive) return;
	try {
		const result = await client.getHighRiskConjunctions();
		if (result?.length > 0) {
			_conjunctions = result;
			_stats = { ..._stats, active_conjunctions: result.length };
		}
	} catch {
		// Keep existing data on error
	}
}

async function refreshBounties() {
	if (!client?.isLive) return;
	try {
		const result = await client.getActiveBounties();
		if (result?.length > 0) {
			_bounties = result;
			_stats = { ..._stats, active_bounties: result.length };
		}
	} catch {
		// Keep existing data on error
	}
}

/** Clean up on unmount */
export function destroy() {
	if (pollInterval) {
		clearInterval(pollInterval);
		pollInterval = null;
	}
	client?.disconnect();
}

// Exported reactive getters
export function getStatus() { return _status; }
export function getObjects() { return _objects; }
export function getConjunctions() { return _conjunctions; }
export function getBounties() { return _bounties; }
export function getStats() { return _stats; }
export function getError() { return _error; }
