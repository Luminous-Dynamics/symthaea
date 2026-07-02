// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain WebSocket client for Mycelix Space.
 *
 * Connects to a running conductor when available, falls back to mock data.
 * The client wraps raw WebSocket calls into typed zome function calls.
 */

const DEFAULT_WS_URL = 'ws://localhost:8888';
const ZOME_PREFIX = '';  // coordinator zome names match exactly

/** @typedef {'connected' | 'disconnected' | 'connecting' | 'mock'} ConnectionStatus */

class HolochainClient {
	/** @type {WebSocket | null} */
	#ws = null;

	/** @type {ConnectionStatus} */
	#status = 'disconnected';

	/** @type {Map<number, { resolve: Function, reject: Function }>} */
	#pending = new Map();

	/** @type {number} */
	#nextId = 1;

	/** @type {string} */
	#url;

	/** @type {((status: ConnectionStatus) => void)[]} */
	#statusListeners = [];

	/** @type {((signal: any) => void)[]} */
	#signalListeners = [];

	/** @type {string | null} */
	#cellId = null;

	constructor(url = DEFAULT_WS_URL) {
		this.#url = url;
	}

	/** Current connection status */
	get status() {
		return this.#status;
	}

	/** Whether we're using live data */
	get isLive() {
		return this.#status === 'connected';
	}

	/** Subscribe to status changes */
	onStatus(fn) {
		this.#statusListeners.push(fn);
		return () => {
			this.#statusListeners = this.#statusListeners.filter(f => f !== fn);
		};
	}

	/** Subscribe to signals from the conductor */
	onSignal(fn) {
		this.#signalListeners.push(fn);
		return () => {
			this.#signalListeners = this.#signalListeners.filter(f => f !== fn);
		};
	}

	#setStatus(status) {
		this.#status = status;
		this.#statusListeners.forEach(fn => fn(status));
	}

	/**
	 * Attempt to connect to a Holochain conductor.
	 * Returns true if connected, false if falling back to mock.
	 */
	async connect() {
		if (this.#status === 'connected') return true;

		this.#setStatus('connecting');

		try {
			await new Promise((resolve, reject) => {
				const ws = new WebSocket(this.#url);
				const timeout = setTimeout(() => {
					ws.close();
					reject(new Error('Connection timeout'));
				}, 3000);

				ws.onopen = () => {
					clearTimeout(timeout);
					this.#ws = ws;
					this.#setupHandlers();
					resolve();
				};

				ws.onerror = () => {
					clearTimeout(timeout);
					reject(new Error('WebSocket error'));
				};
			});

			this.#setStatus('connected');
			return true;
		} catch {
			this.#setStatus('mock');
			return false;
		}
	}

	#setupHandlers() {
		if (!this.#ws) return;

		this.#ws.onmessage = (event) => {
			try {
				const msg = JSON.parse(event.data);

				// Signal from conductor
				if (msg.type === 'signal') {
					this.#signalListeners.forEach(fn => fn(msg.data));
					return;
				}

				// Response to a zome call
				if (msg.id && this.#pending.has(msg.id)) {
					const { resolve, reject } = this.#pending.get(msg.id);
					this.#pending.delete(msg.id);

					if (msg.error) {
						reject(new Error(msg.error));
					} else {
						resolve(msg.data);
					}
				}
			} catch {
				// Ignore malformed messages
			}
		};

		this.#ws.onclose = () => {
			this.#setStatus('disconnected');
			// Reject all pending calls
			for (const [id, { reject }] of this.#pending) {
				reject(new Error('Connection closed'));
			}
			this.#pending.clear();
		};
	}

	/**
	 * Call a zome function on the conductor.
	 *
	 * @param {string} zomeName - Coordinator zome name
	 * @param {string} fnName - Function name
	 * @param {any} payload - Input data (will be JSON-encoded)
	 * @returns {Promise<any>} - The zome function result
	 */
	async callZome(zomeName, fnName, payload = null) {
		if (!this.#ws || this.#status !== 'connected') {
			throw new Error('Not connected to conductor');
		}

		const id = this.#nextId++;

		return new Promise((resolve, reject) => {
			const timeout = setTimeout(() => {
				this.#pending.delete(id);
				reject(new Error(`Zome call timeout: ${zomeName}.${fnName}`));
			}, 30000);

			this.#pending.set(id, {
				resolve: (data) => {
					clearTimeout(timeout);
					resolve(data);
				},
				reject: (err) => {
					clearTimeout(timeout);
					reject(err);
				}
			});

			this.#ws.send(JSON.stringify({
				id,
				type: 'zome_call',
				cell_id: this.#cellId,
				zome_name: zomeName,
				fn_name: fnName,
				payload
			}));
		});
	}

	disconnect() {
		if (this.#ws) {
			this.#ws.close();
			this.#ws = null;
		}
		this.#setStatus('disconnected');
	}
}

// ============================================================================
// Typed API wrappers
// ============================================================================

/**
 * High-level API for Mycelix Space zome calls.
 * Each method calls the corresponding coordinator extern.
 * Falls back to mock data when not connected.
 */
export class SpaceClient {
	/** @type {HolochainClient} */
	#client;

	/** @type {boolean} */
	#useMock = true;

	constructor(wsUrl = DEFAULT_WS_URL) {
		this.#client = new HolochainClient(wsUrl);
	}

	get status() { return this.#client.status; }
	get isLive() { return this.#client.isLive; }
	onStatus(fn) { return this.#client.onStatus(fn); }
	onSignal(fn) { return this.#client.onSignal(fn); }

	/**
	 * Initialize the client. Tries conductor, falls back to mock.
	 */
	async init() {
		const connected = await this.#client.connect();
		this.#useMock = !connected;
		return connected;
	}

	// --- Orbital Objects ---

	async getLatestTles(noradIds) {
		if (this.#useMock) return [];
		return this.#client.callZome('orbital_objects_coordinator', 'get_latest_tles', noradIds);
	}

	async getTlesWithMetadata(noradIds, config = null) {
		if (this.#useMock) return [];
		return this.#client.callZome('orbital_objects_coordinator', 'get_tles_with_metadata', {
			norad_ids: noradIds,
			config
		});
	}

	// --- Observations ---

	async getObservationsForObject(noradId) {
		if (this.#useMock) return [];
		return this.#client.callZome('observations_coordinator', 'get_observations_for_object', noradId);
	}

	async listSensors() {
		if (this.#useMock) return [];
		return this.#client.callZome('observations_coordinator', 'list_sensors', null);
	}

	// --- Conjunctions ---

	async getHighRiskConjunctions() {
		if (this.#useMock) return [];
		return this.#client.callZome('conjunctions_coordinator', 'get_high_risk_conjunctions', null);
	}

	async getConjunctionsForObject(noradId) {
		if (this.#useMock) return [];
		return this.#client.callZome('conjunctions_coordinator', 'get_conjunctions_for_object', noradId);
	}

	async screenConjunction(primaryNoradId, secondaryNoradIds, hoursAhead = 72, pcThreshold = 1e-10) {
		if (this.#useMock) return [];
		return this.#client.callZome('conjunctions_coordinator', 'screen_conjunction', {
			primary_norad_id: primaryNoradId,
			secondary_norad_ids: secondaryNoradIds,
			hours_ahead: hoursAhead,
			pc_threshold: pcThreshold
		});
	}

	// --- Debris Bounties ---

	async getActiveBounties() {
		if (this.#useMock) return [];
		return this.#client.callZome('debris_bounties_coordinator', 'get_active_bounties', null);
	}

	async getBountiesForDebris(noradId) {
		if (this.#useMock) return [];
		return this.#client.callZome('debris_bounties_coordinator', 'get_bounties_for_debris', noradId);
	}

	// --- Traffic Control ---

	async getSessionsForConjunction(conjunctionId) {
		if (this.#useMock) return [];
		return this.#client.callZome('traffic_control_coordinator', 'get_sessions_for_conjunction', conjunctionId);
	}

	disconnect() {
		this.#client.disconnect();
	}
}

export default SpaceClient;
