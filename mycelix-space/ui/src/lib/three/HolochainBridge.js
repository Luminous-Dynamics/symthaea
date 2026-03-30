// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * HolochainBridge - Bridges browser-side conjunction screening results
 * to the Mycelix Space Holochain conjunctions coordinator zome.
 *
 * When a Holochain conductor is available, this submits screened conjunctions
 * as ConjunctionEvent entries on the DHT. When offline, it queues events
 * for later submission.
 *
 * @module HolochainBridge
 */

/**
 * @typedef {Object} BridgeConfig
 * @property {string} [conductorUrl='ws://localhost:8888'] - Holochain conductor WebSocket URL
 * @property {string} [roleName='space_operator'] - DNA role name
 * @property {string} [zomeName='conjunctions_coordinator'] - Conjunctions zome name
 * @property {number} [batchDelayMs=2000] - Delay before flushing queued submissions
 */

/** @type {BridgeConfig} */
const DEFAULT_CONFIG = {
  conductorUrl: 'ws://localhost:8888',
  roleName: 'space_operator',
  zomeName: 'conjunctions_coordinator',
  batchDelayMs: 2000,
};

/**
 * Risk string → RiskLevel enum matching the integrity zome.
 * @type {Record<string, string>}
 */
const RISK_MAP = {
  Emergency: 'Emergency',
  High: 'High',
  Medium: 'Medium',
  Low: 'Low',
  Negligible: 'Negligible',
};

/**
 * Bridge between browser conjunction screening and the Holochain DHT.
 */
export class HolochainBridge {
  /** @type {BridgeConfig} */
  #config;

  /** @type {boolean} */
  #connected = false;

  /** @type {WebSocket|null} */
  #ws = null;

  /** @type {Array<object>} Queued events for batch submission */
  #queue = [];

  /** @type {number|null} */
  #flushTimer = null;

  /** @type {Array<object>} Previously submitted (for dedup) */
  #submitted = [];

  /** @type {function|null} */
  #onStatusChange = null;

  /**
   * @param {Partial<BridgeConfig>} [config]
   */
  constructor(config = {}) {
    this.#config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Attempt to connect to the Holochain conductor.
   * @returns {Promise<boolean>} Whether connection succeeded
   */
  async connect() {
    return new Promise((resolve) => {
      try {
        this.#ws = new WebSocket(this.#config.conductorUrl);

        this.#ws.onopen = () => {
          this.#connected = true;
          this.#emitStatus('connected');
          resolve(true);
        };

        this.#ws.onerror = () => {
          this.#connected = false;
          this.#emitStatus('offline');
          resolve(false);
        };

        this.#ws.onclose = () => {
          this.#connected = false;
          this.#emitStatus('disconnected');
        };

        // Timeout after 3 seconds
        setTimeout(() => {
          if (!this.#connected) {
            this.#ws?.close();
            this.#emitStatus('offline');
            resolve(false);
          }
        }, 3000);
      } catch {
        this.#connected = false;
        this.#emitStatus('offline');
        resolve(false);
      }
    });
  }

  /**
   * Register a status change callback.
   * @param {function(string): void} callback - Called with 'connected', 'disconnected', 'offline', 'submitting'
   */
  onStatusChange(callback) {
    this.#onStatusChange = callback;
  }

  #emitStatus(status) {
    this.#onStatusChange?.(status);
  }

  /** @returns {boolean} */
  get isConnected() {
    return this.#connected;
  }

  /** @returns {number} */
  get queueLength() {
    return this.#queue.length;
  }

  /** @returns {number} */
  get submittedCount() {
    return this.#submitted.length;
  }

  /**
   * Convert a ScreenedConjunction from the browser screener into the
   * CreateEventInput format expected by the conjunctions coordinator zome.
   *
   * @param {import('./ConjunctionScreener.js').ScreenedConjunction} screened
   * @returns {object} CreateEventInput-compatible object
   */
  static toCreateEventInput(screened) {
    const tcaMicros = screened.tca instanceof Date
      ? screened.tca.getTime() * 1000
      : new Date(screened.tca).getTime() * 1000;

    return {
      event_id: screened.id,
      primary_norad_id: screened.primaryId,
      secondary_norad_id: screened.secondaryId,
      tca: { micros: tcaMicros },
      miss_distance_km: screened.missDistanceKm,
      max_pc: screened.pc,
      risk_level: RISK_MAP[screened.risk] || 'Negligible',
      compute_details: false,
      primary_tle: null,
      secondary_tle: null,
    };
  }

  /**
   * Submit a screened conjunction to the DHT.
   * If offline, queues for later submission.
   *
   * @param {import('./ConjunctionScreener.js').ScreenedConjunction} screened
   * @returns {Promise<{ submitted: boolean, queued: boolean, hash?: string }>}
   */
  async submitConjunction(screened) {
    const input = HolochainBridge.toCreateEventInput(screened);

    // Dedup check
    if (this.#submitted.some((s) => s.event_id === input.event_id)) {
      return { submitted: false, queued: false };
    }

    if (!this.#connected) {
      this.#queue.push(input);
      this.#scheduleFlush();
      return { submitted: false, queued: true };
    }

    try {
      const hash = await this.#callZome('create_conjunction_event', input);
      this.#submitted.push(input);
      return { submitted: true, queued: false, hash };
    } catch (err) {
      console.warn('Zome call failed, queuing:', err);
      this.#queue.push(input);
      return { submitted: false, queued: true };
    }
  }

  /**
   * Submit all screened conjunctions from a screening run.
   * Only submits Medium/High/Emergency risk (filters out Low/Negligible).
   *
   * @param {import('./ConjunctionScreener.js').ScreenedConjunction[]} conjunctions
   * @returns {Promise<{ submitted: number, queued: number, skipped: number }>}
   */
  async submitBatch(conjunctions) {
    const SUBMIT_RISKS = new Set(['Emergency', 'High', 'Medium']);
    let submitted = 0;
    let queued = 0;
    let skipped = 0;

    this.#emitStatus('submitting');

    for (const c of conjunctions) {
      if (!SUBMIT_RISKS.has(c.risk)) {
        skipped++;
        continue;
      }

      const result = await this.submitConjunction(c);
      if (result.submitted) submitted++;
      else if (result.queued) queued++;
      else skipped++; // Already submitted (dedup)
    }

    this.#emitStatus(this.#connected ? 'connected' : 'offline');
    return { submitted, queued, skipped };
  }

  /**
   * Flush any queued events (called after reconnection).
   * @returns {Promise<number>} Number of successfully submitted events
   */
  async flushQueue() {
    if (!this.#connected || this.#queue.length === 0) return 0;

    let flushed = 0;
    const remaining = [];

    for (const input of this.#queue) {
      try {
        await this.#callZome('create_conjunction_event', input);
        this.#submitted.push(input);
        flushed++;
      } catch {
        remaining.push(input);
      }
    }

    this.#queue = remaining;
    return flushed;
  }

  /**
   * Get the submission status for a conjunction ID.
   * @param {string} eventId
   * @returns {'submitted' | 'queued' | 'none'}
   */
  getStatus(eventId) {
    if (this.#submitted.some((s) => s.event_id === eventId)) return 'submitted';
    if (this.#queue.some((q) => q.event_id === eventId)) return 'queued';
    return 'none';
  }

  /**
   * Export queued conjunctions as JSON (for offline archival / import-batch.sh).
   * @returns {string} JSON array of CreateEventInput objects
   */
  exportQueue() {
    return JSON.stringify(this.#queue, null, 2);
  }

  /** Disconnect and clean up. */
  disconnect() {
    if (this.#flushTimer) clearTimeout(this.#flushTimer);
    this.#ws?.close();
    this.#ws = null;
    this.#connected = false;
  }

  // ── Private helpers ────────────────────────────────────────────

  #scheduleFlush() {
    if (this.#flushTimer) return;
    this.#flushTimer = setTimeout(async () => {
      this.#flushTimer = null;
      if (this.#connected) {
        await this.flushQueue();
      }
    }, this.#config.batchDelayMs);
  }

  /**
   * Call a zome function via the conductor WebSocket.
   * Uses the Holochain AppWebsocket protocol (JSON-RPC style).
   *
   * @param {string} fnName
   * @param {object} payload
   * @returns {Promise<string>} Action hash
   */
  async #callZome(fnName, payload) {
    if (!this.#ws || this.#ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected to conductor');
    }

    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36).slice(2);

      const message = {
        id,
        type: 'call_zome',
        data: {
          role_name: this.#config.roleName,
          zome_name: this.#config.zomeName,
          fn_name: fnName,
          payload,
        },
      };

      const handler = (event) => {
        try {
          const response = JSON.parse(event.data);
          if (response.id === id) {
            this.#ws.removeEventListener('message', handler);
            if (response.type === 'error') {
              reject(new Error(response.data?.message || 'Zome call failed'));
            } else {
              resolve(response.data);
            }
          }
        } catch {
          // Ignore non-JSON messages
        }
      };

      this.#ws.addEventListener('message', handler);
      this.#ws.send(JSON.stringify(message));

      // Timeout after 10 seconds
      setTimeout(() => {
        this.#ws?.removeEventListener('message', handler);
        reject(new Error('Zome call timed out'));
      }, 10_000);
    });
  }
}
