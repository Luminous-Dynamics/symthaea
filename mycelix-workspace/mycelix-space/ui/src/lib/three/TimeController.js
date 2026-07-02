// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TimeController - Simulation time management for the orbit viewer.
 *
 * Maintains a simulated clock that advances at a configurable warp factor
 * relative to wall-clock time.  Supports pause/resume and jump-to-date.
 *
 * @module TimeController
 */

/**
 * @callback TimeUpdateCallback
 * @param {Date} simulatedTime - The current simulated time
 * @param {number} warp - The active time-warp multiplier
 */

/**
 * Real-time clock that advances by a configurable warp factor.
 */
export class TimeController {
  /** @type {Date} */
  #currentTime;

  /** @type {number} Multiplier applied to wall-clock delta */
  #timeWarp;

  /** @type {boolean} */
  #paused;

  /** @type {number} Last wall-clock timestamp (ms) used for delta calc */
  #lastWallMs;

  /** @type {TimeUpdateCallback[]} */
  #listeners;

  /** @type {number[]} Allowed warp values for quick cycling */
  static WARP_PRESETS = [1, 10, 100, 1000];

  /**
   * @param {Date} [startTime] - Initial simulated time (defaults to now)
   */
  constructor(startTime) {
    this.#currentTime = startTime ? new Date(startTime) : new Date();
    this.#timeWarp = 1;
    this.#paused = false;
    this.#lastWallMs = performance.now();
    this.#listeners = [];
  }

  // ── Public read-only properties ────────────────────────────────────

  /** @returns {Date} Current simulated time */
  get currentTime() {
    return new Date(this.#currentTime);
  }

  /** @returns {number} Active warp multiplier */
  get timeWarp() {
    return this.#timeWarp;
  }

  /** @returns {boolean} Whether the clock is paused */
  get paused() {
    return this.#paused;
  }

  // ── Core update ────────────────────────────────────────────────────

  /**
   * Advance simulated time based on wall-clock delta.
   * Call once per animation frame.
   *
   * @param {number} [wallNowMs] - Current `performance.now()` value.
   *   Passed in so the caller can share a single timestamp across systems.
   * @returns {number} Simulated delta in seconds (may be 0 when paused)
   */
  update(wallNowMs) {
    const now = wallNowMs ?? performance.now();
    const wallDeltaMs = now - this.#lastWallMs;
    this.#lastWallMs = now;

    if (this.#paused || wallDeltaMs <= 0) {
      return 0;
    }

    const simDeltaMs = wallDeltaMs * this.#timeWarp;
    this.#currentTime = new Date(this.#currentTime.getTime() + simDeltaMs);

    // Notify listeners
    for (const cb of this.#listeners) {
      try {
        cb(this.#currentTime, this.#timeWarp);
      } catch (_e) {
        // Swallow listener errors so the clock keeps ticking
      }
    }

    return simDeltaMs / 1000; // return seconds
  }

  // ── Controls ───────────────────────────────────────────────────────

  /**
   * Set the warp multiplier.
   * @param {number} multiplier - Must be > 0
   */
  setWarp(multiplier) {
    if (multiplier > 0) {
      this.#timeWarp = multiplier;
    }
  }

  /** Pause the simulation clock. */
  pause() {
    this.#paused = true;
  }

  /** Resume the simulation clock. */
  resume() {
    this.#paused = false;
    // Reset wall-clock anchor so we don't get a huge delta spike
    this.#lastWallMs = performance.now();
  }

  /** Toggle pause state. */
  togglePause() {
    if (this.#paused) {
      this.resume();
    } else {
      this.pause();
    }
  }

  /**
   * Jump to a specific date/time.
   * @param {Date} date
   */
  jumpTo(date) {
    this.#currentTime = new Date(date);
    this.#lastWallMs = performance.now();

    for (const cb of this.#listeners) {
      try {
        cb(this.#currentTime, this.#timeWarp);
      } catch (_e) {
        // ignore
      }
    }
  }

  // ── Event subscription ─────────────────────────────────────────────

  /**
   * Register a callback that fires each time the simulated time advances.
   * @param {TimeUpdateCallback} callback
   * @returns {() => void} Unsubscribe function
   */
  onTimeUpdate(callback) {
    this.#listeners.push(callback);
    return () => {
      this.#listeners = this.#listeners.filter((cb) => cb !== callback);
    };
  }
}
