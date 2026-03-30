// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ConjunctionRenderer - Conjunction event visualisation.
 *
 * Renders pulsing red spheres at TCA positions, connecting lines between the
 * two objects at closest approach, and colour intensity scaled by collision
 * probability.
 *
 * @module ConjunctionRenderer
 */

import * as THREE from 'three';

// ── Constants ────────────────────────────────────────────────────────

/** Base radius for conjunction warning marker */
const MARKER_BASE_RADIUS = 0.02;

/** Pulse animation speed (radians per second) */
const PULSE_SPEED = 4.0;

/** Pulse scale amplitude (fraction of base radius) */
const PULSE_AMPLITUDE = 0.4;

// ── Conjunction record ───────────────────────────────────────────────

/**
 * @typedef {Object} ConjunctionRecord
 * @property {string} id
 * @property {THREE.Mesh} marker - Pulsing sphere at midpoint
 * @property {THREE.Line} line - Line between two objects
 * @property {number} pc - Collision probability
 * @property {Date} tca - Time of closest approach
 * @property {number} phase - Current animation phase (radians)
 */

// ── ConjunctionRenderer class ────────────────────────────────────────

export class ConjunctionRenderer {
  /** @type {THREE.Group} Root group to be added to the scene */
  group;

  /** @type {Map<string, ConjunctionRecord>} id -> record */
  #conjunctions;

  constructor() {
    this.group = new THREE.Group();
    this.#conjunctions = new Map();
  }

  // ── Public API ─────────────────────────────────────────────────────

  /**
   * Add a conjunction event to the visualisation.
   *
   * @param {string} id - Unique identifier for this conjunction
   * @param {number[]} pos1 - [x,y,z] position of primary object at TCA
   * @param {number[]} pos2 - [x,y,z] position of secondary object at TCA
   * @param {number} pc - Collision probability (0..1)
   * @param {Date} tca - Time of closest approach
   */
  addConjunction(id, pos1, pos2, pc, tca) {
    if (this.#conjunctions.has(id)) {
      this.removeConjunction(id);
    }

    // Midpoint for marker placement
    const mid = [
      (pos1[0] + pos2[0]) / 2,
      (pos1[1] + pos2[1]) / 2,
      (pos1[2] + pos2[2]) / 2,
    ];

    // Colour intensity based on Pc
    // Map Pc to a red intensity: higher Pc = brighter, more saturated red
    const intensity = this._pcToIntensity(pc);
    const color = new THREE.Color();
    color.setRGB(
      0.3 + 0.7 * intensity,
      0.05 * (1 - intensity),
      0.05 * (1 - intensity)
    );

    // ── Pulsing marker sphere ────────────────────────────────────
    const markerGeo = new THREE.SphereGeometry(MARKER_BASE_RADIUS, 16, 12);
    const markerMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.7 + 0.3 * intensity,
    });
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.set(mid[0], mid[1], mid[2]);
    this.group.add(marker);

    // ── Glow shell (slightly larger, more transparent) ───────────
    const glowGeo = new THREE.SphereGeometry(MARKER_BASE_RADIUS * 2.0, 16, 12);
    const glowMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.15 + 0.2 * intensity,
      side: THREE.BackSide,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    marker.add(glow); // child of marker so it pulses together

    // ── Connecting line ──────────────────────────────────────────
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(pos1[0], pos1[1], pos1[2]),
      new THREE.Vector3(pos2[0], pos2[1], pos2[2]),
    ]);
    const lineMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.5 + 0.5 * intensity,
    });
    const line = new THREE.Line(lineGeo, lineMat);
    this.group.add(line);

    /** @type {ConjunctionRecord} */
    const record = {
      id,
      marker,
      line,
      pc,
      tca: new Date(tca),
      phase: 0,
    };

    this.#conjunctions.set(id, record);
  }

  /**
   * Remove a conjunction event.
   * @param {string} id
   */
  removeConjunction(id) {
    const rec = this.#conjunctions.get(id);
    if (!rec) return;

    // Dispose marker + glow child
    this.group.remove(rec.marker);
    rec.marker.geometry.dispose();
    /** @type {THREE.MeshBasicMaterial} */ (rec.marker.material).dispose();
    for (const child of rec.marker.children) {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        /** @type {THREE.MeshBasicMaterial} */ (child.material).dispose();
      }
    }

    // Dispose line
    this.group.remove(rec.line);
    rec.line.geometry.dispose();
    /** @type {THREE.LineBasicMaterial} */ (rec.line.material).dispose();

    this.#conjunctions.delete(id);
  }

  /**
   * Advance pulse animation for all conjunction markers.
   * Call once per frame.
   *
   * @param {number} dt - Delta time in seconds
   */
  updatePulse(dt) {
    for (const rec of this.#conjunctions.values()) {
      rec.phase += dt * PULSE_SPEED;

      // Scale oscillates: 1.0 +/- PULSE_AMPLITUDE
      const scale = 1.0 + PULSE_AMPLITUDE * Math.sin(rec.phase);
      rec.marker.scale.setScalar(scale);

      // Opacity also pulses slightly
      const intensity = this._pcToIntensity(rec.pc);
      const baseOpacity = 0.7 + 0.3 * intensity;
      /** @type {THREE.MeshBasicMaterial} */ (rec.marker.material).opacity =
        baseOpacity * (0.8 + 0.2 * Math.sin(rec.phase));
    }
  }

  /**
   * Update the positions of a conjunction (e.g. as time advances).
   *
   * @param {string} id
   * @param {number[]} pos1 - [x,y,z]
   * @param {number[]} pos2 - [x,y,z]
   */
  updatePositions(id, pos1, pos2) {
    const rec = this.#conjunctions.get(id);
    if (!rec) return;

    const mid = [
      (pos1[0] + pos2[0]) / 2,
      (pos1[1] + pos2[1]) / 2,
      (pos1[2] + pos2[2]) / 2,
    ];
    rec.marker.position.set(mid[0], mid[1], mid[2]);

    // Update line endpoints
    const posAttr = rec.line.geometry.getAttribute('position');
    posAttr.setXYZ(0, pos1[0], pos1[1], pos1[2]);
    posAttr.setXYZ(1, pos2[0], pos2[1], pos2[2]);
    posAttr.needsUpdate = true;
  }

  /**
   * Get all conjunction IDs currently rendered.
   * @returns {string[]}
   */
  listConjunctions() {
    return Array.from(this.#conjunctions.keys());
  }

  /**
   * Get conjunction data for a given id.
   * @param {string} id
   * @returns {{ id: string, pc: number, tca: Date }|null}
   */
  getConjunction(id) {
    const rec = this.#conjunctions.get(id);
    if (!rec) return null;
    return { id: rec.id, pc: rec.pc, tca: rec.tca };
  }

  /** Dispose all GPU resources. */
  dispose() {
    for (const id of this.#conjunctions.keys()) {
      this.removeConjunction(id);
    }
  }

  // ── Private ────────────────────────────────────────────────────────

  /**
   * Map collision probability to a 0..1 intensity value (log scale).
   * @param {number} pc
   * @returns {number}
   */
  _pcToIntensity(pc) {
    if (pc <= 0) return 0;
    if (pc >= 1) return 1;
    // Log scale: 1e-10 -> 0, 1e-2 -> 1
    const logPc = Math.log10(Math.max(pc, 1e-12));
    return Math.min(1, Math.max(0, (logPc + 10) / 8));
  }
}
