// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OrbitRenderer - Satellite markers, orbit trails, selection and labels.
 *
 * Uses InstancedMesh for efficient rendering of many satellite markers,
 * line geometry for orbit trails, Raycaster for click/hover, and canvas
 * textures for label sprites.
 *
 * @module OrbitRenderer
 */

import * as THREE from 'three';

// ── Colour palette ───────────────────────────────────────────────────

/** @enum {number} */
export const SatelliteColors = {
  debris: 0x888888,
  payload: 0x4488ff,
  rocket_body: 0xff8800,
  selected: 0xffff00,
  unknown: 0xaaaaaa,
};

/** Marker sphere radius in scene units */
const MARKER_RADIUS = 0.015;

/** Maximum number of satellite instances in a single InstancedMesh */
const MAX_INSTANCES = 2048;

/** Orbit trail line width (note: WebGL line width is capped at 1 on most GPUs) */
const TRAIL_OPACITY = 0.6;

// ── Helpers ──────────────────────────────────────────────────────────

/**
 * Resolve a satellite type string to its display colour.
 * @param {string} type
 * @returns {number}
 */
function colorForType(type) {
  const key = type.toLowerCase().replace(/[\s-]/g, '_');
  return SatelliteColors[key] ?? SatelliteColors.unknown;
}

/**
 * Create a canvas-based sprite texture with the given text.
 * @param {string} text
 * @param {string} [bgColor='rgba(0,0,0,0.7)']
 * @returns {THREE.Sprite}
 */
function makeLabelSprite(text, bgColor = 'rgba(0,0,0,0.7)') {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Sprite();

  ctx.fillStyle = bgColor;
  ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
  ctx.fill();

  ctx.font = 'bold 28px monospace';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.4, 0.05, 1);
  return sprite;
}

// ── Satellite record ─────────────────────────────────────────────────

/**
 * @typedef {Object} SatelliteRecord
 * @property {number} norad_id
 * @property {string} name
 * @property {string} type - 'payload' | 'debris' | 'rocket_body'
 * @property {Float32Array} positions - Flat [x,y,z, x,y,z, ...] orbit positions
 * @property {number} instanceIndex - Index inside the InstancedMesh
 * @property {THREE.Line} trailLine - The orbit trail line object
 * @property {THREE.Sprite|null} label - Label sprite (only when selected)
 * @property {number} color - Original colour
 */

// ── OrbitRenderer class ──────────────────────────────────────────────

export class OrbitRenderer {
  /** @type {THREE.Group} Root group to be added to the scene */
  group;

  /** @type {THREE.InstancedMesh} Satellite markers */
  #instancedMesh;

  /** @type {Map<number, SatelliteRecord>} norad_id -> record */
  #satellites;

  /** @type {number} Next available instance index */
  #nextIndex;

  /** @type {THREE.Raycaster} For selection */
  #raycaster;

  /** @type {number|null} Currently selected NORAD ID */
  #selectedId;

  /** @type {THREE.Object3D} Scratch object for instance matrix writes */
  #dummy;

  /** @type {((norad_id: number, name: string) => void)|null} */
  #onSelectCallback;

  constructor() {
    this.group = new THREE.Group();
    this.#satellites = new Map();
    this.#nextIndex = 0;
    this.#raycaster = new THREE.Raycaster();
    this.#raycaster.params.Points = { threshold: 0.05 };
    this.#selectedId = null;
    this.#dummy = new THREE.Object3D();
    this.#onSelectCallback = null;

    // ── InstancedMesh for markers ────────────────────────────────
    const markerGeo = new THREE.SphereGeometry(MARKER_RADIUS, 8, 6);
    const markerMat = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0x222222,
    });
    this.#instancedMesh = new THREE.InstancedMesh(markerGeo, markerMat, MAX_INSTANCES);
    this.#instancedMesh.count = 0;
    this.#instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.group.add(this.#instancedMesh);
  }

  // ── Public API ─────────────────────────────────────────────────────

  /**
   * Add a satellite to the renderer.
   *
   * @param {number} norad_id
   * @param {string} name
   * @param {string} type - 'payload' | 'debris' | 'rocket_body'
   * @param {number[][]} positions - Array of [x,y,z] orbit sample points (scene units)
   */
  addSatellite(norad_id, name, type, positions) {
    if (this.#satellites.has(norad_id)) {
      this.removeSatellite(norad_id);
    }

    if (this.#nextIndex >= MAX_INSTANCES) {
      console.warn('[OrbitRenderer] Maximum satellite instances reached');
      return;
    }

    const idx = this.#nextIndex++;
    const color = colorForType(type);

    // Flatten positions for quick access
    const flat = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
      flat[i * 3] = positions[i][0];
      flat[i * 3 + 1] = positions[i][1];
      flat[i * 3 + 2] = positions[i][2];
    }

    // Set initial instance transform (first position)
    this.#dummy.position.set(flat[0], flat[1], flat[2]);
    this.#dummy.updateMatrix();
    this.#instancedMesh.setMatrixAt(idx, this.#dummy.matrix);
    this.#instancedMesh.setColorAt(idx, new THREE.Color(color));
    this.#instancedMesh.count = Math.max(this.#instancedMesh.count, idx + 1);
    this.#instancedMesh.instanceMatrix.needsUpdate = true;
    if (this.#instancedMesh.instanceColor) {
      this.#instancedMesh.instanceColor.needsUpdate = true;
    }

    // Orbit trail
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    const trailMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: TRAIL_OPACITY,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    this.group.add(trailLine);

    /** @type {SatelliteRecord} */
    const record = {
      norad_id,
      name,
      type,
      positions: flat,
      instanceIndex: idx,
      trailLine,
      label: null,
      color,
    };

    this.#satellites.set(norad_id, record);
  }

  /**
   * Remove a satellite from the renderer.
   * @param {number} norad_id
   */
  removeSatellite(norad_id) {
    const rec = this.#satellites.get(norad_id);
    if (!rec) return;

    // Remove trail line
    this.group.remove(rec.trailLine);
    rec.trailLine.geometry.dispose();
    /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).dispose();

    // Remove label
    if (rec.label) {
      this.group.remove(rec.label);
      /** @type {THREE.SpriteMaterial} */ (rec.label.material).map?.dispose();
      /** @type {THREE.SpriteMaterial} */ (rec.label.material).dispose();
    }

    // Zero out the instance (move far away)
    this.#dummy.position.set(0, -1000, 0);
    this.#dummy.updateMatrix();
    this.#instancedMesh.setMatrixAt(rec.instanceIndex, this.#dummy.matrix);
    this.#instancedMesh.instanceMatrix.needsUpdate = true;

    if (this.#selectedId === norad_id) {
      this.#selectedId = null;
    }

    this.#satellites.delete(norad_id);
  }

  /**
   * Update the current position of a satellite (e.g. from propagator).
   * Also moves the instanced marker.
   *
   * @param {number} norad_id
   * @param {number[]} position - [x, y, z] in scene units
   */
  updatePositions(norad_id, position) {
    const rec = this.#satellites.get(norad_id);
    if (!rec) return;

    this.#dummy.position.set(position[0], position[1], position[2]);
    this.#dummy.updateMatrix();
    this.#instancedMesh.setMatrixAt(rec.instanceIndex, this.#dummy.matrix);
    this.#instancedMesh.instanceMatrix.needsUpdate = true;

    // Move label if active
    if (rec.label) {
      rec.label.position.set(
        position[0],
        position[1] + MARKER_RADIUS * 4,
        position[2]
      );
    }
  }

  /**
   * Select a satellite by NORAD ID (highlight + show label).
   * @param {number} norad_id
   */
  selectSatellite(norad_id) {
    // Deselect previous
    if (this.#selectedId !== null) {
      this._deselect(this.#selectedId);
    }

    const rec = this.#satellites.get(norad_id);
    if (!rec) return;

    this.#selectedId = norad_id;

    // Highlight colour
    this.#instancedMesh.setColorAt(rec.instanceIndex, new THREE.Color(SatelliteColors.selected));
    if (this.#instancedMesh.instanceColor) {
      this.#instancedMesh.instanceColor.needsUpdate = true;
    }

    // Trail highlight
    /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).color.setHex(SatelliteColors.selected);
    /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).opacity = 1.0;

    // Label sprite
    if (!rec.label) {
      rec.label = makeLabelSprite(`${rec.norad_id}  ${rec.name}`);
      this.group.add(rec.label);
    }

    // Position label above the marker
    const mat = new THREE.Matrix4();
    this.#instancedMesh.getMatrixAt(rec.instanceIndex, mat);
    const pos = new THREE.Vector3().setFromMatrixPosition(mat);
    rec.label.position.set(pos.x, pos.y + MARKER_RADIUS * 4, pos.z);
    rec.label.visible = true;
  }

  /**
   * Register a callback that fires when a satellite is clicked.
   * @param {(norad_id: number, name: string) => void} callback
   */
  onSelect(callback) {
    this.#onSelectCallback = callback;
  }

  /**
   * Test for satellite selection via mouse click.
   * Call from a pointerdown/click handler.
   *
   * @param {THREE.Vector2} ndc - Normalised device coordinates (-1..1)
   * @param {THREE.Camera} camera
   */
  handleClick(ndc, camera) {
    this.#raycaster.setFromCamera(ndc, camera);
    const hits = this.#raycaster.intersectObject(this.#instancedMesh);

    if (hits.length > 0 && hits[0].instanceId !== undefined) {
      // Find which satellite owns this instance index
      for (const [id, rec] of this.#satellites) {
        if (rec.instanceIndex === hits[0].instanceId) {
          this.selectSatellite(id);
          if (this.#onSelectCallback) {
            this.#onSelectCallback(id, rec.name);
          }
          return;
        }
      }
    }
  }

  /**
   * Get all satellites as an array for UI listing.
   * @returns {{ norad_id: number, name: string, type: string }[]}
   */
  listSatellites() {
    const out = [];
    for (const rec of this.#satellites.values()) {
      out.push({ norad_id: rec.norad_id, name: rec.name, type: rec.type });
    }
    return out;
  }

  /** @returns {number|null} Currently selected NORAD ID */
  get selectedId() {
    return this.#selectedId;
  }

  /** Dispose all GPU resources. */
  dispose() {
    // Dispose instanced mesh
    this.#instancedMesh.geometry.dispose();
    /** @type {THREE.MeshPhongMaterial} */ (this.#instancedMesh.material).dispose();

    // Dispose each satellite's trail + label
    for (const rec of this.#satellites.values()) {
      rec.trailLine.geometry.dispose();
      /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).dispose();
      if (rec.label) {
        /** @type {THREE.SpriteMaterial} */ (rec.label.material).map?.dispose();
        /** @type {THREE.SpriteMaterial} */ (rec.label.material).dispose();
      }
    }

    this.#satellites.clear();
  }

  // ── Private ────────────────────────────────────────────────────────

  /**
   * Revert a satellite to its original colour and hide its label.
   * @param {number} norad_id
   */
  _deselect(norad_id) {
    const rec = this.#satellites.get(norad_id);
    if (!rec) return;

    this.#instancedMesh.setColorAt(rec.instanceIndex, new THREE.Color(rec.color));
    if (this.#instancedMesh.instanceColor) {
      this.#instancedMesh.instanceColor.needsUpdate = true;
    }

    /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).color.setHex(rec.color);
    /** @type {THREE.LineBasicMaterial} */ (rec.trailLine.material).opacity = TRAIL_OPACITY;

    if (rec.label) {
      rec.label.visible = false;
    }
  }
}
