// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * SpaceScene - Top-level Three.js scene for the orbit viewer.
 *
 * Manages the WebGLRenderer, camera, controls, lighting, star field, and
 * the multi-scale scene graph (rootScene -> earthGroup -> LEO/MEO/GEO groups).
 * Delegates to EarthRenderer, OrbitRenderer, and ConjunctionRenderer.
 *
 * @module SpaceScene
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EarthRenderer } from './EarthRenderer.js';
import { OrbitRenderer } from './OrbitRenderer.js';
import { ConjunctionRenderer } from './ConjunctionRenderer.js';
import { TimeController } from './TimeController.js';

// ── Constants ────────────────────────────────────────────────────────

/** Number of background stars */
const STAR_COUNT = 5000;

/** Radius of the star-field sphere */
const STAR_FIELD_RADIUS = 500;

/** Scene-unit scale: 1 unit = 6371 km (Earth radius) */
const EARTH_RADIUS_KM = 6371;

// ── Star field ───────────────────────────────────────────────────────

/**
 * Build a points-based star field on a large sphere.
 * @returns {THREE.Points}
 */
function createStarField() {
  const positions = new Float32Array(STAR_COUNT * 3);
  const colors = new Float32Array(STAR_COUNT * 3);

  for (let i = 0; i < STAR_COUNT; i++) {
    // Uniform random direction on sphere
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = STAR_FIELD_RADIUS;

    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);

    // Slight colour variation (white-blue to warm-yellow)
    const temp = 0.7 + Math.random() * 0.3;
    colors[i * 3] = temp;
    colors[i * 3 + 1] = temp;
    colors[i * 3 + 2] = 0.8 + Math.random() * 0.2;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.8,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.9,
  });

  return new THREE.Points(geo, mat);
}

// ── SpaceScene class ─────────────────────────────────────────────────

export class SpaceScene {
  /** @type {THREE.WebGLRenderer} */
  #renderer;

  /** @type {THREE.PerspectiveCamera} */
  #camera;

  /** @type {OrbitControls} */
  #controls;

  /** @type {THREE.Scene} Root scene */
  #scene;

  /** @type {THREE.Group} Earth-centered group */
  #earthGroup;

  /** @type {THREE.Group} LEO shell group */
  #leoGroup;

  /** @type {THREE.Group} MEO shell group */
  #meoGroup;

  /** @type {THREE.Group} GEO shell group */
  #geoGroup;

  /** @type {EarthRenderer} */
  #earth;

  /** @type {OrbitRenderer} */
  #orbits;

  /** @type {ConjunctionRenderer} */
  #conjunctions;

  /** @type {TimeController} */
  #time;

  /** @type {THREE.Points} Background stars */
  #stars;

  /** @type {THREE.DirectionalLight} Sun light */
  #sunLight;

  /** @type {number} requestAnimationFrame handle */
  #rafHandle;

  /** @type {boolean} */
  #disposed;

  /** @type {HTMLElement} Container element */
  #container;

  /** @type {ResizeObserver} */
  #resizeObserver;

  /**
   * @param {HTMLElement} container - DOM element to render into
   */
  constructor(container) {
    this.#container = container;
    this.#disposed = false;
    this.#rafHandle = 0;

    // ── Renderer ─────────────────────────────────────────────────
    this.#renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.#renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.#renderer.setSize(container.clientWidth, container.clientHeight);
    this.#renderer.setClearColor(0x000510);
    this.#renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.#renderer.toneMappingExposure = 1.0;
    container.appendChild(this.#renderer.domElement);

    // ── Camera ───────────────────────────────────────────────────
    const aspect = container.clientWidth / container.clientHeight;
    this.#camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 2000);
    this.#camera.position.set(0, 2, 5);

    // ── Controls ─────────────────────────────────────────────────
    this.#controls = new OrbitControls(this.#camera, this.#renderer.domElement);
    this.#controls.enableDamping = true;
    this.#controls.dampingFactor = 0.08;
    this.#controls.minDistance = 1.2;
    this.#controls.maxDistance = 50;
    this.#controls.target.set(0, 0, 0);

    // ── Scene graph ──────────────────────────────────────────────
    this.#scene = new THREE.Scene();

    this.#earthGroup = new THREE.Group();
    this.#scene.add(this.#earthGroup);

    this.#leoGroup = new THREE.Group();
    this.#meoGroup = new THREE.Group();
    this.#geoGroup = new THREE.Group();
    this.#earthGroup.add(this.#leoGroup);
    this.#earthGroup.add(this.#meoGroup);
    this.#earthGroup.add(this.#geoGroup);

    // ── Star field ───────────────────────────────────────────────
    this.#stars = createStarField();
    this.#scene.add(this.#stars);

    // ── Lighting ─────────────────────────────────────────────────
    const ambient = new THREE.AmbientLight(0x334466, 0.6);
    this.#scene.add(ambient);

    this.#sunLight = new THREE.DirectionalLight(0xfff8e8, 1.8);
    this.#sunLight.position.set(5, 2, 3);
    this.#scene.add(this.#sunLight);

    // Subtle hemisphere for counter-shadow
    const hemi = new THREE.HemisphereLight(0x446688, 0x111122, 0.3);
    this.#scene.add(hemi);

    // ── Sub-renderers ────────────────────────────────────────────
    this.#earth = new EarthRenderer(this.#sunLight.position.clone().normalize());
    this.#earthGroup.add(this.#earth.group);

    this.#orbits = new OrbitRenderer();
    this.#earthGroup.add(this.#orbits.group);

    this.#conjunctions = new ConjunctionRenderer();
    this.#earthGroup.add(this.#conjunctions.group);

    // ── Time controller ──────────────────────────────────────────
    this.#time = new TimeController();

    // ── Resize handling ──────────────────────────────────────────
    this.#resizeObserver = new ResizeObserver(() => this.#handleResize());
    this.#resizeObserver.observe(container);

    // ── Click handling ───────────────────────────────────────────
    this.#renderer.domElement.addEventListener('pointerdown', (e) => this.#handleClick(e));

    // ── Start render loop ────────────────────────────────────────
    this.#animate();
  }

  // ── Public API ─────────────────────────────────────────────────────

  /** @returns {EarthRenderer} */
  get earth() {
    return this.#earth;
  }

  /** @returns {OrbitRenderer} */
  get orbits() {
    return this.#orbits;
  }

  /** @returns {ConjunctionRenderer} */
  get conjunctionRenderer() {
    return this.#conjunctions;
  }

  /** @returns {TimeController} */
  get time() {
    return this.#time;
  }

  /**
   * Convert altitude in km above Earth surface to scene units.
   * @param {number} altKm
   * @returns {number} Scene-unit radius from center
   */
  static altToSceneRadius(altKm) {
    return (EARTH_RADIUS_KM + altKm) / EARTH_RADIUS_KM;
  }

  /**
   * Convert ECI position in km to scene-unit coordinates.
   * @param {number} xKm
   * @param {number} yKm
   * @param {number} zKm
   * @returns {number[]} [x, y, z] in scene units
   */
  static eciToScene(xKm, yKm, zKm) {
    return [xKm / EARTH_RADIUS_KM, zKm / EARTH_RADIUS_KM, -yKm / EARTH_RADIUS_KM];
  }

  /**
   * Add an orbit from an array of ECI state vectors (in km).
   *
   * @param {number} norad_id
   * @param {string} name
   * @param {string} type - 'payload' | 'debris' | 'rocket_body'
   * @param {number[][]} statesKm - Array of [x,y,z] positions in km (ECI)
   */
  addOrbit(norad_id, name, type, statesKm) {
    const scenePos = statesKm.map((s) => SpaceScene.eciToScene(s[0], s[1], s[2]));
    this.#orbits.addSatellite(norad_id, name, type, scenePos);
  }

  /**
   * Add a conjunction event.
   *
   * @param {{ id: string, pos1Km: number[], pos2Km: number[], pc: number, tca: Date }} event
   */
  addConjunction(event) {
    const p1 = SpaceScene.eciToScene(event.pos1Km[0], event.pos1Km[1], event.pos1Km[2]);
    const p2 = SpaceScene.eciToScene(event.pos2Km[0], event.pos2Km[1], event.pos2Km[2]);
    this.#conjunctions.addConjunction(event.id, p1, p2, event.pc, event.tca);
  }

  /**
   * Remove an orbit by NORAD ID.
   * @param {number} norad_id
   */
  removeOrbit(norad_id) {
    this.#orbits.removeSatellite(norad_id);
  }

  /**
   * Set the time warp multiplier.
   * @param {number} multiplier
   */
  setTimeWarp(multiplier) {
    this.#time.setWarp(multiplier);
  }

  /**
   * Get the current simulated time.
   * @returns {Date}
   */
  getSimulatedTime() {
    return this.#time.currentTime;
  }

  /**
   * Register a callback for satellite selection.
   * @param {(norad_id: number, name: string) => void} callback
   */
  onSatelliteSelect(callback) {
    this.#orbits.onSelect(callback);
  }

  /** Clean up all GPU resources and stop the render loop. */
  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;

    cancelAnimationFrame(this.#rafHandle);
    this.#resizeObserver.disconnect();

    this.#earth.dispose();
    this.#orbits.dispose();
    this.#conjunctions.dispose();

    // Star field
    this.#stars.geometry.dispose();
    /** @type {THREE.PointsMaterial} */ (this.#stars.material).dispose();

    // Controls
    this.#controls.dispose();

    // Renderer
    this.#renderer.dispose();
    if (this.#renderer.domElement.parentElement) {
      this.#renderer.domElement.parentElement.removeChild(this.#renderer.domElement);
    }
  }

  // ── Private ────────────────────────────────────────────────────────

  /** Main animation loop. */
  #animate() {
    if (this.#disposed) return;

    this.#rafHandle = requestAnimationFrame(() => this.#animate());

    const wallNow = performance.now();

    // Advance simulation clock
    const dt = this.#time.update(wallNow);

    // Update Earth rotation
    this.#earth.updateRotation(this.#time.currentTime);

    // Update Earth LOD based on camera distance
    const camDist = this.#camera.position.length();
    this.#earth.updateLOD(camDist);

    // Pulse conjunction markers
    this.#conjunctions.updatePulse(dt || 1 / 60);

    // Update controls
    this.#controls.update();

    // Render
    this.#renderer.render(this.#scene, this.#camera);
  }

  /** Handle container resize. */
  #handleResize() {
    const w = this.#container.clientWidth;
    const h = this.#container.clientHeight;
    if (w === 0 || h === 0) return;

    this.#camera.aspect = w / h;
    this.#camera.updateProjectionMatrix();
    this.#renderer.setSize(w, h);
  }

  /**
   * Handle pointer clicks for satellite selection.
   * @param {PointerEvent} event
   */
  #handleClick(event) {
    const rect = this.#renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    this.#orbits.handleClick(ndc, this.#camera);
  }
}
