// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * EarthRenderer - Procedurally-textured Earth with atmosphere effect.
 *
 * Generates a blue-marble-style texture entirely in code (no external assets).
 * Includes an atmosphere glow using a slightly-larger transparent sphere with
 * a custom Fresnel-like shader, plus togglable latitude/longitude grid lines.
 *
 * @module EarthRenderer
 */

import * as THREE from 'three';

// ── Constants ────────────────────────────────────────────────────────

/** Sidereal day in seconds (Earth rotation period relative to stars) */
const SIDEREAL_DAY_S = 86164.1;

/** Earth radius in scene units (1 unit ~ 6371 km) */
const EARTH_RADIUS = 1.0;

/** Number of lat/lon grid divisions */
const GRID_DIVISIONS = 18; // every 10 degrees

// ── Procedural texture generation ────────────────────────────────────

/**
 * Paint a canvas with a rough blue-marble Earth texture.
 * Uses layered simplex-ish noise approximated via sine combinations.
 *
 * @param {number} width
 * @param {number} height
 * @returns {HTMLCanvasElement}
 */
function generateEarthCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  for (let y = 0; y < height; y++) {
    // latitude in radians  (-PI/2 .. PI/2)
    const lat = ((y / height) - 0.5) * Math.PI;

    for (let x = 0; x < width; x++) {
      // longitude in radians  (-PI .. PI)
      const lon = ((x / width) - 0.5) * 2 * Math.PI;

      // Cheap layered noise for continent shapes
      const n1 = Math.sin(lon * 3.0 + 1.3) * Math.cos(lat * 2.7 + 0.8);
      const n2 = Math.sin(lon * 7.1 - 2.1) * Math.cos(lat * 5.3 + 1.5) * 0.5;
      const n3 = Math.sin(lon * 13.7 + 4.2) * Math.cos(lat * 11.1 - 0.3) * 0.25;
      const n4 = Math.sin(lon * 23.0 - 1.0) * Math.cos(lat * 19.0 + 2.0) * 0.12;
      const noise = n1 + n2 + n3 + n4;

      // Polar ice
      const abslat = Math.abs(lat);
      const ice = abslat > 1.2 ? Math.min(1.0, (abslat - 1.2) * 5.0) : 0;

      let r, g, b;

      if (ice > 0.5) {
        // Ice caps - white/light blue
        r = 220 + Math.floor(Math.random() * 20);
        g = 225 + Math.floor(Math.random() * 15);
        b = 235 + Math.floor(Math.random() * 10);
      } else if (noise > 0.15) {
        // Land - green / brown
        const elevation = (noise - 0.15) / 1.7;
        if (elevation > 0.5) {
          // Mountains - brown/gray
          r = 140 + Math.floor(elevation * 60);
          g = 120 + Math.floor(elevation * 30);
          b = 90 + Math.floor(elevation * 20);
        } else if (abslat > 0.85) {
          // Tundra
          r = 160 + Math.floor(Math.random() * 15);
          g = 170 + Math.floor(Math.random() * 15);
          b = 140;
        } else {
          // Forests / plains
          const tropics = Math.max(0, 1.0 - abslat * 1.8);
          r = 40 + Math.floor(tropics * 30) + Math.floor(elevation * 50);
          g = 100 + Math.floor(tropics * 50) - Math.floor(elevation * 20);
          b = 30 + Math.floor(elevation * 20);
        }
      } else if (noise > 0.0) {
        // Coastal shallows
        r = 30;
        g = 80 + Math.floor(noise * 200);
        b = 160 + Math.floor(noise * 200);
      } else {
        // Deep ocean
        const depth = Math.max(0, Math.min(1, -noise));
        r = 10 + Math.floor(depth * 15);
        g = 30 + Math.floor(depth * 30);
        b = 100 + Math.floor((1 - depth) * 80);
      }

      // Blend ice
      if (ice > 0 && ice <= 0.5) {
        const t = ice * 2;
        r = Math.floor(r * (1 - t) + 230 * t);
        g = Math.floor(g * (1 - t) + 235 * t);
        b = Math.floor(b * (1 - t) + 240 * t);
      }

      const idx = (y * width + x) * 4;
      data[idx] = Math.min(255, Math.max(0, r));
      data[idx + 1] = Math.min(255, Math.max(0, g));
      data[idx + 2] = Math.min(255, Math.max(0, b));
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

// ── Atmosphere shader ────────────────────────────────────────────────

const ATMO_VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const ATMO_FRAGMENT = /* glsl */ `
  uniform vec3 uSunDir;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  void main() {
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
    rim = pow(rim, 3.0);
    float sunFacing = max(dot(vNormal, uSunDir), 0.0) * 0.3 + 0.7;
    vec3 col = mix(vec3(0.3, 0.5, 1.0), vec3(0.6, 0.8, 1.0), rim);
    gl_FragColor = vec4(col * sunFacing, rim * 0.6);
  }
`;

// ── Grid overlay ─────────────────────────────────────────────────────

/**
 * Build a wireframe lat/lon grid as a line-segments object.
 * @param {number} radius
 * @param {number} divisions
 * @returns {THREE.LineSegments}
 */
function buildGrid(radius, divisions) {
  const points = [];
  const r = radius * 1.002; // slightly above surface

  // Latitude lines
  for (let i = 1; i < divisions; i++) {
    const lat = (i / divisions) * Math.PI - Math.PI / 2;
    const cosLat = Math.cos(lat);
    const sinLat = Math.sin(lat);
    const segments = 64;
    for (let j = 0; j < segments; j++) {
      const lon0 = (j / segments) * Math.PI * 2;
      const lon1 = ((j + 1) / segments) * Math.PI * 2;
      points.push(
        new THREE.Vector3(r * cosLat * Math.cos(lon0), r * sinLat, r * cosLat * Math.sin(lon0)),
        new THREE.Vector3(r * cosLat * Math.cos(lon1), r * sinLat, r * cosLat * Math.sin(lon1))
      );
    }
  }

  // Longitude lines
  for (let i = 0; i < divisions * 2; i++) {
    const lon = (i / (divisions * 2)) * Math.PI * 2;
    const cosLon = Math.cos(lon);
    const sinLon = Math.sin(lon);
    const segments = 64;
    for (let j = 0; j < segments; j++) {
      const lat0 = (j / segments) * Math.PI - Math.PI / 2;
      const lat1 = ((j + 1) / segments) * Math.PI - Math.PI / 2;
      points.push(
        new THREE.Vector3(r * Math.cos(lat0) * cosLon, r * Math.sin(lat0), r * Math.cos(lat0) * sinLon),
        new THREE.Vector3(r * Math.cos(lat1) * cosLon, r * Math.sin(lat1), r * Math.cos(lat1) * sinLon)
      );
    }
  }

  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: 0x4488aa, transparent: true, opacity: 0.25 });
  return new THREE.LineSegments(geo, mat);
}

// ── EarthRenderer class ──────────────────────────────────────────────

export class EarthRenderer {
  /** @type {THREE.Group} Root group added to the scene */
  group;

  /** @type {THREE.Mesh} The solid Earth sphere */
  #earthMesh;

  /** @type {THREE.Mesh} Atmosphere glow shell */
  #atmoMesh;

  /** @type {THREE.LineSegments} Lat/lon grid overlay */
  #gridLines;

  /** @type {boolean} Whether the grid is visible */
  #gridVisible;

  /** @type {THREE.ShaderMaterial} Atmosphere material (for uniform updates) */
  #atmoMat;

  /**
   * @param {THREE.Vector3} [sunDirection] - Normalised direction to the sun
   */
  constructor(sunDirection) {
    this.group = new THREE.Group();
    this.#gridVisible = false;

    const sunDir = sunDirection
      ? sunDirection.clone().normalize()
      : new THREE.Vector3(1, 0.3, 0.5).normalize();

    // ── Earth sphere ────────────────────────────────────────────
    const texCanvas = generateEarthCanvas(1024, 512);
    const texture = new THREE.CanvasTexture(texCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;

    const earthGeo = new THREE.SphereGeometry(EARTH_RADIUS, 128, 64);
    const earthMat = new THREE.MeshPhongMaterial({
      map: texture,
      shininess: 15,
      specular: new THREE.Color(0x111122),
    });
    this.#earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.group.add(this.#earthMesh);

    // ── Atmosphere ──────────────────────────────────────────────
    const atmoGeo = new THREE.SphereGeometry(EARTH_RADIUS * 1.025, 64, 32);
    this.#atmoMat = new THREE.ShaderMaterial({
      vertexShader: ATMO_VERTEX,
      fragmentShader: ATMO_FRAGMENT,
      uniforms: {
        uSunDir: { value: sunDir },
      },
      transparent: true,
      side: THREE.FrontSide,
      depthWrite: false,
    });
    this.#atmoMesh = new THREE.Mesh(atmoGeo, this.#atmoMat);
    this.group.add(this.#atmoMesh);

    // ── Grid overlay (off by default) ───────────────────────────
    this.#gridLines = buildGrid(EARTH_RADIUS, GRID_DIVISIONS);
    this.#gridLines.visible = this.#gridVisible;
    this.group.add(this.#gridLines);
  }

  // ── Public API ─────────────────────────────────────────────────────

  /**
   * Update Earth rotation to match simulated time.
   *
   * @param {Date} simTime - Current simulated time
   */
  updateRotation(simTime) {
    // Fraction of sidereal day elapsed since J2000.0 epoch
    const j2000Ms = Date.UTC(2000, 0, 1, 12, 0, 0);
    const elapsedS = (simTime.getTime() - j2000Ms) / 1000;
    const rotations = elapsedS / SIDEREAL_DAY_S;
    this.#earthMesh.rotation.y = (rotations % 1) * Math.PI * 2;
    this.#gridLines.rotation.y = this.#earthMesh.rotation.y;
  }

  /**
   * Set LOD segments based on camera distance.
   * @param {number} distance - Camera distance from Earth center
   */
  updateLOD(distance) {
    const segments = distance < 4 ? 128 : 64;
    const current = this.#earthMesh.geometry.parameters;
    if (current && current.widthSegments !== segments) {
      this.#earthMesh.geometry.dispose();
      this.#earthMesh.geometry = new THREE.SphereGeometry(
        EARTH_RADIUS,
        segments,
        segments / 2
      );
    }
  }

  /** Toggle the latitude/longitude grid overlay. */
  toggleGrid() {
    this.#gridVisible = !this.#gridVisible;
    this.#gridLines.visible = this.#gridVisible;
  }

  /**
   * Set grid visibility explicitly.
   * @param {boolean} visible
   */
  setGridVisible(visible) {
    this.#gridVisible = visible;
    this.#gridLines.visible = visible;
  }

  /**
   * Update the sun direction (e.g. when time warps change lighting).
   * @param {THREE.Vector3} dir
   */
  setSunDirection(dir) {
    this.#atmoMat.uniforms.uSunDir.value.copy(dir).normalize();
  }

  /** Dispose all GPU resources. */
  dispose() {
    this.#earthMesh.geometry.dispose();
    /** @type {THREE.MeshPhongMaterial} */ (this.#earthMesh.material).map?.dispose();
    /** @type {THREE.MeshPhongMaterial} */ (this.#earthMesh.material).dispose();
    this.#atmoMesh.geometry.dispose();
    this.#atmoMat.dispose();
    this.#gridLines.geometry.dispose();
    /** @type {THREE.LineBasicMaterial} */ (this.#gridLines.material).dispose();
  }
}
