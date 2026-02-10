/**
 * ConjunctionScreener - Lightweight browser-side close-approach detection.
 *
 * Propagates satellite pairs forward using satellite.js SGP4 and identifies
 * close approaches below a configurable miss-distance threshold.
 *
 * @module ConjunctionScreener
 */

import { CelestrakClient } from './CelestrakClient.js';

/** Default screening parameters */
const DEFAULT_PARAMS = {
  /** Look-ahead window in hours */
  windowHours: 24,
  /** Time step in minutes for coarse scan */
  stepMinutes: 10,
  /** Time step in minutes for fine scan around close approach */
  fineStepMinutes: 0.5,
  /** Miss distance threshold in km */
  thresholdKm: 50,
  /** Maximum number of satellites to screen (limits O(n²) cost) */
  maxSatellites: 100,
  /** Maximum number of conjunctions to return */
  maxResults: 20,
};

/**
 * @typedef {Object} ScreenedConjunction
 * @property {string} id - Generated ID
 * @property {string} primaryName - Name of primary object
 * @property {string} secondaryName - Name of secondary object
 * @property {number} primaryId - NORAD ID of primary
 * @property {number} secondaryId - NORAD ID of secondary
 * @property {number[]} pos1Km - Primary position at TCA [x,y,z] km ECI
 * @property {number[]} pos2Km - Secondary position at TCA [x,y,z] km ECI
 * @property {number} missDistanceKm - Miss distance at TCA
 * @property {number} pc - Estimated collision probability
 * @property {Date} tca - Time of closest approach
 * @property {string} risk - Risk level string
 */

/**
 * Screen a set of satellites for close approaches.
 *
 * @param {Array<{ satrec: object, name: string, norad_id: number }>} satellites
 * @param {Partial<typeof DEFAULT_PARAMS>} [params]
 * @returns {ScreenedConjunction[]}
 */
export function screenConjunctions(satellites, params = {}) {
  const cfg = { ...DEFAULT_PARAMS, ...params };
  const now = new Date();

  // Limit satellite count to avoid O(n²) explosion
  const sats = satellites.slice(0, cfg.maxSatellites);

  // Pre-propagate all satellites at coarse time steps
  const numSteps = Math.ceil((cfg.windowHours * 60) / cfg.stepMinutes);
  const times = [];
  for (let i = 0; i <= numSteps; i++) {
    times.push(new Date(now.getTime() + i * cfg.stepMinutes * 60_000));
  }

  // Propagate all satellites at all times
  /** @type {Map<number, Array<number[]|null>>} norad_id -> positions at each time step */
  const trajectories = new Map();
  for (const sat of sats) {
    const positions = times.map((t) => {
      const result = CelestrakClient.propagateToECI(sat.satrec, t);
      return result ? result.position : null;
    });
    trajectories.set(sat.norad_id, positions);
  }

  /** @type {ScreenedConjunction[]} */
  const results = [];

  // Pair-wise screening
  for (let i = 0; i < sats.length; i++) {
    for (let j = i + 1; j < sats.length; j++) {
      const posA = trajectories.get(sats[i].norad_id);
      const posB = trajectories.get(sats[j].norad_id);

      // Find minimum distance across coarse time steps
      let minDist = Infinity;
      let minIdx = 0;

      for (let k = 0; k < times.length; k++) {
        const a = posA[k];
        const b = posB[k];
        if (!a || !b) continue;

        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < minDist) {
          minDist = dist;
          minIdx = k;
        }
      }

      if (minDist > cfg.thresholdKm) continue;

      // Fine-grained search around the minimum
      const fineStart = times[Math.max(0, minIdx - 1)].getTime();
      const fineEnd = times[Math.min(times.length - 1, minIdx + 1)].getTime();
      const fineStepMs = cfg.fineStepMinutes * 60_000;

      let bestDist = minDist;
      let bestTime = times[minIdx];
      let bestPosA = posA[minIdx];
      let bestPosB = posB[minIdx];

      for (let t = fineStart; t <= fineEnd; t += fineStepMs) {
        const date = new Date(t);
        const rA = CelestrakClient.propagateToECI(sats[i].satrec, date);
        const rB = CelestrakClient.propagateToECI(sats[j].satrec, date);
        if (!rA || !rB) continue;

        const dx = rA.position[0] - rB.position[0];
        const dy = rA.position[1] - rB.position[1];
        const dz = rA.position[2] - rB.position[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < bestDist) {
          bestDist = dist;
          bestTime = date;
          bestPosA = rA.position;
          bestPosB = rB.position;
        }
      }

      // Estimate Pc from miss distance (simplified spherical hard-body model)
      // Assumes combined cross-section radius of ~10m and position uncertainty ~1km
      const pc = estimatePc(bestDist);
      const risk = classifyRisk(pc);

      results.push({
        id: `SCRN-${sats[i].norad_id}-${sats[j].norad_id}`,
        primaryName: sats[i].name,
        secondaryName: sats[j].name,
        primaryId: sats[i].norad_id,
        secondaryId: sats[j].norad_id,
        pos1Km: bestPosA,
        pos2Km: bestPosB,
        missDistanceKm: bestDist,
        pc,
        tca: bestTime,
        risk,
      });
    }
  }

  // Sort by miss distance (closest first), limit results
  results.sort((a, b) => a.missDistanceKm - b.missDistanceKm);
  return results.slice(0, cfg.maxResults);
}

/**
 * Simplified collision probability estimate using a Gaussian encounter model.
 * Assumes combined hard-body radius of 10m and 1-sigma position uncertainty of 1 km.
 *
 * @param {number} missKm - Miss distance in km
 * @returns {number} Estimated Pc
 */
function estimatePc(missKm) {
  const combinedRadius = 0.01; // 10 m in km
  const sigma = 1.0; // 1 km position uncertainty

  // 2D Gaussian: Pc ≈ (r² / (2σ²)) * exp(-d² / (2σ²))
  // Simplified from the Foster (1992) approximation
  const r2 = combinedRadius * combinedRadius;
  const s2 = sigma * sigma;
  const d2 = missKm * missKm;

  return (r2 / (2 * s2)) * Math.exp(-d2 / (2 * s2));
}

/**
 * Classify risk level from collision probability.
 *
 * @param {number} pc
 * @returns {string}
 */
function classifyRisk(pc) {
  if (pc >= 1e-2) return 'Emergency';
  if (pc >= 1e-4) return 'High';
  if (pc >= 1e-6) return 'Medium';
  if (pc >= 1e-8) return 'Low';
  return 'Negligible';
}
