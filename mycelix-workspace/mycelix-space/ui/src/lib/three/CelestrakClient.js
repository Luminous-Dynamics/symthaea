// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CelestrakClient - Fetches GP (General Perturbation) data from CelesTrak
 * and propagates orbits using satellite.js SGP4 in the browser.
 *
 * @module CelestrakClient
 */

import * as satellite from 'satellite.js';

/** CelesTrak GP JSON API base URL */
const BASE_URL = 'https://celestrak.org/NORAD/elements/gp.php';

/** Available satellite categories */
export const CATEGORIES = {
	stations: 'Space Stations',
	active: 'Active Satellites',
	starlink: 'Starlink',
	visual: 'Brightest (Visible)',
};

/**
 * Fetch satellite data from CelesTrak and propagate with SGP4.
 */
export class CelestrakClient {
	/** @type {string} */
	#baseUrl;

	/**
	 * @param {string} [baseUrl] - Override the CelesTrak GP API URL
	 */
	constructor(baseUrl = BASE_URL) {
		this.#baseUrl = baseUrl;
	}

	/**
	 * Fetch GP data for a category from CelesTrak.
	 *
	 * @param {string} category - One of: 'stations', 'active', 'starlink', 'visual'
	 * @returns {Promise<Array<{ satrec: object, name: string, norad_id: number, type: string }>>}
	 */
	async fetchCategory(category) {
		const url = `${this.#baseUrl}?GROUP=${encodeURIComponent(category)}&FORMAT=json`;

		const response = await fetch(url);
		if (!response.ok) {
			throw new Error(`CelesTrak fetch failed: ${response.status} ${response.statusText}`);
		}

		/** @type {Array<object>} */
		const gpData = await response.json();

		return gpData
			.map((gp) => {
				try {
					const satrec = satellite.twoline2satrec(gp.TLE_LINE1, gp.TLE_LINE2);
					const noradId = parseInt(gp.NORAD_CAT_ID, 10);
					const name = gp.OBJECT_NAME || `SAT-${noradId}`;
					const type = CelestrakClient.classifySatellite(
						name,
						gp.OBJECT_TYPE || ''
					);

					return { satrec, name, norad_id: noradId, type };
				} catch {
					return null; // Skip entries with bad TLE data
				}
			})
			.filter(Boolean);
	}

	/**
	 * Propagate a satellite to a given date, returning ECI position in km.
	 *
	 * @param {object} satrec - satellite.js satrec object
	 * @param {Date} date
	 * @returns {{ position: number[], velocity: number[] } | null}
	 */
	static propagateToECI(satrec, date) {
		const positionAndVelocity = satellite.propagate(satrec, date);
		const posEci = positionAndVelocity.position;
		const velEci = positionAndVelocity.velocity;

		if (!posEci || typeof posEci === 'boolean') {
			return null; // Propagation failed (e.g., decayed object)
		}

		return {
			position: [posEci.x, posEci.y, posEci.z],
			velocity: [velEci.x, velEci.y, velEci.z],
		};
	}

	/**
	 * Generate an orbit trail (one full period of ECI positions).
	 *
	 * @param {object} satrec - satellite.js satrec object
	 * @param {Date} epoch - Start time
	 * @param {number} [numPoints=180] - Number of sample points
	 * @returns {number[][]} Array of [x, y, z] positions in km (ECI)
	 */
	static generateOrbitTrail(satrec, epoch, numPoints = 180) {
		// Estimate orbital period from mean motion (rev/day)
		const meanMotionRevPerDay = satrec.no * (1440 / (2 * Math.PI)); // rad/min → rev/day
		const periodMinutes = meanMotionRevPerDay > 0 ? 1440 / meanMotionRevPerDay : 90;
		const stepMs = (periodMinutes * 60 * 1000) / numPoints;

		const positions = [];
		for (let i = 0; i < numPoints; i++) {
			const t = new Date(epoch.getTime() + i * stepMs);
			const result = CelestrakClient.propagateToECI(satrec, t);
			if (result) {
				positions.push(result.position);
			}
		}

		return positions;
	}

	/**
	 * Classify a satellite by name/type into rendering categories.
	 *
	 * @param {string} name - Satellite name
	 * @param {string} objectType - CelesTrak OBJECT_TYPE field
	 * @returns {'payload' | 'debris' | 'rocket_body'}
	 */
	static classifySatellite(name, objectType = '') {
		const upper = (name + ' ' + objectType).toUpperCase();

		if (upper.includes('DEB') || upper.includes('DEBRIS')) return 'debris';
		if (upper.includes('R/B') || upper.includes('ROCKET')) return 'rocket_body';
		return 'payload';
	}
}
