// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/** Mock data for demo mode (no conductor needed) */

export const trackedObjects = [
	{ norad_id: 25544, name: 'ISS (ZARYA)', type: 'Payload', altitude_km: 420, inclination: 51.6, status: 'active' },
	{ norad_id: 48274, name: 'STARLINK-2649', type: 'Payload', altitude_km: 550, inclination: 53.2, status: 'active' },
	{ norad_id: 48275, name: 'STARLINK-2650', type: 'Payload', altitude_km: 550, inclination: 53.2, status: 'active' },
	{ norad_id: 49044, name: 'COSMOS 1408 DEB', type: 'Debris', altitude_km: 485, inclination: 82.6, status: 'tracked' },
	{ norad_id: 49045, name: 'COSMOS 1408 DEB', type: 'Debris', altitude_km: 510, inclination: 82.5, status: 'tracked' },
	{ norad_id: 49046, name: 'COSMOS 1408 DEB', type: 'Debris', altitude_km: 472, inclination: 82.7, status: 'tracked' },
	{ norad_id: 40108, name: 'CZ-4B R/B', type: 'RocketBody', altitude_km: 610, inclination: 97.8, status: 'tracked' },
	{ norad_id: 43205, name: 'TIANGONG', type: 'Payload', altitude_km: 390, inclination: 41.5, status: 'active' },
];

export const conjunctions = [
	{
		id: 'conj-001',
		primary: { norad_id: 25544, name: 'ISS (ZARYA)' },
		secondary: { norad_id: 49044, name: 'COSMOS 1408 DEB' },
		tca: new Date(Date.now() + 18 * 3600000).toISOString(),
		miss_km: 0.42,
		pc: 1.2e-4,
		risk: 'High',
		status: 'Active',
	},
	{
		id: 'conj-002',
		primary: { norad_id: 25544, name: 'ISS (ZARYA)' },
		secondary: { norad_id: 49045, name: 'COSMOS 1408 DEB' },
		tca: new Date(Date.now() + 42 * 3600000).toISOString(),
		miss_km: 2.15,
		pc: 3.8e-6,
		risk: 'Medium',
		status: 'Active',
	},
	{
		id: 'conj-003',
		primary: { norad_id: 48274, name: 'STARLINK-2649' },
		secondary: { norad_id: 40108, name: 'CZ-4B R/B' },
		tca: new Date(Date.now() + 72 * 3600000).toISOString(),
		miss_km: 8.3,
		pc: 1.1e-8,
		risk: 'Low',
		status: 'Monitoring',
	},
];

export const bounties = [
	{
		bounty_id: 'COSMOS-1408-CLEANUP-001',
		debris_norad_id: 49044,
		debris_name: 'COSMOS 1408 DEB',
		amount: 50000,
		currency: 'USD',
		status: 'Open',
		justification: 'High-risk debris from 2021 ASAT test, multiple ISS conjunctions',
		contributors: 3,
		total_contributed: 75000,
		created_at: '2026-01-15T00:00:00Z',
	},
	{
		bounty_id: 'CZ4B-RB-DEORBIT-001',
		debris_norad_id: 40108,
		debris_name: 'CZ-4B R/B',
		amount: 25000,
		currency: 'USD',
		status: 'Claimed',
		justification: 'Large rocket body in congested orbit',
		contributors: 1,
		total_contributed: 30000,
		created_at: '2026-01-20T00:00:00Z',
	},
	{
		bounty_id: 'COSMOS-1408-CLEANUP-002',
		debris_norad_id: 49045,
		debris_name: 'COSMOS 1408 DEB',
		amount: 40000,
		currency: 'USD',
		status: 'InProgress',
		justification: 'Second priority fragment from COSMOS 1408 breakup',
		contributors: 2,
		total_contributed: 55000,
		created_at: '2026-02-01T00:00:00Z',
	},
];

export const networkStats = {
	total_objects: 8,
	active_conjunctions: 3,
	active_bounties: 3,
	sensors_online: 4,
	agents_connected: 12,
	last_updated: new Date().toISOString(),
};
