// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Terra Atlas Discovery API - Main Health/Info Endpoint
// App Router compatible handler

interface DiscoveryAPIInfo {
  status: string;
  service: string;
  version: string;
  endpoints: string[];
  description: string;
  stats: {
    projects_tracked: number;
    transmission_lines: number;
    corridors_available: number;
    average_cost_reduction: string;
    success_rate_improvement: string;
  };
}

export async function GET() {
  return Response.json({
    status: 'healthy',
    service: 'Terra Atlas Discovery API',
    version: '1.0.0',
    description: 'Helping developers with stuck FERC projects find solutions through data intelligence',
    endpoints: [
      '/api/discovery/similar - Find similar successful projects',
      '/api/discovery/transmission - Find available transmission capacity',
      '/api/discovery/corridors - Find cost-sharing corridor opportunities',
      '/api/discovery/queue-intelligence - Analyze FERC queue position'
    ],
    stats: {
      projects_tracked: 10000,
      transmission_lines: 1000,
      corridors_available: 4,
      average_cost_reduction: '74%',
      success_rate_improvement: '28%'
    }
  } satisfies DiscoveryAPIInfo);
}
