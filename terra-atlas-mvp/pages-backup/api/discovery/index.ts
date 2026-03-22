// Terra Atlas Discovery API - Main Health/Info Endpoint
// TypeScript implementation for Next.js API routes

import type { NextApiRequest, NextApiResponse } from 'next';

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

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<DiscoveryAPIInfo | { error: string }>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  res.status(200).json({
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
  });
}