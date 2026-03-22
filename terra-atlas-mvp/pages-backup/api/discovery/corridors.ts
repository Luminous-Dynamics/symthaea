// Terra Atlas Discovery API - Find Corridor Opportunities
// TypeScript implementation for Next.js API routes

import type { NextApiRequest, NextApiResponse } from 'next';

interface SharedInfrastructure {
  transmission_line: string;
  substations: number;
  cost_per_project: number;
  savings_vs_standalone: string;
}

interface Corridor {
  name: string;
  total_projects: number;
  total_capacity_mw: number;
  shared_infrastructure: SharedInfrastructure;
  open_capacity_mw: number;
  contact: string;
}

interface CorridorsResponse {
  corridors: Corridor[];
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<CorridorsResponse | { error: string }>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { state, capacity } = req.query;
    const capacityMw = parseFloat(capacity as string) || 100;
    
    // In production, would query from database
    // For now, using mock data that demonstrates the concept
    const allCorridors: Corridor[] = [
      {
        name: 'West Texas Renewable Corridor',
        total_projects: 8,
        total_capacity_mw: 2400,
        shared_infrastructure: {
          transmission_line: 'New 500kV line',
          substations: 2,
          cost_per_project: 28000000,
          savings_vs_standalone: '74%'
        },
        open_capacity_mw: 600,
        contact: 'corridor-coordinator@terraatlas.com'
      },
      {
        name: 'California Central Valley Solar Highway',
        total_projects: 12,
        total_capacity_mw: 3600,
        shared_infrastructure: {
          transmission_line: 'Upgraded 500kV corridor',
          substations: 3,
          cost_per_project: 32000000,
          savings_vs_standalone: '68%'
        },
        open_capacity_mw: 800,
        contact: 'ca-corridor@terraatlas.com'
      },
      {
        name: 'Great Lakes Wind Corridor',
        total_projects: 15,
        total_capacity_mw: 4500,
        shared_infrastructure: {
          transmission_line: 'HVDC backbone',
          substations: 4,
          cost_per_project: 35000000,
          savings_vs_standalone: '71%'
        },
        open_capacity_mw: 1200,
        contact: 'greatlakes@terraatlas.com'
      },
      {
        name: 'Southeast Solar Belt',
        total_projects: 10,
        total_capacity_mw: 2000,
        shared_infrastructure: {
          transmission_line: 'Upgraded 345kV network',
          substations: 3,
          cost_per_project: 25000000,
          savings_vs_standalone: '65%'
        },
        open_capacity_mw: 500,
        contact: 'southeast@terraatlas.com'
      }
    ];
    
    // Filter corridors based on state if provided
    let filteredCorridors = allCorridors;
    
    if (state) {
      const stateStr = (state as string).toUpperCase();
      
      // Map states to corridors
      const stateToCorridors: Record<string, string[]> = {
        'TX': ['West Texas Renewable Corridor'],
        'CA': ['California Central Valley Solar Highway'],
        'MI': ['Great Lakes Wind Corridor'],
        'OH': ['Great Lakes Wind Corridor'],
        'IL': ['Great Lakes Wind Corridor'],
        'GA': ['Southeast Solar Belt'],
        'FL': ['Southeast Solar Belt'],
        'NC': ['Southeast Solar Belt'],
        'SC': ['Southeast Solar Belt']
      };
      
      const corridorNames = stateToCorridors[stateStr] || [];
      filteredCorridors = allCorridors.filter(c => 
        corridorNames.includes(c.name) || 
        c.name.toLowerCase().includes(stateStr.toLowerCase())
      );
    }
    
    // Filter by capacity fit
    filteredCorridors = filteredCorridors.filter(c => 
      c.open_capacity_mw >= capacityMw
    );
    
    res.status(200).json({ corridors: filteredCorridors });
  } catch (error) {
    console.error('Error finding corridors:', error);
    res.status(500).json({ error: 'Failed to find corridors' });
  }
}