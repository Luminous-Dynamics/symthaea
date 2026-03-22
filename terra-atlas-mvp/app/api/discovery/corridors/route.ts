// Terra Atlas Discovery API - Find Corridor Opportunities

import { NextRequest, NextResponse } from 'next/server';

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
  states?: string[];
}

const ALL_CORRIDORS: Corridor[] = [
  {
    name: 'West Texas Renewable Corridor',
    total_projects: 8,
    total_capacity_mw: 2400,
    shared_infrastructure: {
      transmission_line: 'New 500kV line',
      substations: 2,
      cost_per_project: 28_000_000,
      savings_vs_standalone: '74%'
    },
    open_capacity_mw: 600,
    contact: 'corridor-coordinator@terraatlas.com',
    states: ['TX', 'NM']
  },
  {
    name: 'California Central Valley Solar Highway',
    total_projects: 12,
    total_capacity_mw: 3600,
    shared_infrastructure: {
      transmission_line: 'Upgraded 500kV corridor',
      substations: 3,
      cost_per_project: 32_000_000,
      savings_vs_standalone: '68%'
    },
    open_capacity_mw: 800,
    contact: 'ca-corridor@terraatlas.com',
    states: ['CA']
  },
  {
    name: 'Great Lakes Wind Corridor',
    total_projects: 15,
    total_capacity_mw: 4500,
    shared_infrastructure: {
      transmission_line: 'HVDC backbone',
      substations: 4,
      cost_per_project: 35_000_000,
      savings_vs_standalone: '71%'
    },
    open_capacity_mw: 1200,
    contact: 'greatlakes@terraatlas.com',
    states: ['MI', 'OH', 'PA', 'NY']
  },
  {
    name: 'Southeast Solar + Storage Corridor',
    total_projects: 10,
    total_capacity_mw: 3000,
    shared_infrastructure: {
      transmission_line: 'Hybrid AC/DC upgrade',
      substations: 3,
      cost_per_project: 27_000_000,
      savings_vs_standalone: '66%'
    },
    open_capacity_mw: 900,
    contact: 'southeast@terraatlas.com',
    states: ['GA', 'AL', 'FL']
  }
];

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const stateFilter = searchParams.get('state')?.toUpperCase();
  const capacityFilter = parseFloat(searchParams.get('capacity') || '0');

  const filtered = ALL_CORRIDORS.filter((corridor) => {
    const matchesState = stateFilter
      ? corridor.states?.some((s) => s.toUpperCase() === stateFilter)
      : true;
    const matchesCapacity = capacityFilter
      ? corridor.open_capacity_mw >= capacityFilter
      : true;
    return matchesState && matchesCapacity;
  });

  const corridors = (filtered.length ? filtered : ALL_CORRIDORS).slice(0, 10);
  return NextResponse.json<{ corridors: Corridor[] }>({ corridors });
}
