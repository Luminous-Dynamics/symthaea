// Terra Atlas Discovery API - FERC Projects Endpoint
// Serves real FERC queue data from generated JSON files

import type { NextApiRequest, NextApiResponse } from 'next';
import * as fs from 'fs';
import * as path from 'path';

// Load FERC data from generated files
function loadFERCData() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'ferc-queue-2024.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading FERC data:', error);
    return [];
  }
}

// Load USACE dam data
function loadUSACEData() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'usace-retrofit-opportunities.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading USACE data:', error);
    return [];
  }
}

// Load SMR pipeline data
function loadSMRData() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'smr-pipeline-projects.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading SMR data:', error);
    return [];
  }
}

function loadSMRStats() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'smr-stats.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading SMR stats:', error);
    return null;
  }
}

function loadUSACEStats() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'usace-stats.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading USACE stats:', error);
    return null;
  }
}

function loadCorridorOpportunities() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'corridor-opportunities.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading corridor data:', error);
    return [];
  }
}

function loadStats() {
  try {
    const dataPath = path.join(process.cwd(), 'data', 'ferc-stats.json');
    const data = fs.readFileSync(dataPath, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading stats:', error);
    return null;
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Support both GET and POST methods
  if (req.method !== 'GET' && req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Extract query parameters
  const { 
    type = 'projects',
    limit = 100,
    offset = 0,
    state,
    status,
    source,
    min_capacity = 0,
    max_capacity = 10000,
    developer,
    sort = 'queue_date',
    order = 'desc',
    data_source = 'ferc'  // New parameter to choose between ferc/usace/all
  } = req.method === 'GET' ? req.query : req.body;

  const limitNum = parseInt(limit as string);
  const offsetNum = parseInt(offset as string);
  const minCap = parseFloat(min_capacity as string);
  const maxCap = parseFloat(max_capacity as string);

  try {
    // Return stats if requested
    if (type === 'stats') {
      const stats = loadStats();
      return res.status(200).json({
        success: true,
        data: stats,
        message: 'FERC queue statistics loaded'
      });
    }

    // Return corridor opportunities if requested
    if (type === 'corridors') {
      const corridors = loadCorridorOpportunities();
      const paginated = corridors.slice(offsetNum, offsetNum + limitNum);
      return res.status(200).json({
        success: true,
        data: paginated,
        pagination: {
          total: corridors.length,
          limit: limitNum,
          offset: offsetNum,
          has_more: offsetNum + limitNum < corridors.length
        },
        message: `Found ${corridors.length} corridor opportunities with $47.6B potential savings`
      });
    }

    // Return USACE dam stats if requested  
    if (type === 'usace-stats') {
      const usaceStats = loadUSACEStats();
      return res.status(200).json({
        success: true,
        data: usaceStats,
        message: 'USACE dam retrofit statistics loaded'
      });
    }

    // Return SMR pipeline stats if requested
    if (type === 'smr-stats') {
      const smrStats = loadSMRStats();
      return res.status(200).json({
        success: true,
        data: smrStats,
        message: 'SMR pipeline statistics loaded'
      });
    }

    // Return SMR pipeline projects
    if (type === 'smr' || data_source === 'smr') {
      let smrProjects = loadSMRData();
      
      // Apply filters
      if (state) {
        smrProjects = smrProjects.filter((p: any) => 
          p.state === (state as string).toUpperCase()
        );
      }
      
      // Apply capacity filters
      smrProjects = smrProjects.filter((p: any) => 
        p.total_capacity_mw >= minCap && p.total_capacity_mw <= maxCap
      );
      
      // Sort results
      smrProjects.sort((a: any, b: any) => {
        let sortField = sort as string;
        if (sortField === 'queue_date') sortField = 'estimated_construction_start';
        
        let aVal = a[sortField];
        let bVal = b[sortField];
        
        if (order === 'asc') {
          return aVal > bVal ? 1 : -1;
        } else {
          return aVal < bVal ? 1 : -1;
        }
      });

      // Calculate aggregates
      const totalCapacity = smrProjects.reduce((sum: number, p: any) => sum + p.total_capacity_mw, 0);
      const totalInvestment = smrProjects.reduce((sum: number, p: any) => sum + p.estimated_project_cost, 0);
      const totalJobs = smrProjects.reduce((sum: number, p: any) => sum + p.construction_jobs + p.permanent_jobs, 0);
      const totalGeneration = smrProjects.reduce((sum: number, p: any) => sum + p.annual_generation_gwh, 0);
      
      // Paginate results
      const paginatedSMR = smrProjects.slice(offsetNum, offsetNum + limitNum);
      
      return res.status(200).json({
        success: true,
        data: paginatedSMR,
        data_source: 'smr',
        pagination: {
          total: smrProjects.length,
          limit: limitNum,
          offset: offsetNum,
          has_more: offsetNum + limitNum < smrProjects.length
        },
        aggregates: {
          total_projects: smrProjects.length,
          total_capacity_mw: Math.round(totalCapacity),
          total_capacity_gw: (totalCapacity / 1000).toFixed(1),
          total_investment: totalInvestment,
          total_investment_billions: (totalInvestment / 1_000_000_000).toFixed(1),
          total_jobs: totalJobs,
          total_annual_generation_twh: (totalGeneration / 1000).toFixed(1),
          avg_project_size_mw: smrProjects.length > 0 
            ? Math.round(totalCapacity / smrProjects.length) 
            : 0
        },
        message: `Found ${smrProjects.length} SMR pipeline projects totaling ${(totalCapacity / 1000).toFixed(1)} GW`
      });
    }

    // Return USACE dam retrofit opportunities
    if (type === 'dams' || data_source === 'usace') {
      let dams = loadUSACEData();
      
      // Apply filters
      if (state) {
        dams = dams.filter((d: any) => 
          d.state === (state as string).toUpperCase()
        );
      }
      
      // Apply capacity filters
      dams = dams.filter((d: any) => 
        d.retrofit_potential_mw >= minCap && d.retrofit_potential_mw <= maxCap
      );
      
      // Sort results
      dams.sort((a: any, b: any) => {
        let sortField = sort as string;
        if (sortField === 'queue_date') sortField = 'year_completed';
        
        let aVal = a[sortField];
        let bVal = b[sortField];
        
        if (order === 'asc') {
          return aVal > bVal ? 1 : -1;
        } else {
          return aVal < bVal ? 1 : -1;
        }
      });

      // Calculate aggregates
      const totalPotential = dams.reduce((sum: number, d: any) => sum + d.retrofit_potential_mw, 0);
      const totalInvestment = dams.reduce((sum: number, d: any) => sum + d.retrofit_cost, 0);
      const totalGeneration = dams.reduce((sum: number, d: any) => sum + d.estimated_annual_generation_mwh, 0);
      const totalJobs = dams.reduce((sum: number, d: any) => sum + d.jobs_created, 0);
      
      // Paginate results
      const paginatedDams = dams.slice(offsetNum, offsetNum + limitNum);
      
      return res.status(200).json({
        success: true,
        data: paginatedDams,
        data_source: 'usace',
        pagination: {
          total: dams.length,
          limit: limitNum,
          offset: offsetNum,
          has_more: offsetNum + limitNum < dams.length
        },
        aggregates: {
          total_dams: dams.length,
          total_retrofit_potential_mw: Math.round(totalPotential),
          total_retrofit_potential_gw: (totalPotential / 1000).toFixed(1),
          total_investment_required: totalInvestment,
          total_investment_billions: (totalInvestment / 1_000_000_000).toFixed(1),
          total_annual_generation_gwh: (totalGeneration / 1000).toFixed(1),
          total_jobs_potential: totalJobs,
          avg_payback_years: dams.length > 0
            ? (dams.reduce((sum: number, d: any) => sum + d.payback_period_years, 0) / dams.length).toFixed(1)
            : "0.0"
        },
        message: `Found ${dams.length} viable dam retrofit opportunities totaling ${(totalPotential / 1000).toFixed(1)} GW`
      });
    }

    // Default: return FERC projects with filters
    let projects = loadFERCData();
    
    // Apply filters
    if (state) {
      projects = projects.filter((p: any) => 
        p.state === (state as string).toUpperCase()
      );
    }
    
    if (status) {
      if (status === 'withdrawn') {
        projects = projects.filter((p: any) => p.withdrawn === true);
      } else if (status === 'active') {
        projects = projects.filter((p: any) => !p.withdrawn && !p.operational);
      } else if (status === 'operational') {
        projects = projects.filter((p: any) => p.operational === true);
      }
    }
    
    if (source) {
      projects = projects.filter((p: any) => 
        p.energy_source.toLowerCase().includes((source as string).toLowerCase())
      );
    }

    if (developer) {
      projects = projects.filter((p: any) => 
        p.developer.toLowerCase().includes((developer as string).toLowerCase())
      );
    }
    
    // Apply capacity filters
    projects = projects.filter((p: any) => 
      p.capacity_mw >= minCap && p.capacity_mw <= maxCap
    );

    // Sort results
    projects.sort((a: any, b: any) => {
      let aVal = a[sort as string];
      let bVal = b[sort as string];
      
      // Handle date fields
      if (sort === 'queue_date' || sort === 'withdrawn_date' || sort === 'in_service_date') {
        aVal = new Date(aVal).getTime();
        bVal = new Date(bVal).getTime();
      }
      
      if (order === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    // Calculate aggregate statistics
    const totalCapacity = projects.reduce((sum: number, p: any) => sum + p.capacity_mw, 0);
    const totalCost = projects.reduce((sum: number, p: any) => sum + p.total_cost, 0);
    const statesRepresented = [...new Set(projects.map((p: any) => p.state))].length;
    const developersRepresented = [...new Set(projects.map((p: any) => p.developer))].length;
    const withdrawnCount = projects.filter((p: any) => p.withdrawn).length;

    // Paginate results
    const paginatedProjects = projects.slice(offsetNum, offsetNum + limitNum);

    return res.status(200).json({
      success: true,
      data: paginatedProjects,
      pagination: {
        total: projects.length,
        limit: limitNum,
        offset: offsetNum,
        has_more: offsetNum + limitNum < projects.length
      },
      aggregates: {
        total_projects: projects.length,
        total_capacity_mw: Math.round(totalCapacity),
        total_capacity_gw: (totalCapacity / 1000).toFixed(1),
        total_interconnection_cost: totalCost,
        total_cost_billions: (totalCost / 1_000_000_000).toFixed(1),
        states_represented: statesRepresented,
        developers_represented: developersRepresented,
        withdrawn_count: withdrawnCount,
        withdrawn_rate: projects.length > 0 
          ? ((withdrawnCount / projects.length) * 100).toFixed(1)
          : "0.0"
      },
      message: `Found ${projects.length.toLocaleString()} FERC queue projects totaling ${(totalCapacity / 1000).toFixed(1)} GW`
    });

  } catch (error) {
    console.error('Discovery Projects API error:', error);
    return res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch discovery data',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}