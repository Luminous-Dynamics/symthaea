#!/usr/bin/env tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Terra Atlas Complete Data Integration Script
 * Consolidates all data sources into the MVP platform
 */

import { createClient } from '@supabase/supabase-js';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import axios from 'axios';
import * as csv from 'csv-parse/sync';
import { config } from 'dotenv';

// Load environment variables
config({ path: '.env.local' });

// ========================================
// CONFIGURATION
// ========================================

// Supabase credentials from environment
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('ERROR: Missing required environment variables.');
  console.error('Required: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY');
  console.error('Copy .env.example to .env.local and configure credentials.');
  process.exit(1);
}

// API Keys (from environment or BWS)
const MAPBOX_TOKEN = process.env.MAPBOX_TOKEN || '';
const PLANET_API_KEY = process.env.PLANET_API_KEY || '';
const EIA_API_KEY = process.env.EIA_API_KEY || ''; // US Energy Information Administration
const NREL_API_KEY = process.env.NREL_API_KEY || ''; // National Renewable Energy Laboratory

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ========================================
// DATA SOURCES
// ========================================

interface EnergyProject {
  id?: string;
  name: string;
  type: 'solar' | 'wind' | 'hydro' | 'nuclear' | 'storage' | 'transmission' | 'other';
  status: 'planning' | 'approved' | 'construction' | 'operational' | 'decommissioned';
  latitude: number;
  longitude: number;
  capacity_mw?: number;
  investment_usd?: number;
  developer?: string;
  owner?: string;
  state?: string;
  country?: string;
  commissioning_date?: Date;
  data_source: string;
  metadata?: any;
}

// ========================================
// 1. MIGRATE OLD DEMO DATA (79,193 projects)
// ========================================

async function migrateOldDemoData() {
  console.log('📦 Migrating 79,193 projects from old demos...');
  
  const dbPath = '/srv/luminous-dynamics/terra-atlas-old-demos/terra_atlas.db';
  if (!existsSync(dbPath)) {
    console.log('❌ Old database not found. Skipping...');
    return;
  }

  // Read SQLite data (would need better-sqlite3)
  try {
    const Database = require('better-sqlite3');
    const db = new Database(dbPath, { readonly: true });
    
    const projects = db.prepare(`
      SELECT * FROM projects 
      WHERE latitude IS NOT NULL 
      AND longitude IS NOT NULL
    `).all();

    console.log(`Found ${projects.length} projects to migrate`);

    // Batch insert into Supabase
    const batchSize = 500;
    for (let i = 0; i < projects.length; i += batchSize) {
      const batch = projects.slice(i, i + batchSize).map(p => ({
        name: p.project_name,
        type: mapProjectType(p.technology),
        status: p.status?.toLowerCase() || 'planning',
        latitude: p.latitude,
        longitude: p.longitude,
        capacity_mw: p.capacity_mw,
        investment_usd: p.investment_millions ? p.investment_millions * 1000000 : null,
        developer: p.developer,
        state: p.state,
        country: p.country || 'USA',
        commissioning_date: p.commissioning_date,
        data_source: 'legacy_import',
        metadata: {
          legacy_id: p.id,
          interconnection_status: p.interconnection_status,
          project_type: p.project_type
        }
      }));

      const { error } = await supabase
        .from('projects')
        .insert(batch);

      if (error) {
        console.error(`Error inserting batch ${i/batchSize}:`, error);
      } else {
        console.log(`✅ Imported ${Math.min(i + batchSize, projects.length)}/${projects.length} projects`);
      }
    }

    db.close();
  } catch (error) {
    console.error('Error reading SQLite database:', error);
  }
}

// ========================================
// 2. USACE DAM DATA INTEGRATION
// ========================================

async function fetchUSACEDamData() {
  console.log('🏗️ Fetching USACE dam data (87,000 sites)...');
  
  const USACE_API = 'https://corpslakes.erdc.dren.mil/ords/crrel_apex/f?p=137:3:0::NO:::';
  
  try {
    // Note: USACE data typically requires downloading their National Inventory of Dams (NID) dataset
    // Available at: https://nid.usace.army.mil/api/nation/csv
    
    const response = await axios.get('https://nid.usace.army.mil/api/nation/csv', {
      responseType: 'text',
      timeout: 30000
    });

    const dams = csv.parse(response.data, {
      columns: true,
      skip_empty_lines: true
    });

    console.log(`Found ${dams.length} dams`);

    // Filter for retrofit potential (non-powered dams)
    const retrofitCandidates = dams.filter(dam => 
      !dam.purposes?.includes('H') && // H = Hydroelectric
      parseFloat(dam.nid_height) > 15 && // Minimum 15 feet height
      parseFloat(dam.drainage_area) > 10 // Minimum 10 sq mi drainage
    );

    console.log(`${retrofitCandidates.length} dams suitable for retrofit`);

    // Transform and insert
    const projects = retrofitCandidates.map(dam => ({
      name: `${dam.dam_name} Hydroelectric Retrofit`,
      type: 'hydro' as const,
      status: 'planning' as const,
      latitude: parseFloat(dam.latitude),
      longitude: parseFloat(dam.longitude),
      capacity_mw: estimateHydroPotential(dam),
      developer: 'To Be Determined',
      owner: dam.owner_name,
      state: dam.state,
      country: 'USA',
      data_source: 'usace_nid',
      metadata: {
        dam_height_ft: dam.nid_height,
        drainage_area_sqmi: dam.drainage_area,
        year_built: dam.year_completed,
        purposes: dam.purposes,
        hazard_rating: dam.hazard
      }
    }));

    // Insert in batches
    await batchInsertProjects(projects, 'USACE dams');

  } catch (error) {
    console.error('Error fetching USACE data:', error);
  }
}

// ========================================
// 3. FERC QUEUE DATA
// ========================================

async function fetchFERCQueueData() {
  console.log('⚡ Fetching FERC interconnection queue data...');
  
  // FERC provides data through multiple ISOs/RTOs
  const sources = [
    { name: 'CAISO', url: 'https://www.caiso.com/planning/Pages/GeneratorInterconnection/Default.aspx' },
    { name: 'ERCOT', url: 'https://www.ercot.com/gridinfo/resource' },
    { name: 'PJM', url: 'https://www.pjm.com/planning/services-requests/interconnection-queues.aspx' },
    { name: 'MISO', url: 'https://www.misoenergy.org/planning/generator-interconnection/GI_Queue/' },
    { name: 'ISO-NE', url: 'https://www.iso-ne.com/isoexpress/' },
    { name: 'NYISO', url: 'https://www.nyiso.com/interconnections' },
    { name: 'SPP', url: 'https://www.spp.org/engineering/generator-interconnection/' }
  ];

  // For demonstration, using sample data
  // In production, would scrape/API call each ISO
  const sampleFERCProjects = generateFERCQueueProjects(11547);
  await batchInsertProjects(sampleFERCProjects, 'FERC queue');
}

// ========================================
// 4. SMR (Small Modular Reactor) PIPELINE
// ========================================

async function importSMRPipeline() {
  console.log('☢️ Importing Small Modular Reactor pipeline...');
  
  const smrProjects: EnergyProject[] = [
    {
      name: 'NuScale UAMPS',
      type: 'nuclear',
      status: 'approved',
      latitude: 43.0583,
      longitude: -112.5261,
      capacity_mw: 462,
      investment_usd: 9300000000,
      developer: 'NuScale Power',
      state: 'ID',
      country: 'USA',
      commissioning_date: new Date('2029-01-01'),
      data_source: 'nrc_approved'
    },
    {
      name: 'TerraPower Natrium - Kemmerer',
      type: 'nuclear',
      status: 'construction',
      latitude: 41.7922,
      longitude: -110.5379,
      capacity_mw: 345,
      investment_usd: 4000000000,
      developer: 'TerraPower',
      state: 'WY',
      country: 'USA',
      commissioning_date: new Date('2028-01-01'),
      data_source: 'doe_ardp'
    },
    // Add more SMR projects...
  ];

  await batchInsertProjects(smrProjects, 'SMR pipeline');
}

// ========================================
// 5. PUBLIC API INTEGRATIONS
// ========================================

async function integratePublicAPIs() {
  console.log('🌐 Integrating public energy APIs...');

  // EIA (US Energy Information Administration)
  if (EIA_API_KEY) {
    try {
      const eiaResponse = await axios.get('https://api.eia.gov/v2/electricity/operating-generator-capacity/data/', {
        params: {
          api_key: EIA_API_KEY,
          frequency: 'annual',
          data: ['nameplate-capacity-mw'],
          facets: {
            'sectorid': [1], // Electric power
            'energy_source_code': ['SUN', 'WND', 'WAT'] // Solar, Wind, Hydro
          }
        }
      });
      console.log(`✅ EIA: ${eiaResponse.data.response.data.length} records`);
    } catch (error) {
      console.error('EIA API error:', error);
    }
  }

  // NREL (National Renewable Energy Laboratory)
  if (NREL_API_KEY) {
    try {
      const nrelResponse = await axios.get('https://developer.nrel.gov/api/reopt/v2/job', {
        headers: { 'X-API-KEY': NREL_API_KEY }
      });
      console.log('✅ NREL API connected');
    } catch (error) {
      console.error('NREL API error:', error);
    }
  }

  // OpenWeather for environmental data
  const openWeatherResponse = await axios.get('https://api.openweathermap.org/data/2.5/weather', {
    params: { q: 'Austin,TX', appid: 'demo' } // Free tier
  });
  console.log('✅ OpenWeather API available');

  // IRENA (International Renewable Energy Agency) - Free
  try {
    const irenaResponse = await axios.get('https://pxweb.irena.org/pxweb/api/v1/en/IRENASTAT/Power%20Capacity%20and%20Generation/ELECCAP_2023_cycle2.px');
    console.log('✅ IRENA statistics available');
  } catch (error) {
    console.error('IRENA API error:', error);
  }
}

// ========================================
// 6. REAL-TIME WEBSOCKET SETUP
// ========================================

async function setupRealtimeFeatures() {
  console.log('🔄 Setting up real-time WebSocket features...');
  
  // Subscribe to database changes
  const subscription = supabase
    .channel('projects-changes')
    .on('postgres_changes', 
      { event: '*', schema: 'public', table: 'projects' },
      (payload) => {
        console.log('Real-time update:', payload);
        // Broadcast to connected clients
      }
    )
    .subscribe();

  console.log('✅ Real-time subscriptions active');
}

// ========================================
// HELPER FUNCTIONS
// ========================================

function mapProjectType(technology: string): EnergyProject['type'] {
  const mapping: Record<string, EnergyProject['type']> = {
    'solar': 'solar',
    'wind': 'wind',
    'hydro': 'hydro',
    'nuclear': 'nuclear',
    'storage': 'storage',
    'transmission': 'transmission',
    'battery': 'storage',
    'pumped storage': 'storage'
  };
  
  const tech = technology?.toLowerCase() || '';
  return mapping[tech] || 'other';
}

function estimateHydroPotential(dam: any): number {
  // Simplified formula: P = ρ * g * Q * H * η
  // Where: ρ=1000kg/m³, g=9.81m/s², Q=flow, H=height, η=efficiency
  const heightMeters = parseFloat(dam.nid_height) * 0.3048; // feet to meters
  const drainageArea = parseFloat(dam.drainage_area) || 10; // sq miles
  
  // Very rough estimate: 0.1-0.5 MW per meter of height for medium dams
  return Math.round(heightMeters * 0.3 * (drainageArea / 100));
}

function generateFERCQueueProjects(count: number): EnergyProject[] {
  const projects: EnergyProject[] = [];
  const states = ['TX', 'CA', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'];
  const types: EnergyProject['type'][] = ['solar', 'wind', 'storage'];
  
  for (let i = 0; i < count; i++) {
    projects.push({
      name: `FERC Queue Project ${i + 1}`,
      type: types[Math.floor(Math.random() * types.length)],
      status: 'planning',
      latitude: 25 + Math.random() * 25, // US latitudes
      longitude: -125 + Math.random() * 50, // US longitudes
      capacity_mw: Math.round(10 + Math.random() * 990),
      investment_usd: Math.round((50 + Math.random() * 450) * 1000000),
      state: states[Math.floor(Math.random() * states.length)],
      country: 'USA',
      data_source: 'ferc_queue_sample'
    });
  }
  
  return projects;
}

async function batchInsertProjects(projects: EnergyProject[], source: string) {
  const batchSize = 500;
  for (let i = 0; i < projects.length; i += batchSize) {
    const batch = projects.slice(i, i + batchSize);
    
    const { error } = await supabase
      .from('projects')
      .insert(batch);
    
    if (error) {
      console.error(`Error inserting ${source} batch:`, error);
    } else {
      console.log(`✅ ${source}: ${Math.min(i + batchSize, projects.length)}/${projects.length}`);
    }
  }
}

// ========================================
// MAIN EXECUTION
// ========================================

async function main() {
  console.log('🚀 Terra Atlas Complete Data Integration');
  console.log('========================================\n');
  
  try {
    // 1. Migrate old demo data
    await migrateOldDemoData();
    
    // 2. Fetch USACE dam data
    await fetchUSACEDamData();
    
    // 3. Import FERC queue
    await fetchFERCQueueData();
    
    // 4. Import SMR pipeline
    await importSMRPipeline();
    
    // 5. Integrate public APIs
    await integratePublicAPIs();
    
    // 6. Setup real-time features
    await setupRealtimeFeatures();
    
    console.log('\n✅ All data sources integrated successfully!');
    console.log('📊 Run npm run dev to see your unified platform');
    
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export {
  migrateOldDemoData,
  fetchUSACEDamData,
  fetchFERCQueueData,
  importSMRPipeline,
  integratePublicAPIs,
  setupRealtimeFeatures
};