// Import SMR Pipeline Data to Database
// TypeScript implementation using Drizzle ORM

import { db } from '../lib/drizzle/db';
import { energyProjects } from '../lib/drizzle/schema-energy';
import { SMR_PIPELINE } from './import-smr-pipeline';

interface SMRProject {
  company: string;
  reactor: string;
  name: string;
  location: string;
  latitude: number;
  longitude: number;
  capacityMw: number;
  moduleCount?: number;
  moduleCapacityMw?: number;
  status: string;
  nrcStatus?: string;
  codDate: string;
  developer: string;
  estimatedCostMillion: number;
  lcoePerMwh: number;
  description: string;
  storageCapacityMw?: number;
}

async function importSMRProjects() {
  console.log('⚛️  Importing SMR Pipeline to Database');
  console.log('=====================================\n');

  let imported = 0;
  let failed = 0;

  for (const smr of SMR_PIPELINE) {
    try {
      // Prepare the data for insertion
      const projectData = {
        projectType: 'nuclear',
        subType: 'smr',
        name: smr.name,
        description: smr.description,
        latitude: smr.latitude.toString(),
        longitude: smr.longitude.toString(),
        state: extractState(smr.location),
        country: extractCountry(smr.location),
        capacityMw: smr.capacityMw.toString(),
        
        // Calculate annual generation (nuclear has ~90% capacity factor)
        annualGenerationGwh: (smr.capacityMw * 8760 * 0.9 / 1000).toString(),
        capacityFactor: '90',
        
        status: mapStatus(smr.status),
        developer: smr.developer,
        owner: smr.developer,
        operator: smr.company,
        
        // Financial metrics
        totalCostMillion: smr.estimatedCostMillion.toString(),
        lcoePerMwh: smr.lcoePerMwh.toString(),
        expectedIrr: calculateIRR(smr.lcoePerMwh).toString(),
        
        // Environmental impact (nuclear = near zero emissions)
        co2AvoidedTonsYear: (smr.capacityMw * 8760 * 0.9 * 0.5).toString(), // 0.5 tons CO2/MWh avoided
        
        // Dates
        plannedCod: new Date(`${smr.codDate}-01-01`),
        
        // Data source
        dataSourceId: 'SMR_PIPELINE',
        dataSourceName: 'Industry SMR Development Pipeline',
        externalId: `SMR_${smr.company.replace(/\s+/g, '_').toUpperCase()}_${smr.name.replace(/\s+/g, '_').toUpperCase()}`,
        
        // Additional properties as JSON
        properties: {
          reactor_type: smr.reactor,
          module_count: smr.moduleCount || 1,
          module_capacity_mw: smr.moduleCapacityMw || smr.capacityMw,
          nrc_status: smr.nrcStatus || 'Pre-application',
          storage_capacity_mw: smr.storageCapacityMw || null,
          company: smr.company,
          location_detail: smr.location
        },

        // Timestamps
        createdAt: new Date(),
        updatedAt: new Date()
      };

      // Insert into database
      await db.insert(energyProjects).values(projectData);
      
      imported++;
      console.log(`✅ Imported: ${smr.name} (${smr.capacityMw}MW)`);
      console.log(`   Location: ${smr.location}`);
      console.log(`   Technology: ${smr.company} ${smr.reactor}`);
      console.log(`   Status: ${smr.status}`);
      console.log(`   COD: ${smr.codDate}`);
      console.log(`   Cost: $${(smr.estimatedCostMillion/1000).toFixed(1)}B`);
      console.log(`   LCOE: $${smr.lcoePerMwh}/MWh\n`);
      
    } catch (error) {
      failed++;
      console.error(`❌ Failed to import ${smr.name}:`, error);
    }
  }

  console.log('=====================================');
  console.log('📊 Import Summary');
  console.log('=====================================');
  console.log(`✅ Successfully imported: ${imported} projects`);
  console.log(`❌ Failed: ${failed} projects`);
  console.log(`💰 Total capacity: ${SMR_PIPELINE.reduce((sum, p) => sum + p.capacityMw, 0).toLocaleString()} MW`);
  console.log(`💵 Total investment: $${(SMR_PIPELINE.reduce((sum, p) => sum + p.estimatedCostMillion, 0)/1000).toFixed(1)}B`);
  
  return { imported, failed };
}

// Helper function to extract state from location string
function extractState(location: string): string {
  const stateMatch = location.match(/,\s*([A-Z]{2})/);
  if (stateMatch) return stateMatch[1];
  
  // Handle international locations
  if (location.includes('Canada')) return 'ON';
  if (location.includes('UK') || location.includes('Wales')) return 'UK';
  if (location.includes('Ukraine')) return 'UA';
  if (location.includes('Romania')) return 'RO';
  
  return 'XX'; // Unknown
}

// Helper function to extract country
function extractCountry(location: string): string {
  if (location.includes('Canada')) return 'Canada';
  if (location.includes('UK') || location.includes('Wales')) return 'United Kingdom';
  if (location.includes('Ukraine')) return 'Ukraine';
  if (location.includes('Romania')) return 'Romania';
  return 'USA'; // Default to USA
}

// Map SMR status to our standard statuses
function mapStatus(status: string): string {
  const statusMap: Record<string, string> = {
    'NRC Approved': 'permitted',
    'Construction Prep': 'construction',
    'Site Selected': 'development',
    'License Application': 'permitting',
    'Construction Permit': 'construction',
    'Planning': 'planning',
    'FEED Study': 'development',
    'Development': 'development',
    'Pre-application': 'planning',
    'Site Assessment': 'planning'
  };
  
  return statusMap[status] || 'planning';
}

// Calculate expected IRR based on LCOE
function calculateIRR(lcoe: number): number {
  // Rough IRR calculation based on LCOE competitiveness
  if (lcoe === 0) return 0; // Demo project
  if (lcoe < 60) return 15; // Very competitive
  if (lcoe < 70) return 12;
  if (lcoe < 80) return 10;
  if (lcoe < 90) return 8;
  return 6; // Higher LCOE = lower returns
}

// Run the import if called directly
if (require.main === module) {
  importSMRProjects()
    .then(result => {
      console.log('\n✅ SMR pipeline import complete!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Import failed:', error);
      process.exit(1);
    });
}

export { importSMRProjects };