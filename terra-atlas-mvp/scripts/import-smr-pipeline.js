#!/usr/bin/env node

// Import Small Modular Reactor (SMR) Pipeline Data
// The future of nuclear - high investor interest

const fs = require('fs');

// SMR Projects currently in development
const SMR_PIPELINE = [
  {
    company: 'NuScale',
    reactor: 'NuScale VOYGR',
    name: 'Carbon Free Power Project',
    location: 'Idaho Falls, ID',
    latitude: 43.4666,
    longitude: -112.0341,
    capacityMw: 462, // 6 x 77MW modules
    moduleCount: 6,
    moduleCapacityMw: 77,
    status: 'NRC Approved',
    nrcStatus: 'Design Certification Approved',
    codDate: '2029',
    developer: 'Utah Associated Municipal Power Systems (UAMPS)',
    estimatedCostMillion: 9300,
    lcoePerMwh: 89,
    description: 'First NRC-approved SMR in the US. Six-module plant at INL.'
  },
  {
    company: 'TerraPower',
    reactor: 'Natrium',
    name: 'Kemmerer Power Station',
    location: 'Kemmerer, WY',
    latitude: 41.7922,
    longitude: -110.5379,
    capacityMw: 345, // Can boost to 500MW for 5.5 hours
    storageCapacityMw: 500,
    status: 'Construction Prep',
    nrcStatus: 'Under Review',
    codDate: '2030',
    developer: 'TerraPower & PacifiCorp',
    estimatedCostMillion: 4000,
    lcoePerMwh: 85,
    description: 'Bill Gates-backed sodium-cooled reactor with molten salt storage. Replacing coal plant.'
  },
  {
    company: 'X-energy',
    reactor: 'Xe-100',
    name: 'Dow Gulf Coast Project',
    location: 'Seadrift, TX',
    latitude: 28.4147,
    longitude: -96.7114,
    capacityMw: 320, // 4 x 80MW modules
    moduleCount: 4,
    moduleCapacityMw: 80,
    status: 'Site Selected',
    nrcStatus: 'Under Review',
    codDate: '2030',
    developer: 'X-energy & Dow Chemical',
    estimatedCostMillion: 2800,
    lcoePerMwh: 75,
    description: 'High-temp gas reactor for industrial heat and power at Dow chemical facility.'
  },
  {
    company: 'GE Hitachi',
    reactor: 'BWRX-300',
    name: 'Ontario Power Generation - Darlington',
    location: 'Clarington, ON, Canada',
    latitude: 43.8728,
    longitude: -78.7194,
    capacityMw: 300,
    moduleCount: 1,
    moduleCapacityMw: 300,
    status: 'License Application',
    nrcStatus: 'Pre-application',
    codDate: '2029',
    developer: 'Ontario Power Generation',
    estimatedCostMillion: 2000,
    lcoePerMwh: 65,
    description: 'First grid-scale SMR in Canada. Natural circulation, passive safety.'
  },
  {
    company: 'Kairos Power',
    reactor: 'KP-FHR',
    name: 'Hermes Low-Power Demonstration',
    location: 'Oak Ridge, TN',
    latitude: 35.9312,
    longitude: -84.3104,
    capacityMw: 35, // Demo reactor
    status: 'Construction Permit',
    nrcStatus: 'Construction Permit Approved',
    codDate: '2027',
    developer: 'Kairos Power',
    estimatedCostMillion: 100,
    lcoePerMwh: 0, // Demo, not commercial
    description: 'Fluoride salt-cooled high-temp reactor demonstration at East Tennessee Technology Park.'
  },
  {
    company: 'Westinghouse',
    reactor: 'AP300',
    name: 'Ukraine Recovery Project',
    location: 'Multiple Sites, Ukraine',
    latitude: 50.4501,
    longitude: 30.5234,
    capacityMw: 300,
    moduleCount: 1,
    moduleCapacityMw: 300,
    status: 'Planning',
    nrcStatus: 'Design Phase',
    codDate: '2032',
    developer: 'Energoatom',
    estimatedCostMillion: 2500,
    lcoePerMwh: 70,
    description: 'Part of Ukraine energy independence strategy. Based on proven AP1000 technology.'
  },
  {
    company: 'Rolls-Royce',
    reactor: 'UK SMR',
    name: 'Trawsfynydd Site',
    location: 'Gwynedd, Wales, UK',
    latitude: 52.9086,
    longitude: -3.9319,
    capacityMw: 470,
    status: 'Site Assessment',
    nrcStatus: 'UK GDA Process',
    codDate: '2032',
    developer: 'Rolls-Royce SMR Ltd',
    estimatedCostMillion: 2800,
    lcoePerMwh: 60,
    description: 'UK-designed PWR at former Magnox site. Part of UK net-zero strategy.'
  },
  {
    company: 'Holtec',
    reactor: 'SMR-160',
    name: 'Oyster Creek Site',
    location: 'Forked River, NJ',
    latitude: 39.8143,
    longitude: -74.2069,
    capacityMw: 160,
    status: 'Pre-application',
    nrcStatus: 'Pre-application',
    codDate: '2031',
    developer: 'Holtec International',
    estimatedCostMillion: 1200,
    lcoePerMwh: 68,
    description: 'Repurposing decommissioned nuclear site. Walk-away safe design.'
  },
  {
    company: 'NuScale',
    reactor: 'NuScale VOYGR-12',
    name: 'Romania Doicesti Project',
    location: 'Doicesti, Romania',
    latitude: 44.6181,
    longitude: 25.2364,
    capacityMw: 462, // 6 x 77MW
    moduleCount: 6,
    moduleCapacityMw: 77,
    status: 'FEED Study',
    nrcStatus: 'NRC Approved (US)',
    codDate: '2031',
    developer: 'Nuclearelectrica',
    estimatedCostMillion: 7000,
    lcoePerMwh: 75,
    description: 'First SMR deployment in Europe. Former coal plant site.'
  },
  {
    company: 'Last Energy',
    reactor: 'PWR-20',
    name: 'UK Industrial Sites',
    location: 'Multiple, UK',
    latitude: 52.3555,
    longitude: -1.1743,
    capacityMw: 20,
    moduleCount: 1,
    moduleCapacityMw: 20,
    status: 'Development',
    nrcStatus: 'Pre-licensing',
    codDate: '2028',
    developer: 'Last Energy',
    estimatedCostMillion: 100,
    lcoePerMwh: 90,
    description: 'Micro-reactor for industrial heat and power. Factory-built, truck-delivered.'
  }
];

function analyzeSMRPipeline() {
  console.log('⚛️  SMR (Small Modular Reactor) Pipeline Analysis');
  console.log('==================================================\n');
  
  // Summary statistics
  const totalProjects = SMR_PIPELINE.length;
  const totalCapacityMW = SMR_PIPELINE.reduce((sum, p) => sum + p.capacityMw, 0);
  const totalInvestment = SMR_PIPELINE.reduce((sum, p) => sum + (p.estimatedCostMillion || 0), 0);
  const avgLCOE = SMR_PIPELINE.filter(p => p.lcoePerMwh > 0)
    .reduce((sum, p, _, arr) => sum + p.lcoePerMwh / arr.length, 0);
  
  // By status
  const byStatus = {};
  SMR_PIPELINE.forEach(p => {
    byStatus[p.status] = (byStatus[p.status] || 0) + 1;
  });
  
  // By company
  const byCompany = {};
  SMR_PIPELINE.forEach(p => {
    byCompany[p.company] = (byCompany[p.company] || 0) + 1;
  });
  
  console.log('📊 PIPELINE SUMMARY');
  console.log('-------------------');
  console.log(`Total Projects: ${totalProjects}`);
  console.log(`Total Capacity: ${totalCapacityMW.toLocaleString()} MW`);
  console.log(`Total Investment: $${(totalInvestment/1000).toFixed(1)} billion`);
  console.log(`Average LCOE: $${avgLCOE.toFixed(0)}/MWh`);
  console.log(`Average Project Size: ${(totalCapacityMW/totalProjects).toFixed(0)} MW\n`);
  
  console.log('📈 BY REGULATORY STATUS');
  console.log('-----------------------');
  Object.entries(byStatus).sort((a,b) => b[1] - a[1]).forEach(([status, count]) => {
    console.log(`${status}: ${count} projects`);
  });
  
  console.log('\n🏢 BY COMPANY');
  console.log('-------------');
  Object.entries(byCompany).sort((a,b) => b[1] - a[1]).forEach(([company, count]) => {
    console.log(`${company}: ${count} project${count > 1 ? 's' : ''}`);
  });
  
  console.log('\n🎯 KEY PROJECTS TO WATCH');
  console.log('------------------------');
  
  // Highlight key projects
  const keyProjects = [
    SMR_PIPELINE.find(p => p.company === 'NuScale' && p.location.includes('Idaho')),
    SMR_PIPELINE.find(p => p.company === 'TerraPower'),
    SMR_PIPELINE.find(p => p.company === 'X-energy'),
    SMR_PIPELINE.find(p => p.company === 'GE Hitachi')
  ];
  
  keyProjects.forEach(project => {
    if (project) {
      console.log(`\n🔹 ${project.name}`);
      console.log(`   Company: ${project.company} (${project.reactor})`);
      console.log(`   Location: ${project.location}`);
      console.log(`   Capacity: ${project.capacityMw} MW`);
      console.log(`   Status: ${project.status}`);
      console.log(`   Target COD: ${project.codDate}`);
      console.log(`   Est. Cost: $${(project.estimatedCostMillion/1000).toFixed(1)}B`);
      console.log(`   LCOE: $${project.lcoePerMwh}/MWh`);
      console.log(`   ${project.description}`);
    }
  });
  
  console.log('\n💡 INVESTMENT INSIGHTS');
  console.log('----------------------');
  console.log('• First movers: NuScale (NRC approved), TerraPower (Gates-backed)');
  console.log('• Industrial focus: X-energy with Dow, Last Energy for direct heat');
  console.log('• Geographic diversity: US, Canada, UK, Europe, Ukraine');
  console.log('• Cost trajectory: $60-90/MWh, competitive with renewables + storage');
  console.log('• Timeline: First deployments 2027-2029, scale-up 2030+');
  
  console.log('\n🚀 MARKET OPPORTUNITY');
  console.log('---------------------');
  console.log('• Near-term (2025-2030): ~2 GW, $15B investment');
  console.log('• Mid-term (2030-2035): ~10 GW, $60B investment');
  console.log('• Long-term (2035-2050): ~100+ GW globally');
  console.log('• Key drivers: Grid stability, industrial heat, coal replacement');
  
  // Database import (commented for demo)
  console.log('\n💾 Database Import');
  console.log('------------------');
  console.log(`Would import ${totalProjects} SMR projects to database`);
  console.log('(Database import disabled for demo)');
  
  // In production:
  // for (const project of SMR_PIPELINE) {
  //   await db.insert(energyProjects).values({
  //     projectType: 'nuclear',
  //     subType: 'smr',
  //     name: project.name,
  //     // ... other fields
  //   });
  // }
  
  return {
    totalProjects,
    totalCapacityMW,
    totalInvestmentBillion: totalInvestment / 1000,
    averageLCOE: avgLCOE.toFixed(0)
  };
}

// Run the analysis
if (require.main === module) {
  const results = analyzeSMRPipeline();
  console.log('\n✅ SMR Pipeline Analysis Complete!');
  process.exit(0);
}

module.exports = { SMR_PIPELINE, analyzeSMRPipeline };