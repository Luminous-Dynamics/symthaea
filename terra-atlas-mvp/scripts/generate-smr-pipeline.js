"use strict";
/**
 * Generate SMR (Small Modular Reactor) Pipeline Projects
 * Based on real data from companies like NuScale, TerraPower, X-energy, etc.
 */
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var path = require("path");
// List of real SMR developers and their reactor designs
var SMR_DEVELOPERS = [
    { company: 'NuScale Power', reactor: 'NuScale Power Module', capacity_mw: 77 },
    { company: 'TerraPower', reactor: 'Natrium', capacity_mw: 345 },
    { company: 'X-energy', reactor: 'Xe-100', capacity_mw: 80 },
    { company: 'GE Hitachi', reactor: 'BWRX-300', capacity_mw: 300 },
    { company: 'Westinghouse', reactor: 'AP300', capacity_mw: 300 },
    { company: 'Rolls-Royce', reactor: 'UK SMR', capacity_mw: 470 },
    { company: 'Holtec', reactor: 'SMR-160', capacity_mw: 160 },
    { company: 'Kairos Power', reactor: 'KP-FHR', capacity_mw: 140 },
    { company: 'Ultra Safe Nuclear', reactor: 'MMR', capacity_mw: 15 },
    { company: 'Oklo', reactor: 'Aurora', capacity_mw: 1.5 }
];
// Potential sites - often at retiring coal plants
var POTENTIAL_SITES = [
    { state: 'WY', site: 'Kemmerer (Naughton Plant)', lat: 41.7880, lon: -110.5380 },
    { state: 'ID', site: 'Idaho Falls INL', lat: 43.5158, lon: -112.0339 },
    { state: 'WA', site: 'Centralia Coal Plant', lat: 46.7162, lon: -122.9543 },
    { state: 'PA', site: 'Beaver Valley', lat: 40.6231, lon: -80.4342 },
    { state: 'TN', site: 'Clinch River', lat: 35.9089, lon: -84.3808 },
    { state: 'OH', site: 'Portsmouth', lat: 39.0114, lon: -83.0158 },
    { state: 'MI', site: 'Palisades', lat: 42.3228, lon: -86.3148 },
    { state: 'MT', site: 'Colstrip', lat: 45.8842, lon: -106.6123 },
    { state: 'ND', site: 'Coal Creek Station', lat: 47.0914, lon: -101.0542 },
    { state: 'UT', site: 'Hunter Power Plant', lat: 39.1542, lon: -111.0269 },
    { state: 'VA', site: 'Surry', lat: 37.1656, lon: -76.6983 },
    { state: 'SC', site: 'V.C. Summer', lat: 34.2954, lon: -81.3175 },
    { state: 'TX', site: 'Comanche Peak', lat: 32.2983, lon: -97.7856 },
    { state: 'AL', site: 'Bellefonte', lat: 34.7117, lon: -85.9325 },
    { state: 'IN', site: 'Michigan City', lat: 41.7075, lon: -86.8950 }
];
// NRC licensing phases
var LICENSING_PHASES = [
    'Pre-Application',
    'Design Certification',
    'Combined License Application',
    'Early Site Permit',
    'Construction Permit',
    'Operating License'
];
function generateSMRProjects() {
    var projects = [];
    var projectCount = 0;
    // Generate 2-3 projects per site
    POTENTIAL_SITES.forEach(function (site, siteIndex) {
        var numProjects = Math.random() > 0.3 ? 2 : 1;
        for (var i = 0; i < numProjects; i++) {
            projectCount++;
            // Select a developer (weighted towards major players)
            var developerIndex = Math.random() > 0.4
                ? Math.floor(Math.random() * 5) // Top 5 developers more likely
                : Math.floor(Math.random() * SMR_DEVELOPERS.length);
            var developer = SMR_DEVELOPERS[developerIndex];
            var numModules = Math.floor(Math.random() * 8) + 1; // 1-8 modules
            var totalCapacity = developer.capacity_mw * numModules;
            // Realistic timeline based on current SMR development
            var currentYear = 2024;
            var constructionStart = currentYear + Math.floor(Math.random() * 4) + 2; // 2026-2029
            var commercialOperation = constructionStart + 4 + Math.floor(Math.random() * 2); // 4-5 years construction
            // Cost estimates based on industry projections
            var costPerKw = 6000 + Math.random() * 3000; // $6,000-9,000/kW
            var projectCost = totalCapacity * 1000 * costPerKw;
            // Determine licensing phase based on timeline
            var licensingPhase = LICENSING_PHASES[0];
            if (constructionStart <= 2026) {
                licensingPhase = LICENSING_PHASES[Math.floor(Math.random() * 3) + 2];
            }
            else if (constructionStart <= 2028) {
                licensingPhase = LICENSING_PHASES[Math.floor(Math.random() * 2) + 1];
            }
            var project = {
                project_id: "SMR-2024-".concat(String(projectCount).padStart(3, '0')),
                project_name: "".concat(site.site, " SMR Project ").concat(i + 1),
                developer: developer.company,
                reactor_type: developer.reactor,
                capacity_mw: developer.capacity_mw,
                number_of_modules: numModules,
                total_capacity_mw: totalCapacity,
                state: site.state,
                site_name: site.site,
                latitude: site.lat + (Math.random() - 0.5) * 0.1,
                longitude: site.lon + (Math.random() - 0.5) * 0.1,
                licensing_phase: licensingPhase,
                nrc_docket_number: "52-".concat(String(Math.floor(Math.random() * 100) + 1).padStart(3, '0')),
                estimated_construction_start: "".concat(constructionStart, "-Q").concat(Math.floor(Math.random() * 4) + 1),
                estimated_commercial_operation: "".concat(commercialOperation, "-Q").concat(Math.floor(Math.random() * 4) + 1),
                estimated_project_cost: Math.round(projectCost),
                cost_per_kw: Math.round(costPerKw),
                construction_jobs: Math.round(totalCapacity * 3.5), // ~3.5 jobs per MW
                permanent_jobs: Math.round(totalCapacity * 0.5), // ~0.5 permanent jobs per MW
                annual_generation_gwh: Math.round(totalCapacity * 8760 * 0.93 / 1000), // 93% capacity factor
                capacity_factor: 0.93,
                carbon_avoided_tons_per_year: Math.round(totalCapacity * 8760 * 0.93 * 0.82), // 0.82 tons CO2/MWh avoided
                water_source: ['River', 'Lake', 'Ocean', 'Groundwater'][Math.floor(Math.random() * 4)],
                cooling_type: Math.random() > 0.3 ? 'Wet Cooling Tower' : 'Air Cooled',
                grid_connection_voltage_kv: totalCapacity > 200 ? 345 : 230,
                status: licensingPhase.includes('Application') ? 'In Licensing' : 'Pre-Development',
                key_milestones: generateMilestones(licensingPhase, constructionStart),
                project_website: Math.random() > 0.5 ? "https://".concat(developer.company.toLowerCase().replace(/\s+/g, '-'), ".com/projects/").concat(site.state.toLowerCase()) : undefined,
                last_updated: new Date().toISOString().split('T')[0]
            };
            projects.push(project);
        }
    });
    return projects;
}
function generateMilestones(phase, constructionStart) {
    var milestones = [];
    if (phase.includes('Design')) {
        milestones.push('Design Certification Application submitted to NRC');
    }
    if (phase.includes('License')) {
        milestones.push('Combined License Application filed');
        milestones.push('NRC acceptance review completed');
    }
    if (phase.includes('Permit')) {
        milestones.push('Environmental Impact Statement initiated');
        milestones.push('Public hearings scheduled');
    }
    // Add future milestones
    milestones.push("Financing expected Q4 ".concat(constructionStart - 1));
    milestones.push("Construction start targeted ".concat(constructionStart));
    return milestones;
}
// Calculate aggregate statistics
function calculateStats(projects) {
    var totalProjects = projects.length;
    var totalCapacity = projects.reduce(function (sum, p) { return sum + p.total_capacity_mw; }, 0);
    var totalInvestment = projects.reduce(function (sum, p) { return sum + p.estimated_project_cost; }, 0);
    var totalJobs = projects.reduce(function (sum, p) { return sum + p.construction_jobs + p.permanent_jobs; }, 0);
    var totalGeneration = projects.reduce(function (sum, p) { return sum + p.annual_generation_gwh; }, 0);
    var totalCarbonAvoided = projects.reduce(function (sum, p) { return sum + p.carbon_avoided_tons_per_year; }, 0);
    var stateCount = new Set(projects.map(function (p) { return p.state; })).size;
    var developerCount = new Set(projects.map(function (p) { return p.developer; })).size;
    // Projects by phase
    var phaseBreakdown = {};
    projects.forEach(function (p) {
        phaseBreakdown[p.licensing_phase] = (phaseBreakdown[p.licensing_phase] || 0) + 1;
    });
    return {
        total_projects: totalProjects,
        total_capacity_mw: Math.round(totalCapacity),
        total_capacity_gw: (totalCapacity / 1000).toFixed(1),
        total_investment: totalInvestment,
        total_investment_billions: (totalInvestment / 1000000000).toFixed(1),
        avg_cost_per_kw: Math.round(totalInvestment / (totalCapacity * 1000)),
        total_jobs: totalJobs,
        construction_jobs: projects.reduce(function (sum, p) { return sum + p.construction_jobs; }, 0),
        permanent_jobs: projects.reduce(function (sum, p) { return sum + p.permanent_jobs; }, 0),
        total_annual_generation_twh: (totalGeneration / 1000).toFixed(1),
        total_carbon_avoided_million_tons: (totalCarbonAvoided / 1000000).toFixed(1),
        states_with_projects: stateCount,
        number_of_developers: developerCount,
        avg_project_size_mw: Math.round(totalCapacity / totalProjects),
        licensing_phase_breakdown: phaseBreakdown,
        earliest_online_date: Math.min.apply(Math, projects.map(function (p) { return parseInt(p.estimated_commercial_operation); })),
        latest_online_date: Math.max.apply(Math, projects.map(function (p) { return parseInt(p.estimated_commercial_operation); }))
    };
}
// Main execution
function main() {
    console.log('🚀 Generating SMR Pipeline Projects...');
    var projects = generateSMRProjects();
    var stats = calculateStats(projects);
    // Save projects data
    var dataDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir);
    }
    fs.writeFileSync(path.join(dataDir, 'smr-pipeline-projects.json'), JSON.stringify(projects, null, 2));
    fs.writeFileSync(path.join(dataDir, 'smr-stats.json'), JSON.stringify(stats, null, 2));
    console.log("\u2705 Generated ".concat(projects.length, " SMR projects"));
    console.log("\uD83D\uDCCA Statistics:");
    console.log("   - Total Capacity: ".concat(stats.total_capacity_gw, " GW"));
    console.log("   - Total Investment: $".concat(stats.total_investment_billions, "B"));
    console.log("   - States Covered: ".concat(stats.states_with_projects));
    console.log("   - Developers: ".concat(stats.number_of_developers));
    console.log("   - Jobs Created: ".concat(stats.total_jobs.toLocaleString()));
    console.log("   - Annual Generation: ".concat(stats.total_annual_generation_twh, " TWh"));
    console.log("   - CO2 Avoided: ".concat(stats.total_carbon_avoided_million_tons, "M tons/year"));
    console.log("   - Timeline: ".concat(stats.earliest_online_date, " - ").concat(stats.latest_online_date));
    console.log('\n📁 Data saved to:');
    console.log('   - data/smr-pipeline-projects.json');
    console.log('   - data/smr-stats.json');
}
main();
