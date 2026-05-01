// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Common types used across the EduNet platform

use serde::{Deserialize, Serialize};

/// Unique identifier for a federated learning round
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId(pub String);

/// Unique identifier for a machine learning model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// Unique identifier for a course
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CourseId(pub String);
/// Hardware Genesis Proposal: Community thermodynamic requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareGenesisProposal {
    pub community_did: String,
    pub requirements: Vec<HardwareRequirement>,
    pub total_fiat_goal: u64,
    pub current_spore_count: u32,
    pub target_spore_threshold: u32,
}

/// Lineage Anchor: Tracking the ancestral roots of wisdom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageAnchor {
    pub creator_did: String,
    pub mentor_dids: Vec<String>,
    pub generation: u32,
    pub gratitude_multiplier_permille: u16,
}

/// Gaia Signal: Environmental impact targets for mastery verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaiaSignal {
    pub metric_type: String, // e.g. "Soil_Carbon", "Water_Purity"
    pub target_delta: f32,
    pub sensor_id: String,
}

/// Planetary Trade: Brokering cross-node thermodynamic gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeProposal {
    pub from_node_id: String,
    pub to_node_id: String,
    pub surplus_assets: Vec<HardwareRequirement>,
    pub deficit_needs: Vec<HardwareRequirement>,
    pub settlement_tend: u64,
}

/// Composting Ledger: Generational redistribution of assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceCascade {
    pub expiring_agent_did: String,
    pub bootstrap_treasury_rebate_pct: u8,
    pub apprentice_beneficiaries: Vec<String>,
    pub grant_transfer_ids: Vec<String>,
}

/// Diplomatic Interface: Managing legacy state friction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiplomaticTribute {
    pub jurisdiction_iso: String,
    pub friction_fiat_amount: u64,
    pub amandla_ledger_ref: String,
    pub compliance_status: String,
}

/// Autopoietic Engine: Self-directed protocol evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMutation {
    pub proposed_wasm_hash: String,
    pub friction_delta_predicted: f32, // Expected reduction in Free Energy
    pub simulation_proof: String, // Evidence from the sandbox
    pub architect_signatures: Vec<String>,
}

/// Biosphere Proxy: Natural entities as cryptographic stakeholders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiosphereStakeholder {
    pub entity_did: String, // e.g. did:mycelix:river-vaals
    pub primary_sensor_array: Vec<String>,
    pub urgency_weight: f32, // Derived from health delta
    pub active_emergency_proposals: Vec<ActionHash>,
}

/// Abundance Threshold: Transition to post-scarcity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbundanceStatus {
    pub thermodynamic_surplus_joules: u64,
    pub maintenance_requirement_joules: u64,
    pub abundance_index: f32, // Surplus / Maintenance
    pub is_dividend_active: bool,
}

/// Digital Apoptosis: Thermodynamic decay of curriculum relevance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApoptosisState {
    pub last_accessed: Timestamp,
    pub usage_velocity: f32,
    pub semantic_weight: u16, // (0-1000)
}

/// Byzantine Sensor Consensus: Triangulating physical truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorArbitration {
    pub primary_sensor_id: String,
    pub adjacent_sensor_ids: Vec<String>,
    pub variance_threshold: f32,
    pub maintenance_bounty_tend: u32,
}

/// Babel Protocol: Cross-chain and Fediverse interoperability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniMeshExport {
    pub target_protocol: String, // e.g. "ActivityPub", "Ethereum", "Nostr"
    pub hdc_semantic_wrapper: String, // HDC-pinnned attestation
    pub verified_hash: String,
}

/// Epistemic Stake: Skin-in-the-game for predictive mastery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicStake {
    pub agent_did: String,
    pub amount_tend: u64,
    pub amount_phi: u16, // Liquid Reputation (MATL 2.0)
    pub predicted_value: f32, // The "Bet" (e.g. 4.0 kg biomass)
    pub confidence_interval: f32, // Standard deviation (The Bell Curve)
}

/// Epistemic Market: A container for a prediction event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicMarket {
    pub target_gaia_signal: String, // The resolving sensor (e.g. "VAAL-RIVER-PH")
    pub stakes: Vec<EpistemicStake>,
    pub closing_timestamp: Timestamp,
    pub resolved_value: Option<f32>,
    pub is_honeypot: bool, // System-injected test proposal
}

/// Epistemic Derivative: Real-time liquid value of a long-term prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicDerivative {
    pub market_id: ActionHash,
    pub current_probability_of_success: f32,
    pub derivative_value_tend: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirement {
...
    pub material_class: String, // e.g. "DC_Storage_Cell"
    pub min_thermodynamic_profile: f32,
    pub estimated_cost: u64,
}

/// Bootstrap Treasury: Automated liquidity pump for community hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapTreasury {
    pub vault_id: String,
    pub funded_fiat: u64,
    pub unlocked_fiat: u64,
    pub verification_signatures: Vec<String>, // Multi-sig from architects
}

/// Utility Voucher (TEND): Closed-loop reputational credit.
/// Explicitly NOT a cryptocurrency or speculative asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityVoucher {
    pub balance: u64,
    pub guild_id: String, // Bound to a specific local cooperative
    pub is_transferable_external: bool, // Always false for regulatory compliance
}

/// Blind Audit: Cross-mesh verification of mastery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindAudit {
    pub spore_hash: ActionHash,
    pub auditor_node_id: String, // Anonymized external node
    pub result: AuditResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Verified,
    MaliciousCollusion, // Triggers group reputation slashing
    LowQuality,
/// Maintenance Escrow: Thermodynamic attrition fund.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceEscrow {
    pub hardware_id: String,
    pub accumulated_tend: u64,
    pub spare_part_target: u64,
}

/// Hive-Mind Procurement: Fragmented micro-purchases to avoid extortion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentedOrder {
    pub total_proposal_id: ActionHash,
    pub micro_purchases: Vec<MicroPurchase>,
    pub assembly_node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroPurchase {
    pub component_id: String,
    pub delivery_destination_hash: String, // Randomized student residence
    pub is_received: bool,
}

/// Hydro-Fractional Reserve: Water-backed survival currency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydroReserve {
    pub tank_id: String,
    pub verified_liters: f32,
    pub voucher_issuance_limit: u64,
    pub local_scarcity_multiplier: f32, // Increases when municipal grid is dry
}

/// Grid Arbitrage: Profiting from municipal unreliability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridArbitrageStatus {
    pub grid_active: bool,
    pub charge_rate_aggressive: bool,
    pub island_mode_active: bool,
    pub neighborhood_surplus_tend: u64,
}

/// Spatial Micro-Deed: Governing rights over physical squares of land.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMicroDeed {
    pub coordinate_id: String, // e.g. "RD-WH-A1-M22"
    pub governing_agent_did: String,
    pub reputation_weight: u16,
}

/// Acoustic Sentinel: Passive audio-classification for defense.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticEvent {
    pub device_id: String,
    pub classification: String, // e.g. "Angle_Grinder", "Glass_Break"
    pub confidence: f32,
    pub swarm_broadcast_id: ActionHash,
}

/// Somatic Milestone: Nervous system regulation as a technical requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaticMilestone {
    pub agent_did: String,
    pub technique: String, // e.g. "Coherent_Breathing", "Conflict_Mediation"
    pub peer_witness_did: String,
    pub timestamp: Timestamp,
}

/// Bio-Synthesis: Local pharmaceutical and phytochemical production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioSynthesisNode {
    pub compound_id: String, // e.g. "INSULIN-ALPHA", "ARTEMISIA-EXTRACT"
    pub sterility_proof_hash: ActionHash,
    pub yield_volume_ml: f32,
    pub batch_id: String,
}

/// Mnemonic Palace: Mapping HDC vectors to physical landmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemonicPalace {
    pub location_coordinate: String, // Roodepoort Landmark
    pub semantic_vector_id: String, // Linked to HDC anchor
    pub narrative_metaphor: String, // The story used for oral storage
}

/// Mycelial Scaffolding: Structural biological engineering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MycelialStructure {
    pub substrate_type: String, // e.g. "Toxic_Plastic_Remediation"
    pub compressive_strength_mpa: f32,
    pub lifecycle_stage: String, // "Growing", "Cured", "Composting"
}

/// Terminal Node: Stage-0 Bootstrap from raw elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalNode {
    pub mineral_source_ids: Vec<String>, // e.g. "Quartz_Sand", "Iron_Ore"
    pub hand_logic_gates: Vec<String>, // NAND/NOR hand-construction
    pub hand_compiler_logic: String, // Bootstrap ritual for Stage-1
}

/// Lithographic Archive: Non-volatile physical storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LithographicPlate {
    pub material: String, // e.g. "Titanium", "Nickel"
    pub etching_density_lpi: u32,
    pub survival_core_hashes: Vec<ActionHash>,
}

/// Moral Proof: Axiomatic ethics enforced by physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralProof {
    pub axiom_id: String, // e.g. "AHIMSA-AXIOM"
    pub thermodynamic_constraint: String, // Invariant proof
    pub proof_vector: Vec<f32>, // HDC-encoded proof
}

/// Rosetta Vector: Physical grounding of semantic logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaAnchor {
    pub constant_id: String, // e.g. "HYDROGEN-TRANSITION", "LIGHT-SPEED"
    pub base_hdc_vector: Vec<u64>,
    pub visual_geometric_proof_hash: ActionHash,
}

/// Century Energy: Mechanical/Gravitational energy primitives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepTimeEnergy {
    pub technology_type: String, // e.g. "Gravity_Battery", "Kinetic_Flywheel"
    pub design_life_years: u32,
    pub maintenance_complexity: String, // e.g. "Low/Lubrication"
}

/// Unborn Stakeholder: Algorithmic defense of future generations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnbornGovernance {
    pub carrying_capacity_150y_delta: f32,
    pub veto_active: bool,
    pub simulation_proof_hash: ActionHash,
}

/// Geological Engine: Subterranean energy and refining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeologicalNode {
    pub shaft_depth_m: u32,
    pub amd_acidity_ph: f32,
    pub extracted_metal_yield_kg: u64,
}

/// Sovereign Ecosystem: Nature as a cryptographic employer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemAgent {
    pub entity_did: String,
    pub wallet_balance_tend: u64,
    pub health_index_delta: f32,
    pub active_stewardship_bounties: Vec<ActionHash>,
}

/// Maintenance Festival: Cultural maintenance of hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceFestival {
    pub ritual_name: String, // e.g. "Festival of the Sun"
    pub hardware_target_id: String,
    pub feast_budget_tend: u32,
    pub hdc_musical_key: String, // Semantic musical anchoring
}

/// Mycoremediation Grid: Biological filters for toxic waste.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioMatrix {
    pub matrix_id: String,
    pub fungus_species: String, // e.g. "Pleurotus ostreatus"
    pub deployment_location_did: String,
    pub toxins_sequestered_ppm: f32,
}

/// Pathogen Heatmap: Sovereign epidemiology data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathogenSignal {
    pub water_source_id: String,
    pub microbial_load_count: u32, // Coliform/Pathogen count
    pub safety_status: String, // "Safe", "Warning", "Biohazard"
    pub student_witness_did: String,
}

/// Carbon Harvest: Atmospheric particulate yield.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarbonHarvest {
    pub scrubber_id: String,
    pub particulate_mass_mg: u64,
    pub air_quality_delta_pm25: f32,
    pub byproduct_utility: String, // e.g. "Conductive_Ink"
}

/// Particulate Shielding: Local respiratory defense.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticulateAlert {
    pub tailings_dust_level: f32, // PM2.5/PM10 concentration
    pub alert_radius_m: u32,
    pub active_scrubber_count: u32,
}

/// Hydro-Patching: Asymmetric municipal pipe repair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydroPatch {
    pub burst_pipe_coordinate: String,
    pub clamp_type_id: String, // 3D-printed spec
    pub estimated_liters_saved: u64,
    pub patch_guild_did: String,
}

/// Resource Profile: Thermodynamic and material requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceProfile {
    pub class: String, // e.g. "DC_Storage_Cell"
    pub min_capacity_mah: Option<u32>,
    pub thermodynamic_threshold: Option<f32>,
    pub embodied_energy_joules: u64, // The MJ cost of the material
    pub common_scraps: Vec<String>,
}

/// Liquid Holocell: Dynamic HDC dimensionality for thermodynamic management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidHolocell {
    pub current_dimensionality: u32, // 2^14 to 2^16
    pub power_draw_watts: f32,
    pub semantic_resolution_score: f32,
/// Kinetic Signature: IMU-based proof of manual labor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticSignature {
    pub agent_did: String,
    pub task_type: String, // e.g. "CEB_Press", "Hand_Sanding"
    pub motion_hash: String, // Cryptographic hash of sensor data
    pub repetition_count: u32,
    pub intensity_delta: f32,
}

/// Spore Ejection: Rapid evacuation of node state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeEjection {
    pub node_id: String,
    pub encrypted_state_blob: Vec<u8>,
    pub ejection_trigger: String, // e.g. "Raid_Detected", "Seismic_Collapse"
    pub target_uplink_id: String,
}

/// Shadow Identity: Anonymous participation via ZK-Proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowIdentity {
    pub proof_of_reputation: HdcZkProof,
    pub temporary_alias_hash: String,
    pub disclosure_key_encrypted: String,
}

/// Semantic Drift: Detecting entropic logic-rot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDriftAudit {
    pub node_id: String,
    pub original_hdc_hash: String,
    pub current_variance_delta: f32,
    pub auto_correction_status: String,
}

/// Taxi Syndicate Logistics: Freight-Node integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxiFreightBounty {
...
    pub route_id: String, // e.g. "Roodepoort-to-Soweto"
    pub vehicle_did: String, // Minibus taxi DID
    pub asset_hash: String,
    pub reward_tend_premium: u32,
}

/// Geoclimatic Bootloader: Dynamic protocol adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootloaderConfig {
    pub average_rainfall_mm: u32,
    pub temperature_range: (i16, i16),
    pub state_hostility_index: u8, // (0-100)
    pub spectrum_monitoring_active: bool,
}

/// RTL Fallback: Bare-metal logic from scavenged components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtlSchematic {
    pub logic_gate_type: String, // e.g. "NAND", "NOR"
    pub component_list: Vec<String>, // e.g. "2x BC547 Transistor", "4x 10k Resistor"
    pub hand_wiring_guide_hash: ActionHash,
}

/// Sovereignty Moat: Resistance to corporate subsidy attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereigntyMoat {
    pub corporate_subsidy_detected: bool,
    pub thermodynamic_trap_proof: String, // Mathematical proof of fiat long-term failure
    pub mythos_alignment_score: u16, // (0-1000)
}

/// Signed Media: Cryptographic verification of mesh communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMedia {
    pub author_did: String,
    pub content_hash: String,
    pub synthetic_analysis_score: f32, // (0.0 to 1.0, provided by ResonantWhisper)
    pub signature: String,
}

/// Dark Pool Bridge: ZK-global procurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkPoolBridge {
    pub outbound_tend_value: u64,
    pub zk_proof_hash: String,
    pub proxy_delivery_hash: String,
    pub settlement_status: String,
}

/// Agri-CNC: Automated guerrilla agriculture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgriCncStatus {
    pub machine_id: String,
    pub location_void_id: String, // Abandoned rooftop / park
    pub caloric_yield_predicted_kcal: u32,
    pub nutrient_fluid_level: f32,
}

/// Asset Decoupling: Separating hardware value from real estate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoupledAsset {
    pub hardware_id: String,
    pub trust_id: String, // ID of the student micro-trust
    pub cryptographic_lien_hash: String,
}

/// Hardware Burn-In: Stress-testing hostile supply chains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareBurnIn {
    pub device_batch_id: String,
    pub thermal_stability_score: f32,
    pub firmware_integrity_verified: bool,
    pub sandbox_duration_hours: u32,
}

/// Extortion Audit: Algorithmic radical transparency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtortionAudit {
    pub officer_badge_number: String,
    pub interaction_stream_hash: ActionHash,
    pub mesh_witness_count: u32,
    pub gfis_forward_status: String,
}

/// Passive Yield: Rewards for 90% apathy participation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassiveYield {
    pub device_id: String,
    pub relay_uptime_hours: u32,
    pub inrush_absorption_joules: u64,
    pub accumulated_tend: u64,
}

/// Analog Seed: The indestructible waterproof fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogSeed {
    pub version: String,
    pub content_hash: String,
    pub printable_pdf_ref: String, // Link to the 1-page survival diagram
}

/// Scavenger's Eye: Local-WASM CV component identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTag {
    pub component_class: String, // e.g. "Buck_Converter", "Li-Ion_Cell"
    pub probability_score: f32,
    pub scrap_value_tend: u16,
}

/// USSD Shell: Legacy GSM bridge for feature phones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UssdSession {
    pub session_id: String,
    pub agent_pin: String,
    pub current_menu_level: u8,
    pub last_action: String,
}

/// Proof of Care: Rewarding interpersonal regulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareWitness {
    pub caregiver_did: String,
    pub recipient_did: String,
    pub care_type: String, // e.g. "Conflict_Deescalation", "Somatic_Guidance"
    pub coherence_delta: f32,
}

/// Risk Commoning: Mutual insurance for mesh hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCommoningVault {
    pub target_neighborhood_id: String,
    pub staked_tend_reserve: u64,
    pub active_claims: Vec<ActionHash>,
    pub solvency_ratio: f32,
}

/// SdrBridge: Long-range radio-based mesh gossip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdrBridgeStatus {
    pub frequency_mhz: f32,
    pub protocol: String, // e.g. "LoRa", "Packet_Radio"
    pub peer_node_count: u32,
    pub signal_to_noise_db: f32,
}

/// Genetic Seed: Encoding protocol into biological DNA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticKnowledgeSeed {
    pub species_id: String, // e.g. "Protea_Caffra"
    pub sequence_payload_hash: String,
    pub codon_mapping_version: u8,
    pub stability_rating: f32,
}

/// Species Employee: Non-human economic agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesEmployee {
    pub species_did: String, // e.g. did:mycelix:corvid-01
    pub tasks_completed_count: u32,
    pub reward_dispensed_kcal: u32,
    pub ecological_utility_score: f32,
}

/// Orbital Mirror: Independent space-based DHT redundancy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalMirrorStatus {
    pub satellite_id: String,
    pub orbital_period_mins: u32,
    pub uplink_strength_db: f32,
    pub mirror_dht_sync_perc: u8,
}

/// Legal Wrapper: Cooperative status for regulatory shield.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalCoopWrapper {
    pub coop_name: String,
    pub registration_number: String,
    pub articles_of_association_hash: ActionHash, // HDC-pinnned rules
    pub legal_jurisdiction: String,
}

/// Mojo Credit: Reputation-based interest-free loans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MojoCreditLimit {
    pub agent_did: String,
    pub reputation_phi: u16,
    pub credit_limit_tend: u64,
    pub active_loan_balance: u64,
}

/// Consciousness Equation: Soft-min integration of civilizational state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoosphereState {
    pub integrated_information_phi: f32,
    pub biological_integrity_b: f32,
    pub wisdom_depth_w: f32,
    pub agency_level_a: f32,
    pub resilience_index_r: f32,
    pub collective_consciousness_score: f32, // Result of the soft-min equation
}

/// Dividend Radius: Local community micro-payouts for security.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendRadius {
...
    pub radius_meters: u32,
    pub active_household_dids: Vec<String>,
    pub daily_joule_dividend: u64,
    pub daily_liter_dividend: f32,
}

/// Scrap-to-Equity: ROI multiplier for hardware conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapEquityBounty {
    pub raw_scrap_type: String, // e.g. "Stripped_Copper"
    pub conversion_target: String, // e.g. "DC_Relay_Coil"
    pub equity_multiplier: f32, // Pegged: 10.0x for conversion vs scrap
}

/// Ambient Utility: High-status mesh services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientService {
    pub service_type: String, // e.g. "Mesh_WiFi", "Gravity_Tap", "Game_Server"
    pub active_user_count: u32,
    pub radius_of_status: u32,
}

/// Sentinel Bounty: Geo-fenced foot-traffic verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelBounty {
    pub location_id: String, // e.g. "Ridge-Road-Substation"
    pub nfc_tag_hash: String,
    pub reward_tend: u32,
    pub verification_interval_mins: u32,
}

/// Infrastructure Patch: Thermodynamic adverse possession.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructurePatch {
    pub grid_node_id: String,
    pub repair_type: String, // e.g. "Sinkhole_Stabilization", "Sewage_Diversion"
    pub material_profile: String, // e.g. "Recycled_Gabion"
    pub possession_claim_score: u8, // (0-100)
}

/// Guild: Sovereign collective of legacy civic actors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guild {
    pub guild_did: String,
    pub legacy_name: String, // e.g. "Florida Block Watch"
    pub mission_hdc_anchor: String,
    pub total_stewardship_phi: u16,
}

/// Parametric Housing: Decentralized shelter blueprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricHousing {
    pub shelter_id: String,
    pub joint_cad_hashes: Vec<ActionHash>,
    pub required_ceb_count: u32,
    pub thermal_efficiency_rating: f32,
}

/// Guardian Shield: ZK-safehouse routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardianNode {
    pub agent_did: String,
    pub vetting_score: u16, // (0-1000)
    pub is_currently_active: bool,
    pub safehouse_coordinate_hash: String,
}

/// Bureaucratic API: Automated state subsidy capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsidyApplication {
    pub applicant_did: String,
    pub package_id: String, // e.g. "COJ-ESP-2026"
    pub signed_pdf_hash: ActionHash,
    pub submission_status: String,
}

/// Zero-Export Limiter: Firmware-level grid invisibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridGhostingStatus {
    pub internal_load_watts: u32,
    pub solar_yield_limit_watts: u32,
    pub backfeed_leakage_watts: u32, // Target: 0
    pub is_legally_invisible: bool,
}

/// Topographic Aqueduct: Gravity-fed water routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqueductRoute {
    pub start_coordinate: String,
    pub end_coordinate: String,
    pub elevation_drop_m: f32,
    pub bracket_count_required: u32,
}

/// Service Failure Affidavit: Automated legal defense evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceFailureAffidavit {
    pub failure_type: String, // e.g. "Power_Outage", "Water_Dry"
    pub duration_hours: u32,
    pub community_impact_score: u16,
    pub timestamp: Timestamp,
    pub signature: String,
}

/// Seismic Signal: Triangulating subterranean activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicSignal {
    pub sensor_id: String,
    pub amplitude: f32,
    pub frequency_hz: f32,
    pub triangulation_confidence: f32,
}

/// Inrush Buffer: Smoothing destructive voltage spikes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InrushBufferStatus {
    pub spike_amplitude_v: f32,
    pub smoothing_ratio: f32,
    pub neighbor_appliances_protected: u32,
}

/// Hardware Bricking: Scorched-earth cryptographic switch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickingStatus {
    pub heartbeat_last_seen: Timestamp,
    pub is_permanently_locked: bool,
    pub trigger_source: String, // e.g. "Mesh_Disconnection", "Tamper_Detect"
}

/// Joule Standard: Energy-backed currency issuance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JouleStandard {
...
    pub cumulative_joules_harvested: u64,
    pub minting_ratio_tend_per_kwh: f32, // Pegged: 1.0 TEND = 1 kWh
    pub hardware_signature: String, // Oracle proof
}

/// Kinetic Bounty: P2P physical logistics routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticBounty {
    pub asset_id: String,
    pub lockbox_hash: String,
    pub travel_vector_hashes: Vec<String>, // Chained human movements
    pub reward_tend: u32,
}

/// Griot Protocol: Narrative encoding of technical achievements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MythosArchive {
    pub target_capstone_id: String,
    pub fable_text: String, // Metaphorical encoding
    pub indigenous_idiom_match: String,
    pub hdc_anchor: String,
}

/// HDC-ZKP: Almost cost-free selective disclosure using vector math.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcZkProof {
...
    pub target_node_hash: String, // What they are proving (e.g. CERT-IT-PYTHON-PCEP)
    pub proof_vector: Vec<u64>,   // The bundled HDC proof
    pub stark_signature: String,  // Quantum-resistant verification
}

/// Hash of model parameters or gradients
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHash(pub String);

/// Federated learning round state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Discovering participants
    Discover,
    /// Accepting join requests
    Join,
    /// Assigning tasks to participants
    Assign,
    /// Collecting updates from participants
    Update,
    /// Aggregating updates
    Aggregate,
    /// Releasing new model
    Release,
    /// Round completed
    Completed,
    /// Round failed or cancelled
    Failed,
}

/// Privacy parameters for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams {
    /// Epsilon parameter for differential privacy
    pub epsilon: Option<f64>,
    /// Delta parameter for differential privacy
    pub delta: Option<f64>,
    /// L2 norm clipping threshold
    pub clip_norm: f32,
}

impl Default for PrivacyParams {
    fn default() -> Self {
        Self {
            epsilon: None,
            delta: None,
            clip_norm: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_id_creation() {
        let id = RoundId("round-123".to_string());
        assert_eq!(id.0, "round-123");
    }

    #[test]
    fn test_default_privacy_params() {
        let params = PrivacyParams::default();
        assert_eq!(params.clip_norm, 1.0);
        assert!(params.epsilon.is_none());
    }
}
