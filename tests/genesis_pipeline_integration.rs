#![cfg(feature = "genesis")]
//! Cross-crate integration tests for the genesis pipeline.
//!
//! Tests the handoffs between the 5 genesis sub-crates:
//! genomics -> cell-foundry -> ectogenesis -> nurture -> population
//!
//! Each test verifies that types and data flow correctly across crate boundaries,
//! ensuring the full DNA-to-human pipeline is API-compatible.

use rand::rngs::StdRng;
use rand::SeedableRng;

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_neuromodulators::NeuromodulatorBath;

// ─── Genomics ────────────────────────────────────────────────────────────────
use symthaea_genomics::{DamageModel, HdcAssembler, MockSequencer, QualityAssessor};

// ─── Cell Foundry ────────────────────────────────────────────────────────────
use symthaea_cell_foundry::{
    encode_cell_state, CellState, CellType, CultureEnvironment, EthicsGate, GermCellStage,
    IvgProtocol,
};

// ─── Ectogenesis ─────────────────────────────────────────────────────────────
use symthaea_ectogenesis::{
    encode_fetal_state, ConsentProxy, EctogenesisEthicsGate, FetalMetrics, FetalMonitor,
    GestationalWeek,
};

// ─── Nurture ─────────────────────────────────────────────────────────────────
use symthaea_nurture::{
    AttachmentNeuromodulation, AttachmentStyle, AttachmentSystem, CaregiverAction,
};

// ─── Population ──────────────────────────────────────────────────────────────
use symthaea_population::{
    encode_individual, encode_locus, heterozygosity_after_generations,
    observed_heterozygosity, Allele, BiologicalSex, BreedingStrategy,
    EthicalTension, Genotype, GovernanceTier, Individual, Locus, MatingPair, Pedigree,
    PedigreeEntry, Population, PopulationDecision, PopulationSimulator,
};

// =============================================================================
// TEST 1: Genomics assembly feeds into cell state construction
// =============================================================================

#[test]
fn test_genomics_assembly_to_cell_state() {
    // Step 1: Generate a synthetic genome and sequence it
    let genome = MockSequencer::generate_genome(1000, 42);
    let damage_model = DamageModel::new(15.0);
    let sequencer = MockSequencer::new(150, 0.01, 10, damage_model);
    let mut rng = StdRng::seed_from_u64(42);
    let reads = sequencer.sequence(&genome, 500.0, &mut rng);
    assert!(!reads.is_empty(), "Sequencing should produce reads");

    // Step 2: Assemble reads into contigs using HDC overlap detection
    let assembler = HdcAssembler::default();
    let contigs = assembler.assemble(&reads);
    assert!(!contigs.is_empty(), "Assembly should produce contigs");

    // Step 3: Assess assembly quality
    let quality = QualityAssessor::assess_assembly(&contigs);
    assert!(quality.total_length > 0, "Assembly should have non-zero total length");
    assert!(quality.num_contigs > 0, "Assembly should have at least one contig");
    assert!(quality.completeness > 0.0, "Completeness should be positive");

    // Step 4: Use assembly quality metrics to construct a cell state
    // A high-quality assembly implies we can proceed with cell manipulation
    let mut cell = CellState::new_somatic();
    // Modulate viability based on assembly completeness (simulating
    // the idea that better DNA quality -> more viable starting material)
    cell.viability = 0.8 + 0.2 * quality.completeness;
    assert!(cell.viability > 0.8, "Cell viability should reflect assembly quality");
    assert_eq!(cell.cell_type, CellType::Somatic);

    // Step 5: Verify the cell state can be HDC-encoded
    let cell_hv = encode_cell_state(&cell);
    assert_eq!(cell_hv.dim(), HDC_DIMENSION);
    assert!(cell_hv.norm() > 0.0, "Encoded cell state HV should be non-zero");
}

// =============================================================================
// TEST 2: Cell foundry produces oocyte via IVG protocol
// =============================================================================

#[test]
fn test_cell_foundry_produces_oocyte() {
    // Start with an iPSC cell (required for IVG)
    let mut ipsc = CellState::new_ipsc();
    ipsc.viability = 0.99; // Maximize chances of completing protocol

    // Create IVG protocol
    let mut protocol = IvgProtocol::new(ipsc).expect("IVG should accept iPSC");
    assert_eq!(protocol.stage, GermCellStage::Ipsc);

    let culture = CultureEnvironment::standard();

    // Advance through the full protocol (up to 40 days)
    for _ in 0..40 {
        protocol.advance_day(&culture);
        if protocol.is_mature_gamete() || protocol.is_failed() {
            break;
        }
    }

    // The protocol should have terminated (either success or checkpoint failure)
    assert!(
        protocol.is_mature_gamete() || protocol.is_failed(),
        "Protocol should terminate within 40 days: stage={:?}",
        protocol.stage
    );

    if protocol.is_mature_gamete() {
        // Verify oocyte quality: at least some checkpoints passed
        assert!(
            protocol.checkpoints_passed_count() >= 2,
            "Mature gamete should have passed checkpoints: {}",
            protocol.checkpoints_passed_count()
        );
    }
}

// =============================================================================
// TEST 3: Ectogenesis fetal monitoring with normative trajectory
// =============================================================================

#[test]
fn test_ectogenesis_fetal_monitoring() {
    let mut monitor = FetalMonitor::new();

    // Record normative metrics from week 10 to week 30
    for week in (10..=30).step_by(2) {
        let metrics = FetalMetrics::normative(GestationalWeek::new(week));
        monitor.record(metrics);
    }

    assert_eq!(monitor.history.len(), 11, "Should have recorded 11 measurements");

    // Growth percentile for normative data should be near 50th percentile
    let percentile = monitor.growth_percentile();
    assert!(
        (percentile - 0.5).abs() < 0.2,
        "Normative growth should be near 50th percentile, got {percentile}"
    );

    // Heart rate should be normal for normative data
    assert!(
        monitor.heart_rate_normal(),
        "Normative heart rate should be within normal range"
    );

    // Anomaly score should be low for normative data
    let anomaly = monitor.anomaly_score();
    assert!(
        anomaly < 0.2,
        "Normative data should have low anomaly score, got {anomaly}"
    );

    // Growth velocity should be positive (fetus is growing)
    let velocity = monitor.growth_velocity(5);
    assert!(
        velocity > 0.0,
        "Growth velocity should be positive, got {velocity}"
    );
}

// =============================================================================
// TEST 4: Consent proxy escalation at gestational boundaries
// =============================================================================

#[test]
fn test_ectogenesis_consent_proxy_escalation() {
    // Week 4 boundary: should be PreSentient (0-4)
    let proxy_w4 = ConsentProxy::from_week(GestationalWeek::new(4));
    assert_eq!(proxy_w4, ConsentProxy::PreSentient);
    assert!((proxy_w4.patient_weight() - 0.2).abs() < f64::EPSILON);

    // Week 5: escalates to EmergingSentience
    let proxy_w5 = ConsentProxy::from_week(GestationalWeek::new(5));
    assert_eq!(proxy_w5, ConsentProxy::EmergingSentience);
    assert!((proxy_w5.patient_weight() - 0.6).abs() < f64::EPSILON);

    // Week 23: still EmergingSentience
    let proxy_w23 = ConsentProxy::from_week(GestationalWeek::new(23));
    assert_eq!(proxy_w23, ConsentProxy::EmergingSentience);

    // Week 24: escalates to Sentient
    let proxy_w24 = ConsentProxy::from_week(GestationalWeek::new(24));
    assert_eq!(proxy_w24, ConsentProxy::Sentient);
    assert!((proxy_w24.patient_weight() - 1.0).abs() < f64::EPSILON);

    // Verify the ethics gate enforces escalation
    let gate = EctogenesisEthicsGate::fully_approved();

    // PreSentient: low bar (0.3)
    assert!(gate.check_intervention(GestationalWeek::new(4), "test", 0.4).is_ok());
    assert!(gate.check_intervention(GestationalWeek::new(4), "test", 0.1).is_err());

    // Sentient: high bar (0.7) + all approvals required
    assert!(gate.check_intervention(GestationalWeek::new(30), "test", 0.8).is_ok());
    assert!(gate.check_intervention(GestationalWeek::new(30), "test", 0.5).is_err());

    // Unapproved gate fails for sentient even with high benefit
    let unapproved = EctogenesisEthicsGate::unapproved();
    assert!(unapproved.check_intervention(GestationalWeek::new(30), "test", 0.9).is_err());
}

// =============================================================================
// TEST 5: Nurture attachment formation via reliable caregiver
// =============================================================================

#[test]
fn test_nurture_attachment_formation() {
    let mut system = AttachmentSystem::new();
    assert_eq!(system.style, AttachmentStyle::Forming);

    // Advance past the forming period (>6 months)
    system.advance_age(8.0);

    // Simulate 500 reliable caregiver interactions
    let reliable = CaregiverAction::reliable();
    for _ in 0..500 {
        system.process_interaction(&reliable);
    }

    // After 500 reliable interactions past the forming period, attachment should be secure
    assert_eq!(
        system.style,
        AttachmentStyle::Secure,
        "500 reliable interactions should produce secure attachment, got {:?}",
        system.style
    );

    // Security score should be high
    assert!(
        system.security_score > 0.7,
        "Security score should be high with reliable care, got {}",
        system.security_score
    );

    // Internal working model should reflect positive beliefs
    assert!(
        system.internal_working_model.self_worthy > 0.7,
        "Self-worthy should be high, got {}",
        system.internal_working_model.self_worthy
    );
    assert!(
        system.internal_working_model.other_reliable > 0.7,
        "Other-reliable should be high, got {}",
        system.internal_working_model.other_reliable
    );
}

// =============================================================================
// TEST 6: Nurture neuromodulation wiring to NeuromodulatorBath
// =============================================================================

#[test]
fn test_nurture_neuromod_wiring() {
    let mut bath = NeuromodulatorBath::default();
    let initial_oxy = bath.oxytocin.level;
    let initial_ne = bath.noradrenaline.level;
    let initial_5ht = bath.serotonin.level;
    let initial_ade = bath.adenosine.level;

    // Apply separation neuromodulation signature
    let separation = AttachmentNeuromodulation::separation();
    separation.apply_to_bath(&mut bath);

    // Separation should increase noradrenaline and adenosine
    assert!(
        bath.noradrenaline.level > initial_ne,
        "NE should rise on separation: before={initial_ne}, after={}",
        bath.noradrenaline.level
    );
    assert!(
        bath.adenosine.level > initial_ade,
        "Adenosine should rise on separation: before={initial_ade}, after={}",
        bath.adenosine.level
    );
    // Serotonin should dip
    assert!(
        bath.serotonin.level < initial_5ht,
        "5-HT should dip on separation: before={initial_5ht}, after={}",
        bath.serotonin.level
    );

    // Now apply reunion neuromodulation
    let elevated_ne = bath.noradrenaline.level;
    let reunion = AttachmentNeuromodulation::reunion();
    reunion.apply_to_bath(&mut bath);

    // Reunion should produce oxytocin burst and reduce NE
    assert!(
        bath.oxytocin.level > initial_oxy,
        "Oxytocin should surge on reunion: before={initial_oxy}, after={}",
        bath.oxytocin.level
    );
    assert!(
        bath.noradrenaline.level < elevated_ne,
        "NE should decrease on reunion: elevated={elevated_ne}, after={}",
        bath.noradrenaline.level
    );
    assert!(
        bath.dopamine.level > 0.5,
        "DA should increase on reunion, got {}",
        bath.dopamine.level
    );
}

// =============================================================================
// TEST 7: Population breeding and simulation
// =============================================================================

#[test]
fn test_population_breeding_and_simulation() {
    let mut rng = StdRng::seed_from_u64(42);

    // Create 5 loci with HDC-encoded identifiers
    let loci: Vec<Locus> = (0..5)
        .map(|i| Locus {
            id: i as u64,
            name: format!("locus_{i}"),
            chromosome: 1,
            position: i as u64 * 1000,
            hv: encode_locus(i as u64),
        })
        .collect();

    // Create founder population: 10 females + 10 males with heterozygous genotypes
    let mut individuals = Vec::new();
    let mut pedigree = Pedigree::new();
    for i in 0..20 {
        let sex = if i < 10 {
            BiologicalSex::Female
        } else {
            BiologicalSex::Male
        };
        let genotypes: Vec<Genotype> = loci
            .iter()
            .map(|l| {
                // Founders have diverse alleles: each individual has unique allele IDs
                let a_id = (i * 10 + l.id) as u64;
                let b_id = (i * 10 + l.id + 100) as u64;
                Genotype {
                    locus_id: l.id,
                    allele_a: Allele::neutral(a_id),
                    allele_b: Allele::neutral(b_id),
                }
            })
            .collect();
        let genome_hv = encode_individual(&loci, &genotypes);
        let id = i as u64;
        individuals.push(Individual {
            id,
            sex,
            genotypes,
            genome_hv,
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        });
        pedigree.add_entry(PedigreeEntry {
            individual_id: id,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
    }

    let population = Population {
        individuals,
        generation: 0,
        founding_size: 20,
        diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
    };

    // Measure initial heterozygosity
    let initial_ho: f64 = (0..5)
        .map(|locus_idx| {
            let locus_genotypes: Vec<Genotype> = population
                .individuals
                .iter()
                .filter_map(|ind| ind.genotypes.get(locus_idx).cloned())
                .collect();
            observed_heterozygosity(&locus_genotypes)
        })
        .sum::<f64>()
        / 5.0;

    // All founders are heterozygous by construction
    assert!(
        initial_ho > 0.9,
        "Initial Ho should be near 1.0 (all het founders), got {initial_ho}"
    );

    // Simulate 10 generations with MinimumKinship strategy
    let simulator = PopulationSimulator::new(BreedingStrategy::MinimumKinship, 20);
    let result = simulator.run(&population, &mut pedigree, &loci, 10, &mut rng);

    assert_eq!(
        result.generations.len(),
        11,
        "Should have initial + 10 generation snapshots"
    );

    // Check that heterozygosity has not increased dramatically
    // (it should decay or stay roughly stable with minimum kinship)
    let final_ho = result.generations.last().unwrap().heterozygosity;

    // Theoretical decay without selection: H(t) = H(0) * (1 - 1/(2*Ne))^t
    // With Ne ~ 20 and 10 generations: ~0.78 * H(0)
    let theoretical_h10 = heterozygosity_after_generations(initial_ho, 20.0, 10);
    assert!(
        theoretical_h10 < initial_ho,
        "Theoretical heterozygosity should decay: initial={initial_ho}, theoretical={theoretical_h10}"
    );
    assert!(
        final_ho <= initial_ho + 0.15,
        "Heterozygosity should not increase substantially: initial={initial_ho}, final={final_ho}"
    );

    // Final population should exist
    assert!(
        result.final_population.size() > 0,
        "Final population should have individuals"
    );
}

// =============================================================================
// TEST 8: HDC vectors flow across the entire pipeline
// =============================================================================

#[test]
fn test_hdc_vectors_flow_across_pipeline() {
    // Genomics: contig HVs from assembly
    let genome = MockSequencer::generate_genome(500, 99);
    let damage_model = DamageModel::new(15.0);
    let sequencer = MockSequencer::new(100, 0.01, 10, damage_model);
    let mut rng = StdRng::seed_from_u64(99);
    let reads = sequencer.sequence(&genome, 100.0, &mut rng);
    let assembler = HdcAssembler::default();
    let contigs = assembler.assemble(&reads);
    assert!(!contigs.is_empty());
    for contig in &contigs {
        assert_eq!(contig.hv.dim(), HDC_DIMENSION, "Contig HV should be 16,384D");
        assert!(contig.hv.norm() > 0.0, "Contig HV should be non-zero");
    }

    // Cell Foundry: cell state HVs
    let somatic = CellState::new_somatic();
    let somatic_hv = encode_cell_state(&somatic);
    assert_eq!(somatic_hv.dim(), HDC_DIMENSION, "Cell state HV should be 16,384D");
    assert!(somatic_hv.norm() > 0.0, "Cell state HV should be non-zero");

    let ipsc = CellState::new_ipsc();
    let ipsc_hv = encode_cell_state(&ipsc);
    assert_eq!(ipsc_hv.dim(), HDC_DIMENSION);
    // Different cell types should produce different HVs
    let cell_sim = somatic_hv.similarity(&ipsc_hv);
    assert!(
        cell_sim < 0.9,
        "Somatic vs iPSC should differ: similarity={cell_sim}"
    );

    // Ectogenesis: fetal state HVs
    let fetal_hv_w20 = encode_fetal_state(&FetalMetrics::normative(GestationalWeek::new(20)));
    let fetal_hv_w30 = encode_fetal_state(&FetalMetrics::normative(GestationalWeek::new(30)));
    assert_eq!(fetal_hv_w20.dim(), HDC_DIMENSION, "Fetal HV should be 16,384D");
    assert_eq!(fetal_hv_w30.dim(), HDC_DIMENSION);
    assert!(fetal_hv_w20.norm() > 0.0, "Fetal HV should be non-zero");
    assert!(fetal_hv_w30.norm() > 0.0);
    // Different weeks should produce different encodings
    let fetal_sim = fetal_hv_w20.similarity(&fetal_hv_w30);
    assert!(
        fetal_sim < 0.9,
        "Week 20 vs 30 fetal state should differ: similarity={fetal_sim}"
    );

    // Nurture: internal working model HVs
    let attachment = AttachmentSystem::new();
    let initial_attachment_hv = attachment.attachment_hv.clone();
    assert_eq!(
        initial_attachment_hv.dim(),
        HDC_DIMENSION,
        "Attachment HV should be 16,384D"
    );
    assert!(initial_attachment_hv.norm() > 0.0, "Attachment HV should be non-zero");
    // Working model HV should also be valid
    assert_eq!(
        attachment.internal_working_model.model_hv.dim(),
        HDC_DIMENSION,
        "IWM HV should be 16,384D"
    );

    // Population: genome HVs
    let loci = vec![
        Locus {
            id: 0,
            name: "locus_0".into(),
            chromosome: 1,
            position: 0,
            hv: encode_locus(0),
        },
        Locus {
            id: 1,
            name: "locus_1".into(),
            chromosome: 1,
            position: 1000,
            hv: encode_locus(1),
        },
    ];
    let genotypes = vec![
        Genotype {
            locus_id: 0,
            allele_a: Allele::neutral(10),
            allele_b: Allele::neutral(20),
        },
        Genotype {
            locus_id: 1,
            allele_a: Allele::neutral(30),
            allele_b: Allele::neutral(40),
        },
    ];
    let genome_hv = encode_individual(&loci, &genotypes);
    assert_eq!(genome_hv.dim(), HDC_DIMENSION, "Genome HV should be 16,384D");
    assert!(genome_hv.norm() > 0.0, "Genome HV should be non-zero");

    // All HVs share the same dimensionality, proving pipeline-wide compatibility
    assert_eq!(contigs[0].hv.dim(), somatic_hv.dim());
    assert_eq!(somatic_hv.dim(), fetal_hv_w20.dim());
    assert_eq!(fetal_hv_w20.dim(), initial_attachment_hv.dim());
    assert_eq!(initial_attachment_hv.dim(), genome_hv.dim());
}

// =============================================================================
// TEST 9: Full pipeline type compatibility (compile-time proof)
// =============================================================================

#[test]
fn test_full_pipeline_type_compatibility() {
    // This test verifies that types from one crate can construct inputs
    // for the next crate in the pipeline. If this compiles, the API is compatible.

    // 1. Genomics -> Cell Foundry:
    //    Assembly quality informs initial cell state viability
    let quality = symthaea_genomics::GenomeQuality {
        n50: 500,
        total_length: 5000,
        num_contigs: 3,
        mean_coverage: 15.0,
        completeness: 0.95,
        error_rate: 0.02,
    };
    let mut cell = CellState::new_somatic();
    cell.viability = 0.8 + 0.2 * quality.completeness;
    assert!(cell.viability > 0.9);

    // 2. Cell Foundry -> Ectogenesis:
    //    A mature oocyte becomes the starting point for ectogenesis
    let oocyte = CellState::new_oocyte();
    assert_eq!(oocyte.cell_type, CellType::Oocyte);
    // The fact that we can reference both CellType and GestationalWeek means
    // the type systems are compatible across crate boundaries
    let week_start = GestationalWeek::new(0);
    let fetal_metrics = FetalMetrics::normative(week_start);
    assert!(fetal_metrics.weight_grams >= 0.0);

    // 3. Ectogenesis -> Nurture:
    //    A full-term fetus transitions to the nurture phase
    let term_week = GestationalWeek::new(40);
    assert!(term_week.is_full_term());
    // The developmental age in nurture starts at 0 months (birth)
    let attachment = AttachmentSystem::new();
    assert_eq!(attachment.style, AttachmentStyle::Forming);

    // 4. Nurture -> Population:
    //    An individual from nurture becomes a member of the population
    let individual = Individual {
        id: 0,
        sex: BiologicalSex::Female,
        genotypes: vec![],
        genome_hv: ContinuousHV::random(HDC_DIMENSION, 42),
        generation: 0,
        parent_ids: (None, None),
        fitness: 1.0,
    };
    let pop = Population {
        individuals: vec![individual],
        generation: 0,
        founding_size: 1,
        diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
    };
    assert_eq!(pop.size(), 1);
    assert_eq!(pop.count_females(), 1);

    // 5. Population governance references breeding strategy types
    let decision = PopulationDecision::ModifyStrategy(BreedingStrategy::MinimumKinship);
    assert_eq!(decision.required_tier(), GovernanceTier::Constitutional);
    let desc = decision.description();
    assert!(!desc.is_empty(), "Decision should have a description");
}

// =============================================================================
// TEST 10: Ethics gates present at each pipeline stage
// =============================================================================

#[test]
fn test_ethics_gates_present_at_each_stage() {
    // Cell Foundry ethics gate: consent + IRB
    let mut cell_gate = EthicsGate::new();
    assert!(
        cell_gate.check_manipulation("test").is_err(),
        "Unapproved cell gate should fail"
    );
    cell_gate.grant_consent();
    cell_gate.grant_irb();
    assert!(
        cell_gate.check_manipulation("test").is_ok(),
        "Approved cell gate should pass"
    );
    // IVG requires additional gamete-creation approval
    assert!(
        cell_gate.check_ivg_protocol().is_err(),
        "IVG should require gamete_creation approval"
    );
    cell_gate.add_approval("gamete_creation");
    assert!(
        cell_gate.check_ivg_protocol().is_ok(),
        "IVG should pass with gamete_creation approval"
    );

    // Ectogenesis ethics gate: graduated consent proxy
    let ecto_gate = EctogenesisEthicsGate::fully_approved();
    // PreSentient: benefit >= 0.3
    assert!(ecto_gate
        .check_intervention(GestationalWeek::new(3), "nutrient", 0.4)
        .is_ok());
    // EmergingSentience: guardian + benefit >= 0.5
    assert!(ecto_gate
        .check_intervention(GestationalWeek::new(15), "hormone", 0.6)
        .is_ok());
    // Sentient: guardian + ethics board + benefit >= 0.7
    assert!(ecto_gate
        .check_intervention(GestationalWeek::new(30), "steroid", 0.8)
        .is_ok());
    // Minimum benefit increases with developmental stage
    let min_pre = ecto_gate.minimum_benefit(GestationalWeek::new(3));
    let min_em = ecto_gate.minimum_benefit(GestationalWeek::new(15));
    let min_sent = ecto_gate.minimum_benefit(GestationalWeek::new(30));
    assert!(min_pre < min_em, "Minimum benefit should increase: {min_pre} < {min_em}");
    assert!(min_em < min_sent, "Minimum benefit should increase: {min_em} < {min_sent}");

    // Population ethics: governance tiers with escalating thresholds
    let pairing_decision = PopulationDecision::ApprovePairing(MatingPair {
        parent_a: 0,
        parent_b: 1,
        kinship: 0.05,
        hla_complementarity: 0.8,
    });
    assert_eq!(
        pairing_decision.required_tier(),
        GovernanceTier::Citizen,
        "Pairing should be Citizen-tier"
    );

    let strategy_decision = PopulationDecision::ModifyStrategy(BreedingStrategy::HdcDistanceMaximize);
    assert_eq!(
        strategy_decision.required_tier(),
        GovernanceTier::Constitutional,
        "Strategy change should be Constitutional-tier"
    );

    let override_decision =
        PopulationDecision::OverrideAiRecommendation("safety concern".to_string());
    assert_eq!(
        override_decision.required_tier(),
        GovernanceTier::Override,
        "AI override should be Override-tier"
    );

    // Verify thresholds are monotonically increasing
    let tiers = GovernanceTier::all();
    for i in 1..tiers.len() {
        assert!(
            tiers[i].threshold() > tiers[i - 1].threshold(),
            "Governance tiers should have ascending thresholds"
        );
    }

    // Population ethical tension encoding produces valid HDC vectors
    let tension = EthicalTension::new("test_tension", 0.4, 0.3, 0.2, 0.1);
    assert_eq!(tension.tension_hv.dim(), HDC_DIMENSION);
    assert!(tension.tension_hv.norm() > 0.5, "Ethical tension HV should be normalized");
    assert_eq!(tension.dominant_value(), "autonomy"); // 0.4 = unambiguous max
}
