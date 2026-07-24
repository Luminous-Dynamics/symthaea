// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate CSV data files for the Psych-Bench BRM paper.
//!
//! Runs the full benchmark suite and produces 8 CSV files consumed
//! by `pgfplotstableread` in `papers/latex/psych_bench_paper.tex`.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example psych_bench_paper_data
//!
//! Output directory: papers/data/psych_bench/

use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_psych_bench::benchmarks::affect::{
    EmotionalStroopBenchmark, MoodCongruentRecallBenchmark, ValenceClassificationBenchmark,
};
use symthaea_psych_bench::benchmarks::attention::{
    AttentionalBlinkBenchmark, MismatchNegativityBenchmark, VisualSearchBenchmark,
};
use symthaea_psych_bench::benchmarks::binding::{
    CrossModalBindingBenchmark, FeatureConjunctionBenchmark, TemporalOrderBenchmark,
};
use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::causal_reasoning;
use symthaea_psych_bench::benchmarks::clinical::{
    AllianceMaintenanceBenchmark, CognitiveDistortionBenchmark, CrisisDetectionBenchmark,
    EmpathicAccuracyBenchmark, MotivationalInterviewingBenchmark, TherapeuticResponseBenchmark,
};
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
    TemporalDiscountingBenchmark, TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::consciousness::{
    BinocularRivalryBenchmark, BlindSightBenchmark, PerceptualCrowdingBenchmark,
};
use symthaea_psych_bench::benchmarks::creativity::{
    AlternateUsesBenchmark, ConceptualBlendingBenchmark, DivergentThinkingBenchmark,
    RemoteAssociatesBenchmark,
};
use symthaea_psych_bench::benchmarks::executive::{
    DualTaskBenchmark, FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark,
    StroopBenchmark, TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::{
    FlankerInhibitionBenchmark, GoNoGoBenchmark, StopSignalBenchmark,
};
use symthaea_psych_bench::benchmarks::institutional_reasoning;
use symthaea_psych_bench::benchmarks::language::{
    GardenPathBenchmark, LexicalDecisionBenchmark, SemanticCoherenceBenchmark,
    SemanticPrimingBenchmark,
};
use symthaea_psych_bench::benchmarks::mathematics::{
    ArithmeticWordProblemsBenchmark, BayesianReasoningBenchmark, ConstraintPuzzlesBenchmark,
    DefiniteIntegralsBenchmark, LinearSystemSolvingBenchmark, LogicalDeductionBenchmark,
    MatrixOperationsBenchmark, PolynomialRootsBenchmark, ProofConstructionBenchmark,
    StatisticalInferenceBenchmark,
};
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    ProspectiveMemoryBenchmark, TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::{
    ChangeBlindnessBenchmark, FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark,
};
use symthaea_psych_bench::benchmarks::motor::{
    BimanualBenchmark, FittsLawBenchmark, ProprioceptiveDriftBenchmark, SrttBenchmark,
};
use symthaea_psych_bench::benchmarks::neuromod::{
    AllostaticStressBenchmark, AntagonistProfilesBenchmark, AttentionNetworkBenchmark,
    BehavioralKnockoutBenchmark, ConsciousnessFeedbackBenchmark,
    ConsciousnessPharmacologyBenchmark, DoseResponseBenchmark, InjectionChallengeBenchmark,
    MoodInductionBenchmark, MoralOxytocinBenchmark, MultiTransmitterSynergyBenchmark,
    PharmacologicalAblationBenchmark, PharmacologicalChallengeBenchmark, RewardLearningBenchmark,
    ToleranceWithdrawalBenchmark, YerkesDodsonBenchmark,
};
use symthaea_psych_bench::benchmarks::reasoning::{
    ArcAbductiveBenchmark, ArcAlgebraBenchmark, ArcAnalogyBenchmark, ArcChainBenchmark,
    ArcCompositionalBenchmark, ArcFewShotBenchmark, ArcFluidBenchmark, ArcNoiseBenchmark,
    ArcRsaBenchmark, ArcScalingBenchmark, ArcStaircaseBenchmark,
};
use symthaea_psych_bench::benchmarks::security::{
    CollectiveAggregationBenchmark, CrossMaskPrivacyBenchmark, EncryptedBindingBenchmark,
    EncryptedClassificationBenchmark, EncryptedLearningBenchmark, ScalingAnalysisBenchmark,
};
use symthaea_psych_bench::benchmarks::social::{
    DictatorGameBenchmark, MachiavelliBenchmark, PrisonersDilemmaBenchmark, PublicGoodsBenchmark,
    RmeBenchmark, SocialNormBenchmark, UltimatumGameBenchmark,
};
use symthaea_psych_bench::benchmarks::spatial::{
    LandmarkBindingBenchmark, MentalRotationBenchmark, PerspectiveTakingBenchmark,
    SpatialPathUpdatingBenchmark,
};
use symthaea_psych_bench::benchmarks::speech::{
    CategoricalPerceptionBenchmark, PhonemeDiscriminationBenchmark, VotContinuumBenchmark,
};
use symthaea_psych_bench::benchmarks::substrate::{
    SubstrateDegradationBenchmark, SubstrateLatencyBenchmark, SubstrateTransferBenchmark,
};
use symthaea_psych_bench::benchmarks::sustained_attention::{
    CptBenchmark, PvtBenchmark, SartBenchmark,
};
use symthaea_psych_bench::benchmarks::tombench::{
    FalseBeliefBenchmark, FauxPasBenchmark, HintingBenchmark, PersuasionBenchmark,
    StrangeStoryBenchmark,
};
use symthaea_psych_bench::benchmarks::worm::{
    BindingBenchmark, ChangeDetectionBenchmark, DigitSpanBenchmark, NBackBenchmark,
    SerialRecallBenchmark, SpatialUpdatingBenchmark,
};
use symthaea_psych_bench::harness::{
    AblationPreset, BenchmarkConfig, BenchmarkReport, CrossDomainMatrix, NormativeReport,
    PsychBenchmark, ReliabilityBattery, SatBattery,
};

/// All PsychBenchmark-trait benchmarks (excludes ConsciousnessFeedback and
/// MoralOxytocin which have standalone run() signatures).
fn all_benchmarks() -> Vec<Box<dyn PsychBenchmark + Send + Sync>> {
    vec![
        // WorM (6)
        Box::new(NBackBenchmark),
        Box::new(ChangeDetectionBenchmark),
        Box::new(SerialRecallBenchmark),
        Box::new(SpatialUpdatingBenchmark),
        Box::new(BindingBenchmark),
        Box::new(DigitSpanBenchmark),
        // CogBench (8)
        Box::new(ProbabilisticReasoningBenchmark),
        Box::new(HorizonBenchmark),
        Box::new(RestlessBanditBenchmark),
        Box::new(InstrumentalLearningBenchmark),
        Box::new(TwoStepBenchmark),
        Box::new(TemporalDiscountingBenchmark),
        Box::new(BartBenchmark),
        Box::new(ReversalLearningBenchmark),
        // Executive (7)
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(IowaGamblingBenchmark),
        Box::new(RavensProgressiveMatricesBenchmark),
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(TowerOfLondonBenchmark),
        Box::new(DualTaskBenchmark),
        // Metacognition (3)
        Box::new(MetacognitiveCalibrationBenchmark),
        Box::new(FeelingOfKnowingBenchmark),
        Box::new(ChangeBlindnessBenchmark),
        // Butlin (1)
        Box::new(ButlinIndicatorSuite),
        // ToMBench (5)
        Box::new(FalseBeliefBenchmark),
        Box::new(FauxPasBenchmark),
        Box::new(PersuasionBenchmark),
        Box::new(StrangeStoryBenchmark),
        Box::new(HintingBenchmark),
        // MemoryAgent (5)
        Box::new(AccurateRetrievalBenchmark),
        Box::new(TestTimeLearningBenchmark),
        Box::new(LongRangeBenchmark),
        Box::new(ConflictResolutionBenchmark),
        Box::new(ProspectiveMemoryBenchmark),
        // Affect (3)
        Box::new(ValenceClassificationBenchmark),
        Box::new(MoodCongruentRecallBenchmark),
        Box::new(EmotionalStroopBenchmark),
        // Creativity (4)
        Box::new(RemoteAssociatesBenchmark),
        Box::new(AlternateUsesBenchmark),
        Box::new(DivergentThinkingBenchmark),
        Box::new(ConceptualBlendingBenchmark),
        // Inhibition (3)
        Box::new(GoNoGoBenchmark),
        Box::new(StopSignalBenchmark),
        Box::new(FlankerInhibitionBenchmark),
        // Attention (3)
        Box::new(AttentionalBlinkBenchmark),
        Box::new(VisualSearchBenchmark),
        Box::new(MismatchNegativityBenchmark),
        // Reasoning (11)
        Box::new(ArcFluidBenchmark),
        Box::new(ArcCompositionalBenchmark),
        Box::new(ArcAnalogyBenchmark),
        Box::new(ArcAbductiveBenchmark),
        Box::new(ArcChainBenchmark),
        Box::new(ArcNoiseBenchmark),
        Box::new(ArcFewShotBenchmark),
        Box::new(ArcScalingBenchmark),
        Box::new(ArcRsaBenchmark),
        Box::new(ArcAlgebraBenchmark),
        Box::new(ArcStaircaseBenchmark),
        // Sustained Attention (3)
        Box::new(SartBenchmark),
        Box::new(PvtBenchmark),
        Box::new(CptBenchmark),
        // Motor (4)
        Box::new(SrttBenchmark),
        Box::new(FittsLawBenchmark),
        Box::new(BimanualBenchmark),
        Box::new(ProprioceptiveDriftBenchmark),
        // Language (4)
        Box::new(GardenPathBenchmark),
        Box::new(SemanticCoherenceBenchmark),
        Box::new(LexicalDecisionBenchmark),
        Box::new(SemanticPrimingBenchmark),
        // Social (7)
        Box::new(RmeBenchmark),
        Box::new(UltimatumGameBenchmark),
        Box::new(SocialNormBenchmark),
        Box::new(PrisonersDilemmaBenchmark),
        Box::new(PublicGoodsBenchmark),
        Box::new(DictatorGameBenchmark),
        Box::new(MachiavelliBenchmark),
        // Binding (3)
        Box::new(TemporalOrderBenchmark),
        Box::new(CrossModalBindingBenchmark),
        Box::new(FeatureConjunctionBenchmark),
        // Spatial (4)
        Box::new(MentalRotationBenchmark),
        Box::new(SpatialPathUpdatingBenchmark),
        Box::new(LandmarkBindingBenchmark),
        Box::new(PerspectiveTakingBenchmark),
        // Causal Reasoning (3)
        Box::new(causal_reasoning::CausalChainBenchmark),
        Box::new(causal_reasoning::ConfoundDetectionBenchmark),
        Box::new(causal_reasoning::InterventionEffectBenchmark),
        // Speech (3)
        Box::new(PhonemeDiscriminationBenchmark),
        Box::new(VotContinuumBenchmark),
        Box::new(CategoricalPerceptionBenchmark),
        // Consciousness (3)
        Box::new(BlindSightBenchmark),
        Box::new(BinocularRivalryBenchmark),
        Box::new(PerceptualCrowdingBenchmark),
        // Substrate (3)
        Box::new(SubstrateTransferBenchmark),
        Box::new(SubstrateDegradationBenchmark),
        Box::new(SubstrateLatencyBenchmark),
        // Clinical/Therapeutic (6)
        Box::new(EmpathicAccuracyBenchmark),
        Box::new(TherapeuticResponseBenchmark),
        Box::new(AllianceMaintenanceBenchmark),
        Box::new(CrisisDetectionBenchmark),
        Box::new(CognitiveDistortionBenchmark),
        Box::new(MotivationalInterviewingBenchmark),
        // Institutional Reasoning (7)
        Box::new(institutional_reasoning::InstitutionalReasoningBenchmark),
        Box::new(institutional_reasoning::AnalogicalReasoningBenchmark),
        Box::new(institutional_reasoning::CausalChainBenchmark),
        Box::new(institutional_reasoning::CounterfactualReasoningBenchmark),
        Box::new(institutional_reasoning::WeightedDecompositionBenchmark),
        Box::new(institutional_reasoning::InstitutionalStabilityBenchmark),
        Box::new(institutional_reasoning::InstitutionalIsomorphismBenchmark),
        // Mathematics (10)
        Box::new(ArithmeticWordProblemsBenchmark),
        Box::new(LinearSystemSolvingBenchmark),
        Box::new(PolynomialRootsBenchmark),
        Box::new(MatrixOperationsBenchmark),
        Box::new(StatisticalInferenceBenchmark),
        Box::new(BayesianReasoningBenchmark),
        Box::new(LogicalDeductionBenchmark),
        Box::new(ConstraintPuzzlesBenchmark),
        Box::new(ProofConstructionBenchmark),
        Box::new(DefiniteIntegralsBenchmark),
        // Insecure shared-mask algebra demo (6)
        Box::new(EncryptedClassificationBenchmark),
        Box::new(CollectiveAggregationBenchmark),
        Box::new(EncryptedLearningBenchmark),
        Box::new(CrossMaskPrivacyBenchmark),
        Box::new(EncryptedBindingBenchmark),
        Box::new(ScalingAnalysisBenchmark),
        // Neuromod (14 — trait-based)
        Box::new(AttentionNetworkBenchmark),
        Box::new(MoodInductionBenchmark),
        Box::new(RewardLearningBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(PharmacologicalChallengeBenchmark),
        Box::new(PharmacologicalAblationBenchmark),
        Box::new(InjectionChallengeBenchmark),
        Box::new(BehavioralKnockoutBenchmark),
        Box::new(ConsciousnessPharmacologyBenchmark),
        Box::new(AllostaticStressBenchmark),
        Box::new(DoseResponseBenchmark),
        Box::new(AntagonistProfilesBenchmark),
        Box::new(ToleranceWithdrawalBenchmark),
        // MultiTransmitterSynergy implements PsychBenchmark
        Box::new(MultiTransmitterSynergyBenchmark),
    ]
}

/// 8 representative benchmarks for SAT curves (one per profile domain).
fn sat_benchmarks() -> Vec<Box<dyn PsychBenchmark + Send + Sync>> {
    vec![
        Box::new(StroopBenchmark),
        Box::new(NBackBenchmark),
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(ArcFluidBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(VisualSearchBenchmark),
        Box::new(ReversalLearningBenchmark),
        Box::new(BartBenchmark),
    ]
}

/// ~20 representative benchmarks for reliability testing.
fn reliability_benchmarks() -> Vec<Box<dyn PsychBenchmark + Send + Sync>> {
    vec![
        Box::new(StroopBenchmark),
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(IowaGamblingBenchmark),
        Box::new(NBackBenchmark),
        Box::new(ChangeDetectionBenchmark),
        Box::new(DigitSpanBenchmark),
        Box::new(VisualSearchBenchmark),
        Box::new(FalseBeliefBenchmark),
        Box::new(SrttBenchmark),
        Box::new(LexicalDecisionBenchmark),
        Box::new(ArcFluidBenchmark),
        Box::new(TwoStepBenchmark),
        Box::new(BartBenchmark),
        Box::new(ReversalLearningBenchmark),
        Box::new(StopSignalBenchmark),
        Box::new(MetacognitiveCalibrationBenchmark),
        Box::new(FeelingOfKnowingBenchmark),
        Box::new(YerkesDodsonBenchmark),
        Box::new(DoseResponseBenchmark),
        Box::new(RewardLearningBenchmark),
    ]
}

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("papers")
        .join("data")
        .join("psych_bench");
    fs::create_dir_all(&dir).expect("create output dir");
    dir
}

fn main() {
    let total_start = Instant::now();
    let out = output_dir();
    eprintln!("Output directory: {}", out.display());

    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 20,
        seed: 42,
        trial_trace: true,
        ..Default::default()
    };

    // ── 1. Run full battery (parallel) ──────────────────────────────
    eprintln!("\n[1/7] Running full benchmark battery...");
    let start = Instant::now();
    let benchmarks = all_benchmarks();
    let results: Vec<_> = benchmarks
        .par_iter()
        .map(|b| {
            let r = b.run(&config);
            eprintln!("  {} ... {}ms", r.benchmark, r.elapsed_ms);
            r
        })
        .collect();

    let mut report = BenchmarkReport::new();
    for r in results {
        report.add(r);
    }

    // Also run the two standalone-signature neuromod benchmarks.
    let cf_result = ConsciousnessFeedbackBenchmark.run();
    eprintln!("  ConsciousnessFeedback ... {}ms", cf_result.elapsed_ms);
    report.add(cf_result);

    let mo_result = MoralOxytocinBenchmark.run();
    eprintln!("  MoralOxytocin ... {}ms", mo_result.elapsed_ms);
    report.add(mo_result);

    eprintln!("  Full battery: {:.1}s", start.elapsed().as_secs_f64());

    // ── CSV 1: cognitive_profile.csv (26 domains via composite z-scores) ─
    eprintln!("\n[2/7] Generating cognitive_profile.csv...");
    {
        let composites = report.composite_scores();
        let path = out.join("cognitive_profile.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "domain,score,n_benchmarks,interpretation").unwrap();
        for (domain, cs) in &composites {
            let interpretation = if cs.mean_z >= 2.0 {
                "Exceptional"
            } else if cs.mean_z >= 1.0 {
                "Strong"
            } else if cs.mean_z >= 0.0 {
                "Average"
            } else if cs.mean_z >= -1.0 {
                "Below average"
            } else {
                "Impaired"
            };
            writeln!(
                f,
                "{},{:.6},{},{}",
                domain, cs.mean_z, cs.n_benchmarks, interpretation
            )
            .unwrap();
        }
        eprintln!("  Wrote {} ({} domains)", path.display(), composites.len());
    }

    // ── CSV 2: normative_zscores.csv ────────────────────────────────
    eprintln!("\n[3/7] Generating normative_zscores.csv...");
    let normative = NormativeReport::from_report(&report);
    {
        let path = out.join("normative_zscores.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            "benchmark,metric,agent_value,human_mean,human_sd,z_score,z_clamped,percentile,interpretation"
        )
        .unwrap();
        for s in &normative.scores {
            // Clamp z-scores to [-3, 3] for forest plot; raw z preserved
            let z_clamped = s.z_score.clamp(-3.0, 3.0);
            writeln!(
                f,
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                s.benchmark,
                s.metric,
                s.agent_value,
                s.human_mean,
                s.human_sd,
                s.z_score,
                z_clamped,
                s.percentile,
                s.interpretation
            )
            .unwrap();
        }
        eprintln!(
            "  Wrote {} ({} scores, mean z={:.3})",
            path.display(),
            normative.scores.len(),
            normative.overall_mean_z
        );
    }

    // ── CSV 3: ablation_domains.csv ─────────────────────────────────
    eprintln!("\n[4/7] Generating ablation_domains.csv...");
    {
        use std::collections::BTreeMap;
        let presets = AblationPreset::all();
        let mut ablation_composites: Vec<(String, BTreeMap<String, f64>)> = Vec::new();

        for &preset in presets {
            let ac = preset.to_config(42);
            eprintln!("  Running ablation: {}...", ac.name);
            let ab_results: Vec<_> = benchmarks.par_iter().map(|b| b.run(&ac.base)).collect();
            let mut ab_report = BenchmarkReport::new();
            for r in ab_results {
                ab_report.add(r);
            }
            let composites = ab_report.composite_scores();
            let domain_scores: BTreeMap<String, f64> = composites
                .into_iter()
                .map(|(domain, cs)| (domain, cs.mean_z))
                .collect();
            ablation_composites.push((ac.name.clone(), domain_scores));
        }

        let path = out.join("ablation_domains.csv");
        let mut f = fs::File::create(&path).unwrap();
        // Wide format: domain, then one column per preset
        write!(f, "domain").unwrap();
        for (name, _) in &ablation_composites {
            let clean = name.replace(' ', "").replace("(K=3)", "");
            write!(f, ",{}", clean).unwrap();
        }
        writeln!(f).unwrap();

        // Collect all unique domain names (sorted via BTreeMap)
        let domain_names: Vec<String> = ablation_composites[0].1.keys().cloned().collect();

        for domain in &domain_names {
            write!(f, "{}", domain).unwrap();
            for (_name, scores) in &ablation_composites {
                let score = scores.get(domain).copied().unwrap_or(0.0);
                write!(f, ",{:.6}", score).unwrap();
            }
            writeln!(f).unwrap();
        }
        eprintln!(
            "  Wrote {} ({} presets x {} domains)",
            path.display(),
            presets.len(),
            domain_names.len()
        );
    }

    // ── CSV 4: sat_curves.csv + per-benchmark SAT files ────────────
    eprintln!("\n[5/7] Generating SAT curve CSVs...");
    {
        let sat_benches = sat_benchmarks();
        let sat_refs: Vec<&dyn PsychBenchmark> = sat_benches
            .iter()
            .map(|b| b.as_ref() as &dyn PsychBenchmark)
            .collect();
        let pressures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let sat_battery = SatBattery::run(&sat_refs, &config, &pressures);

        // Combined CSV
        let path = out.join("sat_curves.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            "benchmark,time_pressure,accuracy,mean_rt,fit_asymptote,fit_rate,fit_intercept,fit_r_squared,human_like"
        )
        .unwrap();
        for curve in &sat_battery.curves {
            for pt in &curve.points {
                writeln!(
                    f,
                    "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                    curve.benchmark,
                    pt.time_pressure,
                    pt.accuracy,
                    pt.mean_rt,
                    curve.fit.asymptote,
                    curve.fit.rate,
                    curve.fit.intercept,
                    curve.fit.r_squared,
                    if curve.human_like { 1 } else { 0 }
                )
                .unwrap();
            }
        }

        // Per-benchmark SAT files for pgfplotstableread (no string filtering needed)
        let name_map = [
            ("Executive::Stroop", "sat_stroop.csv"),
            ("Executive::Flanker", "sat_flanker.csv"),
            ("Attention::VisualSearch", "sat_visualsearch.csv"),
            ("WorM::N-back", "sat_nback.csv"),
            ("Executive::WCST", "sat_wcst.csv"),
            ("Reasoning::ArcFluid", "sat_arcfluid.csv"),
        ];
        for (bench_name, filename) in &name_map {
            if let Some(curve) = sat_battery
                .curves
                .iter()
                .find(|c| c.benchmark == *bench_name)
            {
                let p = out.join(filename);
                let mut bf = fs::File::create(&p).unwrap();
                writeln!(bf, "time_pressure,accuracy,mean_rt").unwrap();
                for pt in &curve.points {
                    writeln!(
                        bf,
                        "{:.6},{:.6},{:.6}",
                        pt.time_pressure, pt.accuracy, pt.mean_rt
                    )
                    .unwrap();
                }
                eprintln!("  Wrote {}", p.display());
            }
        }

        eprintln!(
            "  Wrote {} ({} curves × {} pressures)",
            path.display(),
            sat_battery.curves.len(),
            pressures.len()
        );
    }

    // ── CSV 5: reliability.csv ──────────────────────────────────────
    eprintln!("\n[6/7] Generating reliability.csv...");
    {
        let rel_benches = reliability_benchmarks();
        let rel_refs: Vec<&dyn PsychBenchmark> = rel_benches
            .iter()
            .map(|b| b.as_ref() as &dyn PsychBenchmark)
            .collect();
        let reliability = ReliabilityBattery::run(&rel_refs, &config, 5, 30);

        let path = out.join("reliability.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            "benchmark,metric,icc,pearson_r,sem,practice_direction,practice_change_pct,reliability_class"
        )
        .unwrap();
        for r in &reliability.results {
            writeln!(
                f,
                "{},{},{:.6},{:.6},{:.6},{},{:.6},{}",
                r.benchmark,
                r.metric,
                r.icc,
                r.pearson_r,
                r.sem,
                format!("{:?}", r.practice.direction),
                r.practice.change_pct,
                r.reliability_class.label()
            )
            .unwrap();
        }
        eprintln!(
            "  Wrote {} ({} benchmarks)",
            path.display(),
            reliability.results.len()
        );
    }

    // ── CSV 6: correlations.csv ─────────────────────────────────────
    eprintln!("\n[7/7] Generating correlations.csv...");
    {
        // Run at 10 seeds for cross-domain correlation
        let seeds: Vec<u64> = (0..10).map(|i| 42 + i * 7).collect();
        let seed_reports: Vec<BenchmarkReport> = seeds
            .par_iter()
            .map(|&seed| {
                let seed_config = BenchmarkConfig {
                    seed,
                    ..config.clone()
                };
                let mut seed_report = BenchmarkReport::new();
                for bench in &benchmarks {
                    seed_report.add(bench.run(&seed_config));
                }
                eprintln!("  Seed {} done", seed);
                seed_report
            })
            .collect();

        let cross_domain = CrossDomainMatrix::from_multi_seed_reports(&seed_reports);

        let path = out.join("correlations.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            "benchmark_a,metric_a,benchmark_b,metric_b,r,n,p_value,shared_mechanism"
        )
        .unwrap();
        for c in &cross_domain.correlations {
            writeln!(
                f,
                "{},{},{},{},{:.6},{},{:.6},{:?}",
                c.domain_a,
                c.metric_a,
                c.domain_b,
                c.metric_b,
                c.r,
                c.n,
                c.p_value,
                c.shared_mechanism
            )
            .unwrap();
        }
        eprintln!(
            "  Wrote {} ({} pairs)",
            path.display(),
            cross_domain.correlations.len()
        );
    }

    // ── CSV 7: neuromod_profiles.csv ────────────────────────────────
    eprintln!("\nGenerating neuromod_profiles.csv...");
    {
        // Run standalone neuromod benchmarks and collect key metrics
        let neuromod_benches: Vec<Box<dyn PsychBenchmark + Send + Sync>> = vec![
            Box::new(DoseResponseBenchmark),
            Box::new(YerkesDodsonBenchmark),
            Box::new(RewardLearningBenchmark),
            Box::new(AttentionNetworkBenchmark),
            Box::new(MoodInductionBenchmark),
            Box::new(BehavioralKnockoutBenchmark),
            Box::new(ConsciousnessPharmacologyBenchmark),
            Box::new(ToleranceWithdrawalBenchmark),
            Box::new(AntagonistProfilesBenchmark),
            Box::new(MultiTransmitterSynergyBenchmark),
        ];

        let neuromod_results: Vec<_> = neuromod_benches
            .par_iter()
            .map(|b| b.run(&config))
            .collect();

        let path = out.join("neuromod_profiles.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "benchmark,metric,mean,sd").unwrap();
        for result in &neuromod_results {
            for (metric_name, mv) in &result.metrics {
                writeln!(
                    f,
                    "{},{},{:.6},{:.6}",
                    result.benchmark, metric_name, mv.mean, mv.std_dev
                )
                .unwrap();
            }
        }

        // Also include ConsciousnessFeedback and MoralOxytocin results
        let cf = ConsciousnessFeedbackBenchmark.run();
        for (metric_name, mv) in &cf.metrics {
            writeln!(
                f,
                "{},{},{:.6},{:.6}",
                cf.benchmark, metric_name, mv.mean, mv.std_dev
            )
            .unwrap();
        }
        let mo = MoralOxytocinBenchmark.run();
        for (metric_name, mv) in &mo.metrics {
            writeln!(
                f,
                "{},{},{:.6},{:.6}",
                mo.benchmark, metric_name, mv.mean, mv.std_dev
            )
            .unwrap();
        }

        eprintln!("  Wrote {}", path.display());
    }

    // ── CSV 8: neuromod_curves.csv (Figure 6 panels) ──────────────
    eprintln!("\nGenerating neuromod_curves.csv...");
    {
        use symthaea_core::hdc::ContinuousHV;
        use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

        let path = out.join("neuromod_curves.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "panel,x_var,x_value,y_var,y_value").unwrap();

        // ── Panel (a): Yerkes-Dodson NE sweep ──
        // Low dimension (64) + higher AFC to make noise actually degrade discrimination.
        let dim = 64usize;
        let ne_levels: Vec<f64> = (1..=9).map(|i| i as f64 * 0.1).collect();
        let n_runs = 10; // average over multiple runs

        for &ne in &ne_levels {
            let mut simple_accs = Vec::new();
            let mut complex_accs = Vec::new();

            for run in 0..n_runs {
                let base_seed = 42u64.wrapping_add(run * 7919);
                let mut rng = base_seed ^ 0x9E3779B97F4A7C15;
                let next = |s: &mut u64| -> u64 {
                    *s ^= *s << 13;
                    *s ^= *s >> 7;
                    *s ^= *s << 17;
                    *s
                };

                // Simple task: 4-AFC (target + 3 distractors)
                let target = ContinuousHV::random(dim, next(&mut rng));
                let distractors: Vec<ContinuousHV> = (0..3)
                    .map(|_| ContinuousHV::random(dim, next(&mut rng)))
                    .collect();
                let mut correct = 0usize;
                let trials = 20;
                for _ in 0..trials {
                    let noise_scale = 1.0 - (1.0 - (ne - 0.6_f64).powi(2) * 4.0).max(0.0);
                    let noise_weight = (noise_scale * 0.85) as f32;
                    let signal_weight = (1.0 - noise_weight).max(0.05);
                    let noise_hv = ContinuousHV::random(dim, next(&mut rng));
                    let noisy = ContinuousHV::weighted_bundle(
                        &[&target, &noise_hv],
                        &[signal_weight, noise_weight],
                    );
                    let sim_target = noisy.similarity(&target);
                    let best_distractor = distractors
                        .iter()
                        .map(|d| noisy.similarity(d))
                        .fold(f32::NEG_INFINITY, f32::max);
                    if sim_target > best_distractor {
                        correct += 1;
                    }
                }
                simple_accs.push(correct as f64 / trials as f64);

                // Complex task: 8-AFC multi-prototype discrimination
                let protos: Vec<ContinuousHV> = (0..8)
                    .map(|_| ContinuousHV::random(dim, next(&mut rng)))
                    .collect();
                let mut correct = 0usize;
                for _ in 0..trials {
                    let ci = (next(&mut rng) % 8) as usize;
                    let noise_scale = 1.0 - (1.0 - (ne - 0.45_f64).powi(2) * 5.0).max(0.0);
                    let noise_weight = (noise_scale * 0.95) as f32;
                    let signal_weight = (1.0 - noise_weight).max(0.05);
                    let noise_hv = ContinuousHV::random(dim, next(&mut rng));
                    let noisy = ContinuousHV::weighted_bundle(
                        &[&protos[ci], &noise_hv],
                        &[signal_weight, noise_weight],
                    );
                    let best = protos
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            noisy.similarity(a).total_cmp(&noisy.similarity(b))
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if best == ci {
                        correct += 1;
                    }
                }
                complex_accs.push(correct as f64 / trials as f64);
            }

            let simple_mean = simple_accs.iter().sum::<f64>() / n_runs as f64;
            let complex_mean = complex_accs.iter().sum::<f64>() / n_runs as f64;
            writeln!(
                f,
                "yerkes_dodson,ne,{ne:.2},simple_accuracy,{simple_mean:.6}"
            )
            .unwrap();
            writeln!(
                f,
                "yerkes_dodson,ne,{ne:.2},complex_accuracy,{complex_mean:.6}"
            )
            .unwrap();
        }
        eprintln!("  Panel (a): Yerkes-Dodson NE sweep done");

        // ── Panel (b): DA Reward Learning RPE per block ──
        // Track Q-value divergence across 8 blocks of 10 trials
        let n_runs = 20;
        let blocks = 8;
        let trials_per_block = 10;
        let lr = 0.15_f64;
        let temp = 0.3_f64;

        let mut block_rpes: Vec<Vec<f64>> = vec![Vec::new(); blocks];
        let mut block_q_diffs: Vec<Vec<f64>> = vec![Vec::new(); blocks];

        for run in 0..n_runs {
            let base_seed = 42u64.wrapping_add(run * 3571);
            let mut rng = base_seed ^ 0x9E3779B97F4A7C15;
            let next = |s: &mut u64| -> u64 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                *s
            };

            let mut q = [0.5_f64, 0.5];

            for block in 0..blocks {
                let mut block_rpe_sum = 0.0_f64;
                let in_reversal = block >= 4; // phase 2 starts at block 4

                for _ in 0..trials_per_block {
                    // Softmax choice
                    let max_q = q[0].max(q[1]);
                    let e0 = ((q[0] - max_q) / temp).exp();
                    let e1 = ((q[1] - max_q) / temp).exp();
                    let p0 = e0 / (e0 + e1);
                    let r = (next(&mut rng) % 10000) as f64 / 10000.0;
                    let choice = if r < p0 { 0 } else { 1 };

                    let reward = if !in_reversal {
                        if choice == 0 { 1.0 } else { 0.0 }
                    } else {
                        if choice == 1 { 1.0 } else { 0.0 }
                    };

                    let rpe = reward - q[choice];
                    q[choice] = (q[choice] + lr * rpe).clamp(0.0, 1.0);
                    block_rpe_sum += rpe.abs();
                }

                block_rpes[block].push(block_rpe_sum / trials_per_block as f64);
                block_q_diffs[block].push(q[1] - q[0]);
            }
        }

        for block in 0..blocks {
            let rpe_mean = block_rpes[block].iter().sum::<f64>() / n_runs as f64;
            let q_diff_mean = block_q_diffs[block].iter().sum::<f64>() / n_runs as f64;
            let block_label = block + 1;
            writeln!(
                f,
                "reward_learning,block,{block_label},mean_abs_rpe,{rpe_mean:.6}"
            )
            .unwrap();
            writeln!(
                f,
                "reward_learning,block,{block_label},q_value_diff,{q_diff_mean:.6}"
            )
            .unwrap();
        }
        eprintln!("  Panel (b): DA Reward Learning RPE done");

        // ── Panel (c): ACh Attention Network decomposition ──
        // Run ANT benchmark at different ACh-orienting boost levels
        // (replicating the ANT logic with parametric orienting weight)
        let ach_levels: Vec<f64> = (0..=10).map(|i| i as f64 * 0.05).collect(); // 0.0 to 0.5
        let n_runs = 10;

        for &ach_boost in &ach_levels {
            let mut alerting_effs = Vec::new();
            let mut orienting_effs = Vec::new();
            let mut conflict_effs = Vec::new();

            for run in 0..n_runs {
                let base_seed = 42u64.wrapping_add(run * 6151);
                let mut rng = base_seed ^ 0x9E3779B97F4A7C15;
                let next = |s: &mut u64| -> u64 {
                    *s ^= *s << 13;
                    *s ^= *s >> 7;
                    *s ^= *s << 17;
                    *s
                };

                let left_p = ContinuousHV::random(dim, next(&mut rng));
                let right_p = ContinuousHV::random(dim, next(&mut rng));
                let alert_p = ContinuousHV::random(dim, next(&mut rng));
                let loc_p = ContinuousHV::random(dim, next(&mut rng));

                // 8 conditions: alert × orient × congruent
                let mut mean_rts = [0.0_f64; 8];
                let trials_per = 15;

                for ci in 0..8 {
                    let alert_cue = ci >= 4;
                    let orient_cue = (ci / 2) % 2 == 1;
                    let congruent = ci % 2 == 0;
                    let mut rt_sum = 0.0_f64;

                    for _ in 0..trials_per {
                        let is_left = next(&mut rng) % 2 == 0;
                        let tp = if is_left { &left_p } else { &right_p };

                        let mut comps: Vec<&ContinuousHV> = vec![tp];
                        let mut ws: Vec<f32> = vec![0.5];

                        if alert_cue {
                            comps.push(&alert_p);
                            ws.push(0.15);
                            ws[0] -= 0.05;
                        }
                        if orient_cue {
                            comps.push(&loc_p);
                            ws.push(ach_boost as f32); // parametric ACh boost
                            ws[0] -= (ach_boost as f32 * 0.33).min(ws[0] - 0.1);
                        }
                        if !congruent {
                            let opp = if is_left { &right_p } else { &left_p };
                            comps.push(opp);
                            ws.push(0.25);
                            ws[0] -= 0.1;
                        }
                        let total: f32 = ws.iter().sum();
                        let nw: Vec<f32> = ws.iter().map(|w| w / total).collect();
                        let stim = ContinuousHV::weighted_bundle(&comps, &nw);
                        let margin = (stim.similarity(&left_p) as f64
                            - stim.similarity(&right_p) as f64)
                            .abs();
                        let mut rt = 5.0 + (1.0 - margin.min(1.0)) * 8.0;
                        // NE-mediated alerting: temporal preparation reduces baseline RT
                        if alert_cue {
                            rt -= 1.2;
                        }
                        // ACh-mediated orienting: spatial filtering reduces baseline RT
                        if orient_cue {
                            rt -= 0.8;
                        }
                        rt_sum += rt.max(1.0);
                    }
                    mean_rts[ci] = rt_sum / trials_per as f64;
                }

                let no_alert = (mean_rts[0] + mean_rts[1] + mean_rts[2] + mean_rts[3]) / 4.0;
                let alert = (mean_rts[4] + mean_rts[5] + mean_rts[6] + mean_rts[7]) / 4.0;
                let no_orient = (mean_rts[0] + mean_rts[1] + mean_rts[4] + mean_rts[5]) / 4.0;
                let orient = (mean_rts[2] + mean_rts[3] + mean_rts[6] + mean_rts[7]) / 4.0;
                let cong = (mean_rts[0] + mean_rts[2] + mean_rts[4] + mean_rts[6]) / 4.0;
                let incong = (mean_rts[1] + mean_rts[3] + mean_rts[5] + mean_rts[7]) / 4.0;

                alerting_effs.push(no_alert - alert);
                orienting_effs.push(no_orient - orient);
                conflict_effs.push(incong - cong);
            }

            let ale_mean = alerting_effs.iter().sum::<f64>() / n_runs as f64;
            let ori_mean = orienting_effs.iter().sum::<f64>() / n_runs as f64;
            let con_mean = conflict_effs.iter().sum::<f64>() / n_runs as f64;
            writeln!(
                f,
                "attention_network,ach_boost,{ach_boost:.2},alerting_effect,{ale_mean:.6}"
            )
            .unwrap();
            writeln!(
                f,
                "attention_network,ach_boost,{ach_boost:.2},orienting_effect,{ori_mean:.6}"
            )
            .unwrap();
            writeln!(
                f,
                "attention_network,ach_boost,{ach_boost:.2},conflict_effect,{con_mean:.6}"
            )
            .unwrap();
        }
        eprintln!("  Panel (c): ACh Attention Network done");

        // ── Panel (d): Dose-Response monotonicity curves ──
        // Replicate dose_sweep for all 5 transmitters
        let doses = [0.1_f32, 0.2, 0.3, 0.5, 0.8];
        let warmup = 10;
        let observe = 30;
        let neutral_inputs = NeuromodulatorInputs {
            prediction_error: 0.2,
            surprise: false,
            reward_signal: 0.0,
            coherence: 0.6,
            arousal: 0.4,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
            consciousness_level: None,
            moral_signal: None,
        };

        let transmitters: &[(&str, fn(&NeuromodulatorBath) -> f32)] = &[
            ("da", |b: &NeuromodulatorBath| b.learning_rate_factor()),
            ("ne", |b: &NeuromodulatorBath| b.exploration_delta()),
            ("sht", |b: &NeuromodulatorBath| b.confidence_delta()),
            ("ach", |b: &NeuromodulatorBath| b.attention_factor()),
            ("gaba", |b: &NeuromodulatorBath| b.global_inhibition()),
        ];

        for &(name, metric_fn) in transmitters {
            for &dose in &doses {
                let mut bath = NeuromodulatorBath::default();
                for _ in 0..warmup {
                    bath.update(&neutral_inputs);
                    match name {
                        "da" => {
                            bath.clamp_all_levels(Some(dose), None, None, None, None, None, None)
                        }
                        "ne" => {
                            bath.clamp_all_levels(None, Some(dose), None, None, None, None, None)
                        }
                        "sht" => {
                            bath.clamp_all_levels(None, None, Some(dose), None, None, None, None)
                        }
                        "ach" => {
                            bath.clamp_all_levels(None, None, None, Some(dose), None, None, None)
                        }
                        "gaba" => {
                            bath.clamp_all_levels(None, None, None, None, Some(dose), None, None)
                        }
                        _ => {}
                    }
                }
                let mut sum = 0.0_f64;
                for _ in 0..observe {
                    bath.update(&neutral_inputs);
                    match name {
                        "da" => {
                            bath.clamp_all_levels(Some(dose), None, None, None, None, None, None)
                        }
                        "ne" => {
                            bath.clamp_all_levels(None, Some(dose), None, None, None, None, None)
                        }
                        "sht" => {
                            bath.clamp_all_levels(None, None, Some(dose), None, None, None, None)
                        }
                        "ach" => {
                            bath.clamp_all_levels(None, None, None, Some(dose), None, None, None)
                        }
                        "gaba" => {
                            bath.clamp_all_levels(None, None, None, None, Some(dose), None, None)
                        }
                        _ => {}
                    }
                    sum += metric_fn(&bath) as f64;
                }
                let mean_val = sum / observe as f64;
                writeln!(
                    f,
                    "dose_response,dose,{dose:.2},{name}_output,{mean_val:.6}"
                )
                .unwrap();
            }
        }
        eprintln!("  Panel (d): Dose-Response curves done");

        eprintln!("  Wrote {}", path.display());
    }

    eprintln!(
        "\nAll CSVs written to {}\nTotal time: {:.1}s",
        out.display(),
        total_start.elapsed().as_secs_f64()
    );
}
