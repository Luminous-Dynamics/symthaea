//! Generate CSV data files for the Psych-Bench BRM paper.
//!
//! Runs the full benchmark suite and produces 7 CSV files consumed
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
    AttentionalBlinkBenchmark, VisualSearchBenchmark,
};
use symthaea_psych_bench::benchmarks::butlin::ButlinIndicatorSuite;
use symthaea_psych_bench::benchmarks::cogbench::{
    BartBenchmark, HorizonBenchmark, InstrumentalLearningBenchmark,
    ProbabilisticReasoningBenchmark, RestlessBanditBenchmark, ReversalLearningBenchmark,
    TemporalDiscountingBenchmark, TwoStepBenchmark,
};
use symthaea_psych_bench::benchmarks::creativity::{
    AlternateUsesBenchmark, RemoteAssociatesBenchmark,
};
use symthaea_psych_bench::benchmarks::executive::{
    DualTaskBenchmark, FlankerBenchmark, IowaGamblingBenchmark, RavensProgressiveMatricesBenchmark,
    StroopBenchmark, TowerOfLondonBenchmark, WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::{GoNoGoBenchmark, StopSignalBenchmark};
use symthaea_psych_bench::benchmarks::language::{
    GardenPathBenchmark, LexicalDecisionBenchmark, SemanticCoherenceBenchmark,
    SemanticPrimingBenchmark,
};
use symthaea_psych_bench::benchmarks::memory_agent::{
    AccurateRetrievalBenchmark, ConflictResolutionBenchmark, LongRangeBenchmark,
    ProspectiveMemoryBenchmark, TestTimeLearningBenchmark,
};
use symthaea_psych_bench::benchmarks::metacognition::{
    FeelingOfKnowingBenchmark, MetacognitiveCalibrationBenchmark,
};
use symthaea_psych_bench::benchmarks::motor::{
    BimanualBenchmark, FittsLawBenchmark, SrttBenchmark,
};
use symthaea_psych_bench::benchmarks::neuromod::{
    AllostaticStressBenchmark, AntagonistProfilesBenchmark, AttentionNetworkBenchmark,
    BehavioralKnockoutBenchmark, ConsciousnessFeedbackBenchmark, ConsciousnessPharmacologyBenchmark,
    DoseResponseBenchmark, InjectionChallengeBenchmark,
    MoodInductionBenchmark, MoralOxytocinBenchmark, MultiTransmitterSynergyBenchmark,
    PharmacologicalAblationBenchmark, PharmacologicalChallengeBenchmark, RewardLearningBenchmark,
    ToleranceWithdrawalBenchmark, YerkesDodsonBenchmark,
};
use symthaea_psych_bench::benchmarks::reasoning::{
    ArcAbductiveBenchmark, ArcAlgebraBenchmark, ArcAnalogyBenchmark, ArcChainBenchmark,
    ArcCompositionalBenchmark, ArcFewShotBenchmark, ArcFluidBenchmark, ArcNoiseBenchmark,
    ArcRsaBenchmark, ArcScalingBenchmark, ArcStaircaseBenchmark,
};
use symthaea_psych_bench::benchmarks::social::{
    RmeBenchmark, SocialNormBenchmark, UltimatumGameBenchmark,
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
    AblationPreset, BenchmarkConfig, BenchmarkReport, CognitiveProfile, CrossDomainMatrix,
    NormativeReport, PsychBenchmark, ReliabilityBattery, SatBattery,
};

/// All 76 PsychBenchmark-trait benchmarks (excludes ConsciousnessFeedback and
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
        // Metacognition (2)
        Box::new(MetacognitiveCalibrationBenchmark),
        Box::new(FeelingOfKnowingBenchmark),
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
        // Creativity (2)
        Box::new(RemoteAssociatesBenchmark),
        Box::new(AlternateUsesBenchmark),
        // Inhibition (2)
        Box::new(GoNoGoBenchmark),
        Box::new(StopSignalBenchmark),
        // Attention (2)
        Box::new(AttentionalBlinkBenchmark),
        Box::new(VisualSearchBenchmark),
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
        // Motor (3)
        Box::new(SrttBenchmark),
        Box::new(FittsLawBenchmark),
        Box::new(BimanualBenchmark),
        // Language (4)
        Box::new(GardenPathBenchmark),
        Box::new(SemanticCoherenceBenchmark),
        Box::new(LexicalDecisionBenchmark),
        Box::new(SemanticPrimingBenchmark),
        // Social (3)
        Box::new(RmeBenchmark),
        Box::new(UltimatumGameBenchmark),
        Box::new(SocialNormBenchmark),
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

    // ── CSV 1: cognitive_profile.csv ────────────────────────────────
    eprintln!("\n[2/7] Generating cognitive_profile.csv...");
    let profile = CognitiveProfile::from_report(&report);
    {
        let path = out.join("cognitive_profile.csv");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "domain,score,n_benchmarks,interpretation").unwrap();
        for d in &profile.domains {
            writeln!(
                f,
                "{},{:.6},{},{}",
                d.domain, d.score, d.n_benchmarks, d.interpretation
            )
            .unwrap();
        }
        eprintln!("  Wrote {} ({} domains)", path.display(), profile.domains.len());
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
        let presets = AblationPreset::all();
        let mut ablation_profiles: Vec<(String, CognitiveProfile)> = Vec::new();

        for &preset in presets {
            let ac = preset.to_config(42);
            eprintln!("  Running ablation: {}...", ac.name);
            let ab_results: Vec<_> = benchmarks
                .par_iter()
                .map(|b| b.run(&ac.base))
                .collect();
            let mut ab_report = BenchmarkReport::new();
            for r in ab_results {
                ab_report.add(r);
            }
            let ab_profile = CognitiveProfile::from_report(&ab_report);
            ablation_profiles.push((ac.name.clone(), ab_profile));
        }

        let path = out.join("ablation_domains.csv");
        let mut f = fs::File::create(&path).unwrap();
        // Wide format: domain, then one column per preset
        let preset_names: Vec<&str> = ablation_profiles.iter().map(|(n, _)| n.as_str()).collect();
        write!(f, "domain").unwrap();
        for name in &preset_names {
            // Sanitize column name for pgfplotstableread (no spaces or parens)
            let clean = name
                .replace(' ', "")
                .replace("(K=3)", "");
            write!(f, ",{}", clean).unwrap();
        }
        writeln!(f).unwrap();

        // Collect all unique domain names (sorted)
        let domain_names: Vec<String> = ablation_profiles[0]
            .1
            .domains
            .iter()
            .map(|d| d.domain.clone())
            .collect();

        for domain in &domain_names {
            write!(f, "{}", domain).unwrap();
            for (_name, prof) in &ablation_profiles {
                let score = prof
                    .domains
                    .iter()
                    .find(|d| &d.domain == domain)
                    .map(|d| d.score)
                    .unwrap_or(0.0);
                write!(f, ",{:.6}", score).unwrap();
            }
            writeln!(f).unwrap();
        }
        eprintln!(
            "  Wrote {} ({} presets × {} domains)",
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
            if let Some(curve) = sat_battery.curves.iter().find(|c| c.benchmark == *bench_name) {
                let p = out.join(filename);
                let mut bf = fs::File::create(&p).unwrap();
                writeln!(bf, "time_pressure,accuracy,mean_rt").unwrap();
                for pt in &curve.points {
                    writeln!(bf, "{:.6},{:.6},{:.6}", pt.time_pressure, pt.accuracy, pt.mean_rt)
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
        let reliability = ReliabilityBattery::run(&rel_refs, &config, 3);

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

    eprintln!(
        "\nAll CSVs written to {}\nTotal time: {:.1}s",
        out.display(),
        total_start.elapsed().as_secs_f64()
    );
}
