#![forbid(unsafe_code)]

//! Reproducible wiring probe for the localized context-tree portfolio.
//! This is not a Type-I calibration or power benchmark.

use std::error::Error;

use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
use cmaq_053_compositional_risk_budget::Rational;
use cmaq_059_finite_sample_calibration::scenarios::{generate_stream, ScenarioConfig, ScenarioKind};
use cmaq_062_localized_context_portfolio::{
    ContextNodeConfig, ContextRegion, ContextTreePortfolio,
};

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn make_tree() -> Result<ContextTreePortfolio, Box<dyn Error>> {
    let configs = vec![
        ContextNodeConfig {
            node_id: "root".into(),
            parent_id: None,
            depth: 0,
            region: ContextRegion::root(1)?,
            response_map: FourierFeatureMap::new(1, vec![vec![0.5]], A.into())?,
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], B.into())?,
            prior_weight: Rational::new(1, 2)?,
        },
        ContextNodeConfig {
            node_id: "left".into(),
            parent_id: Some("root".into()),
            depth: 1,
            region: ContextRegion::new(vec![None], vec![Some(0.0)])?,
            response_map: FourierFeatureMap::new(1, vec![vec![1.0]], C.into())?,
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], D.into())?,
            prior_weight: Rational::new(1, 4)?,
        },
        ContextNodeConfig {
            node_id: "right".into(),
            parent_id: Some("root".into()),
            depth: 1,
            region: ContextRegion::new(vec![Some(0.0)], vec![None])?,
            response_map: FourierFeatureMap::new(1, vec![vec![2.0]], E.into())?,
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], F.into())?,
            prior_weight: Rational::new(1, 4)?,
        },
    ];
    Ok(ContextTreePortfolio::new(
        &configs,
        Rational::probability(1, 20)?,
        0.5,
    )?)
}

fn run(kind: ScenarioKind, label: &str) -> Result<(), Box<dyn Error>> {
    let mut config = ScenarioConfig::standard(kind, 256);
    config.covariate_dimension = 1;
    config.amplitude = 0.5;
    let stream = generate_stream("cmaq-062-probe-v1", &config, 31)?;
    let mut monitor = make_tree()?;

    for observation in &stream.observations {
        let update = monitor.observe_pair(
            observation.sequence,
            &format!("{label}-{}", observation.sequence),
            &observation.covariate,
            &observation.x,
            &observation.y,
        )?;
        println!(
            "{label},{},{},{:.12},{:.12},{},{},{}",
            update.sequence,
            update.active_path.join(";"),
            update.log_mixture_e_value,
            update.log_threshold,
            update.total_feature_coordinates_evaluated,
            update.quarantined,
            update.trace_sha256,
        );
        if update.quarantined { break; }
    }
    eprintln!(
        "scenario={} seed={} stream={} selected_counts={:?}",
        label,
        stream.seed_sha256,
        stream.stream_sha256,
        monitor.selected_counts(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("scenario,sequence,active_path,log_mixture_e,log_threshold,feature_coordinates,quarantined,trace_sha256");
    run(ScenarioKind::EqualConditionalIid, "equal-conditional-iid")?;
    run(ScenarioKind::SparseConditionalShift, "sparse-conditional-shift")?;
    run(
        ScenarioKind::EqualMarginalConditionalPairingShift,
        "equal-marginal-conditional-pairing-shift",
    )?;
    Ok(())
}
