//! Write deterministic non-secret configuration files for the Symtropy
//! fresh-process-driving-observation-replay fixture.
//!
//! Usage:
//!   cargo run --locked --manifest-path crates/bridges/symthaea-camera-driving-bridge/Cargo.toml \
//!     --example write_fixture_configs -- bridge-config.json vision-config.json

use std::env;
use std::error::Error;
use std::fs;

use symthaea_camera_driving_bridge::{BridgeConfig, DrivingRunIdentity};
use symthaea_vision_manifold::VisionConfig;

fn run() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: write_fixture_configs <bridge-config.json> <vision-config.json>".into());
    }

    let bridge_config = BridgeConfig {
        identity: DrivingRunIdentity {
            run_id: "fresh-process-driving-fixture-run-v0-1".to_string(),
            episode_id: "fresh-process-driving-episode-v0-1".to_string(),
            scenario_id: "straight-to-curved-road-two-actors-v0-1".to_string(),
            agent_id: "ego".to_string(),
            camera_id: "front-rgb".to_string(),
            simulator_revision: "pre-recorded-kinematic-snapshot-fixture-v1".to_string(),
            policy_revision: "untrained-input-replay-only-v0-1".to_string(),
        },
        max_width: 640,
        max_height: 360,
        max_frame_bytes: 640 * 360 * 3,
        initial_dt_us: 33_333,
        maximum_frame_gap_us: 66_666,
        maximum_observation_delivery_age_us: 5_000,
        maximum_sensor_clock_skew_us: 1_000,
        visual_weight_ppm: 500_000,
        vehicle_encoder_seed_phrase: "public-nonsecret-fresh-process-driving-fixture-seed-v0-1"
            .to_string(),
        expected_camera_calibration_sha256: None,
    };
    bridge_config.validate()?;

    let vision_config = VisionConfig::default();
    if vision_config.hdc_dim != 16_384 || vision_config.enable_depth {
        return Err("default VisionConfig does not meet this fixture's 16,384D no-depth contract".into());
    }
    vision_config.validate()?;

    fs::write(&args[0], serde_json::to_vec_pretty(&bridge_config)?)?;
    fs::write(&args[1], serde_json::to_vec_pretty(&vision_config)?)?;
    println!("wrote bridge config: {}", args[0]);
    println!("wrote vision config: {}", args[1]);
    println!("fixture seed phrase is deterministic and non-secret; do not use it for production");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("write-fixture-configs: {error}");
        std::process::exit(2);
    }
}
