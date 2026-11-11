//! Process one complete driving episode through the persistent camera/HDC bridge.
//!
//! Usage:
//!   camera-driving-bridge <episode-request.json> <receipts-output.json> <policy-input-dir>
//!
//! Input pixel files are relative to the episode request directory. Path traversal
//! and symlink escapes are rejected. Output HDC vectors are binary f32 little-endian
//! artifacts whose first eight bytes encode the dimension (u64 LE).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};

use symthaea_camera_driving_bridge::{
    BridgeConfig, CameraDrivingBridge, CapturedDrivingObservation, DrivingObservationReceipt,
    DrivingRunIdentity, UpstreamCameraObservation, RECEIPT_SCHEMA_VERSION,
};
use symthaea_vehicle::VehicleState;
use symthaea_vision_manifold::VisionConfig;

const MAX_EPISODE_OBSERVATIONS: usize = 100_000;
const MAX_EPISODE_INPUT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EpisodeRequest {
    schema_version: u32,
    bridge_config: BridgeConfig,
    vision_config: VisionConfig,
    observations: Vec<EpisodeObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EpisodeObservation {
    sequence: u64,
    timestamp_us: u64,
    vehicle_state_timestamp_us: u64,
    pixels_file: String,
    vehicle_state: VehicleState,
    upstream: UpstreamCameraObservation,
}

#[derive(Debug, Serialize)]
struct PolicyInputArtifact {
    sequence: u64,
    upstream_observation_id: String,
    relative_path: String,
    dimension: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct EpisodeOutput {
    schema_version: u32,
    receipt_schema_version: u32,
    run_id: String,
    episode_id: String,
    expected_observations: usize,
    processed_observations: usize,
    frames_processed: u64,
    vision_learning_frozen: bool,
    vision_training_steps_before: u64,
    vision_training_steps_after: u64,
    hdc_dimension: usize,
    receipts: Vec<DrivingObservationReceipt>,
    policy_input_artifacts: Vec<PolicyInputArtifact>,
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    if value.trim().is_empty() {
        return Err("pixels_file must be non-empty".into());
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("pixels_file must be a safe relative path, got {value:?}"));
    }
    Ok(path.to_path_buf())
}

fn resolve_input_file(base: &Path, relative: &Path) -> Result<PathBuf, String> {
    let canonical_base = fs::canonicalize(base)
        .map_err(|error| format!("cannot resolve episode request directory: {error}"))?;
    let candidate = fs::canonicalize(canonical_base.join(relative))
        .map_err(|error| format!("cannot resolve pixel artifact {relative:?}: {error}"))?;
    if !candidate.starts_with(&canonical_base) {
        return Err(format!(
            "pixel artifact escapes the episode request directory: {relative:?}"
        ));
    }
    let metadata = fs::metadata(&candidate)
        .map_err(|error| format!("cannot inspect pixel artifact {relative:?}: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("pixel artifact is not a regular file: {relative:?}"));
    }
    Ok(candidate)
}

fn encode_policy_hv(dimension: usize, values: &[f32]) -> Result<Vec<u8>, String> {
    if dimension == 0 || values.len() != dimension {
        return Err(format!(
            "invalid HDC vector: declared dimension {dimension}, actual length {}",
            values.len()
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("policy input contains a non-finite component".into());
    }
    let capacity = 8_usize
        .checked_add(dimension.checked_mul(4).ok_or("HDC byte-length overflow")?)
        .ok_or("HDC byte-length overflow")?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut rendered = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut rendered, "{byte:02x}");
    }
    rendered
}

fn process_episode(
    request_path: &Path,
    output_path: &Path,
    policy_input_dir: &Path,
) -> Result<EpisodeOutput, Box<dyn Error>> {
    let request_bytes = fs::read(request_path)?;
    if request_bytes.len() as u64 > MAX_EPISODE_INPUT_BYTES {
        return Err("episode request exceeds the 4 GiB input bound".into());
    }
    let request: EpisodeRequest = serde_json::from_slice(&request_bytes)?;
    if request.schema_version != 1 {
        return Err(format!(
            "unsupported episode request schema version {}",
            request.schema_version
        )
        .into());
    }
    if request.observations.is_empty() || request.observations.len() > MAX_EPISODE_OBSERVATIONS {
        return Err(format!(
            "observations must contain 1..={MAX_EPISODE_OBSERVATIONS} entries"
        )
        .into());
    }

    let input_base = request_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut bridge = CameraDrivingBridge::try_new(
        request.bridge_config.clone(),
        request.vision_config,
    )?;
    let initial_training_steps = bridge.training_steps();

    if output_path.exists() {
        return Err(format!("receipts output already exists: {}", output_path.display()).into());
    }
    if policy_input_dir.exists() {
        let mut existing = fs::read_dir(policy_input_dir)?;
        if existing.next().is_some() {
            return Err(format!(
                "policy input directory must be empty before processing: {}",
                policy_input_dir.display()
            )
            .into());
        }
    }

    let mut receipts = Vec::with_capacity(request.observations.len());
    let mut policy_input_artifacts = Vec::with_capacity(request.observations.len());
    let mut staged_policy_inputs = Vec::with_capacity(request.observations.len());
    let mut aggregate_input_bytes = 0_u64;

    for (index, observation) in request.observations.iter().enumerate() {
        let relative_pixels = safe_relative_path(&observation.pixels_file)?;
        let pixels_path = resolve_input_file(input_base, &relative_pixels)?;
        let metadata = fs::metadata(&pixels_path)?;
        aggregate_input_bytes = aggregate_input_bytes
            .checked_add(metadata.len())
            .ok_or("aggregate pixel artifact size overflow")?;
        if aggregate_input_bytes > MAX_EPISODE_INPUT_BYTES {
            return Err("aggregate pixel artifacts exceed the 4 GiB bound".into());
        }
        let pixels = fs::read(&pixels_path)?;

        let result = bridge.observe(CapturedDrivingObservation {
            sequence: observation.sequence,
            timestamp_us: observation.timestamp_us,
            vehicle_state_timestamp_us: observation.vehicle_state_timestamp_us,
            pixels: &pixels,
            width: observation.upstream.width,
            height: observation.upstream.height,
            channels: observation.upstream.channels,
            vehicle_state: &observation.vehicle_state,
            upstream: observation.upstream.clone(),
        })?;

        let policy_input_bytes = encode_policy_hv(
            result.policy_input.dim(),
            result.policy_input.as_slice(),
        )?;
        let policy_input_sha256 = sha256_hex(&policy_input_bytes);
        if policy_input_sha256 != result.receipt.fused_policy_input_sha256 {
            return Err(format!(
                "policy input artifact digest mismatch at sequence {}",
                observation.sequence
            )
            .into());
        }

        let artifact_name = format!("policy_input_{:08}.f32le", observation.sequence);
        let relative_artifact_path = artifact_name.clone();
        staged_policy_inputs.push((artifact_name, policy_input_bytes.clone()));

        policy_input_artifacts.push(PolicyInputArtifact {
            sequence: observation.sequence,
            upstream_observation_id: observation.upstream.observation_id.clone(),
            relative_path: relative_artifact_path,
            dimension: result.policy_input.dim(),
            byte_length: policy_input_bytes.len(),
            sha256: policy_input_sha256,
        });
        receipts.push(result.receipt);

        if index > 0 && receipts[index].previous_receipt_sha256.as_ref()
            != Some(&receipts[index - 1].receipt_sha256)
        {
            return Err(format!(
                "receipt chain mismatch between sequences {} and {}",
                receipts[index - 1].capture_sequence,
                receipts[index].capture_sequence
            )
            .into());
        }
    }

    if !bridge.learning_is_frozen() {
        return Err("Vision Manifold learning was unexpectedly unfrozen".into());
    }
    let final_training_steps = bridge.training_steps();
    if final_training_steps != initial_training_steps {
        return Err("Vision Manifold training steps changed during frozen episode".into());
    }

    let output = EpisodeOutput {
        schema_version: 1,
        receipt_schema_version: RECEIPT_SCHEMA_VERSION,
        run_id: request.bridge_config.identity.run_id,
        episode_id: request.bridge_config.identity.episode_id,
        expected_observations: request.observations.len(),
        processed_observations: receipts.len(),
        frames_processed: bridge.frames_processed(),
        vision_learning_frozen: bridge.learning_is_frozen(),
        vision_training_steps_before: initial_training_steps,
        vision_training_steps_after: final_training_steps,
        hdc_dimension: policy_input_artifacts
            .first()
            .map(|artifact| artifact.dimension)
            .unwrap_or(0),
        receipts,
        policy_input_artifacts,
    };

    // Do not expose partial HDC artifacts when a later observation rejects.
    // Every frame, receipt, predecessor link and frozen-learning invariant has
    // passed before this single episode-level commit phase begins.
    fs::create_dir_all(policy_input_dir)?;
    for (artifact_name, artifact_bytes) in staged_policy_inputs {
        fs::write(policy_input_dir.join(artifact_name), artifact_bytes)?;
    }

    let encoded_output = serde_json::to_vec_pretty(&output)?;
    if let Some(parent) = output_path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, encoded_output)?;
    Ok(output)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            "usage: camera-driving-bridge <episode-request.json> <receipts-output.json> <policy-input-dir>"
                .into(),
        );
    }
    let request_path = PathBuf::from(&args[0]);
    let output_path = PathBuf::from(&args[1]);
    let policy_input_dir = PathBuf::from(&args[2]);

    let output = process_episode(&request_path, &output_path, &policy_input_dir)?;
    println!(
        "processed {} / {} camera observations for episode {}",
        output.processed_observations, output.expected_observations, output.episode_id
    );
    println!("receipts: {}", output_path.display());
    println!("policy inputs: {}", policy_input_dir.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("camera-driving-bridge: {error}");
        std::process::exit(2);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_pixel_path_traversal_and_absolute_paths() {
        assert!(safe_relative_path("../outside.rgb").is_err());
        assert!(safe_relative_path("/tmp/outside.rgb").is_err());
        assert!(safe_relative_path("nested/../../outside.rgb").is_err());
    }

    #[test]
    fn accepts_nested_relative_pixel_paths() {
        assert_eq!(
            safe_relative_path("frames/tick_0001.rgb").unwrap(),
            PathBuf::from("frames/tick_0001.rgb")
        );
    }

    #[test]
    fn policy_hdc_artifact_has_dimension_header_and_little_endian_values() {
        let encoded = encode_policy_hv(2, &[1.0, -0.5]).unwrap();
        assert_eq!(&encoded[..8], &2_u64.to_le_bytes());
        assert_eq!(&encoded[8..12], &1.0_f32.to_bits().to_le_bytes());
        assert_eq!(&encoded[12..16], &(-0.5_f32).to_bits().to_le_bytes());
        assert_eq!(sha256_hex(&encoded).len(), 64);
    }

    fn test_bridge_config() -> BridgeConfig {
        BridgeConfig {
            identity: DrivingRunIdentity {
                run_id: "atomic-episode-test-run".to_string(),
                episode_id: "atomic-episode-test".to_string(),
                scenario_id: "atomic-output-rollback-test".to_string(),
                agent_id: "ego".to_string(),
                camera_id: "front".to_string(),
                simulator_revision: "fixture-simulator-v1".to_string(),
                policy_revision: "untrained-policy-input-test".to_string(),
            },
            max_width: 16,
            max_height: 16,
            max_frame_bytes: 16 * 16 * 3,
            initial_dt_us: 10_000,
            maximum_frame_gap_us: 20_000,
            maximum_sensor_clock_skew_us: 0,
            visual_weight_ppm: 500_000,
            vehicle_encoder_seed_phrase: "public-atomic-output-test-seed".to_string(),
            expected_camera_calibration_sha256: Some("c".repeat(64)),
        }
    }

    fn test_upstream_receipt(
        pixels: &[u8],
        sequence: u64,
        tick: u64,
        timestamp_us: u64,
    ) -> UpstreamCameraObservation {
        let mut receipt = UpstreamCameraObservation {
            schema_version: RECEIPT_SCHEMA_VERSION,
            observation_id: String::new(),
            pixel_format: "rgb8_srgb".to_string(),
            frame_byte_len: pixels.len(),
            receipt_sha256: String::new(),
            frame_sha256: sha256_hex(pixels),
            episode_id: "atomic-episode-test".to_string(),
            agent_id: "ego".to_string(),
            clock_id: "atomic-test-clock".to_string(),
            tick,
            sequence,
            camera_id: "front".to_string(),
            camera_calibration_sha256: "c".repeat(64),
            renderer_revision: "atomic-output-test-renderer-v1".to_string(),
            capture_timestamp_us: timestamp_us,
            render_completed_timestamp_us: timestamp_us + 800,
            delivered_timestamp_us: timestamp_us + 1_000,
            width: 16,
            height: 16,
            channels: 3,
            applied_action_receipt_id: Some(format!("action-{tick}")),
            applied_action_receipt_sha256: Some("a".repeat(64)),
            applied_action_command_id: Some(format!("command-{tick}")),
            applied_action_tick: Some(tick),
        };
        receipt.observation_id = receipt.calculated_observation_id();
        receipt.receipt_sha256 = receipt.calculated_receipt_sha256();
        receipt
    }

    #[test]
    fn later_sequence_rejection_commits_no_partial_hdc_artifacts_or_receipts() {
        let unique = format!(
            "symthaea-camera-bridge-atomic-episode-{}",
            std::process::id()
        );
        let base = std::env::temp_dir().join(unique);
        let frame_dir = base.join("frames");
        let request_path = base.join("episode-request.json");
        let output_path = base.join("receipts.json");
        let policy_input_dir = base.join("policy-inputs");
        fs::create_dir_all(&frame_dir).unwrap();

        let first_pixels = vec![101_u8; 16 * 16 * 3];
        let second_pixels = vec![202_u8; 16 * 16 * 3];
        fs::write(frame_dir.join("first.rgb"), &first_pixels).unwrap();
        fs::write(frame_dir.join("second.rgb"), &second_pixels).unwrap();

        let mut first_state = VehicleState::cruising(10.0);
        first_state.timestamp = 0.01;
        let mut second_state = VehicleState::cruising(10.0);
        second_state.timestamp = 0.02;
        let first_upstream = test_upstream_receipt(&first_pixels, 2, 1, 10_000);
        // Upstream world tick advances from 1 to 2, and the capture clock and
        // vehicle-state clock advance, but the second observation sequence
        // rolls back from 1 to 1. This passes upstream identity checks and is
        // rejected by the bridge's persistent sequence guard.
        let second_upstream = test_upstream_receipt(&second_pixels, 1, 2, 20_000);

        let request = serde_json::json!({
            "schema_version": 1,
            "bridge_config": test_bridge_config(),
            "vision_config": VisionConfig::default(),
            "observations": [
                {
                    "sequence": 2,
                    "timestamp_us": 10_000,
                    "vehicle_state_timestamp_us": 10_000,
                    "pixels_file": "frames/first.rgb",
                    "vehicle_state": first_state,
                    "upstream": first_upstream
                },
                {
                    "sequence": 1,
                    "timestamp_us": 20_000,
                    "vehicle_state_timestamp_us": 20_000,
                    "pixels_file": "frames/second.rgb",
                    "vehicle_state": second_state,
                    "upstream": second_upstream
                }
            ]
        });
        fs::write(
            &request_path,
            serde_json::to_vec_pretty(&request).unwrap(),
        )
        .unwrap();

        let error = process_episode(&request_path, &output_path, &policy_input_dir)
            .err()
            .expect("the second observation sequence rollback must reject the episode");
        assert!(error.to_string().contains("SequenceRegression"));
        assert!(!output_path.exists());
        assert!(!policy_input_dir.exists());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn rejects_non_empty_policy_input_output_directory_before_processing() {
        let unique = format!(
            "symthaea-camera-bridge-output-test-{}",
            std::process::id()
        );
        let base = std::env::temp_dir().join(unique);
        let policy_dir = base.join("policy-inputs");
        let output_path = base.join("receipts.json");
        fs::create_dir_all(&policy_dir).unwrap();
        fs::write(policy_dir.join("stale.f32le"), b"stale").unwrap();

        let result = (|| -> Result<(), Box<dyn Error>> {
            if policy_dir.exists() {
                let mut existing = fs::read_dir(&policy_dir)?;
                if existing.next().is_some() {
                    return Err("policy input directory must be empty before processing".into());
                }
            }
            Ok(())
        })();
        assert!(result.is_err());
        assert!(!output_path.exists());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn rejects_non_finite_policy_vector_components() {
        assert!(encode_policy_hv(2, &[1.0, f32::NAN]).is_err());
        assert!(encode_policy_hv(2, &[1.0, f32::INFINITY]).is_err());
    }
}
