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
    UpstreamCameraObservation, RECEIPT_SCHEMA_VERSION,
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

    let input_base = request_path.parent().unwrap_or_else(|| Path::new("."));
    let mut bridge = CameraDrivingBridge::try_new(
        request.bridge_config.clone(),
        request.vision_config,
    )?;
    let initial_training_steps = bridge.training_steps();

    fs::create_dir_all(policy_input_dir)?;
    let mut receipts = Vec::with_capacity(request.observations.len());
    let mut policy_input_artifacts = Vec::with_capacity(request.observations.len());
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
        let artifact_path = policy_input_dir.join(&artifact_name);
        fs::write(&artifact_path, &policy_input_bytes)?;

        let relative_artifact_path = artifact_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("policy input artifact name is not valid UTF-8")?
            .to_string();

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
