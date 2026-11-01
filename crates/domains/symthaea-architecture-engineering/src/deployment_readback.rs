//! Exact-byte flash readback verification for a previously verified deployment artifact.
//!
//! This module proves that supplied readback bytes match the exact model/firmware binary
//! bytes and flash map bound to a VerifiedDeployableArtifact. Device identity, probe identity,
//! tool identity and the acquisition channel are digest-pinned inputs, not authenticated here.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::deployment_artifact::{
    create_deployment_measurement_receipt, verify_deployment_measurement_receipt,
    DeploymentArtifactError, DeploymentMeasurementReceipt, VerifiedDeployableArtifact,
};
use crate::hardware_objectives::{
    validate_hardware_profile, HardwareProfileManifest, ObjectiveObservation,
    ObjectiveRegistry, ScoredMeasurement,
};

const SESSION_SCHEMA: &str = "symthaea.nas.deployment-readback-session";
const RECEIPT_SCHEMA: &str = "symthaea.nas.deployment-readback-receipt";
const MEASUREMENT_SCHEMA: &str = "symthaea.nas.readback-bound-measurement";
const VERSION: &str = "0.1.0";
const REGION_MAP_DOMAIN: &[u8] = b"symthaea.nas.deployment-readback-region-map.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.deployment-readback-receipt.v0.1";
const MEASUREMENT_DOMAIN: &[u8] = b"symthaea.nas.readback-bound-measurement.v0.1";
const SHA256_HEX_LEN: usize = 64;
const MAX_REGIONS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashImageComponent {
    ModelBinary,
    FirmwareBinary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlashRegionBinding {
    pub region_id: String,
    pub component: FlashImageComponent,
    pub flash_address: u64,
    pub source_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadbackRegionSlice {
    pub region_id: String,
    pub observed_flash_address: u64,
    pub capture_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReadbackSessionManifest {
    pub schema: String,
    pub version: String,
    pub artifact_sha256: String,
    pub pipeline_sha256: String,
    pub build_receipt_sha256: String,
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub board_id: String,
    pub board_revision: String,
    pub device_serial: String,
    /// Digest of upstream device-identity evidence; not authenticated by this module.
    pub device_identity_evidence_sha256: String,
    pub flash_probe_id: String,
    pub flash_tool_executable_sha256: String,
    pub readback_protocol_sha256: String,
    pub readback_capture_sha256: String,
    pub readback_region_map_sha256: String,
    pub session_id: String,
    pub session_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReadbackRegionReceipt {
    pub region_id: String,
    pub component: FlashImageComponent,
    pub flash_address: u64,
    pub source_offset: u64,
    pub capture_offset: u64,
    pub byte_length: u64,
    pub expected_source_sha256: String,
    pub readback_bytes_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReadbackReceipt {
    pub schema: String,
    pub version: String,
    pub artifact_sha256: String,
    pub pipeline_sha256: String,
    pub build_receipt_sha256: String,
    pub model_binary_sha256: String,
    pub firmware_binary_sha256: String,
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub board_id: String,
    pub board_revision: String,
    pub device_serial: String,
    pub device_identity_evidence_sha256: String,
    pub flash_probe_id: String,
    pub flash_tool_executable_sha256: String,
    pub readback_protocol_sha256: String,
    pub readback_capture_sha256: String,
    pub readback_region_map_sha256: String,
    pub session_id: String,
    pub session_sequence: u64,
    pub regions: Vec<DeploymentReadbackRegionReceipt>,
    pub readback_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadbackBoundDeploymentMeasurementReceipt {
    pub schema: String,
    pub version: String,
    pub artifact_sha256: String,
    pub readback_receipt_sha256: String,
    pub measurement_receipt: DeploymentMeasurementReceipt,
    pub binding_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentReadbackError {
    pub field: String,
    pub message: String,
}

impl DeploymentReadbackError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for DeploymentReadbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for DeploymentReadbackError {}

/// Non-serializable in-process capability issued only after byte-level verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedDeploymentReadback {
    receipt: DeploymentReadbackReceipt,
}

impl VerifiedDeploymentReadback {
    pub fn receipt(&self) -> &DeploymentReadbackReceipt {
        &self.receipt
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.receipt.artifact_sha256
    }

    pub fn target_id(&self) -> &str {
        &self.receipt.target_id
    }

    pub fn hardware_profile_sha256(&self) -> &str {
        &self.receipt.hardware_profile_sha256
    }

    pub fn readback_receipt_sha256(&self) -> &str {
        &self.receipt.readback_receipt_sha256
    }

    pub fn session_id(&self) -> &str {
        &self.receipt.session_id
    }
}

/// Hash the independently observed region map. The input must already be sorted by region_id.
pub fn compute_readback_region_map_sha256(
    slices: &[ReadbackRegionSlice],
) -> Result<String, DeploymentReadbackError> {
    validate_observed_slices(slices)?;
    let bytes = serde_json::to_vec(slices)
        .map_err(|error| DeploymentReadbackError::new("region_map", error.to_string()))?;
    Ok(digest_bytes(REGION_MAP_DOMAIN, &bytes))
}

/// Verify raw captured flash bytes against the exact model and firmware binary bytes.
///
/// The capture is the compact concatenation of all readback regions in capture_offset order.
/// Every byte of both source binaries must be covered exactly once, and every byte of the
/// capture must be assigned to exactly one region.
pub fn verify_deployment_flash_readback(
    artifact: &VerifiedDeployableArtifact,
    profile: &HardwareProfileManifest,
    session: &DeploymentReadbackSessionManifest,
    bindings: &[FlashRegionBinding],
    observed_slices: &[ReadbackRegionSlice],
    model_binary: &[u8],
    firmware_binary: &[u8],
    readback_capture: &[u8],
) -> Result<VerifiedDeploymentReadback, DeploymentReadbackError> {
    validate_session_scope(artifact, profile, session)?;
    if model_binary.is_empty() || firmware_binary.is_empty() {
        return Err(DeploymentReadbackError::new(
            "binary_payloads",
            "model and firmware binaries must both be non-empty",
        ));
    }
    if sha256_raw(model_binary) != artifact.model_binary_sha256()
        || sha256_raw(firmware_binary) != artifact.firmware_binary_sha256()
    {
        return Err(DeploymentReadbackError::new(
            "binary_payloads",
            "source bytes do not match the exact verified model/firmware binary digests",
        ));
    }
    if readback_capture.is_empty() || sha256_raw(readback_capture) != session.readback_capture_sha256 {
        return Err(DeploymentReadbackError::new(
            "readback_capture_sha256",
            "capture is empty or its raw bytes do not match the session digest",
        ));
    }
    validate_bindings(bindings, model_binary.len(), firmware_binary.len())?;
    validate_observed_slices(observed_slices)?;
    if bindings.len() != observed_slices.len() {
        return Err(DeploymentReadbackError::new(
            "region_count",
            "expected and observed flash-region counts differ",
        ));
    }
    if compute_readback_region_map_sha256(observed_slices)? != session.readback_region_map_sha256 {
        return Err(DeploymentReadbackError::new(
            "readback_region_map_sha256",
            "observed region map does not match the session manifest",
        ));
    }
    validate_capture_coverage(observed_slices, readback_capture.len())?;
    validate_flash_address_non_overlap(bindings)?;

    let mut regions = Vec::with_capacity(bindings.len());
    for (index, binding) in bindings.iter().enumerate() {
        let observed = &observed_slices[index];
        if binding.region_id != observed.region_id {
            return Err(DeploymentReadbackError::new(
                format!("regions[{index}].region_id"),
                "expected and observed region IDs differ",
            ));
        }
        if binding.flash_address != observed.observed_flash_address {
            return Err(DeploymentReadbackError::new(
                format!("regions[{index}].flash_address"),
                "observed flash address differs from the expected linker/build map",
            ));
        }
        if binding.byte_length != observed.byte_length {
            return Err(DeploymentReadbackError::new(
                format!("regions[{index}].byte_length"),
                "observed byte length differs from the expected region length",
            ));
        }

        let source = match binding.component {
            FlashImageComponent::ModelBinary => model_binary,
            FlashImageComponent::FirmwareBinary => firmware_binary,
        };
        let source_start = to_usize(binding.source_offset, "source_offset")?;
        let length = to_usize(binding.byte_length, "byte_length")?;
        let source_end = source_start.checked_add(length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("regions[{index}].source_offset"),
                "source range overflows host address space",
            )
        })?;
        let expected_bytes = source.get(source_start..source_end).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("regions[{index}].source_offset"),
                "source segment is outside the verified binary",
            )
        })?;

        let capture_start = to_usize(observed.capture_offset, "capture_offset")?;
        let capture_end = capture_start.checked_add(length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("regions[{index}].capture_offset"),
                "capture range overflows host address space",
            )
        })?;
        let observed_bytes = readback_capture.get(capture_start..capture_end).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("regions[{index}].capture_offset"),
                "readback segment is outside the raw capture",
            )
        })?;
        if expected_bytes != observed_bytes {
            return Err(DeploymentReadbackError::new(
                format!("regions[{index}]"),
                "readback bytes differ from the exact source binary segment",
            ));
        }

        regions.push(DeploymentReadbackRegionReceipt {
            region_id: binding.region_id.clone(),
            component: binding.component,
            flash_address: binding.flash_address,
            source_offset: binding.source_offset,
            capture_offset: observed.capture_offset,
            byte_length: binding.byte_length,
            expected_source_sha256: sha256_raw(expected_bytes),
            readback_bytes_sha256: sha256_raw(observed_bytes),
        });
    }

    let mut receipt = DeploymentReadbackReceipt {
        schema: RECEIPT_SCHEMA.into(),
        version: VERSION.into(),
        artifact_sha256: artifact.artifact_sha256().into(),
        pipeline_sha256: artifact.pipeline_sha256().into(),
        build_receipt_sha256: artifact.build_receipt_sha256().into(),
        model_binary_sha256: artifact.model_binary_sha256().into(),
        firmware_binary_sha256: artifact.firmware_binary_sha256().into(),
        target_id: artifact.target_id().into(),
        hardware_profile_sha256: artifact.hardware_profile_sha256().into(),
        board_id: profile.board_id.clone(),
        board_revision: profile.board_revision.clone(),
        device_serial: session.device_serial.clone(),
        device_identity_evidence_sha256: session.device_identity_evidence_sha256.clone(),
        flash_probe_id: session.flash_probe_id.clone(),
        flash_tool_executable_sha256: session.flash_tool_executable_sha256.clone(),
        readback_protocol_sha256: session.readback_protocol_sha256.clone(),
        readback_capture_sha256: session.readback_capture_sha256.clone(),
        readback_region_map_sha256: session.readback_region_map_sha256.clone(),
        session_id: session.session_id.clone(),
        session_sequence: session.session_sequence,
        regions,
        readback_receipt_sha256: "0".repeat(SHA256_HEX_LEN),
    };
    receipt.readback_receipt_sha256 = compute_deployment_readback_receipt_sha256(&receipt)?;
    Ok(VerifiedDeploymentReadback { receipt })
}

/// Recompute all region bytes and all receipt fields from original build/readback payloads.
pub fn verify_deployment_readback_receipt(
    artifact: &VerifiedDeployableArtifact,
    profile: &HardwareProfileManifest,
    session: &DeploymentReadbackSessionManifest,
    bindings: &[FlashRegionBinding],
    observed_slices: &[ReadbackRegionSlice],
    model_binary: &[u8],
    firmware_binary: &[u8],
    readback_capture: &[u8],
    supplied: &DeploymentReadbackReceipt,
) -> Result<VerifiedDeploymentReadback, DeploymentReadbackError> {
    validate_receipt_fields(supplied)?;
    if compute_deployment_readback_receipt_sha256(supplied)? != supplied.readback_receipt_sha256 {
        return Err(DeploymentReadbackError::new(
            "readback_receipt_sha256",
            "receipt digest does not match canonical fields",
        ));
    }
    let recomputed = verify_deployment_flash_readback(
        artifact, profile, session, bindings, observed_slices,
        model_binary, firmware_binary, readback_capture,
    )?;
    if recomputed.receipt != *supplied {
        return Err(DeploymentReadbackError::new(
            "readback_receipt",
            "supplied receipt differs from independently recomputed byte-level evidence",
        ));
    }
    Ok(recomputed)
}

pub fn compute_deployment_readback_receipt_sha256(
    receipt: &DeploymentReadbackReceipt,
) -> Result<String, DeploymentReadbackError> {
    validate_receipt_fields(receipt)?;
    let mut canonical = receipt.clone();
    canonical.readback_receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentReadbackError::new("receipt", error.to_string()))?;
    Ok(digest_bytes(RECEIPT_DOMAIN, &bytes))
}

fn validate_session_scope(
    artifact: &VerifiedDeployableArtifact,
    profile: &HardwareProfileManifest,
    session: &DeploymentReadbackSessionManifest,
) -> Result<(), DeploymentReadbackError> {
    if session.schema != SESSION_SCHEMA || session.version != VERSION {
        return Err(DeploymentReadbackError::new(
            "schema_version",
            "unsupported deployment readback session schema/version",
        ));
    }
    validate_hardware_profile(profile)
        .map_err(|error| DeploymentReadbackError::new(
            format!("hardware_profile.{}", error.field),
            error.message,
        ))?;
    if artifact.target_id() != profile.profile_id
        || artifact.hardware_profile_sha256() != profile.profile_sha256
        || session.target_id != artifact.target_id()
        || session.hardware_profile_sha256 != artifact.hardware_profile_sha256()
        || session.artifact_sha256 != artifact.artifact_sha256()
        || session.pipeline_sha256 != artifact.pipeline_sha256()
        || session.build_receipt_sha256 != artifact.build_receipt_sha256()
        || session.board_id != profile.board_id
        || session.board_revision != profile.board_revision
    {
        return Err(DeploymentReadbackError::new(
            "artifact_target_scope",
            "session is not bound to the exact deployable artifact and hardware profile",
        ));
    }
    for (field, value) in [
        ("device_identity_evidence_sha256", session.device_identity_evidence_sha256.as_str()),
        ("flash_tool_executable_sha256", session.flash_tool_executable_sha256.as_str()),
        ("readback_protocol_sha256", session.readback_protocol_sha256.as_str()),
        ("readback_capture_sha256", session.readback_capture_sha256.as_str()),
        ("readback_region_map_sha256", session.readback_region_map_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    for (field, value) in [
        ("device_serial", session.device_serial.as_str()),
        ("flash_probe_id", session.flash_probe_id.as_str()),
        ("session_id", session.session_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentReadbackError::new(field, "must be non-empty"));
        }
    }
    if session.session_sequence == 0 {
        return Err(DeploymentReadbackError::new(
            "session_sequence",
            "must be positive",
        ));
    }
    Ok(())
}

fn validate_bindings(
    bindings: &[FlashRegionBinding],
    model_length: usize,
    firmware_length: usize,
) -> Result<(), DeploymentReadbackError> {
    if bindings.is_empty() || bindings.len() > MAX_REGIONS {
        return Err(DeploymentReadbackError::new(
            "bindings",
            format!("must contain between 1 and {MAX_REGIONS} regions"),
        ));
    }
    if bindings.windows(2).any(|pair| pair[0].region_id >= pair[1].region_id) {
        return Err(DeploymentReadbackError::new(
            "bindings",
            "regions must be strictly sorted by region_id",
        ));
    }
    let mut model_ranges = Vec::new();
    let mut firmware_ranges = Vec::new();
    for (index, binding) in bindings.iter().enumerate() {
        if binding.region_id.trim().is_empty() || binding.byte_length == 0 {
            return Err(DeploymentReadbackError::new(
                format!("bindings[{index}]"),
                "region ID and byte length must be non-empty/positive",
            ));
        }
        binding.flash_address.checked_add(binding.byte_length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("bindings[{index}].flash_address"),
                "flash address range overflows u64",
            )
        })?;
        let source_end = binding.source_offset.checked_add(binding.byte_length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("bindings[{index}].source_offset"),
                "source range overflows u64",
            )
        })?;
        let source_length = match binding.component {
            FlashImageComponent::ModelBinary => model_length as u64,
            FlashImageComponent::FirmwareBinary => firmware_length as u64,
        };
        if source_end > source_length {
            return Err(DeploymentReadbackError::new(
                format!("bindings[{index}].source_offset"),
                "source range is outside the selected binary",
            ));
        }
        match binding.component {
            FlashImageComponent::ModelBinary => model_ranges.push((binding.source_offset, source_end)),
            FlashImageComponent::FirmwareBinary => firmware_ranges.push((binding.source_offset, source_end)),
        }
    }
    validate_source_coverage("model_binary", &mut model_ranges, model_length as u64)?;
    validate_source_coverage("firmware_binary", &mut firmware_ranges, firmware_length as u64)?;
    Ok(())
}

fn validate_source_coverage(
    field: &str,
    ranges: &mut Vec<(u64, u64)>,
    expected_length: u64,
) -> Result<(), DeploymentReadbackError> {
    ranges.sort_unstable_by_key(|range| range.0);
    if ranges.is_empty() {
        return Err(DeploymentReadbackError::new(field, "no region covers this binary"));
    }
    let mut cursor = 0u64;
    for (start, end) in ranges.iter().copied() {
        if start != cursor || end <= start {
            return Err(DeploymentReadbackError::new(
                field,
                "source coverage contains a gap, overlap or empty range",
            ));
        }
        cursor = end;
    }
    if cursor != expected_length {
        return Err(DeploymentReadbackError::new(
            field,
            "flash regions do not cover the complete source binary",
        ));
    }
    Ok(())
}

fn validate_observed_slices(
    slices: &[ReadbackRegionSlice],
) -> Result<(), DeploymentReadbackError> {
    if slices.is_empty() || slices.len() > MAX_REGIONS {
        return Err(DeploymentReadbackError::new(
            "observed_slices",
            format!("must contain between 1 and {MAX_REGIONS} regions"),
        ));
    }
    if slices.windows(2).any(|pair| pair[0].region_id >= pair[1].region_id) {
        return Err(DeploymentReadbackError::new(
            "observed_slices",
            "observed regions must be strictly sorted by region_id",
        ));
    }
    for (index, slice) in slices.iter().enumerate() {
        if slice.region_id.trim().is_empty() || slice.byte_length == 0 {
            return Err(DeploymentReadbackError::new(
                format!("observed_slices[{index}]"),
                "region ID and byte length must be non-empty/positive",
            ));
        }
        slice.observed_flash_address.checked_add(slice.byte_length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("observed_slices[{index}].observed_flash_address"),
                "flash address range overflows u64",
            )
        })?;
        slice.capture_offset.checked_add(slice.byte_length).ok_or_else(|| {
            DeploymentReadbackError::new(
                format!("observed_slices[{index}].capture_offset"),
                "capture range overflows u64",
            )
        })?;
    }
    Ok(())
}

fn validate_capture_coverage(
    slices: &[ReadbackRegionSlice],
    capture_length: usize,
) -> Result<(), DeploymentReadbackError> {
    let mut ranges: Vec<(u64, u64)> = slices
        .iter()
        .map(|slice| (slice.capture_offset, slice.capture_offset + slice.byte_length))
        .collect();
    ranges.sort_unstable_by_key(|range| range.0);
    let mut cursor = 0u64;
    for (start, end) in ranges {
        if start != cursor || end <= start {
            return Err(DeploymentReadbackError::new(
                "capture_coverage",
                "capture regions contain a gap, overlap or empty range",
            ));
        }
        cursor = end;
    }
    if cursor != capture_length as u64 {
        return Err(DeploymentReadbackError::new(
            "capture_coverage",
            "regions do not account for the entire raw capture payload",
        ));
    }
    Ok(())
}

fn validate_flash_address_non_overlap(
    bindings: &[FlashRegionBinding],
) -> Result<(), DeploymentReadbackError> {
    let mut ranges: Vec<(u64, u64)> = bindings
        .iter()
        .map(|binding| (binding.flash_address, binding.flash_address + binding.byte_length))
        .collect();
    ranges.sort_unstable_by_key(|range| range.0);
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(DeploymentReadbackError::new(
            "flash_addresses",
            "expected flash-region address ranges overlap",
        ));
    }
    Ok(())
}

fn validate_receipt_fields(
    receipt: &DeploymentReadbackReceipt,
) -> Result<(), DeploymentReadbackError> {
    if receipt.schema != RECEIPT_SCHEMA || receipt.version != VERSION {
        return Err(DeploymentReadbackError::new(
            "schema_version",
            "unsupported deployment readback receipt schema/version",
        ));
    }
    for (field, value) in [
        ("artifact_sha256", receipt.artifact_sha256.as_str()),
        ("pipeline_sha256", receipt.pipeline_sha256.as_str()),
        ("build_receipt_sha256", receipt.build_receipt_sha256.as_str()),
        ("model_binary_sha256", receipt.model_binary_sha256.as_str()),
        ("firmware_binary_sha256", receipt.firmware_binary_sha256.as_str()),
        ("hardware_profile_sha256", receipt.hardware_profile_sha256.as_str()),
        ("device_identity_evidence_sha256", receipt.device_identity_evidence_sha256.as_str()),
        ("flash_tool_executable_sha256", receipt.flash_tool_executable_sha256.as_str()),
        ("readback_protocol_sha256", receipt.readback_protocol_sha256.as_str()),
        ("readback_capture_sha256", receipt.readback_capture_sha256.as_str()),
        ("readback_region_map_sha256", receipt.readback_region_map_sha256.as_str()),
        ("readback_receipt_sha256", receipt.readback_receipt_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    for (field, value) in [
        ("target_id", receipt.target_id.as_str()),
        ("board_id", receipt.board_id.as_str()),
        ("board_revision", receipt.board_revision.as_str()),
        ("device_serial", receipt.device_serial.as_str()),
        ("flash_probe_id", receipt.flash_probe_id.as_str()),
        ("session_id", receipt.session_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentReadbackError::new(field, "must be non-empty"));
        }
    }
    if receipt.session_sequence == 0 || receipt.regions.is_empty() || receipt.regions.len() > MAX_REGIONS {
        return Err(DeploymentReadbackError::new(
            "session_or_regions",
            "session sequence must be positive and region count within the supported range",
        ));
    }
    if receipt.regions.windows(2).any(|pair| pair[0].region_id >= pair[1].region_id) {
        return Err(DeploymentReadbackError::new(
            "regions",
            "receipt regions must be strictly sorted by region_id",
        ));
    }
    for (index, region) in receipt.regions.iter().enumerate() {
        if region.region_id.trim().is_empty() || region.byte_length == 0 {
            return Err(DeploymentReadbackError::new(
                format!("regions[{index}]"),
                "region ID and length must be non-empty/positive",
            ));
        }
        require_sha256(
            &format!("regions[{index}].expected_source_sha256"),
            &region.expected_source_sha256,
        )?;
        require_sha256(
            &format!("regions[{index}].readback_bytes_sha256"),
            &region.readback_bytes_sha256,
        )?;
    }
    Ok(())
}

fn require_sha256(field: &str, value: &str) -> Result<(), DeploymentReadbackError> {
    if !is_sha256(value) {
        return Err(DeploymentReadbackError::new(
            field,
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn to_usize(value: u64, field: &str) -> Result<usize, DeploymentReadbackError> {
    usize::try_from(value)
        .map_err(|_| DeploymentReadbackError::new(field, "value does not fit host address space"))
}

fn sha256_raw(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(SHA256_HEX_LEN);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let mut encoded = String::with_capacity(SHA256_HEX_LEN);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn measurement_binding_sha256(
    receipt: &ReadbackBoundDeploymentMeasurementReceipt,
) -> Result<String, DeploymentReadbackError> {
    let mut canonical = receipt.clone();
    canonical.binding_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentReadbackError::new("measurement_receipt", error.to_string()))?;
    Ok(digest_bytes(MEASUREMENT_DOMAIN, &bytes))
}

/// Create a measurement receipt that explicitly depends on the verified readback capability.
pub fn create_readback_bound_measurement_receipt(
    artifact: &VerifiedDeployableArtifact,
    readback: &VerifiedDeploymentReadback,
    registry: &ObjectiveRegistry,
    observation: ObjectiveObservation,
) -> Result<ReadbackBoundDeploymentMeasurementReceipt, DeploymentReadbackError> {
    if readback.artifact_sha256() != artifact.artifact_sha256()
        || readback.target_id() != artifact.target_id()
        || readback.hardware_profile_sha256() != artifact.hardware_profile_sha256()
    {
        return Err(DeploymentReadbackError::new(
            "readback_scope",
            "verified readback does not belong to the supplied deployment artifact",
        ));
    }
    let measurement_receipt = create_deployment_measurement_receipt(artifact, registry, observation)
        .map_err(map_artifact_error)?;
    let mut receipt = ReadbackBoundDeploymentMeasurementReceipt {
        schema: MEASUREMENT_SCHEMA.into(),
        version: VERSION.into(),
        artifact_sha256: artifact.artifact_sha256().into(),
        readback_receipt_sha256: readback.readback_receipt_sha256().into(),
        measurement_receipt,
        binding_sha256: "0".repeat(SHA256_HEX_LEN),
    };
    receipt.binding_sha256 = measurement_binding_sha256(&receipt)?;
    Ok(receipt)
}

pub fn verify_readback_bound_measurement_receipt(
    artifact: &VerifiedDeployableArtifact,
    readback: &VerifiedDeploymentReadback,
    registry: &ObjectiveRegistry,
    receipt: &ReadbackBoundDeploymentMeasurementReceipt,
) -> Result<ScoredMeasurement, DeploymentReadbackError> {
    if receipt.schema != MEASUREMENT_SCHEMA || receipt.version != VERSION {
        return Err(DeploymentReadbackError::new(
            "schema_version",
            "unsupported readback-bound measurement schema/version",
        ));
    }
    require_sha256("artifact_sha256", &receipt.artifact_sha256)?;
    require_sha256("readback_receipt_sha256", &receipt.readback_receipt_sha256)?;
    require_sha256("binding_sha256", &receipt.binding_sha256)?;
    if receipt.artifact_sha256 != artifact.artifact_sha256()
        || receipt.readback_receipt_sha256 != readback.readback_receipt_sha256()
        || readback.artifact_sha256() != artifact.artifact_sha256()
    {
        return Err(DeploymentReadbackError::new(
            "readback_scope",
            "measurement does not bind to this exact artifact and verified readback",
        ));
    }
    if measurement_binding_sha256(receipt)? != receipt.binding_sha256 {
        return Err(DeploymentReadbackError::new(
            "binding_sha256",
            "does not match canonical measurement binding",
        ));
    }
    verify_deployment_measurement_receipt(artifact, registry, &receipt.measurement_receipt)
        .map_err(map_artifact_error)
}

fn map_artifact_error(error: DeploymentArtifactError) -> DeploymentReadbackError {
    DeploymentReadbackError::new("deployment_artifact", error.to_string())
}

const SESSION_LEDGER_SCHEMA: &str = "symthaea.nas.deployment-readback-session-ledger";
const SESSION_RECORD_DOMAIN: &[u8] = b"symthaea.nas.deployment-readback-session-record.v0.1";
const SESSION_LEDGER_DOMAIN: &[u8] = b"symthaea.nas.deployment-readback-session-ledger.v0.1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReadbackSessionRecord {
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub device_serial: String,
    pub session_sequence: u64,
    pub session_id: String,
    pub artifact_sha256: String,
    pub readback_receipt_sha256: String,
    pub previous_record_sha256: Option<String>,
    pub record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReadbackSessionLedger {
    pub schema: String,
    pub version: String,
    pub target_id: String,
    pub hardware_profile_sha256: String,
    pub device_serial: String,
    pub records: Vec<DeploymentReadbackSessionRecord>,
    pub ledger_sha256: String,
}

/// Create an empty per-device ledger for verified readback successes.
pub fn create_deployment_readback_session_ledger(
    target_id: impl Into<String>,
    hardware_profile_sha256: impl Into<String>,
    device_serial: impl Into<String>,
) -> Result<DeploymentReadbackSessionLedger, DeploymentReadbackError> {
    let mut ledger = DeploymentReadbackSessionLedger {
        schema: SESSION_LEDGER_SCHEMA.into(),
        version: VERSION.into(),
        target_id: target_id.into(),
        hardware_profile_sha256: hardware_profile_sha256.into(),
        device_serial: device_serial.into(),
        records: Vec::new(),
        ledger_sha256: "0".repeat(SHA256_HEX_LEN),
    };
    validate_session_ledger_fields(&ledger)?;
    ledger.ledger_sha256 = compute_deployment_readback_session_ledger_sha256(&ledger)?;
    Ok(ledger)
}

pub fn compute_deployment_readback_session_ledger_sha256(
    ledger: &DeploymentReadbackSessionLedger,
) -> Result<String, DeploymentReadbackError> {
    validate_session_ledger_fields(ledger)?;
    let mut canonical = ledger.clone();
    canonical.ledger_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentReadbackError::new("session_ledger", error.to_string()))?;
    Ok(digest_bytes(SESSION_LEDGER_DOMAIN, &bytes))
}

/// Append only a capability produced by byte-level readback verification. Caller-supplied
/// success booleans or deserialized receipts cannot be appended through this API.
pub fn append_verified_readback_session(
    ledger: &mut DeploymentReadbackSessionLedger,
    readback: &VerifiedDeploymentReadback,
) -> Result<(), DeploymentReadbackError> {
    verify_deployment_readback_session_ledger(ledger)?;
    let receipt = readback.receipt();
    if receipt.target_id != ledger.target_id
        || receipt.hardware_profile_sha256 != ledger.hardware_profile_sha256
        || receipt.device_serial != ledger.device_serial
    {
        return Err(DeploymentReadbackError::new(
            "session_scope",
            "verified readback belongs to a different target, profile or physical device",
        ));
    }
    if ledger.records.iter().any(|record| record.session_id == receipt.session_id) {
        return Err(DeploymentReadbackError::new(
            "session_id",
            "readback session ID has already been consumed by this ledger",
        ));
    }
    if ledger.records.last().is_some_and(|last| receipt.session_sequence <= last.session_sequence) {
        return Err(DeploymentReadbackError::new(
            "session_sequence",
            "readback session sequence must strictly increase for this device ledger",
        ));
    }

    let mut record = DeploymentReadbackSessionRecord {
        target_id: receipt.target_id.clone(),
        hardware_profile_sha256: receipt.hardware_profile_sha256.clone(),
        device_serial: receipt.device_serial.clone(),
        session_sequence: receipt.session_sequence,
        session_id: receipt.session_id.clone(),
        artifact_sha256: receipt.artifact_sha256.clone(),
        readback_receipt_sha256: receipt.readback_receipt_sha256.clone(),
        previous_record_sha256: ledger.records.last().map(|last| last.record_sha256.clone()),
        record_sha256: "0".repeat(SHA256_HEX_LEN),
    };
    record.record_sha256 = compute_deployment_readback_session_record_sha256(&record)?;
    ledger.records.push(record);
    ledger.ledger_sha256 = compute_deployment_readback_session_ledger_sha256(ledger)?;
    Ok(())
}

pub fn compute_deployment_readback_session_record_sha256(
    record: &DeploymentReadbackSessionRecord,
) -> Result<String, DeploymentReadbackError> {
    validate_session_record_fields(record)?;
    let mut canonical = record.clone();
    canonical.record_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DeploymentReadbackError::new("session_record", error.to_string()))?;
    Ok(digest_bytes(SESSION_RECORD_DOMAIN, &bytes))
}

pub fn verify_deployment_readback_session_ledger(
    ledger: &DeploymentReadbackSessionLedger,
) -> Result<(), DeploymentReadbackError> {
    validate_session_ledger_fields(ledger)?;
    if !is_sha256(&ledger.ledger_sha256)
        || compute_deployment_readback_session_ledger_sha256(ledger)? != ledger.ledger_sha256
    {
        return Err(DeploymentReadbackError::new(
            "ledger_sha256",
            "does not match the canonical readback session ledger",
        ));
    }
    Ok(())
}

fn validate_session_ledger_fields(
    ledger: &DeploymentReadbackSessionLedger,
) -> Result<(), DeploymentReadbackError> {
    if ledger.schema != SESSION_LEDGER_SCHEMA || ledger.version != VERSION {
        return Err(DeploymentReadbackError::new(
            "schema_version",
            "unsupported readback session ledger schema/version",
        ));
    }
    if ledger.target_id.trim().is_empty() || ledger.device_serial.trim().is_empty() {
        return Err(DeploymentReadbackError::new(
            "ledger_identity",
            "target ID and device serial must be non-empty",
        ));
    }
    require_sha256("hardware_profile_sha256", &ledger.hardware_profile_sha256)?;
    require_sha256("ledger_sha256", &ledger.ledger_sha256)?;

    let mut session_ids = HashSet::with_capacity(ledger.records.len());
    let mut previous_sequence = 0u64;
    let mut previous_digest: Option<&str> = None;
    for (index, record) in ledger.records.iter().enumerate() {
        validate_session_record_fields(record).map_err(|error| {
            DeploymentReadbackError::new(format!("records[{index}].{}", error.field), error.message)
        })?;
        if record.target_id != ledger.target_id
            || record.hardware_profile_sha256 != ledger.hardware_profile_sha256
            || record.device_serial != ledger.device_serial
        {
            return Err(DeploymentReadbackError::new(
                format!("records[{index}].scope"),
                "record target/profile/device differs from its ledger",
            ));
        }
        if record.session_sequence <= previous_sequence {
            return Err(DeploymentReadbackError::new(
                format!("records[{index}].session_sequence"),
                "session sequence must strictly increase",
            ));
        }
        if !session_ids.insert(record.session_id.as_str()) {
            return Err(DeploymentReadbackError::new(
                format!("records[{index}].session_id"),
                "duplicate session ID in the verified readback ledger",
            ));
        }
        if record.previous_record_sha256.as_deref() != previous_digest {
            return Err(DeploymentReadbackError::new(
                format!("records[{index}].previous_record_sha256"),
                "hash-chain predecessor does not match the previous record",
            ));
        }
        if compute_deployment_readback_session_record_sha256(record)? != record.record_sha256 {
            return Err(DeploymentReadbackError::new(
                format!("records[{index}].record_sha256"),
                "does not match canonical session-record fields",
            ));
        }
        previous_sequence = record.session_sequence;
        previous_digest = Some(record.record_sha256.as_str());
    }
    Ok(())
}

fn validate_session_record_fields(
    record: &DeploymentReadbackSessionRecord,
) -> Result<(), DeploymentReadbackError> {
    for (field, value) in [
        ("target_id", record.target_id.as_str()),
        ("device_serial", record.device_serial.as_str()),
        ("session_id", record.session_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(DeploymentReadbackError::new(field, "must be non-empty"));
        }
    }
    if record.session_sequence == 0 {
        return Err(DeploymentReadbackError::new(
            "session_sequence",
            "must be positive",
        ));
    }
    for (field, value) in [
        ("hardware_profile_sha256", record.hardware_profile_sha256.as_str()),
        ("artifact_sha256", record.artifact_sha256.as_str()),
        ("readback_receipt_sha256", record.readback_receipt_sha256.as_str()),
        ("record_sha256", record.record_sha256.as_str()),
    ] {
        require_sha256(field, value)?;
    }
    if let Some(previous) = record.previous_record_sha256.as_deref() {
        require_sha256("previous_record_sha256", previous)?;
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment_artifact::{
        compute_deployment_build_receipt_sha256, compute_deployment_pipeline_sha256,
        verify_deployable_artifact, BuildDisposition, DeploymentBuildReceipt,
        DeploymentPipelineManifest, PipelineComponent, PipelineComponentKind,
    };
    use crate::hardware_objectives::{
        compute_hardware_profile_sha256, compute_objective_registry_sha256,
        HardwareProfileManifest, ObjectiveDefinition, ObjectiveDirection,
        ObjectiveNormalization, ObjectiveRegistry, ObjectiveSource,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    struct Fixture {
        artifact: VerifiedDeployableArtifact,
        profile: HardwareProfileManifest,
        model: Vec<u8>,
        firmware: Vec<u8>,
        bindings: Vec<FlashRegionBinding>,
        slices: Vec<ReadbackRegionSlice>,
        capture: Vec<u8>,
        session: DeploymentReadbackSessionManifest,
        registry: ObjectiveRegistry,
    }

    fn fixture() -> Fixture {
        let model = vec![1, 2, 3, 4];
        let firmware = vec![10, 11, 12, 13, 14, 15];
        let mut profile = HardwareProfileManifest {
            profile_id: "board-profile-01".into(),
            device_model: "nRF5340".into(),
            board_id: "board-01".into(),
            board_revision: "rev-a".into(),
            firmware_sha256: sha('b'),
            runtime_sha256: sha('c'),
            compiler_toolchain_sha256: sha('d'),
            quantization_spec_sha256: sha('e'),
            measurement_protocol_sha256: sha('f'),
            instrument_model: "PPK2".into(),
            instrument_id: "ppk2-01".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = compute_hardware_profile_sha256(&profile).unwrap();

        let mut registry = ObjectiveRegistry {
            schema: "symthaea.nas.objective-registry".into(),
            version: "0.1.0".into(),
            benchmark_id: "readback-test-benchmark".into(),
            objectives: vec![ObjectiveDefinition {
                objective_id: "latency_ms".into(),
                value_unit: "ms".into(),
                direction: ObjectiveDirection::Minimize,
                source: ObjectiveSource::PhysicalDeviceMeasurement,
                normalization: ObjectiveNormalization { raw_min: 0.0, raw_max: 100.0 },
                measurement_protocol_sha256: profile.measurement_protocol_sha256.clone(),
                hardware_profile_sha256: Some(profile.profile_sha256.clone()),
                measurement_cost_unit: "queries".into(),
                measurement_cost_model_sha256: sha('a'),
            }],
            hardware_profiles: vec![profile.clone()],
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = compute_objective_registry_sha256(&registry).unwrap();

        let kinds = vec![
            (PipelineComponentKind::ArchitectureGraph, sha('1')),
            (PipelineComponentKind::QuantizationSpec, profile.quantization_spec_sha256.clone()),
            (PipelineComponentKind::CalibrationDataset, sha('2')),
            (PipelineComponentKind::CompilerToolchain, profile.compiler_toolchain_sha256.clone()),
            (PipelineComponentKind::CompilerFlags, sha('3')),
            (PipelineComponentKind::OperatorLibrary, sha('4')),
            (PipelineComponentKind::MemoryPlanner, sha('5')),
            (PipelineComponentKind::Firmware, profile.firmware_sha256.clone()),
            (PipelineComponentKind::Runtime, profile.runtime_sha256.clone()),
            (PipelineComponentKind::BoardSupportPackage, sha('6')),
            (PipelineComponentKind::LinkerScript, sha('7')),
            (PipelineComponentKind::BuildEnvironment, sha('8')),
        ];
        let mut pipeline = DeploymentPipelineManifest {
            schema: "symthaea.nas.deployment-pipeline".into(),
            version: "0.1.0".into(),
            benchmark_id: registry.benchmark_id.clone(),
            candidate_id: "candidate-01".into(),
            candidate_config_sha256: sha('9'),
            target_id: profile.profile_id.clone(),
            hardware_profile_sha256: profile.profile_sha256.clone(),
            objective_registry_sha256: registry.registry_sha256.clone(),
            components: kinds.into_iter().map(|(kind, digest)| PipelineComponent {
                kind, version: "v1".into(), sha256: digest,
            }).collect(),
            pipeline_sha256: String::new(),
        };
        pipeline.pipeline_sha256 = compute_deployment_pipeline_sha256(&pipeline).unwrap();

        let mut build = DeploymentBuildReceipt {
            schema: "symthaea.nas.deployment-build-receipt".into(),
            version: "0.1.0".into(),
            pipeline_sha256: pipeline.pipeline_sha256.clone(),
            disposition: BuildDisposition::Built,
            build_log_sha256: Some(sha('a')),
            model_binary_sha256: Some(sha256_raw(&model)),
            firmware_binary_sha256: Some(sha256_raw(&firmware)),
            linker_map_sha256: Some(sha('b')),
            flash_report_sha256: Some(sha('c')),
            ram_report_sha256: Some(sha('d')),
            reason_code: None,
            build_receipt_sha256: String::new(),
        };
        build.build_receipt_sha256 = compute_deployment_build_receipt_sha256(&build).unwrap();
        let artifact = verify_deployable_artifact(&pipeline, &profile, &registry, &build).unwrap();

        let bindings = vec![
            FlashRegionBinding {
                region_id: "firmware-a".into(),
                component: FlashImageComponent::FirmwareBinary,
                flash_address: 0x0000_0000,
                source_offset: 0,
                byte_length: 3,
            },
            FlashRegionBinding {
                region_id: "firmware-b".into(),
                component: FlashImageComponent::FirmwareBinary,
                flash_address: 0x0000_0003,
                source_offset: 3,
                byte_length: 3,
            },
            FlashRegionBinding {
                region_id: "model-a".into(),
                component: FlashImageComponent::ModelBinary,
                flash_address: 0x0008_0000,
                source_offset: 0,
                byte_length: 4,
            },
        ];
        let slices = vec![
            ReadbackRegionSlice {
                region_id: "firmware-a".into(),
                observed_flash_address: 0x0000_0000,
                capture_offset: 0,
                byte_length: 3,
            },
            ReadbackRegionSlice {
                region_id: "firmware-b".into(),
                observed_flash_address: 0x0000_0003,
                capture_offset: 3,
                byte_length: 3,
            },
            ReadbackRegionSlice {
                region_id: "model-a".into(),
                observed_flash_address: 0x0008_0000,
                capture_offset: 6,
                byte_length: 4,
            },
        ];
        let mut capture = firmware.clone();
        capture.extend_from_slice(&model);

        let mut session = DeploymentReadbackSessionManifest {
            schema: SESSION_SCHEMA.into(),
            version: VERSION.into(),
            artifact_sha256: artifact.artifact_sha256().into(),
            pipeline_sha256: artifact.pipeline_sha256().into(),
            build_receipt_sha256: artifact.build_receipt_sha256().into(),
            target_id: artifact.target_id().into(),
            hardware_profile_sha256: artifact.hardware_profile_sha256().into(),
            board_id: profile.board_id.clone(),
            board_revision: profile.board_revision.clone(),
            device_serial: "device-serial-01".into(),
            device_identity_evidence_sha256: sha('a'),
            flash_probe_id: "probe-01".into(),
            flash_tool_executable_sha256: sha('b'),
            readback_protocol_sha256: sha('c'),
            readback_capture_sha256: sha256_raw(&capture),
            readback_region_map_sha256: String::new(),
            session_id: "session-01".into(),
            session_sequence: 1,
        };
        session.readback_region_map_sha256 = compute_readback_region_map_sha256(&slices).unwrap();

        Fixture {
            artifact, profile, model, firmware, bindings, slices, capture, session, registry,
        }
    }

    #[test]
    fn exact_readback_bytes_issue_a_verified_capability() {
        let f = fixture();
        let verified = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        assert_eq!(verified.artifact_sha256(), f.artifact.artifact_sha256());
        assert_eq!(verified.receipt().regions.len(), 3);
    }

    #[test]
    fn one_changed_firmware_byte_is_rejected() {
        let f = fixture();
        let mut changed = f.capture.clone();
        changed[1] ^= 0xff;
        let mut session = f.session.clone();
        session.readback_capture_sha256 = sha256_raw(&changed);
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &changed,
        ).is_err());
    }

    #[test]
    fn wrong_observed_address_is_rejected_even_with_rehashed_region_map() {
        let f = fixture();
        let mut slices = f.slices.clone();
        slices[0].observed_flash_address += 1;
        let mut session = f.session.clone();
        session.readback_region_map_sha256 = compute_readback_region_map_sha256(&slices).unwrap();
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &session, &f.bindings, &slices,
            &f.model, &f.firmware, &f.capture,
        ).is_err());
    }

    #[test]
    fn truncated_capture_is_rejected() {
        let f = fixture();
        let mut capture = f.capture.clone();
        capture.pop();
        let mut session = f.session.clone();
        session.readback_capture_sha256 = sha256_raw(&capture);
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &capture,
        ).is_err());
    }

    #[test]
    fn missing_source_segment_and_overlapping_flash_regions_are_rejected() {
        let f = fixture();
        let mut bindings = f.bindings.clone();
        bindings[0].source_offset = 1;
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).is_err());

        let mut bindings = f.bindings.clone();
        bindings[1].flash_address = 0;
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).is_err());
    }

    #[test]
    fn different_board_profile_or_artifact_is_rejected() {
        let f = fixture();
        let mut session = f.session.clone();
        session.board_revision = "rev-b".into();
        assert!(verify_deployment_flash_readback(
            &f.artifact, &f.profile, &session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).is_err());
    }

    #[test]
    fn receipt_verifier_recomputes_from_original_bytes() {
        let f = fixture();
        let verified = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let replayed = verify_deployment_readback_receipt(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture, verified.receipt(),
        ).unwrap();
        assert_eq!(verified, replayed);
    }

    #[test]
    fn physical_measurement_must_reference_the_same_readback_capability() {
        let f = fixture();
        let readback = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let observation = ObjectiveObservation {
            candidate_id: "candidate-01".into(),
            candidate_config_sha256: sha('9'),
            objective_id: "latency_ms".into(),
            registry_sha256: f.registry.registry_sha256.clone(),
            measurement_protocol_sha256: f.profile.measurement_protocol_sha256.clone(),
            hardware_profile_sha256: Some(f.profile.profile_sha256.clone()),
            outcome: crate::hardware_objectives::MeasurementOutcome::Measured {
                raw_value_bits: 12.5_f64.to_bits(),
                replicate_id: "rep-01".into(),
                sample_count: 10,
                source_artifact_sha256: sha('e'),
                cost: crate::hardware_objectives::MeasurementCost {
                    resource_unit: "queries".into(),
                    units: 10,
                },
            },
        };
        let bound = create_readback_bound_measurement_receipt(
            &f.artifact, &readback, &f.registry, observation,
        ).unwrap();
        let scored = verify_readback_bound_measurement_receipt(
            &f.artifact, &readback, &f.registry, &bound,
        ).unwrap();
        assert!(matches!(scored, ScoredMeasurement::Measured { .. }));
    }

    #[test]
    fn only_verified_readback_capabilities_can_advance_the_device_session_ledger() {
        let f = fixture();
        let readback = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let mut ledger = create_deployment_readback_session_ledger(
            f.profile.profile_id.clone(),
            f.profile.profile_sha256.clone(),
            "device-serial-01",
        ).unwrap();
        append_verified_readback_session(&mut ledger, &readback).unwrap();
        verify_deployment_readback_session_ledger(&ledger).unwrap();
        assert_eq!(ledger.records.len(), 1);

        assert!(append_verified_readback_session(&mut ledger, &readback).is_err());
        assert_eq!(ledger.records.len(), 1);
    }

    #[test]
    fn session_ledger_rejects_repeated_ids_and_non_monotonic_sequences() {
        let f = fixture();
        let first = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let mut ledger = create_deployment_readback_session_ledger(
            f.profile.profile_id.clone(),
            f.profile.profile_sha256.clone(),
            "device-serial-01",
        ).unwrap();
        append_verified_readback_session(&mut ledger, &first).unwrap();

        let mut reused = f.session.clone();
        reused.session_id = "session-02".into();
        let reused = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &reused, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        assert!(append_verified_readback_session(&mut ledger, &reused).is_err());
    }

    #[test]
    fn session_ledger_rejects_tampered_chain_even_after_outer_state_is_serialized() {
        let f = fixture();
        let readback = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let mut ledger = create_deployment_readback_session_ledger(
            f.profile.profile_id.clone(),
            f.profile.profile_sha256.clone(),
            "device-serial-01",
        ).unwrap();
        append_verified_readback_session(&mut ledger, &readback).unwrap();
        ledger.records[0].artifact_sha256 = sha('0');
        assert!(verify_deployment_readback_session_ledger(&ledger).is_err());
    }

    #[test]
    fn nested_unknown_fields_are_rejected_on_receipt_deserialization() {
        let f = fixture();
        let verified = verify_deployment_flash_readback(
            &f.artifact, &f.profile, &f.session, &f.bindings, &f.slices,
            &f.model, &f.firmware, &f.capture,
        ).unwrap();
        let encoded = serde_json::to_string(verified.receipt()).unwrap();
        let injected = encoded.replacen(
            "\"readback_receipt_sha256\"",
            "\"unknown\":true,\"readback_receipt_sha256\"",
            1,
        );
        assert!(serde_json::from_str::<DeploymentReadbackReceipt>(&injected).is_err());
    }
}
