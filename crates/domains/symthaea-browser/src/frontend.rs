// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frontend-specific browser evidence capture.
//!
//! This module joins three independent browser observations: an accessibility
//! snapshot, a bounded layout snapshot, and a viewport PNG screenshot. Artifacts are
//! content-addressed so a benchmark runner can persist them separately and
//! verify that the receipt still refers to the exact captured bytes.
//!
//! The layout probe reads only fixed, browser-owned properties and a bounded
//! set of semantic elements. It does not execute page-supplied selectors or
//! extract arbitrary element text. Captures are sequential rather than an
//! atomic browser transaction; SFEB tasks must use deterministic fixtures.

use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{CdpSession, PageObservation};

const MAX_LAYOUT_ELEMENTS: usize = 256;
const MAX_VIEWPORT_DIMENSION: u32 = 32_768;
const MAX_SCREENSHOT_DIMENSION: u32 = 4_096;
const MAX_SCREENSHOT_BYTES: usize = 32 * 1024 * 1024;
const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

const LAYOUT_PROBE: &str = r#"(() => {
  const maxElements = 256;
  const selector = [
    'main', 'header', 'nav', 'footer', 'aside', 'section', 'article',
    'h1', 'h2', 'h3', 'button', 'a[href]', 'input', 'textarea', 'select',
    '[role]', '[data-sfeb-id]'
  ].join(',');
  const round = value => Math.round(value * 100) / 100;
  const elements = Array.from(document.querySelectorAll(selector))
    .slice(0, maxElements)
    .map(element => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      const rawId = element.getAttribute('data-sfeb-id');
      const rawRole = element.getAttribute('role');
      return {
        data_sfeb_id: rawId ? rawId.slice(0, 128) : null,
        tag_name: element.tagName.toLowerCase().slice(0, 32),
        role: rawRole ? rawRole.slice(0, 64) : null,
        rect: {
          x: round(rect.left + window.scrollX),
          y: round(rect.top + window.scrollY),
          width: round(rect.width),
          height: round(rect.height)
        },
        visible: rect.width > 0 && rect.height > 0 &&
          style.display !== 'none' && style.visibility !== 'hidden',
        display: style.display.slice(0, 32),
        position: style.position.slice(0, 32),
        overflow_x: style.overflowX.slice(0, 32),
        overflow_y: style.overflowY.slice(0, 32)
      };
    });
  const root = document.documentElement;
  const scrollWidth = root ? root.scrollWidth : 0;
  const scrollHeight = root ? root.scrollHeight : 0;
  return JSON.stringify({
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      device_scale_factor: window.devicePixelRatio
    },
    document: {
      scroll_width: scrollWidth,
      scroll_height: scrollHeight,
      has_horizontal_overflow: scrollWidth > window.innerWidth,
      has_vertical_overflow: scrollHeight > window.innerHeight
    },
    elements
  });
})()"#;

/// Viewport dimensions reported by the rendered browser page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendViewport {
    pub width: u32,
    pub height: u32,
    pub device_scale_factor: f64,
}

/// Document-level overflow observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendDocumentMetrics {
    pub scroll_width: u32,
    pub scroll_height: u32,
    pub has_horizontal_overflow: bool,
    pub has_vertical_overflow: bool,
}

/// Document-coordinate rectangle for a semantic page element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendLayoutRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Bounded geometry and computed layout properties for one semantic element.
///
/// `data_sfeb_id` is the stable bridge from a benchmark fixture's component
/// identity to the rendered DOM. It is optional so uninstrumented pages remain
/// observable, but SFEB reference fixtures should annotate important regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendLayoutElement {
    pub data_sfeb_id: Option<String>,
    pub tag_name: String,
    pub role: Option<String>,
    pub rect: FrontendLayoutRect,
    pub visible: bool,
    pub display: String,
    pub position: String,
    pub overflow_x: String,
    pub overflow_y: String,
}

/// Bounded layout observation captured from the live rendered page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendLayoutSnapshot {
    pub viewport: FrontendViewport,
    pub document: FrontendDocumentMetrics,
    pub elements: Vec<FrontendLayoutElement>,
}

impl FrontendLayoutSnapshot {
    /// Reject malformed or unexpectedly unbounded browser output.
    pub fn validate(&self) -> Result<()> {
        if self.viewport.width == 0
            || self.viewport.height == 0
            || self.viewport.width > MAX_VIEWPORT_DIMENSION
            || self.viewport.height > MAX_VIEWPORT_DIMENSION
        {
            bail!("frontend viewport dimensions are outside the supported range");
        }
        if !self.viewport.device_scale_factor.is_finite()
            || !(0.5..=4.0).contains(&self.viewport.device_scale_factor)
        {
            bail!("frontend device scale factor is outside the supported range");
        }
        if self.document.scroll_width > MAX_VIEWPORT_DIMENSION * 4
            || self.document.scroll_height > MAX_VIEWPORT_DIMENSION * 4
        {
            bail!("frontend document dimensions exceed the supported range");
        }
        if self.elements.len() > MAX_LAYOUT_ELEMENTS {
            bail!("frontend layout element count exceeds the capture limit");
        }

        let mut seen_source_ids = BTreeSet::new();
        for element in &self.elements {
            if element.tag_name.is_empty()
                || element.tag_name.len() > 32
                || !element
                    .tag_name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                || element.role.as_ref().is_some_and(|value| value.len() > 64)
            {
                bail!("frontend layout element contains an invalid identifier");
            }
            if let Some(source_id) = &element.data_sfeb_id {
                if source_id.is_empty()
                    || source_id.len() > 128
                    || !source_id.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')
                    })
                    || !seen_source_ids.insert(source_id)
                {
                    bail!("frontend source identifiers must be unique and use a safe token alphabet");
                }
            }
            for value in [
                element.rect.x,
                element.rect.y,
                element.rect.width,
                element.rect.height,
            ] {
                if !value.is_finite() || value.abs() > 131_072.0 {
                    bail!("frontend layout rectangle contains an invalid coordinate");
                }
            }
            if element.rect.width < 0.0 || element.rect.height < 0.0 {
                bail!("frontend layout rectangle has negative dimensions");
            }
            for value in [
                &element.display,
                &element.position,
                &element.overflow_x,
                &element.overflow_y,
            ] {
                if value.len() > 32 {
                    bail!("frontend computed style value exceeds the capture limit");
                }
            }
        }
        Ok(())
    }
}

/// Inputs supplied by the benchmark orchestrator for one capture.
///
/// Environment identity is a digest of a separately persisted, canonical
/// environment manifest (toolchain, resolved dependencies, browser build,
/// OS image, fonts, viewport policy, fixture revision and evaluator revision).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendEvidenceContext {
    pub run_id: String,
    pub task_id: String,
    pub fixture_id: String,
    pub source_commit: String,
    pub environment_manifest_sha256: String,
}

impl FrontendEvidenceContext {
    fn validate(&self) -> Result<()> {
        validate_identifier("run_id", &self.run_id)?;
        validate_identifier("task_id", &self.task_id)?;
        validate_identifier("fixture_id", &self.fixture_id)?;
        if !matches!(self.source_commit.len(), 40 | 64)
            || !self.source_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("source_commit must be a 40- or 64-character hexadecimal object ID");
        }
        validate_sha256("environment_manifest_sha256", &self.environment_manifest_sha256)
    }
}

fn validate_png_header(bytes: &[u8]) -> Result<(u32, u32)> {
    if bytes.len() > MAX_SCREENSHOT_BYTES {
        bail!("frontend screenshot exceeds the compressed artifact size limit");
    }
    if bytes.len() < 24 || !bytes.starts_with(PNG_SIGNATURE) {
        bail!("browser screenshot is truncated or does not have a PNG signature");
    }
    if bytes.get(12..16) != Some(&b"IHDR"[..]) {
        bail!("browser screenshot does not begin with a PNG IHDR chunk");
    }

    let width = u32::from_be_bytes(
        bytes[16..20]
            .try_into()
            .context("decode PNG width")?,
    );
    let height = u32::from_be_bytes(
        bytes[20..24]
            .try_into()
            .context("decode PNG height")?,
    );
    if width == 0
        || height == 0
        || width > MAX_SCREENSHOT_DIMENSION
        || height > MAX_SCREENSHOT_DIMENSION
    {
        bail!("frontend screenshot dimensions are outside the supported range");
    }
    Ok((width, height))
}

/// Content-addressed description of one persisted artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendArtifactDigest {
    pub kind: String,
    pub media_type: String,
    pub sha256: String,
    pub byte_length: u64,
}

/// Receipt for one frontend observation.
///
/// The receipt contains digests and bounded metadata, not the screenshot or
/// accessibility payload. The corresponding artifacts are held separately in
/// `FrontendEvidenceBundle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendEvidenceReceipt {
    pub schema: String,
    pub run_id: String,
    pub task_id: String,
    pub fixture_id: String,
    pub source_commit: String,
    pub environment_manifest_sha256: String,
    pub captured_at_unix_ms: u64,
    pub final_url_redacted: String,
    pub page_title: String,
    pub viewport: FrontendViewport,
    pub document: FrontendDocumentMetrics,
    pub observed_layout_elements: usize,
    pub artifacts: Vec<FrontendArtifactDigest>,
}

impl FrontendEvidenceReceipt {
    /// SHA-256 of the serialized receipt. Persist this alongside the receipt.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Receipt plus the exact bytes to be persisted by the benchmark runner.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontendEvidenceBundle {
    pub receipt: FrontendEvidenceReceipt,
    pub screenshot_png: Vec<u8>,
    pub accessibility_json: Vec<u8>,
    pub layout_json: Vec<u8>,
}

impl FrontendEvidenceBundle {
    pub fn new(
        context: FrontendEvidenceContext,
        mut observation: PageObservation,
        layout: FrontendLayoutSnapshot,
        screenshot_png: Vec<u8>,
    ) -> Result<Self> {
        context.validate()?;
        layout.validate()?;
        let (png_width, png_height) = validate_png_header(&screenshot_png)?;
        let expected_width =
            (f64::from(layout.viewport.width) * layout.viewport.device_scale_factor).round();
        let expected_height =
            (f64::from(layout.viewport.height) * layout.viewport.device_scale_factor).round();
        if f64::from(png_width) != expected_width
            || f64::from(png_height) != expected_height
        {
            bail!(
                "viewport screenshot dimensions {png_width}x{png_height} do not match declared viewport at device scale factor {} (expected {}x{})",
                layout.viewport.device_scale_factor,
                expected_width,
                expected_height
            );
        }

        // Re-apply the observation boundary before serializing a durable
        // artifact: sensitive field values are redacted and page strings bounded.
        observation.sanitize(256);
        let accessibility_json = serde_json::to_vec(&observation)
            .context("serialize sanitized accessibility observation")?;
        let layout_json =
            serde_json::to_vec(&layout).context("serialize frontend layout snapshot")?;

        let artifacts = vec![
            artifact_digest("screenshot", "image/png", &screenshot_png),
            artifact_digest(
                "accessibility",
                "application/json",
                &accessibility_json,
            ),
            artifact_digest("layout", "application/json", &layout_json),
        ];

        let captured_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64;

        let receipt = FrontendEvidenceReceipt {
            schema: "sfeb-frontend-evidence/v1".to_string(),
            run_id: context.run_id,
            task_id: context.task_id,
            fixture_id: context.fixture_id,
            source_commit: context.source_commit,
            environment_manifest_sha256: context.environment_manifest_sha256,
            captured_at_unix_ms,
            final_url_redacted: observation.redacted_url(),
            page_title: observation.title.clone(),
            viewport: layout.viewport.clone(),
            document: layout.document.clone(),
            observed_layout_elements: layout.elements.len(),
            artifacts,
        };

        Ok(Self {
            receipt,
            screenshot_png,
            accessibility_json,
            layout_json,
        })
    }

    /// Verify every artifact against the receipt before evaluation or storage.
    pub fn verify_integrity(&self) -> bool {
        if self.receipt.schema != "sfeb-frontend-evidence/v1"
            || self.receipt.artifacts.len() != 3
        {
            return false;
        }
        let expected = [
            ("screenshot", "image/png", &self.screenshot_png),
            (
                "accessibility",
                "application/json",
                &self.accessibility_json,
            ),
            ("layout", "application/json", &self.layout_json),
        ];
        expected.iter().all(|(kind, media_type, bytes)| {
            self.receipt
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == *kind)
                .is_some_and(|artifact| {
                    artifact.media_type == *media_type
                        && artifact.byte_length == bytes.len() as u64
                        && artifact.sha256 == sha256_hex(bytes)
                })
        })
    }
}

impl CdpSession {
    /// Capture accessibility, layout and viewport screenshot evidence.
    ///
    /// These are sequential read-only observations, not an atomic browser
    /// transaction. Use deterministic fixtures with animations and external
    /// network activity controlled by the harness for benchmark comparisons.
    pub async fn capture_frontend_evidence(
        &self,
        context: FrontendEvidenceContext,
    ) -> Result<FrontendEvidenceBundle> {
        context.validate()?;
        let layout = self.frontend_layout_snapshot().await?;
        let observation = self.observe().await?;
        let screenshot = self.screenshot_viewport().await?;
        FrontendEvidenceBundle::new(context, observation, layout, screenshot)
    }

    /// Capture bounded geometry and computed layout data for semantic elements.
    pub async fn frontend_layout_snapshot(&self) -> Result<FrontendLayoutSnapshot> {
        let raw = self
            .eval_js(LAYOUT_PROBE)
            .await
            .context("frontend layout probe failed")?;
        let snapshot: FrontendLayoutSnapshot =
            serde_json::from_str(&raw).context("decode frontend layout probe")?;
        snapshot.validate()?;
        Ok(snapshot)
    }
}

fn artifact_digest(kind: &str, media_type: &str, bytes: &[u8]) -> FrontendArtifactDigest {
    FrontendArtifactDigest {
        kind: kind.to_string(),
        media_type: media_type.to_string(),
        sha256: sha256_hex(bytes),
        byte_length: bytes.len() as u64,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        bail!("{label} must be 1-128 ASCII letters, digits, '.', '_' or '-'");
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 64-character SHA-256 digest");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::AccessibleElement;

    fn context() -> FrontendEvidenceContext {
        FrontendEvidenceContext {
            run_id: "run-001".into(),
            task_id: "SFEB-L1-001".into(),
            fixture_id: "fixture-dashboard".into(),
            source_commit: "a".repeat(40),
            environment_manifest_sha256: "b".repeat(64),
        }
    }

    fn layout() -> FrontendLayoutSnapshot {
        FrontendLayoutSnapshot {
            viewport: FrontendViewport {
                width: 1280,
                height: 800,
                device_scale_factor: 1.0,
            },
            document: FrontendDocumentMetrics {
                scroll_width: 1280,
                scroll_height: 1200,
                has_horizontal_overflow: false,
                has_vertical_overflow: true,
            },
            elements: vec![FrontendLayoutElement {
                data_sfeb_id: Some("primary-content".into()),
                tag_name: "main".into(),
                role: Some("main".into()),
                rect: FrontendLayoutRect {
                    x: 0.0,
                    y: 64.0,
                    width: 1280.0,
                    height: 900.0,
                },
                visible: true,
                display: "block".into(),
                position: "static".into(),
                overflow_x: "visible".into(),
                overflow_y: "visible".into(),
            }],
        }
    }

    fn observation() -> PageObservation {
        PageObservation {
            url: "https://fixture.example.test/page?token=secret#private".into(),
            title: "Fixture dashboard".into(),
            elements: vec![AccessibleElement {
                backend_node_id: 1,
                role: "textbox".into(),
                name: "API token".into(),
                value: Some("must-not-persist".into()),
                description: None,
                focused: false,
                disabled: false,
            }],
            focused_element: None,
        }
    }

    fn png_fixture() -> Vec<u8> {
        // Header-only fixture for metadata and integrity tests. Browser
        // integration tests must use a real Chromium-produced PNG.
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&13_u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&1280_u32.to_be_bytes());
        bytes.extend_from_slice(&800_u32.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes
    }

    #[test]
    fn validates_layout_and_rejects_non_finite_geometry() {
        assert!(layout().validate().is_ok());
        let mut malformed = layout();
        malformed.elements[0].rect.width = f64::NAN;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn rejects_unbounded_element_count() {
        let mut malformed = layout();
        malformed.elements = vec![malformed.elements[0].clone(); MAX_LAYOUT_ELEMENTS + 1];
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn rejects_duplicate_or_unsafe_source_anchors() {
        let mut malformed = layout();
        malformed.elements.push(malformed.elements[0].clone());
        assert!(malformed.validate().is_err());

        let mut malformed = layout();
        malformed.elements[0].data_sfeb_id = Some("../component".into());
        assert!(malformed.validate().is_err());
    }


    #[test]
    fn rejects_invalid_context_before_capture() {
        let mut malformed = context();
        malformed.source_commit = "not-a-commit".into();
        assert!(malformed.validate().is_err());
        malformed = context();
        malformed.environment_manifest_sha256 = "bad-digest".into();
        assert!(malformed.validate().is_err());
        malformed = context();
        malformed.task_id = "../secrets".into();
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn bundle_redacts_sensitive_input_and_url_secrets() {
        let bundle = FrontendEvidenceBundle::new(
            context(),
            observation(),
            layout(),
            png_fixture(),
        )
        .expect("valid evidence bundle");
        let accessibility = String::from_utf8(bundle.accessibility_json.clone()).unwrap();
        assert!(!accessibility.contains("must-not-persist"));
        assert!(accessibility.contains("[REDACTED]"));
        assert!(!bundle.receipt.final_url_redacted.contains("token=secret"));
        assert!(!bundle.receipt.final_url_redacted.contains("#private"));
    }

    #[test]
    fn bundle_artifact_digests_verify_and_detect_tampering() {
        let mut bundle = FrontendEvidenceBundle::new(
            context(),
            observation(),
            layout(),
            png_fixture(),
        )
        .expect("valid evidence bundle");
        assert!(bundle.verify_integrity());
        assert_eq!(bundle.receipt.artifacts.len(), 3);
        assert_eq!(bundle.receipt.sha256().unwrap().len(), 64);

        bundle.receipt.artifacts[0].media_type = "text/plain".into();
        assert!(!bundle.verify_integrity());
        bundle.receipt.artifacts[0].media_type = "image/png".into();

        bundle.screenshot_png.push(0);
        assert!(!bundle.verify_integrity());
    }

    #[test]
    fn rejects_truncated_png_headers_and_invalid_ihdr() {
        assert!(validate_png_header(b"\x89PNG\r\n\x1a\n").is_err());

        let mut invalid = png_fixture();
        invalid[12..16].copy_from_slice(b"IDAT");
        assert!(validate_png_header(&invalid).is_err());
    }

    #[test]
    fn rejects_zero_and_oversized_png_dimensions() {
        let mut zero_width = png_fixture();
        zero_width[16..20].copy_from_slice(&0_u32.to_be_bytes());
        assert!(validate_png_header(&zero_width).is_err());

        let mut oversized = png_fixture();
        oversized[16..20].copy_from_slice(&(MAX_SCREENSHOT_DIMENSION + 1).to_be_bytes());
        assert!(validate_png_header(&oversized).is_err());
    }

    #[test]
    fn rejects_screenshot_that_does_not_match_viewport_scale() {
        let mut screenshot = png_fixture();
        screenshot[16..20].copy_from_slice(&1279_u32.to_be_bytes());
        let result =
            FrontendEvidenceBundle::new(context(), observation(), layout(), screenshot);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_png_screenshot_bytes() {
        let result =
            FrontendEvidenceBundle::new(context(), observation(), layout(), b"not png".to_vec());
        assert!(result.is_err());
    }
}
