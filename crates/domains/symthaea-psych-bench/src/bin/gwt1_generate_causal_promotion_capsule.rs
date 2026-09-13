// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(not(feature = "symthaea-backend"))]
fn main() {
    eprintln!("gwt1_generate_causal_promotion_capsule requires --features symthaea-backend");
    std::process::exit(2);
}

#[cfg(feature = "symthaea-backend")]
fn main() {
    if let Err(error) = trusted_main() {
        eprintln!("GWT-1 causal promotion capsule generation failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "symthaea-backend")]
fn trusted_main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use serde::Deserialize;
    use symthaea_psych_bench::benchmarks::butlin::{
        GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1, GWT1_CAUSAL_PROMOTION_POLICY_V1,
        GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1, GWT1_CAUSAL_TRUSTED_REPOSITORY_V1,
        Gwt1CausalEvidenceEnvelopeResolutionV1, Gwt1CausalEvidenceEnvelopeV1,
        Gwt1CausalPromotionCapsuleV1, SupportTier,
        map_gwt1_causal_outcome_to_eligibility_v1, resolve_gwt1_causal_envelope_v1,
        validate_gwt1_causal_promotion_capsule_v1,
    };

    #[derive(Debug, Deserialize)]
    struct WorkflowProvenance {
        schema: String,
        evidence_subject_sha: String,
        trusted_builder_sha: String,
        trusted_builder_ref: String,
        repository: String,
        run_id: String,
        run_attempt: String,
    }

    fn required_env(name: &str) -> Result<String, io::Error> {
        env::var(name)
            .map_err(|_| io::Error::other(format!("missing required environment variable {name}")))
    }

    fn sha256sum(path: &Path) -> Result<String, io::Error> {
        let output = Command::new("sha256sum").arg(path).output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "sha256sum failed for {}: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let digest = stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| io::Error::other("sha256sum produced no digest"))?
            .to_string();
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(io::Error::other(format!(
                "sha256sum produced malformed digest {digest:?}"
            )));
        }
        Ok(digest)
    }

    fn read_json<T: for<'de> Deserialize<'de>>(
        path: &Path,
    ) -> Result<T, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    let evidence_dir = PathBuf::from(required_env("SYMTHAEA_GWT1_CAUSAL_EVIDENCE_DIR")?);
    let archive_path = PathBuf::from(required_env("GWT1_CAUSAL_ARCHIVE_PATH")?);
    let attestation_bundle_path =
        PathBuf::from(required_env("GWT1_CAUSAL_ARCHIVE_ATTESTATION_BUNDLE")?);
    let verification_path =
        PathBuf::from(required_env("GWT1_CAUSAL_ARCHIVE_ATTESTATION_VERIFICATION")?);
    let output_path = PathBuf::from(required_env("GWT1_CAUSAL_PROMOTION_OUTPUT")?);
    let trusted_builder_sha = required_env("TRUSTED_GWT1_CAUSAL_BUILDER_SHA")?;
    let trusted_builder_ref = required_env("TRUSTED_GWT1_CAUSAL_BUILDER_REF")?;
    let repository = required_env("GITHUB_REPOSITORY")?;

    if repository != GWT1_CAUSAL_TRUSTED_REPOSITORY_V1 {
        return Err(io::Error::other(format!(
            "unexpected repository {repository:?}; expected {GWT1_CAUSAL_TRUSTED_REPOSITORY_V1:?}"
        ))
        .into());
    }

    let causal_raw = fs::read(evidence_dir.join("causal_observations.json"))?;
    let matched_sham_raw = fs::read(evidence_dir.join("matched_sham_observations.json"))?;
    let envelope: Gwt1CausalEvidenceEnvelopeV1 =
        read_json(&evidence_dir.join("causal_evidence_envelope.json"))?;
    let stored_resolution: Gwt1CausalEvidenceEnvelopeResolutionV1 =
        read_json(&evidence_dir.join("resolution.json"))?;

    let recomputed =
        resolve_gwt1_causal_envelope_v1(&envelope, &causal_raw, &matched_sham_raw);
    if recomputed != stored_resolution {
        return Err(io::Error::other(format!(
            "stored causal resolution differs from promotion-time recomputation: stored={:?}, recomputed={:?}",
            stored_resolution.outcome, recomputed.outcome
        ))
        .into());
    }

    let provenance: WorkflowProvenance = read_json(&evidence_dir.join("workflow_provenance.json"))?;
    if provenance.schema != "butlin-gwt1-causal-trusted-builder-provenance-v1" {
        return Err(io::Error::other(format!(
            "unexpected workflow provenance schema {:?}",
            provenance.schema
        ))
        .into());
    }
    if provenance.repository != repository {
        return Err(io::Error::other("workflow provenance repository mismatch").into());
    }
    if provenance.trusted_builder_sha != trusted_builder_sha {
        return Err(io::Error::other("workflow provenance trusted builder SHA mismatch").into());
    }
    if provenance.trusted_builder_ref != trusted_builder_ref {
        return Err(io::Error::other("workflow provenance trusted builder ref mismatch").into());
    }
    if provenance.evidence_subject_sha != envelope.execution_identity.source_commit_sha {
        return Err(io::Error::other("workflow provenance evidence subject mismatch").into());
    }
    let provenance_run_id = format!("{}/{}", provenance.run_id, provenance.run_attempt);
    if provenance_run_id != envelope.execution_identity.execution_run_id {
        return Err(io::Error::other(format!(
            "workflow provenance execution run mismatch: provenance={provenance_run_id:?}, envelope={:?}",
            envelope.execution_identity.execution_run_id
        ))
        .into());
    }

    let verification_bytes = fs::read(&verification_path)?;
    let verification_json: serde_json::Value = serde_json::from_slice(&verification_bytes)?;
    if verification_json.is_null() {
        return Err(io::Error::other("archive attestation verification record is JSON null").into());
    }
    if fs::metadata(&attestation_bundle_path)?.len() == 0 {
        return Err(io::Error::other("archive attestation bundle is empty").into());
    }
    if fs::metadata(&archive_path)?.len() == 0 {
        return Err(io::Error::other("causal evidence archive is empty").into());
    }

    let capsule = Gwt1CausalPromotionCapsuleV1 {
        schema: GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1.to_string(),
        policy: GWT1_CAUSAL_PROMOTION_POLICY_V1.to_string(),
        indicator_id: "GWT-1".to_string(),
        repository,
        trusted_builder_workflow: GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1.to_string(),
        trusted_builder_sha,
        trusted_builder_ref,
        causal_archive_sha256: sha256sum(&archive_path)?,
        archive_attestation_bundle_sha256: sha256sum(&attestation_bundle_path)?,
        archive_attestation_verification_sha256: sha256sum(&verification_path)?,
        evidence_subject: envelope.execution_identity.clone(),
        scientific_outcome: recomputed.outcome,
        eligible_evidence_outcome: map_gwt1_causal_outcome_to_eligibility_v1(recomputed.outcome),
        tier_ceiling: SupportTier::CausallySupported,
    };

    let failures = validate_gwt1_causal_promotion_capsule_v1(&capsule);
    if !failures.is_empty() {
        return Err(io::Error::other(format!(
            "promotion capsule failed internal validation: {failures:?}"
        ))
        .into());
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_vec_pretty(&capsule)?)?;

    println!(
        "GWT-1 causal promotion capsule generated: scientific={:?}, eligible={:?}",
        capsule.scientific_outcome, capsule.eligible_evidence_outcome
    );
    println!("Promotion capsule: {}", output_path.display());
    Ok(())
}
