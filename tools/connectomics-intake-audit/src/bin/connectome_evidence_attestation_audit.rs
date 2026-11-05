//! CLI for verifying signed connectomics evidence statements against a pinned trust policy.
//!
//! Cryptographic authentication is not a claim of scientific truth.

#[path = "../evidence_attestation.rs"]
mod evidence_attestation;

use clap::Parser;
use evidence_attestation::{
    DsseEnvelope, TrustPolicy, dsse_pae, sha256, verify_envelope,
};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-evidence-attestation-audit",
    version,
    about = "Verify DSSE-signed connectomics evidence against a pinned role-bound trust policy"
)]
struct Args {
    /// DSSE envelope JSON.
    #[arg(long)]
    envelope: PathBuf,
    /// Separately maintained trust policy JSON.
    #[arg(long)]
    trust_policy: PathBuf,
    /// Caller-supplied SHA-256 pin for the trust policy.
    #[arg(long)]
    expected_trust_policy_sha256: String,
    /// Deterministic Unix time at which key and statement validity is evaluated.
    #[arg(long)]
    verification_time_unix: i64,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let envelope_bytes = match fs::read(&args.envelope) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-evidence-attestation-audit: envelope read failed: {error}");
            return ExitCode::from(2);
        }
    };
    let policy_bytes = match fs::read(&args.trust_policy) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-evidence-attestation-audit: trust-policy read failed: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match verify_envelope(
        &envelope_bytes,
        &policy_bytes,
        &args.expected_trust_policy_sha256,
        args.verification_time_unix,
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("connectome-evidence-attestation-audit: {error}");
            return ExitCode::from(3);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!(
                "connectome-evidence-attestation-audit: report serialization failed: {error}"
            );
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-evidence-attestation-audit: report write failed: {error}");
            return ExitCode::from(2);
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};
    use evidence_attestation::EXPECTED_PAYLOAD_TYPE;
    use serde_json;

    fn fixture_envelope() -> Vec<u8> {
        include_bytes!(
            "../../../../docs/neuroscience/examples/CONNECTOME_SIGNED_EVIDENCE_ENVELOPE.example.json"
        )
        .to_vec()
    }

    fn fixture_policy() -> Vec<u8> {
        include_bytes!(
            "../../../../docs/neuroscience/examples/CONNECTOME_SIGNED_EVIDENCE_TRUST_POLICY.example.json"
        )
        .to_vec()
    }

    fn verify(envelope: &[u8], policy: &[u8]) -> Result<evidence_attestation::VerificationReceipt, String> {
        verify_envelope(
            envelope,
            policy,
            &sha256(policy),
            1_793_900_000,
        )
    }

    #[test]
    fn valid_three_role_dsse_envelope_is_accepted() {
        let envelope = fixture_envelope();
        let policy = fixture_policy();
        let report = verify(&envelope, &policy).unwrap();
        assert_eq!(report.verified_signer_count, 3);
        assert_eq!(
            report.covered_roles,
            vec![
                "data_steward".to_string(),
                "independent_reviewer".to_string(),
                "observer".to_string()
            ]
        );
        assert!(report.policy_accepted);
        assert_eq!(report.scientific_truth_claim, "NOT_ESTABLISHED");
    }

    #[test]
    fn policy_digest_pin_is_mandatory() {
        let envelope = fixture_envelope();
        let policy = fixture_policy();
        assert!(verify_envelope(&envelope, &policy, &"0".repeat(64), 1_793_900_000).is_err());
    }

    #[test]
    fn tampered_signed_payload_is_rejected() {
        let mut envelope: DsseEnvelope = serde_json::from_slice(&fixture_envelope()).unwrap();
        let mut payload = STANDARD.decode(&envelope.payload).unwrap();
        payload.push(b' ');
        envelope.payload = STANDARD.encode(payload);
        let envelope = serde_json::to_vec(&envelope).unwrap();
        assert!(verify(&envelope, &fixture_policy()).is_err());
    }

    #[test]
    fn wrong_payload_type_is_rejected() {
        let mut envelope: DsseEnvelope = serde_json::from_slice(&fixture_envelope()).unwrap();
        envelope.payload_type = "application/json".into();
        let envelope = serde_json::to_vec(&envelope).unwrap();
        assert!(verify(&envelope, &fixture_policy()).is_err());
    }

    #[test]
    fn duplicate_signer_is_rejected() {
        let mut envelope: DsseEnvelope = serde_json::from_slice(&fixture_envelope()).unwrap();
        envelope.signatures.push(envelope.signatures[0].clone());
        let envelope = serde_json::to_vec(&envelope).unwrap();
        assert!(verify(&envelope, &fixture_policy()).is_err());
    }

    #[test]
    fn missing_required_role_is_rejected() {
        let envelope = fixture_envelope();
        let mut policy: TrustPolicy = serde_json::from_slice(&fixture_policy()).unwrap();
        policy.required_roles.push("functional_validator".into());
        let policy = serde_json::to_vec(&policy).unwrap();
        assert!(verify(&envelope, &policy).is_err());
    }

    #[test]
    fn same_public_key_cannot_be_relabelled_as_a_second_role() {
        let envelope = fixture_envelope();
        let mut policy: TrustPolicy = serde_json::from_slice(&fixture_policy()).unwrap();
        policy.trusted_keys[1].public_key_hex = policy.trusted_keys[0].public_key_hex.clone();
        let policy = serde_json::to_vec(&policy).unwrap();
        let error = verify(&envelope, &policy).unwrap_err();
        assert!(error.contains("reuses Ed25519 public-key material"));
    }

    #[test]
    fn revoked_trust_key_is_rejected_even_for_previously_issued_statement() {
        let envelope = fixture_envelope();
        let mut policy: TrustPolicy = serde_json::from_slice(&fixture_policy()).unwrap();
        policy.trusted_keys[0].revoked_at_unix = Some(1_700_000_000);
        let policy = serde_json::to_vec(&policy).unwrap();
        assert!(verify(&envelope, &policy).is_err());
    }

    #[test]
    fn expired_trust_key_is_rejected() {
        let envelope = fixture_envelope();
        let mut policy: TrustPolicy = serde_json::from_slice(&fixture_policy()).unwrap();
        policy.trusted_keys[0].valid_until_unix = Some(1_700_000_000);
        let policy = serde_json::to_vec(&policy).unwrap();
        assert!(verify(&envelope, &policy).is_err());
    }

    #[test]
    fn future_issued_statement_is_rejected() {
        let mut envelope: DsseEnvelope = serde_json::from_slice(&fixture_envelope()).unwrap();
        let mut payload: evidence_attestation::EvidenceStatement =
            serde_json::from_slice(&STANDARD.decode(&envelope.payload).unwrap()).unwrap();
        payload.issued_at_unix = 1_900_000_000;
        envelope.payload = STANDARD.encode(serde_json::to_vec(&payload).unwrap());
        let envelope = serde_json::to_vec(&envelope).unwrap();
        assert!(verify(&envelope, &fixture_policy()).is_err());
    }

    #[test]
    fn dsse_pae_binds_type_and_exact_payload_bytes() {
        let payload = br#"{"x":1}"#;
        let pae = dsse_pae("application/test", payload);
        assert_eq!(pae, b"DSSEv1 16 application/test 7 {\"x\":1}".to_vec());
    }

    #[test]
    fn fixture_uses_the_expected_signed_media_type() {
        let envelope: DsseEnvelope = serde_json::from_slice(&fixture_envelope()).unwrap();
        assert_eq!(envelope.payload_type, EXPECTED_PAYLOAD_TYPE);
    }
}
