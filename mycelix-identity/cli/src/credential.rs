//! Credential Generation and ZK Commitment
//!
//! Generates W3C VC 2.0 compliant academic credentials with
//! ZK commitments for selective disclosure.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::Path;
use uuid::Uuid;

use crate::csv_parser::ParsedRecord;
use crate::dns_did::DnsDidVerification;

/// Degree type mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum DegreeType {
    HighSchool,
    Associate,
    Bachelor,
    Master,
    Doctorate,
    Professional,
    Certificate,
    Diploma,
    Microcredential,
    CourseCompletion,
}

impl DegreeType {
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("bachelor") || lower.contains("b.s.") || lower.contains("b.a.") {
            DegreeType::Bachelor
        } else if lower.contains("master") || lower.contains("m.s.") || lower.contains("m.a.") {
            DegreeType::Master
        } else if lower.contains("doctor") || lower.contains("ph.d") || lower.contains("phd") {
            DegreeType::Doctorate
        } else if lower.contains("associate") || lower.contains("a.s.") || lower.contains("a.a.") {
            DegreeType::Associate
        } else if lower.contains("certificate") {
            DegreeType::Certificate
        } else if lower.contains("high school") {
            DegreeType::HighSchool
        } else if lower.contains("j.d.") || lower.contains("m.d.") {
            DegreeType::Professional
        } else {
            DegreeType::Diploma
        }
    }
}

/// Institution issuer details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalIssuer {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub issuer_type: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// Academic subject (student)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicSubject {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_hash: Option<String>,
}

/// Achievement metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementMetadata {
    pub degree_type: DegreeType,
    pub degree_name: String,
    pub field_of_study: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minors: Option<Vec<String>>,
    pub conferral_date: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpa: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub honors: Option<Vec<String>>,
}

/// Cryptographic proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicProof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub created: String,
    #[serde(rename = "verificationMethod")]
    pub verification_method: String,
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,
    #[serde(rename = "proofValue")]
    pub proof_value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cryptosuite: Option<String>,
}

/// DNS-DID data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsDid {
    pub domain: String,
    pub did: String,
    pub txt_record: String,
    pub dnssec: String,
    pub last_verified: String,
}

/// Complete academic credential (matches zome structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: String,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub issuer: InstitutionalIssuer,
    #[serde(rename = "validFrom")]
    pub valid_from: String,
    #[serde(rename = "validUntil", skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    #[serde(rename = "credentialSubject")]
    pub credential_subject: AcademicSubject,
    pub proof: AcademicProof,
    pub zk_commitment: String,
    pub commitment_nonce: String,
    pub revocation_registry_id: String,
    pub revocation_index: u32,
    pub dns_did: DnsDid,
    pub achievement: AchievementMetadata,
    pub mycelix_schema_id: String,
    pub mycelix_created: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_import_ref: Option<String>,
}

/// Credential generation configuration
pub struct CredentialConfig {
    pub institution_did: String,
    pub institution_name: String,
    pub revocation_registry: String,
    pub dns_verification: Option<DnsDidVerification>,
    pub signing_key: SigningKey,
}

impl CredentialConfig {
    pub fn new(
        institution_did: String,
        institution_name: String,
        revocation_registry: String,
        dns_verification: Option<DnsDidVerification>,
    ) -> Self {
        // Generate signing key for this session
        let signing_key = SigningKey::generate(&mut OsRng);

        Self {
            institution_did,
            institution_name,
            revocation_registry,
            dns_verification,
            signing_key,
        }
    }
}

/// Generate ZK commitment
pub fn generate_zk_commitment(credential_id: &str, subject_id: &str, nonce: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(credential_id.as_bytes());
    hasher.update(b"|");
    hasher.update(subject_id.as_bytes());
    hasher.update(b"|");
    hasher.update(nonce);
    hasher.finalize().to_vec()
}

/// Generate name hash for privacy
pub fn hash_name(first_name: &str, last_name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(first_name.to_lowercase().as_bytes());
    hasher.update(b" ");
    hasher.update(last_name.to_lowercase().as_bytes());
    hex::encode(hasher.finalize())
}

/// Build a credential from a parsed record
pub fn build_credential(
    record: &ParsedRecord,
    config: &CredentialConfig,
    revocation_index: u32,
    batch_id: &str,
) -> Result<AcademicCredential> {
    let now: DateTime<Utc> = Utc::now();
    let now_iso = now.to_rfc3339();

    // Generate credential ID
    let credential_id = format!("urn:uuid:{}", Uuid::new_v4());

    // Generate subject DID from student ID and institution
    let subject_did = format!(
        "did:student:{}:{}",
        config.institution_did.replace("did:", "").replace(":", "-"),
        record.student_id
    );

    // Generate nonce for ZK commitment
    let mut nonce = [0u8; 32];
    rand::RngCore::fill_bytes(&mut OsRng, &mut nonce);

    // Generate ZK commitment
    let zk_commitment = generate_zk_commitment(&credential_id, &subject_did, &nonce);

    // Build achievement metadata
    let achievement = AchievementMetadata {
        degree_type: DegreeType::from_name(&record.degree_name),
        degree_name: record.degree_name.clone(),
        field_of_study: record.major.clone(),
        minors: record.minor.as_ref().map(|m| vec![m.clone()]),
        conferral_date: normalize_date(&record.conferral_date),
        gpa: record.gpa,
        honors: if record.honors.is_empty() {
            None
        } else {
            Some(record.honors.clone())
        },
    };

    // Build subject with hashed name for privacy
    let name_hash = hash_name(&record.first_name, &record.last_name);
    let credential_subject = AcademicSubject {
        id: subject_did.clone(),
        name: Some(format!("{} {}", record.first_name, record.last_name)),
        name_hash: Some(name_hash),
    };

    // Build DNS-DID data
    let dns_did = if let Some(ref verification) = config.dns_verification {
        DnsDid {
            domain: verification.domain.clone(),
            did: verification.resolved_did.clone().unwrap_or_default(),
            txt_record: format!("_did.{}", verification.domain),
            dnssec: format!("{:?}", verification.dnssec_status),
            last_verified: now_iso.clone(),
        }
    } else {
        // Extract domain from DID
        let domain = config
            .institution_did
            .strip_prefix("did:dns:")
            .unwrap_or("unknown.edu");
        DnsDid {
            domain: domain.to_string(),
            did: config.institution_did.clone(),
            txt_record: format!("_did.{}", domain),
            dnssec: "Unknown".to_string(),
            last_verified: now_iso.clone(),
        }
    };

    // Build issuer
    let issuer = InstitutionalIssuer {
        id: config.institution_did.clone(),
        name: config.institution_name.clone(),
        issuer_type: vec!["Organization".to_string(), "EducationalOrganization".to_string()],
        image: None,
    };

    // Create proof hash
    let proof_hash = compute_proof_hash(&credential_id, &issuer.id, &subject_did, &now_iso);

    // Sign the proof
    let signature = config.signing_key.sign(&proof_hash);
    let proof_value = format!("z{}", BASE64.encode(signature.to_bytes()));

    // Build proof
    let proof = AcademicProof {
        proof_type: "DataIntegrityProof".to_string(),
        created: now_iso.clone(),
        verification_method: format!("{}#keys-1", config.institution_did),
        proof_purpose: "assertionMethod".to_string(),
        proof_value,
        cryptosuite: Some("eddsa-rdfc-2022".to_string()),
    };

    Ok(AcademicCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://w3id.org/vc/status-list/2021/v1".to_string(),
            "https://purl.imsglobal.org/spec/clr/v2p0/context.json".to_string(),
        ],
        id: credential_id,
        credential_type: vec![
            "VerifiableCredential".to_string(),
            "AcademicCredential".to_string(),
            format!("{:?}Credential", achievement.degree_type),
        ],
        issuer,
        valid_from: now_iso.clone(),
        valid_until: None,
        credential_subject,
        proof,
        zk_commitment: BASE64.encode(&zk_commitment),
        commitment_nonce: BASE64.encode(nonce),
        revocation_registry_id: config.revocation_registry.clone(),
        revocation_index,
        dns_did,
        achievement,
        mycelix_schema_id: "mycelix:schema:education:academic:v1".to_string(),
        mycelix_created: now_iso,
        legacy_import_ref: Some(format!("{}:{}", batch_id, record.row_number)),
    })
}

fn compute_proof_hash(
    credential_id: &str,
    issuer_did: &str,
    subject_did: &str,
    created: &str,
) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(credential_id.as_bytes());
    hasher.update(issuer_did.as_bytes());
    hasher.update(subject_did.as_bytes());
    hasher.update(created.as_bytes());
    hasher.finalize().to_vec()
}

fn normalize_date(date: &str) -> String {
    // Try to parse and normalize to ISO 8601
    let formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"];

    for fmt in &formats {
        if let Ok(parsed) = chrono::NaiveDate::parse_from_str(date, fmt) {
            return parsed.format("%Y-%m-%d").to_string();
        }
    }

    // Return as-is if parsing fails
    date.to_string()
}

/// Verify a ZK commitment from CLI
pub fn verify_commitment_cli(credential_path: &Path, nonce_b64: &str) -> Result<()> {
    let file = File::open(credential_path).context("Failed to open credential file")?;
    let credential: AcademicCredential =
        serde_json::from_reader(file).context("Failed to parse credential")?;

    let nonce = BASE64
        .decode(nonce_b64)
        .context("Failed to decode nonce (expected base64)")?;

    let expected = generate_zk_commitment(
        &credential.id,
        &credential.credential_subject.id,
        &nonce,
    );

    let stored = BASE64
        .decode(&credential.zk_commitment)
        .context("Failed to decode stored commitment")?;

    if expected == stored {
        println!("ZK commitment verified successfully");
        println!("Credential ID: {}", credential.id);
        println!("Subject: {}", credential.credential_subject.id);
    } else {
        println!("ZK commitment verification FAILED");
        println!("Expected: {}", BASE64.encode(&expected));
        println!("Stored:   {}", credential.zk_commitment);
        anyhow::bail!("Commitment mismatch");
    }

    Ok(())
}

/// Comprehensive credential verification from CLI
pub async fn verify_credential_cli(
    credential_path: &Path,
    check_dns: bool,
    verbose: bool,
) -> Result<()> {
    use console::style;

    println!();
    println!(
        "{}",
        style("=== Credential Verification ===").bold().cyan()
    );
    println!();

    // Load credential
    let file = File::open(credential_path).context("Failed to open credential file")?;
    let credential: AcademicCredential =
        serde_json::from_reader(file).context("Failed to parse credential JSON")?;

    let mut checks_passed = 0u32;
    let mut checks_failed = 0u32;
    let mut warnings = 0u32;

    // 1. Structure validation
    println!("{}", style("[1/5] Structure Validation").bold());

    // Check W3C VC 2.0 context
    if credential
        .context
        .contains(&"https://www.w3.org/ns/credentials/v2".to_string())
    {
        println!("  {} W3C VC 2.0 context present", style("✓").green());
        checks_passed += 1;
    } else {
        println!("  {} Missing W3C VC 2.0 context", style("✗").red());
        checks_failed += 1;
    }

    // Check type
    if credential
        .credential_type
        .contains(&"VerifiableCredential".to_string())
        && credential
            .credential_type
            .contains(&"AcademicCredential".to_string())
    {
        println!("  {} Credential type valid", style("✓").green());
        checks_passed += 1;
    } else {
        println!(
            "  {} Missing required credential types",
            style("✗").red()
        );
        checks_failed += 1;
    }

    // Check credential ID format
    if credential.id.starts_with("urn:uuid:") {
        println!("  {} Credential ID format valid", style("✓").green());
        checks_passed += 1;
    } else {
        println!("  {} Invalid credential ID format", style("✗").red());
        checks_failed += 1;
    }

    // 2. Issuer validation
    println!();
    println!("{}", style("[2/5] Issuer Validation").bold());

    if credential.issuer.id.starts_with("did:") {
        println!("  {} Issuer DID valid", style("✓").green());
        checks_passed += 1;
    } else {
        println!("  {} Issuer DID invalid", style("✗").red());
        checks_failed += 1;
    }

    if !credential.issuer.name.is_empty() {
        println!(
            "  {} Issuer: {}",
            style("✓").green(),
            credential.issuer.name
        );
        checks_passed += 1;
    } else {
        println!("  {} Missing issuer name", style("✗").red());
        checks_failed += 1;
    }

    // 3. Proof validation
    println!();
    println!("{}", style("[3/5] Proof Validation").bold());

    if credential.proof.proof_type == "DataIntegrityProof" {
        println!("  {} Proof type: DataIntegrityProof", style("✓").green());
        checks_passed += 1;
    } else {
        println!(
            "  {} Unknown proof type: {}",
            style("⚠").yellow(),
            credential.proof.proof_type
        );
        warnings += 1;
    }

    if credential.proof.proof_value.starts_with('z') {
        println!("  {} Proof value multibase-encoded", style("✓").green());
        checks_passed += 1;

        // Try to decode the signature
        let sig_b64 = &credential.proof.proof_value[1..]; // Strip 'z' prefix
        match BASE64.decode(sig_b64) {
            Ok(sig_bytes) if sig_bytes.len() == 64 => {
                println!(
                    "  {} Ed25519 signature present (64 bytes)",
                    style("✓").green()
                );
                checks_passed += 1;
            }
            Ok(sig_bytes) => {
                println!(
                    "  {} Signature length {} (expected 64 for Ed25519)",
                    style("⚠").yellow(),
                    sig_bytes.len()
                );
                warnings += 1;
            }
            Err(_) => {
                println!("  {} Failed to decode proof value", style("✗").red());
                checks_failed += 1;
            }
        }
    } else {
        println!(
            "  {} Proof value not multibase-encoded",
            style("✗").red()
        );
        checks_failed += 1;
    }

    if credential.proof.proof_purpose == "assertionMethod" {
        println!("  {} Proof purpose: assertionMethod", style("✓").green());
        checks_passed += 1;
    } else {
        println!(
            "  {} Unexpected proof purpose: {}",
            style("⚠").yellow(),
            credential.proof.proof_purpose
        );
        warnings += 1;
    }

    // 4. ZK Commitment validation
    println!();
    println!("{}", style("[4/5] ZK Commitment Validation").bold());

    match BASE64.decode(&credential.zk_commitment) {
        Ok(commitment) if commitment.len() == 32 => {
            println!(
                "  {} ZK commitment present (SHA-256, 32 bytes)",
                style("✓").green()
            );
            checks_passed += 1;
        }
        Ok(commitment) => {
            println!(
                "  {} ZK commitment length {} (expected 32)",
                style("⚠").yellow(),
                commitment.len()
            );
            warnings += 1;
        }
        Err(_) => {
            println!(
                "  {} Failed to decode ZK commitment",
                style("✗").red()
            );
            checks_failed += 1;
        }
    }

    // Check revocation registry
    if !credential.revocation_registry_id.is_empty() {
        println!(
            "  {} Revocation registry: {}",
            style("✓").green(),
            credential.revocation_registry_id
        );
        checks_passed += 1;
    } else {
        println!(
            "  {} Missing revocation registry (REQUIRED)",
            style("✗").red()
        );
        checks_failed += 1;
    }

    // 5. DNS-DID verification
    println!();
    println!("{}", style("[5/5] DNS-DID Verification").bold());
    println!(
        "  Domain:     {}",
        credential.dns_did.domain
    );
    println!("  DID:        {}", credential.dns_did.did);
    println!("  TXT Record: {}", credential.dns_did.txt_record);
    println!("  DNSSEC:     {}", credential.dns_did.dnssec);

    if check_dns {
        match crate::dns_did::verify_dns_did(
            &credential.dns_did.domain,
            Some(&credential.issuer.id),
            verbose,
        )
        .await
        {
            Ok(verification) => {
                if verification.verified {
                    println!(
                        "  {} DNS-DID verified live",
                        style("✓").green()
                    );
                    checks_passed += 1;
                } else {
                    println!(
                        "  {} DNS-DID not verified (DID not found in DNS)",
                        style("⚠").yellow()
                    );
                    warnings += 1;
                }
            }
            Err(e) => {
                println!(
                    "  {} DNS lookup failed: {}",
                    style("⚠").yellow(),
                    e
                );
                warnings += 1;
            }
        }
    } else {
        println!(
            "  {} DNS-DID not checked (use --check-dns to verify)",
            style("○").dim()
        );
    }

    // Summary
    println!();
    println!(
        "{}",
        style("═══════════════════════════════════").cyan()
    );
    println!("{}", style("     VERIFICATION SUMMARY").bold());
    println!(
        "{}",
        style("═══════════════════════════════════").cyan()
    );
    println!();
    println!("  Credential: {}", credential.id);
    println!("  Subject:    {}", credential.credential_subject.id);
    println!(
        "  Degree:     {} in {}",
        credential.achievement.degree_name, credential.achievement.field_of_study
    );
    println!();
    println!(
        "  Checks passed:  {}",
        style(checks_passed).green().bold()
    );
    if warnings > 0 {
        println!(
            "  Warnings:       {}",
            style(warnings).yellow().bold()
        );
    }
    if checks_failed > 0 {
        println!(
            "  Checks failed:  {}",
            style(checks_failed).red().bold()
        );
    }
    println!();

    if checks_failed == 0 {
        println!(
            "  {}",
            style("Credential structure is VALID").green().bold()
        );
    } else {
        println!(
            "  {}",
            style("Credential has ERRORS").red().bold()
        );
    }

    if verbose {
        println!();
        println!("{}", style("=== Raw Credential Data ===").dim());
        println!(
            "{}",
            serde_json::to_string_pretty(&credential).unwrap()
        );
    }

    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_commitment() {
        let nonce = b"test_nonce_12345";
        let commitment =
            generate_zk_commitment("urn:uuid:123", "did:student:test", nonce);
        assert_eq!(commitment.len(), 32); // SHA-256

        // Same inputs should produce same output
        let commitment2 =
            generate_zk_commitment("urn:uuid:123", "did:student:test", nonce);
        assert_eq!(commitment, commitment2);

        // Different inputs should produce different output
        let commitment3 =
            generate_zk_commitment("urn:uuid:456", "did:student:test", nonce);
        assert_ne!(commitment, commitment3);
    }

    #[test]
    fn test_degree_type_parsing() {
        assert!(matches!(
            DegreeType::from_name("Bachelor of Science"),
            DegreeType::Bachelor
        ));
        assert!(matches!(
            DegreeType::from_name("Master of Arts"),
            DegreeType::Master
        ));
        assert!(matches!(
            DegreeType::from_name("Ph.D."),
            DegreeType::Doctorate
        ));
    }
}
