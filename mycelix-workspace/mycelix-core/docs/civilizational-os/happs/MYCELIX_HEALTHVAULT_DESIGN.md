# Mycelix-HealthVault: Decentralized Health Records

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - High Synergy

---

## Executive Summary

Mycelix-HealthVault is a patient-controlled, privacy-preserving health records system built on Holochain. It enables individuals to own their complete health history while allowing selective, cryptographically-verified sharing with healthcare providers, researchers, and family members.

### Why HealthVault?

Healthcare data is uniquely sensitive yet uniquely valuable:
- **Personal**: Medical history reveals intimate details about individuals
- **Critical**: Access can be life-or-death in emergencies
- **Valuable**: Aggregated data advances medical research
- **Fragmented**: Currently siloed across providers, insurers, nations

HealthVault resolves these tensions through agent-centric data ownership with selective disclosure.

---

## Core Principles

### 1. Patient Sovereignty
The patient owns their data. Period. No platform, provider, or government has access without explicit consent.

### 2. Selective Disclosure
Share only what's needed. A dermatologist doesn't need your psychiatric history. A researcher doesn't need your identity.

### 3. Emergency Access
Life-saving information must be accessible in emergencies, with appropriate safeguards against abuse.

### 4. Research Enablement
Aggregate insights can be extracted without exposing individual records, using federated learning.

### 5. Provider Verification
Only verified healthcare providers can receive sensitive information, using Praxis credential chains.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Patient Interface                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Mobile App   │  │  Web Portal  │  │  Wearable Sync           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                        HealthVault hApp                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Coordinator Zomes                           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Records  │ │ Sharing  │ │ Emergency│ │ Research │           ││
│  │  │ Manager  │ │ Manager  │ │ Access   │ │ Consent  │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                        ││
│  │  │ Provider │ │ Audit    │ │ Analytics│                        ││
│  │  │ Verify   │ │ Trail    │ │ (Local)  │                        ││
│  │  └──────────┘ └──────────┘ └──────────┘                        ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │                      Integrity Zomes                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Health   │ │ Consent  │ │ Access   │ │ Audit    │           ││
│  │  │ Record   │ │ Grant    │ │ Log      │ │ Entry    │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                     Cross-hApp Integrations                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Praxis       │  │ 0TML         │  │ Bridge Protocol          │   │
│  │ (Provider    │  │ (Federated   │  │ (Reputation,             │   │
│  │  Credentials)│  │  Learning)   │  │  Identity)               │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A single health record entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HealthRecord {
    /// Unique record identifier
    pub record_id: String,
    /// Type of record (lab, imaging, procedure, medication, etc.)
    pub record_type: HealthRecordType,
    /// When the health event occurred
    pub event_date: Timestamp,
    /// When this record was created
    pub created_at: Timestamp,
    /// Provider who created this record (if applicable)
    pub provider: Option<AgentPubKey>,
    /// Provider's verified credential hash
    pub provider_credential: Option<ActionHash>,
    /// Encrypted record content
    pub encrypted_content: EncryptedData,
    /// Content hash for verification without decryption
    pub content_hash: String,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
    /// Attachments (images, PDFs, etc.)
    pub attachments: Vec<AttachmentRef>,
    /// Tags for organization
    pub tags: Vec<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthRecordType {
    // Clinical
    Diagnosis,
    Procedure,
    Medication,
    Allergy,
    Immunization,
    VitalSigns,

    // Laboratory
    LabResult,
    PathologyReport,
    GeneticTest,

    // Imaging
    XRay,
    MRI,
    CTScan,
    Ultrasound,

    // Documentation
    ClinicalNote,
    DischargeSummary,
    Referral,

    // Patient-Generated
    Symptom,
    LifestyleLog,
    WearableData,

    // Administrative
    Insurance,
    Billing,
    Consent,
}

/// Consent grant for sharing records
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsentGrant {
    /// Who is granting consent (patient)
    pub grantor: AgentPubKey,
    /// Who receives access
    pub grantee: AgentPubKey,
    /// Grantee's verified credential (must be valid provider)
    pub grantee_credential: Option<ActionHash>,
    /// What record types are accessible
    pub record_types: Vec<HealthRecordType>,
    /// Specific records (if not type-based)
    pub specific_records: Vec<ActionHash>,
    /// Date range of accessible records
    pub date_range: Option<DateRange>,
    /// When this consent expires
    pub expires_at: Option<Timestamp>,
    /// Purpose of access
    pub purpose: AccessPurpose,
    /// Can grantee re-share with others?
    pub allow_delegation: bool,
    /// Revocation status
    pub revoked: bool,
    /// Revocation timestamp
    pub revoked_at: Option<Timestamp>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessPurpose {
    Treatment,
    SecondOpinion,
    EmergencyCare,
    Research { study_id: String, irb_approval: String },
    Insurance,
    Legal,
    Personal,
    FamilyCaregiver,
}

/// Emergency access configuration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmergencyAccessConfig {
    /// Patient this config belongs to
    pub patient: AgentPubKey,
    /// Is emergency access enabled?
    pub enabled: bool,
    /// What information is accessible in emergencies
    pub emergency_visible: Vec<HealthRecordType>,
    /// Designated emergency contacts
    pub emergency_contacts: Vec<EmergencyContact>,
    /// Required verification level for emergency access
    pub verification_level: EmergencyVerificationLevel,
    /// Auto-notify these contacts on emergency access
    pub notify_on_access: Vec<AgentPubKey>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct EmergencyContact {
    pub agent: AgentPubKey,
    pub relationship: String,
    pub priority: u8, // 1 = highest
    pub can_make_decisions: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum EmergencyVerificationLevel {
    /// Any verified healthcare provider
    AnyProvider,
    /// Verified emergency services
    EmergencyServicesOnly,
    /// Requires N of M emergency contacts to approve
    MultiSigContacts { required: u8, total: u8 },
    /// Time-delayed access (accessor notifies, patient has N hours to deny)
    TimeDelayed { hours: u8 },
}

/// Access log entry (immutable audit trail)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AccessLog {
    /// Who accessed
    pub accessor: AgentPubKey,
    /// Accessor's credential at time of access
    pub accessor_credential: Option<ActionHash>,
    /// What was accessed
    pub records_accessed: Vec<ActionHash>,
    /// When
    pub accessed_at: Timestamp,
    /// Under what consent/authorization
    pub authorization: AccessAuthorization,
    /// Access type
    pub access_type: AccessType,
    /// Client info (for security monitoring)
    pub client_info: Option<ClientInfo>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessAuthorization {
    ConsentGrant(ActionHash),
    EmergencyAccess { justification: String },
    PatientSelf,
    LegalOrder { court: String, case_number: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessType {
    View,
    Download,
    Share,
    Print,
    Export,
}
```

### Link Types

```rust
#[hdk_link_types]
pub enum LinkTypes {
    // Patient → Records
    PatientToRecord,
    PatientToConsentGrant,
    PatientToEmergencyConfig,
    PatientToAccessLog,

    // Record organization
    RecordToTag,
    RecordToAttachment,
    RecordToRecord, // Related records

    // Provider links
    ProviderToPatientGrant,
    ProviderToAccessLog,

    // Temporal indexing
    YearToRecord,
    MonthToRecord,

    // Type indexing
    RecordTypeToRecord,

    // Emergency
    EmergencyContactToPatient,
}
```

---

## Zome Specifications

### 1. Records Manager Zome

```rust
// records_manager/src/lib.rs

/// Create a new health record
#[hdk_extern]
pub fn create_record(input: CreateRecordInput) -> ExternResult<ActionHash> {
    // Validate record type
    // Encrypt content with patient's key
    // Create entry
    // Link to patient path
    // Link to temporal indexes
    // Link to type index
    // Return hash
}

/// Get patient's records with optional filters
#[hdk_extern]
pub fn get_my_records(filters: RecordFilters) -> ExternResult<Vec<HealthRecord>> {
    let patient = agent_info()?.agent_latest_pubkey;

    // Get records based on filters
    // Decrypt content
    // Return filtered list
}

/// Get records shared with me (as provider)
#[hdk_extern]
pub fn get_shared_records(patient: AgentPubKey) -> ExternResult<Vec<HealthRecord>> {
    let me = agent_info()?.agent_latest_pubkey;

    // Check consent grants from patient to me
    // Filter records by consent scope
    // Log access
    // Return permitted records
}

/// Update a record (creates new version)
#[hdk_extern]
pub fn update_record(input: UpdateRecordInput) -> ExternResult<ActionHash> {
    // Verify ownership
    // Create new entry
    // Link as update
    // Maintain version history
}

/// Delete a record (soft delete with audit)
#[hdk_extern]
pub fn delete_record(record_hash: ActionHash) -> ExternResult<()> {
    // Verify ownership
    // Mark as deleted
    // Log deletion
    // Note: Data may be retained for legal/medical requirements
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RecordFilters {
    pub record_types: Option<Vec<HealthRecordType>>,
    pub date_range: Option<DateRange>,
    pub providers: Option<Vec<AgentPubKey>>,
    pub tags: Option<Vec<String>>,
    pub search_text: Option<String>,
}
```

### 2. Sharing Manager Zome

```rust
// sharing_manager/src/lib.rs

/// Grant access to a provider
#[hdk_extern]
pub fn grant_access(input: GrantAccessInput) -> ExternResult<ActionHash> {
    // Verify I am the patient
    // Verify provider's Praxis credential
    let credential = verify_provider_credential(&input.provider)?;

    // Create consent grant
    let grant = ConsentGrant {
        grantor: agent_info()?.agent_latest_pubkey,
        grantee: input.provider,
        grantee_credential: Some(credential),
        record_types: input.record_types,
        specific_records: input.specific_records,
        date_range: input.date_range,
        expires_at: input.expires_at,
        purpose: input.purpose,
        allow_delegation: input.allow_delegation,
        revoked: false,
        revoked_at: None,
    };

    // Store and link
    let hash = create_entry(&EntryTypes::ConsentGrant(grant))?;

    // Notify provider
    send_signal_to_agent(&input.provider, Signal::ConsentGranted(hash.clone()))?;

    Ok(hash)
}

/// Revoke access
#[hdk_extern]
pub fn revoke_access(grant_hash: ActionHash) -> ExternResult<()> {
    // Get grant
    // Verify I am grantor
    // Update as revoked
    // Notify grantee
}

/// List my consent grants
#[hdk_extern]
pub fn get_my_grants() -> ExternResult<Vec<ConsentGrant>> {
    // Get all grants where I am grantor
}

/// List consents I've received (as provider)
#[hdk_extern]
pub fn get_received_consents() -> ExternResult<Vec<ConsentGrant>> {
    // Get all grants where I am grantee
    // Filter out revoked/expired
}

/// Generate shareable summary (for external sharing)
#[hdk_extern]
pub fn generate_share_package(input: SharePackageInput) -> ExternResult<SharePackage> {
    // Select records
    // Generate encrypted package
    // Create one-time access token
    // Return package with access instructions
}

/// Verify provider credential via Praxis bridge
fn verify_provider_credential(provider: &AgentPubKey) -> ExternResult<ActionHash> {
    // Call Praxis via Bridge Protocol
    let credential = bridge_call(
        "praxis",
        "verify_provider",
        provider,
    )?;

    // Verify credential is valid healthcare provider
    if !credential.is_valid || !credential.has_claim("healthcare_provider") {
        return Err(WasmError::Guest("Invalid provider credential".into()));
    }

    Ok(credential.hash)
}
```

### 3. Emergency Access Zome

```rust
// emergency_access/src/lib.rs

/// Configure emergency access settings
#[hdk_extern]
pub fn configure_emergency_access(config: EmergencyAccessConfig) -> ExternResult<ActionHash> {
    // Verify I am the patient
    // Validate configuration
    // Store config
    // Link emergency contacts
}

/// Request emergency access (for providers)
#[hdk_extern]
pub fn request_emergency_access(input: EmergencyAccessRequest) -> ExternResult<EmergencyAccessResult> {
    let requester = agent_info()?.agent_latest_pubkey;
    let patient = input.patient;

    // Verify requester is verified provider
    let credential = verify_provider_credential(&requester)?;

    // Get patient's emergency config
    let config = get_emergency_config(&patient)?;

    if !config.enabled {
        return Err(WasmError::Guest("Patient has disabled emergency access".into()));
    }

    // Check verification level requirements
    match config.verification_level {
        EmergencyVerificationLevel::AnyProvider => {
            // Credential already verified
        }
        EmergencyVerificationLevel::EmergencyServicesOnly => {
            if !credential.has_claim("emergency_services") {
                return Err(WasmError::Guest("Only emergency services can access".into()));
            }
        }
        EmergencyVerificationLevel::MultiSigContacts { required, total } => {
            // Initiate multi-sig approval process
            return initiate_multisig_emergency(input, required)?;
        }
        EmergencyVerificationLevel::TimeDelayed { hours } => {
            // Initiate time-delayed access
            return initiate_delayed_emergency(input, hours)?;
        }
    }

    // Grant immediate access
    let access_grant = grant_emergency_access(&patient, &requester, &input.justification)?;

    // Notify patient and emergency contacts
    notify_emergency_access(&patient, &config.notify_on_access, &access_grant)?;

    // Log access
    create_access_log(AccessLog {
        accessor: requester,
        accessor_credential: Some(credential.hash),
        records_accessed: vec![], // Will be populated on actual access
        accessed_at: sys_time()?,
        authorization: AccessAuthorization::EmergencyAccess {
            justification: input.justification,
        },
        access_type: AccessType::View,
        client_info: input.client_info,
    })?;

    // Get emergency-visible records
    let records = get_emergency_records(&patient, &config.emergency_visible)?;

    Ok(EmergencyAccessResult {
        granted: true,
        records,
        expires_at: sys_time()?.add_hours(24), // 24-hour emergency access
    })
}

/// Get emergency info without full access (for first responders)
#[hdk_extern]
pub fn get_emergency_summary(patient: AgentPubKey) -> ExternResult<EmergencySummary> {
    // Returns only critical info: allergies, medications, emergency contacts
    // Does not require full provider verification
    // Still logs access attempt

    let config = get_emergency_config(&patient)?;

    if !config.enabled {
        return Err(WasmError::Guest("Patient has disabled emergency access".into()));
    }

    // Get critical information only
    let allergies = get_records_by_type(&patient, HealthRecordType::Allergy)?;
    let medications = get_records_by_type(&patient, HealthRecordType::Medication)?;
    let contacts = config.emergency_contacts;

    Ok(EmergencySummary {
        allergies,
        current_medications: medications,
        emergency_contacts: contacts,
        blood_type: get_blood_type(&patient)?,
        critical_conditions: get_critical_conditions(&patient)?,
    })
}
```

### 4. Research Consent Zome

```rust
// research_consent/src/lib.rs

/// Consent to participate in a research study
#[hdk_extern]
pub fn consent_to_research(input: ResearchConsentInput) -> ExternResult<ActionHash> {
    // Verify study exists and has IRB approval
    let study = verify_research_study(&input.study_id)?;

    // Create consent
    let consent = ResearchConsent {
        patient: agent_info()?.agent_latest_pubkey,
        study_id: input.study_id,
        study_title: study.title,
        irb_approval: study.irb_approval,
        data_types: input.data_types,
        anonymization_level: input.anonymization_level,
        can_recontact: input.can_recontact,
        consented_at: sys_time()?,
        expires_at: input.expires_at,
        withdrawn: false,
    };

    create_entry(&EntryTypes::ResearchConsent(consent))
}

/// Contribute data to federated learning study
#[hdk_extern]
pub fn contribute_to_fl_study(input: FLContributionInput) -> ExternResult<ActionHash> {
    // Verify consent exists for this study
    let consent = get_research_consent(&input.study_id)?;

    // Verify study is active
    let study = verify_research_study(&input.study_id)?;

    // Prepare data according to consent scope
    let data = prepare_fl_data(&consent, &input.data_request)?;

    // Apply differential privacy
    let dp_data = apply_differential_privacy(data, study.privacy_budget)?;

    // Submit to 0TML via bridge
    let contribution = bridge_call(
        "zerotrustml",
        "submit_contribution",
        FLContribution {
            study_id: input.study_id,
            data: dp_data,
            consent_hash: consent.hash,
        },
    )?;

    // Log contribution
    create_entry(&EntryTypes::ResearchContribution(ResearchContribution {
        consent: consent.hash,
        study_id: input.study_id,
        contributed_at: sys_time()?,
        data_types: consent.data_types,
        privacy_budget_used: study.privacy_budget,
    }))
}

/// Withdraw from research
#[hdk_extern]
pub fn withdraw_research_consent(consent_hash: ActionHash) -> ExternResult<()> {
    // Get consent
    // Verify I am patient
    // Mark as withdrawn
    // Notify study via bridge
}

#[derive(Clone, Serialize, Deserialize)]
pub enum AnonymizationLevel {
    /// Full data with identifier removed
    Pseudonymized,
    /// K-anonymity applied
    KAnonymous { k: u32 },
    /// Differential privacy applied
    DifferentiallyPrivate { epsilon: f64 },
    /// Only aggregate statistics
    AggregateOnly,
}
```

### 5. Provider Verification Zome

```rust
// provider_verify/src/lib.rs

/// Verify a provider's credentials (calls Praxis)
#[hdk_extern]
pub fn verify_provider(provider: AgentPubKey) -> ExternResult<ProviderVerification> {
    // Query Praxis via Bridge
    let credentials = bridge_call::<Vec<VerifiableCredential>>(
        "praxis",
        "get_agent_credentials",
        provider.clone(),
    )?;

    // Check for healthcare-related credentials
    let healthcare_creds: Vec<_> = credentials
        .iter()
        .filter(|c| is_healthcare_credential(c))
        .collect();

    if healthcare_creds.is_empty() {
        return Ok(ProviderVerification {
            verified: false,
            provider,
            credentials: vec![],
            specialties: vec![],
            licenses: vec![],
            trust_score: 0.0,
        });
    }

    // Get MATL trust score
    let trust = bridge_call::<f64>(
        "bridge",
        "get_cross_happ_reputation",
        provider.clone(),
    )?;

    // Extract specialties and licenses
    let specialties = extract_specialties(&healthcare_creds);
    let licenses = extract_licenses(&healthcare_creds);

    Ok(ProviderVerification {
        verified: true,
        provider,
        credentials: healthcare_creds.into_iter().map(|c| c.hash).collect(),
        specialties,
        licenses,
        trust_score: trust,
    })
}

/// Check if provider can access specific record type
#[hdk_extern]
pub fn can_access_record_type(
    provider: AgentPubKey,
    record_type: HealthRecordType,
) -> ExternResult<bool> {
    let verification = verify_provider(provider)?;

    if !verification.verified {
        return Ok(false);
    }

    // Some record types require specific specialties
    match record_type {
        HealthRecordType::GeneticTest => {
            Ok(verification.specialties.contains(&"genetics".to_string()))
        }
        HealthRecordType::PathologyReport => {
            Ok(verification.specialties.contains(&"pathology".to_string()) ||
               verification.specialties.contains(&"oncology".to_string()))
        }
        // Most types accessible to any verified provider
        _ => Ok(true),
    }
}

fn is_healthcare_credential(credential: &VerifiableCredential) -> bool {
    credential.credential_type.iter().any(|t| {
        matches!(t.as_str(),
            "MedicalDegree" |
            "NursingLicense" |
            "MedicalLicense" |
            "PharmacyLicense" |
            "HealthcareProviderCredential" |
            "EmergencyMedicalTechnician" |
            "Paramedic"
        )
    })
}
```

### 6. Audit Trail Zome

```rust
// audit_trail/src/lib.rs

/// Create access log entry (called by other zomes)
#[hdk_extern]
pub fn log_access(input: AccessLog) -> ExternResult<ActionHash> {
    // Validate log entry
    // Create immutable entry
    let hash = create_entry(&EntryTypes::AccessLog(input.clone()))?;

    // Link to patient path
    create_link(
        patient_path(&input.records_accessed[0])?,
        hash.clone(),
        LinkTypes::PatientToAccessLog,
        (),
    )?;

    // Link to accessor path (for providers to see their access history)
    create_link(
        accessor_path(&input.accessor)?,
        hash.clone(),
        LinkTypes::ProviderToAccessLog,
        (),
    )?;

    // Check for suspicious patterns
    check_access_patterns(&input)?;

    Ok(hash)
}

/// Get my access history (as patient)
#[hdk_extern]
pub fn get_my_access_history(filters: AccessLogFilters) -> ExternResult<Vec<AccessLog>> {
    let patient = agent_info()?.agent_latest_pubkey;

    // Get all access logs for my records
    let path = patient_access_path(&patient);
    let links = get_links(
        GetLinksInputBuilder::try_new(path.path_entry_hash()?, LinkTypes::PatientToAccessLog)?
            .build()
    )?;

    // Filter and return
    filter_access_logs(links, filters)
}

/// Get my access history (as provider)
#[hdk_extern]
pub fn get_my_provider_access_history() -> ExternResult<Vec<AccessLog>> {
    let provider = agent_info()?.agent_latest_pubkey;

    // Get all my access logs
    let path = accessor_path(&provider);
    let links = get_links(
        GetLinksInputBuilder::try_new(path.path_entry_hash()?, LinkTypes::ProviderToAccessLog)?
            .build()
    )?;

    collect_access_logs(links)
}

/// Check for suspicious access patterns
fn check_access_patterns(log: &AccessLog) -> ExternResult<()> {
    // Get recent logs for this accessor
    let recent_logs = get_recent_access_by_accessor(&log.accessor, 24)?; // Last 24 hours

    // Check for anomalies
    let anomalies = detect_anomalies(&recent_logs, log);

    if !anomalies.is_empty() {
        // Alert Sentinel via bridge
        bridge_call::<()>(
            "sentinel",
            "report_anomaly",
            SecurityAnomaly {
                source: "healthvault".to_string(),
                anomaly_type: AnomalyType::UnusualAccessPattern,
                details: anomalies,
                agent: log.accessor.clone(),
                timestamp: sys_time()?,
            },
        )?;

        // Notify patient
        send_signal_to_patient(
            &log.records_accessed[0],
            Signal::SuspiciousAccessDetected(anomalies),
        )?;
    }

    Ok(())
}

fn detect_anomalies(recent: &[AccessLog], current: &AccessLog) -> Vec<AnomalyDetail> {
    let mut anomalies = vec![];

    // Unusual time (outside normal hours for this provider)
    // Unusual volume (many records in short time)
    // Unusual types (accessing types never accessed before)
    // Unusual patient (accessing patient never treated before)
    // Geographic anomaly (access from unusual location)

    // ... anomaly detection logic ...

    anomalies
}
```

---

## Cross-hApp Integration

### Praxis Integration
```rust
// Verify provider credentials
let credentials = bridge_call::<Vec<VerifiableCredential>>(
    "praxis",
    "get_agent_credentials",
    provider,
)?;
```

### 0TML Integration
```rust
// Participate in federated learning
let contribution = bridge_call::<FLReceipt>(
    "zerotrustml",
    "submit_contribution",
    FLContribution { data, study_id, consent },
)?;
```

### Sentinel Integration
```rust
// Report security anomalies
bridge_call::<()>(
    "sentinel",
    "report_anomaly",
    anomaly_details,
)?;
```

### Attest Integration
```rust
// Verify patient identity for emergency situations
let identity = bridge_call::<IdentityAttestation>(
    "attest",
    "verify_identity",
    patient,
)?;
```

---

## Privacy & Security

### Encryption Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encryption Layers                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Record Encryption                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Patient's personal key encrypts all record content          ││
│  │ Only patient can decrypt by default                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Layer 2: Shared Key for Consent                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ When consent granted, re-encrypt with shared key            ││
│  │ Recipient receives derived key for their access scope       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Layer 3: Emergency Key Sharding                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Emergency key split among emergency contacts                 ││
│  │ Shamir's Secret Sharing: k-of-n threshold                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Layer 4: Network Encryption                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ All DHT operations encrypted in transit                     ││
│  │ Holochain's native encryption layer                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Compliance Considerations

| Regulation | Approach |
|------------|----------|
| HIPAA (US) | Patient controls PHI, audit trails, encryption at rest and transit |
| GDPR (EU) | Right to erasure (soft delete with audit), data portability, consent |
| PIPEDA (CA) | Meaningful consent, individual access, security safeguards |
| HITECH | Breach notification via Sentinel, access logs |

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Unauthorized access | Consent grants, credential verification, encryption |
| Insider threat (provider) | Access logs, anomaly detection, limited scope grants |
| Data breach | Client-side encryption, no central database to breach |
| Identity theft | Praxis credential chains, multi-factor for sensitive ops |
| Coercion | Dead man's switch, time-delayed access options |

---

## User Experience

### Patient Flows

#### Adding a Record
```
1. Patient creates record in app
2. Content encrypted with patient's key
3. Record stored in source chain
4. Indexed by type, date, tags
5. Visible only to patient until shared
```

#### Sharing with Provider
```
1. Provider requests access (or patient initiates)
2. Patient reviews provider's credentials (from Praxis)
3. Patient selects scope (record types, date range, duration)
4. Consent grant created
5. Provider notified
6. Provider can now access permitted records
7. All access logged
```

#### Emergency Scenario
```
1. First responder scans patient's emergency QR/NFC
2. Emergency summary displayed (allergies, meds, contacts)
3. If more access needed, responder requests emergency access
4. Based on patient's config:
   - Immediate access (any verified provider)
   - Multi-sig from emergency contacts
   - Time-delayed (patient can deny)
5. Emergency contacts notified
6. Full audit trail created
```

### Provider Flows

#### Receiving Patient Records
```
1. Patient grants access (or provider requests)
2. Provider receives notification
3. Provider verifies their credentials are current
4. Provider accesses records within scope
5. Access logged
6. Provider can add notes/records (with patient consent)
```

#### Contributing to Research
```
1. Provider enrolls patients in study (with consent)
2. Patient data prepared according to consent scope
3. Differential privacy applied
4. Data submitted to federated learning via 0TML
5. Model updates received
6. No raw data leaves patient's control
```

---

## Epistemic Classification

Health data carries inherent epistemic properties:

| Record Type | Typical E-Level | Typical N-Level | Typical M-Level |
|-------------|-----------------|-----------------|-----------------|
| Lab Result | E3 (Cryptographic) | N2 (Network) | M2 (Persistent) |
| Diagnosis | E2 (Private Verify) | N1 (Communal) | M2 (Persistent) |
| Patient Symptom | E1 (Testimonial) | N0 (Personal) | M1 (Temporal) |
| Wearable Data | E2 (Private Verify) | N0 (Personal) | M1 (Temporal) |
| Genetic Test | E4 (Reproducible) | N2 (Network) | M3 (Foundational) |
| Prescription | E3 (Cryptographic) | N2 (Network) | M2 (Persistent) |

This classification informs:
- How claims should be verified
- Who should agree on validity
- How long to retain

---

## Technical Specifications

### Performance Requirements

| Metric | Target |
|--------|--------|
| Record creation | <500ms |
| Record retrieval | <200ms |
| Consent grant | <1s |
| Emergency access | <5s |
| Search (local) | <1s for 10,000 records |

### Storage Estimates

| Data Type | Size/Record | Retention |
|-----------|-------------|-----------|
| Clinical note | 5-50 KB | Permanent |
| Lab result | 1-5 KB | Permanent |
| Image reference | 100 KB (metadata) | Permanent |
| Wearable data | 100 bytes/reading | 1 year rolling |
| Access log | 500 bytes | Permanent |

### Scalability

- Patient with 10,000+ records over lifetime
- Provider with 1,000+ patient relationships
- Research study with 10,000+ participants

---

## Development Roadmap

### Phase 1: Core Records (Q1 2026)
- [ ] Record creation/retrieval
- [ ] Encryption layer
- [ ] Basic consent grants
- [ ] Access logging

### Phase 2: Provider Integration (Q2 2026)
- [ ] Praxis credential verification
- [ ] Provider onboarding
- [ ] Specialty-based access
- [ ] Audit trail enhancement

### Phase 3: Emergency Access (Q3 2026)
- [ ] Emergency configuration
- [ ] Multi-sig emergency contacts
- [ ] First responder interface
- [ ] Notification system

### Phase 4: Research & FL (Q4 2026)
- [ ] Research consent management
- [ ] 0TML integration
- [ ] Differential privacy
- [ ] Study enrollment

### Phase 5: Production (Q1 2027)
- [ ] Security audit
- [ ] Compliance verification
- [ ] Healthcare system integrations
- [ ] Mobile apps

---

## Success Metrics

| Metric | 1 Year | 3 Years | 5 Years |
|--------|--------|---------|---------|
| Active patients | 1,000 | 50,000 | 500,000 |
| Verified providers | 100 | 5,000 | 50,000 |
| Research studies | 5 | 50 | 200 |
| FL contributions | 1,000 | 100,000 | 1M |
| Emergency accesses | 10 | 1,000 | 10,000 |
| Breaches | 0 | 0 | 0 |

---

## References

- HIPAA Privacy Rule: https://www.hhs.gov/hipaa/
- HL7 FHIR: https://www.hl7.org/fhir/
- W3C Verifiable Credentials: https://www.w3.org/TR/vc-data-model/
- Differential Privacy: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- Mycelix 0TML: /srv/luminous-dynamics/Mycelix-Core/0TML/

---

*"Your health data belongs to you. We're just the mycelium that helps it flow where you need it."*
