# Mycelix Legacy Bridge: Transition Infrastructure

## Overview

The Legacy Bridge provides infrastructure for seamless transition from existing Web2/traditional systems to the Mycelix ecosystem. It enables progressive adoption, data migration, and continued interoperability with legacy systems.

---

## Design Philosophy

### Progressive Sovereignty

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADOPTION PATHWAY                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: OBSERVE                                                       │
│  • Read-only access to Mycelix data                                    │
│  • No identity required                                                │
│  • Explore communities and resources                                   │
│                                                                         │
│  PHASE 2: PARTICIPATE                                                   │
│  • Create identity (L0-L1)                                             │
│  • Join communities                                                    │
│  • Basic interactions                                                  │
│  • Keep legacy accounts                                                │
│                                                                         │
│  PHASE 3: INTEGRATE                                                     │
│  • Import data from legacy systems                                     │
│  • Build reputation (MATL)                                             │
│  • Use multiple hApps                                                  │
│  • Maintain bridges to legacy                                          │
│                                                                         │
│  PHASE 4: MIGRATE                                                       │
│  • Primary activity in Mycelix                                         │
│  • Export from legacy systems                                          │
│  • Full identity (L3+)                                                 │
│  • Reduce legacy dependency                                            │
│                                                                         │
│  PHASE 5: SOVEREIGN                                                     │
│  • Full participation                                                  │
│  • May maintain bridges for interop                                    │
│  • Contribute to ecosystem                                             │
│  • Help others transition                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bridge Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LEGACY BRIDGE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      LEGACY SYSTEMS                              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ Social  │ │ Banking │ │ Health  │ │ Govt ID │ │Commerce │   │   │
│  │  │ Media   │ │  APIs   │ │ Systems │ │ Systems │ │Platforms│   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │   │
│  └───────┼──────────┼──────────┼──────────┼──────────┼───────────┘   │
│          │          │          │          │          │               │
│          ▼          ▼          ▼          ▼          ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BRIDGE ADAPTERS                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ OAuth/  │ │ Open    │ │ HL7/    │ │ Gov     │ │ Payment │   │   │
│  │  │ OIDC    │ │ Banking │ │ FHIR    │ │ APIs    │ │ Rails   │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │   │
│  └───────┼──────────┼──────────┼──────────┼──────────┼───────────┘   │
│          │          │          │          │          │               │
│          ▼          ▼          ▼          ▼          ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BRIDGE CORE                                   │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │ Data Transform│ │ Identity Map  │ │ Credential Convert    │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │ Sync Engine   │ │ Conflict Res  │ │ Audit Trail           │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │          │          │          │          │               │
│          ▼          ▼          ▼          ▼          ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MYCELIX ECOSYSTEM                             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ Weave   │ │Treasury │ │HealthV  │ │ Attest  │ │Marketpl │   │   │
│  │  │ (Social)│ │(Finance)│ │(Health) │ │(Identity│ │(Commerce│   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bridge Categories

### 1. Identity Bridges

```rust
pub struct IdentityBridge {
    pub bridge_id: String,
    pub bridge_type: IdentityBridgeType,
    pub status: BridgeStatus,
    pub mycelix_identity: AgentPubKey,
    pub external_identity: ExternalIdentity,
    pub verification_status: VerificationStatus,
    pub last_sync: Option<Timestamp>,
}

pub enum IdentityBridgeType {
    // Social
    OAuth2 {
        provider: String,               // google, github, etc.
        scopes: Vec<String>,
    },
    SocialMedia {
        platform: SocialPlatform,
        username: String,
    },

    // Government
    GovernmentID {
        country: String,
        id_type: GovernmentIDType,
    },
    eIDAS {
        trust_level: EIDASLevel,
    },

    // Professional
    LinkedIn,
    ProfessionalLicense {
        license_type: String,
        issuer: String,
    },

    // Financial
    BankAccount {
        institution: String,
        verified: bool,
    },

    // Crypto
    EthereumAddress(String),
    BitcoinAddress(String),
    ENS(String),
}

pub enum GovernmentIDType {
    Passport,
    NationalID,
    DriversLicense,
    StateID,
    ResidencePermit,
}
```

### 2. Data Migration Bridges

```rust
pub struct DataMigrationBridge {
    pub bridge_id: String,
    pub source_system: SourceSystem,
    pub target_happ: HAppId,
    pub migration_config: MigrationConfig,
    pub status: MigrationStatus,
    pub progress: MigrationProgress,
}

pub enum SourceSystem {
    // Social
    Facebook { export_file: String },
    Twitter { archive_path: String },
    Instagram { export_path: String },
    LinkedIn { export_path: String },

    // Communication
    Email { format: EmailFormat },
    Slack { workspace: String },
    Discord { server_id: String },

    // Financial
    BankStatement { format: StatementFormat },
    Quickbooks { company_file: String },
    Mint { export_path: String },
    Venmo { export_path: String },

    // Health
    AppleHealth { export_path: String },
    GoogleFit { export_path: String },
    MyChart { patient_id: String },
    FHIR { endpoint: String },

    // Productivity
    GoogleDrive { folder_id: String },
    Dropbox { path: String },
    Notion { workspace: String },
    Evernote { notebook: String },

    // Commerce
    Amazon { order_history: String },
    Ebay { export_path: String },
    Etsy { shop_id: String },

    // Custom
    CSV { schema: Schema },
    JSON { schema: Schema },
    API { endpoint: String, auth: AuthConfig },
}

pub struct MigrationConfig {
    pub field_mappings: Vec<FieldMapping>,
    pub transformations: Vec<Transformation>,
    pub conflict_resolution: ConflictResolution,
    pub privacy_filter: PrivacyFilter,
    pub batch_size: u32,
    pub rate_limit: Option<RateLimit>,
}
```

### 3. Real-Time Sync Bridges

```rust
pub struct SyncBridge {
    pub bridge_id: String,
    pub sync_type: SyncType,
    pub direction: SyncDirection,
    pub source: SyncEndpoint,
    pub target: SyncEndpoint,
    pub sync_config: SyncConfig,
    pub status: SyncStatus,
}

pub enum SyncType {
    // Continuous sync
    Bidirectional,
    MirrorFromLegacy,
    MirrorToLegacy,

    // Triggered sync
    OnDemand,
    Scheduled { cron: String },
    WebhookTriggered,
}

pub enum SyncDirection {
    LegacyToMycelix,
    MycelixToLegacy,
    Bidirectional,
}

pub struct SyncConfig {
    pub conflict_strategy: ConflictStrategy,
    pub filter: SyncFilter,
    pub transform: Option<TransformPipeline>,
    pub batch_size: u32,
    pub retry_config: RetryConfig,
    pub audit_all: bool,
}

pub enum ConflictStrategy {
    LegacyWins,
    MycelixWins,
    MostRecent,
    Manual,
    Merge { merge_strategy: MergeStrategy },
}
```

### 4. Payment Bridges

```rust
pub struct PaymentBridge {
    pub bridge_id: String,
    pub bridge_type: PaymentBridgeType,
    pub mycelix_treasury: ActionHash,
    pub external_account: ExternalAccount,
    pub supported_operations: Vec<PaymentOperation>,
}

pub enum PaymentBridgeType {
    // Traditional
    BankTransfer {
        routing_number: String,
        account_type: AccountType,
    },
    CreditCard {
        processor: String,
        merchant_id: String,
    },
    PayPal,
    Stripe,
    Square,

    // Crypto
    Ethereum {
        contract_address: Option<String>,
    },
    Bitcoin {
        network: BitcoinNetwork,
    },
    Stablecoin {
        token: String,
        chain: String,
    },

    // Regional
    PIX,                               // Brazil
    UPI,                               // India
    Alipay,
    WeChatPay,
    SEPA,                              // EU
}

pub enum PaymentOperation {
    Deposit,
    Withdrawal,
    Escrow,
    Recurring,
    Refund,
}
```

---

## Specific Bridge Designs

### Social Media Import

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOCIAL MEDIA IMPORT                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SUPPORTED PLATFORMS:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Facebook  → Weave (relationships), Hearth (posts), Chronicle   │   │
│  │  Twitter   → Hearth (posts), Weave (followers)                  │   │
│  │  Instagram → Hearth (media), Weave (relationships)              │   │
│  │  LinkedIn  → Attest (credentials), Weave (professional)         │   │
│  │  Reddit    → Hearth (posts), community membership               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  IMPORT FLOW:                                                           │
│  1. User requests data export from platform                            │
│  2. User uploads export to Bridge                                      │
│  3. Bridge parses and transforms data                                  │
│  4. User reviews what will be imported                                 │
│  5. User selects privacy levels for each category                      │
│  6. Import executed to appropriate hApps                               │
│  7. Original data can be deleted from legacy platform                  │
│                                                                         │
│  MAPPING EXAMPLES:                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Facebook friends → Weave relationships (need confirmation)     │   │
│  │  Facebook posts → Hearth works (private by default)            │   │
│  │  Facebook groups → Potential Agora communities                  │   │
│  │  Facebook events → Nexus events (historical)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Banking Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BANKING INTEGRATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  OPEN BANKING (where available):                                       │
│  • Account balance sync                                                │
│  • Transaction history import                                          │
│  • Payment initiation                                                  │
│  • Identity verification                                               │
│                                                                         │
│  INTEGRATION FLOW:                                                      │
│  1. User authorizes via Open Banking consent                           │
│  2. Bridge connects to bank API                                        │
│  3. Transactions mapped to Accord categories                           │
│  4. Balance available in Treasury view                                 │
│  5. Payments can be initiated from Treasury                           │
│                                                                         │
│  PRIVACY:                                                               │
│  • User controls what data syncs                                       │
│  • Categories can be anonymized                                        │
│  • No data shared without explicit consent                             │
│  • Audit trail of all access                                           │
│                                                                         │
│  SUPPORTED STANDARDS:                                                   │
│  • Open Banking UK (PSD2)                                              │
│  • EU PSD2                                                             │
│  • US Open Banking (emerging)                                          │
│  • Australia CDR                                                       │
│  • Brazil Open Finance                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Health Data Import

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HEALTH DATA IMPORT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SOURCES:                                                               │
│  • Apple Health (HealthKit export)                                     │
│  • Google Fit                                                          │
│  • Fitbit                                                              │
│  • Electronic Health Records (via FHIR)                                │
│  • MyChart/Epic patient portal                                         │
│  • Lab results (various formats)                                       │
│                                                                         │
│  TARGET: HealthVault                                                    │
│                                                                         │
│  FHIR RESOURCES MAPPED:                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Patient → HealthVault profile                                  │   │
│  │  Condition → Health conditions list                             │   │
│  │  MedicationStatement → Medication tracking                      │   │
│  │  Observation → Vitals, lab results                             │   │
│  │  Immunization → Vaccination records                             │   │
│  │  Procedure → Medical history                                    │   │
│  │  AllergyIntolerance → Allergy list                             │   │
│  │  DocumentReference → Medical documents                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PRIVACY:                                                               │
│  • All health data encrypted at rest                                   │
│  • User controls all sharing                                           │
│  • Emergency access protocols respected                                │
│  • HIPAA/GDPR compliance maintained                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Government ID Verification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GOVERNMENT ID BRIDGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PURPOSE: Elevate Attest identity level using government verification  │
│                                                                         │
│  METHODS:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  eIDAS (EU): Connect to qualified trust services                │   │
│  │  India: Aadhaar verification (optional)                         │   │
│  │  USA: ID.me, Login.gov integration                              │   │
│  │  UK: GOV.UK Verify                                              │   │
│  │  Estonia: e-Residency                                           │   │
│  │  Document scan: AI verification + human review                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  IDENTITY LEVEL MAPPING:                                               │
│  • Government verified → Attest L4                                     │
│  • Biometric verified → Attest L5                                      │
│                                                                         │
│  PRIVACY PRESERVING:                                                   │
│  • Only verification status stored, not ID details                    │
│  • ZKP for "over 18" without revealing DOB                            │
│  • Jurisdiction stored, not address                                   │
│  • User can revoke and re-verify                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Export Capabilities

### Data Portability

Every hApp MUST support data export in standard formats:

```rust
pub struct ExportCapability {
    pub happ: HAppId,
    pub supported_formats: Vec<ExportFormat>,
    pub export_scope: ExportScope,
    pub scheduling: ExportScheduling,
}

pub enum ExportFormat {
    // Universal
    JSON,
    CSV,
    XML,

    // Domain specific
    ActivityPub,                       // Social data
    FHIR,                              // Health data
    iCalendar,                         // Events
    vCard,                             // Contacts
    OpenDocument,                      // Documents
    MBOX,                              // Messages

    // Interop
    W3CVerifiableCredential,
    W3CDID,
}

pub enum ExportScope {
    Everything,
    ByDateRange { start: Timestamp, end: Timestamp },
    ByType(Vec<String>),
    Custom { filter: ExportFilter },
}

pub enum ExportScheduling {
    OnDemand,
    Recurring { cron: String, destination: ExportDestination },
    Automatic { destination: ExportDestination },
}
```

---

## Bridge Security

### Security Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BRIDGE SECURITY                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  AUTHENTICATION                                                         │
│  • OAuth 2.0 for external service auth                                 │
│  • Credentials stored encrypted                                        │
│  • Token refresh handled automatically                                 │
│  • Revocation propagated immediately                                   │
│                                                                         │
│  AUTHORIZATION                                                          │
│  • Minimum necessary permissions requested                             │
│  • Scopes clearly explained to user                                    │
│  • Regular permission audits                                           │
│  • Easy revocation                                                     │
│                                                                         │
│  DATA IN TRANSIT                                                        │
│  • TLS 1.3 for all connections                                        │
│  • Certificate pinning for known services                              │
│  • End-to-end encryption where possible                               │
│                                                                         │
│  DATA AT REST                                                           │
│  • Bridge credentials encrypted                                        │
│  • Temporary data purged after migration                               │
│  • Audit logs preserved                                                │
│                                                                         │
│  MONITORING                                                             │
│  • All bridge operations logged                                        │
│  • Anomaly detection for unusual patterns                              │
│  • User notification of bridge activity                                │
│  • Regular security reviews                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Experience

### Onboarding Flow

```
Welcome to Mycelix!
        │
        ▼
┌─────────────────┐
│ Create Identity │ ← Basic pseudonymous identity
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Would you like to import existing   │
│ accounts and data?                  │
│                                     │
│ [Yes, help me migrate]              │
│ [No, start fresh]                   │
│ [Maybe later]                       │
└────────┬────────────────────────────┘
         │
         ▼ (if yes)
┌─────────────────────────────────────┐
│ Select what you'd like to import:   │
│                                     │
│ [ ] Social connections (Facebook,   │
│     Twitter, etc.)                  │
│ [ ] Financial history (banks)       │
│ [ ] Health records                  │
│ [ ] Professional credentials        │
│ [ ] Photos and media                │
│ [ ] Messages and communications     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Privacy Review                      │
│                                     │
│ For each category, choose who can   │
│ see this data in Mycelix:          │
│                                     │
│ ○ Only me                          │
│ ○ Trusted circle                    │
│ ○ My communities                    │
│ ○ Public                           │
└─────────────────────────────────────┘
```

---

## Conclusion

The Legacy Bridge enables graceful transition from existing systems to the Mycelix ecosystem. By supporting progressive adoption, comprehensive data migration, and continued interoperability, it ensures that the path to sovereignty doesn't require abandoning everything at once.

*"The bridge to a new world must be sturdy enough to carry the old. Legacy Bridge ensures no one is left behind."*

---

*Document Version: 1.0*
*Applies to: All external integrations*
