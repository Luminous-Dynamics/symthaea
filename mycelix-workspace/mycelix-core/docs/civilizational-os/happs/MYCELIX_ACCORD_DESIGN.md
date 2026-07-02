# Mycelix-Accord: Regulatory Harmony & Tax Compliance

## Vision Statement

*"Sovereignty does not mean isolation. Mycelix-Accord enables agents and communities to operate transparently within existing legal frameworks while preserving the option to transition to self-determined economic models. We render unto Caesar what is Caesar's - on our own terms, with our own records, under our own control."*

---

## Executive Summary

Mycelix-Accord provides the infrastructure for tax compliance, regulatory reporting, and legal harmony across jurisdictions. It enables:

1. **Agent-controlled compliance** - You decide what to report, when, and to whom
2. **Multi-jurisdictional support** - Templates for 50+ tax jurisdictions
3. **Automated event tracking** - Captures taxable events from all economic hApps
4. **Privacy-preserving proofs** - ZKP attestations of compliance without revealing details
5. **Transition pathways** - Supports community-defined contribution models

Accord is not surveillance infrastructure - it's sovereignty-preserving compliance tooling that puts agents in control of their regulatory relationships.

---

## Problem Statement

### The Current Reality

Agents operating in the Mycelix ecosystem face real-world obligations:

1. **Income reporting** - Earnings from Marketplace, Collab, and other hApps
2. **Capital gains** - Token/asset appreciation and trades
3. **Business obligations** - VAT/GST, payroll taxes for organizations
4. **Cross-border complexity** - Digital nomads, remote work, international trade
5. **Audit risk** - Inability to produce records creates legal exposure

### The Existing Options (All Bad)

| Option | Problem |
|--------|---------|
| Ignore taxes | Legal risk, limits adoption |
| Manual tracking | Error-prone, time-consuming |
| Third-party services | Privacy invasion, custody risk |
| Centralized reporting | Against sovereignty principles |

### The Accord Solution

Agent-controlled compliance infrastructure that:
- Captures events automatically from ecosystem activity
- Generates jurisdiction-specific reports on demand
- Allows selective disclosure with cryptographic proofs
- Supports transition to community-defined models

---

## Core Principles

### 1. Agent Sovereignty Over Compliance Data
```
Your economic data belongs to you.
You decide what gets reported.
You control when reports are generated.
You choose which jurisdictions to comply with.
```

### 2. Compliance as Service, Not Surveillance
```
Accord helps YOU comply with YOUR obligations.
It does not report to authorities on your behalf.
It does not share data without explicit consent.
It is a tool, not a monitor.
```

### 3. Jurisdiction Agnosticism
```
No jurisdiction is privileged over another.
Templates are community-maintained.
Local expertise via Guild-certified professionals.
Support for emerging/alternative frameworks.
```

### 4. Transition-Ready Architecture
```
Current: Nation-state tax compliance
Future: Community-defined contribution models
Accord supports both and the transition between.
```

---

## Architecture Overview

### System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ACCORD ECOSYSTEM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                        ┌─────────────────┐                             │
│                        │     AGENT       │                             │
│                        │   Dashboard     │                             │
│                        └────────┬────────┘                             │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        MYCELIX-ACCORD                            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │  │
│  │  │   Event    │ │ Jurisdiction│ │  Report    │ │    ZKP       │  │  │
│  │  │  Tracker   │ │  Templates │ │ Generator  │ │  Attestor    │  │  │
│  │  └──────┬─────┘ └──────┬─────┘ └──────┬─────┘ └──────┬───────┘  │  │
│  │         │              │              │              │           │  │
│  │         └──────────────┴──────────────┴──────────────┘           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                 │                                       │
│         ┌───────────────────────┼───────────────────────┐              │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│  ┌─────────────┐       ┌─────────────┐         ┌─────────────┐        │
│  │ Marketplace │       │  Treasury   │         │   Collab    │        │
│  │  (trades)   │       │ (payments)  │         │ (earnings)  │        │
│  └─────────────┘       └─────────────┘         └─────────────┘        │
│         │                       │                       │              │
│         └───────────────────────┼───────────────────────┘              │
│                                 │                                       │
│                                 ▼                                       │
│                        ┌─────────────┐                                 │
│                        │   Oracle    │                                 │
│                        │(FX rates,   │                                 │
│                        │ valuations) │                                 │
│                        └─────────────┘                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Zome Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ACCORD ZOMES                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INTEGRITY ZOMES                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  accord_types                                                   │   │
│  │  ├── TaxableEvent          (income, gain, transfer, etc.)      │   │
│  │  ├── JurisdictionProfile   (agent's tax residency/obligations) │   │
│  │  ├── TaxTemplate           (jurisdiction-specific rules)       │   │
│  │  ├── ComplianceReport      (generated report)                  │   │
│  │  ├── ZKPAttestation        (proof of compliance)               │   │
│  │  ├── CommunityContribution (alternative to traditional tax)    │   │
│  │  └── AuditTrail            (record of report generation)       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COORDINATOR ZOMES                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  event_tracker                                                  │   │
│  │  ├── subscribe_to_economic_events()                            │   │
│  │  ├── record_taxable_event()                                    │   │
│  │  ├── categorize_event()                                        │   │
│  │  ├── query_events_by_period()                                  │   │
│  │  └── export_events()                                           │   │
│  │                                                                  │   │
│  │  jurisdiction_manager                                           │   │
│  │  ├── set_tax_residency()                                       │   │
│  │  ├── add_jurisdiction()                                        │   │
│  │  ├── get_applicable_rules()                                    │   │
│  │  ├── query_templates()                                         │   │
│  │  └── calculate_obligations()                                   │   │
│  │                                                                  │   │
│  │  report_generator                                               │   │
│  │  ├── generate_report()                                         │   │
│  │  ├── preview_report()                                          │   │
│  │  ├── export_format() // CSV, PDF, tax software formats         │   │
│  │  ├── schedule_report()                                         │   │
│  │  └── archive_report()                                          │   │
│  │                                                                  │   │
│  │  zkp_attestor                                                   │   │
│  │  ├── generate_compliance_proof()                               │   │
│  │  ├── verify_compliance_proof()                                 │   │
│  │  ├── generate_income_range_proof()                             │   │
│  │  └── generate_tax_paid_proof()                                 │   │
│  │                                                                  │   │
│  │  transition_manager                                             │   │
│  │  ├── register_community_model()                                │   │
│  │  ├── calculate_community_contribution()                        │   │
│  │  ├── track_contribution_history()                              │   │
│  │  └── generate_dual_report() // traditional + community         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A taxable economic event captured from ecosystem activity
#[hdk_entry_helper]
pub struct TaxableEvent {
    pub event_id: String,
    pub event_type: TaxableEventType,
    pub timestamp: Timestamp,
    pub source_happ: HAppId,
    pub source_action: ActionHash,

    // Financial details
    pub amount: Decimal,
    pub currency: Currency,
    pub fiat_value_at_time: Option<FiatValue>,
    pub cost_basis: Option<Decimal>,

    // Parties
    pub from_agent: Option<AgentPubKey>,
    pub to_agent: Option<AgentPubKey>,
    pub counterparty_jurisdiction: Option<String>,

    // Classification
    pub category: EventCategory,
    pub subcategory: Option<String>,
    pub description: String,

    // For gains/losses
    pub acquisition_date: Option<Timestamp>,
    pub acquisition_cost: Option<Decimal>,
    pub disposal_proceeds: Option<Decimal>,

    // Metadata
    pub tags: Vec<String>,
    pub attachments: Vec<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TaxableEventType {
    // Income events
    Income {
        income_type: IncomeType,
    },

    // Capital events
    CapitalGain {
        asset_type: AssetType,
        holding_period: Duration,
        gain_amount: Decimal,
    },
    CapitalLoss {
        asset_type: AssetType,
        loss_amount: Decimal,
    },

    // Transfer events
    Transfer {
        transfer_type: TransferType,
    },

    // Business events
    BusinessExpense {
        expense_category: String,
        deductible: bool,
    },

    // Token events
    TokenReceived {
        token_type: TokenType,
        fair_market_value: Decimal,
    },
    TokenDisposed {
        token_type: TokenType,
        method: DisposalMethod,
    },

    // Staking/rewards
    StakingReward {
        protocol: String,
    },

    // Airdrops/gifts
    Airdrop {
        source: String,
    },
    GiftReceived {
        from_relationship: Option<String>,
    },
    GiftGiven {
        to_relationship: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IncomeType {
    Employment,
    SelfEmployment,
    ContractWork,
    Freelance,
    Royalties,
    Interest,
    Dividends,
    Rental,
    CommunityReward,
    BountyPayment,
    Other(String),
}

/// Agent's tax profile and jurisdictional obligations
#[hdk_entry_helper]
pub struct JurisdictionProfile {
    pub profile_id: String,
    pub agent: AgentPubKey,

    // Primary tax residency
    pub primary_jurisdiction: Jurisdiction,
    pub tax_id: Option<EncryptedTaxId>, // Encrypted, agent-controlled

    // Additional obligations
    pub additional_jurisdictions: Vec<Jurisdiction>,

    // Entity type
    pub entity_type: EntityType,

    // Fiscal year
    pub fiscal_year_end: MonthDay,

    // Reporting thresholds
    pub reporting_thresholds: HashMap<String, Decimal>,

    // Preferences
    pub auto_categorization: bool,
    pub notification_preferences: NotificationPreferences,

    // Cost basis method
    pub cost_basis_method: CostBasisMethod,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Jurisdiction {
    pub country_code: String,      // ISO 3166-1 alpha-2
    pub subdivision: Option<String>, // State/province
    pub tax_treaty_status: Vec<TreatyStatus>,
    pub template_hash: ActionHash,  // Link to tax template
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EntityType {
    Individual,
    SoleProprietor,
    Partnership,
    Corporation,
    LLC,
    DAO,
    Cooperative,
    Trust,
    NonProfit,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CostBasisMethod {
    FIFO,       // First In First Out
    LIFO,       // Last In First Out
    HIFO,       // Highest In First Out
    SpecificId, // Specific identification
    Average,    // Average cost
}

/// Jurisdiction-specific tax template
#[hdk_entry_helper]
pub struct TaxTemplate {
    pub template_id: String,
    pub jurisdiction: String,
    pub version: String,
    pub effective_from: Timestamp,
    pub effective_until: Option<Timestamp>,

    // Tax rates
    pub income_tax_brackets: Vec<TaxBracket>,
    pub capital_gains_rates: CapitalGainsRates,
    pub corporate_rate: Option<Decimal>,

    // Rules
    pub rules: Vec<TaxRule>,

    // Forms
    pub required_forms: Vec<FormTemplate>,

    // Deductions and credits
    pub available_deductions: Vec<DeductionType>,
    pub available_credits: Vec<CreditType>,

    // Reporting requirements
    pub reporting_thresholds: HashMap<String, Decimal>,
    pub filing_deadlines: Vec<FilingDeadline>,

    // Maintainer
    pub maintained_by: AgentPubKey,
    pub guild_certified: bool,
    pub certification_hash: Option<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaxBracket {
    pub min_income: Decimal,
    pub max_income: Option<Decimal>,
    pub rate: Decimal,
    pub filing_status: Option<FilingStatus>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CapitalGainsRates {
    pub short_term_rates: Vec<TaxBracket>,
    pub long_term_rates: Vec<TaxBracket>,
    pub holding_period_threshold: Duration,
    pub exemptions: Vec<Exemption>,
}

/// Generated compliance report
#[hdk_entry_helper]
pub struct ComplianceReport {
    pub report_id: String,
    pub agent: AgentPubKey,
    pub jurisdiction: String,
    pub tax_year: u32,
    pub report_type: ReportType,

    // Period
    pub period_start: Timestamp,
    pub period_end: Timestamp,

    // Summary
    pub total_income: Decimal,
    pub total_deductions: Decimal,
    pub taxable_income: Decimal,
    pub total_gains: Decimal,
    pub total_losses: Decimal,
    pub net_capital_gain: Decimal,
    pub estimated_tax: Decimal,

    // Details
    pub income_by_category: HashMap<String, Decimal>,
    pub deductions_by_category: HashMap<String, Decimal>,
    pub capital_events: Vec<ActionHash>,

    // Events included
    pub events_included: Vec<ActionHash>,
    pub events_excluded: Vec<ExcludedEvent>,

    // Generated outputs
    pub form_data: HashMap<String, FormData>,
    pub export_formats: Vec<ExportFormat>,

    // Audit trail
    pub generated_at: Timestamp,
    pub generation_parameters: GenerationParameters,
    pub checksum: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ReportType {
    AnnualIncomeTax,
    QuarterlyEstimate,
    CapitalGainsSummary,
    TransactionHistory,
    AuditPackage,
    CommunityContribution,
    Custom(String),
}

/// ZKP attestation of compliance
#[hdk_entry_helper]
pub struct ZKPAttestation {
    pub attestation_id: String,
    pub agent: AgentPubKey,
    pub attestation_type: AttestationType,
    pub jurisdiction: String,
    pub tax_year: u32,

    // The proof
    pub proof: ZKProof,

    // What it proves (without revealing details)
    pub claim: ComplianceClaim,

    // Verification
    pub verifiable_by: Vec<AgentPubKey>, // Who can verify
    pub expires_at: Option<Timestamp>,

    // Metadata
    pub created_at: Timestamp,
    pub revoked: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AttestationType {
    /// Proves income is within a range without revealing exact amount
    IncomeRange {
        min: Decimal,
        max: Decimal,
    },

    /// Proves tax was paid without revealing amount
    TaxPaid {
        year: u32,
    },

    /// Proves compliance with specific regulation
    RegulatoryCompliance {
        regulation: String,
    },

    /// Proves residency without revealing address
    TaxResidency {
        jurisdiction: String,
    },

    /// Proves business expense without revealing vendor
    BusinessExpenseValid {
        category: String,
        amount_range: (Decimal, Decimal),
    },
}

/// Community-defined contribution model (for transition)
#[hdk_entry_helper]
pub struct CommunityContributionModel {
    pub model_id: String,
    pub community: ActionHash, // Agora governance space
    pub name: String,
    pub description: String,

    // Contribution rules
    pub contribution_type: ContributionType,
    pub calculation_method: CalculationMethod,
    pub rates: Vec<ContributionRate>,

    // Distribution
    pub distribution_pools: Vec<DistributionPool>,

    // Governance
    pub governed_by: ActionHash,
    pub amendment_process: AmendmentProcess,

    // Relationship to traditional tax
    pub replaces_traditional: bool,
    pub parallel_compliance: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ContributionType {
    /// Percentage of income
    IncomePercentage,

    /// Percentage of transactions
    TransactionFee,

    /// Land/resource value
    GeorgistLVT,

    /// Wealth-based
    WealthContribution,

    /// Time-based
    TimeContribution,

    /// Voluntary with social pressure
    VoluntaryWithTransparency,

    /// Custom formula
    Custom { formula: String },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContributionRate {
    pub bracket_min: Decimal,
    pub bracket_max: Option<Decimal>,
    pub rate: Decimal,
    pub applies_to: ContributionBase,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DistributionPool {
    pub pool_name: String,
    pub percentage: Decimal,
    pub purpose: String,
    pub governed_by: ActionHash,
}
```

---

## Key Workflows

### Workflow 1: Automatic Event Capture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVENT CAPTURE FLOW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Economic hApp (Marketplace/Collab/Treasury)                           │
│       │                                                                 │
│       │ Transaction completed                                          │
│       ▼                                                                 │
│  ┌─────────────────┐                                                   │
│  │  Bridge Event   │                                                   │
│  │  Emission       │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ACCORD EVENT TRACKER                          │   │
│  │                                                                  │   │
│  │  1. Receive economic event                                      │   │
│  │  2. Check agent's subscription preferences                      │   │
│  │  3. Query Oracle for FMV at transaction time                    │   │
│  │  4. Auto-categorize based on event type                         │   │
│  │  5. Calculate cost basis (if disposal)                          │   │
│  │  6. Create TaxableEvent entry                                   │   │
│  │  7. Link to source transaction                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  Agent's Local  │                                                   │
│  │  Tax Ledger     │                                                   │
│  └─────────────────┘                                                   │
│                                                                         │
│  Note: All data stored in agent's source chain                         │
│  Agent can disable auto-capture at any time                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 2: Report Generation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REPORT GENERATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent Request                                                          │
│  "Generate 2025 US Federal Income Tax Report"                          │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. LOAD JURISDICTION PROFILE                                   │   │
│  │     • US Federal (primary)                                      │   │
│  │     • California State (additional)                             │   │
│  │     • Cost basis method: FIFO                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. LOAD TAX TEMPLATE                                           │   │
│  │     • US Federal 2025 template                                  │   │
│  │     • Guild-certified: Yes                                      │   │
│  │     • Last updated: 2025-01-15                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. QUERY TAXABLE EVENTS                                        │   │
│  │     • Period: 2025-01-01 to 2025-12-31                         │   │
│  │     • Sources: Marketplace, Collab, Treasury, Mutual            │   │
│  │     • Total events: 847                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  4. CATEGORIZE & CALCULATE                                      │   │
│  │     • Income: $127,450                                          │   │
│  │       - Collab earnings: $95,000                                │   │
│  │       - Marketplace sales: $28,200                              │   │
│  │       - Staking rewards: $4,250                                 │   │
│  │     • Capital Gains: $12,800                                    │   │
│  │       - Short-term: $3,200                                      │   │
│  │       - Long-term: $9,600                                       │   │
│  │     • Deductions: $18,500                                       │   │
│  │       - Home office: $4,800                                     │   │
│  │       - Equipment: $8,200                                       │   │
│  │       - Software: $5,500                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  5. APPLY TAX RULES                                             │   │
│  │     • Taxable income: $108,950                                  │   │
│  │     • Federal tax: $21,234                                      │   │
│  │     • Self-employment tax: $13,516                              │   │
│  │     • Total estimated: $34,750                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  6. GENERATE OUTPUTS                                            │   │
│  │     • Form 1040 data (JSON)                                     │   │
│  │     • Schedule C data (JSON)                                    │   │
│  │     • Schedule D data (JSON)                                    │   │
│  │     • TurboTax import file (.txf)                              │   │
│  │     • CSV transaction history                                   │   │
│  │     • PDF summary report                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  7. ARCHIVE & AUDIT TRAIL                                       │   │
│  │     • Store report hash in Chronicle                            │   │
│  │     • Create audit trail entry                                  │   │
│  │     • Generate report checksum                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 3: ZKP Compliance Attestation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZKP ATTESTATION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Use Case: Proving tax compliance to a landlord without                │
│  revealing exact income                                                 │
│                                                                         │
│  Agent Request                                                          │
│  "Generate proof that my income is above $80,000"                      │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. GATHER PRIVATE INPUTS                                       │   │
│  │     • Actual income: $127,450 (never revealed)                  │   │
│  │     • Tax return hash                                           │   │
│  │     • Supporting event hashes                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. GENERATE ZK CIRCUIT                                         │   │
│  │     • Prove: income >= $80,000                                  │   │
│  │     • Without revealing: exact amount                           │   │
│  │     • Binding to: agent identity, tax year                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. CREATE ATTESTATION                                          │   │
│  │     {                                                           │   │
│  │       "claim": "Income >= $80,000 for 2025",                   │   │
│  │       "jurisdiction": "US",                                     │   │
│  │       "proof": <ZK_PROOF>,                                      │   │
│  │       "verifiable_by": [landlord_pub_key],                     │   │
│  │       "expires": "2026-04-15"                                   │   │
│  │     }                                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  4. SHARE WITH VERIFIER                                         │   │
│  │     • Landlord receives attestation                             │   │
│  │     • Verifies proof cryptographically                          │   │
│  │     • Confirms claim is valid                                   │   │
│  │     • Never learns actual income                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 4: Community Contribution Transition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSITION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: Dual Compliance                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Agent operates in:                                             │   │
│  │  • Traditional jurisdiction (required)                          │   │
│  │  • Community contribution model (voluntary)                     │   │
│  │                                                                  │   │
│  │  Accord generates:                                              │   │
│  │  • Traditional tax report                                       │   │
│  │  • Community contribution report                                │   │
│  │  • Comparison analysis                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 2: Community Recognition                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Community achieves:                                            │   │
│  │  • Special economic zone status, or                             │   │
│  │  • Treaty with traditional jurisdiction, or                     │   │
│  │  • Sufficient autonomy for self-governance                      │   │
│  │                                                                  │   │
│  │  Accord supports:                                               │   │
│  │  • Tracking dual obligations during transition                  │   │
│  │  • Generating compliance proofs for both systems               │   │
│  │  • Historical comparison for policy analysis                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Phase 3: Community Primary                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Agent's primary obligation is community contribution           │   │
│  │                                                                  │   │
│  │  Accord generates:                                              │   │
│  │  • Community contribution report (primary)                      │   │
│  │  • Traditional compliance (as required for external dealings)  │   │
│  │  • Transparency reports for community governance                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### Required Integrations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ACCORD INTEGRATIONS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ECONOMIC hApps (Event Sources)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Marketplace ──► Sales, purchases, trades                       │   │
│  │  Treasury ────► Payments, distributions, grants                 │   │
│  │  Collab ──────► Earnings, contractor payments                   │   │
│  │  Mutual ──────► Insurance payouts, premiums                     │   │
│  │  Cycle ───────► Lending interest, borrowing costs               │   │
│  │  Seed ────────► Investment returns, startup income              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  INFRASTRUCTURE (Support Services)                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Oracle ──────► Exchange rates, FMV at transaction time        │   │
│  │  Attest ──────► Identity verification, tax residency proofs    │   │
│  │  Chronicle ───► Report archival, audit trail                   │   │
│  │  Guild ───────► Certified tax template maintainers             │   │
│  │  Agora ───────► Community contribution model governance        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Event Subscriptions

```rust
/// Accord subscribes to these Bridge events:

const ACCORD_SUBSCRIPTIONS: &[EventSubscription] = &[
    // Marketplace events
    EventSubscription {
        source_happ: HAppId::Marketplace,
        event_types: vec![
            "SaleCompleted",
            "PurchaseCompleted",
            "TradeExecuted",
            "RefundIssued",
        ],
    },

    // Treasury events
    EventSubscription {
        source_happ: HAppId::Treasury,
        event_types: vec![
            "PaymentSent",
            "PaymentReceived",
            "DistributionClaimed",
            "GrantAwarded",
        ],
    },

    // Collab events
    EventSubscription {
        source_happ: HAppId::Collab,
        event_types: vec![
            "CompensationPaid",
            "MilestonePayment",
            "BountyAwarded",
        ],
    },

    // Mutual events
    EventSubscription {
        source_happ: HAppId::Mutual,
        event_types: vec![
            "ClaimPaid",
            "PremiumPaid",
        ],
    },

    // Cycle events
    EventSubscription {
        source_happ: HAppId::Cycle,
        event_types: vec![
            "LoanRepayment",
            "InterestPaid",
            "InterestReceived",
        ],
    },
];
```

---

## Supported Jurisdictions (Initial)

### Tier 1: Full Support (Launch)

| Jurisdiction | Template Status | Guild Maintainer |
|--------------|-----------------|------------------|
| United States (Federal) | Complete | US Tax Guild |
| United States (50 States) | Complete | US Tax Guild |
| United Kingdom | Complete | UK Tax Guild |
| European Union (VAT) | Complete | EU Tax Guild |
| Germany | Complete | DE Tax Guild |
| Canada | Complete | CA Tax Guild |
| Australia | Complete | AU Tax Guild |

### Tier 2: Community Maintained

| Jurisdiction | Template Status | Maintainer |
|--------------|-----------------|------------|
| Japan | In Progress | Community |
| South Korea | In Progress | Community |
| Singapore | Complete | Community |
| Switzerland | Complete | Community |
| Portugal | Complete | Community |
| Estonia | Complete | Community |

### Tier 3: Crypto-Specific

| Jurisdiction | Notes |
|--------------|-------|
| El Salvador | Bitcoin legal tender considerations |
| UAE | Crypto-friendly, no income tax |
| Malta | Crypto regulatory framework |
| Liechtenstein | Blockchain Act compliance |

---

## Privacy Architecture

### Data Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRIVACY LEVELS                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 4: ULTRA-SENSITIVE (Never Shared)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Tax identification numbers                                   │   │
│  │  • Bank account details                                         │   │
│  │  • Social security numbers                                      │   │
│  │  • Full transaction history                                     │   │
│  │                                                                  │   │
│  │  Storage: Encrypted, agent's source chain only                  │   │
│  │  Access: Agent only                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 3: SENSITIVE (Selective Sharing)                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Generated reports                                            │   │
│  │  • Income amounts                                               │   │
│  │  • Tax calculations                                             │   │
│  │                                                                  │   │
│  │  Storage: Encrypted, agent-controlled                           │   │
│  │  Access: Agent + explicit grants                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 2: ATTESTABLE (ZKP Provable)                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Income ranges                                                │   │
│  │  • Compliance status                                            │   │
│  │  • Residency jurisdiction                                       │   │
│  │                                                                  │   │
│  │  Storage: Private, but provable                                 │   │
│  │  Access: Via ZKP attestation only                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 1: AGGREGATE (Community Stats)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Community-wide contribution totals                           │   │
│  │  • Anonymized compliance rates                                  │   │
│  │  • Template usage statistics                                    │   │
│  │                                                                  │   │
│  │  Storage: Differential privacy aggregates                       │   │
│  │  Access: Public (no individual identification)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Interface Concepts

### Dashboard View

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ACCORD - Tax Compliance Dashboard                            [Agent]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TAX YEAR 2025                                   [Change Year ▼]│   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  INCOME SUMMARY                    CAPITAL SUMMARY              │   │
│  │  ┌─────────────────────┐          ┌─────────────────────┐       │   │
│  │  │ Total: $127,450     │          │ Net Gain: $12,800   │       │   │
│  │  │ ████████████████░░░ │          │ ██████████░░░░░░░░░ │       │   │
│  │  │                     │          │                     │       │   │
│  │  │ Collab:    $95,000  │          │ Short-term: $3,200  │       │   │
│  │  │ Market:    $28,200  │          │ Long-term:  $9,600  │       │   │
│  │  │ Staking:   $4,250   │          │                     │       │   │
│  │  └─────────────────────┘          └─────────────────────┘       │   │
│  │                                                                  │   │
│  │  ESTIMATED TAX                     FILING STATUS                │   │
│  │  ┌─────────────────────┐          ┌─────────────────────┐       │   │
│  │  │ Federal:   $21,234  │          │ Jurisdiction: US-CA │       │   │
│  │  │ State:     $8,716   │          │ Entity: Individual  │       │   │
│  │  │ SE Tax:    $13,516  │          │ Method: FIFO        │       │   │
│  │  │ ─────────────────── │          │                     │       │   │
│  │  │ Total:     $43,466  │          │ [Edit Profile]      │       │   │
│  │  └─────────────────────┘          └─────────────────────┘       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  RECENT EVENTS                                    [View All →]  │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Dec 28  Collab        Project Payment      +$4,500    Income   │   │
│  │  Dec 26  Marketplace   NFT Sale             +$890      Cap.Gain │   │
│  │  Dec 24  Treasury      Community Grant      +$1,200    Income   │   │
│  │  Dec 22  Marketplace   Equipment Purchase   -$450      Expense  │   │
│  │  Dec 20  Staking       Weekly Rewards       +$82       Income   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐   │
│  │ Generate Report │ │ Create ZK Proof │ │ Export Transactions     │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Community Contribution View

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ACCORD - Community Contribution                              [Agent]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  YOUR COMMUNITIES                                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │  🌿 Mycelix Commons DAO                                    │  │   │
│  │  │                                                            │  │   │
│  │  │  Model: Progressive Income (5-15%)                        │  │   │
│  │  │  Your 2025 Contribution: $8,450                           │  │   │
│  │  │  Distributed to:                                          │  │   │
│  │  │    • Infrastructure (40%): $3,380                         │  │   │
│  │  │    • Community Fund (30%): $2,535                         │  │   │
│  │  │    • Development (20%): $1,690                            │  │   │
│  │  │    • Emergency Reserve (10%): $845                        │  │   │
│  │  │                                                            │  │   │
│  │  │  [View Details] [Governance →]                            │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │  🏘️ Local Bioregion Cooperative                           │  │   │
│  │  │                                                            │  │   │
│  │  │  Model: Land Value Contribution (Georgist)                │  │   │
│  │  │  Your 2025 Contribution: $2,400                           │  │   │
│  │  │  Based on: Land value assessment                          │  │   │
│  │  │                                                            │  │   │
│  │  │  [View Details] [Governance →]                            │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  COMPARISON: Traditional vs Community                           │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  Traditional Tax (US Federal + CA):     $43,466                │   │
│  │  Community Contributions:               $10,850                 │   │
│  │                                                                  │   │
│  │  Note: Community contributions may be deductible as             │   │
│  │  charitable donations. Consult Guild-certified advisor.        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Tax data theft | All sensitive data encrypted, agent-controlled keys |
| False reporting | Cryptographic binding to source transactions |
| Template manipulation | Guild certification, version pinning, audit trail |
| Government surveillance | No central database, ZKP for compliance proofs |
| Jurisdiction shopping | Transparent residency tracking, no hiding |

### Audit Trail

Every report generation creates an immutable audit trail:

```rust
pub struct AuditEntry {
    pub action: AuditAction,
    pub agent: AgentPubKey,
    pub timestamp: Timestamp,
    pub parameters: HashMap<String, String>,
    pub result_hash: String,
    pub events_included: u64,
    pub template_version: String,
}
```

---

## Implementation Priority

### Phase 1: Core Infrastructure
1. Event tracking from Marketplace, Collab, Treasury
2. US Federal template (most common need)
3. Basic report generation
4. CSV/PDF export

### Phase 2: Expanded Jurisdiction
5. US State templates (CA, NY, TX, FL priority)
6. UK, EU VAT templates
7. Multi-jurisdiction profiles
8. Tax software integrations (TurboTax, H&R Block)

### Phase 3: Advanced Features
9. ZKP attestations
10. Guild certification system
11. Community contribution models
12. Transition pathway tooling

### Phase 4: Ecosystem Integration
13. Full Oracle integration (real-time FMV)
14. Arbiter integration (tax disputes)
15. Diplomat integration (cross-community treaties)

---

## Conclusion

Mycelix-Accord enables the ecosystem to operate legitimately within existing legal frameworks while building the infrastructure for communities to transition to self-determined contribution models. It embodies the principle that sovereignty and compliance are not opposites - agents can meet their obligations while maintaining control over their data and their future.

*"Pay your taxes. Keep your sovereignty. Build the alternative."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Oracle, Marketplace, Treasury, Collab, Chronicle, Attest, Guild, Agora*
