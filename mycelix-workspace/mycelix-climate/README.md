# Mycelix Climate

**Climate coordination and carbon management for the Mycelix Civilizational OS**

## Overview

Mycelix Climate provides decentralized climate coordination infrastructure including carbon credit tracking, Renewable Energy Certificates (RECs), an offset marketplace, and GHG Protocol emissions reporting. This hApp integrates with the Energy hApp for renewable generation verification and Supply Chain for Scope 3 emissions tracking.

## Zomes

### carbon
Carbon credit management with provenance tracking:
- **CarbonCredit**: Verified carbon offset credits with full provenance
- **CarbonProject**: Reforestation, renewable energy, methane capture projects
- **Retirement**: Permanently retired credits (prevents double-counting)
- **Verification**: Third-party verification records (Verra VCS, Gold Standard, etc.)

### recs
Renewable Energy Certificate management:
- **RenewableCertificate**: RECs from verified renewable sources
- **Generator**: Registered renewable energy generators
- **Transfer**: REC ownership transfer records
- **Retirement**: Retired RECs for compliance purposes

### offsets
Offset marketplace for credits and RECs:
- **OffsetListing**: Credits/RECs available for purchase
- **Purchase**: Completed offset purchases
- **Subscription**: Recurring offset purchases (monthly neutrality)
- **PriceDiscovery**: Market pricing and orderbook

### reporting
GHG Protocol emissions tracking:
- **EmissionsReport**: Organizational emissions data
- **Scope1**: Direct emissions (owned sources)
- **Scope2**: Indirect emissions (purchased energy)
- **Scope3**: Value chain emissions (supply chain)
- **Reduction**: Verified emissions reductions
- **Target**: Net-zero commitments and tracking

### bridge
Cross-hApp integration:
- Link to Energy hApp for renewable generation verification
- Link to Supply Chain hApp for Scope 3 emissions data
- MATL reputation for verified green claims
- Export compliance reports (CDP, SBTi, etc.)

## Verification Standards

Supported verification standards:
- **Verra VCS** (Verified Carbon Standard)
- **Gold Standard** (CDM and VER projects)
- **American Carbon Registry** (ACR)
- **Climate Action Reserve** (CAR)

## Architecture

```
mycelix-climate/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── carbon/
│   │   ├── integrity/        # Carbon credit validation
│   │   └── coordinator/      # Credit management
│   ├── recs/
│   │   ├── integrity/        # REC validation
│   │   └── coordinator/      # Certificate management
│   ├── offsets/
│   │   ├── integrity/        # Marketplace validation
│   │   └── coordinator/      # Trading operations
│   ├── reporting/
│   │   ├── integrity/        # Emissions validation
│   │   └── coordinator/      # Report management
│   └── bridge/
│       ├── integrity/        # Integration validation
│       └── coordinator/      # Cross-hApp ops
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Carbon Credit Lifecycle

```
Project Registration
        │
        ▼
   Verification (3rd party)
        │
        ▼
   Credit Issuance
        │
        ├──► Trading (marketplace)
        │
        ▼
   Retirement (permanent)
```

## Integration Points

- **mycelix-energy**: Renewable generation verification for RECs
- **mycelix-supplychain**: Scope 3 emissions data
- **mycelix-identity**: Organization and verifier credentials
- **mycelix-governance**: Climate policy decisions
- **mycelix-finance**: Payment for offsets

## Building

```bash
# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package the hApp
hc app pack .
```

## Example Usage

```typescript
// Create a carbon project
const project = await climateClient.createCarbonProject({
  name: "Amazon Reforestation Initiative",
  projectType: "reforestation",
  location: { lat: -3.4653, lng: -62.2159 },
  estimatedTonnesCO2: 100000,
  standard: "verra_vcs",
  methodology: "VM0007",
});

// Register verification
const verification = await climateClient.registerVerification({
  projectId: project.id,
  verifier: "SCS Global Services",
  standard: "verra_vcs",
  vintage: 2024,
  verifiedTonnes: 95000,
});

// Issue credits
const credits = await climateClient.issueCredits({
  projectId: project.id,
  verificationId: verification.id,
  tonnes: 95000,
  serialNumberStart: "VCS-2024-001-0001",
});

// Retire credits (permanent)
await climateClient.retireCredits({
  creditId: credits.id,
  tonnes: 10,
  beneficiary: "Luminous Dynamics",
  retirementReason: "2024 Q1 Carbon Neutrality",
});
```

## License

Apache-2.0
