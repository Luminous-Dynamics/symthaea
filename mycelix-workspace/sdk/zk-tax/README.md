# Mycelix ZK Tax SDK

Zero-knowledge proofs for privacy-preserving tax compliance. **Prove your tax bracket without revealing your income.**

[![Crates.io](https://img.shields.io/crates/v/mycelix-zk-tax)](https://crates.io/crates/mycelix-zk-tax)
[![npm](https://img.shields.io/npm/v/@mycelix/zk-tax)](https://www.npmjs.com/package/@mycelix/zk-tax)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)

## Features

- 🔐 **Privacy-Preserving**: STARK proofs via RISC Zero zkVM - income never revealed
- 🌍 **58 Countries**: Americas, Europe, Asia-Pacific, Middle East, and Africa
- 🇺🇸 **Sub-national**: US states (all 50), Swiss cantons (26), Canadian provinces
- ⚡ **High Performance**: Real ZK proofs in ~60-80s, dev mode <1ms
- 💾 **Smart Caching**: LRU eviction, TTL, global cache singleton
- 🌐 **Multi-Platform**: CLI, Rust library, WASM/npm, Python/PyPI
- 🔗 **Holochain Ready**: DHT integration for decentralized identity

## Quick Start

### CLI

```bash
# Install
cargo install mycelix-zk-tax --features cli

# Generate a proof (dev mode - instant)
zk-tax prove --income 85000 --jurisdiction US --status single

# Generate a real ZK proof (~60-80s)
zk-tax prove --income 85000 --jurisdiction US --real

# List all G20 jurisdictions
zk-tax list jurisdictions

# Show brackets for a country
zk-tax list brackets --jurisdiction CN
```

### Rust Library

```rust
use mycelix_zk_tax::{TaxBracketProver, Jurisdiction, FilingStatus};

// Create a prover (dev mode for testing)
let prover = TaxBracketProver::dev_mode();

// Generate proof - income stays PRIVATE!
let proof = prover.prove(
    85_000,                     // Your income (never revealed)
    Jurisdiction::US,
    FilingStatus::Single,
    2024,
).unwrap();

// Public output (safe to share)
println!("Bracket: {}", proof.bracket_index);  // 2
println!("Rate: {}%", proof.rate_bps / 100);   // 22%
println!("Commitment: {}", proof.commitment);  // Cryptographic proof
```

### JavaScript/TypeScript (WASM)

```typescript
import init, { WasmTaxProver } from '@mycelix/zk-tax';

await init();

const prover = new WasmTaxProver();
const proof = prover.prove(BigInt(85000), "US", "single", 2024);

console.log(`Bracket: ${proof.bracket_index}`);      // 2
console.log(`Rate: ${proof.rate_percent}%`);         // 22%
console.log(`Valid: ${proof.verify()}`);             // true
```

## Installation

### Rust

```toml
[dependencies]
mycelix-zk-tax = { version = "0.1", features = ["prover"] }
```

### npm

```bash
npm install @mycelix/zk-tax
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `prover` | Full proof generation with Risc0 zkVM |
| `verifier-only` | Lightweight verification only |
| `wasm` | WebAssembly support for browsers |
| `cli` | Command-line interface |
| `cuda` | GPU acceleration (10-20x faster) |
| `metal` | Apple Silicon acceleration |

## Supported Jurisdictions

### 58 Countries Worldwide

| Region | Countries |
|--------|-----------|
| **Americas (10)** | 🇺🇸 US, 🇨🇦 CA, 🇲🇽 MX, 🇧🇷 BR, 🇦🇷 AR, 🇨🇱 CL, 🇨🇴 CO, 🇵🇪 PE, 🇪🇨 EC, 🇺🇾 UY |
| **Europe (23)** | 🇬🇧 UK, 🇩🇪 DE, 🇫🇷 FR, 🇮🇹 IT, 🇪🇸 ES, 🇳🇱 NL, 🇧🇪 BE, 🇦🇹 AT, 🇵🇹 PT, 🇮🇪 IE, 🇵🇱 PL, 🇸🇪 SE, 🇩🇰 DK, 🇫🇮 FI, 🇳🇴 NO, 🇨🇭 CH, 🇨🇿 CZ, 🇬🇷 GR, 🇭🇺 HU, 🇷🇴 RO, 🇷🇺 RU, 🇹🇷 TR, 🇺🇦 UA |
| **Asia-Pacific (15)** | 🇯🇵 JP, 🇨🇳 CN, 🇮🇳 IN, 🇰🇷 KR, 🇮🇩 ID, 🇦🇺 AU, 🇳🇿 NZ, 🇸🇬 SG, 🇭🇰 HK, 🇹🇼 TW, 🇲🇾 MY, 🇹🇭 TH, 🇻🇳 VN, 🇵🇭 PH, 🇵🇰 PK |
| **Middle East (5)** | 🇸🇦 SA, 🇦🇪 AE, 🇮🇱 IL, 🇪🇬 EG, 🇶🇦 QA |
| **Africa (5)** | 🇿🇦 ZA, 🇳🇬 NG, 🇰🇪 KE, 🇲🇦 MA, 🇬🇭 GH |

### Sub-national Jurisdictions

#### US States

```rust
use mycelix_zk_tax::subnational::{USState, find_state_bracket};

// California (1-13.3% progressive)
let bracket = find_state_bracket(100_000, USState::CA, 2024, FilingStatus::Single)?;

// Texas (no income tax)
let bracket = find_state_bracket(1_000_000, USState::TX, 2024, FilingStatus::Single)?;
assert_eq!(bracket.rate_bps, 0);
```

**Progressive tax states**: CA, NY, NJ, OR, MN, HI, VT, IA, WI, ME
**Flat tax states**: IL (4.95%), MA (5%), CO (4.4%), NC (4.75%)
**No income tax**: TX, FL, WA, NV, WY, SD, AK, TN, NH

#### Canadian Provinces

```rust
use mycelix_zk_tax::subnational::{CanadianProvince, find_provincial_bracket};

// Ontario (5.05-13.16%)
let bracket = find_provincial_bracket(100_000, CanadianProvince::ON, 2024, FilingStatus::Single)?;

// Quebec (highest: 14-25.75%)
let bracket = find_provincial_bracket(150_000, CanadianProvince::QC, 2024, FilingStatus::Single)?;
```

**Provinces**: ON, QC, BC, AB (flat 10%), MB, SK, NS, NB, NL, PE, NT, YT, NU

## Filing Statuses

| Status | Code | Available In |
|--------|------|--------------|
| Single | `single` | All countries |
| Married Filing Jointly | `mfj` | US, DE |
| Married Filing Separately | `mfs` | US |
| Head of Household | `hoh` | US |

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Dev Mode Proof | <1ms | For testing only |
| Real ZK Proof (CPU) | ~60-80s | Full STARK proof |
| Real ZK Proof (CUDA) | ~5-8s | GPU accelerated |
| Verification | ~50ms | Constant time |

## Integrations

### Solidity Smart Contract

```solidity
import "./TaxBracketVerifier.sol";

// Verify a proof on-chain
TaxBracketVerifier verifier = new TaxBracketVerifier();

TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
    jurisdiction: TaxBracketVerifier.Jurisdiction.US,
    taxYear: 2024,
    bracketIndex: 2,
    rateBps: 2200,
    bracketLower: 47150,
    bracketUpper: 100525,
    commitment: commitment,
    isDevMode: false,
    risc0Receipt: receipt
});

verifier.verifyProof(input);

// Query proofs
bool hasProof = verifier.hasProofInRange(
    userAddress,
    TaxBracketVerifier.Jurisdiction.US,
    2024,
    2,  // min bracket
    4   // max bracket
);
```

### Holochain Zome

The `tax_compliance` zome stores ZK proof receipts in the DHT:

```rust
// Store a proof receipt
let receipt = TaxProofReceipt::new(
    proof_id,
    "US".to_string(),
    "Single".to_string(),
    2024,
    2,      // bracket_index
    2200,   // rate_bps (22%)
    47150,  // bracket_lower
    100525, // bracket_upper
    commitment_hex,
    false,  // is_dev_mode
);

create_proof_receipt(receipt)?;

// Query proofs for verification
let proofs = get_proofs_by_year(2024)?;
```

## Advanced Proof Types

### Effective Tax Rate Proofs

Prove your effective (average) tax rate without revealing income:

```rust
use mycelix_zk_tax::{EffectiveTaxRateProof, EffectiveTaxRateBuilder, Jurisdiction, FilingStatus};

// Direct proof
let proof = EffectiveTaxRateProof::prove_dev(
    85_000,
    Jurisdiction::US,
    FilingStatus::Single,
    2024,
)?;

println!("Effective rate: {:.1}%", proof.effective_rate_percent());  // ~14.5%
println!("Marginal rate: {:.1}%", proof.marginal_rate_percent());    // 22%
println!("Progressive savings: {:.1}%", proof.progressive_tax_savings_bps() as f64 / 100.0);

// Builder pattern with range
let proof = EffectiveTaxRateBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
    .with_range(1000, 2000)  // Prove effective rate is between 10-20%
    .build()?;
```

### Cross-Jurisdiction Proofs

Prove consistent tax bracket across multiple countries:

```rust
use mycelix_zk_tax::{CrossJurisdictionProofBuilder, Jurisdiction, FilingStatus};

// Compare tax treatment across countries
let proof = CrossJurisdictionProofBuilder::new(100_000, 2024, FilingStatus::Single)
    .add(Jurisdiction::US)?
    .add(Jurisdiction::UK)?
    .add(Jurisdiction::DE)?
    .add(Jurisdiction::CA)?
    .build()?;

println!("Average rate: {:.1}%", proof.average_rate_percent());
println!("Rate spread: {:.1}%", proof.rate_spread_percent());
println!("All within 5%? {}", proof.rates_within_tolerance(500));

// OECD preset (common OECD members)
let oecd = CrossJurisdictionProofBuilder::oecd_common(85_000, 2024)?;

// G20 preset
let g20 = CrossJurisdictionProofBuilder::g20(85_000, 2024)?;
```

### Deduction Proofs

Prove total deductions without revealing individual amounts:

```rust
use mycelix_zk_tax::{DeductionProofBuilder, DeductionCategory, FilingStatus};

let proof = DeductionProofBuilder::new(2024, FilingStatus::Single)
    .charitable(5_000)?      // Charitable giving
    .mortgage_interest(12_000)?
    .salt(10_000)?           // State & local taxes (capped at $10k)
    .retirement(6_500)?      // IRA contribution
    .build()?;

println!("Total deductions: ${}", proof.total_lower);
println!("Exceeds standard? {}", proof.exceeds_standard_deduction());  // $14,600 for 2024
println!("Categories: {}", proof.category_count());

// Get anonymous summary
for summary in proof.category_summaries() {
    println!("{}: ${}-${}", summary.category.name(), summary.lower, summary.upper);
}
```

### Composite Proofs

Combine multiple proof types into one attestation:

```rust
use mycelix_zk_tax::{CompositeProofBuilder, DeductionCategory, Jurisdiction, FilingStatus};

let proof = CompositeProofBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
    .with_bracket()                           // Include bracket proof
    .with_effective_rate()                    // Include effective rate
    .with_range(50_000, 150_000)              // Prove income in range
    .with_deduction(DeductionCategory::Charitable, 5_000)
    .with_deduction(DeductionCategory::MortgageInterest, 12_000)
    .build_dev()?;

println!("Components: {}", proof.component_count());
println!("Has bracket? {}", proof.has_bracket());
println!("Has effective? {}", proof.has_effective_rate());
assert!(proof.verify().is_ok());
```

### Proof Chaining (Audit Trails)

Create immutable chains of proofs for compliance history:

```rust
use mycelix_zk_tax::{ProofChain, TaxBracketProver, EffectiveTaxRateProof, Jurisdiction, FilingStatus};

let prover = TaxBracketProver::dev_mode();
let mut chain = ProofChain::new();

// Add proofs over time (each links to previous via SHA3-256)
let proof_2022 = prover.prove(75_000, Jurisdiction::US, FilingStatus::Single, 2022)?;
chain.add_bracket_proof(&proof_2022);

let proof_2023 = prover.prove(82_000, Jurisdiction::US, FilingStatus::Single, 2023)?;
chain.add_bracket_proof(&proof_2023);

let proof_2024 = prover.prove(88_000, Jurisdiction::US, FilingStatus::Single, 2024)?;
chain.add_bracket_proof(&proof_2024);

// Verify chain integrity
assert!(chain.verify());
println!("Chain length: {}", chain.len());
println!("Chain hash: {}", chain.chain_hash());

// Export for storage
let json = chain.to_json();
```

### Multi-Year Batch Proofs

Prove bracket history across multiple years:

```rust
use mycelix_zk_tax::{BatchProofBuilder, Jurisdiction, FilingStatus};

let proof = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
    .add_year(2022, 75_000)?
    .add_year(2023, 82_000)?
    .add_year(2024, 88_000)?
    .build()?;

println!("Years covered: {:?}", proof.year_range());
println!("Consistent bracket? {}", proof.is_consistent_bracket());
println!("Shows growth? {}", proof.shows_income_growth());

for summary in proof.year_summaries() {
    println!("{}: Bracket {} ({}%)", summary.year, summary.bracket_index, summary.rate_percent());
}
```

### Income Range Proofs

Prove income falls within a range without revealing exact amount:

```rust
use mycelix_zk_tax::{RangeProofBuilder, IncomeRangeProof};

// Prove income is between $50k-$100k
let proof = RangeProofBuilder::new(85_000, 2024)
    .prove_between(50_000, 100_000)?;

// Prove income is above $50k
let proof = RangeProofBuilder::new(85_000, 2024)
    .prove_above(50_000)?;

// Prove income is below $200k
let proof = RangeProofBuilder::new(85_000, 2024)
    .prove_below(200_000)?;

// Prove income is a multiple of $1000 (within tolerance)
let proof = RangeProofBuilder::new(85_000, 2024)
    .prove_multiple_of(1_000)?;
```

## Subnational Jurisdictions

### Swiss Cantons

```rust
use mycelix_zk_tax::subnational::{SwissCanton, get_cantonal_brackets};

// Zug (famous low-tax canton)
let brackets = get_cantonal_brackets(SwissCanton::ZG, 2024, FilingStatus::Single)?;
println!("Zug: {} brackets, top rate {:.1}%", brackets.len(), brackets.last().unwrap().rate_bps as f64 / 100.0);

// Geneva (high-tax, French-speaking)
let brackets = get_cantonal_brackets(SwissCanton::GE, 2024, FilingStatus::Single)?;

// Canton info
println!("Language: {}", SwissCanton::ZH.language());  // "German"
println!("Name: {}", SwissCanton::TI.name());          // "Ticino"
```

### UK Nations (Scotland's Different Rates)

```rust
use mycelix_zk_tax::subnational::{UKNation, get_uk_nation_brackets};

// Scotland has different (more progressive) rates
let scotland = get_uk_nation_brackets(UKNation::Scotland, 2024, FilingStatus::Single)?;
let england = get_uk_nation_brackets(UKNation::England, 2024, FilingStatus::Single)?;

// Scotland 2024/25: Starter (19%), Basic (20%), Intermediate (21%),
//                   Higher (42%), Advanced (45%), Top (48%)
println!("Scotland brackets: {}", scotland.len());  // 6
println!("England brackets: {}", england.len());    // 3
```

### Brazilian States (IRPF)

```rust
use mycelix_zk_tax::subnational::{BrazilianState, get_brazilian_brackets};

// Brazil uses federal IRPF (same rates nationwide)
let sp_brackets = get_brazilian_brackets(BrazilianState::SP, 2024, FilingStatus::Single)?;

// State info
println!("Region: {}", BrazilianState::RJ.region());  // "Southeast"
println!("Name: {}", BrazilianState::BA.name());      // "Bahia"
```

## WASM Subnational Bindings

```typescript
import { WasmSubnationalProver } from '@mycelix/zk-tax';

const prover = new WasmSubnationalProver();

// US State proof
const caProof = prover.prove_us_state(BigInt(150000), "CA", "single", 2024);
console.log(`California: ${caProof.rate_percent}%`);  // 9.3%
console.log(`Has income tax: ${caProof.has_income_tax}`);

// Swiss Canton proof
const zhProof = prover.prove_swiss_canton(BigInt(200000), "ZH", "single", 2024);
console.log(`Zürich: ${zhProof.rate_percent}%`);

// List all regions
console.log(WasmSubnationalProver.list_us_states());
console.log(WasmSubnationalProver.list_swiss_cantons());
console.log(WasmSubnationalProver.list_canadian_provinces());
```

## Use Cases

1. **KYC-Lite**: Prove income tier without exact amount
2. **Loan Applications**: Qualify for income requirements privately
3. **Rental Verification**: Landlords verify tenant income bracket
4. **Benefits Eligibility**: Prove bracket without full disclosure
5. **Tax Advisory**: Share bracket info with advisors safely
6. **International Compliance**: Prove tax status across jurisdictions
7. **Mortgage Qualification**: Multi-year income stability proofs
8. **Deduction Verification**: Prove itemization without revealing details
9. **Cross-Border Workers**: Prove consistent treatment across countries
10. **Audit Defense**: Immutable proof chains for compliance history

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Income    │────▶│   Risc0 zkVM    │────▶│   STARK Proof   │
│   (PRIVATE)     │     │   Verification   │     │    (PUBLIC)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  Never revealed         Proves income          Anyone can verify
                         is in bracket          without learning
                                                actual income
```

1. **Witness Creation**: Income combined with tax bracket bounds
2. **zkVM Execution**: Risc0 zkVM verifies income is within claimed bracket
3. **STARK Proof**: Proof generated (income never leaves your device)
4. **Verification**: Anyone can verify without learning the income

## Security

- ✅ Income is **NEVER** revealed in the proof
- ✅ Proofs are cryptographically bound to specific brackets (FNV-1a commitment)
- ✅ Real proofs use Risc0's STARK proving system
- ⚠️ Dev mode is **NOT** secure - use only for testing
- ✅ Deterministic proofs for same inputs

## Project Structure

```
sdk/zk-tax/
├── src/
│   ├── brackets.rs      # Tax bracket definitions for all G20
│   ├── subnational.rs   # US states & Canadian provinces
│   ├── jurisdiction.rs  # Jurisdiction & filing status enums
│   ├── prover.rs        # ZK proof generation (Risc0)
│   ├── wasm.rs          # WebAssembly bindings
│   └── ...
├── contracts/
│   ├── TaxBracketVerifier.sol    # Solidity on-chain verifier
│   └── TaxBracketVerifier.t.sol  # Foundry tests
├── methods/
│   └── guest/           # Risc0 guest program
├── pkg/                 # Built WASM package
└── tests/
    └── holochain_integration.rs  # DHT compatibility tests
```

## Development

```bash
# Run all tests
cargo test --lib --no-default-features

# Build WASM package
wasm-pack build --target web --features wasm --no-default-features

# Build CLI
cargo build --bin zk-tax --features "cli prover" --release

# Generate real ZK proof (release mode recommended)
cargo run --bin zk-tax --features "cli prover" --release -- prove --income 85000 --jurisdiction US --real
```

## Related Projects

- [Mycelix Network](https://mycelix.net) - Decentralized identity infrastructure
- [Risc0](https://risczero.com) - Zero-knowledge virtual machine
- [Holochain](https://holochain.org) - Agent-centric distributed computing

## License

MIT OR Apache-2.0

---

Built with ❤️ by [Luminous Dynamics](https://luminousdynamics.org)
