# @mycelix/zk-tax

Zero-knowledge tax bracket proofs for privacy-preserving compliance.

## Overview

This SDK enables users to prove they fall within a specific tax bracket **without revealing their actual income**. Powered by RISC Zero zkVM.

### Use Cases

- **Privacy-preserving income verification** - Prove income range without disclosure
- **Compliance attestations** - Demonstrate tax bracket membership cryptographically
- **Decentralized identity** - Add verifiable income credentials to DID documents
- **Loan applications** - Prove income requirements without bank statements

## Installation

```bash
npm install @mycelix/zk-tax
# or
yarn add @mycelix/zk-tax
# or
pnpm add @mycelix/zk-tax
```

## Quick Start

```typescript
import { ZkTax } from '@mycelix/zk-tax';

// Initialize the SDK
const zktax = await ZkTax.init();

// Generate a proof (income is NEVER revealed)
const proof = await zktax.prove(85000, 'US', 'single', 2024);
console.log(`Bracket: ${proof.bracket_index}`);  // "Bracket: 2"
console.log(`Rate: ${proof.rate_bps / 100}%`);   // "Rate: 22%"

// Verify the proof
const isValid = await zktax.verify(proof);
console.log(`Valid: ${isValid}`);  // "Valid: true"
```

## Features

### 58 Jurisdictions Worldwide

```typescript
const jurisdictions = zktax.getJurisdictions();
console.log(`${jurisdictions.length} countries supported`);  // "58 countries supported"

// Filter by region
const europe = zktax.getJurisdictionsByRegion('Europe');
const asiaPacific = zktax.getJurisdictionsByRegion('Asia-Pacific');
```

**Supported Regions:**
- 🌎 **Americas** (10): US, CA, MX, BR, AR, CL, CO, PE, EC, UY
- 🌍 **Europe** (23): UK, DE, FR, IT, ES, NL, BE, AT, PT, IE, PL, SE, DK, FI, NO, CH, CZ, GR, HU, RO, RU, TR, UA
- 🌏 **Asia-Pacific** (15): JP, CN, IN, KR, ID, AU, NZ, SG, HK, TW, MY, TH, VN, PH, PK
- 🏜️ **Middle East** (5): SA, AE, IL, EG, QA
- 🌍 **Africa** (5): ZA, NG, KE, MA, GH

### Range Proofs

Prove income falls within bounds without revealing exact amount:

```typescript
const rangeProof = await zktax.proveRange(85000, 2024, {
  minIncome: 50000,
  maxIncome: 100000
});
// Proves: 50,000 ≤ income ≤ 100,000
```

### Multi-Year Batch Proofs

Prove multiple years efficiently:

```typescript
const batchProof = await zktax.proveBatch('US', 'single', {
  years: [
    { year: 2022, income: 75000 },
    { year: 2023, income: 82000 },
    { year: 2024, income: 85000 }
  ]
});
```

### Proof Compression

Compress proofs for efficient storage:

```typescript
const compressed = zktax.compress(proof);
console.log(`Size: ${compressed.compressed_data.length} bytes`);

// Later...
const decompressed = zktax.decompress(compressed);
```

### Tax Bracket Information

Query tax brackets for any jurisdiction:

```typescript
const brackets = zktax.getBrackets('US', 2024, 'single');
brackets.forEach(b => {
  console.log(`${b.rate_percent}%: $${b.lower.toLocaleString()} - $${b.upper.toLocaleString()}`);
});
// 10%: $0 - $11,600
// 12%: $11,600 - $47,150
// 22%: $47,150 - $100,525
// ...
```

## API Reference

### ZkTax Class

#### `ZkTax.init(wasmPath?: string): Promise<ZkTax>`

Initialize the SDK. Call this once before using other methods.

#### `prove(income, jurisdiction, filingStatus, taxYear, options?): Promise<TaxBracketProof>`

Generate a zero-knowledge proof of tax bracket membership.

| Parameter | Type | Description |
|-----------|------|-------------|
| `income` | `number` | Annual gross income (private) |
| `jurisdiction` | `JurisdictionCode` | Country code (e.g., "US") |
| `filingStatus` | `FilingStatusCode` | Filing status |
| `taxYear` | `TaxYear` | Tax year (2020-2025) |

#### `verify(proof): Promise<boolean>`

Verify a tax bracket proof.

#### `compress(proof): CompressedProof`

Compress a proof for efficient storage.

#### `getBrackets(jurisdiction, taxYear, filingStatus): BracketInfo[]`

Get tax brackets for a jurisdiction.

### Types

```typescript
type JurisdictionCode = 'US' | 'CA' | 'UK' | ... // 58 codes
type FilingStatusCode = 'single' | 'mfj' | 'mfs' | 'hoh';
type TaxYear = 2020 | 2021 | 2022 | 2023 | 2024 | 2025;
type Region = 'Americas' | 'Europe' | 'Asia-Pacific' | 'Middle East' | 'Africa';

interface TaxBracketProof {
  jurisdiction: JurisdictionCode;
  filing_status: FilingStatusCode;
  tax_year: TaxYear;
  bracket_index: number;
  rate_bps: number;
  bracket_lower: number;
  bracket_upper: number;
  commitment: number[];
  receipt_bytes: number[];
  image_id: number[];
}
```

## Dev Mode vs Production

By default in browsers, the SDK uses **dev mode** for fast proof generation. Dev mode proofs are valid for testing but not cryptographically secure.

```typescript
import { isDevMode } from '@mycelix/zk-tax';

const proof = await zktax.prove(85000, 'US', 'single', 2024);
if (isDevMode(proof)) {
  console.log('This is a dev mode proof - for testing only');
}
```

For production proofs, use the Rust SDK with RISC Zero zkVM.

## Browser Support

This package uses WebAssembly. Supported browsers:
- Chrome 57+
- Firefox 53+
- Safari 11+
- Edge 16+

## License

MIT OR Apache-2.0

## Links

- [GitHub](https://github.com/Luminous-Dynamics/mycelix)
- [Documentation](https://docs.mycelix.net/zk-tax)
- [Rust SDK](https://crates.io/crates/mycelix-zk-tax)
- [Python SDK](https://pypi.org/project/mycelix-zk-tax)
