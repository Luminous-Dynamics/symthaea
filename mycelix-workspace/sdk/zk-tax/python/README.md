# ZK Tax Python SDK

Zero-knowledge tax bracket proofs for privacy-preserving compliance.

## Installation

```bash
pip install zk-tax
```

Or build from source with maturin:

```bash
pip install maturin
maturin develop
```

## Quick Start

```python
from zk_tax import Jurisdiction, FilingStatus, find_bracket, get_brackets

# Find the tax bracket for a specific income
bracket = find_bracket(
    income=85000,
    jurisdiction=Jurisdiction.US,
    year=2024,
    filing_status=FilingStatus.SINGLE
)

print(f"Bracket: {bracket.index}")
print(f"Rate: {bracket.rate_percent}%")
print(f"Range: ${bracket.lower:,} - ${bracket.upper:,}")

# Get all brackets for a jurisdiction
brackets = get_brackets(Jurisdiction.US, 2024, FilingStatus.SINGLE)
for b in brackets:
    upper = "∞" if b.upper == 2**64 - 1 else f"${b.upper:,}"
    print(f"  Bracket {b.index}: ${b.lower:,} - {upper} @ {b.rate_percent}%")
```

## Supported Jurisdictions

58 jurisdictions across 6 continents:

### Americas
- 🇺🇸 US, 🇨🇦 CA, 🇲🇽 MX, 🇧🇷 BR, 🇦🇷 AR, 🇨🇱 CL, 🇨🇴 CO, 🇵🇪 PE, 🇪🇨 EC, 🇺🇾 UY

### Europe
- 🇬🇧 UK, 🇩🇪 DE, 🇫🇷 FR, 🇮🇹 IT, 🇪🇸 ES, 🇳🇱 NL, 🇧🇪 BE, 🇦🇹 AT, 🇵🇹 PT, 🇮🇪 IE
- 🇵🇱 PL, 🇸🇪 SE, 🇩🇰 DK, 🇫🇮 FI, 🇳🇴 NO, 🇨🇭 CH, 🇨🇿 CZ, 🇬🇷 GR, 🇭🇺 HU, 🇷🇴 RO
- 🇷🇺 RU, 🇹🇷 TR, 🇺🇦 UA

### Asia-Pacific
- 🇯🇵 JP, 🇨🇳 CN, 🇮🇳 IN, 🇰🇷 KR, 🇮🇩 ID, 🇦🇺 AU, 🇳🇿 NZ, 🇸🇬 SG, 🇭🇰 HK, 🇹🇼 TW
- 🇲🇾 MY, 🇹🇭 TH, 🇻🇳 VN, 🇵🇭 PH, 🇵🇰 PK

### Middle East
- 🇸🇦 SA, 🇦🇪 AE, 🇮🇱 IL, 🇪🇬 EG, 🇶🇦 QA

### Africa
- 🇿🇦 ZA, 🇳🇬 NG, 🇰🇪 KE, 🇲🇦 MA, 🇬🇭 GH

## Filing Statuses

- `FilingStatus.SINGLE` - Single filer
- `FilingStatus.MARRIED_FILING_JOINTLY` - Married filing jointly
- `FilingStatus.MARRIED_FILING_SEPARATELY` - Married filing separately
- `FilingStatus.HEAD_OF_HOUSEHOLD` - Head of household

Note: Not all filing statuses are available in all jurisdictions. Most countries outside the US only support `SINGLE`.

## Working with Proofs

```python
from zk_tax import TaxBracketProof

# Load a proof from JSON
proof = TaxBracketProof.from_json(json_string)

# Verify the proof
try:
    proof.verify()
    print("Proof is valid!")
    print(f"  Bracket: {proof.bracket_index}")
    print(f"  Rate: {proof.rate_percent}%")
    print(f"  Year: {proof.tax_year}")
except ValueError as e:
    print(f"Invalid proof: {e}")

# Serialize to JSON
json_output = proof.to_json()
```

## API Reference

### Functions

#### `find_bracket(income, jurisdiction, year, filing_status) -> TaxBracket`
Find the tax bracket for a given income.

#### `get_brackets(jurisdiction, year, filing_status) -> List[TaxBracket]`
Get all tax brackets for a jurisdiction.

#### `list_jurisdictions() -> List[Tuple[str, str]]`
List all supported jurisdictions as (code, name) tuples.

#### `version() -> str`
Get the SDK version.

### Classes

#### `TaxBracket`
- `index: int` - Bracket index (0-based)
- `rate_bps: int` - Rate in basis points
- `rate_percent: float` - Rate as percentage
- `lower: int` - Lower bound (inclusive)
- `upper: int` - Upper bound (exclusive)
- `contains(income: int) -> bool` - Check if income is in bracket

#### `TaxBracketProof`
- `jurisdiction: Jurisdiction` - The jurisdiction
- `filing_status: FilingStatus` - The filing status
- `tax_year: int` - The tax year
- `bracket_index: int` - The proven bracket
- `rate_bps: int` - Rate in basis points
- `rate_percent: float` - Rate as percentage
- `commitment: str` - Cryptographic commitment (hex)
- `verify() -> bool` - Verify the proof
- `to_json() -> str` - Serialize to JSON
- `from_json(s: str) -> TaxBracketProof` - Deserialize from JSON

## License

MIT OR Apache-2.0
