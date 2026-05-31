# Mycelix SDK Type Generation

This document describes how to generate TypeScript types from the Rust SDK using ts-rs.

## Overview

The Mycelix SDK uses [ts-rs](https://github.com/Aleph-Alpha/ts-rs) to automatically generate TypeScript type definitions from Rust structs. This ensures type safety across the stack and eliminates manual type maintenance.

## Quick Start

```bash
# Generate types from Rust to TypeScript
cd sdk
./scripts/generate-types.sh
```

## How It Works

1. **Annotate Rust types** with `#[derive(TS)]` and `#[ts(export)]`
2. **Build with feature** `cargo build --features ts-export`
3. **Run tests** to trigger type generation
4. **Copy to TypeScript SDK** in `sdk-ts/src/generated/`

## Adding ts-rs to a Type

```rust
// Before
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVector {
    pub k_r: f32,
    pub k_a: f32,
    // ...
}

// After
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-export", ts(export))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVector {
    pub k_r: f32,
    pub k_a: f32,
    // ...
}
```

## Types to Export

### MATL Module (matl/)
- `KVector` - 8-dimensional trust profile
- `KVectorWeights` - Configurable component weights
- `KVectorDimension` - Dimension identifiers
- `GovernanceTier` - Trust-based governance levels
- `ProofOfGradientQuality` - PoGQ validation record
- `RbBftConsensus` - Byzantine consensus state

### Epistemic Module (epistemic/)
- `EpistemicClaim` - Claim with E/N/M classification
- `EmpiricalLevel` - Empirical evidence levels (E0-E4)
- `NormativeLevel` - Normative levels (N0-N4)
- `MaterialityLevel` - Materiality levels (M0-M4)
- `Evidence` - Supporting evidence record

### DKG Module (dkg/)
- `VerifiableTriple` - Knowledge graph triple
- `ConfidenceScore` - 5-factor confidence
- `Attestation` - Claim attestation

### FL Module (fl/)
- `GradientUpdate` - Federated learning update
- `AggregatedGradient` - Aggregated result
- `FLConfig` - Federated learning config
- `Participant` - FL participant state

### Economics Module (economics/)
- `Currency` - Supported currencies
- `FeeTier` - Transaction fee tiers
- `MemberEconomics` - Member economic state

## Generated Output

Types are generated to:
```
sdk-ts/src/generated/
├── index.ts       # Re-exports all types
├── matl.ts        # MATL types
├── epistemic.ts   # Epistemic types
├── dkg.ts         # DKG types
├── fl.ts          # FL types
├── bridge.ts      # Bridge types
└── economics.ts   # Economics types
```

## Using Generated Types

```typescript
// Import from generated module
import { KVector, GovernanceTier, EpistemicClaim } from '@mycelix/sdk/generated';

// Or use the re-exports
import { matl, epistemic } from '@mycelix/sdk';
const kv: matl.KVector = { k_r: 0.8, ... };
```

## CI Integration

Add to your CI pipeline:

```yaml
- name: Generate TypeScript types
  run: |
    cd sdk
    ./scripts/generate-types.sh

- name: Type check
  run: |
    cd sdk-ts
    pnpm typecheck
```

## Troubleshooting

### Types not generating
- Ensure `ts-export` feature is enabled
- Check that types have `#[ts(export)]` attribute
- Verify ts-rs is version 9.x

### Type mismatch errors
- Regenerate types: `./scripts/generate-types.sh`
- Check for Rust type changes that need manual adjustment
- Verify serde attributes match expected JSON format

### Missing dependency
```bash
cargo add ts-rs --features chrono-impl,serde-compat
```

## Best Practices

1. **Always regenerate** types after changing Rust structs
2. **Don't edit** generated files manually - changes will be overwritten
3. **Add tests** that verify TypeScript types match Rust behavior
4. **Document** custom serde attributes that affect JSON serialization
5. **Use cfg_attr** to make ts-rs optional for production builds

## Related Files

- `Cargo.toml` - Feature flags for ts-export
- `scripts/generate-types.sh` - Generation script
- `sdk-ts/src/generated/` - Generated output
- `sdk-ts/package.json` - Build scripts for TypeScript
