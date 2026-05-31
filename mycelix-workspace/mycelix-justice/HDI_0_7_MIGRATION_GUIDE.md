# HDI 0.5 to HDI 0.7 Migration Guide

## Overview

The Justice hApp currently uses HDI 0.5 / HDK 0.4. Upgrading to HDI 0.7 / HDK 0.6 requires addressing breaking changes in the Holochain macro system.

## Breaking Changes Identified

### 1. `#[hdk_entry_helper]` Macro Changes

In HDI 0.5, the macro automatically derived `Serialize, Deserialize, Debug` for entry types. In HDI 0.7, the macro behavior changed:

- The macro still derives serialization traits, but there appear to be conflicts when using complex nested types
- Entry types with many nested enums/structs may require manual trait implementations

### 2. `#[hdk_entry_types]` Macro Changes

The `#[hdk_entry_types]` macro now internally uses `hdk_entry_types_conversions` which has stricter trait bounds:

- Requires `SerializedBytes: TryFrom<&Type>` for all entry types
- Requires `WasmErrorInner: From<Infallible>` for error handling
- May require `skip_hdk_extern = true` flag for non-WASM contexts

### 3. Dependency Version Requirements

HDI 0.7 requires:
- `derive_more = "2.1"` (was 0.99)
- `getrandom = { version = "0.2", features = ["js"] }` (must pin to 0.2 for WASM)
- `holochain_serialized_bytes = "0.0.56"`

## Required Changes

### Entry Type Definitions

Current (HDI 0.5):
```rust
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Case {
    pub id: String,
    // ... many nested types
}
```

Required for HDI 0.7 (needs investigation):
```rust
// Option 1: Explicit derives
#[hdk_entry_helper]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    // ...
}

// Option 2: Use holochain_serialized_bytes derive
use holochain_serialized_bytes::SerializedBytes;

#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub struct Case {
    // ...
}
```

### Nested Types

All nested types (enums, structs used in entry types) must implement:
- `Serialize` + `Deserialize` (serde)
- `Debug`
- Possibly `SerializedBytes` explicitly

Current Justice hApp has complex nested types:
- `CaseType`, `CasePhase`, `CaseStatus`, `CaseSeverity`, `CaseContext`
- `EvidenceType`, `EvidenceStatus`, `AuthenticationType`
- `MediationStatus`, `ArbitrationStatus`, `DecisionOutcome`
- And many more

All of these already have `Serialize, Deserialize`, but may need additional traits.

## Recommended Migration Strategy

1. **Create a clean branch** for the migration
2. **Start with a minimal entry type** - create a test integrity zome with one simple entry type
3. **Verify the minimal case compiles** with HDI 0.7
4. **Incrementally add complexity** - nested types one at a time
5. **Identify which traits are required** for complex nested structures
6. **Update all entry types** once the pattern is established
7. **Test thoroughly** before merging

## Resources

- HDI 0.7 source: https://github.com/holochain/holochain/tree/main/crates/hdi
- Migration examples: Check `hdi-0.7.0/tests/` in Cargo cache
- Holochain Discord: #core-dev channel

## Status

- **Current**: HDI 0.5 / HDK 0.4 (working)
- **Target**: HDI 0.7 / HDK 0.6
- **Blocker**: Macro trait bound requirements need deeper investigation

## Estimated Effort

Medium complexity - requires understanding HDI 0.7 macro internals and potentially refactoring entry type definitions. Estimate 4-8 hours of focused work.
