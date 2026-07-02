# HDK 0.4.4 Quick Reference Guide

**For**: Zero-TrustML Identity DNA Development
**Updated**: November 11, 2025

## Essential Patterns

### 1. Entry Type Definitions

```rust
use hdk::prelude::*;

// ✅ CORRECT (HDK 0.4.4)
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    MyEntry(MyEntry),
    #[entry_type]
    AnotherEntry(AnotherEntry),
}

// ❌ WRONG (HDK 0.3.x - deprecated)
#[hdk_entry_defs]
pub enum EntryTypes {
    #[entry_def]
    MyEntry(MyEntry),
}
```

### 2. Creating Entries

```rust
// ✅ CORRECT - Pass reference
let action_hash = create_entry(&EntryTypes::MyEntry(data.clone()))?;

// ❌ WRONG - Pass owned value
let action_hash = create_entry(EntryTypes::MyEntry(data))?;
```

### 3. Path Operations

```rust
// ✅ CORRECT - Must call .typed() before .ensure()
let path = Path::from(format!("prefix.{}", id))
    .typed(LinkTypes::MyLink)?;
path.ensure()?;

// ❌ WRONG - .ensure() without .typed()
let path = Path::from(format!("prefix.{}", id));
path.ensure()?;  // ERROR: no method named `ensure`
```

### 4. Error Conversion for SerializedBytes

```rust
// ✅ CORRECT - Map SerializedBytesError to WasmError
let entry: MyEntry = MyEntry::try_from(
    SerializedBytes::from(UnsafeBytes::from(bytes.to_vec()))
).map_err(|e| wasm_error!(e))?;

// ❌ WRONG - Direct ? operator
let entry: MyEntry = MyEntry::try_from(
    SerializedBytes::from(UnsafeBytes::from(bytes.to_vec()))
)?;  // ERROR: can't convert SerializedBytesError to WasmError
```

### 5. Link Target Hash Conversion

```rust
// ✅ CORRECT - Use .into_any_dht_hash()
let target_hash = link.target.clone()
    .into_any_dht_hash()
    .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

if let Some(record) = get(target_hash, GetOptions::default())? {
    // Process record
}

// ❌ WRONG - Direct conversion
let target_hash = link.target;  // Wrong type for get()
```

### 6. LinkTag with Byte Slices

```rust
// ✅ CORRECT - Use .as_bytes()
LinkTag::new(my_string.as_bytes())

// ❌ WRONG - Pass &String
LinkTag::new(&my_string)  // ERROR: trait bound not satisfied
```

### 7. HDK Extern Functions (Single Parameter Only!)

```rust
// ✅ CORRECT - Wrap in struct
#[derive(Serialize, Deserialize, Debug)]
pub struct MyFunctionInput {
    pub field1: String,
    pub field2: u32,
}

#[hdk_extern]
pub fn my_function(input: MyFunctionInput) -> ExternResult<Output> {
    // Access input.field1, input.field2
}

// ❌ WRONG - Multiple parameters
#[hdk_extern]
pub fn my_function(field1: String, field2: u32) -> ExternResult<Output> {
    // ERROR: hdk_extern functions must take a single parameter or none
}
```

### 8. Calling HDK Extern Functions

```rust
// ⚠️ IMPORTANT: Cannot call #[hdk_extern] functions from within the same zome!
// They are only accessible via external calls through the conductor.

// ❌ WRONG - Internal call
#[hdk_extern]
pub fn store_entry(input: StoreInput) -> ExternResult<ActionHash> {
    // This will fail to compile:
    let existing = get_entry(input.id.clone())?;  // ERROR: function not found
    // ...
}

#[hdk_extern]
pub fn get_entry(id: String) -> ExternResult<Option<Entry>> {
    // ...
}

// ✅ CORRECT - Extract to helper function
pub fn store_entry(input: StoreInput) -> ExternResult<ActionHash> {
    let existing = internal_get_entry(&input.id)?;  // OK: internal function
    // ...
}

fn internal_get_entry(id: &str) -> ExternResult<Option<Entry>> {
    // Shared logic accessible internally
}

#[hdk_extern]
pub fn get_entry(id: String) -> ExternResult<Option<Entry>> {
    internal_get_entry(&id)
}
```

### 9. Explicit Type Annotations (When Needed)

```rust
// ✅ CORRECT - Annotate ambiguous numeric types
let score: f64 = match count {
    0 => 0.0,
    1 => 0.5,
    _ => 1.0,
};

let bonus: f64 = if condition { 0.2 } else { 0.0 };
Ok((score + bonus).min(1.0_f64))

// ❌ WRONG - Ambiguous type
let score = match count {
    0 => 0.0,  // f32 or f64?
    1 => 0.5,
    _ => 1.0,
};
Ok((score + bonus).min(1.0))  // ERROR: ambiguous numeric type
```

## Common Compilation Errors & Solutions

### Error: "no method named `ensure`"
```
error[E0599]: no method named `ensure` found for struct `Path`
```
**Solution**: Call `.typed(LinkTypes::YourLink)?` before `.ensure()`

### Error: "can't convert SerializedBytesError"
```
error[E0277]: `?` couldn't convert the error to `WasmError`
```
**Solution**: Add `.map_err(|e| wasm_error!(e))?`

### Error: "trait bound not satisfied" (Entry conversion)
```
error[E0277]: the trait bound `Entry: TryFrom<EntryTypes>` is not satisfied
```
**Solution**: Pass reference to `create_entry(&EntryTypes::Variant(data))`

### Error: "hdk_extern functions must take single parameter"
```
error: hdk_extern functions must take a single parameter or none
```
**Solution**: Wrap multiple parameters in a struct

### Error: "can't find function X in this scope"
```
error[E0425]: cannot find function `some_hdk_extern_function` in this scope
```
**Solution**: `#[hdk_extern]` functions cannot be called internally. Extract to helper function.

## Build & Test Commands

```bash
# Enter Nix development environment (ALWAYS!)
cd /srv/luminous-dynamics/Mycelix-Core/0TML/zerotrustml-identity-dna
nix-shell

# Build single zome
cargo build --release --target wasm32-unknown-unknown --package zome_name

# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Pack DNA bundle
hc dna pack .

# Verify DNA hash
hc dna hash zerotrustml_identity.dna

# Create test sandbox
hc sandbox create --in-process-lair -d test-conductor

# Check WASM artifacts
ls -lh target/wasm32-unknown-unknown/release/*.wasm
```

## Workspace Structure

```
zerotrustml-identity-dna/
├── Cargo.toml          # Workspace config (HDK 0.4.4)
├── dna.yaml            # DNA manifest
├── shell.nix           # Nix build environment
├── zomes/
│   ├── did_registry/
│   │   └── src/lib.rs
│   ├── identity_store/
│   │   └── src/lib.rs
│   ├── reputation_sync/
│   │   └── src/lib.rs
│   ├── guardian_graph/
│   │   └── src/lib.rs
│   └── governance_record/
│       └── src/lib.rs
└── target/
    └── wasm32-unknown-unknown/
        └── release/
            └── *.wasm  # Compiled zomes
```

## Helpful Resources

- **HDK 0.4.4 Docs**: https://docs.rs/hdk/0.4.4/hdk/
- **Holochain Examples**: `/tmp/holochain/crates/test_utils/wasm/wasm_workspace/`
- **Our Completion Report**: `PHASE_1_1_HOLOCHAIN_INTEGRATION_COMPLETE.md`
- **W3C DID Spec**: https://www.w3.org/TR/did-core/
- **Holochain Docs**: https://docs.holochain.org/

## Tips & Best Practices

1. **Always use nix-shell**: Direct cargo builds will fail without WASM target
2. **Read compiler errors carefully**: Most HDK errors have clear solutions
3. **Check Holochain examples**: The test_utils workspace has working patterns
4. **Keep link types explicit**: Always specify which LinkType you're using
5. **Prefer internal helpers**: Extract logic to internal functions instead of calling `#[hdk_extern]`
6. **Test incrementally**: Build after each major change to catch errors early

## Version Info

- **HDK**: 0.4.4
- **Rust**: 1.91.1
- **Holochain**: 0.5.6 (hc CLI)
- **Target**: wasm32-unknown-unknown

---

**Last Updated**: November 11, 2025
**Maintainer**: Zero-TrustML Team
