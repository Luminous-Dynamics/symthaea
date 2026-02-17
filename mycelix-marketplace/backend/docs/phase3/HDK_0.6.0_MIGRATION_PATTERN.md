# HDK 0.6.0 Coordinator Zome Migration Pattern

**Status**: ✅ Pattern established from successful listings_coordinator migration
**Date**: December 30, 2025

---

## Overview

This document captures the complete migration pattern for upgrading coordinator zomes from older HDK versions to HDK 0.6.0, based on the successful migration of `listings_coordinator`.

## Required Changes

### 1. Agent Info Field Rename ✅ AUTOMATED

**Old**:
```rust
agent_info.agent_latest_pubkey
```

**New**:
```rust
agent_info.agent_initial_pubkey
```

**Fix**: Applied automatically via `apply-hdk-0.6-pattern.sh`

### 2. Path.ensure() Removal ✅ AUTOMATED

**Old**:
```rust
let path = Path::from("some_path");
path.ensure()?;  // NO LONGER EXISTS
create_link(path.path_entry_hash()?, ...)?;
```

**New**:
```rust
let path = Path::from("some_path");
// Paths are auto-created in HDK 0.6.0 - no ensure() needed
create_link(path.path_entry_hash()?, ...)?;
```

**Fix**: Applied automatically via `apply-hdk-0.6-pattern.sh`

### 3. get_links() API Change ⚠️ MANUAL

**Old** (Pre-HDK 0.6.0):
```rust
use GetLinksInputBuilder;

let links = get_links(
    GetLinksInputBuilder::try_new(
        base,
        LinkTypes::SomeLinkType,
    )?.build()
)?;
```

**New** (HDK 0.6.0):
```rust
let links = get_links(
    LinkQuery::try_new(base, LinkTypes::SomeLinkType)?,
    GetStrategy::Local,
)?;
```

**Key Changes**:
- Use `LinkQuery::try_new()` instead of `GetLinksInputBuilder`
- `try_new()` handles the `TryInto<LinkTypeFilter>` conversion automatically
- Must specify `GetStrategy::Local` or `GetStrategy::Network`
- LinkTypes enum works directly without manual conversion

**Important**: Do NOT try to manually convert LinkTypes to LinkTypeFilter. The `try_new()` method handles this via the `TryInto<LinkTypeFilter, Error = WasmError>` trait bound.

### 4. to_app_option() Error Handling ⚠️ MANUAL

**Old**:
```rust
let entry: MyType = record.entry().to_app_option()?
    .ok_or(wasm_error!("Could not deserialize"))?;
```

**New**:
```rust
let entry: MyType = record
    .entry()
    .to_app_option()
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
    .ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not deserialize entry".into()
    )))?;
```

**Reason**: `to_app_option()` now returns `SerializedBytesError` instead of `WasmError`, requiring explicit error type conversion.

### 5. Value Ownership Issues ⚠️ MANUAL

**Problem**: Values used multiple times need cloning

**Example**:
```rust
let agent_info = agent_info()?;
let agent_path = agent_info.agent_initial_pubkey.clone(); // Clone for multiple uses

create_link(
    agent_path.clone(),  // Clone here since used again later
    entry_hash.clone(),
    LinkTypes::AgentToListings,
    (),
)?;

// Later used again
monitoring::emit_metric(
    ...
    Some(agent_path.clone()),  // Can use again
    ...
)?;
```

---

## Complete Example: get_all_listings()

### Before (Pre-HDK 0.6.0):
```rust
pub fn get_all_listings(_: ()) -> ExternResult<ListingsResponse> {
    let path = Path::from("all_listings");
    path.ensure()?;  // NO LONGER EXISTS

    let links = get_links(
        GetLinksInputBuilder::try_new(  // OLD API
            path.path_entry_hash()?,
            LinkTypes::AllListings,
        )?.build()
    )?;

    let mut listings = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            let listing: Listing = record.entry().to_app_option()?  // WRONG ERROR TYPE
                .ok_or(...)?;
            listings.push(listing);
        }
    }

    Ok(ListingsResponse { listings })
}
```

### After (HDK 0.6.0):
```rust
pub fn get_all_listings(_: ()) -> ExternResult<ListingsResponse> {
    let path = Path::from("all_listings");
    // HDK 0.6.0: Path.ensure() removed - paths auto-created

    let links = get_links(
        LinkQuery::try_new(path.path_entry_hash()?, LinkTypes::AllListings)?,
        GetStrategy::Local,
    )?;

    let mut listings = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(listing_output) = get_listing(action_hash)? {
                if listing_output.listing.status != ListingStatus::Deleted {
                    listings.push(listing_output);
                }
            }
        }
    }

    Ok(ListingsResponse { listings })
}
```

---

## Migration Workflow

### Step 1: Apply Automatic Fixes
```bash
./apply-hdk-0.6-pattern.sh
```

This handles:
- `agent_latest_pubkey` → `agent_initial_pubkey`
- Removing `Path.ensure()` calls

### Step 2: Fix get_links() Calls Manually

Search for all `get_links(` calls:
```bash
grep -n "get_links(" zomes/*/coordinator/src/lib.rs
```

For each call, apply the pattern:
```rust
let links = get_links(
    LinkQuery::try_new(base, LinkTypes::X)?,
    GetStrategy::Local,
)?;
```

### Step 3: Fix to_app_option() Error Handling

Search for all `.to_app_option()` calls:
```bash
grep -n "to_app_option()" zomes/*/coordinator/src/lib.rs
```

Add `.map_err()` wrapper:
```rust
.to_app_option()
.map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
```

### Step 4: Fix Ownership Issues

Compilation will show "value moved here" errors. Add `.clone()` strategically:
```rust
let value = something.clone(); // Clone if used multiple times
create_link(value.clone(), ...)?; // Clone on use if needed later
```

### Step 5: Test Compilation

```bash
cargo check -p <zome_name>_coordinator
```

Repeat until:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in X.XXs
```

---

## Common Errors and Solutions

### Error: `get_links` takes 2 arguments but X was supplied
**Solution**: Use `LinkQuery::try_new()` with `GetStrategy::Local` as second argument

### Error: trait bound `_: Into<LinkTypeFilter>` is not satisfied
**Solution**: Use `LinkQuery::try_new()` instead of `LinkQuery::new()` - it handles conversion

### Error: borrow of moved value
**Solution**: Add `.clone()` before first use of value that will be used again

### Error: `?` operator has incompatible types (SerializedBytesError vs WasmError)
**Solution**: Add `.map_err(|e| wasm_error!(WasmErrorInner::Guest(...)))?`

---

## Verification Checklist

For each coordinator zome:

- [ ] All `agent_latest_pubkey` → `agent_initial_pubkey`
- [ ] All `Path.ensure()` calls removed
- [ ] All `get_links()` use `LinkQuery::try_new()` pattern
- [ ] All `to_app_option()` have `.map_err()` wrapper
- [ ] All ownership errors resolved with `.clone()`
- [ ] `cargo check -p <zome>_coordinator` shows "Finished"
- [ ] Zero compilation errors
- [ ] Only expected warnings (if any)

---

## Success Metrics

**listings_coordinator**: ✅ COMPLETE
- 0 errors
- 0 warnings (except unrelated monitoring zome)
- Compilation time: 2.28s

**Expected for other zomes**:
- Similar compilation success
- Pattern replication should be straightforward
- Estimated time: 15-30 minutes per coordinator zome

---

## Next Steps

1. Apply pattern to `reputation_coordinator`
2. Apply pattern to `transactions_coordinator`
3. Apply pattern to `arbitration_coordinator`
4. Apply pattern to `messaging_coordinator`
5. Test all coordinator zomes together
6. Build WASM and package hApp

---

**Pattern established**: 2025-12-30
**By**: HDK 0.6.0 Migration Process
**Status**: ✅ Ready for systematic application
