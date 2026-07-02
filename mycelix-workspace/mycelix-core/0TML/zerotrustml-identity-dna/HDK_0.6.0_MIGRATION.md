# HDK 0.6.0-rc.0 Migration Guide

## Overview

This document details the migration of the Zero-TrustML Identity DNA from HDK 0.4.4 to HDK 0.6.0-rc.0, completed on November 12, 2025.

## Migration Summary

- **From**: HDK 0.4.4
- **To**: HDK 0.6.0-rc.0
- **Zomes Migrated**: 5 (did_registry, identity_store, reputation_sync, guardian_graph, governance_record)
- **Build Status**: ✅ All zomes compile successfully
- **DNA Bundle**: ✅ Successfully packed (4.0MB)

## Breaking Changes Fixed

### 1. getrandom WASM Compatibility

**Issue**: getrandom v0.3.4 doesn't support wasm32-unknown-unknown target by default

**Solution**: Created `.cargo/config.toml` to configure custom randomness backend:

```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="custom"']
```

This tells getrandom to use Holochain's host-provided randomness instead of trying to use WASM-unsupported system calls.

### 2. Missing Dependency: holochain_serialized_bytes

**Issue**: `#[hdk_entry_helper]` macro requires `holochain_serialized_bytes` crate

**Solution**: Added to all zome Cargo.toml files:

```toml
[dependencies]
holochain_serialized_bytes = "0.0.56"
```

**Affected Files**:
- `zomes/did_registry/Cargo.toml`
- `zomes/identity_store/Cargo.toml`
- `zomes/reputation_sync/Cargo.toml`
- `zomes/guardian_graph/Cargo.toml`
- `zomes/governance_record/Cargo.toml`

### 3. AgentInfo Field Rename

**Issue**: `agent_latest_pubkey` renamed to `agent_initial_pubkey`

**Before**:
```rust
let caller = agent_info()?.agent_latest_pubkey;
```

**After**:
```rust
let caller = agent_info()?.agent_initial_pubkey;
```

**Affected Files**:
- `zomes/guardian_graph/src/lib.rs` (2 occurrences at lines 74, 133)
- `zomes/did_registry/src/lib.rs` (3 occurrences at lines 72, 150, 197)

### 4. get_links() Function Signature Change

**Issue**: `get_links()` now takes 2 parameters: `LinkQuery` and `GetStrategy`

**Before**:
```rust
let links = get_links(
    GetLinksInputBuilder::try_new(
        path.path_entry_hash()?,
        LinkTypes::SomeLink
    )?.build()
)?;
```

**After**:
```rust
let links = get_links(
    LinkQuery::try_new(
        path.path_entry_hash()?,
        LinkTypes::SomeLink
    )?,
    GetStrategy::default()
)?;
```

**Affected Files**:
- `zomes/identity_store/src/lib.rs` (4 occurrences)
- `zomes/reputation_sync/src/lib.rs` (4 occurrences)
- `zomes/guardian_graph/src/lib.rs` (5 occurrences)
- `zomes/governance_record/src/lib.rs` (7 occurrences)
- `zomes/did_registry/src/lib.rs` (2 occurrences)

### 5. delete_link() Function Signature Change

**Issue**: `delete_link()` now requires `GetOptions` parameter

**Before**:
```rust
delete_link(link.create_link_hash)?;
```

**After**:
```rust
delete_link(link.create_link_hash, GetOptions::default())?;
```

**Affected Files**:
- `zomes/guardian_graph/src/lib.rs` (2 occurrences at lines 166, 172)
- `zomes/did_registry/src/lib.rs` (1 occurrence at line 213)

### 6. SerializedBytes Deserialization Pattern Change

**Issue**: Entry::App deserialization pattern changed

**Before**:
```rust
if let Ok(did_doc) = DIDDocument::try_from(
    SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
) {
```

**After**:
```rust
if let Ok(did_doc) = DIDDocument::try_from(
    SerializedBytes::from(bytes.to_owned())
).map_err(|e| wasm_error!(e)) {
```

**Affected Files**:
- `zomes/did_registry/src/lib.rs` (lines 337-339)

## Build Instructions

### Prerequisites
- NixOS or Linux with Nix (flakes enabled)
- Rust 1.91.1+ (provided by nix-shell)
- Holochain tools (hc CLI)

### Building All Zomes

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/zerotrustml-identity-dna

# Enter Nix development environment
nix-shell

# Build all zomes to WASM
cargo build --release --target wasm32-unknown-unknown

# Pack DNA bundle
hc dna pack .
```

### Build Output

**WASM Artifacts** (in `target/wasm32-unknown-unknown/release/`):
- `did_registry.wasm` (4.3 MB)
- `identity_store.wasm` (4.5 MB)
- `reputation_sync.wasm` (4.3 MB)
- `guardian_graph.wasm` (4.3 MB)
- `governance_record.wasm` (3.5 MB)

**DNA Bundle**:
- `zerotrustml_identity.dna` (4.0 MB)

## Warnings (Non-Critical)

The following warnings are present but do not affect functionality:

1. **Unused variables** (guardian_graph, identity_store, did_registry)
2. **Unused imports** (guardian_graph: HashSet)

These can be addressed with:
```bash
cargo fix --lib -p <zome_name>
```

## Testing Recommendations

After migration, test the following:

1. **DID Operations**:
   - Create DID document
   - Resolve DID
   - Update DID
   - Deactivate DID

2. **Identity Store**:
   - Add identity factors
   - Issue verifiable credentials
   - Query credentials by type

3. **Reputation Sync**:
   - Add reputation entries
   - Query reputation by DID
   - Aggregate reputation scores

4. **Guardian Graph**:
   - Add guardian relationships
   - Remove relationships
   - Query guardians
   - Compute metrics

5. **Governance Record**:
   - Create proposals
   - Cast votes
   - Tally results
   - Record governance decisions

## Files Changed

### Configuration Files
- `.cargo/config.toml` (created)
- `Cargo.toml` (workspace HDK version updated)
- `zomes/*/Cargo.toml` (5 files - added holochain_serialized_bytes)

### Source Files
- `zomes/did_registry/src/lib.rs`
- `zomes/identity_store/src/lib.rs`
- `zomes/reputation_sync/src/lib.rs`
- `zomes/guardian_graph/src/lib.rs`
- `zomes/governance_record/src/lib.rs`

## Migration Tools

A Python script was created to automate the API pattern fixes:
- `/tmp/fix_hdk_api.py` - Automated get_links() and delete_link() fixes

## Next Steps

1. ✅ **HDK Upgrade** - Complete
2. ✅ **Compilation** - All zomes build successfully
3. ✅ **DNA Packing** - Bundle created
4. ⏳ **Testing** - Pending (Phase 1.2)
5. ⏳ **Integration** - Connect to Python DHT client (Phase 1.2)
6. ⏳ **Deployment** - Conductor deployment testing (Phase 1.3)

## References

- [Holochain HDK 0.6.0-rc.0 Documentation](https://docs.rs/hdk/0.6.0-rc.0/)
- [Zero-TrustML Architecture](../docs/06-architecture/matl_architecture.md)
- [Development Roadmap](../README.md)

## Troubleshooting

### Issue: WASM target not found
```bash
# Outside nix-shell
rustup target add wasm32-unknown-unknown
```

### Issue: getrandom compilation error
Ensure `.cargo/config.toml` exists with proper rustflags configuration.

### Issue: Missing holochain_serialized_bytes
Add dependency to zome's Cargo.toml: `holochain_serialized_bytes = "0.0.56"`

### Issue: get_links() signature mismatch
Use `LinkQuery::try_new()` + `GetStrategy::default()` instead of `GetLinksInputBuilder`.

### Issue: delete_link() signature mismatch
Add `GetOptions::default()` as second parameter.

---

**Migration Completed**: November 12, 2025
**Migrated By**: Claude Code Max (AI Assistant)
**Verified**: All zomes compile and DNA packs successfully
