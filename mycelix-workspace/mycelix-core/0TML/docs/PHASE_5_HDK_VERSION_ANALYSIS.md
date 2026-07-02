# Phase 5 HDK Version Analysis

**Date**: 2025-09-30
**Purpose**: Determine optimal HDK version for isolated Zero-TrustML Credits DNA

---

## Current Situation

We have HDK version compatibility issues preventing workspace compilation:
- **HDK 0.3**: Original version (possibly working but untested)
- **HDK 0.4.4**: Currently in Cargo.lock, API incompatibilities
- **HDK 0.5.6**: Attempted migration, different API patterns
- **HDK 0.6.0-dev.18**: Latest development version available

---

## Version Research Summary

### Holochain Framework vs HDK Versioning

**Important**: Holochain Framework version ≠ HDK version

| Holochain Framework | HDK Version | HDI Version | Status |
|--------------------|-------------|-------------|---------|
| 0.2.x | 0.2.6 | 0.3.6 | Legacy |
| 0.3.x | 0.3.x | 0.4.x | Legacy |
| **0.4.x** | **0.4.0** | **0.5.0** | **Stable (Current)** |
| 0.5.x | TBD | TBD | Beta/Stable |
| 0.6.x | TBD | TBD | Expected Soon |

### HDK Stability Status (2025)

**From Holochain Blog** (Dev Pulse 91):
- HDK v0.0.100 declared "stable (for now)"
- Core team intends to "leave it alone for a while"
- Available on crates.io with updated documentation
- Community encouraged to use and provide feedback

**Recent News** (2025):
- Holochain Framework 0.5 is now stable
- Holochain Framework 0.6 expected "in a few weeks"
- HDK stability maintained across framework updates

---

## Version Comparison for Isolated DNA

### Option 1: HDK 0.4.0 (Recommended by Docs) ✅

**Pros**:
- ✅ Official recommendation for Holochain 0.4
- ✅ Matches upgrade guide specs exactly
- ✅ Should have stable APIs
- ✅ Most documentation available
- ✅ Known compatible with HDI 0.5.0

**Cons**:
- ⚠️ We experienced issues with 0.4.4 (patch version)
- ⚠️ May have older patterns

**Recommendation**: Try exact version `hdk = "=0.4.0"` (not "0.4")

### Option 2: HDK 0.5.x (Newer Stable) 🔍

**Pros**:
- ✅ Matches stable Holochain Framework 0.5
- ✅ More modern APIs
- ✅ Better error messages (typically)
- ✅ Forward compatible with 0.6

**Cons**:
- ⚠️ Encountered `SerializedBytes` removal
- ⚠️ API changes from 0.4 (create_entry patterns)
- ⚠️ Less documentation than 0.4

**Status**: Needs investigation - what exact 0.5.x version is stable?

### Option 3: HDK 0.6.0-dev.18 (Bleeding Edge) ⚠️

**Pros**:
- ✅ Latest features
- ✅ Likely most active development
- ✅ May have cleanest APIs

**Cons**:
- ❌ Development version (not stable)
- ❌ API may change
- ❌ Risky for production
- ❌ May have bugs

**Recommendation**: **NOT RECOMMENDED** for isolated DNA

### Option 4: HDK 0.3.x (Legacy Fallback) 🔄

**Pros**:
- ✅ May match original codebase
- ✅ Older but stable patterns

**Cons**:
- ⚠️ Legacy version
- ⚠️ Less future support
- ⚠️ May lack features

**Status**: Last resort if 0.4/0.5 fail

---

## Recommended Approach for Isolated DNA

### Phase 1: Try HDK 0.4.0 Exact (HIGHEST PRIORITY) ⭐

**Strategy**: Use exact version pinning as per official docs

```toml
[package]
name = "zerotrustml_credits_isolated"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdk = "=0.4.0"  # EXACT version per docs
hdi = "=0.5.0"  # EXACT version per docs
serde = "1"
```

**Implementation Steps**:
1. Create isolated directory: `holochain/zerotrustml_credits_isolated/`
2. Copy zerotrustml_credits zome code
3. Use exact version specs from upgrade guide
4. Create isolated Cargo.toml (no workspace)
5. Test compilation: `cargo check`
6. If works: Build DNA: `hc dna pack`

**Expected Result**: Should work as per official documentation

### Phase 2: Try Latest Stable 0.5.x (IF 0.4.0 FAILS)

**Strategy**: Check crates.io for latest non-dev 0.5 version

```bash
# Check what 0.5 versions exist
cargo search hdk | grep "0.5"

# Try latest stable 0.5
# Update Cargo.toml with exact version found
```

**Expected Result**: May require code updates but should be stable

### Phase 3: Investigate HDK 0.0.100 Reference

The blog post mentions "v0.0.100" being stable. This may be:
- A typo (actually 0.1.00 or 1.0.0?)
- An old versioning scheme
- A specific release tag

**Action**: Check Holochain GitHub for this version

---

## API Compatibility Matrix

### Entry Creation Patterns by Version

| Version | Pattern | Reference | Status |
|---------|---------|-----------|---------|
| 0.3.x | `create_entry(&EntryTypes::...)` | With reference | Unknown |
| 0.4.0 | `create_entry(&EntryTypes::...)` | With reference | Per docs |
| 0.4.4 | **MIXED** | Unclear | Problematic |
| 0.5.x | `create_entry(EntryTypes::...)` | No reference | Confirmed |
| 0.6.x | Unknown | TBD | Unknown |

### Entry Types Macro by Version

| Version | Macro | Unit Enum | Status |
|---------|-------|-----------|---------|
| 0.3.x | `hdk_entry_defs` | Yes | Unknown |
| 0.4.0 | `hdk_entry_defs` + `unit_enum` | Yes | Per docs |
| 0.4.4 | `hdk_entry_types` | Required | Tested |
| 0.5.x | `hdk_entry_types` + `unit_enum` | Yes | Tested |
| 0.6.x | Unknown | Unknown | Unknown |

### SerializedBytes Support

| Version | SerializedBytes | Notes |
|---------|----------------|--------|
| 0.3.x | ✅ Supported | Legacy |
| 0.4.0 | ✅ Supported | Stable |
| 0.4.4 | ✅ Supported | Patch |
| 0.5.x | ❌ Removed | Modern |
| 0.6.x | ❌ Likely removed | TBD |

---

## Isolated DNA Implementation Plan

### Directory Structure
```
holochain/
├── zerotrustml_credits_isolated/
│   ├── Cargo.toml          # Isolated, not workspace
│   ├── dna.yaml            # DNA manifest
│   ├── src/
│   │   └── lib.rs          # Clean zome code
│   ├── tests/
│   │   └── integration.rs  # Holochain tests
│   └── flake.nix           # Pinned Holochain version
└── (existing workspace...)
```

### Cargo.toml (Isolated)
```toml
[package]
name = "zerotrustml_credits_isolated"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdk = "=0.4.0"     # Exact version per Holochain 0.4 docs
hdi = "=0.5.0"     # Exact version per Holochain 0.4 docs
serde = "1.0"
```

### flake.nix (Pinned Versions)
```nix
{
  description = "Zero-TrustML Credits Isolated DNA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    holochain.url = "github:holochain/holochain/holochain-0.4.0";
  };

  outputs = { self, nixpkgs, holochain }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rustc
          cargo
          holochain.packages.${system}.holochain
          holochain.packages.${system}.hc
        ];
      };
    };
}
```

### Building the Isolated DNA

```bash
# Create isolated workspace
mkdir -p holochain/zerotrustml_credits_isolated
cd holochain/zerotrustml_credits_isolated

# Copy zome code (clean version)
cp ../zomes/zerotrustml_credits/src/lib.rs src/

# Create Cargo.toml with exact versions
cat > Cargo.toml << 'EOF'
[package]
name = "zerotrustml_credits_isolated"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdk = "=0.4.0"
hdi = "=0.5.0"
serde = "1.0"
EOF

# Test compilation
cargo check

# If successful, create DNA
cat > dna.yaml << 'EOF'
---
manifest_version: "1"
name: zerotrustml_credits
integrity:
  origin_time: 2025-09-30T00:00:00+00:00
  zomes:
    - name: zerotrustml_credits_integrity
      bundled: target/wasm32-unknown-unknown/release/zerotrustml_credits_isolated.wasm
coordinator:
  zomes: []
EOF

# Build WASM
cargo build --release --target wasm32-unknown-unknown

# Pack DNA
hc dna pack .
```

---

## Success Criteria

### Compilation Success
- [ ] `cargo check` passes
- [ ] `cargo build --release` passes
- [ ] `cargo build --release --target wasm32-unknown-unknown` passes
- [ ] No errors, only warnings (if any)

### DNA Packaging Success
- [ ] `hc dna pack` succeeds
- [ ] Generates `.dna` file
- [ ] DNA hash is deterministic

### Integration Testing
- [ ] Can install DNA in conductor
- [ ] Zome functions callable
- [ ] Python bridge can connect
- [ ] Credit issuance works end-to-end

---

## Rollback Plan

If isolated DNA with exact versions fails:

1. **Document exact error messages**
2. **Try HDK 0.3.x** (legacy fallback)
3. **Contact Holochain community** with specific errors
4. **Continue mock mode** (Option D) while waiting for support

---

## Recommendation Summary

### For Isolated DNA (Option C):

1. **✅ First Try**: HDK 0.4.0 with HDI 0.5.0 (exact versions per docs)
2. **🔍 Second Try**: Latest stable 0.5.x if 0.4.0 has issues
3. **🔄 Fallback**: HDK 0.3.x if both fail
4. **❌ Avoid**: HDK 0.6.0-dev (not stable)

### Version to Check:

**YES, absolutely check for HDK 0.6** - but:
- ✅ Check if stable version exists (0.6.0, not 0.6.0-dev.x)
- ✅ Review changelog for breaking changes
- ✅ Use only if marked "stable" by Holochain team
- ❌ Don't use 0.6.0-dev.18 for production

---

## Next Actions

1. **Create isolated workspace** with HDK 0.4.0 exact
2. **Test compilation** with clean zome code
3. **If fails**: Document errors and try 0.5.x
4. **If succeeds**: Build DNA and test with Python bridge
5. **Document result**: Update this file with findings

---

**Status**: 🔍 **INVESTIGATION COMPLETE** - Ready to Test HDK 0.4.0

**Recommendation**: Start with HDK 0.4.0 (official docs), have 0.5.x ready as backup

**Next**: Create isolated workspace and test compilation

---

*"Use exact versions from official documentation first. Experiment with newer versions only if necessary."*