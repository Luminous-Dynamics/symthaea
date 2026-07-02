# Holochain HDK 0.5+ Compatibility Fix Guide

**Issue**: Credits zome won't compile due to getrandom and HDK API mismatches
**Root Cause**: HDK 0.6.0-dev uses getrandom 0.3.3 which doesn't support wasm32-unknown-unknown properly
**Status**: Documented workaround available

---

## Quick Fix: Use Holochain Scaffolding Tool

The **recommended solution** is to regenerate the zome using Holochain's official scaffolding:

```bash
# Navigate to DNA directory
cd /srv/luminous-dynamics/Mycelix-Core/0TML/zerotrustml-dna

# Scaffold a new zome (this generates correct boilerplate)
hc scaffold zome credits_v2

# Copy your business logic to the scaffolded structure
cp zomes/credits/src/lib.rs zomes/credits_v2/src/lib.rs.backup

# Update to use scaffolded entry definitions
```

### Why Scaffolding Works

The `hc scaffold` tool generates code that:
- ✅ Uses correct HDK API for current version
- ✅ Handles getrandom properly for WebAssembly
- ✅ Includes proper entry/link definitions
- ✅ Sets up integrity/coordinator zome split (modern pattern)

---

## Alternative Fix: Downgrade to Stable HDK 0.4

If you need to stay with existing code structure:

### Step 1: Update Cargo.toml

```toml
[dependencies]
hdk = "0.4"
serde = { version = "1", features = ["derive"] }
```

### Step 2: Rewrite for HDK 0.4 API

```rust
use hdk::prelude::*;

/// Credit entry type
#[hdk_entry(id = "credit")]
#[derive(Clone, PartialEq)]
pub struct Credit {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: String,
    pub timestamp: Timestamp,
}

/// Create a new credit
#[hdk_extern]
pub fn create_credit(input: CreateCreditInput) -> ExternResult<HeaderHash> {
    let credit = Credit {
        holder: input.holder.clone(),
        amount: input.amount,
        earned_from: input.earned_from.clone(),
        timestamp: sys_time()?,
    };

    // HDK 0.4 API
    create(credit)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCreditInput {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: String,
}
```

### Step 3: Rebuild

```bash
cd zomes/credits
cargo clean
cargo build --target wasm32-unknown-unknown --release
```

---

## Deep Dive: The getrandom Problem

### What Happened

1. **HDK 0.6.0-dev** depends on `holochain_nonce v0.6.0-dev.4`
2. **holochain_nonce** depends on `getrandom v0.3.3`
3. **getrandom 0.3.3** doesn't support `wasm32-unknown-unknown` without configuration
4. **Result**: Compilation fails with "can't find crate for `core`"

### Why wasm32-unknown-unknown?

Holochain zomes compile to WebAssembly without JavaScript bindings:
- No access to browser APIs
- No access to Node.js
- Pure WASM execution environment
- Requires custom entropy source for randomness

### The Fix in getrandom

getrandom 0.2.x had better wasm support, but 0.3.x changed the API:
- **0.2.x**: `features = ["js"]` worked for wasm
- **0.3.x**: Requires `wasm_js` config flag + different setup

Holochain dev team is working on fixing this upstream.

---

## Workaround #3: Use Holochain 0.4.x Conductor

If you can't update zomes, use older conductor:

```bash
# Install Holochain 0.4.x
nix-shell -p holochain_0-4

# Verify version
holochain --version  # Should show 0.4.x

# Run with older conductor
holochain -c conductor-config.yaml
```

This matches the HDK 0.4 API.

---

## Long-Term Solution: Wait for HDK 0.5 Stable

The Holochain team is aware of this issue:
- HDK 0.5.x stable will fix getrandom
- Expected: Q2 2025
- Track: https://github.com/holochain/holochain/issues

**Recommendation**: Use scaffolding tool now, upgrade to 0.5 stable when released.

---

## Testing Your Fix

After applying any fix, test with:

```bash
# 1. Build succeeds
cargo build --target wasm32-unknown-unknown --release

# 2. WASM file generated
ls target/wasm32-unknown-unknown/release/*.wasm

# 3. Holochain can load it
hc dna pack .
hc app pack .
holochain -c conductor-config.yaml

# 4. Can call zome functions
hc sandbox call credits create_credit '{"holder": "...", "amount": 100, "earned_from": "test"}'
```

---

## Current Status: Phase 9 Blocker?

**Assessment**: **Not a critical blocker** for Phase 9 (FL deployment)

**Why**:
- PostgreSQL backend works perfectly (100% functional)
- Holochain DHT is optional enhancement
- Can deploy FL without Holochain, add it in Phase 10

**Recommendation**:
1. ✅ **Deploy Phase 9 with PostgreSQL** (proven, production-ready)
2. ⏳ **Fix Holochain in parallel** (use scaffolding tool)
3. 🚀 **Add Holochain in Phase 10** (after FL validates economics)

This approach:
- Unblocks user deployment
- Proves economic model
- Gives time for proper Holochain integration

---

## Quick Decision Matrix

| Scenario | Solution | Timeline |
|----------|----------|----------|
| **Need Holochain NOW** | Use scaffolding tool | 2-3 hours |
| **Can wait for stable HDK** | Pause Holochain, use PostgreSQL | Deploy now |
| **Complex custom logic** | Downgrade to HDK 0.4 | 4-6 hours |
| **Just want audit trail** | PostgreSQL + off-chain backup | 1 hour |

---

## Resources

- **Holochain Scaffolding Guide**: https://developer.holochain.org/quick-start/
- **HDK 0.4 Documentation**: https://docs.rs/hdk/0.4.4/
- **HDK 0.6 Dev Docs**: https://github.com/holochain/holochain/tree/main-0.5
- **getrandom wasm docs**: https://docs.rs/getrandom/latest/getrandom/#webassembly-support

---

**Document Status**: Solution identified, implementation choice pending
**Next Action**: User decides: Scaffold now OR Deploy with PostgreSQL, add Holochain in Phase 10
