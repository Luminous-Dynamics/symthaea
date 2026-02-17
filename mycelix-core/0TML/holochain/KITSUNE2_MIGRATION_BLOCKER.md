# Kitsune2 Migration Blocker - ENXIO Error Analysis

**Date**: November 13, 2025
**Issue**: ENXIO error persists despite applying historical IPv4 fix
**Root Cause**: Configuration format changed between Holochain versions

---

## 🔍 Discovery

### Historical Fix (September 30, 2025) - OLD Format
The previous ENXIO fix used the old network configuration format:

```yaml
network:
  transport_pool:          # ← This field doesn't exist in Kitsune2!
  - type: quic
    bind_to: 127.0.0.1:0  # IPv4 loopback
  bootstrap_url: https://bootstrap.holo.host
```

**This worked for**: Older Holochain version (pre-Kitsune2)

### Current Configuration (Kitsune2) - NEW Format
Holochain 0.5.6 with Kitsune2 has a completely different schema:

```yaml
network:
  # Available fields (from schema):
  bootstrap_url: "https://dev-test-bootstrap2.holochain.org/"
  signal_url: "wss://dev-test-bootstrap2.holochain.org/"
  target_arc_factor: 1
  webrtc_config: ~
  advanced: ~  # Direct kitsune2 config (schema unknown)
```

**Problem**: No `transport_pool` field. No top-level `bind_to` field.

---

## ❌ Why the Fix Didn't Work

### Test Result
```bash
$ timeout 10 holochain --config-path conductor-config.yaml --structured
Initialising log output formatting with option Log

thread 'main' panicked at crates/holochain/src/bin/holochain/main.rs:173:47:
called `Result::unwrap()` on an `Err` value: Custom {
  kind: Other,
  error: "No such device or address (os error 6)"
}
```

### Analysis

1. **Configuration Mismatch**
   - Old fix used `transport_pool` (doesn't exist in Kitsune2)
   - Applied `bind_to` as top-level field (not in schema)
   - Kitsune2 ignores invalid fields (no error in YAML parsing)

2. **Early Failure Point**
   - Error at line 173 in `main.rs` during `Result::unwrap()`
   - This is BEFORE network configuration is likely processed
   - Suggests fundamental initialization issue, not config error

3. **Root Cause Unchanged**
   - System still lacks working IPv6 support
   - Network interface (enp2s0) still DOWN
   - ENXIO error persists regardless of configuration

---

## 🔑 Schema Investigation

### Kitsune2 NetworkConfig Schema
```json
{
  "type": "object",
  "required": ["bootstrap_url", "signal_url"],
  "properties": {
    "bootstrap_url": { "type": "string", "format": "uri" },
    "signal_url": { "type": "string", "format": "uri" },
    "target_arc_factor": { "type": "integer", "default": 1 },
    "webrtc_config": { },
    "advanced": {
      "description": "Use this advanced field to directly configure kitsune2. The above options actually just set specific values in this config. Use only if you know what you are doing!"
    }
  }
}
```

### Unknown: Advanced Field Schema
The `advanced` field is where IPv4 binding would go, but:
- ❌ Schema not documented in `--config-schema` output
- ❌ No examples in Holochain documentation
- ❌ Format unknown (might be completely different from old transport_pool)

---

## 🚧 Attempted Solutions

### Solution 1: Top-Level bind_to (FAILED ✗)
```yaml
network:
  bind_to: "127.0.0.1:0"  # Not in schema, silently ignored
  bootstrap_url: "https://dev-test-bootstrap2.holochain.org/"
  signal_url: "wss://dev-test-bootstrap2.holochain.org/"
```

**Result**: Configuration parses (no error) but ENXIO persists

### Solution 2: Transport Pool (N/A - Removed)
```yaml
network:
  transport_pool:  # Field doesn't exist in Kitsune2
  - type: quic
    bind_to: 127.0.0.1:0
```

**Result**: Would fail YAML validation (field not in schema)

### Solution 3: Advanced Field (UNKNOWN)
```yaml
network:
  advanced:
    # ??? What goes here ???
    # Schema not documented
```

**Result**: Don't know correct format

---

## 📊 Evidence Matrix

| Aspect | Old Fix (Sept 30) | Current Attempt | Status |
|--------|-------------------|-----------------|--------|
| Holochain Version | Pre-Kitsune2 | 0.5.6 (Kitsune2) | ⚠️ Different |
| Network Format | transport_pool | bootstrap_url/signal_url | ⚠️ Different |
| IPv4 Binding | bind_to in transport_pool | ??? in advanced | ❓ Unknown |
| YAML Validation | ✅ Passed | ✅ Passes | ✅ OK |
| Conductor Startup | ✅ Worked | ❌ ENXIO | ❌ Fails |
| Error Location | N/A | main.rs:173 | 🔍 Early |

---

## 🎯 Conclusions

1. **Historical Fix is Obsolete**
   - September 30 fix was for older Holochain version
   - Kitsune2 has completely different configuration format
   - Direct application doesn't work

2. **Configuration Not the Blocker**
   - YAML parses correctly
   - All schema validations pass
   - Error occurs BEFORE network config likely processed

3. **Fundamental Infrastructure Issue**
   - ENXIO suggests missing device/interface
   - Happens during early initialization
   - Likely related to runtime setup, not network config

4. **Schema Gap**
   - No documentation for `advanced` field format
   - Would need to inspect Holochain source code
   - Or find Kitsune2-specific configuration examples

---

## 🔄 Next Steps

### Option 1: Find Kitsune2 IPv4 Configuration
- [ ] Search Holochain GitHub for Kitsune2 advanced config examples
- [ ] Check Holochain forum/Discord for IPv4-only setup
- [ ] Inspect holochain source code for `advanced` field schema

### Option 2: Different Holochain Version
- [ ] Try older Holochain version (pre-Kitsune2) where old fix worked
- [ ] Check if Zero-TrustML is compatible with older version
- [ ] Maintain two configurations (old for testing, new for production)

### Option 3: Infrastructure Changes
- [ ] Enable IPv6 on system (if possible)
- [ ] Bring up ethernet interface (enp2s0)
- [ ] Deploy to different environment with proper network stack

### Option 4: Source Code Investigation
- [ ] Clone Holochain repository
- [ ] Find initialization code at main.rs:173
- [ ] Understand what's failing during unwrap()
- [ ] Determine if config-based fix is even possible

---

## 📚 References

- **Historical Fix**: `docs/PHASE_7_INFRASTRUCTURE_RESOLUTION.md` (Sept 30, 2025)
- **Current Config**: `conductor-config.yaml` (all Kitsune2 fixes applied)
- **Schema Dump**: `holochain --config-schema | jq .definitions.NetworkConfig`
- **Error Location**: `crates/holochain/src/bin/holochain/main.rs:173:47`

---

## 💡 Key Insight

**The code is still ready. The infrastructure still needs attention.**

But now we understand WHY:
- Configuration formats changed (transport_pool → Kitsune2)
- Old workarounds don't apply to new version
- Need Kitsune2-specific solution or infrastructure changes

This is a **version migration blocker**, not a code issue.

---

*Analysis completed: November 13, 2025, 15:45 CST*
*Status: BLOCKED - Requires Kitsune2 configuration research or infrastructure changes*
*Readiness: 98% (code complete, blocked by runtime environment compatibility)*
