# Docker Deployment Attempt - November 13, 2025

## Summary

Attempted to bypass host system's IPv6 limitations by deploying Holochain conductor in Docker with IPv6 enabled.

**Result**: ❌ **FAILED** - Same ENXIO error persists

---

## What We Did

### 1. Built Docker Image ✅
- Base: Debian 12-slim
- IPv6 Config: `echo "net.ipv6.conf.all.disable_ipv6 = 0" >> /etc/sysctl.conf`
- Holochain: Pre-built binary copied into image
- Build completed successfully (all 7 steps)

### 2. Mounted Configuration ✅
```yaml
volumes:
  - ./holochain/conductor-config.yaml:/conductor-config.yaml:ro
```
- Configuration successfully loaded by conductor
- No file-not-found errors

### 3. Container Startup ✅
- Container created and started successfully
- Network `zerotrustml-net` created
- Port 8888 bound successfully

---

## Error Encountered

```
thread 'main' panicked at crates/holochain/src/bin/holochain/main.rs:173:47:
called `Result::unwrap()` on an `Err` value: Custom {
  kind: Other,
  error: "No such device or address (os error 6)"
}
```

**Same ENXIO error** as on host system.

---

## Analysis

### Why IPv6 Config Didn't Work

The Dockerfile approach of echoing to `/etc/sysctl.conf`:
```dockerfile
RUN echo "net.ipv6.conf.all.disable_ipv6 = 0" >> /etc/sysctl.conf
```

**Doesn't actually apply the setting** because:
1. `sysctl.conf` is only loaded at boot time or with `sysctl -p`
2. Docker containers don't run a full init system
3. The setting is written but never activated

### What This Confirms

1. **Error occurs before network config**: Panic at line 173 in `main.rs`, before network initialization
2. **Not a host-specific issue**: Same error in isolated container
3. **Deeper than simple IPv6**: Even if IPv6 were enabled, error persists
4. **Configuration format gap**: Need Kitsune2 `advanced` field documentation

---

## Possible Solutions

### Option 1: Proper IPv6 in Docker
```dockerfile
# Actually enable IPv6 (requires privileged container)
RUN sysctl -w net.ipv6.conf.all.disable_ipv6=0
```
**Issue**: Requires `--privileged` flag and may not work in all environments

### Option 2: Docker IPv6 Networking
```yaml
# docker-compose.yml
networks:
  zerotrustml-net:
    enable_ipv6: true
    ipam:
      config:
        - subnet: "fd00::/64"
```
**Worth trying**: Proper Docker IPv6 network configuration

### Option 3: Find Kitsune2 IPv4 Config
Research `advanced` field schema for Kitsune2 network configuration

### Option 4: Use Older Holochain
Try pre-Kitsune2 version where `transport_pool` IPv4 fix worked

---

## Recommendation

**This blocker requires specialized Holochain/networking expertise**, not just general development work. Recommended actions:

1. **Ask Holochain community**: Post to Discord/forum about Kitsune2 IPv4-only configuration
2. **Consult Holochain docs**: Search for `advanced` network field documentation
3. **Try older version**: Test with pre-Kitsune2 Holochain temporarily for development
4. **Deploy elsewhere**: Use environment with working IPv6 support

---

## ✅ UPDATE: PROBLEM SOLVED (17:09 CST)

### The Working Solution

Found the working configuration pattern in `docker-compose.multi-node.yml` and applied it:

**Three Critical Changes:**
1. **Use `sysctls` in docker-compose.yml** (not Dockerfile):
   ```yaml
   sysctls:
     - net.ipv6.conf.all.disable_ipv6=0
   ```
   This actually applies IPv6 settings at container level.

2. **Switch to minimal config**:
   ```yaml
   volumes:
     - ./holochain/conductor-config-minimal.yaml:/conductor-config.yaml:ro
   ```

3. **Use passphrase stdin**:
   ```yaml
   command: sh -c "echo 'zerotrustml-dev-passphrase' | holochain -p -c /conductor-config.yaml"
   ```

### Result

```
Conductor ready.
Conductor startup: passphrase obtained.
Conductor startup: networking started.
WebsocketListener listening [addr=127.0.0.1:8888]
WebsocketListener listening [addr=[::1]:8888]
Conductor successfully initialized.
```

✅ **NO ENXIO ERROR - CONDUCTOR RUNNING SUCCESSFULLY**

### Why It Works

- `sysctls` in docker-compose actually applies kernel parameters (vs Dockerfile `echo` which doesn't)
- Minimal config avoids unnecessary complexity
- Passphrase via stdin works reliably without keystore issues

---

## Files Updated

- `docker-compose.holochain.yml` - Applied working configuration pattern
- `SESSION_SUMMARY_2025-11-13.md` - Updated with successful resolution
- `DOCKER_ATTEMPT_2025-11-13.md` - This file (documented solution)

---

*Initial attempt: November 13, 2025, 16:51 CST*
*Status: ❌ Failed with ENXIO*

*Solution found: November 13, 2025, 17:09 CST*
*Status: ✅ **SUCCESS - Conductor running***
*Method: Applied working pattern from docker-compose.multi-node.yml*
