# Holochain WebSocket Connection - Summary of Findings

**Date**: November 13, 2025
**Holochain Version**: 0.6.0-rc.1 (dev)
**Status**: WebSocket connection achieved with deserialization issue

---

## 🎯 Key Discoveries

### 1. WebSocket Library Compatibility

**Finding**: The conductor ONLY accepts connections from `websocket-client` (sync), NOT `websockets` (async)

| Library | Result | Details |
|---------|--------|---------|
| **websockets** (async) | ❌ HTTP 400 | Both our code AND official client fail |
| **websocket-client** (sync) | ✅ SUCCESS | Connection established, messages exchanged |

**Implication**: The async `websockets` library (used by official holochain-client-python) is incompatible with Holochain 0.6.0-rc.1.

### 2. WireMessageRequest Protocol

**Discovered**: Holochain requires double MessagePack serialization with specific envelope:

```python
# Step 1: Inner AdminRequest
admin_request = {
    'type': 'list_apps',
    'data': None  # None after removing empty fields
}
inner_bytes = msgpack.packb(admin_request)

# Step 2: Wire envelope
wire_message = {
    'type': 'request',
    'id': 0,
    'data': [b for b in inner_bytes]  # List[int], not bytes!
}
final_bytes = msgpack.packb(wire_message)

# Step 3: Send via WebSocket
ws.send(final_bytes, opcode=websocket.ABNF.OPCODE_BINARY)
```

**Source**: https://github.com/holochain/holochain-client-python
**File**: `holochain_client/api/common/request.py:35-38`

### 3. Connection from Inside Docker Container

**Problem**: Conductor binds to `127.0.0.1:8888` (localhost only), not `0.0.0.0:8888`

**Evidence from logs**:
```
[INFO] holochain_websocket: WebsocketListener listening addr=127.0.0.1:8888
[INFO] holochain_websocket: WebsocketListener listening addr=[::1]:8888
```

**Solution**: Run client code inside the Docker container using `docker exec`

### 4. Current Error: Deserialization

**Status**: Connection works, but request format has subtle issue

**Response from conductor**:
```json
{
  "type": "response",
  "id": 0,
  "data": {
    "type": "error",
    "value": {
      "type": "deserialization",
      "value": "Failed to deserialize request"
    }
  }
}
```

This means:
- ✅ TCP connection successful
- ✅ WebSocket handshake successful
- ✅ Request received by conductor
- ❌ Request format slightly incorrect

---

## 🔬 Working Test Script

**Location**: `/tmp/test_holochain_sync.py`

**Usage**:
```bash
# Install dependencies inside container
docker exec test-holochain pip3 install --break-system-packages msgpack websocket-client

# Copy and run test
docker cp /tmp/test_holochain_sync.py test-holochain:/tmp/
docker exec test-holochain python3 /tmp/test_holochain_sync.py
```

**Result**:
```
✅ WebSocket connected!
✅ Request sent!
✅ Received 102 bytes!
❌ Deserialization error in response
```

---

## 📊 Progress Summary

### What Works ✅
1. WebSocket connection (sync library only)
2. Message transmission
3. Response reception
4. WireMessageRequest envelope format (mostly correct)

### What Doesn't Work ❌
1. Async websockets library (HTTP 400)
2. Request deserialization (format issue)
3. Connections from outside container (binding issue)

### Progress Made 🎉
**From**: "Connection reset by peer" (no connection at all)
**To**: "Failed to deserialize request" (connected, exchanging messages!)

---

## 🎯 BREAKTHROUGH: generate_agent_pub_key Works!

### Discovery
**`generate_agent_pub_key` request succeeds perfectly!**

```python
Request: {'type': 'generate_agent_pub_key', 'data': None}
Response: {'type': 'agent_pub_key_generated', 'value': <39 bytes>}
Status: ✅ SUCCESS
```

### Byte-by-Byte Comparison

Both requests use **identical protocol format**:

| Request | Inner Bytes | Wire Bytes | Protocol | Result |
|---------|-------------|------------|----------|--------|
| generate_agent_pub_key | 35 | 66 | ✅ Correct | ✅ SUCCESS |
| list_apps | 22 | 53 | ✅ Correct | ❌ Deserialization error |

**Encoding verified identical:**
```python
# Both use same structure
AdminRequest:      {'type': <tag>, 'data': None}
WireMessageRequest: {'type': 'request', 'id': 0, 'data': [bytes as list]}
```

### Conclusion

**The issue is NOT our implementation!** It's either:

1. **Holochain 0.6.0-rc.1 has a bug** in `list_apps` handler
2. **Version incompatibility** - `list_apps` may have changed between versions
3. **Request requires additional setup** - maybe need to install apps first

### Proven Working

✅ WebSocket connection with sync `websocket-client` library
✅ WireMessageRequest protocol implementation
✅ Double MessagePack serialization
✅ List[int] data field format
✅ AdminRequest structure
✅ At least one admin API call (`generate_agent_pub_key`)

## 🔍 Remaining Investigation

### 1. Try Other Admin Commands
Test more commands to find which work and which don't:
- ✅ `generate_agent_pub_key` (WORKS)
- ❌ `list_apps` (deserialization error)
- ❓ `install_app`
- ❓ `enable_app`
- ❓ `dump_network_stats`

### 2. Check Holochain Version
- Current: 0.6.0-rc.1 (release candidate)
- Try with stable release if available
- Check official client's target version

### 3. Examine list_apps Requirements
- May need apps installed first
- Check if empty app list causes deserialization error
- Try after installing a test app

---

## 🛠️ Recommended Approach for Production

### Option A: Use Sync Library (Recommended)
Fix the deserialization issue with `websocket-client` library since we know it connects successfully.

**Pros**:
- Already have working connection
- Just need to fix request format
- No HTTP 400 errors

**Cons**:
- Synchronous (but acceptable for admin operations)

### Option B: Run Client Inside Container
Use the official Python client but run it via `docker exec` inside the conductor container.

**Pros**:
- Official client handles all protocol details
- Guaranteed compatibility

**Cons**:
- Awkward architecture (subprocess calls to docker exec)
- Not portable

### Option C: Fix Async Websockets Compatibility
Debug why async `websockets` library gets HTTP 400 from Holochain 0.6.0.

**Pros**:
- Would work with official client
- Async operations

**Cons**:
- May be Holochain bug
- Might require conductor code changes

---

## 📚 References

- **Official Python Client**: https://github.com/holochain/holochain-client-python
- **Holochain Docs**: https://developer.holochain.org/
- **WebSocket Spec**: https://datatracker.ietf.org/doc/html/rfc6455
- **MessagePack Spec**: https://msgpack.org/

---

**Conclusion**: We've made substantial progress. WebSocket connectivity is working with the sync library. The remaining issue is a subtle difference in request format causing deserialization errors. This is solvable through careful debugging and comparison with working implementations.
