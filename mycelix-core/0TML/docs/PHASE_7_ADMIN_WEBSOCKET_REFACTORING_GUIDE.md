# Phase 7: AdminWebsocket API Refactoring Guide

**Date**: 2025-09-30
**Purpose**: Step-by-step guide for refactoring rust-bridge to use holochain_client's AdminWebsocket API
**Prerequisites**: Dependencies resolved with holochain_client v0.8.0-dev.20 ✅

---

## 🎯 Objective

Replace manual WebSocket + MessagePack serialization with the official `holochain_client::AdminWebsocket` API for production-ready admin operations.

---

## 📊 Current vs Target Architecture

### Current Implementation (Manual WebSocket)
```rust
// Manual WebSocket connection
Arc<Mutex<Option<WebSocketStream>>>

// Manual serialization
let request = AdminRequest::GenerateAgentPubKey;
let bytes = rmp_serde::to_vec(&request)?;
ws.send(Message::Binary(bytes)).await?;

// Manual response handling
loop {
    match ws.next().await {
        Message::Binary(bytes) => break bytes,
        Message::Ping(data) => ws.send(Message::Pong(data)).await?,
        // ... handle other messages
    }
}
let response: AdminResponse = rmp_serde::from_slice(&bytes)?;
```

### Target Implementation (AdminWebsocket API)
```rust
// High-level AdminWebsocket
Arc<Mutex<Option<AdminWebsocket>>>

// Direct API calls
let admin_ws = AdminWebsocket::connect(url).await?;
let pubkey = admin_ws.generate_agent_pub_key().await?;
```

**Benefits**:
- ✅ Handles Ping/Pong automatically
- ✅ Manages serialization/deserialization
- ✅ Provides type-safe API
- ✅ Eliminates 200+ lines of manual protocol handling

---

## 🔧 Step 1: Update Imports

**File**: `rust-bridge/src/lib.rs`
**Lines**: 1-16

### Current Imports
```rust
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};
use tokio_tungstenite::MaybeTlsStream;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::http::Request;

// Holochain official types for admin API
use holochain_conductor_api::{AdminRequest, AdminResponse};
use holochain_types::prelude::*;

type WebSocketStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>;
```

### Target Imports
```rust
// holochain_client provides everything we need
use holochain_client::{
    AdminWebsocket,
    InstallAppPayload,
    AgentPubKey,
};

// Keep these for compatibility
use holochain_conductor_api::{AdminRequest, AdminResponse};
use holochain_types::prelude::*;
```

**Changes**:
- ❌ Remove: `tokio_tungstenite`, `futures_util`, `Message`, `WebSocketStream` imports
- ✅ Add: `holochain_client::AdminWebsocket` and related types

---

## 🔧 Step 2: Update HolochainBridge Structure

**File**: `rust-bridge/src/lib.rs`
**Lines**: ~165-180

### Current Structure
```rust
#[pyclass]
struct HolochainBridge {
    ws_url: String,
    enabled: bool,
    ws_client: Arc<Mutex<Option<WebSocketStream>>>,  // Manual WebSocket
    runtime: Arc<tokio::runtime::Runtime>,
}
```

### Target Structure
```rust
#[pyclass]
struct HolochainBridge {
    ws_url: String,
    enabled: bool,
    admin_ws: Arc<Mutex<Option<AdminWebsocket>>>,  // AdminWebsocket API
    runtime: Arc<tokio::runtime::Runtime>,
}
```

**Changes**:
- ❌ Remove: `ws_client: Arc<Mutex<Option<WebSocketStream>>>`
- ✅ Add: `admin_ws: Arc<Mutex<Option<AdminWebsocket>>>`

---

## 🔧 Step 3: Update connect() Method

**File**: `rust-bridge/src/lib.rs`
**Lines**: ~209-239

### Current Implementation (Manual WebSocket)
```rust
fn connect(&self, py: Python) -> PyResult<bool> {
    py.allow_threads(|| {
        self.runtime.block_on(async {
            let ws_client = Arc::clone(&self.ws_client);
            let mut ws_guard = ws_client.lock().await;

            // Manual WebSocket connection
            let request = Request::builder()
                .uri(&self.ws_url)
                .body(())
                .unwrap();

            let (ws_stream, _) = connect_async(request).await
                .map_err(|e| PyRuntimeError::new_err(format!("Connection failed: {}", e)))?;

            *ws_guard = Some(ws_stream);
            Ok(true)
        })
    })
}
```

### Target Implementation (AdminWebsocket API)
```rust
fn connect(&self, py: Python) -> PyResult<bool> {
    py.allow_threads(|| {
        self.runtime.block_on(async {
            let admin_ws_arc = Arc::clone(&self.admin_ws);
            let mut admin_guard = admin_ws_arc.lock().await;

            // High-level AdminWebsocket connection
            let admin_ws = AdminWebsocket::connect(&self.ws_url).await
                .map_err(|e| PyRuntimeError::new_err(format!("Connection failed: {}", e)))?;

            *admin_guard = Some(admin_ws);
            Ok(true)
        })
    })
}
```

**Changes**:
- ❌ Remove: Manual `Request::builder()` and `connect_async()`
- ✅ Replace with: `AdminWebsocket::connect(url).await`

**Lines affected**: ~30 lines simplified to ~10 lines

---

## 🔧 Step 4: Refactor generate_agent_key()

**File**: `rust-bridge/src/lib.rs`
**Lines**: 673-749

### Current Implementation (77 lines)
```rust
fn generate_agent_key(&self, py: Python) -> PyResult<String> {
    py.allow_threads(|| {
        self.runtime.block_on(async {
            let ws_client = Arc::clone(&self.ws_client);
            let mut ws_guard = ws_client.lock().await;
            let ws = ws_guard.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Not connected"))?;

            // Manual serialization
            let request = AdminRequest::GenerateAgentPubKey;
            let request_bytes = rmp_serde::to_vec(&request)
                .map_err(|e| PyRuntimeError::new_err(format!("Serialize failed: {}", e)))?;

            // Manual send
            ws.send(Message::Binary(request_bytes)).await
                .map_err(|e| PyRuntimeError::new_err(format!("Send failed: {}", e)))?;

            // Manual response handling with Ping/Pong
            let response_bytes = loop {
                let response_msg = ws.next().await
                    .ok_or_else(|| PyRuntimeError::new_err("Connection closed"))?
                    .map_err(|e| PyRuntimeError::new_err(format!("Receive failed: {}", e)))?;

                match response_msg {
                    Message::Binary(bytes) => break bytes,
                    Message::Ping(data) => {
                        ws.send(Message::Pong(data)).await
                            .map_err(|e| PyRuntimeError::new_err(format!("Pong failed: {}", e)))?;
                    },
                    Message::Pong(_) => {},
                    Message::Text(text) => {
                        return Err(PyRuntimeError::new_err(format!("Expected binary: {}", text)));
                    },
                    Message::Close(_) => {
                        return Err(PyRuntimeError::new_err("Connection closed"));
                    },
                    _ => {
                        return Err(PyRuntimeError::new_err("Unexpected message"));
                    }
                }
            };

            // Manual deserialization
            let response: AdminResponse = rmp_serde::from_slice(&response_bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("Deserialize failed: {}", e)))?;

            // Extract from response
            match response {
                AdminResponse::AgentPubKeyGenerated(agent_key) => {
                    Ok(format!("{:?}", agent_key))
                }
                _ => Err(PyRuntimeError::new_err("Unexpected response type"))
            }
        })
    })
}
```

### Target Implementation (~20 lines)
```rust
fn generate_agent_key(&self, py: Python) -> PyResult<String> {
    py.allow_threads(|| {
        self.runtime.block_on(async {
            let admin_ws_arc = Arc::clone(&self.admin_ws);
            let mut admin_guard = admin_ws_arc.lock().await;

            let admin_ws = admin_guard.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Not connected to conductor"))?;

            // High-level API call - handles everything internally
            let agent_key = admin_ws.generate_agent_pub_key().await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to generate agent key: {}", e)))?;

            Ok(format!("{:?}", agent_key))
        })
    })
}
```

**Changes**:
- ❌ Remove: ~60 lines of manual protocol handling
- ✅ Replace with: Single `admin_ws.generate_agent_pub_key().await` call

**Reduction**: 77 lines → 20 lines (73% reduction)

---

## 🔧 Step 5: Refactor install_app()

**File**: `rust-bridge/src/lib.rs`
**Lines**: 494-670

### Current Implementation (177 lines)
```rust
fn install_app(&self, py: Python, app_id: String, happ_path: String) -> PyResult<String> {
    // Read hApp bundle
    let happ_bytes = fs::read(Path::new(&happ_path))
        .map_err(|e| PyIOError::new_err(format!("Failed to read: {}", e)))?;

    // Generate agent key (77 lines)
    let agent_key = self.generate_agent_key(py)?;

    py.allow_threads(|| {
        self.runtime.block_on(async {
            let ws_client = Arc::clone(&self.ws_client);
            let mut ws_guard = ws_client.lock().await;
            let ws = ws_guard.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Not connected"))?;

            // Manual bundle encoding
            use base64::{Engine as _, engine::general_purpose};
            let happ_b64 = general_purpose::STANDARD.encode(&happ_bytes);

            // Manual request structure
            #[derive(serde::Serialize)]
            struct InstallAppRequest {
                id: u32,
                #[serde(rename = "type")]
                req_type: String,
                data: InstallAppData,
            }
            // ... more manual structures

            let request = InstallAppRequest {
                id: 2,
                req_type: "install_app".to_string(),
                data: InstallAppData {
                    installed_app_id: app_id.clone(),
                    agent_key: agent_key.clone(),
                    bundle: BundleData { bundled: happ_b64 },
                    membrane_proofs: serde_json::json!({}),
                    network_seed: None,
                },
            };

            // Manual serialization
            let request_bytes = rmp_serde::to_vec(&request)
                .map_err(|e| PyRuntimeError::new_err(format!("Serialize failed: {}", e)))?;

            // Manual send
            ws.send(Message::Binary(request_bytes)).await
                .map_err(|e| PyRuntimeError::new_err(format!("Send failed: {}", e)))?;

            // Manual response loop (same 40+ lines as generate_agent_key)
            let response_bytes = loop {
                // ... Ping/Pong handling ...
            };

            // Manual deserialization
            let response: Response = rmp_serde::from_slice(&response_bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("Deserialize failed: {}", e)))?;

            // Enable app (another 40+ lines of manual protocol)
            let enable_request = EnableAppRequest { /* ... */ };
            let enable_bytes = rmp_serde::to_vec(&enable_request)?;
            ws.send(Message::Binary(enable_bytes)).await?;
            // ... more manual handling ...

            Ok(app_id)
        })
    })
}
```

### Target Implementation (~50 lines)
```rust
fn install_app(&self, py: Python, app_id: String, happ_path: String) -> PyResult<String> {
    use std::path::Path;
    use std::fs;

    if !self.enabled {
        return Err(PyRuntimeError::new_err("Bridge not enabled"));
    }

    // Read hApp bundle
    let happ_bytes = fs::read(Path::new(&happ_path))
        .map_err(|e| PyIOError::new_err(format!("Failed to read hApp file: {}", e)))?;

    println!("📦 Installing hApp: {} ({} bytes)", app_id, happ_bytes.len());

    // Generate agent key (now only ~20 lines internally)
    let agent_key_str = self.generate_agent_key(py)?;

    // Parse agent key from string representation
    // Note: AdminWebsocket API may return AgentPubKey directly in newer versions
    let agent_key = AgentPubKey::from_raw_39(
        hex::decode(&agent_key_str.trim_matches(|c| c == '(' || c == ')'))
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid agent key format: {}", e)))?
    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to parse agent key: {}", e)))?;

    py.allow_threads(|| {
        self.runtime.block_on(async {
            let admin_ws_arc = Arc::clone(&self.admin_ws);
            let mut admin_guard = admin_ws_arc.lock().await;

            let admin_ws = admin_guard.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Not connected to conductor"))?;

            // Create install payload using official types
            let payload = InstallAppPayload {
                agent_key,
                installed_app_id: Some(app_id.clone().into()),
                membrane_proofs: HashMap::new(),
                network_seed: None,
                // Bundle should be loaded from happ_bytes
                // Check holochain_client API for exact bundle format
            };

            // High-level API call - handles serialization, protocol, and response
            let installed_app = admin_ws.install_app(payload).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to install app: {}", e)))?;

            println!("✅ hApp installed successfully");

            // Enable the app - single API call
            admin_ws.enable_app(app_id.clone().into()).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to enable app: {}", e)))?;

            println!("✅ hApp enabled");

            Ok(app_id)
        })
    })
}
```

**Changes**:
- ❌ Remove: ~120 lines of manual protocol handling
- ✅ Replace with: 2 API calls (`install_app()`, `enable_app()`)

**Reduction**: 177 lines → 50 lines (72% reduction)

---

## 🔧 Step 6: Update __new__() Constructor

**File**: `rust-bridge/src/lib.rs`
**Lines**: ~186-208

### Current Implementation
```rust
#[new]
fn new(
    ws_url: Option<String>,
    enabled: Option<bool>,
) -> PyResult<Self> {
    Ok(Self {
        ws_url: ws_url.unwrap_or_else(|| "ws://localhost:8888".to_string()),
        enabled: enabled.unwrap_or(true),
        ws_client: Arc::new(Mutex::new(None)),  // Manual WebSocket
        runtime: Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Runtime failed: {}", e)))?
        ),
    })
}
```

### Target Implementation
```rust
#[new]
fn new(
    ws_url: Option<String>,
    enabled: Option<bool>,
) -> PyResult<Self> {
    Ok(Self {
        ws_url: ws_url.unwrap_or_else(|| "ws://localhost:8888".to_string()),
        enabled: enabled.unwrap_or(true),
        admin_ws: Arc::new(Mutex::new(None)),  // AdminWebsocket
        runtime: Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Runtime failed: {}", e)))?
        ),
    })
}
```

**Changes**:
- ❌ Remove: `ws_client: Arc::new(Mutex::new(None))`
- ✅ Add: `admin_ws: Arc::new(Mutex::new(None))`

---

## 🔧 Step 7: Remove Unused Helper Methods

**File**: `rust-bridge/src/lib.rs`
**Lines**: 762-843

### Methods to Remove
```rust
async fn ensure_connected(&self) -> Result<(), String> {
    // No longer needed - AdminWebsocket handles reconnection
}

async fn try_connect_internal(&self) -> Result<bool, String> {
    // No longer needed - use connect() instead
}
```

**Reason**: AdminWebsocket handles connection management internally.

---

## 📊 Summary of Changes

### Code Reduction
| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| **generate_agent_key()** | 77 lines | 20 lines | 73% |
| **install_app()** | 177 lines | 50 lines | 72% |
| **connect()** | 30 lines | 10 lines | 67% |
| **Helper methods** | 80 lines | 0 lines | 100% |
| **Total** | ~364 lines | ~80 lines | **78%** |

### Complexity Reduction
- ❌ Manual Ping/Pong handling: Eliminated
- ❌ Manual MessagePack serialization: Eliminated
- ❌ Manual response parsing: Eliminated
- ❌ Manual error handling for WebSocket states: Eliminated
- ✅ Type-safe API calls: Added
- ✅ Automatic protocol handling: Added

---

## 🧪 Testing After Refactoring

### 1. Build Test
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
cargo check
```

**Expected**: Clean build with no errors

### 2. Python Build
```bash
maturin develop --release
```

**Expected**: Python module builds successfully

### 3. Connection Test
```python
import holochain_credits_bridge

bridge = holochain_credits_bridge.HolochainBridge(
    ws_url="ws://localhost:8888",
    enabled=True
)

success = bridge.connect()
print(f"Connection: {success}")
```

**Expected**: `Connection: True`

### 4. Agent Key Test
```python
agent_key = bridge.generate_agent_key()
print(f"Agent key: {agent_key}")
```

**Expected**: Valid agent public key (84 hex characters)

### 5. DNA Installation Test
```bash
# Ensure conductor is running
holochain --config-path conductor-ipv4-only.yaml --piped &

# Run test
python3 test_install_dna.py
```

**Expected**: DNA installs successfully, all zome functions work

---

## ⚠️ Potential Issues & Solutions

### Issue 1: AdminWebsocket API Differences

**Problem**: API method signatures might differ from expectations

**Solution**:
```bash
# Check API documentation
cargo doc --package holochain_client --no-deps --open

# Or check source
cat ~/.cargo/registry/src/*/holochain_client-0.8.0-dev.20/src/admin_websocket.rs
```

### Issue 2: AgentPubKey Format

**Problem**: generate_agent_key() returns string but install_app() needs AgentPubKey

**Current**: `format!("{:?}", agent_key)` gives Debug output

**Solution**: Store raw AgentPubKey:
```rust
// Return serialized bytes instead
Ok(hex::encode(agent_key.as_ref()))

// Or use base64
use base64::{Engine as _, engine::general_purpose};
Ok(general_purpose::STANDARD.encode(agent_key.as_ref()))
```

### Issue 3: InstallAppPayload Structure

**Problem**: Exact structure of InstallAppPayload unclear

**Solution**: Check holochain_client source:
```rust
// Look for
pub struct InstallAppPayload {
    pub source: AppBundleSource,
    pub agent_key: AgentPubKey,
    pub installed_app_id: Option<InstalledAppId>,
    pub membrane_proofs: HashMap<RoleName, MembraneProof>,
    pub network_seed: Option<NetworkSeed>,
}
```

### Issue 4: Version Compatibility

**Problem**: conductor 0.5.6 with holochain_client 0.8.0-dev.20

**Test first**: Current configuration might work (forward compatibility)

**If incompatible**: Upgrade conductor to 0.6.x:
```bash
# Check available versions
nix search nixpkgs holochain

# Or compile from source
git clone https://github.com/holochain/holochain
cd holochain
git checkout holochain-0.6.x
cargo build --release
```

---

## 📋 Checklist for Refactoring

- [ ] **Step 1**: Update imports (remove tokio-tungstenite, add holochain_client)
- [ ] **Step 2**: Change `ws_client` to `admin_ws` in struct
- [ ] **Step 3**: Refactor `connect()` to use `AdminWebsocket::connect()`
- [ ] **Step 4**: Refactor `generate_agent_key()` to use API
- [ ] **Step 5**: Refactor `install_app()` to use API
- [ ] **Step 6**: Update `__new__()` constructor
- [ ] **Step 7**: Remove unused helper methods
- [ ] **Step 8**: Run `cargo check` - verify compilation
- [ ] **Step 9**: Run `maturin develop --release` - build Python module
- [ ] **Step 10**: Test connection with Python
- [ ] **Step 11**: Test agent key generation
- [ ] **Step 12**: Test DNA installation
- [ ] **Step 13**: Test all zome functions
- [ ] **Step 14**: Update documentation

---

## 🎯 Expected Outcome

After completing this refactoring:

1. ✅ **Code Quality**: 78% reduction in complexity
2. ✅ **Maintainability**: Using official API instead of manual protocol
3. ✅ **Reliability**: AdminWebsocket handles edge cases automatically
4. ✅ **Type Safety**: Compile-time checking of admin operations
5. ✅ **Future Proof**: Compatible with Holochain ecosystem updates

---

## 📚 Resources

### Documentation
- holochain_client API: `cargo doc --package holochain_client --no-deps --open`
- Holochain Admin API: https://docs.rs/holochain_conductor_api/latest/
- MessagePack format: https://msgpack.org/

### Source Code
- holochain_client: `~/.cargo/registry/src/*/holochain_client-0.8.0-dev.20/`
- Current implementation: `rust-bridge/src/lib.rs`

### Related Docs
- PHASE_7_HOLOCHAIN_CLIENT_INTEGRATION_STATUS.md - Dependency resolution
- PHASE_7_EXECUTIVE_SUMMARY.md - Overall Phase 7 status
- QUICK_START_PHASE_7.md - Quick reference

---

## 🔮 Next Steps After Refactoring

1. **Test thoroughly** with conductor 0.5.6
2. **If incompatible**: Upgrade to conductor 0.6.x
3. **Document compatibility** requirements
4. **Update QUICK_START** with new usage
5. **Create Phase 7 completion report**
6. **Move to Phase 8**: Trust metrics integration

---

**Created**: 2025-09-30
**Version**: v1.0
**Purpose**: Comprehensive refactoring guide for AdminWebsocket integration
**Estimated Time**: 1-2 hours for complete refactoring + testing

🎯 This guide provides everything needed to complete the holochain_client integration!
