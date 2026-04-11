# Mycelix Pulse — Implementation Plan (All 21 Items)

## Critical Path (Items 1-3): Make It Real

### Item 1: Real Key Exchange
**Problem**: crypto.rs derives encryption keys from the nonce (public data). Completely broken.
**Solution**: Port the backend's X25519 ECDH from `happ/backend-rs/src/services/crypto.rs` to WASM.

**Steps**:
1. Add `x25519-dalek` + `hkdf` + `sha2` to Cargo.toml (all are no_std compatible, work in WASM)
2. On first app load: generate X25519 identity keypair, store in localStorage (encrypted by browser)
3. Call `publish_pre_key_bundle()` to publish our public key to DHT
4. When composing: call `get_pre_key_bundle(recipient)` → perform X3DH → derive shared secret
5. Encrypt with AES-256-GCM using the HKDF-derived key, NOT the nonce
6. Call `consume_pre_key()` to mark the OTPK as used

**Dependencies**: Working conductor connection (Item 2)
**Complexity**: High — but backend code exists to port

### Item 2: Conductor Authentication
**Problem**: BrowserWsTransport connects but auth handshake times out.
**Root cause**: The load functions fire before connection settles. The exponential backoff helps but doesn't resolve the underlying auth timing.

**Steps**:
1. Add detailed logging to each step of connect() — WebSocket open, authenticate, app_info
2. The app port 8888 was added via `hc sandbox call --running 33800 add-app-ws 8888` — this creates an UNAUTHENTICATED interface (no token needed). Skip the authenticate step.
3. Debug app_info response parsing — the cell_map population might fail silently
4. Add connection status reactive signal that the UI observes
5. Implement auto-reconnect on WebSocket close

**Complexity**: Medium — mostly debugging, not new code

### Item 3: Real Zome Call Round-Trip
**Problem**: Never proven end-to-end.
**Prerequisite**: Item 2 (working auth)

**Steps**:
1. Fix conductor connection (Item 2)
2. Call `get_folders()` — init creates 7 system folders, this is the simplest test
3. Call `create_contact()` with test data — verify it persists
4. Call `send_email()` with properly encrypted payload (Item 1)
5. Call `get_inbox()` — verify the sent email appears
6. Call `rotate_keys()` — verify key revocation works

**Complexity**: Low once Item 2 is fixed

---

## Functional Items (4-10): Wire the Buttons

### Item 4: Contact Import File Parsing
**Fix**: The FileReader onload closure works in Chrome/Firefox. For Brave, add a fallback that reads the file synchronously if async fails.

### Item 5: Backup Restore
**Fix**: Same FileReader pattern. Add schema validation on import.

### Item 6: Calendar Grid Event Positioning
**Fix**: The TimeGrid component renders hourly slots but events use absolute positioning. Need to calculate `top` and `height` from event start/end times relative to the visible day range.

### Item 7: Chat P2P Signal Dispatch
**Fix**: Wire `remote_signal` calls through the conductor for chat messages. In mock mode, messages already go to local state.

### Item 8: Meet WebRTC Connection
**Fix**: Port the TypeScript `SignalingService` pattern to Rust/WASM. Use `web_sys::RtcPeerConnection` for the WebRTC API. Signaling goes through the federation zome's `send_signal`.

### Item 9: IMAP Bridge Deployment
**Fix**: Build the bridge binary, configure with real Gmail credentials, enable the NixOS service. Test with `sudo systemctl start mail-bridge`.

### Item 10: OAuth2 Callback Handler
**Fix**: Add a route `/oauth/callback` that receives the authorization code, exchanges it for tokens via the token endpoint, and stores the refresh token in the account config.

---

## Polish Items (11-15): Make It Smooth

### Item 11: Welcome Modal Persistence in Headless
**Not a real issue** — only affects automated testing. Real users dismiss it once.

### Item 12: Settings Create/Edit for Signatures, Labels, Filters
**Fix**: Add inline creation forms for each section. Pattern: show/hide form with input fields + save button.

### Item 13: Keyboard Shortcut Full Remapping
**Fix**: Store custom keybindings in `UserPreferences.custom_keybindings: HashMap<String, String>`. On keypress, check custom map first, fall back to defaults.

### Item 14: NLP Calendar Parsing Enhancement
**Fix**: Add month name parsing ("April 15"), relative month ("next month"), and time range ("10am-4pm").

### Item 15: Compiler Warnings Cleanup
**Fix**: `cargo fix --lib --allow-dirty` then manual cleanup of remaining.

---

## Missing Items (16-21): Build New

### Item 16: Email Decryption in Read View
**Fix**: When opening an email in read view, call `decrypt_aes_gcm(encrypted_body, nonce)` with the shared secret (from Item 1). Display decrypted content. If decryption fails (revoked key), show "Content revoked" message.

### Item 17: Attachment Upload
**Fix**: Add drag-and-drop zone to compose page. Use `web_sys::FileReader` to read files as `ArrayBuffer`. Encrypt file content with the same session key. Chunk large files (max 10MB per chunk per the zome spec).

### Item 18: Thread Reply Linking
**Fix**: When replying, set `in_reply_to: Some(original_hash)` and `thread_id` on the new email. The inbox thread grouping already handles display.

### Item 19: Notification Permission Prompt
**Fix**: On first email receive (or first "Explore Demo" click), show an inline banner: "Enable notifications to know when new messages arrive" with Allow/Dismiss buttons. Only prompt once.

### Item 20: Mobile Swipe Gestures
**Fix**: Add touchstart/touchmove/touchend handlers to email cards. Track horizontal displacement. Threshold: 80px. Left swipe = swipe_left action, right swipe = swipe_right action. Show colored background during swipe (green for archive, red for delete).

### Item 21: Symthaea-Core Integration
**Fix**: Add `symthaea-core` with `features = ["hdc"]` to Cargo.toml (behind the `symthaea` feature flag). Replace the 1024D TextEncoder with the real 16,384D version. Wire `MoralParser` for ethical content flags.

---

## Build Priority (Sequenced)

### Sprint 1: Fix the Foundation
- Item 2 (conductor auth debugging)
- Item 15 (cleanup warnings)
- Loading spinner in index.html (Brave fix)

### Sprint 2: Real Encryption
- Item 1 (X25519 ECDH key exchange)
- Item 16 (decrypt in read view)
- Item 3 (end-to-end zome round-trip proof)

### Sprint 3: Wire the Buttons
- Items 4, 5 (file import/export robustness)
- Item 10 (OAuth callback)
- Item 18 (thread reply linking)
- Item 17 (attachment upload)

### Sprint 4: Polish + Mobile
- Items 12, 13, 14 (settings, keyboard, NLP)
- Item 20 (swipe gestures)
- Item 19 (notification prompt)

### Sprint 5: Full Integration
- Item 7 (chat P2P signals)
- Item 8 (WebRTC video)
- Item 9 (IMAP bridge)
- Item 21 (Symthaea-core)
- Item 6 (calendar grid)

---

## Estimated Effort

| Sprint | Items | LOC Estimate | Sessions |
|--------|-------|-------------|----------|
| Sprint 1 | 2, 15, loading spinner | ~200 LOC | 1 |
| Sprint 2 | 1, 3, 16 | ~500 LOC | 1-2 |
| Sprint 3 | 4, 5, 10, 17, 18 | ~400 LOC | 1 |
| Sprint 4 | 12, 13, 14, 19, 20 | ~600 LOC | 1-2 |
| Sprint 5 | 6, 7, 8, 9, 21 | ~1000 LOC | 2-3 |

**Total: ~2,700 LOC across 6-9 sessions**

The current codebase is 12,000+ LOC. This would bring it to ~15,000 LOC with everything wired, encrypted, and real.
