# Deprecated Services

These Python services have been replaced by the Rust backend (`backend-rs/`).

## Why Deprecated

### did-registry/
- **Problem**: Centralized SQLite database for DID resolution
- **Replacement**: DID resolution now happens via Holochain DHT
- **Location**: `backend-rs/src/services/holochain.rs` → `resolve_did()`

### matl-bridge/
- **Problem**: Separate Python service for MATL trust sync
- **Replacement**: Trust operations integrated into Rust backend with caching
- **Location**: `backend-rs/src/services/trust_cache.rs`

## Migration

All functionality is now available through:
1. **Holochain DNA** - `mail_messages` zome has `register_my_did()` and `resolve_did()`
2. **Rust Backend** - Axum server with trust caching and DHT-based DID resolution

## Do Not Use

These services should not be used in new deployments. They are kept for reference only.
