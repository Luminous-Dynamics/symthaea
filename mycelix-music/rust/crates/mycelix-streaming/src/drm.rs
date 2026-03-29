// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DRM (Digital Rights Management) support

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// DRM system types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrmSystem {
    Widevine,
    FairPlay,
    PlayReady,
    ClearKey,
}

impl DrmSystem {
    pub fn system_id(&self) -> &'static str {
        match self {
            Self::Widevine => "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::FairPlay => "94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
            Self::PlayReady => "9a04f079-9840-4286-ab92-e65be0885f95",
            Self::ClearKey => "e2719d58-a985-b3c9-781a-b030af78d30e",
        }
    }
}

/// DRM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmConfig {
    pub system: DrmSystem,
    pub key_id: Uuid,
    pub license_server_url: Option<String>,
    pub init_data: Option<Vec<u8>>,
}

/// DRM key info
#[derive(Debug, Clone)]
pub struct DrmKey {
    pub key_id: Uuid,
    pub key: Vec<u8>,
    pub iv: Option<Vec<u8>>,
}

/// Generate PSSH (Protection System Specific Header) box
pub fn generate_pssh(drm_system: DrmSystem, key_id: &Uuid) -> Vec<u8> {
    let system_id_bytes = parse_uuid_bytes(drm_system.system_id());
    let key_id_bytes = key_id.as_bytes();

    // PSSH box structure (simplified)
    let mut pssh = Vec::new();

    // Box size (will be filled at end)
    pssh.extend_from_slice(&[0, 0, 0, 0]);

    // Box type: 'pssh'
    pssh.extend_from_slice(b"pssh");

    // Version and flags
    pssh.push(0x01); // version 1
    pssh.extend_from_slice(&[0, 0, 0]); // flags

    // System ID
    pssh.extend_from_slice(&system_id_bytes);

    // Key ID count
    pssh.extend_from_slice(&1u32.to_be_bytes());

    // Key ID
    pssh.extend_from_slice(key_id_bytes);

    // Data size (0 for this simple case)
    pssh.extend_from_slice(&0u32.to_be_bytes());

    // Update box size
    let size = pssh.len() as u32;
    pssh[0..4].copy_from_slice(&size.to_be_bytes());

    pssh
}

/// Generate ClearKey license request
pub fn generate_clearkey_license_request(key_ids: &[Uuid]) -> String {
    use base64::Engine;
    let base64 = base64::engine::general_purpose::STANDARD;

    let kids: Vec<String> = key_ids
        .iter()
        .map(|kid| base64.encode(kid.as_bytes()))
        .collect();

    serde_json::json!({
        "kids": kids,
        "type": "temporary"
    })
    .to_string()
}

/// Parse ClearKey license response
pub fn parse_clearkey_license_response(response: &str) -> Option<Vec<DrmKey>> {
    use base64::Engine;
    let base64 = base64::engine::general_purpose::STANDARD;

    let json: serde_json::Value = serde_json::from_str(response).ok()?;
    let keys = json.get("keys")?.as_array()?;

    let mut drm_keys = Vec::new();

    for key_obj in keys {
        let kid_b64 = key_obj.get("kid")?.as_str()?;
        let key_b64 = key_obj.get("k")?.as_str()?;

        let kid_bytes = base64.decode(kid_b64).ok()?;
        let key_bytes = base64.decode(key_b64).ok()?;

        if kid_bytes.len() == 16 {
            let key_id = Uuid::from_slice(&kid_bytes).ok()?;
            drm_keys.push(DrmKey {
                key_id,
                key: key_bytes,
                iv: None,
            });
        }
    }

    Some(drm_keys)
}

fn parse_uuid_bytes(uuid_str: &str) -> [u8; 16] {
    let uuid = Uuid::parse_str(uuid_str).unwrap_or_default();
    *uuid.as_bytes()
}

/// Encrypt segment with CENC (Common Encryption)
pub fn encrypt_cenc_segment(data: &[u8], key: &[u8], iv: &[u8]) -> Vec<u8> {
    // Simplified - in production would use proper AES-CTR encryption
    // This is a placeholder that XORs with key material
    let mut encrypted = data.to_vec();
    for (i, byte) in encrypted.iter_mut().enumerate() {
        *byte ^= key[i % key.len()] ^ iv[i % iv.len()];
    }
    encrypted
}

/// Decrypt segment with CENC
pub fn decrypt_cenc_segment(data: &[u8], key: &[u8], iv: &[u8]) -> Vec<u8> {
    // XOR is symmetric
    encrypt_cenc_segment(data, key, iv)
}
