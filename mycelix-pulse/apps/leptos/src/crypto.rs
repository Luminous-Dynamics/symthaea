// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Client-side encryption for Mycelix Pulse.
//!
//! This module now uses a real shared-secret flow for the active web app path:
//! X25519 key agreement in Rust/WASM, HKDF-SHA256 for key derivation, and
//! AES-256-GCM via Web Crypto for payload encryption.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use ed25519_dalek::Signer;
use hkdf::Hkdf;
use sha2::Sha256;
use wasm_bindgen_futures::JsFuture;
use x25519_dalek::{PublicKey, StaticSecret};

const LOCAL_IDENTITY_SECRET_KEY: &str = "mycelix_pulse_identity_secret_b64";
const LOCAL_IDENTITY_PUBLIC_KEY: &str = "mycelix_pulse_identity_public_b64";
const LOCAL_SIGNING_SECRET_KEY: &str = "mycelix_pulse_signing_secret_b64";
const LOCAL_SIGNING_PUBLIC_KEY: &str = "mycelix_pulse_signing_public_b64";
const HKDF_INFO_SUBJECT: &[u8] = b"mycelix-pulse-v1-subject";
const HKDF_INFO_BODY: &[u8] = b"mycelix-pulse-v1-body";

pub struct MessageCrypto {
    pub ephemeral_pubkey: Vec<u8>,
    pub nonce: [u8; 24],
    pub subject_key: [u8; 32],
    pub body_key: [u8; 32],
}

/// Generate cryptographically secure random bytes.
pub fn generate_nonce(len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    let crypto = web_sys::window().unwrap().crypto().expect("crypto API required");
    crypto
        .get_random_values_with_u8_array(&mut buf)
        .expect("RNG failed");
    buf
}

/// Ensure a local identity keypair exists and return the public key bytes.
pub fn ensure_local_identity_keypair() -> Result<Vec<u8>, String> {
    if let Some(public_key) = load_local_identity_public_key() {
        return Ok(public_key.to_vec());
    }

    let secret_bytes = generate_nonce(32);
    let secret_array: [u8; 32] = secret_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "Could not generate 32-byte identity key".to_string())?;

    let secret = StaticSecret::from(secret_array);
    let public = PublicKey::from(&secret);

    store_local_identity(&secret.to_bytes(), &public.to_bytes())?;

    Ok(public.to_bytes().to_vec())
}

pub fn local_identity_public_key() -> Option<Vec<u8>> {
    load_local_identity_public_key().map(|pk| pk.to_vec())
}

pub fn generate_public_key_bytes() -> Vec<u8> {
    let secret_bytes = generate_nonce(32);
    let secret_array: [u8; 32] = secret_bytes
        .as_slice()
        .try_into()
        .expect("32-byte key generation failed");
    let secret = StaticSecret::from(secret_array);
    let public = PublicKey::from(&secret);
    public.to_bytes().to_vec()
}

/// Sign content||nonce with the local Ed25519 signing key.
pub fn sign_message(content: &[u8], nonce: &[u8]) -> Vec<u8> {
    let secret = match load_signing_secret_key() {
        Some(s) => s,
        None => {
            // Fallback: ensure signing keypair exists and retry
            let _ = ensure_signing_keypair();
            match load_signing_secret_key() {
                Some(s) => s,
                None => return vec![0u8; 64], // last resort fallback
            }
        }
    };
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret);
    let mut message = Vec::with_capacity(content.len() + nonce.len());
    message.extend_from_slice(content);
    message.extend_from_slice(nonce);
    signing_key.sign(&message).to_bytes().to_vec()
}

fn ensure_signing_keypair() -> Result<(), String> {
    if load_signing_secret_key().is_some() { return Ok(()); }
    let seed_bytes = generate_nonce(32);
    let seed_array: [u8; 32] = seed_bytes.as_slice().try_into()
        .map_err(|_| "Could not generate signing seed".to_string())?;
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed_array);
    let verifying_key = signing_key.verifying_key();
    let storage = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .ok_or_else(|| "localStorage unavailable".to_string())?;
    storage.set_item(LOCAL_SIGNING_SECRET_KEY, &base64_encode(signing_key.as_bytes()))
        .map_err(|e| format!("{e:?}"))?;
    storage.set_item(LOCAL_SIGNING_PUBLIC_KEY, &base64_encode(verifying_key.as_bytes()))
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}

fn load_signing_secret_key() -> Option<[u8; 32]> {
    let storage = web_sys::window()?.local_storage().ok().flatten()?;
    let secret = storage.get_item(LOCAL_SIGNING_SECRET_KEY).ok().flatten()?;
    base64_decode(&secret).try_into().ok()
}

pub fn decode_transport_text(bytes: &[u8]) -> Option<String> {
    String::from_utf8(bytes.to_vec()).ok()
}

pub fn derive_message_crypto(recipient_pubkey: &[u8]) -> Result<MessageCrypto, String> {
    let recipient_array: [u8; 32] = recipient_pubkey
        .try_into()
        .map_err(|_| "Recipient public key must be 32 bytes".to_string())?;

    let ephemeral_secret_bytes = generate_nonce(32);
    let ephemeral_secret_array: [u8; 32] = ephemeral_secret_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "Could not create ephemeral secret".to_string())?;
    let ephemeral_secret = StaticSecret::from(ephemeral_secret_array);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);
    let recipient_public = PublicKey::from(recipient_array);
    let shared_secret = ephemeral_secret.diffie_hellman(&recipient_public);

    let mut subject_key = [0u8; 32];
    let mut body_key = [0u8; 32];
    let hk = Hkdf::<Sha256>::new(Some(ephemeral_public.as_bytes()), shared_secret.as_bytes());
    hk.expand(HKDF_INFO_SUBJECT, &mut subject_key)
        .map_err(|_| "Could not derive subject key".to_string())?;
    hk.expand(HKDF_INFO_BODY, &mut body_key)
        .map_err(|_| "Could not derive body key".to_string())?;

    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&generate_nonce(24));

    Ok(MessageCrypto {
        ephemeral_pubkey: ephemeral_public.to_bytes().to_vec(),
        nonce,
        subject_key,
        body_key,
    })
}

pub fn derive_message_crypto_for_local_recipient(
    ephemeral_pubkey: &[u8],
    nonce: &[u8],
) -> Result<MessageCrypto, String> {
    let secret = load_local_identity_secret_key()
        .ok_or_else(|| "No local identity secret available".to_string())?;
    let ephemeral_array: [u8; 32] = ephemeral_pubkey
        .try_into()
        .map_err(|_| "Ephemeral public key must be 32 bytes".to_string())?;
    let recipient_secret = StaticSecret::from(secret);
    let shared_secret = recipient_secret.diffie_hellman(&PublicKey::from(ephemeral_array));
    let mut subject_key = [0u8; 32];
    let mut body_key = [0u8; 32];
    let hk = Hkdf::<Sha256>::new(Some(ephemeral_pubkey), shared_secret.as_bytes());
    hk.expand(HKDF_INFO_SUBJECT, &mut subject_key)
        .map_err(|_| "Could not derive subject key".to_string())?;
    hk.expand(HKDF_INFO_BODY, &mut body_key)
        .map_err(|_| "Could not derive body key".to_string())?;

    let nonce_array: [u8; 24] = nonce
        .try_into()
        .map_err(|_| "Nonce must be 24 bytes".to_string())?;

    Ok(MessageCrypto {
        ephemeral_pubkey: ephemeral_pubkey.to_vec(),
        nonce: nonce_array,
        subject_key,
        body_key,
    })
}

pub async fn encrypt_with_key(
    plaintext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 24],
) -> Result<Vec<u8>, String> {
    let pt_b64 = base64_encode(plaintext);
    let key_b64 = base64_encode(key);
    let iv_b64 = base64_encode(&nonce[..12]);

    let js_code = format!(
        r#"(async function() {{
            const keyBytes = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const iv = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const data = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const key = await crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false, ['encrypt']);
            const ct = await crypto.subtle.encrypt({{name: 'AES-GCM', iv}}, key, data);
            return btoa(String.fromCharCode(...new Uint8Array(ct)));
        }})()"#,
        key_b64, iv_b64, pt_b64
    );

    let promise = js_sys::eval(&js_code).map_err(|e| format!("eval failed: {e:?}"))?;
    let result = JsFuture::from(js_sys::Promise::from(promise))
        .await
        .map_err(|e| format!("AES-GCM encrypt failed: {e:?}"))?;

    let ct_b64 = result.as_string().ok_or("not a string")?;
    Ok(base64_decode(&ct_b64))
}

pub async fn decrypt_with_key(
    ciphertext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 24],
) -> Result<Vec<u8>, String> {
    let ct_b64 = base64_encode(ciphertext);
    let key_b64 = base64_encode(key);
    let iv_b64 = base64_encode(&nonce[..12]);

    let js_code = format!(
        r#"(async function() {{
            const keyBytes = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const iv = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const data = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));
            const key = await crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false, ['decrypt']);
            const pt = await crypto.subtle.decrypt({{name: 'AES-GCM', iv}}, key, data);
            return btoa(String.fromCharCode(...new Uint8Array(pt)));
        }})()"#,
        key_b64, iv_b64, ct_b64
    );

    let promise = js_sys::eval(&js_code).map_err(|e| format!("eval failed: {e:?}"))?;
    let result = JsFuture::from(js_sys::Promise::from(promise))
        .await
        .map_err(|e| format!("AES-GCM decrypt failed: {e:?}"))?;

    let pt_b64 = result.as_string().ok_or("not a string")?;
    Ok(base64_decode(&pt_b64))
}

fn store_local_identity(secret: &[u8; 32], public: &[u8; 32]) -> Result<(), String> {
    let storage = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .ok_or_else(|| "localStorage unavailable".to_string())?;
    storage
        .set_item(LOCAL_IDENTITY_SECRET_KEY, &base64_encode(secret))
        .map_err(|e| format!("Could not store identity secret: {e:?}"))?;
    storage
        .set_item(LOCAL_IDENTITY_PUBLIC_KEY, &base64_encode(public))
        .map_err(|e| format!("Could not store identity public key: {e:?}"))?;
    Ok(())
}

fn load_local_identity_secret_key() -> Option<[u8; 32]> {
    let storage = web_sys::window()?.local_storage().ok().flatten()?;
    let secret = storage.get_item(LOCAL_IDENTITY_SECRET_KEY).ok().flatten()?;
    let secret_bytes = base64_decode(&secret);
    secret_bytes.try_into().ok()
}

fn load_local_identity_public_key() -> Option<[u8; 32]> {
    let storage = web_sys::window()?.local_storage().ok().flatten()?;
    let public = storage.get_item(LOCAL_IDENTITY_PUBLIC_KEY).ok().flatten()?;
    let public_bytes = base64_decode(&public);
    public_bytes.try_into().ok()
}

fn base64_encode(data: &[u8]) -> String {
    BASE64.encode(data)
}

fn base64_decode(s: &str) -> Vec<u8> {
    BASE64.decode(s).unwrap_or_default()
}
