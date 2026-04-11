// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{bail, Context, Result};
use blake2::{Blake2b512, Digest};
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::fs;

use crate::client::MycellixClient;
use crate::config::Config;

/// Initialize Mycelix Mail
///
/// Sets up user profile, generates keys, and registers DID
pub async fn handle_init(
    email: Option<String>,
    import_keys: Option<String>,
    conductor_url: &str,
    did_registry_url: &str,
    matl_bridge_url: &str,
) -> Result<()> {
    println!("🍄 Initializing Mycelix Mail...");
    println!();

    // 1. Check if already initialized
    let config_file = Config::config_file()?;
    if config_file.exists() {
        let existing_config = Config::load()?;
        if existing_config.identity.did.is_some() {
            bail!(
                "Already initialized! DID: {}\nUse 'mycelix-mail did whoami' to see your identity.",
                existing_config.identity.did.unwrap()
            );
        }
    }

    // 2. Load or create configuration
    let mut config = Config::load_or_create()?;

    // 3. Generate or import keys
    let (signing_key, verifying_key) = if let Some(keys_path) = import_keys {
        println!("🔑 Importing existing keys from {}...", keys_path);
        // TODO: Implement key import from file
        bail!("Key import not yet implemented. Please generate new keys instead.");
    } else {
        println!("🔑 Generating new Ed25519 keypair...");

        // Generate cryptographically secure random keypair
        use rand::RngCore;
        let mut rng = rand::rngs::OsRng;

        // Generate 32 random bytes for the secret key
        let mut secret_bytes = [0u8; 32];
        rng.fill_bytes(&mut secret_bytes);

        // Create signing key from secret bytes
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();

        (signing_key, verifying_key)
    };

    // 4. Save keys to disk
    println!(
        "💾 Saving keys to {}...",
        config.identity.private_key_path.display()
    );

    // Ensure keys directory exists
    if let Some(parent) = config.identity.private_key_path.parent() {
        fs::create_dir_all(parent).context("Failed to create keys directory")?;
    }

    // Save private key (32 bytes) - IMPORTANT: Keep this secure!
    fs::write(&config.identity.private_key_path, signing_key.to_bytes())
        .context("Failed to write private key")?;

    // Save public key (32 bytes)
    fs::write(&config.identity.public_key_path, verifying_key.to_bytes())
        .context("Failed to write public key")?;

    println!("✅ Keys saved successfully");
    println!(
        "   Private key: {}",
        config.identity.private_key_path.display()
    );
    println!(
        "   Public key: {}",
        config.identity.public_key_path.display()
    );

    // 5. Create DID from public key
    println!();
    println!("🆔 Creating DID...");
    let did = create_did(&verifying_key);
    println!("   DID: {}", did);

    // 6. Create agent public key (hex encoded verifying key for Holochain)
    let agent_pub_key = hex::encode(verifying_key.to_bytes());

    // 7. Register DID with registry
    println!();
    println!("📡 Registering DID with registry ({})...", did_registry_url);

    let client = MycellixClient::new(
        conductor_url,
        did_registry_url,
        matl_bridge_url,
        config.clone(),
    )
    .await?;

    match client
        .register_did(did.clone(), agent_pub_key.clone())
        .await
    {
        Ok(_) => {
            println!("✅ DID registered successfully");
        }
        Err(e) => {
            println!("⚠️  Warning: DID registration failed: {}", e);
            println!("   This is okay - you can register later when the registry is online.");
            println!("   Continuing with local setup...");
        }
    }

    // 8. Update configuration with DID and keys
    config.set_did(did.clone())?;
    config.set_agent_key(agent_pub_key)?;

    if let Some(email_addr) = email {
        println!("✉️  Setting email: {}", email_addr);
        config.set_email(email_addr)?;
    }

    println!();
    println!("✅ Initialization complete!");
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Your Identity:");
    println!("  DID:    {}", did);
    println!("  Config: {}", config_file.display());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Next steps:");
    println!("  • Run 'mycelix-mail did whoami' to verify your identity");
    println!(
        "  • Run 'mycelix-mail send <did> --subject \"Hello\" --body \"Test\"' to send a message"
    );
    println!("  • Run 'mycelix-mail inbox' to check for messages");
    println!();

    Ok(())
}

/// Create a DID from a verifying key
///
/// Format: did:mycelix:{base58(blake2b(pubkey))}
///
/// This creates a deterministic DID from the public key using:
/// 1. Blake2b-512 hash of the public key (for collision resistance)
/// 2. Take first 32 bytes of the hash (256 bits of entropy)
/// 3. Base58 encode for human-friendly representation
fn create_did(verifying_key: &VerifyingKey) -> String {
    // Hash the public key with Blake2b-512
    let mut hasher = Blake2b512::new();
    hasher.update(verifying_key.to_bytes());
    let hash = hasher.finalize();

    // Take first 32 bytes of hash and encode as base58
    let hash_bytes = &hash[..32];
    let encoded = bs58::encode(hash_bytes).into_string();

    format!("did:mycelix:{}", encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_did_creation() {
        use rand::RngCore;

        // Generate a test keypair
        let mut rng = rand::rngs::OsRng;
        let mut secret_bytes = [0u8; 32];
        rng.fill_bytes(&mut secret_bytes);

        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();

        // Create DID
        let did = create_did(&verifying_key);

        // Verify format
        assert!(did.starts_with("did:mycelix:"));
        assert!(did.len() > 20); // Should be reasonably long

        // Verify deterministic (same key = same DID)
        let did2 = create_did(&verifying_key);
        assert_eq!(did, did2);
    }

    #[test]
    fn test_different_keys_different_dids() {
        use rand::RngCore;
        let mut rng = rand::rngs::OsRng;

        // Generate first key
        let mut secret1 = [0u8; 32];
        rng.fill_bytes(&mut secret1);
        let key1 = SigningKey::from_bytes(&secret1).verifying_key();

        // Generate second key
        let mut secret2 = [0u8; 32];
        rng.fill_bytes(&mut secret2);
        let key2 = SigningKey::from_bytes(&secret2).verifying_key();

        let did1 = create_did(&key1);
        let did2 = create_did(&key2);

        assert_ne!(did1, did2);
    }
}
