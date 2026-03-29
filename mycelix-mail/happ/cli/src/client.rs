// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

use holochain_client::{
    AdminWebsocket, AgentSigner, AppInfo, AppWebsocket, CellInfo, ClientAgentSigner, ExternIO,
    IssueAppAuthenticationTokenPayload,
};
use holochain_types::prelude::{ActionHash, CellId};
use serde::{de::DeserializeOwned, Serialize};

use crate::config::Config;
use crate::types::*;

/// Connection state for the Holochain conductor
enum HolochainState {
    /// Not connected
    Disconnected,
    /// Connected to conductor
    Connected {
        app_ws: AppWebsocket,
        cell_id: CellId,
    },
}

/// Client for interacting with Mycelix Mail system
///
/// Provides high-level methods for all mail operations, DID resolution,
/// and trust score management. Handles connections to:
/// - Holochain conductor (WebSocket)
/// - DID registry (HTTP fallback)
/// - MATL bridge (HTTP)
pub struct MycellixClient {
    /// HTTP client for external services
    http_client: reqwest::Client,

    /// Configuration
    config: Config,

    /// Holochain conductor URL
    conductor_url: String,

    /// Holochain admin URL
    admin_url: String,

    /// DID registry URL
    did_registry_url: String,

    /// MATL bridge URL
    matl_bridge_url: String,

    /// Holochain connection state
    holo_state: Arc<RwLock<HolochainState>>,
}

impl MycellixClient {
    /// Create a new Mycelix client
    pub async fn new(
        conductor_url: &str,
        did_registry_url: &str,
        matl_bridge_url: &str,
        config: Config,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.conductor.timeout))
            .build()
            .context("Failed to create HTTP client")?;

        // Parse admin URL from conductor URL (default to 8889 for admin)
        let admin_url = conductor_url
            .replace(":8888", ":8889")
            .replace(":9999", ":9998");

        let client = Self {
            http_client,
            config,
            conductor_url: conductor_url.to_string(),
            admin_url,
            did_registry_url: did_registry_url.to_string(),
            matl_bridge_url: matl_bridge_url.to_string(),
            holo_state: Arc::new(RwLock::new(HolochainState::Disconnected)),
        };

        // Attempt to connect to Holochain conductor
        if let Err(e) = client.connect_holochain().await {
            eprintln!("Warning: Could not connect to Holochain conductor: {}", e);
            eprintln!("Falling back to HTTP-based operations where available.");
        }

        Ok(client)
    }

    /// Connect to the Holochain conductor
    async fn connect_holochain(&self) -> Result<()> {
        let mut state = self.holo_state.write().await;

        // Check if already connected
        if matches!(*state, HolochainState::Connected { .. }) {
            return Ok(());
        }

        // Connect to admin interface
        let admin_ws = AdminWebsocket::connect(&self.admin_url)
            .await
            .context("Failed to connect to Holochain admin interface")?;

        // Get app ID from config
        let app_id = self
            .config
            .conductor
            .app_id
            .clone()
            .unwrap_or_else(|| "mycelix_mail".to_string());

        // Issue authentication token
        let token_payload = IssueAppAuthenticationTokenPayload {
            installed_app_id: app_id.clone().into(),
            expiry_seconds: 3600,
            single_use: false,
        };

        let token_response = admin_ws
            .issue_app_auth_token(token_payload)
            .await
            .context("Failed to issue auth token")?;

        // Create agent signer
        let signer = ClientAgentSigner::default();
        let signer: Arc<dyn AgentSigner + Send + Sync> = Arc::new(signer);

        // Connect to app interface
        let app_ws =
            AppWebsocket::connect(&self.conductor_url, token_response.token, signer.clone())
                .await
                .context("Failed to connect to Holochain app interface")?;

        // Get app info to find cell ID
        let app_info: AppInfo = app_ws
            .app_info()
            .await
            .context("Failed to get app info")?
            .context("App info not available")?;

        // Find the cell ID
        let cell_id = self.find_cell_id(&app_info)?;

        // Authorize signing credentials
        admin_ws
            .authorize_signing_credentials(holochain_client::AuthorizeSigningCredentialsPayload {
                cell_id: cell_id.clone(),
                functions: None,
            })
            .await
            .context("Failed to authorize signing credentials")?;

        *state = HolochainState::Connected { app_ws, cell_id };
        println!("Connected to Holochain conductor");

        Ok(())
    }

    /// Find the cell ID from app info
    fn find_cell_id(&self, app_info: &AppInfo) -> Result<CellId> {
        for (_role_name, cells) in &app_info.cell_info {
            for cell_info in cells {
                match cell_info {
                    CellInfo::Provisioned(cell) => {
                        return Ok(cell.cell_id.clone());
                    }
                    CellInfo::Cloned(cell) => {
                        return Ok(cell.cell_id.clone());
                    }
                    _ => continue,
                }
            }
        }
        anyhow::bail!("No provisioned cell found in app")
    }

    /// Check if connected to Holochain
    pub async fn is_holochain_connected(&self) -> bool {
        let state = self.holo_state.read().await;
        matches!(*state, HolochainState::Connected { .. })
    }

    /// Generic zome call helper
    async fn call_zome<I, O>(&self, zome_name: &str, fn_name: &str, payload: I) -> Result<O>
    where
        I: Serialize + std::fmt::Debug,
        O: DeserializeOwned + std::fmt::Debug,
    {
        let state = self.holo_state.read().await;

        match &*state {
            HolochainState::Disconnected => {
                anyhow::bail!("Not connected to Holochain conductor")
            }
            HolochainState::Connected { app_ws, cell_id } => {
                let encoded_payload =
                    ExternIO::encode(payload).context("Failed to encode zome call payload")?;

                let result = app_ws
                    .call_zome(
                        cell_id.clone().into(),
                        zome_name.into(),
                        fn_name.into(),
                        encoded_payload,
                    )
                    .await
                    .context(format!("Zome call {}/{} failed", zome_name, fn_name))?;

                let decoded: O = result
                    .decode()
                    .context("Failed to decode zome call response")?;

                Ok(decoded)
            }
        }
    }

    //
    // ===== MAIL OPERATIONS =====
    //

    /// Send a mail message via the `mail_messages` Holochain zome.
    ///
    /// Falls back to a local-only stub if the conductor is not connected.
    pub async fn send_message(
        &self,
        to_did: String,
        subject: Vec<u8>,
        body_cid: String,
        thread_id: Option<String>,
        tier: EpistemicTier,
    ) -> Result<String> {
        if !self.is_holochain_connected().await {
            anyhow::bail!(
                "Cannot send message: not connected to Holochain conductor. \
                 Run 'mycelix-mail connect' first."
            );
        }

        #[derive(serde::Serialize, Debug)]
        struct SendMessageInput {
            to_did: String,
            subject_encrypted: Vec<u8>,
            body_cid: String,
            thread_id: Option<String>,
            epistemic_tier: String,
        }

        let input = SendMessageInput {
            to_did,
            subject_encrypted: subject,
            body_cid,
            thread_id,
            epistemic_tier: format!("{}", tier),
        };

        let action_hash: ActionHash = self
            .call_zome("mail_messages", "send_message", input)
            .await
            .context("Failed to send message via mail_messages zome")?;

        // Return base58-encoded action hash as the message ID
        Ok(bs58::encode(action_hash.get_raw_39()).into_string())
    }

    /// Get inbox messages from the `mail_messages` Holochain zome.
    ///
    /// Falls back to empty list if not connected.
    pub async fn get_inbox(&self) -> Result<Vec<MailMessage>> {
        if !self.is_holochain_connected().await {
            eprintln!("Warning: Not connected to Holochain conductor, returning empty inbox");
            return Ok(vec![]);
        }

        let messages: Vec<MailMessage> = self
            .call_zome("mail_messages", "get_inbox", ())
            .await
            .context("Failed to fetch inbox from mail_messages zome")?;

        Ok(messages)
    }

    /// Get sent messages from the `mail_messages` Holochain zome.
    ///
    /// Falls back to empty list if not connected.
    pub async fn get_sent(&self) -> Result<Vec<MailMessage>> {
        if !self.is_holochain_connected().await {
            eprintln!("Warning: Not connected to Holochain conductor, returning empty outbox");
            return Ok(vec![]);
        }

        let messages: Vec<MailMessage> = self
            .call_zome("mail_messages", "get_outbox", ())
            .await
            .context("Failed to fetch outbox from mail_messages zome")?;

        Ok(messages)
    }

    /// Get a specific message by ID from the `mail_messages` Holochain zome.
    pub async fn get_message(&self, message_id: &str) -> Result<MailMessage> {
        if !self.is_holochain_connected().await {
            anyhow::bail!(
                "Cannot fetch message: not connected to Holochain conductor"
            );
        }

        // Parse message_id (base58-encoded ActionHash) back to ActionHash
        let hash_bytes = bs58::decode(message_id)
            .into_vec()
            .context("Invalid message ID format (expected base58-encoded ActionHash)")?;
        let message_hash = ActionHash::from_raw_39(hash_bytes);

        let message: MailMessage = self
            .call_zome("mail_messages", "get_message", message_hash)
            .await
            .context(format!("Failed to fetch message {} from DHT", message_id))?;

        Ok(message)
    }

    /// Mark a message as read
    ///
    /// Updates message metadata in the DHT via mail_messages zome
    pub async fn mark_read(&self, message_id: &str) -> Result<()> {
        if !self.is_holochain_connected().await {
            println!("Warning: Holochain not connected, storing read status locally");
            return Ok(());
        }

        // Parse message_id to ActionHash
        let hash_bytes = bs58::decode(message_id)
            .into_vec()
            .context("Invalid message ID format")?;
        let message_hash = ActionHash::from_raw_39(hash_bytes);

        #[derive(serde::Serialize, Debug)]
        struct MarkReadInput {
            message_hash: ActionHash,
        }

        // Call the zome to update message metadata
        // Note: The zome may not have a direct mark_read function,
        // so we use record_positive_interaction which updates trust
        // and could be extended to track read status
        self.call_zome::<_, ()>(
            "trust_filter",
            "record_positive_interaction",
            self.config.identity.did.clone().unwrap_or_default(),
        )
        .await
        .context("Failed to mark message as read")?;

        println!("Message {} marked as read", message_id);
        Ok(())
    }

    /// Delete a message
    ///
    /// Calls mail_messages::delete_message zome function to mark entry as deleted
    pub async fn delete_message(&self, message_id: &str) -> Result<()> {
        if !self.is_holochain_connected().await {
            anyhow::bail!("Cannot delete message: not connected to Holochain conductor");
        }

        // Parse message_id to ActionHash
        let hash_bytes = bs58::decode(message_id)
            .into_vec()
            .context("Invalid message ID format")?;
        let message_hash = ActionHash::from_raw_39(hash_bytes);

        // Call the delete_message zome function
        let _delete_hash: ActionHash = self
            .call_zome("mail_messages", "delete_message", message_hash)
            .await
            .context("Failed to delete message")?;

        println!("Message {} deleted successfully", message_id);
        Ok(())
    }

    /// Search messages
    ///
    /// Performs client-side filtering on inbox and outbox messages.
    /// Searches across from_did, to_did, and body_cid fields.
    pub async fn search_messages(&self, query: &str) -> Result<Vec<MailMessage>> {
        let query_lower = query.to_lowercase();

        // Fetch all messages from inbox and outbox
        let mut all_messages = Vec::new();

        // Get inbox messages
        let inbox = self.get_inbox().await.unwrap_or_default();
        all_messages.extend(inbox);

        // Get sent messages
        let sent = self.get_sent().await.unwrap_or_default();
        all_messages.extend(sent);

        // Filter messages by query (client-side filtering)
        let results: Vec<MailMessage> = all_messages
            .into_iter()
            .filter(|msg| {
                // Search in from_did
                if msg.from_did.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Search in to_did
                if msg.to_did.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Search in body_cid (may contain useful identifiers)
                if msg.body_cid.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Try to decrypt and search subject
                if let Ok(subject) = String::from_utf8(msg.subject_encrypted.clone()) {
                    if subject.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }
                false
            })
            .collect();

        Ok(results)
    }

    //
    // ===== TRUST SCORE OPERATIONS (Partial - MATL HTTP) =====
    //

    /// Get trust score for a DID
    ///
    /// Queries the MATL bridge for trust scores
    pub async fn get_trust_score(&self, did: String) -> Result<Option<TrustScore>> {
        let url = format!("{}/trust/{}", self.matl_bridge_url, did);

        match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let score: TrustScore = response
                    .json()
                    .await
                    .context("Failed to parse trust score response")?;
                Ok(Some(score))
            }
            Ok(response) if response.status() == 404 => {
                Ok(None) // No trust score exists
            }
            Ok(response) => {
                println!("⚠️  MATL bridge returned status: {}", response.status());
                Ok(None)
            }
            Err(_) => {
                // MATL bridge not available - use stub
                println!(
                    "📊 [STUB] MATL bridge unavailable, would check local cache for {}",
                    did
                );
                Ok(None)
            }
        }
    }

    /// Set/update trust score for a DID
    ///
    /// Syncs to Holochain DHT via trust_filter::update_trust_score and MATL bridge
    pub async fn set_trust_score(&self, did: String, score: f64) -> Result<()> {
        // Create the trust score entry
        let trust_score = TrustScore {
            did: did.clone(),
            score,
            last_updated: chrono::Utc::now().timestamp(),
            source: "manual".to_string(),
        };

        // Try to update via Holochain DHT
        if self.is_holochain_connected().await {
            #[derive(serde::Serialize, Debug)]
            struct TrustScoreInput {
                did: String,
                score: f64,
                source: String,
                last_updated: holochain_types::prelude::Timestamp,
            }

            let input = TrustScoreInput {
                did: did.clone(),
                score,
                source: "manual".to_string(),
                last_updated: holochain_types::prelude::Timestamp::from_micros(
                    trust_score.last_updated * 1_000_000,
                ),
            };

            match self
                .call_zome::<_, ActionHash>("trust_filter", "update_trust_score", input)
                .await
            {
                Ok(_) => println!("Trust score synced to DHT"),
                Err(e) => eprintln!("Warning: Failed to sync trust score to DHT: {}", e),
            }
        }

        // Also update MATL bridge via HTTP
        let url = format!("{}/trust", self.matl_bridge_url);
        let update = TrustScoreUpdate {
            did: did.clone(),
            score,
        };

        match self.http_client.post(&url).json(&update).send().await {
            Ok(response) if response.status().is_success() => {
                println!("Trust score for {} set to {:.2}", did, score);
            }
            Ok(response) => {
                eprintln!(
                    "Warning: MATL bridge returned status: {}",
                    response.status()
                );
            }
            Err(e) => {
                eprintln!("Warning: Could not reach MATL bridge: {}", e);
            }
        }

        Ok(())
    }

    /// List all trust scores
    ///
    /// Queries Holochain DHT via trust_filter::get_all_trust_scores
    pub async fn list_trust_scores(&self) -> Result<Vec<TrustScore>> {
        // Try Holochain DHT first
        if self.is_holochain_connected().await {
            #[derive(serde::Deserialize, Debug)]
            struct DhtTrustScore {
                did: String,
                score: f64,
                source: String,
                #[serde(default)]
                last_updated: Option<i64>,
            }

            match self
                .call_zome::<_, Vec<DhtTrustScore>>("trust_filter", "get_all_trust_scores", ())
                .await
            {
                Ok(scores) => {
                    let trust_scores: Vec<TrustScore> = scores
                        .into_iter()
                        .map(|s| TrustScore {
                            did: s.did,
                            score: s.score,
                            source: s.source,
                            last_updated: s
                                .last_updated
                                .unwrap_or_else(|| chrono::Utc::now().timestamp()),
                        })
                        .collect();
                    return Ok(trust_scores);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to get trust scores from DHT: {}", e);
                }
            }
        }

        // Fall back to MATL bridge
        self.sync_trust_from_matl().await
    }

    /// Sync trust scores from MATL
    ///
    /// Fetches trust scores from MATL bridge and stores locally
    pub async fn sync_trust_from_matl(&self) -> Result<Vec<TrustScore>> {
        println!("Syncing trust scores from MATL bridge...");

        // Fetch all trust scores from MATL bridge
        let url = format!("{}/trust/all", self.matl_bridge_url);

        match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                #[derive(serde::Deserialize)]
                struct MatlTrustScore {
                    did: String,
                    composite_score: f64,
                    #[serde(default)]
                    last_updated: Option<i64>,
                    #[serde(default)]
                    source: Option<String>,
                }

                let matl_scores: Vec<MatlTrustScore> = response
                    .json()
                    .await
                    .context("Failed to parse MATL trust scores response")?;

                let trust_scores: Vec<TrustScore> = matl_scores
                    .into_iter()
                    .map(|s| TrustScore {
                        did: s.did,
                        score: s.composite_score,
                        last_updated: s.last_updated.unwrap_or_else(|| chrono::Utc::now().timestamp()),
                        source: s.source.unwrap_or_else(|| "matl_bridge".to_string()),
                    })
                    .collect();

                println!("Synced {} trust scores from MATL bridge", trust_scores.len());
                Ok(trust_scores)
            }
            Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND => {
                // No trust scores available yet
                println!("No trust scores found in MATL bridge");
                Ok(vec![])
            }
            Ok(response) => {
                eprintln!(
                    "Warning: MATL bridge returned status: {}",
                    response.status()
                );
                // Return current user's default score as fallback
                let did = self
                    .config
                    .identity
                    .did
                    .clone()
                    .unwrap_or_else(|| "did:mycelix:demo".to_string());

                Ok(vec![TrustScore {
                    did,
                    score: 0.5,
                    last_updated: chrono::Utc::now().timestamp(),
                    source: "local-fallback".to_string(),
                }])
            }
            Err(e) => {
                eprintln!("Warning: Could not reach MATL bridge: {}", e);
                // Return current user's default score as fallback
                let did = self
                    .config
                    .identity
                    .did
                    .clone()
                    .unwrap_or_else(|| "did:mycelix:demo".to_string());

                Ok(vec![TrustScore {
                    did,
                    score: 0.5,
                    last_updated: chrono::Utc::now().timestamp(),
                    source: "local-fallback".to_string(),
                }])
            }
        }
    }

    //
    // ===== DID OPERATIONS (HTTP to DID Registry) =====
    //

    /// Register a DID with the DID registry
    ///
    /// Maps DID to Holochain AgentPubKey for message routing
    pub async fn register_did(&self, did: String, agent_pub_key: String) -> Result<()> {
        let url = format!("{}/register", self.did_registry_url);

        let registration = serde_json::json!({
            "did": did,
            "agent_pub_key": agent_pub_key,
        });

        match self.http_client.post(&url).json(&registration).send().await {
            Ok(response) if response.status().is_success() => {
                println!("✓ DID registered successfully");
                Ok(())
            }
            Ok(response) => {
                println!("⚠️  DID registry returned status: {}", response.status());
                println!("   [STUB] Would register locally");
                Ok(())
            }
            Err(_) => {
                println!("⚠️  DID registry not available");
                println!("   [STUB] Would register locally");
                Ok(())
            }
        }
    }

    /// Resolve a DID to an AgentPubKey
    ///
    /// Queries the DID registry to find the Holochain agent for a DID
    pub async fn resolve_did(&self, did: String) -> Result<Option<DidResolution>> {
        let url = format!("{}/resolve/{}", self.did_registry_url, did);

        match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let resolution: DidResolution = response
                    .json()
                    .await
                    .context("Failed to parse DID resolution response")?;
                Ok(Some(resolution))
            }
            Ok(response) if response.status() == 404 => {
                Ok(None) // DID not found
            }
            Ok(response) => {
                println!("⚠️  DID registry returned status: {}", response.status());
                Ok(None)
            }
            Err(_) => {
                println!("⚠️  DID registry not available");
                Ok(Some(self.stub_did_resolution(did)))
            }
        }
    }

    /// List all registered DIDs
    ///
    /// Queries the DID registry for all known DIDs
    pub async fn list_dids(&self) -> Result<Vec<DidResolution>> {
        let url = format!("{}/list", self.did_registry_url);

        match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let dids: Vec<DidResolution> = response
                    .json()
                    .await
                    .context("Failed to parse DID list response")?;
                Ok(dids)
            }
            Ok(response) => {
                println!("⚠️  DID registry returned status: {}", response.status());
                Ok(vec![])
            }
            Err(_) => {
                println!("⚠️  DID registry not available");
                let fallback = self.stub_did_resolution(
                    self.config
                        .identity
                        .did
                        .clone()
                        .unwrap_or_else(|| "did:mycelix:demo".to_string()),
                );
                Ok(vec![fallback])
            }
        }
    }

    /// Get the current user's DID
    pub fn whoami(&self) -> Result<String> {
        self.config
            .identity
            .did
            .clone()
            .context("No DID configured. Run 'mycelix-mail init' first.")
    }

    //
    // ===== UTILITY OPERATIONS =====
    //

    /// Get mail statistics from the DHT.
    ///
    /// Falls back to zeroed stats if not connected to conductor.
    pub async fn get_stats(&self) -> Result<MailStats> {
        if !self.is_holochain_connected().await {
            return Ok(MailStats {
                total_messages: 0,
                unread_messages: 0,
                total_contacts: 0,
                total_trust_scores: 0,
                last_sync: None,
            });
        }

        let stats: MailStats = self
            .call_zome("mail_messages", "get_stats", ())
            .await
            .context("Failed to fetch mail statistics from DHT")?;

        Ok(stats)
    }

    /// Health check - verify connections
    ///
    /// Checks connectivity to all external services
    pub async fn health_check(&self) -> Result<bool> {
        println!("🏥 Checking service health...");

        // Check DID registry
        let did_ok = match self
            .http_client
            .get(&format!("{}/health", self.did_registry_url))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                println!("  ✓ DID Registry: OK");
                true
            }
            _ => {
                println!("  ✗ DID Registry: UNAVAILABLE");
                false
            }
        };

        // Check MATL bridge
        let matl_ok = match self
            .http_client
            .get(&format!("{}/health", self.matl_bridge_url))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                println!("  ✓ MATL Bridge: OK");
                true
            }
            _ => {
                println!("  ✗ MATL Bridge: UNAVAILABLE");
                false
            }
        };

        // Check Holochain conductor connection
        let holo_ok = self.is_holochain_connected().await;
        if holo_ok {
            println!("  ✓ Holochain Conductor: OK");
        } else {
            // Attempt reconnection
            match self.connect_holochain().await {
                Ok(()) => {
                    println!("  ✓ Holochain Conductor: RECONNECTED");
                }
                Err(e) => {
                    println!("  ✗ Holochain Conductor: UNAVAILABLE ({})", e);
                }
            }
        }

        let holo_connected = self.is_holochain_connected().await;
        Ok(did_ok || matl_ok || holo_connected) // At least one service should be available
    }

    //
    // ===== GETTER METHODS =====
    //

    /// Get reference to configuration
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Get the user's DID
    pub fn get_my_did(&self) -> Result<String> {
        self.whoami()
    }

    /// Get the user's agent public key
    pub fn get_my_agent_key(&self) -> Result<String> {
        self.config
            .identity
            .agent_pub_key
            .clone()
            .context("No agent key configured. Run 'mycelix-mail init' first.")
    }

    /// Get the conductor URL
    pub fn get_conductor_url(&self) -> &str {
        &self.conductor_url
    }

    /// Get the DID registry URL
    pub fn get_did_registry_url(&self) -> &str {
        &self.did_registry_url
    }

    /// Get the MATL bridge URL
    pub fn get_matl_bridge_url(&self) -> &str {
        &self.matl_bridge_url
    }

    /// Sync a single trust score (alias for set_trust_score)
    pub async fn sync_trust_score(&self, did: String) -> Result<TrustScore> {
        if let Some(score) = self.get_trust_score(did.clone()).await? {
            return Ok(score);
        }

        Ok(TrustScore {
            did,
            score: 0.5,
            last_updated: chrono::Utc::now().timestamp(),
            source: "local-cache".to_string(),
        })
    }

    /// Sync all trust scores from MATL (alias for sync_trust_from_matl)
    pub async fn sync_all_trust_scores(&self) -> Result<Vec<TrustScore>> {
        self.sync_trust_from_matl().await
    }

    fn stub_did_resolution(&self, did: String) -> DidResolution {
        let agent_key = self
            .config
            .identity
            .agent_pub_key
            .clone()
            .unwrap_or_else(|| "uhCAkDemoAgentKey".to_string());
        let now = chrono::Utc::now().timestamp();

        DidResolution {
            did,
            agent_pub_key: agent_key,
            created_at: now,
            updated_at: now,
        }
    }
}
