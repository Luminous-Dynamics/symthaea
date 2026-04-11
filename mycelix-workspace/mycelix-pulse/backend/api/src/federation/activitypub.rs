// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ActivityPub Federation Module
//!
//! Implements ActivityPub protocol for federated email and trust networks:
//! - Actor representation (Person, Organization)
//! - Activities (Create, Update, Delete, Follow, Trust)
//! - Inbox/Outbox handling
//! - WebFinger discovery
//! - HTTP Signatures for authentication

use axum::{
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Utc};
use ring::{rand, signature};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Constants
// ============================================================================

pub const ACTIVITY_STREAMS_CONTEXT: &str = "https://www.w3.org/ns/activitystreams";
pub const SECURITY_CONTEXT: &str = "https://w3id.org/security/v1";
pub const MYCELIX_CONTEXT: &str = "https://mycelix.mail/ns/v1";

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Actor {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: String,
    #[serde(rename = "type")]
    pub actor_type: String,
    pub preferred_username: String,
    pub name: Option<String>,
    pub summary: Option<String>,
    pub inbox: String,
    pub outbox: String,
    pub followers: String,
    pub following: String,
    pub public_key: PublicKey,
    pub endpoints: Option<Endpoints>,
    pub icon: Option<MediaObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mycelix_trust_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mycelix_attestations: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicKey {
    pub id: String,
    pub owner: String,
    pub public_key_pem: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Endpoints {
    pub shared_inbox: Option<String>,
    pub oauth_authorization_endpoint: Option<String>,
    pub oauth_token_endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaObject {
    #[serde(rename = "type")]
    pub media_type: String,
    pub url: String,
    pub media_type_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Activity {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: String,
    #[serde(rename = "type")]
    pub activity_type: String,
    pub actor: String,
    pub object: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cc: Option<Vec<String>>,
    pub published: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    #[serde(rename = "type")]
    pub note_type: String,
    pub id: String,
    pub attributed_to: String,
    pub content: String,
    pub published: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_reply_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cc: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachment: Option<Vec<MediaObject>>,
}

/// Mycelix-specific trust attestation activity
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrustAttestation {
    #[serde(rename = "type")]
    pub attestation_type: String,
    pub id: String,
    pub actor: String,          // Who is attesting
    pub object: String,         // Who is being attested
    pub trust_level: f64,       // 0.0 to 1.0
    pub trust_category: String, // professional, personal, verified, etc.
    pub evidence: Option<String>,
    pub expires: Option<DateTime<Utc>>,
    pub published: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFingerResponse {
    pub subject: String,
    pub aliases: Vec<String>,
    pub links: Vec<WebFingerLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFingerLink {
    pub rel: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub link_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub href: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderedCollection {
    #[serde(rename = "@context")]
    pub context: String,
    pub id: String,
    #[serde(rename = "type")]
    pub collection_type: String,
    pub total_items: usize,
    pub first: Option<String>,
    pub last: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderedCollectionPage {
    #[serde(rename = "@context")]
    pub context: String,
    pub id: String,
    #[serde(rename = "type")]
    pub page_type: String,
    pub part_of: String,
    pub ordered_items: Vec<Value>,
    pub next: Option<String>,
    pub prev: Option<String>,
}

// ============================================================================
// State
// ============================================================================

pub struct FederationState {
    pub domain: String,
    pub private_key: ring::signature::RsaKeyPair,
    pub public_key_pem: String,
    pub actors: dashmap::DashMap<String, Actor>,
    pub inbox_queue: dashmap::DashMap<String, Vec<Activity>>,
    pub outbox: dashmap::DashMap<String, Vec<Activity>>,
    pub followers: dashmap::DashMap<String, Vec<String>>,
    pub following: dashmap::DashMap<String, Vec<String>>,
}

impl FederationState {
    pub fn new(domain: String, private_key_pem: &str) -> Self {
        // Parse RSA private key
        let private_key_der = pem::parse(private_key_pem)
            .expect("Invalid PEM")
            .contents;
        let private_key = signature::RsaKeyPair::from_pkcs8(&private_key_der)
            .expect("Invalid RSA key");

        // Extract public key PEM (simplified)
        let public_key_pem = private_key_pem
            .replace("PRIVATE", "PUBLIC")
            .lines()
            .filter(|l| !l.contains("PRIVATE"))
            .collect::<Vec<_>>()
            .join("\n");

        Self {
            domain,
            private_key,
            public_key_pem,
            actors: dashmap::DashMap::new(),
            inbox_queue: dashmap::DashMap::new(),
            outbox: dashmap::DashMap::new(),
            followers: dashmap::DashMap::new(),
            following: dashmap::DashMap::new(),
        }
    }
}

// ============================================================================
// Routes
// ============================================================================

pub fn federation_routes() -> Router<Arc<FederationState>> {
    Router::new()
        // WebFinger
        .route("/.well-known/webfinger", get(webfinger))
        // Nodeinfo
        .route("/.well-known/nodeinfo", get(nodeinfo_links))
        .route("/nodeinfo/2.0", get(nodeinfo))
        // Actor endpoints
        .route("/users/:username", get(get_actor))
        .route("/users/:username/inbox", post(inbox))
        .route("/users/:username/outbox", get(outbox).post(post_outbox))
        .route("/users/:username/followers", get(followers))
        .route("/users/:username/following", get(following))
        // Trust-specific
        .route("/trust/:id", get(get_trust_attestation))
}

// ============================================================================
// Handlers
// ============================================================================

async fn webfinger(
    State(state): State<Arc<FederationState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let resource = match params.get("resource") {
        Some(r) => r,
        None => return (StatusCode::BAD_REQUEST, "Missing resource").into_response(),
    };

    // Parse acct:user@domain format
    let account = resource.strip_prefix("acct:").unwrap_or(resource);
    let parts: Vec<&str> = account.split('@').collect();

    if parts.len() != 2 {
        return (StatusCode::BAD_REQUEST, "Invalid resource format").into_response();
    }

    let username = parts[0];
    let domain = parts[1];

    if domain != state.domain {
        return (StatusCode::NOT_FOUND, "Unknown domain").into_response();
    }

    // Check if actor exists
    if !state.actors.contains_key(username) {
        return (StatusCode::NOT_FOUND, "User not found").into_response();
    }

    let actor_url = format!("https://{}/users/{}", state.domain, username);

    let response = WebFingerResponse {
        subject: format!("acct:{}@{}", username, domain),
        aliases: vec![actor_url.clone()],
        links: vec![
            WebFingerLink {
                rel: "self".to_string(),
                link_type: Some("application/activity+json".to_string()),
                href: Some(actor_url),
                template: None,
            },
            WebFingerLink {
                rel: "http://webfinger.net/rel/profile-page".to_string(),
                link_type: Some("text/html".to_string()),
                href: Some(format!("https://{}/profile/{}", state.domain, username)),
                template: None,
            },
        ],
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/jrd+json")],
        Json(response),
    )
        .into_response()
}

async fn nodeinfo_links(State(state): State<Arc<FederationState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "links": [
            {
                "rel": "http://nodeinfo.diaspora.software/ns/schema/2.0",
                "href": format!("https://{}/nodeinfo/2.0", state.domain)
            }
        ]
    }))
}

async fn nodeinfo(State(state): State<Arc<FederationState>>) -> impl IntoResponse {
    let user_count = state.actors.len();

    Json(serde_json::json!({
        "version": "2.0",
        "software": {
            "name": "mycelix-mail",
            "version": "1.0.0"
        },
        "protocols": ["activitypub"],
        "services": {
            "inbound": ["email"],
            "outbound": ["email"]
        },
        "openRegistrations": false,
        "usage": {
            "users": {
                "total": user_count,
                "activeMonth": user_count,
                "activeHalfyear": user_count
            },
            "localPosts": 0
        },
        "metadata": {
            "features": ["trust-attestations", "post-quantum-encryption"],
            "mycelixVersion": "1.0.0"
        }
    }))
}

async fn get_actor(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // Check if requesting ActivityPub format
    if !accept.contains("application/activity+json")
        && !accept.contains("application/ld+json")
    {
        // Could redirect to profile page for HTML requests
    }

    match state.actors.get(&username) {
        Some(actor) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/activity+json")],
            Json(actor.clone()),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "Actor not found").into_response(),
    }
}

async fn inbox(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
    headers: HeaderMap,
    Json(activity): Json<Activity>,
) -> impl IntoResponse {
    // Verify HTTP signature
    if let Err(e) = verify_http_signature(&headers, &activity.actor).await {
        return (StatusCode::UNAUTHORIZED, format!("Invalid signature: {}", e)).into_response();
    }

    // Process activity
    match activity.activity_type.as_str() {
        "Follow" => {
            // Add to followers
            state
                .followers
                .entry(username.clone())
                .or_default()
                .push(activity.actor.clone());

            // Send Accept activity back
            let accept = create_accept_activity(&state, &username, &activity);
            deliver_activity(&accept, &activity.actor).await.ok();
        }

        "Undo" => {
            // Handle unfollow, etc.
            if let Some(obj) = activity.object.as_object() {
                if obj.get("type").and_then(|t| t.as_str()) == Some("Follow") {
                    state.followers.entry(username.clone()).and_modify(|f| {
                        f.retain(|a| a != &activity.actor);
                    });
                }
            }
        }

        "Create" | "Update" | "Delete" => {
            // Queue for processing
            state
                .inbox_queue
                .entry(username.clone())
                .or_default()
                .push(activity);
        }

        "TrustAttestation" => {
            // Handle Mycelix trust attestation
            if let Ok(attestation) = serde_json::from_value::<TrustAttestation>(activity.object) {
                process_trust_attestation(&state, &username, attestation).await;
            }
        }

        _ => {
            // Unknown activity type
        }
    }

    StatusCode::ACCEPTED.into_response()
}

async fn outbox(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
) -> impl IntoResponse {
    let activities = state
        .outbox
        .get(&username)
        .map(|a| a.clone())
        .unwrap_or_default();

    let collection = OrderedCollection {
        context: ACTIVITY_STREAMS_CONTEXT.to_string(),
        id: format!("https://{}/users/{}/outbox", state.domain, username),
        collection_type: "OrderedCollection".to_string(),
        total_items: activities.len(),
        first: Some(format!(
            "https://{}/users/{}/outbox?page=1",
            state.domain, username
        )),
        last: None,
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/activity+json")],
        Json(collection),
    )
}

async fn post_outbox(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
    headers: HeaderMap,
    Json(mut activity): Json<Activity>,
) -> impl IntoResponse {
    // Verify authorization (would check auth token)

    // Generate ID if not present
    if activity.id.is_empty() {
        activity.id = format!(
            "https://{}/activities/{}",
            state.domain,
            Uuid::new_v4()
        );
    }

    // Set published time
    activity.published = Utc::now();

    // Store in outbox
    state
        .outbox
        .entry(username.clone())
        .or_default()
        .push(activity.clone());

    // Deliver to recipients
    let recipients = collect_recipients(&activity);
    for recipient in recipients {
        deliver_activity(&activity, &recipient).await.ok();
    }

    (
        StatusCode::CREATED,
        [(header::LOCATION, activity.id.as_str())],
    )
        .into_response()
}

async fn followers(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
) -> impl IntoResponse {
    let follower_list = state
        .followers
        .get(&username)
        .map(|f| f.clone())
        .unwrap_or_default();

    let collection = OrderedCollection {
        context: ACTIVITY_STREAMS_CONTEXT.to_string(),
        id: format!("https://{}/users/{}/followers", state.domain, username),
        collection_type: "OrderedCollection".to_string(),
        total_items: follower_list.len(),
        first: None,
        last: None,
    };

    Json(collection)
}

async fn following(
    State(state): State<Arc<FederationState>>,
    Path(username): Path<String>,
) -> impl IntoResponse {
    let following_list = state
        .following
        .get(&username)
        .map(|f| f.clone())
        .unwrap_or_default();

    let collection = OrderedCollection {
        context: ACTIVITY_STREAMS_CONTEXT.to_string(),
        id: format!("https://{}/users/{}/following", state.domain, username),
        collection_type: "OrderedCollection".to_string(),
        total_items: following_list.len(),
        first: None,
        last: None,
    };

    Json(collection)
}

async fn get_trust_attestation(
    State(state): State<Arc<FederationState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Would fetch from database
    (StatusCode::NOT_FOUND, "Attestation not found").into_response()
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_accept_activity(state: &FederationState, username: &str, follow: &Activity) -> Activity {
    Activity {
        context: vec![ACTIVITY_STREAMS_CONTEXT.to_string()],
        id: format!("https://{}/activities/{}", state.domain, Uuid::new_v4()),
        activity_type: "Accept".to_string(),
        actor: format!("https://{}/users/{}", state.domain, username),
        object: serde_json::to_value(follow).unwrap(),
        target: None,
        to: Some(vec![follow.actor.clone()]),
        cc: None,
        published: Utc::now(),
    }
}

fn collect_recipients(activity: &Activity) -> Vec<String> {
    let mut recipients = Vec::new();

    if let Some(to) = &activity.to {
        recipients.extend(to.iter().cloned());
    }
    if let Some(cc) = &activity.cc {
        recipients.extend(cc.iter().cloned());
    }

    // Filter out public addressing
    recipients.retain(|r| !r.contains("Public"));

    recipients
}

async fn deliver_activity(activity: &Activity, recipient: &str) -> Result<(), String> {
    // Fetch recipient's inbox
    let client = reqwest::Client::new();

    // Would need to:
    // 1. Fetch actor to get inbox URL
    // 2. Sign the request
    // 3. POST to inbox

    let inbox_url = format!("{}/inbox", recipient.trim_end_matches("/inbox"));

    let response = client
        .post(&inbox_url)
        .header("Content-Type", "application/activity+json")
        .json(activity)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("Delivery failed: {}", response.status()))
    }
}

async fn verify_http_signature(headers: &HeaderMap, actor_id: &str) -> Result<(), String> {
    // Parse Signature header
    let sig_header = headers
        .get("Signature")
        .ok_or("Missing Signature header")?
        .to_str()
        .map_err(|_| "Invalid Signature header")?;

    // In production:
    // 1. Parse signature header components
    // 2. Fetch actor's public key
    // 3. Verify signature against request

    Ok(())
}

async fn process_trust_attestation(
    state: &FederationState,
    username: &str,
    attestation: TrustAttestation,
) {
    // Process and store trust attestation
    // Would interact with Holochain DNA
}

/// Sign an HTTP request for ActivityPub delivery
pub fn sign_request(
    state: &FederationState,
    method: &str,
    path: &str,
    host: &str,
    date: &str,
    digest: Option<&str>,
    key_id: &str,
) -> String {
    let mut signed_string = format!(
        "(request-target): {} {}\nhost: {}\ndate: {}",
        method.to_lowercase(),
        path,
        host,
        date
    );

    let mut headers = "(request-target) host date";

    if let Some(d) = digest {
        signed_string.push_str(&format!("\ndigest: {}", d));
        headers = "(request-target) host date digest";
    }

    // Sign with RSA-SHA256
    let rng = rand::SystemRandom::new();
    let mut signature = vec![0u8; state.private_key.public_modulus_len()];

    state
        .private_key
        .sign(
            &signature::RSA_PKCS1_SHA256,
            &rng,
            signed_string.as_bytes(),
            &mut signature,
        )
        .expect("Signing failed");

    let sig_b64 = BASE64.encode(&signature);

    format!(
        r#"keyId="{}",algorithm="rsa-sha256",headers="{}",signature="{}""#,
        key_id, headers, sig_b64
    )
}

/// Calculate digest for request body
pub fn calculate_digest(body: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body);
    let result = hasher.finalize();
    format!("SHA-256={}", BASE64.encode(result))
}

// ============================================================================
// Actor Creation
// ============================================================================

pub fn create_actor(state: &FederationState, username: &str, name: Option<&str>) -> Actor {
    let base_url = format!("https://{}/users/{}", state.domain, username);

    Actor {
        context: vec![
            ACTIVITY_STREAMS_CONTEXT.to_string(),
            SECURITY_CONTEXT.to_string(),
            MYCELIX_CONTEXT.to_string(),
        ],
        id: base_url.clone(),
        actor_type: "Person".to_string(),
        preferred_username: username.to_string(),
        name: name.map(String::from),
        summary: None,
        inbox: format!("{}/inbox", base_url),
        outbox: format!("{}/outbox", base_url),
        followers: format!("{}/followers", base_url),
        following: format!("{}/following", base_url),
        public_key: PublicKey {
            id: format!("{}#main-key", base_url),
            owner: base_url,
            public_key_pem: state.public_key_pem.clone(),
        },
        endpoints: Some(Endpoints {
            shared_inbox: Some(format!("https://{}/inbox", state.domain)),
            oauth_authorization_endpoint: Some(format!("https://{}/auth/authorize", state.domain)),
            oauth_token_endpoint: Some(format!("https://{}/auth/token", state.domain)),
        }),
        icon: None,
        mycelix_trust_score: None,
        mycelix_attestations: None,
    }
}
