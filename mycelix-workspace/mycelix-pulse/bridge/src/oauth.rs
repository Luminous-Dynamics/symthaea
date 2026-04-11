// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! OAuth2 token management for email providers.
//!
//! Handles:
//! - Authorization URL generation
//! - Code → token exchange
//! - Token refresh
//! - XOAUTH2 SASL string generation for IMAP

use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OAuthProvider {
    pub name: String,
    pub auth_url: String,
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OAuthTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: u64,
    pub token_type: String,
}

impl OAuthProvider {
    /// Gmail OAuth2 configuration.
    pub fn gmail(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            name: "Gmail".into(),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            scopes: vec![
                "https://mail.google.com/".into(), // Full IMAP + SMTP
            ],
            redirect_uri: redirect_uri.into(),
        }
    }

    /// Microsoft (Outlook/Hotmail/Live) OAuth2 configuration.
    pub fn microsoft(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            name: "Microsoft".into(),
            auth_url: "https://login.microsoftonline.com/common/oauth2/v2/authorize".into(),
            token_url: "https://login.microsoftonline.com/common/oauth2/v2/token".into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            scopes: vec![
                "https://outlook.office365.com/IMAP.AccessAsUser.All".into(),
                "https://outlook.office365.com/SMTP.Send".into(),
                "offline_access".into(),
            ],
            redirect_uri: redirect_uri.into(),
        }
    }

    /// Yahoo OAuth2 configuration.
    pub fn yahoo(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            name: "Yahoo".into(),
            auth_url: "https://api.login.yahoo.com/oauth2/request_auth".into(),
            token_url: "https://api.login.yahoo.com/oauth2/get_token".into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            scopes: vec!["mail-r".into(), "mail-w".into()],
            redirect_uri: redirect_uri.into(),
        }
    }

    /// Generate the authorization URL that the user opens in their browser.
    pub fn auth_url(&self, state: &str) -> String {
        format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent&state={}",
            self.auth_url,
            urlencoding::encode(&self.client_id),
            urlencoding::encode(&self.redirect_uri),
            urlencoding::encode(&self.scopes.join(" ")),
            urlencoding::encode(state),
        )
    }

    /// Exchange authorization code for tokens.
    pub async fn exchange_code(&self, code: &str) -> Result<OAuthTokens> {
        let client = reqwest::Client::new();
        let response = client.post(&self.token_url)
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("code", code),
                ("redirect_uri", self.redirect_uri.as_str()),
                ("grant_type", "authorization_code"),
            ])
            .send()
            .await?;

        let body: serde_json::Value = response.json().await?;

        if let Some(error) = body.get("error") {
            return Err(anyhow::anyhow!("OAuth error: {} - {}",
                error.as_str().unwrap_or("unknown"),
                body.get("error_description").and_then(|d| d.as_str()).unwrap_or("")
            ));
        }

        let expires_in = body["expires_in"].as_u64().unwrap_or(3600);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(OAuthTokens {
            access_token: body["access_token"].as_str().unwrap_or("").into(),
            refresh_token: body["refresh_token"].as_str().map(String::from),
            expires_at: now + expires_in,
            token_type: body["token_type"].as_str().unwrap_or("Bearer").into(),
        })
    }

    /// Refresh an expired access token.
    pub async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthTokens> {
        let client = reqwest::Client::new();
        let response = client.post(&self.token_url)
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await?;

        let body: serde_json::Value = response.json().await?;

        let expires_in = body["expires_in"].as_u64().unwrap_or(3600);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(OAuthTokens {
            access_token: body["access_token"].as_str().unwrap_or("").into(),
            refresh_token: body["refresh_token"].as_str().map(String::from)
                .or_else(|| Some(refresh_token.to_string())),
            expires_at: now + expires_in,
            token_type: body["token_type"].as_str().unwrap_or("Bearer").into(),
        })
    }
}

impl OAuthTokens {
    /// Check if the access token has expired.
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now >= self.expires_at.saturating_sub(60) // 60s buffer
    }

    /// Generate XOAUTH2 SASL string for IMAP authentication.
    /// Format: base64("user=" + email + "\x01auth=Bearer " + token + "\x01\x01")
    pub fn xoauth2_string(&self, email: &str) -> String {
        let sasl = format!("user={}\x01auth=Bearer {}\x01\x01", email, self.access_token);
        base64_encode(sasl.as_bytes())
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// URL encoding helper.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars().map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            _ => format!("%{:02X}", c as u8),
        }).collect()
    }
}
