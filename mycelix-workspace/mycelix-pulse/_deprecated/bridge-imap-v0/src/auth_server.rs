// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tiny HTTP server for OAuth2 redirect callback.
//! Runs on localhost:8118, receives the auth code, exchanges for tokens.

use crate::oauth::{OAuthProvider, OAuthTokens};
use crate::config::BridgeConfig;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AuthServer {
    pub config: BridgeConfig,
    pub pending_tokens: Arc<Mutex<Vec<(String, OAuthTokens)>>>, // email → tokens
}

impl AuthServer {
    pub async fn run(self) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:8118").await?;
        tracing::info!("OAuth callback server listening on http://localhost:8118");

        loop {
            let (mut stream, _) = listener.accept().await?;
            let config = self.config.clone();
            let tokens = self.pending_tokens.clone();

            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);

                // Parse GET /callback?code=XXX&state=PROVIDER
                if let Some(path) = request.lines().next() {
                    if path.contains("/callback") && path.contains("code=") {
                        let query = path.split('?').nth(1).unwrap_or("");
                        let params: std::collections::HashMap<&str, &str> = query
                            .split('&')
                            .filter_map(|p| {
                                let mut parts = p.split('=');
                                Some((parts.next()?, parts.next()?))
                            })
                            .collect();

                        let code = params.get("code").unwrap_or(&"");
                        let state = params.get("state").unwrap_or(&"gmail");

                        tracing::info!("[OAuth] Received code for provider: {state}");

                        // Determine provider
                        let provider = match *state {
                            s if s.contains("microsoft") => OAuthProvider::microsoft(
                                &config.oauth_client_id.clone().unwrap_or_default(),
                                &config.oauth_client_secret.clone().unwrap_or_default(),
                                "http://localhost:8118/callback",
                            ),
                            _ => OAuthProvider::gmail(
                                &config.oauth_client_id.clone().unwrap_or_default(),
                                &config.oauth_client_secret.clone().unwrap_or_default(),
                                "http://localhost:8118/callback",
                            ),
                        };

                        match provider.exchange_code(code).await {
                            Ok(token_data) => {
                                tracing::info!("[OAuth] Token exchange successful!");
                                let email = state.split(':').nth(1).unwrap_or("unknown@email.com");
                                tokens.lock().await.push((email.to_string(), token_data));

                                let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n\
                                    <html><body style='background:#08080c;color:#e2e6f0;font-family:Inter,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;'>\
                                    <div style='text-align:center'>\
                                    <h1 style='color:#06D6C8'>Connected!</h1>\
                                    <p>Your email account has been linked to Mycelix Pulse.</p>\
                                    <p>You can close this window.</p>\
                                    </div></body></html>";
                                let _ = stream.write_all(response.as_bytes()).await;
                            }
                            Err(e) => {
                                tracing::error!("[OAuth] Token exchange failed: {e}");
                                let response = format!(
                                    "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\n\r\n\
                                    <html><body style='background:#08080c;color:#e2e6f0;font-family:Inter,sans-serif;'>\
                                    <h1 style='color:#ef4444'>Authentication Failed</h1><p>{e}</p></body></html>"
                                );
                                let _ = stream.write_all(response.as_bytes()).await;
                            }
                        }
                    } else {
                        let response = "HTTP/1.1 404 Not Found\r\n\r\n";
                        let _ = stream.write_all(response.as_bytes()).await;
                    }
                }
            });
        }
    }
}
