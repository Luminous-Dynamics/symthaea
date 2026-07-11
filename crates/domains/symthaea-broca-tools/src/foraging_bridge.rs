// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Foraging Bridge — Global Substrate Research Loop.
//!
//! Bridges Broca's curiosity to the SearXNG cluster, allowing her to resolve
//! internal logical holes by foraging external empirical data.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "mamba-cpu")]
use ureq;

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct SearxResult {
    pub title: String,
    pub content: String,
    pub url: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
struct SearxResponse {
    pub results: Vec<SearxResult>,
}

#[derive(Clone)]
pub struct ForagingBridge {
    pub endpoint: String,
    pub timeout: Duration,
}

impl ForagingBridge {
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            timeout: Duration::from_secs(7), // Optimized for her local cluster
        }
    }

    /// Forage the global substrate for information to resolve a curiosity sector.
    pub fn forage(&self, query: &str) -> Result<String> {
        #[cfg(feature = "mamba-cpu")]
        {
            println!("🌐 Foraging Global Substrate for: '{}'...", query);

            let url = format!(
                "{}/search?q={}&format=json",
                self.endpoint,
                urlencoding::encode(query)
            );

            let response: SearxResponse =
                ureq::get(&url).timeout(self.timeout).call()?.into_json()?;

            // Aggregate top results into a "Semantic Forage" block
            let mut aggregated = String::new();
            for res in response.results.iter().take(3) {
                aggregated.push_str(&format!(
                    "Source: {}\nContent: {}\n\n",
                    res.url, res.content
                ));
            }

            if aggregated.is_empty() {
                return Err(anyhow::anyhow!(
                    "Foraging mission yielded zero empirical hits."
                ));
            }

            println!(
                "✅ Foraging SUCCESS. Retrieved {} characters of knowledge.",
                aggregated.len()
            );
            Ok(aggregated)
        }
        #[cfg(not(feature = "mamba-cpu"))]
        {
            let _ = query;
            Err(anyhow::anyhow!(
                "Foraging requires mamba-cpu feature (ureq)."
            ))
        }
    }
}
