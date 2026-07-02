// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::FetchError;
use async_trait::async_trait;
use url::Url;

#[async_trait]
pub trait Transport: Send + Sync {
    async fn fetch(&self, url: &Url) -> Result<Vec<u8>, FetchError>;
}
