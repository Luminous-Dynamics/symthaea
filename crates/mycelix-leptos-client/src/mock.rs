// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock transport for demo/development mode.
//!
//! All zome calls return an error indicating mock mode. This allows apps
//! to compile and run without a conductor, falling back to mock data.

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::{ConnectConfig, ConnectionStatus};
use std::future::Future;
use std::pin::Pin;

/// A transport that always returns errors, enabling demo/offline mode.
///
/// Use this when no conductor is available — app components should
/// handle the error by falling back to mock data.
#[derive(Clone, Debug)]
pub struct MockTransport;

impl MockTransport {
    pub fn new() -> Self {
        Self
    }
}

impl HolochainTransport for MockTransport {
    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        _payload: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>> {
        let _info = format!("{}.{}.{}", role_name, zome_name, fn_name);
        Box::pin(async move { Err(ClientError::NotConnected) })
    }

    fn status(&self) -> ConnectionStatus {
        ConnectionStatus::Disconnected
    }

    fn connect(
        &self,
        _config: ConnectConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>> {
        Box::pin(async { Ok(()) })
    }

    fn disconnect(&self) {}
}
