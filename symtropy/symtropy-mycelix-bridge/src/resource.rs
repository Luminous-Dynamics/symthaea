// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use flume::{Receiver, Sender, TrySendError};

use crate::events::{MycelixRequest, MycelixResponse};

/// Bevy `Resource` holding the sending end of the request channel.
///
/// Bevy systems obtain this via `Res<MycelixClient>` and call [`send`] to
/// dispatch zome calls to the tokio background task.
///
/// [`send`]: MycelixClient::send
#[derive(Resource, Clone)]
pub struct MycelixClient {
    tx: Sender<MycelixRequest>,
}

impl MycelixClient {
    pub(crate) fn new(tx: Sender<MycelixRequest>) -> Self {
        Self { tx }
    }

    /// Enqueue a zome call. Returns [`MycelixSendError::Full`] if the request
    /// channel has reached its inflight budget — this is non-blocking, so the
    /// Bevy schedule never stalls on a saturated conductor.
    pub fn send(&self, request: MycelixRequest) -> Result<(), MycelixSendError> {
        match self.tx.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(MycelixSendError::Full),
            Err(TrySendError::Disconnected(_)) => Err(MycelixSendError::Disconnected),
        }
    }
}

/// Reasons [`MycelixClient::send`] can fail.
#[derive(Debug, thiserror::Error)]
pub enum MycelixSendError {
    #[error("request channel is full (inflight budget reached)")]
    Full,
    #[error("request channel is disconnected — background task exited")]
    Disconnected,
}

/// Internal resource holding the receiving end of the response channel.
///
/// Pumped each frame by [`pump_responses`] into the Bevy event stream.
///
/// [`pump_responses`]: crate::systems::pump_responses
#[derive(Resource)]
pub(crate) struct MycelixResponseInbox {
    pub(crate) rx: Receiver<MycelixResponse>,
}

/// Internal resource holding the receiving end of the request channel so the
/// tokio background task can claim it at startup.
#[derive(Resource)]
pub(crate) struct MycelixRequestOutbox {
    pub(crate) rx: Option<Receiver<MycelixRequest>>,
    pub(crate) response_tx: Option<Sender<MycelixResponse>>,
}
