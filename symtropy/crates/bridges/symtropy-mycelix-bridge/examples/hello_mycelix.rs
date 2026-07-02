// SPDX-License-Identifier: AGPL-3.0-or-later

//! Milestone 1 smoke test — send one `GetActiveProposals` request from a
//! Bevy system and log the response.
//!
//! # Running
//!
//! 1. Start the shared Holochain conductor (`hc sandbox ...` or the
//!    mycelix-praxis flake). Ensure it's listening on `ws://localhost:8888`
//!    with the `mycelix-unified` hApp installed.
//!
//! 2. Export an app-auth token in the environment:
//!    ```bash
//!    export MYCELIX_APP_TOKEN="<base64-token-from-admin-interface>"
//!    ```
//!    (Obtain via `mycelix-conductor-bridge` or `hc admin issue-app-auth-token`.)
//!
//! 3. `cargo run --example hello_mycelix` (from the `symtropy-mycelix-bridge/`
//!    directory).
//!
//! Successful output: an `ActiveProposals` response with N proposals, followed
//! by graceful exit.
//!
//! If the conductor is unreachable, you'll see an `Error` response with the
//! underlying connection error — this is still a successful M1 result (the
//! plumbing works; the conductor just isn't up).

use bevy::prelude::*;
use symtropy_mycelix_bridge::{
    BevyMycelixPlugin, MycelixClient, MycelixConfig, MycelixRequest, MycelixResponse,
};

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins(bevy::log::LogPlugin::default())
        .add_plugins(BevyMycelixPlugin::new(MycelixConfig::default()))
        .insert_resource(RequestSent(false))
        .add_systems(Update, (send_once, log_and_exit))
        .run();
}

/// Track whether we've already fired our one request.
#[derive(Resource)]
struct RequestSent(bool);

fn send_once(client: Res<MycelixClient>, mut sent: ResMut<RequestSent>, time: Res<Time>) {
    // Wait one second so the dispatcher task has a chance to connect.
    if sent.0 || time.elapsed_secs() < 1.0 {
        return;
    }
    match client.send(MycelixRequest::GetActiveProposals {
        requester: Entity::PLACEHOLDER,
    }) {
        Ok(()) => info!("sent GetActiveProposals"),
        Err(err) => error!(?err, "failed to send request"),
    }
    sent.0 = true;
}

fn log_and_exit(mut reader: MessageReader<MycelixResponse>, mut exit: MessageWriter<AppExit>) {
    for response in reader.read() {
        match response {
            MycelixResponse::ActiveProposals {
                requester,
                proposals,
            } => {
                info!(
                    ?requester,
                    count = proposals.len(),
                    "received ActiveProposals"
                );
            }
            MycelixResponse::Error { requester, reason } => {
                warn!(?requester, %reason, "received Error (plumbing OK, conductor state unclear)");
            }
            // M2 variants — not expected in this M1 smoke test, but covered
            // for exhaustiveness.
            MycelixResponse::ProposalSubmitted {
                requester,
                action_hash,
            } => {
                info!(?requester, %action_hash, "received ProposalSubmitted (unexpected for M1 smoke)");
            }
            MycelixResponse::VoteCast {
                requester,
                proposal_id,
            } => {
                info!(?requester, %proposal_id, "received VoteCast (unexpected for M1 smoke)");
            }
            MycelixResponse::TendBalance { requester, balance } => {
                info!(
                    ?requester,
                    ?balance,
                    "received TendBalance (unexpected for M1 smoke)"
                );
            }
            MycelixResponse::Proposal {
                requester,
                proposal_id,
                record,
            } => {
                info!(?requester, %proposal_id, found = record.is_some(), "received Proposal (unexpected for M1 smoke)");
            }
        }
        exit.write(AppExit::Success);
    }
}
