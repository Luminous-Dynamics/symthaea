// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Governance tally poller — polls the Holochain conductor for completed
//! proposal tallies and emits `GovernanceOutcomeEvent` back to Symthaea.
//!
//! This is the return path of the governance bridge: while `dispatch_loop`
//! sends commands *to* Holochain, the poller reads results *from* Holochain
//! and feeds them back through the outcome channel.

use std::collections::HashSet;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use crate::conductor::HolochainConductor;
use crate::dispatch_loop::GovernanceOutcomeEvent;
use crate::dispatcher::GovernanceDispatcher;

/// Polls the Holochain governance zome for completed tallies.
///
/// Maintains a set of already-reported proposal IDs to avoid duplicate
/// outcome events. Polls at a configurable interval.
pub struct GovernancePoller {
    gov: GovernanceDispatcher,
    conductor: Arc<HolochainConductor>,
    /// Proposal IDs already reported as completed (avoids duplicates).
    reported: HashSet<String>,
}

impl GovernancePoller {
    /// Create a new poller backed by the given conductor.
    pub fn new(conductor: Arc<HolochainConductor>) -> Self {
        Self {
            gov: GovernanceDispatcher::new(conductor.clone()),
            conductor,
            reported: HashSet::new(),
        }
    }

    /// Poll for currently active proposals.
    ///
    /// Returns the raw JSON values from the `get_active_proposals` zome call.
    pub async fn poll_active_proposals(&self) -> crate::Result<Vec<serde_json::Value>> {
        self.gov.get_active_proposals().await
    }

    /// Check active proposals for completed tallies and emit outcome events.
    ///
    /// For each proposal with a `"status"` of `"passed"` or `"failed"`,
    /// emits a `ProposalPassed` or `ProposalFailed` event (once per proposal).
    /// Returns the number of new outcomes found.
    pub async fn poll_tally_results(
        &mut self,
        outcome_tx: &Sender<GovernanceOutcomeEvent>,
    ) -> crate::Result<usize> {
        let proposals = self.gov.get_active_proposals().await?;
        let mut new_outcomes = 0;

        for proposal in &proposals {
            let id = match proposal.get("proposal_id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None => continue,
            };

            // Skip already-reported proposals
            if self.reported.contains(&id) {
                continue;
            }

            let status = match proposal.get("status").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => continue,
            };

            let tally_yes = proposal
                .get("tally_yes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let tally_no = proposal
                .get("tally_no")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let event = match status {
                "passed" => Some(GovernanceOutcomeEvent::ProposalPassed {
                    proposal_id: id.clone(),
                    tally_yes,
                    tally_no,
                }),
                "failed" => Some(GovernanceOutcomeEvent::ProposalFailed {
                    proposal_id: id.clone(),
                    tally_yes,
                    tally_no,
                }),
                _ => None,
            };

            if let Some(evt) = event {
                if outcome_tx.send(evt).is_ok() {
                    self.reported.insert(id);
                    new_outcomes += 1;
                } else {
                    tracing::warn!("Governance outcome channel closed — stopping poll");
                    break;
                }
            }
        }

        Ok(new_outcomes)
    }

    /// Run the poll loop indefinitely at the given interval.
    ///
    /// Ensures the conductor is connected before each poll. Logs errors
    /// but does not stop on transient failures.
    pub async fn run_poll_loop(
        mut self,
        outcome_tx: Sender<GovernanceOutcomeEvent>,
        interval: Duration,
    ) {
        tracing::info!(
            interval_secs = interval.as_secs(),
            "Starting governance tally poll loop"
        );

        loop {
            tokio::time::sleep(interval).await;

            // Ensure connected
            if !self.conductor.is_connected() {
                if let Err(e) = self.conductor.connect().await {
                    tracing::warn!(error = %e, "Conductor reconnect failed in poller — skipping poll");
                    continue;
                }
            }

            match self.poll_tally_results(&outcome_tx).await {
                Ok(n) if n > 0 => {
                    tracing::info!(new_outcomes = n, "Governance poller found completed tallies");
                }
                Ok(_) => {
                    tracing::trace!("Governance poller: no new tallies");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Governance poller failed to fetch tallies");
                }
            }
        }
    }

    /// Clear the set of reported proposals (useful for testing or reset).
    pub fn clear_reported(&mut self) {
        self.reported.clear();
    }

    /// Number of proposals already reported.
    pub fn reported_count(&self) -> usize {
        self.reported.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poller_creation() {
        let conductor = Arc::new(HolochainConductor::new(
            "127.0.0.1:4444",
            "127.0.0.1:4445",
            "mycelix-unified",
        ));
        let poller = GovernancePoller::new(conductor);
        assert_eq!(poller.reported_count(), 0);
    }

    #[test]
    fn test_reported_dedup() {
        let conductor = Arc::new(HolochainConductor::new(
            "127.0.0.1:4444",
            "127.0.0.1:4445",
            "mycelix-unified",
        ));
        let mut poller = GovernancePoller::new(conductor);
        poller.reported.insert("prop-1".to_string());
        assert_eq!(poller.reported_count(), 1);
        assert!(poller.reported.contains("prop-1"));

        poller.clear_reported();
        assert_eq!(poller.reported_count(), 0);
    }
}
