// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! World bridge: threaded simulation with channel-based communication.

use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};

use mycelix_multiworld_sim::{
    config::PolicyConfig, factions::FactionEngine, stochastic::StochasticEngine, world::World,
};

use crate::scale::WorldScale;
use crate::snapshot::{
    EconomySummary, FactionSummary, GovernanceSummary, PlayerAction, SimEvent, SimSnapshot,
};
use crate::time_control::TimeControl;

/// The world bridge connects the simulation thread to the game thread.
pub struct WorldBridge {
    /// Receive snapshots from the sim thread.
    snapshot_rx: Receiver<SimSnapshot>,
    /// Send player actions to the sim thread.
    action_tx: Sender<PlayerAction>,
    /// Most recent snapshot (for game thread to read).
    pub current: SimSnapshot,
    /// Previous snapshot (for interpolation).
    pub previous: SimSnapshot,
    /// Time control (owned by game thread).
    pub time: TimeControl,
    /// Current world scale.
    pub scale: WorldScale,
    /// Handle to the sim thread.
    _sim_thread: Option<thread::JoinHandle<()>>,
    /// Signal to stop the sim thread.
    stop: Arc<Mutex<bool>>,
}

impl WorldBridge {
    /// Create a new world bridge and spawn the simulation thread.
    ///
    /// `world` is the initial world state. `seed` controls the RNG.
    pub fn new(world: World, seed: u64) -> Self {
        Self::with_capacity(world, seed, 4)
    }

    /// Create with explicit channel capacity.
    pub fn with_capacity(world: World, seed: u64, channel_capacity: usize) -> Self {
        let (snapshot_tx, snapshot_rx) = bounded(channel_capacity);
        let (action_tx, action_rx) = bounded(64);
        let stop = Arc::new(Mutex::new(false));
        let stop_clone = stop.clone();

        let sim_thread = thread::spawn(move || {
            run_sim_loop(world, seed, snapshot_tx, action_rx, stop_clone);
        });

        Self {
            snapshot_rx,
            action_tx,
            current: SimSnapshot::initial(),
            previous: SimSnapshot::initial(),
            time: TimeControl::new(),
            scale: WorldScale::Micro,
            _sim_thread: Some(sim_thread),
            stop,
        }
    }

    /// Create a bridge without spawning a thread (for testing).
    pub fn new_offline() -> Self {
        let (_snapshot_tx, snapshot_rx) = bounded(1);
        let (action_tx, _action_rx) = bounded(1);

        Self {
            snapshot_rx,
            action_tx,
            current: SimSnapshot::initial(),
            previous: SimSnapshot::initial(),
            time: TimeControl::new(),
            scale: WorldScale::Micro,
            _sim_thread: None,
            stop: Arc::new(Mutex::new(false)),
        }
    }

    /// Poll for new snapshots from the sim thread. Call once per game frame.
    pub fn poll(&mut self) {
        loop {
            match self.snapshot_rx.try_recv() {
                Ok(snapshot) => {
                    self.previous = std::mem::replace(&mut self.current, snapshot);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Send a player action to the sim thread.
    pub fn send_action(&self, action: PlayerAction) -> bool {
        self.action_tx.try_send(action).is_ok()
    }

    /// Get the interpolated snapshot for the current frame.
    pub fn interpolated(&self) -> SimSnapshot {
        let t = self.time.interpolation_factor();
        crate::snapshot::interpolate_snapshots(&self.previous, &self.current, t)
    }

    /// Current governance stability.
    pub fn stability(&self) -> f64 {
        self.current.governance.stability
    }

    /// Current Gini coefficient.
    pub fn gini(&self) -> f64 {
        self.current.economy.gini
    }

    /// Current oppression index.
    pub fn oppression(&self) -> f64 {
        self.current.governance.oppression_index
    }

    /// Events from the current snapshot.
    pub fn events(&self) -> &[SimEvent] {
        &self.current.events
    }

    /// Signal the sim thread to stop.
    pub fn shutdown(&self) {
        if let Ok(mut stop) = self.stop.lock() {
            *stop = true;
        }
    }

    /// Whether the sim thread is still running.
    pub fn is_running(&self) -> bool {
        self._sim_thread
            .as_ref()
            .map(|t| !t.is_finished())
            .unwrap_or(false)
    }
}

impl Drop for WorldBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The simulation loop on the background thread.
fn run_sim_loop(
    world: World,
    seed: u64,
    snapshot_tx: Sender<SimSnapshot>,
    action_rx: Receiver<PlayerAction>,
    stop: Arc<Mutex<bool>>,
) {
    let mut governance = world.governance.clone();
    let economy = world.economy.clone();
    let factions = FactionEngine::new();
    let mut rng = StochasticEngine::new(seed);
    let mut _policy = PolicyConfig::default();
    let mut tick: u64 = 0;

    loop {
        if stop.lock().map(|s| *s).unwrap_or(true) {
            break;
        }

        // Process player actions
        while let Ok(action) = action_rx.try_recv() {
            match action {
                PlayerAction::SetPolicy { key, value } => {
                    apply_policy(&mut _policy, &key, value);
                }
                _ => {} // Other actions handled game-side or logged
            }
        }

        // Run one simulation tick
        tick += 1;
        let gov_events = governance.tick_governance(&world, tick as u32, &mut rng);

        let mut events: Vec<SimEvent> = Vec::new();
        for _event in &gov_events {
            events.push(SimEvent::Log(format!("Governance event at tick {tick}")));
        }

        let snapshot = SimSnapshot {
            tick,
            governance: GovernanceSummary {
                stability: governance.stability_score,
                oppression_index: governance.oppression_index,
                emergency_active: governance.emergency_powers_active,
                veto_count: governance.veto_count,
                override_count: governance.veto_override_count,
                authority_level: format!("{:?}", governance.authority_level),
            },
            economy: EconomySummary {
                gini: economy.gini_coefficient,
                total_production: economy.total_production,
                currency_supply: economy.currency_supply,
            },
            factions: factions
                .factions
                .iter()
                .map(|f| FactionSummary {
                    name: f.name.clone(),
                    ideology: f.ideology,
                    population_fraction: f.member_count as f64 / 100.0, // approximate
                    in_conflict: false,
                })
                .collect(),
            events,
        };

        let _ = snapshot_tx.try_send(snapshot);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn apply_policy(policy: &mut PolicyConfig, key: &str, value: f64) {
    match key {
        "care_effectiveness" => policy.care_effectiveness = value,
        "pair_bond_rate" => policy.pair_bond_rate = value,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_bridge_works() {
        let bridge = WorldBridge::new_offline();
        assert_eq!(bridge.current.tick, 0);
        assert_eq!(bridge.stability(), 0.0);
    }

    #[test]
    fn poll_empty_channel_no_crash() {
        let mut bridge = WorldBridge::new_offline();
        bridge.poll();
        assert_eq!(bridge.current.tick, 0);
    }

    #[test]
    fn convenience_accessors() {
        let mut bridge = WorldBridge::new_offline();
        bridge.current.governance.stability = 0.75;
        bridge.current.governance.oppression_index = 0.1;
        bridge.current.economy.gini = 0.35;

        assert!((bridge.stability() - 0.75).abs() < 1e-10);
        assert!((bridge.oppression() - 0.1).abs() < 1e-10);
        assert!((bridge.gini() - 0.35).abs() < 1e-10);
    }

    #[test]
    fn interpolated_returns_current_when_no_progress() {
        let bridge = WorldBridge::new_offline();
        let snap = bridge.interpolated();
        assert_eq!(snap.tick, 0);
    }
}
