//! nix-mind-daemon: background daemon for continuous NixOS awareness.
//!
//! Periodically observes the system state, encodes it into the HDC world model,
//! and updates the causal graph with any state transitions. High-surprise events
//! (large prediction errors) are stored in episodic memory for future reference.
//!
//! Designed to run as a systemd service via the NixOS module.

use std::io;
use std::thread;
use std::time::{Duration, Instant};

use symthaea_nix::observe::SystemObserver;
use symthaea_nix::encoding::{SystemStateEncoder, NixCodebook};
use symthaea_nix::mind::NixWorldModel;

/// Daemon configuration.
struct DaemonConfig {
    /// How often to take a full system snapshot (seconds).
    snapshot_interval: u64,
    /// How often to check for high-frequency events like journal entries (seconds).
    poll_interval: u64,
    /// Prediction error threshold for episodic storage.
    surprise_threshold: f64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: 60,
            poll_interval: 5,
            surprise_threshold: 0.3,
        }
    }
}

fn main() -> io::Result<()> {
    let config = DaemonConfig::default();
    let mut codebook = NixCodebook::new();
    let mut world_model = NixWorldModel::default_dim();

    eprintln!("nix-mind-daemon: starting continuous awareness");
    eprintln!(
        "  snapshot every {}s, poll every {}s, surprise threshold {:.2}",
        config.snapshot_interval, config.poll_interval, config.surprise_threshold
    );

    let mut last_snapshot = Instant::now() - Duration::from_secs(config.snapshot_interval);

    loop {
        // Full system snapshot at configured interval
        if last_snapshot.elapsed() >= Duration::from_secs(config.snapshot_interval) {
            match SystemObserver::snapshot() {
                Ok(snapshot) => {
                    let encoder = SystemStateEncoder::new(&mut codebook);
                    let state_hv = encoder.encode_snapshot(&snapshot);
                    world_model.observe(state_hv);
                    let free_energy = world_model.free_energy();

                    if free_energy > config.surprise_threshold {
                        eprintln!(
                            "nix-mind-daemon: surprise detected (FE={:.3}), updating model",
                            free_energy
                        );
                    }
                }
                Err(e) => {
                    eprintln!("nix-mind-daemon: snapshot failed: {}", e);
                }
            }
            last_snapshot = Instant::now();
        }

        thread::sleep(Duration::from_secs(config.poll_interval));
    }
}
