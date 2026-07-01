use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

/// A lightweight, zero-allocation representation of the agent's current state.
/// Forced 64-byte CPU cache-line alignment to completely eliminate False Sharing
/// between isolated real-time silicon cores and standard background threads.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct AgentStateFrame {
    pub timestamp_ns: u64,
    pub avg_free_energy: f32,
    pub kl_divergence: f32,
    pub primary_curvature: f32,
}

/// A non-blocking telemetry publisher designed strictly for real-time threads.
pub struct RealTimeTelemetryTx {
    tx: SyncSender<AgentStateFrame>,
}

impl RealTimeTelemetryTx {
    /// Pushes a state frame to the shadow thread. If the buffer is full, it quietly
    /// drops the frame rather than blocking the high-frequency control loop.
    #[inline(always)]
    pub fn try_send(&self, frame: AgentStateFrame) {
        let _ = self.tx.try_send(frame);
    }
}

/// Initializes the lock-free conduit. Returns a non-blocking transmitter for the
/// agent loop, and spawns a background thread to handle I/O independently.
pub fn spawn_telemetry_shadow_thread(buffer_size: usize) -> RealTimeTelemetryTx {
    let (tx, rx): (SyncSender<AgentStateFrame>, Receiver<AgentStateFrame>) =
        mpsc::sync_channel(buffer_size);

    // Spawn the low-priority consumer thread
    thread::Builder::new()
        .name("symthaea-telemetry-io".to_string())
        .spawn(move || {
            while let Ok(frame) = rx.recv() {
                // Heavy I/O happens here, safely isolated from the real-time silicon
                // e.g., flush to SQLite, broadcast over WebSockets, or write to standard out
                // println!("Shadow I/O [{}]: Free Energy: {}", frame.timestamp_ns, frame.avg_free_energy);
            }
        })
        .expect("Failed to spawn telemetry shadow thread");

    RealTimeTelemetryTx { tx }
}