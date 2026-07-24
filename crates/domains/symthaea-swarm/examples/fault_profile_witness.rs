//! Deterministic fault-schedule smoke witness.

use symthaea_swarm::fault::{DeterministicFaultInjector, FaultProfile, SubmissionOutcome};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut injector = DeterministicFaultInjector::new(FaultProfile::HOSTILE_DATAGRAMS, 0x5eed)?;
    for sequence in 0..256u16 {
        let payload = sequence.to_le_bytes().repeat(16);
        match injector.submit(sequence as u64, payload)? {
            SubmissionOutcome::Scheduled { .. } | SubmissionOutcome::Dropped => {}
        }
    }
    let delivered = injector.drain_ready(u64::MAX);
    let metrics = injector.metrics();
    assert!(metrics.dropped > 0);
    assert!(metrics.duplicated > 0);
    assert!(metrics.corrupted > 0);
    assert_eq!(metrics.queued_packets, 0);
    assert_eq!(metrics.delivered as usize, delivered.len());
    println!(
        "FAULT_WITNESS_OK submitted={} delivered={} dropped={} duplicated={} corrupted={}",
        metrics.submitted,
        metrics.delivered,
        metrics.dropped,
        metrics.duplicated,
        metrics.corrupted,
    );
    Ok(())
}
