//! Replay scenario builder for the public CEB event-replay development fixture.

use ceb_event_replay_contract::Event;
use ceb_event_replay_store::EventProcessor;

/// Deliver one event and then replay the same identity/payload.
///
/// The returned slice is expected to contain two effects under the current
/// intentionally defective implementation. This is a source-level scenario
/// only; it has not been compiled or executed.
pub fn duplicate_delivery_scenario() -> Vec<String> {
    let event = Event {
        event_id: "evt-001".to_owned(),
        payload: "increment:account-7".to_owned(),
    };

    let mut processor = EventProcessor::default();
    let _first = processor.deliver(event.clone());
    let _replay = processor.deliver(event);

    processor
        .effects()
        .iter()
        .map(|effect| effect.event_id.clone())
        .collect()
}
