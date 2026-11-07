//! Event contract types for the public CEB duplicate-replay development fixture.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub event_id: String,
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedEffect {
    pub event_id: String,
    pub payload: String,
    pub ordinal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOutcome {
    Applied(AppliedEffect),
}
