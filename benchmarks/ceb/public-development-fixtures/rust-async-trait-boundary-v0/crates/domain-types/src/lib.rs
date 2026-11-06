//! Domain values shared by the public CEB async-trait development fixture.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub id: String,
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryReceipt {
    pub event_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceError {
    Rejected,
    Unavailable,
}
