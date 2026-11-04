//! Optional audit API, compiled only when the `audit` feature is enabled.

use symthaea_ceb_domain::UserId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessRecord {
    pub user_id: UserId,
    pub allowed: bool,
}

pub fn record_access(user_id: UserId, allowed: bool) -> AccessRecord {
    AccessRecord { user_id, allowed }
}
