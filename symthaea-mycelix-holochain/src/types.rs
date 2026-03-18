//! Response types matching the Mycelix SDK for conductor zome call results.

use serde::{Deserialize, Serialize};

/// SAP currency balance for a member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceResponse {
    pub member_did: String,
    pub currency: String,
    pub balance: u64,
    pub available: bool,
}

/// TEND time-currency balance for a member, including MYCEL score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TendBalanceResponse {
    pub member_did: String,
    pub balance: i32,
    pub mycel_score: f64,
    pub available: bool,
}

/// Fee tier information derived from MYCEL score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTierResponse {
    pub member_did: String,
    pub mycel_score: f64,
    pub tier_name: String,
    pub base_fee_rate: f64,
}
