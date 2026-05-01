// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance shared types — mirrors integrity zome entries without HDI dependency.

use serde::{Deserialize, Serialize};

// ── Payments ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Payment {
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub fee: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub status: TransferStatus,
    pub memo: Option<String>,
    pub created: i64,
    pub completed: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PaymentType {
    Direct,
    TreasuryContribution,
    CommonsContribution,
    Escrow,
    Recurring,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TransferStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
    Refunded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SapBalance {
    pub member_did: String,
    pub balance: u64,
    pub last_demurrage_at: i64,
}

// ── Treasury ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Treasury {
    pub id: String,
    pub name: String,
    pub description: String,
    pub currency: String,
    pub balance: u64,
    pub reserve_ratio: f64,
    pub managers: Vec<String>,
    pub created: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contribution {
    pub id: String,
    pub treasury_id: String,
    pub contributor_did: String,
    pub amount: u64,
    pub currency: String,
    pub contribution_type: ContributionType,
    pub timestamp: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ContributionType {
    Deposit,
    Yield,
    Fee,
    Grant,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommonsPool {
    pub id: String,
    pub dao_did: String,
    pub inalienable_reserve: u64,
    pub available_balance: u64,
    pub demurrage_exempt: bool,
    pub created_at: i64,
}

// ── Staking ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollateralStake {
    pub id: String,
    pub staker_did: String,
    pub sap_amount: u64,
    pub mycel_score: f32,
    pub stake_weight: f32,
    pub staked_at: i64,
    pub unbonding_until: Option<i64>,
    pub status: StakeStatus,
    pub pending_rewards: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StakeStatus {
    Active,
    Unbonding,
    Withdrawn,
    Slashed,
    Jailed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlashingEvent {
    pub id: String,
    pub stake_id: String,
    pub reason: SlashingReason,
    pub slash_percentage: u8,
    pub sap_slashed: u64,
    pub slashed_at: i64,
    pub jailed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SlashingReason {
    DoubleSigning,
    Downtime,
    ByzantineConsensus,
    InvalidGradient,
    CartelActivity,
    GovernanceManipulation,
}

// ── Recognition (MYCEL Score) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecognitionEvent {
    pub recognizer_did: String,
    pub recipient_did: String,
    pub weight: f64,
    pub cycle_id: String,
    pub timestamp: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemberMycelState {
    pub member_did: String,
    pub mycel_score: f64,
    pub participation: f64,
    pub recognition: f64,
    pub validation: f64,
    pub longevity: f64,
    pub active_months: u32,
    pub is_apprentice: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn payment_roundtrip() {
        let p = Payment {
            id: "p-1".into(),
            from_did: "a".into(),
            to_did: "b".into(),
            amount: 500,
            fee: 5,
            currency: "SAP".into(),
            payment_type: PaymentType::Direct,
            status: TransferStatus::Completed,
            memo: None,
            created: 0,
            completed: Some(100),
        };
        let bytes = rmp_serde::to_vec_named(&p).unwrap();
        let decoded: Payment = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.amount, 500);
        assert_eq!(decoded.status, TransferStatus::Completed);
    }
}
