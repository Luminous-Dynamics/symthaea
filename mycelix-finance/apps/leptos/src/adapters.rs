// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapters — map finance wire types into frontend-safe display views.

use finance_leptos_types as view;
use finance_wire_types as wire;

pub fn map_tend_balance(balance: wire::BalanceInfo, now: i64) -> view::TendBalanceView {
    view::TendBalanceView {
        member_did: balance.member_did,
        dao_did: balance.dao_did,
        balance: balance.balance,
        total_provided: balance.total_provided,
        total_received: balance.total_received,
        exchange_count: balance.exchange_count,
        last_activity: now,
    }
}

pub fn map_sap_balance(balance: wire::SapBalanceResponse) -> view::SapBalanceView {
    view::SapBalanceView {
        member_did: balance.member_did,
        balance: balance.effective_balance,
        last_demurrage_at: balance.last_demurrage_at,
        demurrage_pending: balance.pending_demurrage,
    }
}

pub fn map_mycel_score(score: wire::MemberMycelState) -> view::MycelScoreView {
    view::MycelScoreView {
        member_did: score.member_did,
        score: score.mycel_score,
        participation: score.participation,
        recognition: score.recognition,
        validation: score.validation,
        longevity: score.longevity,
        active_months: score.active_months,
        tier: view::MycelTier::from_score(score.mycel_score),
        last_updated: score.last_updated as i64,
    }
}

pub fn map_payment(payment: wire::Payment) -> view::SapPaymentView {
    view::SapPaymentView {
        hash: String::new(),
        id: payment.id,
        from_did: payment.from_did,
        to_did: payment.to_did,
        amount: payment.amount,
        fee: payment.fee,
        memo: payment.memo,
        status: map_payment_status(payment.status),
        created: payment.created as i64,
    }
}

pub fn map_exchange(exchange: wire::ExchangeRecord) -> view::TendExchangeView {
    view::TendExchangeView {
        hash: String::new(),
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: map_service_category(exchange.service_category),
        status: map_exchange_status(exchange.status),
        created: exchange.timestamp as i64,
    }
}

pub fn map_oracle_state(oracle: wire::OracleStateResponse, now: i64) -> view::OracleStateView {
    view::OracleStateView {
        vitality: oracle.vitality,
        tier: match oracle.tier_name.as_str() {
            "Elevated" => view::OracleTier::Elevated,
            "High" => view::OracleTier::High,
            "Emergency" => view::OracleTier::Emergency,
            _ => view::OracleTier::Normal,
        },
        updated_at: now,
    }
}

pub fn map_recognition(event: wire::RecognitionEvent) -> view::RecognitionEventView {
    view::RecognitionEventView {
        hash: String::new(),
        recognizer_did: event.recognizer_did,
        recipient_did: event.recipient_did,
        contribution_type: map_contribution_type(event.contribution_type),
        weight: event.weight,
        cycle_id: event.cycle_id,
        created: event.timestamp as i64,
    }
}

pub fn map_stake(stake: wire::CollateralStake) -> view::StakeView {
    view::StakeView {
        hash: String::new(),
        id: stake.id,
        staker_did: stake.staker_did,
        sap_amount: stake.sap_amount,
        mycel_score: stake.mycel_score,
        stake_weight: stake.stake_weight,
        status: map_stake_status(stake.status),
        created: stake.staked_at as i64,
        unbonding_until: stake.unbonding_until.map(|value| value as i64),
    }
}

pub fn map_treasury(treasury: wire::Treasury) -> view::TreasuryView {
    view::TreasuryView {
        hash: String::new(),
        id: treasury.id,
        name: treasury.name,
        balance: treasury.balance,
        reserve_ratio: treasury.reserve_ratio,
        inalienable_reserve: 0,
        available: treasury.balance,
        currency: treasury.currency,
        created: treasury.created as i64,
    }
}

pub fn map_commons_pool(pool: wire::CommonsPool) -> view::TreasuryView {
    view::TreasuryView {
        hash: String::new(),
        id: pool.id,
        name: format!("{} Commons Pool", pool.dao_did),
        balance: pool.inalienable_reserve.saturating_add(pool.available_balance),
        reserve_ratio: 1.0,
        inalienable_reserve: pool.inalienable_reserve,
        available: pool.available_balance,
        currency: "SAP".into(),
        created: pool.created_at as i64,
    }
}

pub fn map_service_category(category: wire::ServiceCategory) -> view::ServiceCategory {
    match category {
        wire::ServiceCategory::CareWork => view::ServiceCategory::CareWork,
        wire::ServiceCategory::HomeServices => view::ServiceCategory::HomeServices,
        wire::ServiceCategory::FoodServices => view::ServiceCategory::FoodServices,
        wire::ServiceCategory::Transportation => view::ServiceCategory::Transportation,
        wire::ServiceCategory::Education => view::ServiceCategory::Education,
        wire::ServiceCategory::GeneralAssistance => view::ServiceCategory::GeneralAssistance,
        wire::ServiceCategory::Administrative => view::ServiceCategory::Administrative,
        wire::ServiceCategory::Creative => view::ServiceCategory::Creative,
        wire::ServiceCategory::TechSupport => view::ServiceCategory::TechSupport,
        wire::ServiceCategory::Wellness => view::ServiceCategory::Wellness,
        wire::ServiceCategory::Gardening => view::ServiceCategory::Gardening,
        wire::ServiceCategory::Custom(value) => view::ServiceCategory::Custom(value),
    }
}

pub fn map_contribution_type(kind: wire::ContributionType) -> view::ContributionType {
    match kind {
        wire::ContributionType::Technical => view::ContributionType::Technical,
        wire::ContributionType::Community => view::ContributionType::Community,
        wire::ContributionType::Care => view::ContributionType::Care,
        wire::ContributionType::Governance => view::ContributionType::Governance,
        wire::ContributionType::Creative => view::ContributionType::Creative,
        wire::ContributionType::Education => view::ContributionType::Education,
        wire::ContributionType::General => view::ContributionType::General,
    }
}

fn map_payment_status(status: wire::TransferStatus) -> view::PaymentStatus {
    match status {
        wire::TransferStatus::Pending => view::PaymentStatus::Pending,
        wire::TransferStatus::Processing => view::PaymentStatus::Processing,
        wire::TransferStatus::Completed => view::PaymentStatus::Completed,
        wire::TransferStatus::Failed(_) => view::PaymentStatus::Failed,
        wire::TransferStatus::Cancelled => view::PaymentStatus::Cancelled,
        wire::TransferStatus::Refunded => view::PaymentStatus::Refunded,
    }
}

fn map_exchange_status(status: wire::ExchangeStatus) -> view::ExchangeStatus {
    match status {
        wire::ExchangeStatus::Proposed => view::ExchangeStatus::Proposed,
        wire::ExchangeStatus::Confirmed => view::ExchangeStatus::Confirmed,
        wire::ExchangeStatus::Disputed => view::ExchangeStatus::Disputed,
        wire::ExchangeStatus::Cancelled => view::ExchangeStatus::Cancelled,
        wire::ExchangeStatus::Resolved => view::ExchangeStatus::Resolved,
    }
}

fn map_stake_status(status: wire::StakeStatus) -> view::StakeStatus {
    match status {
        wire::StakeStatus::Active => view::StakeStatus::Active,
        wire::StakeStatus::Unbonding => view::StakeStatus::Unbonding,
        wire::StakeStatus::Withdrawn => view::StakeStatus::Withdrawn,
        wire::StakeStatus::Slashed | wire::StakeStatus::Jailed => view::StakeStatus::Slashed,
    }
}
