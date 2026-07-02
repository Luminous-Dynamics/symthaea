// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock data for Finance cluster — three-currency system (TEND/SAP/MYCEL).

use finance_leptos_types::*;

pub fn tend_balance() -> TendBalanceView {
    TendBalanceView {
        member_did: "did:mycelix:user-001".into(),
        dao_did: "did:mycelix:dao-001".into(),
        balance: 12,
        total_provided: 48.5,
        total_received: 36.5,
        exchange_count: 15,
        last_activity: 1714521600,
    }
}

pub fn sap_balance() -> SapBalanceView {
    SapBalanceView {
        member_did: "did:mycelix:user-001".into(),
        balance: 2_450_000,
        last_demurrage_at: 1714521600,
        demurrage_pending: 12_000,
    }
}

pub fn mycel_score() -> MycelScoreView {
    MycelScoreView {
        member_did: "did:mycelix:user-001".into(),
        score: 0.62,
        participation: 0.75,
        recognition: 0.58,
        validation: 0.45,
        longevity: 0.70,
        active_months: 14,
        tier: MycelTier::Steward,
        last_updated: 1714521600,
    }
}

pub fn tend_exchanges() -> Vec<TendExchangeView> {
    vec![
        TendExchangeView {
            hash: "h1".into(),
            id: "TE-001".into(),
            provider_did: "did:mycelix:user-001".into(),
            receiver_did: "did:mycelix:user-002".into(),
            hours: 3.0,
            service_description: "Garden maintenance".into(),
            service_category: ServiceCategory::Gardening,
            status: ExchangeStatus::Confirmed,
            created: 1712000000,
        },
        TendExchangeView {
            hash: "h2".into(),
            id: "TE-002".into(),
            provider_did: "did:mycelix:user-003".into(),
            receiver_did: "did:mycelix:user-001".into(),
            hours: 1.5,
            service_description: "Computer repair".into(),
            service_category: ServiceCategory::TechSupport,
            status: ExchangeStatus::Confirmed,
            created: 1713000000,
        },
    ]
}

pub fn sap_payments() -> Vec<SapPaymentView> {
    vec![SapPaymentView {
        hash: "p1".into(),
        id: "SP-001".into(),
        from_did: "did:mycelix:user-001".into(),
        to_did: "did:mycelix:user-004".into(),
        amount: 500_000,
        fee: 5_000,
        memo: Some("Cooperative dues".into()),
        status: PaymentStatus::Completed,
        created: 1713500000,
    }]
}

pub fn treasury() -> TreasuryView {
    TreasuryView {
        hash: "t1".into(),
        id: "TR-001".into(),
        name: "Commons Treasury".into(),
        balance: 15_000_000,
        reserve_ratio: 0.25,
        inalienable_reserve: 3_750_000,
        available: 11_250_000,
        currency: "SAP".into(),
        created: 1700000000,
    }
}

pub fn stakes() -> Vec<StakeView> {
    vec![StakeView {
        hash: "s1".into(),
        id: "ST-001".into(),
        staker_did: "did:mycelix:user-001".into(),
        sap_amount: 1_000_000,
        mycel_score: 0.62,
        stake_weight: 0.78,
        status: StakeStatus::Active,
        created: 1710000000,
        unbonding_until: None,
    }]
}

pub fn oracle_state() -> OracleStateView {
    OracleStateView {
        vitality: 85,
        tier: OracleTier::Normal,
        updated_at: 1714521600,
    }
}

pub fn recognitions() -> Vec<RecognitionEventView> {
    vec![RecognitionEventView {
        hash: "r1".into(),
        recognizer_did: "did:mycelix:user-002".into(),
        recipient_did: "did:mycelix:user-001".into(),
        contribution_type: ContributionType::Technical,
        weight: 0.8,
        cycle_id: "cycle-2024-q2".into(),
        created: 1713000000,
    }]
}
