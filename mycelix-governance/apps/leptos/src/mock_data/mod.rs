// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock civic data: proposals growing, councils deliberating,
//! a charter that breathes, currencies flowing.

use governance_leptos_types::*;
use finance_leptos_types::*;

fn now() -> i64 {
    (js_sys::Date::now() * 1000.0) as i64
}

fn hours_ago(h: i64) -> i64 {
    now() - h * 3_600_000_000
}

fn days_ago(d: i64) -> i64 {
    now() - d * 86_400_000_000
}

fn days_from_now(d: i64) -> i64 {
    now() + d * 86_400_000_000
}

pub fn mock_proposals() -> Vec<ProposalView> {
    vec![
        ProposalView {
            hash: "proposal_001".into(),
            id: "MIP-0042".into(),
            title: "Establish community seed library in the commons".into(),
            description: "A proposal to allocate 500 SAP from the treasury to establish \
                a seed library accessible to all community members. Seeds are the \
                beginning of food sovereignty.".into(),
            proposal_type: ProposalType::Funding,
            author: "did:mycelix:aria".into(),
            status: ProposalStatus::Active,
            actions: "[]".into(),
            discussion_url: None,
            voting_starts: days_ago(2),
            voting_ends: days_from_now(5),
            created: days_ago(7),
            updated: hours_ago(3),
            version: 1,
        },
        ProposalView {
            hash: "proposal_002".into(),
            id: "MIP-0043".into(),
            title: "Reduce constitutional amendment quorum from 40% to 35%".into(),
            description: "As the community grows, the 40% quorum for constitutional \
                amendments becomes increasingly difficult to meet. This proposal \
                seeks to lower it to 35% while maintaining the 67% supermajority \
                requirement.".into(),
            proposal_type: ProposalType::Constitutional,
            author: "did:mycelix:kai".into(),
            status: ProposalStatus::Draft,
            actions: "[]".into(),
            discussion_url: None,
            voting_starts: days_from_now(3),
            voting_ends: days_from_now(10),
            created: days_ago(2),
            updated: hours_ago(12),
            version: 1,
        },
        ProposalView {
            hash: "proposal_003".into(),
            id: "MIP-0041".into(),
            title: "Update TEND credit limit oracle sensitivity".into(),
            description: "Adjust the oracle vitality thresholds to better reflect \
                seasonal patterns in community exchange activity.".into(),
            proposal_type: ProposalType::Parameter,
            author: "did:mycelix:sol".into(),
            status: ProposalStatus::Approved,
            actions: r#"[{"type":"update_parameter","name":"tend_limit_threshold_normal","value":"45"}]"#.into(),
            discussion_url: None,
            voting_starts: days_ago(14),
            voting_ends: days_ago(7),
            created: days_ago(21),
            updated: days_ago(5),
            version: 2,
        },
        ProposalView {
            hash: "proposal_004".into(),
            id: "MIP-0039".into(),
            title: "Emergency water infrastructure repair fund".into(),
            description: "Allocate emergency funds for the repair of the community \
                rainwater capture system damaged in last week's storm.".into(),
            proposal_type: ProposalType::Emergency,
            author: "did:mycelix:river".into(),
            status: ProposalStatus::Executed,
            actions: "[]".into(),
            discussion_url: None,
            voting_starts: days_ago(10),
            voting_ends: days_ago(9),
            created: days_ago(10),
            updated: days_ago(8),
            version: 1,
        },
    ]
}

pub fn mock_votes() -> Vec<PhiVoteView> {
    vec![
        PhiVoteView {
            hash: "vote_001".into(),
            proposal_id: "MIP-0042".into(),
            voter_did: "did:mycelix:mock-citizen".into(),
            tier: ProposalTier::Basic,
            choice: VoteChoice::For,
            effective_weight: 0.72,
            phi_score: 0.55,
            reasoning: Some("Seed libraries are foundational to commons resilience.".into()),
            delegated: false,
            created: days_ago(1),
        },
    ]
}

pub fn mock_councils() -> Vec<CouncilView> {
    vec![
        CouncilView {
            hash: "council_001".into(),
            id: "council-root".into(),
            name: "Root Council".into(),
            purpose: "Steward the constitutional commons and coordinate between domain councils.".into(),
            council_type: CouncilType::Root,
            parent_council_id: None,
            phi_threshold: 0.6,
            quorum: 0.4,
            member_count: 7,
            status: "Active".into(),
            created: days_ago(365),
        },
        CouncilView {
            hash: "council_002".into(),
            id: "council-commons".into(),
            name: "Commons Council".into(),
            purpose: "Govern shared resources: land, water, food, tools.".into(),
            council_type: CouncilType::Domain { domain: "Commons".into() },
            parent_council_id: Some("council-root".into()),
            phi_threshold: 0.4,
            quorum: 0.25,
            member_count: 12,
            status: "Active".into(),
            created: days_ago(300),
        },
        CouncilView {
            hash: "council_003".into(),
            id: "council-finance".into(),
            name: "Finance Council".into(),
            purpose: "Steward the TEND/SAP/MYCEL economic system and treasury allocations.".into(),
            council_type: CouncilType::Domain { domain: "Finance".into() },
            parent_council_id: Some("council-root".into()),
            phi_threshold: 0.5,
            quorum: 0.3,
            member_count: 5,
            status: "Active".into(),
            created: days_ago(300),
        },
    ]
}

pub fn mock_charter() -> CharterView {
    CharterView {
        hash: "charter_001".into(),
        preamble: "We, the members of this commons, commit to governing ourselves \
            through deliberation, mutual care, and consciousness-guided decision-making. \
            No individual or group shall hold permanent power over others.".into(),
        articles: vec![
            ArticleView {
                number: 1,
                title: "Sovereignty and Exit".into(),
                content: "Every member retains the right to exit the commons at any time, \
                    with their personal data and a fair accounting of their contributions.".into(),
            },
            ArticleView {
                number: 2,
                title: "Consciousness Gating".into(),
                content: "Governance participation is gated by consciousness credentials, \
                    ensuring that decision-making weight reflects demonstrated care and engagement.".into(),
            },
            ArticleView {
                number: 3,
                title: "Anti-Tyranny".into(),
                content: "Emergency powers expire automatically. No council may serve more than \
                    three consecutive sessions without a mandatory cooling period.".into(),
            },
        ],
        rights: vec![
            "Right to exit".into(),
            "Right to fork".into(),
            "Consciousness gating".into(),
            "Term limits".into(),
            "Emergency power limits".into(),
            "Oversight funding".into(),
        ],
        version: 1,
        adopted: days_ago(365),
    }
}

// ============================================================================
// Finance Mock Data
// ============================================================================

pub fn mock_tend_balance() -> TendBalanceView {
    TendBalanceView {
        member_did: "did:mycelix:mock-citizen".into(),
        dao_did: "dao:mycelix:commons".into(),
        balance: 12,
        total_provided: 47.5,
        total_received: 35.5,
        exchange_count: 23,
        last_activity: hours_ago(6),
    }
}

pub fn mock_sap_balance() -> SapBalanceView {
    SapBalanceView {
        member_did: "did:mycelix:mock-citizen".into(),
        balance: 1_850_000_000, // 1,850 SAP
        last_demurrage_at: days_ago(7),
        demurrage_pending: 2_534_000, // ~2.5 SAP pending demurrage
    }
}

pub fn mock_mycel_score() -> MycelScoreView {
    MycelScoreView {
        member_did: "did:mycelix:mock-citizen".into(),
        score: 0.52,
        participation: 0.65,
        recognition: 0.48,
        validation: 0.40,
        longevity: 0.55,
        active_months: 14,
        tier: MycelTier::Member,
        last_updated: days_ago(1),
    }
}

pub fn mock_tend_exchanges() -> Vec<TendExchangeView> {
    vec![
        TendExchangeView {
            hash: "exchange_001".into(),
            id: "tend-ex-001".into(),
            provider_did: "did:mycelix:mock-citizen".into(),
            receiver_did: "did:mycelix:aria".into(),
            hours: 2.0,
            service_description: "Helped repair the community garden irrigation system".into(),
            service_category: ServiceCategory::Gardening,
            status: ExchangeStatus::Confirmed,
            created: days_ago(3),
        },
        TendExchangeView {
            hash: "exchange_002".into(),
            id: "tend-ex-002".into(),
            provider_did: "did:mycelix:sol".into(),
            receiver_did: "did:mycelix:mock-citizen".into(),
            hours: 1.5,
            service_description: "Tutored in basic programming concepts".into(),
            service_category: ServiceCategory::Education,
            status: ExchangeStatus::Confirmed,
            created: days_ago(5),
        },
        TendExchangeView {
            hash: "exchange_003".into(),
            id: "tend-ex-003".into(),
            provider_did: "did:mycelix:mock-citizen".into(),
            receiver_did: "did:mycelix:kai".into(),
            hours: 3.0,
            service_description: "Elder care visit and meal preparation".into(),
            service_category: ServiceCategory::CareWork,
            status: ExchangeStatus::Proposed,
            created: hours_ago(4),
        },
    ]
}

pub fn mock_sap_payments() -> Vec<SapPaymentView> {
    vec![
        SapPaymentView {
            hash: "payment_001".into(),
            id: "sap-pay-001".into(),
            from_did: "did:mycelix:mock-citizen".into(),
            to_did: "treasury:commons".into(),
            amount: 50_000_000, // 50 SAP
            fee: 250_000,
            memo: Some("Monthly commons contribution".into()),
            status: PaymentStatus::Completed,
            created: days_ago(14),
        },
    ]
}

pub fn mock_treasury() -> TreasuryView {
    TreasuryView {
        hash: "treasury_001".into(),
        id: "treasury-commons".into(),
        name: "Commons Treasury".into(),
        balance: 125_000_000_000, // 125,000 SAP
        reserve_ratio: 0.25,
        inalienable_reserve: 31_250_000_000,
        available: 93_750_000_000,
        currency: "SAP".into(),
        created: days_ago(365),
    }
}

pub fn mock_oracle_state() -> OracleStateView {
    OracleStateView {
        vitality: 72,
        tier: OracleTier::Normal,
        updated_at: hours_ago(2),
    }
}

pub fn mock_listings() -> Vec<ServiceListingView> {
    vec![
        ServiceListingView {
            hash: "listing_001".into(),
            id: "listing-001".into(),
            provider_did: "did:mycelix:river".into(),
            dao_did: "dao:mycelix:commons".into(),
            title: "Bicycle repair and maintenance".into(),
            description: "I can fix most common bicycle issues — flats, brakes, gears, chains.".into(),
            category: ServiceCategory::GeneralAssistance,
            estimated_hours: 1.5,
            active: true,
            created: days_ago(10),
        },
        ServiceListingView {
            hash: "listing_002".into(),
            id: "listing-002".into(),
            provider_did: "did:mycelix:aria".into(),
            dao_did: "dao:mycelix:commons".into(),
            title: "Sourdough bread baking lessons".into(),
            description: "Learn to make sourdough from starter to loaf. All ages welcome.".into(),
            category: ServiceCategory::FoodServices,
            estimated_hours: 3.0,
            active: true,
            created: days_ago(5),
        },
    ]
}

pub fn mock_requests() -> Vec<ServiceRequestView> {
    vec![
        ServiceRequestView {
            hash: "request_001".into(),
            id: "request-001".into(),
            requester_did: "did:mycelix:kai".into(),
            dao_did: "dao:mycelix:commons".into(),
            title: "Help moving furniture this weekend".into(),
            description: "Downsizing to a smaller home. Need help carrying furniture on Saturday.".into(),
            category: ServiceCategory::HomeServices,
            estimated_hours: 4.0,
            open: true,
            created: days_ago(1),
        },
    ]
}

pub fn mock_recognitions() -> Vec<RecognitionEventView> {
    vec![
        RecognitionEventView {
            hash: "recog_001".into(),
            recognizer_did: "did:mycelix:aria".into(),
            recipient_did: "did:mycelix:mock-citizen".into(),
            contribution_type: ContributionType::Community,
            weight: 0.48,
            cycle_id: "2026-03".into(),
            created: days_ago(5),
        },
        RecognitionEventView {
            hash: "recog_002".into(),
            recognizer_did: "did:mycelix:sol".into(),
            recipient_did: "did:mycelix:mock-citizen".into(),
            contribution_type: ContributionType::Care,
            weight: 0.52,
            cycle_id: "2026-03".into(),
            created: days_ago(12),
        },
    ]
}
