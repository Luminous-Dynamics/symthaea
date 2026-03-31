// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

mod home;
pub mod proposals;
mod voting;
mod councils;
mod tend;
mod payments;
mod recognition;
mod treasury;
mod budgeting;
mod constitution;
mod execution;
mod staking;
mod oracle;
mod profile;

pub use home::HomePage;
pub use proposals::{ProposalListPage, ProposalDetailPage, CreateProposalPage};
pub use voting::VotingPage;
pub use councils::CouncilsPage;
pub use tend::TendPage;
pub use payments::PaymentsPage;
pub use recognition::RecognitionPage;
pub use treasury::TreasuryPage;
pub use budgeting::BudgetingPage;
pub use constitution::ConstitutionPage;
pub use execution::ExecutionPage;
pub use staking::StakingPage;
pub use oracle::OraclePage;
pub use profile::ProfilePage;
