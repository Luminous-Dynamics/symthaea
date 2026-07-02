// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

mod home;
mod did;
mod mfa;
mod credentials;
mod recovery;
mod trust;

pub use home::HomePage;
pub use did::DidPage;
pub use mfa::MfaPage;
pub use credentials::CredentialsPage;
pub use recovery::RecoveryPage;
pub use trust::TrustPage;
