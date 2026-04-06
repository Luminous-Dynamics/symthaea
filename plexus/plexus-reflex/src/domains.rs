// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain classification lists for the reflex arc.
//!
//! Safe domains = known E4 public knowledge sources (never require consent to broadcast).
//! Private domains = known authenticated/financial services (always Private zone).

/// Known safe domains — E4 public knowledge, always classified as Public zone.
pub const SAFE_DOMAINS: &[&str] = &[
    // Encyclopedias
    "wikipedia.org",
    "wikimedia.org",
    "wiktionary.org",
    "wikisource.org",
    // Documentation
    "docs.rs",
    "doc.rust-lang.org",
    "developer.mozilla.org",
    "devdocs.io",
    "man7.org",
    // Academic
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com",
    "doi.org",
    // Standards bodies
    "w3.org",
    "ietf.org",
    "rfc-editor.org",
    "unicode.org",
    // Government data
    "data.gov",
    "who.int",
    "fao.org",
    "noaa.gov",
    "nasa.gov",
    // Open source
    "github.com",
    "gitlab.com",
    "crates.io",
    "npmjs.com",
    "pypi.org",
];

/// Known private domains — always classified as Private zone.
pub const PRIVATE_DOMAINS: &[&str] = &[
    // Banking / financial
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "capitalone.com",
    "citibank.com",
    "usbank.com",
    "paypal.com",
    "venmo.com",
    "schwab.com",
    "fidelity.com",
    "vanguard.com",
    // Email
    "mail.google.com",
    "outlook.live.com",
    "mail.yahoo.com",
    "protonmail.com",
    "proton.me",
    "tutanota.com",
    // Cloud / corporate
    "console.aws.amazon.com",
    "portal.azure.com",
    "console.cloud.google.com",
    "app.slack.com",
    "teams.microsoft.com",
    // Medical
    "mychart.com",
    "patient.portal",
    // Tax / government
    "irs.gov",
    "ssa.gov",
];

/// URL path patterns that indicate private/authenticated content.
pub const PRIVATE_PATHS: &[&str] = &[
    "/login",
    "/signin",
    "/sign-in",
    "/signup",
    "/sign-up",
    "/account",
    "/accounts",
    "/admin",
    "/dashboard",
    "/settings",
    "/profile",
    "/checkout",
    "/payment",
    "/billing",
    "/oauth",
    "/auth",
    "/api/auth",
    "/password",
    "/reset-password",
    "/two-factor",
    "/mfa",
];
