// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Template domain module for Mycelix Sensorium.
//!
//! # Creating a New Domain
//!
//! 1. Copy this crate to `mycelix-sensorium/crates/domain-yourdomain/`
//! 2. Rename `TemplateDomain` to `YourDomain`
//! 3. Fill in the trait methods with your domain's real data
//! 4. Add to `mycelix-sensorium/Cargo.toml`:
//!    ```toml
//!    domain-yourdomain = { path = "crates/domain-yourdomain", optional = true }
//!    # ...
//!    [features]
//!    yourdomain = ["dep:domain-yourdomain"]
//!    ```
//! 5. Add to `sensorium-shell/src/main.rs` in `build_registry()`:
//!    ```rust
//!    #[cfg(feature = "yourdomain")]
//!    registry.register(Box::new(domain_yourdomain::YourDomain));
//!    ```
//!
//! # External Frontend (Alternative)
//!
//! If you prefer to run your frontend as a separate web app instead of
//! compiling into the Sensorium, create a JSON manifest:
//!
//! ```json
//! {
//!     "id": "yourdomain",
//!     "name": "Your Domain",
//!     "version": "1.0.0",
//!     "author": "your-name",
//!     "bio_name": "Your Metaphor",
//!     "color_primary": "#22C55E",
//!     "color_glow": "#86EFAC",
//!     "min_tier": "Participant",
//!     "frontend_url": "https://yourdomain.example.com",
//!     "required_clusters": ["identity"],
//!     "optional_clusters": [],
//!     "description": "What your domain does"
//! }
//! ```
//!
//! Place it in `~/.mycelix/extensions.json` (array of manifests).
//! The portal will show your domain as an orbital node that opens
//! your frontend URL in a new tab.

use portal_domain_trait::*;

/// Your domain module. Rename this to match your domain.
pub struct TemplateDomain;

impl DomainModule for TemplateDomain {
    fn id(&self) -> &'static str { "template" }

    fn name(&self) -> &'static str { "Template" }

    fn bio_name(&self) -> &'static str { "Seedling" }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#22C55E",
            glow: "#86EFAC",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Overview",
                bio_label: "Root",
                path: "/template",
            },
            NavItem {
                label: "Data",
                bio_label: "Leaves",
                path: "/template/data",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Participant
    }

    fn key_context(&self) -> &'static [u8] {
        b"mycelix-template-v1-encryption"
    }

    fn happ_role(&self) -> &'static str { "template" }

    fn zomes(&self) -> &'static [&'static str] {
        &["template_core", "template_bridge"]
    }

    fn description(&self) -> &'static str {
        "A template domain — replace this with your domain's description"
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                required: true,
                reason: "DID resolution for user identity",
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                zome: "template_core",
                label: "Template Record",
                sensitivity: DataSensitivity::Community,
            },
        ]
    }
}
