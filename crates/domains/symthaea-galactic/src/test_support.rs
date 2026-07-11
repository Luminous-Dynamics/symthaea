// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared helpers for `#[ignore]`d integration tests that need the real
//! SPARC dataset. Only compiled under `#[cfg(test)]`.

use std::path::{Path, PathBuf};

/// Default data dir relative to this crate; override with
/// `SYMTHAEA_SPARC_DATA_DIR` (absolute or CWD-relative).
pub(crate) fn sparc_data_dir() -> PathBuf {
    std::env::var("SYMTHAEA_SPARC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/benchmarks/sparc")
        })
}
