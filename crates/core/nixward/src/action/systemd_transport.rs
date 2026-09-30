// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Narrow, read-only transport boundary for governed systemd observations.
//!
//! This module deliberately returns the exact property projection as UTF-8
//! text. It does not interpret state, authorize an effect, or mint authority.
//! Semantic interpretation belongs to the closed typed evidence parser.

use std::process::Command;

const GOVERNED_PROPERTIES: &str = "Id,LoadState,ActiveState,SubState,UnitFileState,CanStart,CanStop,CanReload";

/// Read the exact governed systemd property projection for one canonical unit.
///
/// The returned text is intentionally unparsed. Callers must pass it through
/// the strict typed evidence parser rather than creating a second semantic
/// representation.
///
/// The transport is observational only and may become stale immediately after
/// the command returns; it is never execution authority.
pub fn observe_service_properties(unit: &str) -> Result<String, std::io::Error> {
    let output = Command::new("systemctl")
        .args([
            "show",
            unit,
            "--no-pager",
            &format!("--property={GOVERNED_PROPERTIES}"),
        ])
        .output()?;

    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "systemctl show failed for '{}': {}",
            unit,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    std::str::from_utf8(&output.stdout)
        .map(str::to_owned)
        .map_err(|error| {
            std::io::Error::other(format!(
                "systemctl show returned invalid UTF-8 for '{}': {error}",
                unit
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governed_property_projection_is_frozen() {
        assert_eq!(
            GOVERNED_PROPERTIES,
            "Id,LoadState,ActiveState,SubState,UnitFileState,CanStart,CanStop,CanReload"
        );
    }
}
