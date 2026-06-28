//! Public alpha stability annotations.
//!
//! Alpha.10 introduces a small, dependency-free vocabulary for documenting which
//! crate surfaces are intended to remain stable across nearby alpha releases and
//! which surfaces are still research scaffolding. These annotations are not a
//! SemVer guarantee; they are operator-facing release notes encoded as data.

/// Stability status for an alpha surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaStability {
    /// Intended to remain source-compatible across nearby alpha releases unless
    /// a release note says otherwise.
    StableAlpha,
    /// Useful but still actively changing while the research protocol evolves.
    Experimental,
    /// Placeholder for planned external integration boundaries.
    FutureBoundary,
    /// Kept for compatibility but no longer recommended for new examples.
    DeprecatedAlpha,
}

impl AlphaStability {
    /// Returns a stable lowercase status label.
    pub fn label(self) -> &'static str {
        match self {
            Self::StableAlpha => "stable-alpha",
            Self::Experimental => "experimental",
            Self::FutureBoundary => "future-boundary",
            Self::DeprecatedAlpha => "deprecated-alpha",
        }
    }
}

/// One documented crate surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilityRecord {
    /// Surface name, usually a module, report type, or CLI command.
    pub name: &'static str,
    /// Current alpha stability status.
    pub status: AlphaStability,
    /// Short human-readable purpose.
    pub purpose: &'static str,
    /// Caveat that should be preserved when documenting this surface.
    pub caveat: &'static str,
}

impl StabilityRecord {
    /// Returns a compact line-oriented record.
    pub fn to_text(&self) -> String {
        format!(
            "surface={} status={} purpose={} caveat={}",
            self.name,
            self.status.label(),
            self.purpose,
            self.caveat,
        )
    }

    /// Returns a Markdown table row.
    pub fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {} | {} |",
            self.name,
            self.status.label(),
            self.purpose,
            self.caveat,
        )
    }
}

/// Returns the alpha.10 public surface stability catalog.
pub fn stability_catalog() -> Vec<StabilityRecord> {
    vec![
        StabilityRecord {
            name: "BinaryHypervector",
            status: AlphaStability::StableAlpha,
            purpose: "classical HDC baseline for binding probes",
            caveat: "baseline implementation, not a consciousness metric",
        },
        StabilityRecord {
            name: "PhaseHypervector",
            status: AlphaStability::Experimental,
            purpose: "quantum-inspired phase binding sketch",
            caveat: "quantum-inspired local simulation only",
        },
        StabilityRecord {
            name: "CorrelationBindingSketch",
            status: AlphaStability::Experimental,
            purpose: "parity/correlation-style binding sketch",
            caveat: "research comparison primitive only",
        },
        StabilityRecord {
            name: "BindingProbeRunner",
            status: AlphaStability::StableAlpha,
            purpose: "single reproducible binding probe runner",
            caveat: "local implementation probe, not physical backend evidence",
        },
        StabilityRecord {
            name: "NoiseSweepRunner",
            status: AlphaStability::StableAlpha,
            purpose: "controlled local noise degradation sweep",
            caveat: "noise model is synthetic and must be reported",
        },
        StabilityRecord {
            name: "ExperimentMatrixRunner",
            status: AlphaStability::Experimental,
            purpose: "dimension-by-noise replicated grid runner",
            caveat: "pilot matrix helper, not a formal benchmark suite",
        },
        StabilityRecord {
            name: "ResearchArtifactReceipt",
            status: AlphaStability::FutureBoundary,
            purpose: "local receipt shape for future Mycelix attestation",
            caveat: "non-cryptographic; not a signed source-chain entry",
        },
        StabilityRecord {
            name: "IntegrationDeclaration",
            status: AlphaStability::FutureBoundary,
            purpose: "explicit authority boundary for future adapters",
            caveat: "declaration only; adapters must provide their own validation",
        },
        StabilityRecord {
            name: "symthaea-quantum-comp CLI",
            status: AlphaStability::Experimental,
            purpose: "minimal dependency-free operator CLI",
            caveat: "ergonomic wrapper around local runs only",
        },
    ]
}

/// Returns true when the catalog contains no deprecated alpha surfaces.
pub fn catalog_has_no_deprecated_surfaces(records: &[StabilityRecord]) -> bool {
    !records
        .iter()
        .any(|record| record.status == AlphaStability::DeprecatedAlpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stability_catalog_has_core_surfaces() {
        let catalog = stability_catalog();
        assert!(catalog.iter().any(|r| r.name == "BindingProbeRunner"));
        assert!(catalog_has_no_deprecated_surfaces(&catalog));
        assert!(catalog[0].to_text().contains("surface="));
    }
}
