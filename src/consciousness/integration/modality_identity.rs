// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Stable identity and enumeration contracts for cross-modal modality variants.
//!
//! Deterministic modality behavior must never depend on the incidental layout of
//! a Rust enum. This module therefore owns the explicit stable IDs used for noise
//! domains and deterministic ordering.
//!
//! Enumeration is deliberately a separate contract from identity. The historical
//! `Modality::all()` helper selects the default root channel set; it is not an
//! exhaustive list of every defined modality. Likewise `Modality::sensory()` is
//! the legacy amodal-convergence subset rather than an exhaustive scientific
//! classification.
//!
//! Olfaction, gustation, and chemesthesis are now *defined* canonical modality
//! identities with stable IDs 13–15, but they remain intentionally absent from
//! both legacy runtime topology sets. Defining a sense is not the same operation
//! as enabling it in cognition.

use super::cross_modal_binding::Modality;

/// Exhaustive list of every modality variant currently defined by the enum.
///
/// Use this for identity/invariant checks that genuinely mean "every defined
/// variant". Do not substitute `Modality::all()`: that helper is a legacy root
/// runtime selection and intentionally has a different contract.
pub const DEFINED_MODALITIES: [Modality; 16] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Textual,
    Modality::Linguistic,
    Modality::Proprioceptive,
    Modality::Somatosensory,
    Modality::Motor,
    Modality::Temporal,
    Modality::Spatial,
    Modality::Affective,
    Modality::Emotional,
    Modality::Interoceptive,
    Modality::Abstract,
    Modality::Olfactory,
    Modality::Gustatory,
    Modality::Chemesthetic,
];

/// Historical set instantiated by `MultiModalIntegrator` through
/// `Modality::all()`.
///
/// This is named explicitly because it is not exhaustive. Chemical integration
/// must opt a modality into this topology deliberately; merely defining a new
/// modality must not create a live root channel.
pub const LEGACY_ROOT_MODALITIES: [Modality; 7] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Linguistic,
    Modality::Somatosensory,
    Modality::Motor,
    Modality::Emotional,
    Modality::Interoceptive,
];

/// Historical set returned by `Modality::sensory()` and used by the legacy
/// amodal convergence zone.
pub const LEGACY_SENSORY_MODALITIES: [Modality; 4] = [
    Modality::Visual,
    Modality::Auditory,
    Modality::Linguistic,
    Modality::Somatosensory,
];

/// Stable IDs assigned to the chemical-sensing modalities.
pub const OLFACTORY_STABLE_ID: u16 = 13;
pub const GUSTATORY_STABLE_ID: u16 = 14;
pub const CHEMESTHETIC_STABLE_ID: u16 = 15;

/// Stable numeric identity for every currently-defined modality.
///
/// New deterministic code must use this instead of `modality as u64` /
/// `modality as u8`. Existing historical modalities intentionally preserve their
/// original numeric identities `0..=12`; the new chemical modalities occupy the
/// previously reserved IDs `13..=15`.
pub const fn stable_modality_id(modality: Modality) -> u16 {
    match modality {
        Modality::Visual => 0,
        Modality::Auditory => 1,
        Modality::Textual => 2,
        Modality::Linguistic => 3,
        Modality::Proprioceptive => 4,
        Modality::Somatosensory => 5,
        Modality::Motor => 6,
        Modality::Temporal => 7,
        Modality::Spatial => 8,
        Modality::Affective => 9,
        Modality::Emotional => 10,
        Modality::Interoceptive => 11,
        Modality::Abstract => 12,
        Modality::Olfactory => OLFACTORY_STABLE_ID,
        Modality::Gustatory => GUSTATORY_STABLE_ID,
        Modality::Chemesthetic => CHEMESTHETIC_STABLE_ID,
    }
}

/// Stable deterministic seed domain for modality-specific stochastic transforms.
pub const fn modality_seed(modality: Modality) -> u64 {
    stable_modality_id(modality) as u64
}

/// Stable ordering key for deterministic modality iteration.
pub const fn modality_sort_key(modality: Modality) -> u16 {
    stable_modality_id(modality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn exhaustive_defined_set_has_every_variant_once() {
        let ids: BTreeSet<u16> = DEFINED_MODALITIES
            .into_iter()
            .map(stable_modality_id)
            .collect();
        assert_eq!(DEFINED_MODALITIES.len(), 16);
        assert_eq!(ids.len(), DEFINED_MODALITIES.len());
    }

    #[test]
    fn declared_discriminants_match_explicit_stable_ids() {
        for modality in DEFINED_MODALITIES {
            assert_eq!(
                modality as u16,
                stable_modality_id(modality),
                "reordering or inserting before an existing modality changes deterministic identity"
            );
        }
    }

    #[test]
    fn legacy_root_enumeration_is_explicit_not_exhaustive() {
        assert_eq!(Modality::all(), LEGACY_ROOT_MODALITIES.to_vec());
        assert!(LEGACY_ROOT_MODALITIES.len() < DEFINED_MODALITIES.len());
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Proprioceptive));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Temporal));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Spatial));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Abstract));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Olfactory));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Gustatory));
        assert!(!LEGACY_ROOT_MODALITIES.contains(&Modality::Chemesthetic));
    }

    #[test]
    fn legacy_sensory_enumeration_is_pinned_separately() {
        assert_eq!(Modality::sensory(), LEGACY_SENSORY_MODALITIES.to_vec());
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Proprioceptive));
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Interoceptive));
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Olfactory));
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Gustatory));
        assert!(!LEGACY_SENSORY_MODALITIES.contains(&Modality::Chemesthetic));
    }

    #[test]
    fn stable_ids_are_unique() {
        let ids: BTreeSet<u16> = DEFINED_MODALITIES
            .into_iter()
            .map(stable_modality_id)
            .collect();
        assert_eq!(ids.len(), DEFINED_MODALITIES.len());
    }

    #[test]
    fn chemical_modalities_are_defined_but_not_root_configured() {
        for (modality, expected_id) in [
            (Modality::Olfactory, OLFACTORY_STABLE_ID),
            (Modality::Gustatory, GUSTATORY_STABLE_ID),
            (Modality::Chemesthetic, CHEMESTHETIC_STABLE_ID),
        ] {
            assert!(DEFINED_MODALITIES.contains(&modality));
            assert_eq!(stable_modality_id(modality), expected_id);
            assert!(!LEGACY_ROOT_MODALITIES.contains(&modality));
            assert!(!LEGACY_SENSORY_MODALITIES.contains(&modality));
            assert!(!Modality::all().contains(&modality));
            assert!(!Modality::sensory().contains(&modality));
        }

        assert_eq!(OLFACTORY_STABLE_ID, 13);
        assert_eq!(GUSTATORY_STABLE_ID, 14);
        assert_eq!(CHEMESTHETIC_STABLE_ID, 15);
    }
}
