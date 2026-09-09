// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gluonic hadrons and glueball-channel HDC encodings.
//!
//! This module complements `hadrons.rs`, whose current API focuses on
//! conventional quark-model baryons and mesons. Glueballs are not encoded as a
//! fixed count of constituent gluons here. Instead, they are represented as
//! color-singlet, self-bound gluonic QCD states tagged by their J^PC channel.
//!
//! This is a semantic/HDC bridge, not a lattice-QCD eigensolver and not a
//! substitute for spectroscopy evidence. Numerical masses and interpretation
//! status live in `symthaea-particle-physics`.

use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Low-lying glueball J^PC channels represented in the core HDC ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GlueballChannel {
    /// Scalar 0++ channel.
    Scalar0PlusPlus,
    /// Tensor 2++ channel.
    Tensor2PlusPlus,
    /// Pseudoscalar 0-+ channel.
    Pseudoscalar0MinusPlus,
}

impl GlueballChannel {
    fn domain_label(self) -> &'static str {
        match self {
            Self::Scalar0PlusPlus => "hadron::glueball::jpc_0pp",
            Self::Tensor2PlusPlus => "hadron::glueball::jpc_2pp",
            Self::Pseudoscalar0MinusPlus => "hadron::glueball::jpc_0mp",
        }
    }
}

/// HDC representation of the low-lying gluonic bound-state sector.
#[derive(Debug, Clone)]
pub struct GluonicHadrons {
    /// Semantic concept: color-singlet gluonic state.
    pub color_singlet: ContinuousHV,
    /// Semantic concept: non-Abelian gluon self-interaction.
    pub self_interaction: ContinuousHV,
    /// Semantic concept: confined/bound QCD state.
    pub bound_state: ContinuousHV,
    /// Scalar 0++ glueball channel.
    pub scalar_0pp: ContinuousHV,
    /// Tensor 2++ glueball channel.
    pub tensor_2pp: ContinuousHV,
    /// Pseudoscalar 0-+ glueball channel.
    pub pseudoscalar_0mp: ContinuousHV,
}

impl GluonicHadrons {
    /// Construct glueball-channel concepts from the Standard Model gluon.
    ///
    /// The construction deliberately avoids encoding a constituent-gluon
    /// number. In non-perturbative QCD, physical glueball states are better
    /// treated as gauge-invariant gluonic field excitations whose observable
    /// identities are determined by quantum numbers and spectroscopy.
    pub fn from_model(model: &StandardModel, genesis: &GenesisSeed) -> Self {
        let color_singlet = genesis.hv("hadron::gluonic::color_singlet", PHYSICS_DIM);
        let self_interaction = genesis.hv("qcd::gluon_self_interaction", PHYSICS_DIM);
        let bound_state = genesis.hv("hadron::gluonic::bound_state", PHYSICS_DIM);

        let gluonic_base = model
            .gluon
            .bind(&self_interaction)
            .bind(&color_singlet)
            .bind(&bound_state);

        let scalar_0pp = Self::encode_channel(&gluonic_base, genesis, GlueballChannel::Scalar0PlusPlus);
        let tensor_2pp = Self::encode_channel(&gluonic_base, genesis, GlueballChannel::Tensor2PlusPlus);
        let pseudoscalar_0mp =
            Self::encode_channel(&gluonic_base, genesis, GlueballChannel::Pseudoscalar0MinusPlus);

        Self {
            color_singlet,
            self_interaction,
            bound_state,
            scalar_0pp,
            tensor_2pp,
            pseudoscalar_0mp,
        }
    }

    fn encode_channel(
        gluonic_base: &ContinuousHV,
        genesis: &GenesisSeed,
        channel: GlueballChannel,
    ) -> ContinuousHV {
        gluonic_base
            .bind(&genesis.hv(channel.domain_label(), PHYSICS_DIM))
            .normalize()
    }

    /// Get the HDC vector for a glueball J^PC channel.
    pub fn glueball(&self, channel: GlueballChannel) -> &ContinuousHV {
        match channel {
            GlueballChannel::Scalar0PlusPlus => &self.scalar_0pp,
            GlueballChannel::Tensor2PlusPlus => &self.tensor_2pp,
            GlueballChannel::Pseudoscalar0MinusPlus => &self.pseudoscalar_0mp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_all_low_lying_glueball_channels() {
        let genesis = GenesisSeed::from_phrase("gluonic hadron channels");
        let model = StandardModel::from_genesis(&genesis);
        let gluonic = GluonicHadrons::from_model(&model, &genesis);

        assert_eq!(gluonic.scalar_0pp.dim(), PHYSICS_DIM);
        assert_eq!(gluonic.tensor_2pp.dim(), PHYSICS_DIM);
        assert_eq!(gluonic.pseudoscalar_0mp.dim(), PHYSICS_DIM);
    }

    #[test]
    fn channels_share_gluonic_semantic_foundation() {
        let genesis = GenesisSeed::from_phrase("gluonic semantic foundation");
        let model = StandardModel::from_genesis(&genesis);
        let gluonic = GluonicHadrons::from_model(&model, &genesis);

        assert_eq!(gluonic.color_singlet.dim(), PHYSICS_DIM);
        assert_eq!(gluonic.self_interaction.dim(), PHYSICS_DIM);
        assert_eq!(gluonic.bound_state.dim(), PHYSICS_DIM);
        assert_eq!(
            gluonic.glueball(GlueballChannel::Pseudoscalar0MinusPlus).dim(),
            PHYSICS_DIM
        );
    }
}
