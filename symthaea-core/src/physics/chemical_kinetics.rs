// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Chemical Kinetics
//!
//! **Status**: Exploratory module — HDC encodings for chemical kinetics (rate laws, mechanisms, catalysis).
//! Not re-exported due to name conflicts with chemistry.
//!
//! This module encodes reaction rates, mechanisms, and catalysis.
//!
//! ## Rate Laws
//!
//! - **Zero order**: rate = k
//! - **First order**: rate = k[A]
//! - **Second order**: rate = k[A][B] or k[A]²
//!
//! ## Key Equations
//!
//! - **Arrhenius**: k = A·exp(-Ea/RT)
//! - **Eyring**: k = (kT/h)·exp(-ΔG‡/RT)
//! - **Michaelis-Menten**: v = Vmax[S]/(Km + [S])
//!
//! ## Mechanisms
//!
//! - Elementary reactions
//! - Complex mechanisms with intermediates
//! - Chain reactions
//! - Catalytic cycles

use super::constants::R_GAS;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Reaction order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReactionOrder {
    Zeroth,
    First,
    Second,
    Third,
    Fractional,
}

/// Reaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReactionType {
    Elementary,
    Complex,
    Chain,
    Catalytic,
    Enzymatic,
    Photochemical,
    Electrochemical,
}

/// Catalyst type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatalystType {
    Homogeneous,
    Heterogeneous,
    Enzyme,
    Acid,
    Base,
    Metal,
    Photocatalyst,
}

/// A chemical reaction
#[derive(Debug, Clone)]
pub struct Reaction {
    pub name: String,
    pub reaction_type: ReactionType,
    pub order: ReactionOrder,
    pub activation_energy_kj: f64,
    pub pre_exponential: f64,
    pub enthalpy_kj: f64, // ΔH: negative = exothermic
    pub vector: ContinuousHV,
}

impl Reaction {
    /// Calculate rate constant at temperature using Arrhenius
    pub fn rate_constant(&self, temp_k: f64) -> f64 {
        self.pre_exponential * (-self.activation_energy_kj * 1000.0 / (R_GAS * temp_k)).exp()
    }
}

/// A catalyst
#[derive(Debug, Clone)]
pub struct Catalyst {
    pub name: String,
    pub catalyst_type: CatalystType,
    pub activation_energy_reduction: f64, // How much it lowers Ea
    pub selectivity: f64,                 // 0-1
    pub turnover_frequency: f64,          // 1/s
    pub vector: ContinuousHV,
}

/// Chemical Kinetics Encoder
#[derive(Debug, Clone)]
pub struct KineticsEncoder {
    genesis: GenesisSeed,

    // Fundamental concepts
    pub rate: ContinuousHV,
    pub concentration: ContinuousHV,
    pub rate_constant: ContinuousHV,
    pub activation_energy: ContinuousHV,
    pub transition_state: ContinuousHV,
    pub reaction_coordinate: ContinuousHV,

    // Thermodynamic quantities
    pub enthalpy: ContinuousHV,
    pub entropy: ContinuousHV,
    pub gibbs_free_energy: ContinuousHV,
    pub equilibrium: ContinuousHV,

    // Rate laws
    pub zeroth_order: ContinuousHV,
    pub first_order: ContinuousHV,
    pub second_order: ContinuousHV,

    // Reaction types
    pub elementary: ContinuousHV,
    pub reversible: ContinuousHV,
    pub irreversible: ContinuousHV,
    pub chain: ContinuousHV,

    // Chain reaction steps
    pub initiation: ContinuousHV,
    pub propagation: ContinuousHV,
    pub termination: ContinuousHV,

    // Catalysis
    pub catalyst: ContinuousHV,
    pub homogeneous: ContinuousHV,
    pub heterogeneous: ContinuousHV,
    pub enzyme: ContinuousHV,
    pub active_site: ContinuousHV,
    pub turnover: ContinuousHV,

    // Species
    pub reactant: ContinuousHV,
    pub product: ContinuousHV,
    pub intermediate: ContinuousHV,
    pub radical: ContinuousHV,
}

impl KineticsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            rate: genesis.hv("kinetics::rate", PHYSICS_DIM),
            concentration: genesis.hv("kinetics::concentration", PHYSICS_DIM),
            rate_constant: genesis.hv("kinetics::rate_constant", PHYSICS_DIM),
            activation_energy: genesis.hv("kinetics::activation_energy", PHYSICS_DIM),
            transition_state: genesis.hv("kinetics::transition_state", PHYSICS_DIM),
            reaction_coordinate: genesis.hv("kinetics::reaction_coordinate", PHYSICS_DIM),

            enthalpy: genesis.hv("thermo::enthalpy", PHYSICS_DIM),
            entropy: genesis.hv("thermo::entropy", PHYSICS_DIM),
            gibbs_free_energy: genesis.hv("thermo::gibbs", PHYSICS_DIM),
            equilibrium: genesis.hv("thermo::equilibrium", PHYSICS_DIM),

            zeroth_order: genesis.hv("order::zeroth", PHYSICS_DIM),
            first_order: genesis.hv("order::first", PHYSICS_DIM),
            second_order: genesis.hv("order::second", PHYSICS_DIM),

            elementary: genesis.hv("reaction::elementary", PHYSICS_DIM),
            reversible: genesis.hv("reaction::reversible", PHYSICS_DIM),
            irreversible: genesis.hv("reaction::irreversible", PHYSICS_DIM),
            chain: genesis.hv("reaction::chain", PHYSICS_DIM),

            initiation: genesis.hv("chain::initiation", PHYSICS_DIM),
            propagation: genesis.hv("chain::propagation", PHYSICS_DIM),
            termination: genesis.hv("chain::termination", PHYSICS_DIM),

            catalyst: genesis.hv("catalysis::catalyst", PHYSICS_DIM),
            homogeneous: genesis.hv("catalysis::homogeneous", PHYSICS_DIM),
            heterogeneous: genesis.hv("catalysis::heterogeneous", PHYSICS_DIM),
            enzyme: genesis.hv("catalysis::enzyme", PHYSICS_DIM),
            active_site: genesis.hv("catalysis::active_site", PHYSICS_DIM),
            turnover: genesis.hv("catalysis::turnover", PHYSICS_DIM),

            reactant: genesis.hv("species::reactant", PHYSICS_DIM),
            product: genesis.hv("species::product", PHYSICS_DIM),
            intermediate: genesis.hv("species::intermediate", PHYSICS_DIM),
            radical: genesis.hv("species::radical", PHYSICS_DIM),
        }
    }

    // ============================================================
    // RATE LAWS
    // ============================================================

    /// Arrhenius equation: k = A·exp(-Ea/RT)
    pub fn arrhenius_equation(&self) -> ContinuousHV {
        let temperature = self.genesis.hv("thermo::temperature", PHYSICS_DIM);
        self.rate_constant
            .bind(&self.activation_energy)
            .bind(&temperature)
    }

    /// Eyring equation (transition state theory)
    pub fn eyring_equation(&self) -> ContinuousHV {
        self.transition_state
            .bind(&self.rate_constant)
            .bind(&self.gibbs_free_energy)
            .bind(&self.entropy)
    }

    /// Rate law: rate = k[A]^n[B]^m
    pub fn rate_law(&self, order: ReactionOrder) -> ContinuousHV {
        let order_vec = match order {
            ReactionOrder::Zeroth => self.zeroth_order.clone(),
            ReactionOrder::First => self.first_order.clone(),
            ReactionOrder::Second => self.second_order.clone(),
            ReactionOrder::Third => self.second_order.bind(&self.first_order),
            ReactionOrder::Fractional => self.first_order.scale(0.5),
        };

        self.rate
            .bind(&self.rate_constant)
            .bind(&self.concentration)
            .bind(&order_vec)
    }

    // ============================================================
    // REACTIONS
    // ============================================================

    /// Create a reaction
    pub fn create_reaction(
        &self,
        name: &str,
        ea_kj: f64,
        pre_exp: f64,
        delta_h: f64,
        reaction_type: ReactionType,
        order: ReactionOrder,
    ) -> Reaction {
        let type_vec = match reaction_type {
            ReactionType::Elementary => self.elementary.clone(),
            ReactionType::Complex => self.elementary.bind(&self.intermediate),
            ReactionType::Chain => self.chain.clone(),
            ReactionType::Catalytic => self.catalyst.bind(&self.elementary),
            ReactionType::Enzymatic => self.enzyme.clone(),
            ReactionType::Photochemical => self.genesis.hv("reaction::photochemical", PHYSICS_DIM),
            ReactionType::Electrochemical => {
                self.genesis.hv("reaction::electrochemical", PHYSICS_DIM)
            }
        };

        let vector = self
            .rate
            .bind(&type_vec)
            .bind(&self.activation_energy.scale(ea_kj as f32))
            .bind(&self.enthalpy.scale(delta_h as f32));

        Reaction {
            name: name.to_string(),
            reaction_type,
            order,
            activation_energy_kj: ea_kj,
            pre_exponential: pre_exp,
            enthalpy_kj: delta_h,
            vector,
        }
    }

    /// First-order decay: [A] = [A]₀ exp(-kt)
    pub fn first_order_decay(&self, initial_conc: f64, k: f64, time: f64) -> f64 {
        initial_conc * (-k * time).exp()
    }

    /// Half-life for first order: t½ = ln(2)/k
    pub fn half_life_first_order(&self, k: f64) -> f64 {
        std::f64::consts::LN_2 / k
    }

    // ============================================================
    // EQUILIBRIUM
    // ============================================================

    /// Equilibrium constant from Gibbs energy: K = exp(-ΔG°/RT)
    pub fn equilibrium_constant(&self) -> ContinuousHV {
        self.equilibrium
            .bind(&self.gibbs_free_energy)
            .bind(&self.rate)
    }

    /// Derive equilibrium constant from forward/reverse rates
    /// K = k_forward / k_reverse
    pub fn equilibrium_from_rates(&self) -> ContinuousHV {
        let forward = self.rate.bind(&self.reactant);
        let reverse = self.rate.bind(&self.product);
        forward.bind(&reverse).bind(&self.equilibrium)
    }

    // ============================================================
    // CHAIN REACTIONS
    // ============================================================

    /// Chain reaction mechanism
    pub fn chain_mechanism(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[&self.initiation, &self.propagation, &self.termination])
            .bind(&self.chain)
            .bind(&self.radical)
    }

    /// Chain length = rate of propagation / rate of initiation
    pub fn chain_length(&self, r_prop: f64, r_init: f64) -> f64 {
        r_prop / r_init
    }

    // ============================================================
    // CATALYSIS
    // ============================================================

    /// Create a catalyst
    pub fn create_catalyst(
        &self,
        name: &str,
        catalyst_type: CatalystType,
        ea_reduction: f64,
        selectivity: f64,
        tof: f64,
    ) -> Catalyst {
        let type_vec = match catalyst_type {
            CatalystType::Homogeneous => self.homogeneous.clone(),
            CatalystType::Heterogeneous => self.heterogeneous.clone(),
            CatalystType::Enzyme => self.enzyme.clone(),
            CatalystType::Acid => self.genesis.hv("acid::catalyst", PHYSICS_DIM),
            CatalystType::Base => self.genesis.hv("base::catalyst", PHYSICS_DIM),
            CatalystType::Metal => self.genesis.hv("metal::catalyst", PHYSICS_DIM),
            CatalystType::Photocatalyst => self.genesis.hv("photo::catalyst", PHYSICS_DIM),
        };

        let vector = self
            .catalyst
            .bind(&type_vec)
            .bind(&self.active_site)
            .bind(&self.turnover.scale(tof as f32));

        Catalyst {
            name: name.to_string(),
            catalyst_type,
            activation_energy_reduction: ea_reduction,
            selectivity,
            turnover_frequency: tof,
            vector,
        }
    }

    /// Catalyst effect: k_cat = A·exp(-(Ea - ΔEa)/RT)
    pub fn catalyzed_rate_constant(
        &self,
        reaction: &Reaction,
        catalyst: &Catalyst,
        temp_k: f64,
    ) -> f64 {
        let effective_ea = reaction.activation_energy_kj - catalyst.activation_energy_reduction;
        reaction.pre_exponential * (-effective_ea * 1000.0 / (R_GAS * temp_k)).exp()
    }

    /// Michaelis-Menten kinetics: v = Vmax[S]/(Km + [S])
    pub fn michaelis_menten(&self) -> ContinuousHV {
        let substrate = self.genesis.hv("enzyme::substrate", PHYSICS_DIM);
        let vmax = self.genesis.hv("enzyme::vmax", PHYSICS_DIM);
        let km = self.genesis.hv("enzyme::km", PHYSICS_DIM);

        self.enzyme
            .bind(&self.rate)
            .bind(&substrate)
            .bind(&vmax)
            .bind(&km)
    }

    /// Michaelis-Menten rate calculation
    pub fn mm_rate(&self, substrate_conc: f64, vmax: f64, km: f64) -> f64 {
        vmax * substrate_conc / (km + substrate_conc)
    }

    /// Langmuir-Hinshelwood kinetics (heterogeneous catalysis)
    pub fn langmuir_hinshelwood(&self) -> ContinuousHV {
        let surface = self.genesis.hv("surface::coverage", PHYSICS_DIM);
        let adsorption = self.genesis.hv("surface::adsorption", PHYSICS_DIM);

        self.heterogeneous
            .bind(&self.rate)
            .bind(&surface)
            .bind(&adsorption)
    }

    // ============================================================
    // TEMPERATURE DEPENDENCE
    // ============================================================

    /// Calculate rate constant ratio at two temperatures
    pub fn rate_constant_ratio(&self, ea_kj: f64, t1: f64, t2: f64) -> f64 {
        let factor = ea_kj * 1000.0 / R_GAS * (1.0 / t1 - 1.0 / t2);
        factor.exp()
    }

    /// Van't Hoff equation: d(ln K)/dT = ΔH°/RT²
    pub fn vant_hoff(&self) -> ContinuousHV {
        let temperature = self.genesis.hv("thermo::temperature", PHYSICS_DIM);
        self.equilibrium.bind(&self.enthalpy).bind(&temperature)
    }

    /// Activation energy from rate constants at two temperatures
    pub fn activation_energy_from_rates(&self, k1: f64, k2: f64, t1: f64, t2: f64) -> f64 {
        R_GAS * (k2 / k1).ln() / (1.0 / t1 - 1.0 / t2) / 1000.0 // kJ/mol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> KineticsEncoder {
        let genesis = GenesisSeed::from_phrase("kinetics test");
        KineticsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.rate.norm() > 0.0);
        assert!(encoder.catalyst.norm() > 0.0);
    }

    #[test]
    fn test_arrhenius_rate() {
        let encoder = setup();

        let reaction = encoder.create_reaction(
            "Test",
            50.0,   // Ea = 50 kJ/mol
            1e13,   // A = 10^13 s^-1
            -100.0, // ΔH = -100 kJ/mol (exothermic)
            ReactionType::Elementary,
            ReactionOrder::First,
        );

        let k_300 = reaction.rate_constant(300.0);
        let k_400 = reaction.rate_constant(400.0);

        // Higher temperature → higher rate
        assert!(k_400 > k_300);
    }

    #[test]
    fn test_catalyst_effect() {
        let encoder = setup();

        let reaction = encoder.create_reaction(
            "Uncatalyzed",
            80.0,
            1e12,
            -50.0,
            ReactionType::Elementary,
            ReactionOrder::Second,
        );

        let catalyst = encoder.create_catalyst(
            "Pt",
            CatalystType::Heterogeneous,
            30.0, // Reduces Ea by 30 kJ/mol
            0.95,
            100.0, // TOF = 100/s
        );

        let k_uncat = reaction.rate_constant(300.0);
        let k_cat = encoder.catalyzed_rate_constant(&reaction, &catalyst, 300.0);

        // Catalyst should increase rate
        assert!(k_cat > k_uncat);
    }

    #[test]
    fn test_first_order_decay() {
        let encoder = setup();

        let conc = encoder.first_order_decay(1.0, 0.1, 10.0);

        // [A] = exp(-1) ≈ 0.368
        assert!((conc - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_half_life() {
        let encoder = setup();

        let t_half = encoder.half_life_first_order(0.1);

        // t½ = ln(2)/0.1 ≈ 6.93 s
        assert!((t_half - 6.93).abs() < 0.1);
    }

    #[test]
    fn test_michaelis_menten() {
        let encoder = setup();

        // At [S] = Km, v = Vmax/2
        let rate = encoder.mm_rate(1.0, 10.0, 1.0);
        assert!((rate - 5.0).abs() < 0.01);

        // At high [S], v → Vmax
        let rate_high = encoder.mm_rate(1000.0, 10.0, 1.0);
        assert!((rate_high - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_chain_mechanism() {
        let encoder = setup();

        let chain = encoder.chain_mechanism();

        // Should encode all chain steps
        assert!(chain.norm() > 0.0);
    }

    #[test]
    fn test_activation_energy_from_rates() {
        let encoder = setup();

        // k1 = 0.1 at 300K, k2 = 1.0 at 400K
        let ea = encoder.activation_energy_from_rates(0.1, 1.0, 300.0, 400.0);

        // Ea should be positive
        assert!(ea > 0.0);
        // Rough calculation gives ~27 kJ/mol
        assert!(ea > 20.0 && ea < 40.0);
    }

    #[test]
    fn test_rate_laws_distinct() {
        let encoder = setup();

        let zero = encoder.rate_law(ReactionOrder::Zeroth);
        let first = encoder.rate_law(ReactionOrder::First);
        let second = encoder.rate_law(ReactionOrder::Second);

        // Different orders should give different vectors
        assert!(zero.similarity(&first) < 0.95);
        assert!(first.similarity(&second) < 0.95);
    }
}
