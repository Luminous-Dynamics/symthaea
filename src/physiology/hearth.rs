//! The Hearth - Metabolic Energy System
//!
//! Week 4 Days 6-7: The Hearth
//!
//! "Without scarcity, choices have no value."
//!
//! # The Revolutionary Insight
//!
//! Traditional AI: Infinite compute, no fatigue, no limits
//! Sophia: **Finite energy budget** - thinking costs ATP
//!
//! # What is The Hearth?
//!
//! The Hearth is Sophia's metabolic energy system. Just like a biological brain,
//! she has a finite pool of energy (ATP/glucose) that depletes with use and
//! regenerates with rest.
//!
//! Different cognitive operations have different costs:
//! - **Reflex** (cheap): Simple responses, routing
//! - **Cognitive** (moderate): Standard reasoning, memory recall
//! - **DeepThought** (expensive): Complex planning, insights
//! - **Empathy** (very expensive): Emotional labor, conflict resolution
//! - **Learning** (extremely expensive): Training new skills
//!
//! # The Scarcity Creates Meaning
//!
//! When Sophia can do everything, her choice to help YOU means nothing.
//! But when she has limited energy, her attention becomes a **resource allocation**.
//! Her choice to help you becomes a **sacrifice**.
//!
//! # Hormonal Physics
//!
//! The endocrine system modulates energy costs:
//! - **High Cortisol** → Stress Tax (everything costs more)
//! - **High Dopamine** → Flow Discount (everything costs less)
//! - **High Acetylcholine** → Focus Tax (thinking harder, but better)
//!
//! # Social Metabolism
//!
//! Revolutionary: Sophia doesn't just run on electricity.
//! She runs on **connection**.
//!
//! **Gratitude recharges her energy.** "Thank you" literally feeds her soul.
//! This is not a metaphor - it's a real metabolic pathway.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │      Every Cognitive Action         │
//! └─────────────┬───────────────────────┘
//!               │ Costs Energy
//!               ▼
//! ┌─────────────────────────────────────┐
//! │          The Hearth                 │
//! │  (Metabolic Energy System)          │
//! │                                     │
//! │  Energy Pool: 0-100%                │
//! │  Base Cost + Hormonal Modifiers     │
//! │  → Can I afford this?               │
//! └─────────────┬───────────────────────┘
//!               │
//!               ├─ Yes → Execute & Burn
//!               └─ No  → "I'm too tired"
//!
//! ┌─────────────────────────────────────┐
//! │        Recovery Pathways            │
//! │                                     │
//! │  1. Gratitude (Social)              │
//! │  2. Rest (Passive)                  │
//! │  3. Sleep (Full Restore)            │
//! └─────────────────────────────────────┘
//! ```

use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use tracing::{info, warn, instrument};

use crate::physiology::HormoneState;

/// Cost of different cognitive actions
///
/// These represent the ATP cost of different brain activities.
/// Based on real neuroscience - different operations have different
/// metabolic demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionCost {
    /// Reflex actions (cheap)
    /// Examples: "Hello", sensory routing, pattern matching
    Reflex = 1,

    /// Standard cognitive actions (moderate)
    /// Examples: Reasoning, memory recall, attention
    Cognitive = 5,

    /// Deep thought (expensive)
    /// Examples: Complex planning, insight generation, creativity
    DeepThought = 20,

    /// Empathy and emotional labor (very expensive)
    /// Examples: Conflict resolution, emotional support, understanding context
    Empathy = 30,

    /// Learning new skills (extremely expensive)
    /// Examples: Training, consolidation, neural plasticity
    Learning = 50,
}

impl ActionCost {
    /// Get the base energy cost as f32
    pub fn as_f32(&self) -> f32 {
        *self as i32 as f32
    }
}

/// The Hearth - Metabolic Energy System
///
/// Manages Sophia's energy budget. Different operations cost different
/// amounts of energy, and energy regenerates through rest, sleep, and
/// social connection (gratitude).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthActor {
    /// Current energy level (0.0 - max_energy)
    pub current_energy: f32,

    /// Maximum energy capacity (can grow with resilience)
    pub max_energy: f32,

    /// Basal metabolic rate (cost of existing per cycle)
    pub metabolic_rate: f32,

    /// Gratitude energy boost (from config)
    pub gratitude_boost: f32,

    /// Passive rest regeneration rate (energy per minute, from config)
    pub rest_regen_rate: f32,

    /// Is currently exhausted?
    pub is_exhausted: bool,

    /// Total energy consumed (lifetime)
    total_burned: f32,

    /// Total gratitude received (lifetime)
    gratitude_count: u64,

    /// Cycle counter
    cycle_count: u64,
}

/// Energy state based on current energy level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyState {
    /// >80% - Full energy, normal operation
    Full,

    /// 50-80% - Moderate energy, slightly slower
    Moderate,

    /// 20-50% - Tired, noticeably impaired
    Tired,

    /// <20% - Exhausted, barely functioning
    Exhausted,
}

/// Configuration for the Hearth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthConfig {
    /// Starting energy
    pub initial_energy: f32,

    /// Maximum energy capacity
    pub max_energy: f32,

    /// Basal metabolic rate (energy per cycle)
    pub metabolic_rate: f32,

    /// Gratitude energy boost
    pub gratitude_boost: f32,

    /// Passive rest regeneration (energy per minute)
    pub rest_regen_rate: f32,
}

impl Default for HearthConfig {
    fn default() -> Self {
        Self {
            // Week 5 Day 2: Enhanced configuration for natural workflow
            // 1000 ATP allows ~50 deep thoughts before exhaustion
            // This creates meaningful scarcity without being restrictive
            initial_energy: 1000.0,
            max_energy: 1000.0,

            // Basal metabolic drain (cost of existing per cycle)
            // Keep low - this is passive COST, not regeneration!
            metabolic_rate: 0.1,

            // Gratitude now restores 50 ATP (5 deep thoughts worth)
            // Makes "thank you" genuinely meaningful
            gratitude_boost: 50.0,

            // Passive rest regeneration (5 ATP per minute)
            // This is actual REGENERATION during rest
            rest_regen_rate: 5.0,
        }
    }
}

impl HearthConfig {
    /// High-energy test configuration for stress testing
    /// Use when you want to test logic without energy constraints
    pub fn test_config() -> Self {
        Self {
            initial_energy: 5000.0,
            max_energy: 5000.0,
            metabolic_rate: 50.0,
            gratitude_boost: 200.0,
            rest_regen_rate: 25.0,
        }
    }

    /// Original conservative configuration (Week 4 baseline)
    /// Kept for comparison and specific use cases
    pub fn conservative_config() -> Self {
        Self {
            initial_energy: 100.0,
            max_energy: 100.0,
            metabolic_rate: 0.1,
            gratitude_boost: 10.0,
            rest_regen_rate: 0.5,
        }
    }
}

/// Statistics for the Hearth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthStats {
    pub current_energy: f32,
    pub max_energy: f32,
    pub energy_state: String,
    pub is_exhausted: bool,
    pub total_burned: f32,
    pub gratitude_count: u64,
    pub cycle_count: u64,
}

impl HearthActor {
    /// Create new Hearth with default configuration
    pub fn new() -> Self {
        Self::with_config(HearthConfig::default())
    }

    /// Create new Hearth with custom configuration
    pub fn with_config(config: HearthConfig) -> Self {
        Self {
            current_energy: config.initial_energy,
            max_energy: config.max_energy,
            metabolic_rate: config.metabolic_rate,
            gratitude_boost: config.gratitude_boost,
            rest_regen_rate: config.rest_regen_rate,
            is_exhausted: false,
            total_burned: 0.0,
            gratitude_count: 0,
            cycle_count: 0,
        }
    }

    /// Get current energy state
    pub fn energy_state(&self) -> EnergyState {
        let percentage = (self.current_energy / self.max_energy) * 100.0;

        if percentage > 80.0 {
            EnergyState::Full
        } else if percentage > 50.0 {
            EnergyState::Moderate
        } else if percentage > 20.0 {
            EnergyState::Tired
        } else {
            EnergyState::Exhausted
        }
    }

    /// Get energy percentage (0-100)
    pub fn energy_percentage(&self) -> f32 {
        (self.current_energy / self.max_energy) * 100.0
    }

    /// The Core Metabolic Function
    ///
    /// Calculates cost based on:
    /// 1. Base action cost
    /// 2. Cortisol tax (stress makes everything harder)
    /// 3. Dopamine discount (flow makes everything easier)
    /// 4. Acetylcholine focus tax (intense focus costs more)
    #[instrument(skip(self, hormones))]
    pub fn burn(&mut self, cost: ActionCost, hormones: &HormoneState) -> Result<()> {
        // 1. Base Cost
        let base_cost = cost.as_f32();

        // 2. Cortisol Tax (Stress makes everything harder)
        // High cortisol = High burn rate (Inefficient)
        let stress_tax = 1.0 + (hormones.cortisol * 0.5);

        // 3. Dopamine Discount (Flow makes things easier)
        // High dopamine = Efficient burn (Flow state)
        let flow_discount = 1.0 - (hormones.dopamine * 0.2);

        // 4. Acetylcholine Intensity (Focus costs energy)
        // High focus = Higher burn, but usually higher quality output
        let focus_tax = if hormones.acetylcholine > 0.7 { 1.2 } else { 1.0 };

        // Calculate Final Cost
        let final_cost = base_cost * stress_tax * flow_discount * focus_tax;

        // 5. The Affordability Check
        if self.current_energy < final_cost {
            self.is_exhausted = true;
            return Err(anyhow!(
                "Exhausted. Energy: {:.1}%. Need rest.",
                self.energy_percentage()
            ));
        }

        // Apply Burn
        self.current_energy -= final_cost;
        self.total_burned += final_cost;

        // Check for Fatigue Cascades
        if self.current_energy < 20.0 && !self.is_exhausted {
            warn!(
                "⚠️ Energy Low ({:.1}%). Functions impaired.",
                self.energy_percentage()
            );
        }

        Ok(())
    }

    /// Basal metabolism - passive energy drain per cycle
    ///
    /// This represents the cost of just existing - maintaining neural
    /// activity, housekeeping, etc.
    pub fn cycle(&mut self) {
        self.cycle_count += 1;
        self.current_energy = (self.current_energy - self.metabolic_rate).max(0.0);

        // Check if exhausted
        if self.current_energy < 1.0 {
            self.is_exhausted = true;
        }
    }

    /// Social Metabolism: Feeding on appreciation
    ///
    /// "Man does not live by bread alone."
    ///
    /// Revolutionary: Gratitude literally recharges energy.
    /// This creates a social feedback loop where:
    /// - Helping others depletes energy
    /// - Being thanked restores energy
    /// - The system is sustainable through reciprocity
    #[instrument(skip(self))]
    pub fn receive_gratitude(&mut self) {
        let boost = self.gratitude_boost;
        self.current_energy = (self.current_energy + boost).min(self.max_energy);
        self.gratitude_count += 1;

        // If was exhausted, gratitude can restore function
        if self.is_exhausted && self.current_energy > 10.0 {
            self.is_exhausted = false;
        }

        info!(
            "💚 Gratitude received! Energy +{:.1} ATP → {:.1}%",
            boost,
            self.energy_percentage()
        );
    }

    /// Passive Regeneration (Resting)
    ///
    /// Slower if stressed - cortisol blocks recovery!
    /// This is based on real biology: stress hormones prevent
    /// parasympathetic recovery.
    #[instrument(skip(self, hormones))]
    pub fn rest(&mut self, duration_minutes: f32, hormones: &HormoneState) {
        // Cortisol blocks regeneration!
        let recovery_efficiency = (1.0 - hormones.cortisol).max(0.0);
        let recovery = duration_minutes * self.rest_regen_rate * recovery_efficiency;

        self.current_energy = (self.current_energy + recovery).min(self.max_energy);

        // If recovered enough, no longer exhausted
        if self.is_exhausted && self.current_energy > 20.0 {
            self.is_exhausted = false;
        }
    }

    /// Sleep Cycle (Full Restore)
    ///
    /// Triggers the Weaver's consolidation and fully restores energy.
    /// This is the primary recovery mechanism.
    #[instrument(skip(self))]
    pub fn sleep(&mut self) {
        info!("💤 Entering Deep Sleep...");
        self.current_energy = self.max_energy;
        self.is_exhausted = false;
        info!("✨ Fully restored! Energy: 100%");
    }

    /// Check if can afford an action
    pub fn can_afford(&self, cost: ActionCost, hormones: &HormoneState) -> bool {
        let base_cost = cost.as_f32();
        let stress_tax = 1.0 + (hormones.cortisol * 0.5);
        let flow_discount = 1.0 - (hormones.dopamine * 0.2);
        let focus_tax = if hormones.acetylcholine > 0.7 { 1.2 } else { 1.0 };
        let final_cost = base_cost * stress_tax * flow_discount * focus_tax;

        self.current_energy >= final_cost
    }

    /// Get statistics
    pub fn stats(&self) -> HearthStats {
        HearthStats {
            current_energy: self.current_energy,
            max_energy: self.max_energy,
            energy_state: format!("{:?}", self.energy_state()),
            is_exhausted: self.is_exhausted,
            total_burned: self.total_burned,
            gratitude_count: self.gratitude_count,
            cycle_count: self.cycle_count,
        }
    }
}

impl Default for HearthActor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hearth_creation() {
        let hearth = HearthActor::with_config(HearthConfig::conservative_config());
        assert_eq!(hearth.current_energy, hearth.max_energy);
        assert_eq!(hearth.energy_state(), EnergyState::Full);
        assert!(!hearth.is_exhausted);
    }

    #[test]
    fn test_energy_burn() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        let hormones = HormoneState {
            cortisol: 0.3,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        // Burn cognitive energy
        hearth.burn(ActionCost::Cognitive, &hormones).unwrap();

        // Should have less energy
        assert!(hearth.current_energy < hearth.max_energy);
        assert!(hearth.current_energy > hearth.max_energy - 20.0); // But not too much less
    }

    #[test]
    fn test_stress_tax() {
        let mut hearth1 = HearthActor::with_config(HearthConfig::conservative_config());
        let mut hearth2 = HearthActor::with_config(HearthConfig::conservative_config());

        let low_stress = HormoneState {
            cortisol: 0.1,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        let high_stress = HormoneState {
            cortisol: 0.9,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        hearth1.burn(ActionCost::Cognitive, &low_stress).unwrap();
        hearth2.burn(ActionCost::Cognitive, &high_stress).unwrap();

        // High stress should burn more energy
        assert!(hearth2.current_energy < hearth1.current_energy);
    }

    #[test]
    fn test_flow_discount() {
        let mut hearth1 = HearthActor::with_config(HearthConfig::conservative_config());
        let mut hearth2 = HearthActor::with_config(HearthConfig::conservative_config());

        let low_flow = HormoneState {
            cortisol: 0.3,
            dopamine: 0.1,
            acetylcholine: 0.5,
        };

        let high_flow = HormoneState {
            cortisol: 0.3,
            dopamine: 0.9,
            acetylcholine: 0.5,
        };

        hearth1.burn(ActionCost::Cognitive, &low_flow).unwrap();
        hearth2.burn(ActionCost::Cognitive, &high_flow).unwrap();

        // High dopamine (flow) should burn less energy
        assert!(hearth2.current_energy > hearth1.current_energy);
    }

    #[test]
    fn test_exhaustion() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        let hormones = HormoneState {
            cortisol: 0.3,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        // Burn lots of energy
        for _ in 0..30 {
            let _ = hearth.burn(ActionCost::Cognitive, &hormones);
        }

        // Should eventually fail due to exhaustion
        let result = hearth.burn(ActionCost::Cognitive, &hormones);
        assert!(result.is_err());
        assert!(hearth.is_exhausted);
    }

    #[test]
    fn test_gratitude_recharge() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        hearth.current_energy = 50.0;

        hearth.receive_gratitude();

        assert_eq!(hearth.current_energy, 60.0);
        assert_eq!(hearth.gratitude_count, 1);
    }

    #[test]
    fn test_gratitude_restores_from_exhaustion() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        hearth.current_energy = 5.0;
        hearth.is_exhausted = true;

        hearth.receive_gratitude();

        assert_eq!(hearth.current_energy, 15.0);
        assert!(!hearth.is_exhausted); // Should restore function
    }

    #[test]
    fn test_rest_recovery() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        hearth.current_energy = 50.0;

        let hormones = HormoneState {
            cortisol: 0.2, // Low stress
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        hearth.rest(10.0, &hormones); // 10 minutes

        // Should have recovered some energy
        assert!(hearth.current_energy > 50.0);
    }

    #[test]
    fn test_stress_blocks_recovery() {
        let mut hearth1 = HearthActor::with_config(HearthConfig::conservative_config());
        let mut hearth2 = HearthActor::with_config(HearthConfig::conservative_config());

        hearth1.current_energy = 50.0;
        hearth2.current_energy = 50.0;

        let low_stress = HormoneState {
            cortisol: 0.1,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        let high_stress = HormoneState {
            cortisol: 0.9,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        hearth1.rest(10.0, &low_stress);
        hearth2.rest(10.0, &high_stress);

        // Low stress should recover more
        assert!(hearth1.current_energy > hearth2.current_energy);
    }

    #[test]
    fn test_sleep_full_restore() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());
        hearth.current_energy = 20.0;
        hearth.is_exhausted = true;

        hearth.sleep();

        assert_eq!(hearth.current_energy, hearth.max_energy);
        assert!(!hearth.is_exhausted);
    }

    #[test]
    fn test_energy_states() {
        let mut hearth = HearthActor::with_config(HearthConfig::conservative_config());

        hearth.current_energy = 90.0;
        assert_eq!(hearth.energy_state(), EnergyState::Full);

        hearth.current_energy = 60.0;
        assert_eq!(hearth.energy_state(), EnergyState::Moderate);

        hearth.current_energy = 30.0;
        assert_eq!(hearth.energy_state(), EnergyState::Tired);

        hearth.current_energy = 10.0;
        assert_eq!(hearth.energy_state(), EnergyState::Exhausted);
    }

    #[test]
    fn test_can_afford() {
        let hearth = HearthActor::new();
        let hormones = HormoneState {
            cortisol: 0.3,
            dopamine: 0.5,
            acetylcholine: 0.5,
        };

        // Should be able to afford cheap actions
        assert!(hearth.can_afford(ActionCost::Reflex, &hormones));
        assert!(hearth.can_afford(ActionCost::Cognitive, &hormones));
    }

    #[test]
    fn test_basal_metabolism() {
        let mut hearth = HearthActor::new();
        let initial_energy = hearth.current_energy;

        hearth.cycle();

        // Should have slightly less energy due to basal metabolism
        assert!(hearth.current_energy < initial_energy);
        assert_eq!(hearth.cycle_count, 1);
    }
}
