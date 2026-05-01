// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Colony Projects: multi-tick construction and development efforts.
//!
//! The missing heartbeat of civilization simulation. Every colony is defined
//! by what it chose to build. Projects consume labor and materials over months
//! to years, creating the fundamental governance decision: what do we build next?
//!
//! # Design
//!
//! A project has:
//! - **Blueprint**: what it builds (reactor, greenhouse, habitat module, etc.)
//! - **Duration**: how many ticks to complete
//! - **Labor cost**: engineering-hours per tick (diverted from other work)
//! - **Material cost**: resources consumed over the project's lifetime
//! - **Effect**: what changes when complete (power increase, food production, etc.)
//!
//! Projects compete for scarce labor and materials. Starting a reactor means
//! NOT starting a greenhouse. This creates the strategic tension that drives
//! governance decisions.

use serde::{Deserialize, Serialize};

/// A project under construction in a colony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveProject {
    /// What is being built.
    pub blueprint: ProjectBlueprint,
    /// Ticks remaining until completion.
    pub ticks_remaining: u32,
    /// Total ticks for the project (for progress calculation).
    pub total_ticks: u32,
    /// Engineering labor-hours consumed per tick.
    pub labor_per_tick: f64,
    /// Material units consumed per tick.
    pub materials_per_tick: f64,
    /// Whether the project is stalled (insufficient resources).
    pub stalled: bool,
    /// Ticks spent stalled (delays completion).
    pub stall_ticks: u32,
}

impl ActiveProject {
    pub fn progress_fraction(&self) -> f64 {
        1.0 - (self.ticks_remaining as f64 / self.total_ticks.max(1) as f64)
    }

    pub fn is_complete(&self) -> bool {
        self.ticks_remaining == 0
    }
}

/// What a project builds. Each blueprint defines costs and effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectBlueprint {
    /// Fission reactor module. 100 kW power. 24 months construction.
    FissionReactor,
    /// Greenhouse module. Feeds 50 people. 18 months construction.
    GreenhouseModule,
    /// Habitat expansion. +500 max population. 36 months construction.
    HabitatExpansion,
    /// Medical facility. Unlocks surgery + pharmacy. 12 months.
    MedicalFacility,
    /// Communications array. Reduces latency penalty. 6 months.
    CommsArray,
    /// Fabrication workshop. Enables local manufacturing. 24 months.
    FabricationWorkshop,
    /// Launch pad (for interplanetary missions). 36 months.
    LaunchPad,
    /// Centrifuge habitat (enables reproduction at low-g). 48 months.
    CentrifugeHabitat,
    /// Radiation shelter (reduces radiation damage). 12 months.
    RadiationShelter,
    /// Water extraction plant. 10x water production. 18 months.
    WaterExtractionPlant,
    /// Exploration vehicle (submersible, rover, drone). 12 months.
    ExplorationVehicle,
}

impl ProjectBlueprint {
    /// Duration in ticks (months).
    pub fn duration(&self) -> u32 {
        match self {
            Self::FissionReactor => 24,
            Self::GreenhouseModule => 18,
            Self::HabitatExpansion => 36,
            Self::MedicalFacility => 12,
            Self::CommsArray => 6,
            Self::FabricationWorkshop => 24,
            Self::LaunchPad => 36,
            Self::CentrifugeHabitat => 48,
            Self::RadiationShelter => 12,
            Self::WaterExtractionPlant => 18,
            Self::ExplorationVehicle => 12,
        }
    }

    /// Engineering labor-hours per tick required.
    pub fn labor_per_tick(&self) -> f64 {
        match self {
            Self::FissionReactor => 500.0,
            Self::GreenhouseModule => 200.0,
            Self::HabitatExpansion => 800.0,
            Self::MedicalFacility => 150.0,
            Self::CommsArray => 100.0,
            Self::FabricationWorkshop => 400.0,
            Self::LaunchPad => 600.0,
            Self::CentrifugeHabitat => 1000.0,
            Self::RadiationShelter => 200.0,
            Self::WaterExtractionPlant => 300.0,
            Self::ExplorationVehicle => 150.0,
        }
    }

    /// Material units consumed per tick.
    pub fn materials_per_tick(&self) -> f64 {
        match self {
            Self::FissionReactor => 15.0,
            Self::GreenhouseModule => 8.0,
            Self::HabitatExpansion => 20.0,
            Self::MedicalFacility => 5.0,
            Self::CommsArray => 3.0,
            Self::FabricationWorkshop => 12.0,
            Self::LaunchPad => 18.0,
            Self::CentrifugeHabitat => 25.0,
            Self::RadiationShelter => 6.0,
            Self::WaterExtractionPlant => 10.0,
            Self::ExplorationVehicle => 4.0,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::FissionReactor => "Fission Reactor",
            Self::GreenhouseModule => "Greenhouse Module",
            Self::HabitatExpansion => "Habitat Expansion",
            Self::MedicalFacility => "Medical Facility",
            Self::CommsArray => "Communications Array",
            Self::FabricationWorkshop => "Fabrication Workshop",
            Self::LaunchPad => "Launch Pad",
            Self::CentrifugeHabitat => "Centrifuge Habitat",
            Self::RadiationShelter => "Radiation Shelter",
            Self::WaterExtractionPlant => "Water Extraction Plant",
            Self::ExplorationVehicle => "Exploration Vehicle",
        }
    }
}

/// Colony project manager — decides what to build and tracks progress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectManager {
    /// Currently active projects (typically 1-3 at most).
    pub active: Vec<ActiveProject>,
    /// Completed projects (historical record).
    pub completed: Vec<ProjectBlueprint>,
    /// Projects queued for future start.
    pub queue: Vec<ProjectBlueprint>,
    /// Total labor-hours currently committed to projects.
    pub committed_labor: f64,
}

impl ProjectManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new project with realistic variance.
    /// Duration varies ±30% (Flyvbjerg 2002: construction overruns average 20-50%).
    pub fn start_project_with_variance(
        &mut self,
        blueprint: ProjectBlueprint,
        rng: &mut crate::stochastic::StochasticEngine,
    ) {
        // Duration: base ± 30% gaussian
        let variance = rng.next_gaussian(1.0, 0.15).clamp(0.7, 1.5);
        let actual_duration = ((blueprint.duration() as f64 * variance) as u32).max(1);
        self.active.push(ActiveProject {
            blueprint,
            ticks_remaining: actual_duration,
            total_ticks: actual_duration,
            labor_per_tick: blueprint.labor_per_tick(),
            materials_per_tick: blueprint.materials_per_tick(),
            stalled: false,
            stall_ticks: 0,
        });
    }

    /// Start a new project (deterministic duration, for testing).
    pub fn start_project(&mut self, blueprint: ProjectBlueprint) {
        self.active.push(ActiveProject {
            blueprint,
            ticks_remaining: blueprint.duration(),
            total_ticks: blueprint.duration(),
            labor_per_tick: blueprint.labor_per_tick(),
            materials_per_tick: blueprint.materials_per_tick(),
            stalled: false,
            stall_ticks: 0,
        });
    }

    /// Advance all active projects by one tick with realistic variance.
    /// Includes 3% setback chance and 0.5% critical failure per tick.
    pub fn tick_with_variance(
        &mut self,
        available_labor: f64,
        available_materials: f64,
        rng: &mut crate::stochastic::StochasticEngine,
    ) -> (Vec<ProjectBlueprint>, f64, f64, Vec<String>) {
        let mut events = Vec::new();
        let (completed, labor, materials) = self.tick_inner(available_labor, available_materials);

        // Setbacks and failures for active projects
        let mut failed_indices = Vec::new();
        for (i, project) in self.active.iter_mut().enumerate() {
            if project.stalled {
                continue;
            }
            // 3% chance of setback: adds 1-3 months
            if rng.bernoulli(0.03) {
                let delay = (rng.next_f64() * 3.0).ceil() as u32;
                project.ticks_remaining += delay;
                project.total_ticks += delay;
                events.push(format!(
                    "{} setback: +{} months delay",
                    project.blueprint.name(),
                    delay
                ));
            }
            // 0.5% chance of critical failure: project abandoned
            if rng.bernoulli(0.005) {
                events.push(format!(
                    "{} CRITICAL FAILURE: project abandoned, materials lost",
                    project.blueprint.name()
                ));
                failed_indices.push(i);
            }
        }
        // Remove failed projects (reverse order to preserve indices)
        for &i in failed_indices.iter().rev() {
            self.active.remove(i);
        }

        (completed, labor, materials, events)
    }

    /// Advance all active projects by one tick (deterministic, no variance).
    pub fn tick(
        &mut self,
        available_labor: f64,
        available_materials: f64,
    ) -> (Vec<ProjectBlueprint>, f64, f64) {
        self.tick_inner(available_labor, available_materials)
    }

    fn tick_inner(
        &mut self,
        available_labor: f64,
        available_materials: f64,
    ) -> (Vec<ProjectBlueprint>, f64, f64) {
        let mut completed = Vec::new();
        let mut total_labor_used = 0.0;
        let mut total_materials_used = 0.0;
        let mut remaining_labor = available_labor;
        let mut remaining_materials = available_materials;

        for project in &mut self.active {
            if remaining_labor >= project.labor_per_tick
                && remaining_materials >= project.materials_per_tick
            {
                // Project advances
                project.stalled = false;
                project.ticks_remaining = project.ticks_remaining.saturating_sub(1);
                remaining_labor -= project.labor_per_tick;
                remaining_materials -= project.materials_per_tick;
                total_labor_used += project.labor_per_tick;
                total_materials_used += project.materials_per_tick;
            } else {
                // Project stalls
                project.stalled = true;
                project.stall_ticks += 1;
            }
        }

        // Collect completed projects
        self.active.retain(|p| {
            if p.is_complete() {
                completed.push(p.blueprint);
                false
            } else {
                true
            }
        });
        self.completed.extend(&completed);

        self.committed_labor = total_labor_used;

        // Start queued projects if capacity available
        while let Some(&next) = self.queue.first() {
            if remaining_labor >= next.labor_per_tick()
                && remaining_materials >= next.materials_per_tick()
            {
                self.queue.remove(0);
                self.start_project(next);
            } else {
                break;
            }
        }

        (completed, total_labor_used, total_materials_used)
    }

    /// Number of active (non-stalled) projects.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|p| !p.stalled).count()
    }

    /// Has a specific type of project been completed?
    pub fn has_completed(&self, blueprint: ProjectBlueprint) -> bool {
        self.completed.contains(&blueprint)
    }
}

/// Decide what the colony should build next based on its needs.
/// This is the AI governor's decision logic.
pub fn prioritize_projects(
    population: usize,
    power_deficit: f64, // power_demand - power_generation (positive = need more)
    food_fraction: f64, // food stock / capacity
    _max_population: usize,
    pop_near_capacity: bool, // population > 80% of max
    has_medical: bool,
    has_fabrication: bool,
    location: &str,
    completed: &[ProjectBlueprint],
) -> Vec<ProjectBlueprint> {
    let mut priorities = Vec::new();

    // Power crisis: build reactor first
    if power_deficit > 50.0 && !completed.contains(&ProjectBlueprint::FissionReactor) {
        priorities.push(ProjectBlueprint::FissionReactor);
    }

    // Food crisis: build greenhouse
    if food_fraction < 0.3 {
        priorities.push(ProjectBlueprint::GreenhouseModule);
    }

    // Population pressure: expand habitat
    if pop_near_capacity && population > 100 {
        priorities.push(ProjectBlueprint::HabitatExpansion);
    }

    // Medical: critical for colony health
    if !has_medical && population > 50 {
        priorities.push(ProjectBlueprint::MedicalFacility);
    }

    // Fabrication: enables self-sufficiency
    if !has_fabrication && population > 200 {
        priorities.push(ProjectBlueprint::FabricationWorkshop);
    }

    // Low-g colonies: centrifuge for reproduction
    if matches!(location, "Moon" | "Europa" | "Titan")
        && !completed.contains(&ProjectBlueprint::CentrifugeHabitat)
        && population > 100
    {
        priorities.push(ProjectBlueprint::CentrifugeHabitat);
    }

    // Radiation shelter for Europa
    if location == "Europa" && !completed.contains(&ProjectBlueprint::RadiationShelter) {
        priorities.push(ProjectBlueprint::RadiationShelter);
    }

    // Water plant for water-poor worlds
    if location == "Mars" && !completed.contains(&ProjectBlueprint::WaterExtractionPlant) {
        priorities.push(ProjectBlueprint::WaterExtractionPlant);
    }

    priorities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_lifecycle() {
        let mut pm = ProjectManager::new();
        pm.start_project(ProjectBlueprint::CommsArray); // 6 ticks
        assert_eq!(pm.active.len(), 1);

        // Advance 6 ticks with sufficient resources
        for _ in 0..6 {
            let (completed, _, _) = pm.tick(1000.0, 1000.0);
            if !completed.is_empty() {
                assert_eq!(completed[0], ProjectBlueprint::CommsArray);
            }
        }
        assert!(pm.has_completed(ProjectBlueprint::CommsArray));
        assert_eq!(pm.active.len(), 0);
    }

    #[test]
    fn test_project_stalls_without_resources() {
        let mut pm = ProjectManager::new();
        pm.start_project(ProjectBlueprint::FissionReactor);

        // Try to advance with zero labor
        let (completed, _, _) = pm.tick(0.0, 1000.0);
        assert!(completed.is_empty());
        assert!(pm.active[0].stalled);
        assert_eq!(pm.active[0].stall_ticks, 1);
    }

    #[test]
    fn test_prioritization() {
        let priorities =
            prioritize_projects(200, 100.0, 0.2, 500, false, false, false, "Europa", &[]);
        // Power deficit → reactor first, food crisis → greenhouse
        assert!(priorities.contains(&ProjectBlueprint::FissionReactor));
        assert!(priorities.contains(&ProjectBlueprint::GreenhouseModule));
        // Europa needs radiation shelter
        assert!(priorities.contains(&ProjectBlueprint::RadiationShelter));
    }
}
