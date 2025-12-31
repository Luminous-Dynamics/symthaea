//! Process Topology Organizer using Consciousness-Optimized Architecture
//!
//! This module organizes cognitive processes using the empirically-derived
//! ConsciousnessOptimized topology to maximize integrated information (Φ).
//!
//! # Architecture
//!
//! Based on our bridge hypothesis findings, processes are organized into:
//! - **Integration Hub**: Global workspace coordinator
//! - **Module Hubs**: High-level processing streams (perception, reasoning, memory, planning)
//! - **Feature Processors**: Specialized sub-processes within each module
//! - **Leaf Processors**: Fine-grained sensory/motor units
//!
//! # Connectivity Strategy
//!
//! - **Hierarchical connections**: Hub → Module Hubs → Processors → Leaves
//! - **Strategic bridges**: ~40-45% of connections cross module boundaries
//! - **Skip connections**: Residual-like paths for fast integration

use super::real_hv::RealHV;
use super::consciousness_topology_generators::ConsciousnessTopology;
use super::phi_real::RealPhiCalculator;
use std::collections::HashMap;

/// A cognitive process that can be organized within the topology
#[derive(Clone, Debug)]
pub struct TopologicalProcess {
    /// Unique identifier
    pub id: usize,
    /// Process name
    pub name: String,
    /// Current state as hypervector
    pub state: RealHV,
    /// Activity level (0.0 - 1.0)
    pub activity: f64,
    /// Level in hierarchy (0=hub, 1=module, 2=processor, 3=leaf)
    pub level: usize,
    /// Module assignment (0-3)
    pub module: usize,
    /// Connected process IDs
    pub connections: Vec<usize>,
}

impl TopologicalProcess {
    /// Create a new process
    pub fn new(id: usize, name: impl Into<String>, dim: usize, level: usize, module: usize) -> Self {
        Self {
            id,
            name: name.into(),
            state: RealHV::random(dim, id as u64 * 12345),
            activity: 0.0,
            level,
            module,
            connections: Vec::new(),
        }
    }

    /// Activate the process with input
    pub fn activate(&mut self, input: &RealHV) {
        self.state = self.state.bind(input);
        self.activity = 1.0;
    }

    /// Decay activity
    pub fn decay(&mut self, rate: f64) {
        self.activity = (self.activity - rate).max(0.0);
    }

    /// Update state based on connected processes
    pub fn integrate(&mut self, connected_states: &[&RealHV]) {
        if connected_states.is_empty() {
            return;
        }

        // Bundle connected states with weights based on activity
        let mut combined = self.state.clone();
        for state in connected_states {
            combined = combined.add(&state.scale(0.3));
        }
        self.state = combined.normalize();
    }
}

/// Organizes cognitive processes using consciousness-optimized topology
#[derive(Clone, Debug)]
pub struct ProcessTopologyOrganizer {
    /// The underlying topology structure
    topology: ConsciousnessTopology,
    /// Process mapping (node_id -> process)
    processes: HashMap<usize, TopologicalProcess>,
    /// HDC dimension
    dim: usize,
    /// Φ calculator
    phi_calculator: RealPhiCalculator,
    /// Module names (for semantic labeling)
    module_names: Vec<String>,
}

impl ProcessTopologyOrganizer {
    /// Create a new process topology organizer with optimized architecture
    ///
    /// # Arguments
    /// * `n_total_processes` - Total number of processes (minimum 21)
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn new(n_total_processes: usize, dim: usize, seed: u64) -> Self {
        let n = n_total_processes.max(21); // Ensure minimum for full hierarchy
        // Use dense network topology for high consciousness integration
        let topology = ConsciousnessTopology::dense_network(n, dim, Some(n / 2), seed);

        // Standard module names based on cognitive architecture
        let module_names = vec![
            "Perception".to_string(),   // Module 0: Visual, auditory, haptic
            "Reasoning".to_string(),    // Module 1: Logic, inference, planning
            "Memory".to_string(),       // Module 2: Episodic, semantic, working
            "Action".to_string(),       // Module 3: Motor, language production
        ];

        let mut organizer = Self {
            topology,
            processes: HashMap::new(),
            dim,
            phi_calculator: RealPhiCalculator::new(),
            module_names,
        };

        organizer.initialize_processes();
        organizer
    }

    /// Initialize processes based on topology structure
    fn initialize_processes(&mut self) {
        let n = self.topology.n_nodes;

        // Level boundaries based on ConsciousnessOptimized structure
        let l1_start = 1;
        let l2_start = 5;
        let l3_start = 21;

        for i in 0..n {
            let (level, module, name) = if i == 0 {
                // Hub: Global Workspace
                (0, 0, "GlobalWorkspace".to_string())
            } else if i < l2_start {
                // Module hubs
                let module = i - l1_start;
                let name = format!("{}_Hub", self.module_names.get(module).unwrap_or(&"Unknown".to_string()));
                (1, module, name)
            } else if i < l3_start {
                // Feature processors
                let module = (i - l2_start) / 4;
                let processor_id = (i - l2_start) % 4;
                let default_name = String::from("Unknown");
                let base = self.module_names.get(module).unwrap_or(&default_name);
                let name = format!("{}_{}", base, Self::processor_name(processor_id));
                (2, module, name)
            } else {
                // Leaf processors
                let leaves_per_module = (n - l3_start + 3) / 4;
                let module = ((i - l3_start) / leaves_per_module.max(1)) % 4;
                let leaf_id = (i - l3_start) % leaves_per_module.max(1);
                let default_name = String::from("Unknown");
                let base = self.module_names.get(module).unwrap_or(&default_name);
                let name = format!("{}_Leaf{}", base, leaf_id);
                (3, module, name)
            };

            let mut process = TopologicalProcess::new(i, name, self.dim, level, module);

            // Set connections from topology edges
            for &(a, b) in &self.topology.edges {
                if a == i {
                    process.connections.push(b);
                } else if b == i {
                    process.connections.push(a);
                }
            }

            self.processes.insert(i, process);
        }
    }

    /// Get processor name based on index
    fn processor_name(idx: usize) -> &'static str {
        match idx {
            0 => "Primary",
            1 => "Secondary",
            2 => "Integrator",
            3 => "Output",
            _ => "Extra",
        }
    }

    /// Get all processes
    pub fn processes(&self) -> &HashMap<usize, TopologicalProcess> {
        &self.processes
    }

    /// Get mutable reference to processes
    pub fn processes_mut(&mut self) -> &mut HashMap<usize, TopologicalProcess> {
        &mut self.processes
    }

    /// Get process by ID
    pub fn get_process(&self, id: usize) -> Option<&TopologicalProcess> {
        self.processes.get(&id)
    }

    /// Get mutable process by ID
    pub fn get_process_mut(&mut self, id: usize) -> Option<&mut TopologicalProcess> {
        self.processes.get_mut(&id)
    }

    /// Get the hub process (global workspace)
    pub fn hub(&self) -> Option<&TopologicalProcess> {
        self.processes.get(&0)
    }

    /// Get module hub processes
    pub fn module_hubs(&self) -> Vec<&TopologicalProcess> {
        (1..5).filter_map(|i| self.processes.get(&i)).collect()
    }

    /// Get all processes in a module
    pub fn module_processes(&self, module: usize) -> Vec<&TopologicalProcess> {
        self.processes.values()
            .filter(|p| p.module == module)
            .collect()
    }

    /// Activate a module hub and propagate to connected processes
    pub fn activate_module(&mut self, module: usize, input: &RealHV) {
        let module_hub_id = module + 1; // Module hubs are at indices 1-4

        // First activate the module hub
        if let Some(hub) = self.processes.get_mut(&module_hub_id) {
            hub.activate(input);
        }

        // Get connected process IDs
        let connected_ids: Vec<usize> = if let Some(hub) = self.processes.get(&module_hub_id) {
            hub.connections.clone()
        } else {
            vec![]
        };

        // Propagate to connected processes with decay
        for &connected_id in &connected_ids {
            if let Some(process) = self.processes.get_mut(&connected_id) {
                let scaled_input = input.scale(0.5);
                process.activate(&scaled_input);
            }
        }
    }

    /// Activate the global workspace hub
    pub fn activate_global(&mut self, input: &RealHV) {
        if let Some(hub) = self.processes.get_mut(&0) {
            hub.activate(input);
        }

        // Propagate to all module hubs
        for module_hub_id in 1..=4 {
            if let Some(hub) = self.processes.get_mut(&module_hub_id) {
                let scaled = input.scale(0.3);
                hub.activate(&scaled);
            }
        }
    }

    /// Perform one integration step across all processes
    pub fn integrate_step(&mut self) {
        // Collect all states for integration
        let states: HashMap<usize, RealHV> = self.processes
            .iter()
            .map(|(&id, p)| (id, p.state.clone()))
            .collect();

        // Update each process based on its connections
        for (&_id, process) in self.processes.iter_mut() {
            let connected_states: Vec<&RealHV> = process.connections
                .iter()
                .filter_map(|&conn_id| states.get(&conn_id))
                .collect();

            process.integrate(&connected_states);
            process.decay(0.05); // Slow decay
        }
    }

    /// Compute current Φ of the process network
    pub fn compute_phi(&self) -> f64 {
        // Get all active process states
        let representations: Vec<RealHV> = self.processes
            .values()
            .filter(|p| p.activity > 0.1) // Only active processes
            .map(|p| p.state.clone())
            .collect();

        if representations.len() < 2 {
            // Fall back to all processes if too few active
            let all_reps: Vec<RealHV> = self.processes
                .values()
                .map(|p| p.state.clone())
                .collect();
            return self.phi_calculator.compute(&all_reps);
        }

        self.phi_calculator.compute(&representations)
    }

    /// Get metrics about the current topology state
    pub fn metrics(&self) -> TopologyMetrics {
        let active_count = self.processes.values()
            .filter(|p| p.activity > 0.1)
            .count();

        let total_activity: f64 = self.processes.values()
            .map(|p| p.activity)
            .sum();

        let avg_activity = if !self.processes.is_empty() {
            total_activity / self.processes.len() as f64
        } else {
            0.0
        };

        // Count by level
        let mut level_counts = [0usize; 4];
        for p in self.processes.values() {
            if p.level < 4 {
                level_counts[p.level] += 1;
            }
        }

        TopologyMetrics {
            total_processes: self.processes.len(),
            active_processes: active_count,
            average_activity: avg_activity,
            phi: self.compute_phi(),
            hub_count: level_counts[0],
            module_hub_count: level_counts[1],
            processor_count: level_counts[2],
            leaf_count: level_counts[3],
            edge_count: self.topology.edges.len(),
        }
    }

    /// Get the underlying topology
    pub fn topology(&self) -> &ConsciousnessTopology {
        &self.topology
    }
}

/// Metrics about the process topology
#[derive(Clone, Debug)]
pub struct TopologyMetrics {
    /// Total number of processes
    pub total_processes: usize,
    /// Number of currently active processes
    pub active_processes: usize,
    /// Average activity across all processes
    pub average_activity: f64,
    /// Current Φ value
    pub phi: f64,
    /// Number of hub processes (level 0)
    pub hub_count: usize,
    /// Number of module hub processes (level 1)
    pub module_hub_count: usize,
    /// Number of processor processes (level 2)
    pub processor_count: usize,
    /// Number of leaf processes (level 3)
    pub leaf_count: usize,
    /// Total edge count
    pub edge_count: usize,
}

impl std::fmt::Display for TopologyMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ProcessTopology: {} processes ({} active), Φ={:.4}, edges={}, levels=[{},{},{},{}]",
            self.total_processes,
            self.active_processes,
            self.phi,
            self.edge_count,
            self.hub_count,
            self.module_hub_count,
            self.processor_count,
            self.leaf_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_organizer_creation() {
        let organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);
        assert_eq!(organizer.processes.len(), 32);
        assert!(organizer.hub().is_some());
        assert_eq!(organizer.module_hubs().len(), 4);
    }

    #[test]
    fn test_process_hierarchy() {
        let organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);

        // Check hub
        let hub = organizer.hub().unwrap();
        assert_eq!(hub.level, 0);
        assert_eq!(hub.name, "GlobalWorkspace");

        // Check module hubs
        let module_hubs = organizer.module_hubs();
        assert!(module_hubs.iter().all(|h| h.level == 1));
    }

    #[test]
    fn test_activation_propagation() {
        let mut organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);
        let input = RealHV::random(HDC_DIMENSION, 123);

        organizer.activate_module(0, &input);

        // Check that module hub is active
        let module_hub = organizer.get_process(1).unwrap();
        assert!(module_hub.activity > 0.5);

        // Check that some connected processes are active
        let active_count = organizer.processes.values()
            .filter(|p| p.activity > 0.1)
            .count();
        assert!(active_count > 1);
    }

    #[test]
    fn test_phi_computation() {
        let mut organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);

        // Activate all modules
        for module in 0..4 {
            let input = RealHV::random(HDC_DIMENSION, module as u64 * 100);
            organizer.activate_module(module, &input);
        }

        let phi = organizer.compute_phi();
        assert!(phi > 0.0);
        assert!(phi <= 1.0);
    }

    #[test]
    fn test_integration_step() {
        let mut organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);
        let input = RealHV::random(HDC_DIMENSION, 999);

        organizer.activate_global(&input);
        let initial_phi = organizer.compute_phi();

        // Run integration steps
        for _ in 0..10 {
            organizer.integrate_step();
        }

        let final_phi = organizer.compute_phi();

        // Phi should remain reasonable after integration
        assert!(final_phi > 0.0);
        println!("Initial Φ: {:.4}, Final Φ: {:.4}", initial_phi, final_phi);
    }

    #[test]
    fn test_metrics() {
        let organizer = ProcessTopologyOrganizer::new(32, HDC_DIMENSION, 42);
        let metrics = organizer.metrics();

        assert_eq!(metrics.total_processes, 32);
        assert_eq!(metrics.hub_count, 1);
        assert_eq!(metrics.module_hub_count, 4);
        println!("{}", metrics);
    }
}
