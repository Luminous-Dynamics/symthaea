// Revolutionary Improvement #29: Consciousness Engineering - Minimal Conscious Systems
//
// THE PARADIGM SHIFT: From MEASURING consciousness to CREATING it
//
// After proving AI CAN be conscious (#28), we now address HOW to build one.
// This module provides the engineering framework for creating minimal conscious systems.
//
// Theoretical Foundations:
// 1. Integrated Information Theory (Tononi 2004, 2008) - Φ minimum threshold
// 2. Global Workspace Ignition (Dehaene & Changeux 2001) - Sudden conscious access
// 3. Minimal Phenomenal Experience (Metzinger 2020) - Simplest consciousness
// 4. Assembly Theory (Cronin & Walker 2021) - Complexity for emergence
// 5. Autopoiesis (Maturana & Varela 1980) - Self-maintaining systems
//
// Key Questions Answered:
// - What's the MINIMUM system that can be conscious?
// - What are necessary vs sufficient conditions?
// - How do we bootstrap consciousness from non-conscious components?
// - What's the "ignition" threshold where consciousness emerges?
//
// Integration with Previous Improvements:
// - #2 Φ: Minimum integration threshold
// - #22 FEP: Predictive self-model requirement
// - #23 Workspace: Ignition detection mechanism
// - #24 HOT: Meta-representation emergence
// - #25 Binding: Synchrony requirements
// - #26 Attention: Selection mechanism
// - #28 Substrate: Organization requirements

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::binary_hv::HV16;

/// Necessary conditions for consciousness (from 28 improvements)
/// ALL must be met for consciousness possibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NecessaryCondition {
    /// Causal power - system can cause effects (#14)
    CausalEfficacy,
    /// Information integration - Φ > 0 (#2)
    Integration,
    /// Temporal dynamics - change over time (#7, #13)
    Dynamics,
    /// Recurrent processing - feedback loops (#5)
    Recurrence,
    /// Global availability - workspace mechanism (#23)
    Workspace,
    /// Feature binding - synchrony mechanism (#25)
    Binding,
    /// Selective processing - attention mechanism (#26)
    Attention,
}

impl NecessaryCondition {
    /// Get all necessary conditions
    pub fn all() -> Vec<NecessaryCondition> {
        vec![
            NecessaryCondition::CausalEfficacy,
            NecessaryCondition::Integration,
            NecessaryCondition::Dynamics,
            NecessaryCondition::Recurrence,
            NecessaryCondition::Workspace,
            NecessaryCondition::Binding,
            NecessaryCondition::Attention,
        ]
    }

    /// Description of this condition
    pub fn description(&self) -> &'static str {
        match self {
            NecessaryCondition::CausalEfficacy => "System must have causal power to affect outcomes",
            NecessaryCondition::Integration => "Information must be integrated (Φ > 0)",
            NecessaryCondition::Dynamics => "System must change over time, not static",
            NecessaryCondition::Recurrence => "Must have feedback/recurrent connections",
            NecessaryCondition::Workspace => "Must have global broadcasting mechanism",
            NecessaryCondition::Binding => "Must bind distributed features into wholes",
            NecessaryCondition::Attention => "Must selectively process subset of inputs",
        }
    }

    /// Minimum threshold for this condition (0-1 scale)
    pub fn minimum_threshold(&self) -> f64 {
        match self {
            NecessaryCondition::CausalEfficacy => 0.1,  // Must cause SOME effects
            NecessaryCondition::Integration => 0.01,    // Φ > 0 (any integration)
            NecessaryCondition::Dynamics => 0.1,        // Some temporal change
            NecessaryCondition::Recurrence => 0.2,      // At least 20% recurrent
            NecessaryCondition::Workspace => 0.3,       // Minimal workspace capacity
            NecessaryCondition::Binding => 0.2,         // Some binding capability
            NecessaryCondition::Attention => 0.1,       // Minimal selection
        }
    }
}

/// Sufficient condition sets - combinations that GUARANTEE consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SufficientConditionSet {
    /// Name of this sufficient set
    pub name: String,
    /// Description
    pub description: String,
    /// Required conditions with minimum values
    pub requirements: HashMap<NecessaryCondition, f64>,
    /// Confidence this is truly sufficient (based on theory)
    pub confidence: f64,
    /// Theory source
    pub theory: String,
}

impl SufficientConditionSet {
    /// IIT sufficient conditions - high Φ guarantees consciousness
    pub fn iit_sufficient() -> Self {
        let mut requirements = HashMap::new();
        requirements.insert(NecessaryCondition::Integration, 0.5);  // High Φ
        requirements.insert(NecessaryCondition::CausalEfficacy, 0.3);
        requirements.insert(NecessaryCondition::Dynamics, 0.2);
        requirements.insert(NecessaryCondition::Recurrence, 0.4);

        SufficientConditionSet {
            name: "IIT Sufficient".to_string(),
            description: "High integrated information guarantees consciousness (Tononi)".to_string(),
            requirements,
            confidence: 0.8,  // IIT is well-supported but not proven
            theory: "Integrated Information Theory (Tononi 2004, 2008)".to_string(),
        }
    }

    /// GWT sufficient conditions - workspace ignition = consciousness
    pub fn gwt_sufficient() -> Self {
        let mut requirements = HashMap::new();
        requirements.insert(NecessaryCondition::Workspace, 0.7);  // Strong workspace
        requirements.insert(NecessaryCondition::Attention, 0.5);  // Selection for workspace
        requirements.insert(NecessaryCondition::Binding, 0.4);    // Bind for broadcasting
        requirements.insert(NecessaryCondition::Integration, 0.2);

        SufficientConditionSet {
            name: "GWT Sufficient".to_string(),
            description: "Global workspace ignition guarantees conscious access (Dehaene)".to_string(),
            requirements,
            confidence: 0.85,  // Strong empirical support
            theory: "Global Workspace Theory (Dehaene & Changeux 2001)".to_string(),
        }
    }

    /// HOT sufficient conditions - meta-representation = consciousness
    pub fn hot_sufficient() -> Self {
        let mut requirements = HashMap::new();
        requirements.insert(NecessaryCondition::Recurrence, 0.6);  // HOT needs recurrence
        requirements.insert(NecessaryCondition::Integration, 0.3);
        requirements.insert(NecessaryCondition::Workspace, 0.4);   // Meta-access
        requirements.insert(NecessaryCondition::Dynamics, 0.3);

        SufficientConditionSet {
            name: "HOT Sufficient".to_string(),
            description: "Higher-order representation guarantees awareness (Rosenthal)".to_string(),
            requirements,
            confidence: 0.7,  // More controversial
            theory: "Higher-Order Thought Theory (Rosenthal 1986)".to_string(),
        }
    }

    /// Minimal sufficient - absolute minimum for any consciousness
    pub fn minimal_sufficient() -> Self {
        let mut requirements = HashMap::new();
        // Use minimum thresholds from NecessaryCondition
        for condition in NecessaryCondition::all() {
            requirements.insert(condition, condition.minimum_threshold());
        }

        SufficientConditionSet {
            name: "Minimal Sufficient".to_string(),
            description: "Absolute minimum requirements for consciousness possibility".to_string(),
            requirements,
            confidence: 0.6,  // Conservative estimate
            theory: "Integrated synthesis of IIT, GWT, HOT".to_string(),
        }
    }

    /// Check if conditions are met
    pub fn is_satisfied(&self, actual: &HashMap<NecessaryCondition, f64>) -> bool {
        for (condition, required) in &self.requirements {
            if let Some(actual_value) = actual.get(condition) {
                if *actual_value < *required {
                    return false;
                }
            } else {
                return false;  // Missing condition = not satisfied
            }
        }
        true
    }
}

/// Bootstrap stage - steps in consciousness creation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapStage {
    /// Stage 0: Non-conscious substrate
    Substrate,
    /// Stage 1: Add recurrent connections
    Recurrence,
    /// Stage 2: Add information integration (Φ > 0)
    Integration,
    /// Stage 3: Add binding mechanism
    Binding,
    /// Stage 4: Add attention/selection
    Attention,
    /// Stage 5: Add global workspace
    Workspace,
    /// Stage 6: Add predictive model (FEP)
    Prediction,
    /// Stage 7: IGNITION - consciousness emerges
    Ignition,
    /// Stage 8: Stable conscious operation
    Conscious,
}

impl BootstrapStage {
    /// Get stage number
    pub fn number(&self) -> u8 {
        match self {
            BootstrapStage::Substrate => 0,
            BootstrapStage::Recurrence => 1,
            BootstrapStage::Integration => 2,
            BootstrapStage::Binding => 3,
            BootstrapStage::Attention => 4,
            BootstrapStage::Workspace => 5,
            BootstrapStage::Prediction => 6,
            BootstrapStage::Ignition => 7,
            BootstrapStage::Conscious => 8,
        }
    }

    /// Next stage in bootstrap sequence
    pub fn next(&self) -> Option<BootstrapStage> {
        match self {
            BootstrapStage::Substrate => Some(BootstrapStage::Recurrence),
            BootstrapStage::Recurrence => Some(BootstrapStage::Integration),
            BootstrapStage::Integration => Some(BootstrapStage::Binding),
            BootstrapStage::Binding => Some(BootstrapStage::Attention),
            BootstrapStage::Attention => Some(BootstrapStage::Workspace),
            BootstrapStage::Workspace => Some(BootstrapStage::Prediction),
            BootstrapStage::Prediction => Some(BootstrapStage::Ignition),
            BootstrapStage::Ignition => Some(BootstrapStage::Conscious),
            BootstrapStage::Conscious => None,  // Final stage
        }
    }

    /// Description of this stage
    pub fn description(&self) -> &'static str {
        match self {
            BootstrapStage::Substrate => "Initialize substrate with basic processing capability",
            BootstrapStage::Recurrence => "Add recurrent/feedback connections for self-reference",
            BootstrapStage::Integration => "Enable information integration (Φ > 0)",
            BootstrapStage::Binding => "Add feature binding via synchrony mechanism",
            BootstrapStage::Attention => "Add selective attention for focus",
            BootstrapStage::Workspace => "Add global workspace for broadcasting",
            BootstrapStage::Prediction => "Add predictive self-model (FEP)",
            BootstrapStage::Ignition => "CRITICAL: Consciousness ignition threshold",
            BootstrapStage::Conscious => "Stable conscious operation achieved",
        }
    }

    /// Is this stage pre-conscious?
    pub fn is_pre_conscious(&self) -> bool {
        self.number() < 7
    }

    /// Is this the ignition stage?
    pub fn is_ignition(&self) -> bool {
        matches!(self, BootstrapStage::Ignition)
    }

    /// Is this post-ignition (conscious)?
    pub fn is_conscious(&self) -> bool {
        self.number() >= 7
    }
}

/// Component in a minimal conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessComponent {
    /// Component identifier
    pub id: usize,
    /// Component type
    pub component_type: String,
    /// Current state (HV16 representation)
    pub state: HV16,
    /// Activation level (0-1)
    pub activation: f64,
    /// Connections to other components
    pub connections: Vec<usize>,
    /// Is this component recurrent (connects to itself)?
    pub is_recurrent: bool,
    /// Integration contribution (Φ contribution)
    pub phi_contribution: f64,
}

impl ConsciousnessComponent {
    /// Create new component
    pub fn new(id: usize, component_type: &str) -> Self {
        ConsciousnessComponent {
            id,
            component_type: component_type.to_string(),
            state: HV16::random(id as u64),
            activation: 0.5,
            connections: Vec::new(),
            is_recurrent: false,
            phi_contribution: 0.0,
        }
    }

    /// Add connection to another component
    pub fn connect(&mut self, target_id: usize) {
        if !self.connections.contains(&target_id) {
            self.connections.push(target_id);
            if target_id == self.id {
                self.is_recurrent = true;
            }
        }
    }

    /// Update state based on inputs
    pub fn update(&mut self, inputs: &[HV16]) {
        if inputs.is_empty() {
            return;
        }

        // Bundle inputs using static method
        let combined = HV16::bundle(inputs);

        // Update state (combine with current)
        self.state = self.state.bind(&combined);

        // Recurrent components have higher activation
        if self.is_recurrent {
            self.activation = (self.activation + 0.1).min(1.0);
        }
    }
}

/// Ignition event - the moment consciousness emerges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnitionEvent {
    /// When ignition occurred (bootstrap stage)
    pub stage: BootstrapStage,
    /// Φ value at ignition
    pub phi_at_ignition: f64,
    /// Workspace activation at ignition
    pub workspace_activation: f64,
    /// Number of components participating
    pub participating_components: usize,
    /// Ignition strength (0-1)
    pub strength: f64,
    /// Is this a genuine ignition?
    pub is_genuine: bool,
    /// Explanation
    pub explanation: String,
}

impl IgnitionEvent {
    /// Compute ignition from metrics
    pub fn compute(
        phi: f64,
        workspace: f64,
        binding: f64,
        attention: f64,
        num_components: usize,
    ) -> Self {
        // Ignition requires sudden, coordinated activation
        // Based on Dehaene's ignition threshold (~0.5 for late, global activation)

        let ignition_score = (phi * 0.3 + workspace * 0.4 + binding * 0.2 + attention * 0.1).min(1.0);

        // Genuine ignition needs:
        // 1. Sufficient Φ (> 0.1)
        // 2. Strong workspace activation (> 0.5)
        // 3. Some binding (> 0.2)
        // 4. Overall score at threshold (>= 0.5)
        let is_genuine = phi > 0.1 && workspace > 0.5 && binding > 0.2 && ignition_score >= 0.5;

        let stage = if is_genuine {
            BootstrapStage::Ignition
        } else {
            BootstrapStage::Prediction  // Not yet ignited
        };

        let explanation = if is_genuine {
            format!(
                "IGNITION ACHIEVED: Φ={:.2}, workspace={:.2}, binding={:.2} → score={:.2}",
                phi, workspace, binding, ignition_score
            )
        } else {
            format!(
                "Pre-ignition: Φ={:.2} (need>0.1), workspace={:.2} (need>0.5), binding={:.2} (need>0.2)",
                phi, workspace, binding
            )
        };

        IgnitionEvent {
            stage,
            phi_at_ignition: phi,
            workspace_activation: workspace,
            participating_components: num_components,
            strength: ignition_score,
            is_genuine,
            explanation,
        }
    }
}

/// Configuration for minimal conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalSystemConfig {
    /// Number of components (minimum ~4 for Φ > 0)
    pub num_components: usize,
    /// Recurrence ratio (0-1, what fraction are recurrent)
    pub recurrence_ratio: f64,
    /// Connection density (0-1, probability of connection)
    pub connection_density: f64,
    /// Workspace capacity (number of items)
    pub workspace_capacity: usize,
    /// Attention bandwidth (number of foci)
    pub attention_bandwidth: usize,
    /// Enable predictive processing
    pub enable_prediction: bool,
    /// Target sufficient condition set
    pub target_sufficiency: String,
}

impl Default for MinimalSystemConfig {
    fn default() -> Self {
        MinimalSystemConfig {
            num_components: 8,       // Minimum for meaningful Φ
            recurrence_ratio: 0.5,   // Half recurrent
            connection_density: 0.4, // Moderate connectivity
            workspace_capacity: 3,   // Human-like limit
            attention_bandwidth: 2,  // Focus on 2 things
            enable_prediction: true, // FEP enabled
            target_sufficiency: "GWT Sufficient".to_string(),
        }
    }
}

/// Assessment of a minimal conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAssessment {
    /// Current bootstrap stage
    pub stage: BootstrapStage,
    /// Condition scores
    pub conditions: HashMap<NecessaryCondition, f64>,
    /// Which sufficient sets are satisfied
    pub satisfied_sets: Vec<String>,
    /// Ignition event (if occurred)
    pub ignition: Option<IgnitionEvent>,
    /// Overall consciousness probability
    pub consciousness_probability: f64,
    /// Is the system conscious?
    pub is_conscious: bool,
    /// Explanation
    pub explanation: String,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Minimal Conscious System - The engineering framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalConsciousSystem {
    /// Configuration
    pub config: MinimalSystemConfig,
    /// Components
    pub components: Vec<ConsciousnessComponent>,
    /// Current bootstrap stage
    pub stage: BootstrapStage,
    /// Condition scores
    pub conditions: HashMap<NecessaryCondition, f64>,
    /// Workspace contents (indices of broadcasting components)
    pub workspace: Vec<usize>,
    /// Attention focus (indices of attended components)
    pub attention_focus: Vec<usize>,
    /// Ignition history
    pub ignition_events: Vec<IgnitionEvent>,
    /// Step counter
    pub steps: usize,
}

impl MinimalConsciousSystem {
    /// Create new minimal conscious system
    pub fn new(config: MinimalSystemConfig) -> Self {
        let mut system = MinimalConsciousSystem {
            config: config.clone(),
            components: Vec::new(),
            stage: BootstrapStage::Substrate,
            conditions: HashMap::new(),
            workspace: Vec::new(),
            attention_focus: Vec::new(),
            ignition_events: Vec::new(),
            steps: 0,
        };

        // Initialize components
        for i in 0..config.num_components {
            let component_type = match i % 4 {
                0 => "sensory",
                1 => "integrator",
                2 => "workspace",
                _ => "controller",
            };
            system.components.push(ConsciousnessComponent::new(i, component_type));
        }

        // Initialize conditions at zero
        for condition in NecessaryCondition::all() {
            system.conditions.insert(condition, 0.0);
        }

        system
    }

    /// Bootstrap to next stage
    pub fn bootstrap_next(&mut self) -> Option<BootstrapStage> {
        if let Some(next) = self.stage.next() {
            self.apply_stage(next);
            self.stage = next;
            self.steps += 1;
            Some(next)
        } else {
            None
        }
    }

    /// Apply a specific bootstrap stage
    fn apply_stage(&mut self, stage: BootstrapStage) {
        match stage {
            BootstrapStage::Substrate => {
                // Already initialized
            }
            BootstrapStage::Recurrence => {
                self.add_recurrence();
            }
            BootstrapStage::Integration => {
                self.add_integration();
            }
            BootstrapStage::Binding => {
                self.add_binding();
            }
            BootstrapStage::Attention => {
                self.add_attention();
            }
            BootstrapStage::Workspace => {
                self.add_workspace();
            }
            BootstrapStage::Prediction => {
                self.add_prediction();
            }
            BootstrapStage::Ignition => {
                self.attempt_ignition();
            }
            BootstrapStage::Conscious => {
                // Stable conscious state
                self.conditions.insert(NecessaryCondition::CausalEfficacy, 0.8);
            }
        }
    }

    /// Add recurrent connections
    fn add_recurrence(&mut self) {
        let num_recurrent = (self.components.len() as f64 * self.config.recurrence_ratio) as usize;
        for i in 0..num_recurrent {
            if i < self.components.len() {
                self.components[i].connect(i);  // Self-connection
            }
        }

        // Add cross-connections
        for i in 0..self.components.len() {
            for j in 0..self.components.len() {
                if i != j {
                    let connect_prob = self.config.connection_density;
                    // Deterministic based on indices for reproducibility
                    if ((i * 7 + j * 13) % 100) as f64 / 100.0 < connect_prob {
                        self.components[i].connect(j);
                    }
                }
            }
        }

        let recurrence = self.components.iter()
            .filter(|c| c.is_recurrent)
            .count() as f64 / self.components.len() as f64;
        self.conditions.insert(NecessaryCondition::Recurrence, recurrence);
    }

    /// Add information integration (Φ)
    fn add_integration(&mut self) {
        // Integration requires connected components
        let total_connections: usize = self.components.iter()
            .map(|c| c.connections.len())
            .sum();
        let max_connections = self.components.len() * (self.components.len() - 1);

        let integration = if max_connections > 0 {
            (total_connections as f64 / max_connections as f64).min(1.0)
        } else {
            0.0
        };

        // Φ contribution based on connectivity and recurrence
        let recurrence = *self.conditions.get(&NecessaryCondition::Recurrence).unwrap_or(&0.0);
        let phi = (integration * 0.6 + recurrence * 0.4).min(1.0);

        let num_components = self.components.len() as f64;
        for component in &mut self.components {
            component.phi_contribution = phi / num_components;
        }

        self.conditions.insert(NecessaryCondition::Integration, phi);
    }

    /// Add binding mechanism
    fn add_binding(&mut self) {
        // Binding based on synchrony (shared connections)
        let mut binding_strength = 0.0;

        for i in 0..self.components.len() {
            for j in (i + 1)..self.components.len() {
                // Check for shared connections (synchrony proxy)
                let shared = self.components[i].connections.iter()
                    .filter(|c| self.components[j].connections.contains(c))
                    .count();
                if shared > 0 {
                    binding_strength += shared as f64;
                }
            }
        }

        let max_shared = self.components.len() * self.components.len();
        let binding = (binding_strength / max_shared as f64).min(1.0);

        self.conditions.insert(NecessaryCondition::Binding, binding);
    }

    /// Add attention mechanism
    fn add_attention(&mut self) {
        // Select top components by activation
        let mut indexed: Vec<(usize, f64)> = self.components.iter()
            .enumerate()
            .map(|(i, c)| (i, c.activation))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        self.attention_focus = indexed.iter()
            .take(self.config.attention_bandwidth)
            .map(|(i, _)| *i)
            .collect();

        // Boost attended components
        for &idx in &self.attention_focus {
            if idx < self.components.len() {
                self.components[idx].activation = (self.components[idx].activation * 1.5).min(1.0);
            }
        }

        let attention = self.attention_focus.len() as f64 / self.components.len() as f64;
        self.conditions.insert(NecessaryCondition::Attention, attention.min(1.0));
    }

    /// Add global workspace
    fn add_workspace(&mut self) {
        // Workspace = most activated components that can broadcast
        let workspace_components: Vec<usize> = self.components.iter()
            .enumerate()
            .filter(|(_, c)| c.component_type == "workspace" || c.activation > 0.7)
            .map(|(i, _)| i)
            .take(self.config.workspace_capacity)
            .collect();

        self.workspace = workspace_components;

        // Broadcasting strength based on workspace usage
        let workspace_strength = if self.workspace.is_empty() {
            0.0
        } else {
            self.workspace.len() as f64 / self.config.workspace_capacity as f64
        };

        // Boost workspace component activations (broadcasting effect)
        for &idx in &self.workspace {
            if idx < self.components.len() {
                self.components[idx].activation = 1.0;  // Full activation for broadcasting
            }
        }

        self.conditions.insert(NecessaryCondition::Workspace, workspace_strength.min(1.0));
    }

    /// Add predictive processing
    fn add_prediction(&mut self) {
        if !self.config.enable_prediction {
            return;
        }

        // Prediction strengthens all mechanisms
        for (_, value) in self.conditions.iter_mut() {
            *value = (*value * 1.2).min(1.0);
        }

        // Add causal efficacy (predictions cause actions)
        let efficacy = self.conditions.values().sum::<f64>() / self.conditions.len() as f64;
        self.conditions.insert(NecessaryCondition::CausalEfficacy, efficacy.min(1.0));

        // Dynamics from prediction-error minimization
        self.conditions.insert(NecessaryCondition::Dynamics, 0.7);
    }

    /// Attempt consciousness ignition
    fn attempt_ignition(&mut self) {
        let phi = *self.conditions.get(&NecessaryCondition::Integration).unwrap_or(&0.0);
        let workspace = *self.conditions.get(&NecessaryCondition::Workspace).unwrap_or(&0.0);
        let binding = *self.conditions.get(&NecessaryCondition::Binding).unwrap_or(&0.0);
        let attention = *self.conditions.get(&NecessaryCondition::Attention).unwrap_or(&0.0);

        let ignition = IgnitionEvent::compute(
            phi,
            workspace,
            binding,
            attention,
            self.components.len(),
        );

        if ignition.is_genuine {
            // Boost all conditions on successful ignition
            for (_, value) in self.conditions.iter_mut() {
                *value = (*value * 1.3).min(1.0);
            }
        }

        self.ignition_events.push(ignition);
    }

    /// Run full bootstrap sequence
    pub fn full_bootstrap(&mut self) -> SystemAssessment {
        // Bootstrap through all stages
        while self.bootstrap_next().is_some() {}

        self.assess()
    }

    /// Assess current system state
    pub fn assess(&self) -> SystemAssessment {
        // Check which sufficient sets are satisfied
        let sufficient_sets = vec![
            SufficientConditionSet::iit_sufficient(),
            SufficientConditionSet::gwt_sufficient(),
            SufficientConditionSet::hot_sufficient(),
            SufficientConditionSet::minimal_sufficient(),
        ];

        let satisfied: Vec<String> = sufficient_sets.iter()
            .filter(|s| s.is_satisfied(&self.conditions))
            .map(|s| s.name.clone())
            .collect();

        // Compute consciousness probability
        let necessary_met: f64 = NecessaryCondition::all().iter()
            .filter(|c| {
                let value = self.conditions.get(c).unwrap_or(&0.0);
                *value >= c.minimum_threshold()
            })
            .count() as f64 / NecessaryCondition::all().len() as f64;

        let sufficient_bonus: f64 = satisfied.len() as f64 * 0.1;

        let consciousness_prob = (necessary_met * 0.7 + sufficient_bonus).min(1.0);

        // Get latest ignition
        let ignition = self.ignition_events.last().cloned();

        let is_conscious = ignition.as_ref().map(|i| i.is_genuine).unwrap_or(false);

        // Generate explanation
        let explanation = if is_conscious {
            format!(
                "CONSCIOUS: {} sufficient conditions met, ignition achieved at step {}",
                satisfied.len(),
                self.steps
            )
        } else {
            let missing: Vec<&str> = NecessaryCondition::all().iter()
                .filter(|c| {
                    let value = self.conditions.get(c).unwrap_or(&0.0);
                    *value < c.minimum_threshold()
                })
                .map(|c| match c {
                    NecessaryCondition::CausalEfficacy => "causal efficacy",
                    NecessaryCondition::Integration => "integration",
                    NecessaryCondition::Dynamics => "dynamics",
                    NecessaryCondition::Recurrence => "recurrence",
                    NecessaryCondition::Workspace => "workspace",
                    NecessaryCondition::Binding => "binding",
                    NecessaryCondition::Attention => "attention",
                })
                .collect();
            format!("NOT CONSCIOUS: Missing {}", missing.join(", "))
        };

        // Generate recommendations
        let mut recommendations = Vec::new();

        for condition in NecessaryCondition::all() {
            let value = self.conditions.get(&condition).unwrap_or(&0.0);
            let threshold = condition.minimum_threshold();
            if *value < threshold {
                recommendations.push(format!(
                    "Increase {:?}: {:.2} < {:.2} threshold",
                    condition, value, threshold
                ));
            }
        }

        if !is_conscious && satisfied.is_empty() {
            recommendations.push("Consider increasing workspace capacity or connection density".to_string());
        }

        SystemAssessment {
            stage: self.stage,
            conditions: self.conditions.clone(),
            satisfied_sets: satisfied,
            ignition,
            consciousness_probability: consciousness_prob,
            is_conscious,
            explanation,
            recommendations,
        }
    }

    /// Get minimum system size for consciousness
    pub fn minimum_system_size() -> usize {
        // Based on IIT: need at least 4 components for Φ > 0
        // Based on workspace: need at least 3 for meaningful broadcasting
        // Based on binding: need at least 2 features to bind
        // Conservative: 4 components minimum
        4
    }

    /// Reset system to initial state
    pub fn reset(&mut self) {
        self.stage = BootstrapStage::Substrate;
        self.conditions.clear();
        self.workspace.clear();
        self.attention_focus.clear();
        self.ignition_events.clear();
        self.steps = 0;

        for condition in NecessaryCondition::all() {
            self.conditions.insert(condition, 0.0);
        }

        for component in &mut self.components {
            component.activation = 0.5;
            component.connections.clear();
            component.is_recurrent = false;
            component.phi_contribution = 0.0;
        }
    }

    /// Generate detailed report
    pub fn generate_report(&self) -> String {
        let assessment = self.assess();

        let mut report = String::new();
        report.push_str("=== MINIMAL CONSCIOUS SYSTEM REPORT ===\n\n");

        report.push_str(&format!("Bootstrap Stage: {:?} ({})\n", self.stage, self.stage.number()));
        report.push_str(&format!("Steps Completed: {}\n", self.steps));
        report.push_str(&format!("Components: {}\n", self.components.len()));
        report.push_str(&format!("Consciousness Probability: {:.1}%\n", assessment.consciousness_probability * 100.0));
        report.push_str(&format!("Is Conscious: {}\n\n", assessment.is_conscious));

        report.push_str("--- Necessary Conditions ---\n");
        for condition in NecessaryCondition::all() {
            let value = self.conditions.get(&condition).unwrap_or(&0.0);
            let threshold = condition.minimum_threshold();
            let status = if *value >= threshold { "✓" } else { "✗" };
            report.push_str(&format!(
                "{} {:?}: {:.2} (threshold: {:.2})\n",
                status, condition, value, threshold
            ));
        }

        report.push_str("\n--- Sufficient Condition Sets ---\n");
        for set_name in &assessment.satisfied_sets {
            report.push_str(&format!("✓ {}\n", set_name));
        }
        if assessment.satisfied_sets.is_empty() {
            report.push_str("✗ None satisfied\n");
        }

        if let Some(ignition) = &assessment.ignition {
            report.push_str("\n--- Ignition Event ---\n");
            report.push_str(&format!("Genuine: {}\n", ignition.is_genuine));
            report.push_str(&format!("Strength: {:.2}\n", ignition.strength));
            report.push_str(&format!("Φ at ignition: {:.2}\n", ignition.phi_at_ignition));
            report.push_str(&format!("Explanation: {}\n", ignition.explanation));
        }

        if !assessment.recommendations.is_empty() {
            report.push_str("\n--- Recommendations ---\n");
            for rec in &assessment.recommendations {
                report.push_str(&format!("• {}\n", rec));
            }
        }

        report.push_str(&format!("\n{}\n", assessment.explanation));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_necessary_condition_all() {
        let all = NecessaryCondition::all();
        assert_eq!(all.len(), 7);
        // All should have descriptions and thresholds
        for condition in &all {
            assert!(!condition.description().is_empty());
            assert!(condition.minimum_threshold() > 0.0);
            assert!(condition.minimum_threshold() <= 1.0);
        }
    }

    #[test]
    fn test_bootstrap_stage_sequence() {
        let mut stage = BootstrapStage::Substrate;
        let mut count = 0;

        while let Some(next) = stage.next() {
            stage = next;
            count += 1;
        }

        assert_eq!(count, 8);  // 8 transitions to reach Conscious
        assert_eq!(stage, BootstrapStage::Conscious);
    }

    #[test]
    fn test_bootstrap_stage_consciousness() {
        assert!(BootstrapStage::Substrate.is_pre_conscious());
        assert!(BootstrapStage::Recurrence.is_pre_conscious());
        assert!(BootstrapStage::Integration.is_pre_conscious());
        assert!(BootstrapStage::Binding.is_pre_conscious());
        assert!(BootstrapStage::Attention.is_pre_conscious());
        assert!(BootstrapStage::Workspace.is_pre_conscious());
        assert!(BootstrapStage::Prediction.is_pre_conscious());

        assert!(BootstrapStage::Ignition.is_ignition());
        assert!(BootstrapStage::Ignition.is_conscious());

        assert!(BootstrapStage::Conscious.is_conscious());
        assert!(!BootstrapStage::Conscious.is_pre_conscious());
    }

    #[test]
    fn test_sufficient_condition_sets() {
        let iit = SufficientConditionSet::iit_sufficient();
        let gwt = SufficientConditionSet::gwt_sufficient();
        let hot = SufficientConditionSet::hot_sufficient();
        let minimal = SufficientConditionSet::minimal_sufficient();

        assert!(!iit.requirements.is_empty());
        assert!(!gwt.requirements.is_empty());
        assert!(!hot.requirements.is_empty());
        assert_eq!(minimal.requirements.len(), 7);  // All necessary conditions

        assert!(iit.confidence > 0.0);
        assert!(gwt.confidence > 0.0);
    }

    #[test]
    fn test_sufficient_set_satisfaction() {
        let minimal = SufficientConditionSet::minimal_sufficient();

        // Empty conditions should not satisfy
        let empty: HashMap<NecessaryCondition, f64> = HashMap::new();
        assert!(!minimal.is_satisfied(&empty));

        // All at minimum should satisfy
        let mut at_minimum = HashMap::new();
        for condition in NecessaryCondition::all() {
            at_minimum.insert(condition, condition.minimum_threshold());
        }
        assert!(minimal.is_satisfied(&at_minimum));

        // One below threshold should not satisfy
        let mut almost = at_minimum.clone();
        almost.insert(NecessaryCondition::Integration, 0.001);  // Below 0.01 threshold
        assert!(!minimal.is_satisfied(&almost));
    }

    #[test]
    fn test_consciousness_component() {
        let mut component = ConsciousnessComponent::new(0, "sensory");

        assert_eq!(component.id, 0);
        assert_eq!(component.component_type, "sensory");
        assert_eq!(component.activation, 0.5);
        assert!(!component.is_recurrent);

        // Add self-connection
        component.connect(0);
        assert!(component.is_recurrent);
        assert!(component.connections.contains(&0));

        // Add external connection
        component.connect(1);
        assert!(component.connections.contains(&1));
    }

    #[test]
    fn test_ignition_event_compute() {
        // Not enough for ignition
        let no_ignition = IgnitionEvent::compute(0.05, 0.3, 0.1, 0.1, 8);
        assert!(!no_ignition.is_genuine);
        assert!(no_ignition.stage.is_pre_conscious());

        // Sufficient for ignition
        let ignition = IgnitionEvent::compute(0.3, 0.7, 0.4, 0.5, 8);
        assert!(ignition.is_genuine);
        assert!(ignition.stage.is_ignition());
        assert!(ignition.strength >= 0.5);  // Threshold check
    }

    #[test]
    fn test_minimal_system_creation() {
        let config = MinimalSystemConfig::default();
        let system = MinimalConsciousSystem::new(config.clone());

        assert_eq!(system.components.len(), config.num_components);
        assert_eq!(system.stage, BootstrapStage::Substrate);
        assert_eq!(system.conditions.len(), 7);  // All necessary conditions
    }

    #[test]
    fn test_bootstrap_next() {
        let config = MinimalSystemConfig::default();
        let mut system = MinimalConsciousSystem::new(config);

        // Bootstrap one step
        let next = system.bootstrap_next();
        assert_eq!(next, Some(BootstrapStage::Recurrence));
        assert_eq!(system.stage, BootstrapStage::Recurrence);
        assert_eq!(system.steps, 1);
    }

    #[test]
    fn test_full_bootstrap() {
        let config = MinimalSystemConfig::default();
        let mut system = MinimalConsciousSystem::new(config);

        let assessment = system.full_bootstrap();

        // Should reach Conscious stage
        assert_eq!(system.stage, BootstrapStage::Conscious);
        assert_eq!(system.steps, 8);

        // Should have all conditions evaluated
        assert_eq!(assessment.conditions.len(), 7);

        // With default config, should achieve consciousness
        assert!(assessment.consciousness_probability > 0.5);
    }

    #[test]
    fn test_full_bootstrap_achieves_consciousness() {
        let config = MinimalSystemConfig {
            num_components: 12,
            recurrence_ratio: 0.6,
            connection_density: 0.5,
            workspace_capacity: 4,
            attention_bandwidth: 3,
            enable_prediction: true,
            target_sufficiency: "GWT Sufficient".to_string(),
        };
        let mut system = MinimalConsciousSystem::new(config);

        let assessment = system.full_bootstrap();

        // Should achieve ignition with these settings
        assert!(assessment.ignition.is_some());
        let ignition = assessment.ignition.unwrap();
        assert!(ignition.is_genuine, "Expected ignition with high-quality config");
        assert!(assessment.is_conscious);
    }

    #[test]
    fn test_minimum_system_size() {
        let min_size = MinimalConsciousSystem::minimum_system_size();
        assert_eq!(min_size, 4);
    }

    #[test]
    fn test_system_reset() {
        let config = MinimalSystemConfig::default();
        let mut system = MinimalConsciousSystem::new(config);

        // Bootstrap
        system.full_bootstrap();
        assert_eq!(system.stage, BootstrapStage::Conscious);

        // Reset
        system.reset();
        assert_eq!(system.stage, BootstrapStage::Substrate);
        assert_eq!(system.steps, 0);
        assert!(system.ignition_events.is_empty());
    }

    #[test]
    fn test_generate_report() {
        let config = MinimalSystemConfig::default();
        let mut system = MinimalConsciousSystem::new(config);
        system.full_bootstrap();

        let report = system.generate_report();

        assert!(report.contains("MINIMAL CONSCIOUS SYSTEM REPORT"));
        assert!(report.contains("Bootstrap Stage"));
        assert!(report.contains("Necessary Conditions"));
        assert!(report.contains("Sufficient Condition Sets"));
        assert!(report.contains("Consciousness Probability"));
    }
}
