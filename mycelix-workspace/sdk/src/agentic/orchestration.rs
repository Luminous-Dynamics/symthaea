//! # Agent Orchestration Layer
//!
//! High-level primitives for spawning and managing agent swarms.
//! Provides automatic trust propagation, lifecycle management, and swarm coordination.
//!
//! ## Features
//!
//! - **Swarm Management**: Spawn, monitor, and terminate agent groups
//! - **Trust Propagation**: Automatic trust inheritance and sharing
//! - **KREDIT Pooling**: Shared resource pools for agent groups
//! - **Health Monitoring**: Automated health checks and recovery
//! - **Lifecycle Hooks**: Pre/post spawn, terminate, and restart callbacks
//!
//! ## Example
//!
//! ```rust,ignore
//! use mycelix_sdk::agentic::orchestration::{
//!     Orchestrator, SwarmConfig, SpawnPolicy, AgentTemplate,
//! };
//!
//! // Create orchestrator
//! let mut orchestrator = Orchestrator::new(OrchestratorConfig::default());
//!
//! // Define agent template
//! let template = AgentTemplate::new("worker")
//!     .with_initial_trust(0.5)
//!     .with_kredit(1000);
//!
//! // Spawn a swarm
//! let swarm_id = orchestrator.spawn_swarm(SwarmConfig {
//!     name: "workers".into(),
//!     template,
//!     min_agents: 3,
//!     max_agents: 10,
//!     spawn_policy: SpawnPolicy::Demand,
//! });
//!
//! // Monitor health
//! let health = orchestrator.swarm_health(&swarm_id);
//! println!("Swarm health: {:?}", health);
//! ```

use super::coordination::{AgentGroup, CoordinationConfig};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Agent Templates
// ============================================================================

/// Template for spawning agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTemplate {
    /// Template name
    pub name: String,
    /// Initial trust score
    pub initial_trust: f64,
    /// Initial KREDIT allocation
    pub initial_kredit: u64,
    /// Initial K-Vector (optional, uses defaults if None)
    pub initial_kvector: Option<KVector>,
    /// Agent class
    pub agent_class: AgentClass,
    /// Capabilities/permissions
    pub capabilities: Vec<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Agent class for orchestration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentClass {
    /// Fully autonomous
    Autonomous,
    /// Supervised by sponsor
    Supervised,
    /// Assistive (limited autonomy)
    Assistive,
    /// Worker (task-specific)
    Worker,
    /// Coordinator (manages other agents)
    Coordinator,
}

impl AgentTemplate {
    /// Create a new agent template
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            initial_trust: 0.5,
            initial_kredit: 1000,
            initial_kvector: None,
            agent_class: AgentClass::Worker,
            capabilities: vec![],
            metadata: HashMap::new(),
        }
    }

    /// Set initial trust
    pub fn with_trust(mut self, trust: f64) -> Self {
        self.initial_trust = trust.clamp(0.0, 1.0);
        self
    }

    /// Set initial KREDIT
    pub fn with_kredit(mut self, kredit: u64) -> Self {
        self.initial_kredit = kredit;
        self
    }

    /// Set agent class
    pub fn with_class(mut self, class: AgentClass) -> Self {
        self.agent_class = class;
        self
    }

    /// Add capability
    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ============================================================================
// Swarm Configuration
// ============================================================================

/// Configuration for a swarm of agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Swarm name
    pub name: String,
    /// Agent template
    pub template: AgentTemplate,
    /// Minimum number of agents
    pub min_agents: usize,
    /// Maximum number of agents
    pub max_agents: usize,
    /// Spawn policy
    pub spawn_policy: SpawnPolicy,
    /// Whether agents share a KREDIT pool
    pub shared_kredit_pool: bool,
    /// Trust propagation mode
    pub trust_propagation: TrustPropagation,
    /// Health check interval (ms)
    pub health_check_interval_ms: u64,
    /// Auto-restart failed agents
    pub auto_restart: bool,
    /// Maximum restart attempts
    pub max_restart_attempts: u32,
}

/// When to spawn new agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpawnPolicy {
    /// Spawn all at once
    Immediate,
    /// Spawn on demand
    Demand,
    /// Spawn based on load
    LoadBased,
    /// Manual spawn only
    Manual,
}

/// How trust propagates within swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustPropagation {
    /// No trust sharing
    None,
    /// Trust inherits from template
    Inherited,
    /// Trust averages across swarm
    Averaged,
    /// Trust flows from coordinator
    Hierarchical,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            name: "default-swarm".into(),
            template: AgentTemplate::new("default"),
            min_agents: 1,
            max_agents: 10,
            spawn_policy: SpawnPolicy::Immediate,
            shared_kredit_pool: false,
            trust_propagation: TrustPropagation::Inherited,
            health_check_interval_ms: 30_000,
            auto_restart: true,
            max_restart_attempts: 3,
        }
    }
}

// ============================================================================
// Orchestrated Agent
// ============================================================================

/// An agent managed by the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratedAgent {
    /// Unique agent ID
    pub id: String,
    /// Swarm this agent belongs to
    pub swarm_id: String,
    /// Current status
    pub status: AgentStatus,
    /// Trust score
    pub trust: f64,
    /// KREDIT balance (or None if using pool)
    pub kredit: Option<u64>,
    /// Spawn timestamp
    pub spawned_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Restart count
    pub restart_count: u32,
    /// Health score (0.0-1.0)
    pub health: f64,
    /// Error history
    pub errors: VecDeque<AgentError>,
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Starting up
    Starting,
    /// Running normally
    Running,
    /// Temporarily paused
    Paused,
    /// Unhealthy (needs attention)
    Unhealthy,
    /// Being restarted
    Restarting,
    /// Terminated
    Terminated,
}

/// Agent error record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentError {
    /// Error timestamp
    pub timestamp: u64,
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Whether it was recovered
    pub recovered: bool,
}

// ============================================================================
// Swarm State
// ============================================================================

/// State of a swarm
#[derive(Debug, Clone)]
pub struct Swarm {
    /// Swarm ID
    pub id: String,
    /// Configuration
    pub config: SwarmConfig,
    /// Agent IDs in this swarm
    pub agent_ids: HashSet<String>,
    /// Shared KREDIT pool (if enabled)
    pub kredit_pool: Option<KreditPool>,
    /// Coordination group (if any)
    pub coordination_group: Option<AgentGroup>,
    /// Swarm status
    pub status: SwarmStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Aggregate health
    pub aggregate_health: f64,
}

/// Swarm status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmStatus {
    /// Being created
    Creating,
    /// Running normally
    Running,
    /// Scaling up
    ScalingUp,
    /// Scaling down
    ScalingDown,
    /// Degraded (some agents unhealthy)
    Degraded,
    /// Stopped
    Stopped,
}

/// Shared KREDIT pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KreditPool {
    /// Total KREDIT in pool
    pub total: u64,
    /// Available KREDIT
    pub available: u64,
    /// Reserved KREDIT (per agent)
    pub reservations: HashMap<String, u64>,
    /// Maximum per-agent withdrawal
    pub max_withdrawal: u64,
}

impl KreditPool {
    /// Create a new pool
    pub fn new(total: u64) -> Self {
        Self {
            total,
            available: total,
            reservations: HashMap::new(),
            max_withdrawal: total / 10, // 10% max per withdrawal
        }
    }

    /// Reserve KREDIT for an agent
    pub fn reserve(&mut self, agent_id: &str, amount: u64) -> bool {
        if amount > self.available || amount > self.max_withdrawal {
            return false;
        }
        self.available -= amount;
        *self.reservations.entry(agent_id.to_string()).or_insert(0) += amount;
        true
    }

    /// Release reserved KREDIT
    pub fn release(&mut self, agent_id: &str, amount: u64) {
        if let Some(reserved) = self.reservations.get_mut(agent_id) {
            let release_amount = amount.min(*reserved);
            *reserved -= release_amount;
            self.available += release_amount;
        }
    }

    /// Get agent's reserved amount
    pub fn reserved_for(&self, agent_id: &str) -> u64 {
        self.reservations.get(agent_id).copied().unwrap_or(0)
    }
}

// ============================================================================
// Orchestrator
// ============================================================================

/// Main orchestrator for managing agent swarms
pub struct Orchestrator {
    /// Configuration
    config: OrchestratorConfig,
    /// All swarms
    swarms: HashMap<String, Swarm>,
    /// All agents
    agents: HashMap<String, OrchestratedAgent>,
    /// Event log
    events: VecDeque<OrchestratorEvent>,
    /// Lifecycle hooks
    hooks: LifecycleHooks,
    /// Next agent ID counter
    next_agent_id: u64,
}

/// Orchestrator configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum total agents across all swarms
    pub max_total_agents: usize,
    /// Maximum swarms
    pub max_swarms: usize,
    /// Default health check interval
    pub default_health_interval_ms: u64,
    /// Event log size
    pub event_log_size: usize,
    /// Enable trust propagation
    pub enable_trust_propagation: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_total_agents: 1000,
            max_swarms: 100,
            default_health_interval_ms: 30_000,
            event_log_size: 1000,
            enable_trust_propagation: true,
        }
    }
}

/// Orchestrator event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Event type
    pub event_type: EventType,
    /// Swarm ID (if applicable)
    pub swarm_id: Option<String>,
    /// Agent ID (if applicable)
    pub agent_id: Option<String>,
    /// Event data
    pub data: HashMap<String, String>,
}

/// Event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Swarm created
    SwarmCreated,
    /// Swarm stopped
    SwarmStopped,
    /// Agent spawned
    AgentSpawned,
    /// Agent terminated
    AgentTerminated,
    /// Agent restarted
    AgentRestarted,
    /// Health check
    HealthCheck,
    /// Trust updated
    TrustUpdated,
    /// KREDIT transaction
    KreditTransaction,
    /// Error occurred
    Error,
}

/// Lifecycle hooks
#[derive(Default)]
pub struct LifecycleHooks {
    /// Called before spawning an agent
    pub pre_spawn: Option<Box<dyn Fn(&AgentTemplate) -> bool + Send + Sync>>,
    /// Called after spawning an agent
    pub post_spawn: Option<Box<dyn Fn(&OrchestratedAgent) + Send + Sync>>,
    /// Called before terminating an agent
    pub pre_terminate: Option<Box<dyn Fn(&OrchestratedAgent) -> bool + Send + Sync>>,
    /// Called after terminating an agent
    pub post_terminate: Option<Box<dyn Fn(&str) + Send + Sync>>,
    /// Called on health check
    pub on_health_check: Option<Box<dyn Fn(&OrchestratedAgent) -> f64 + Send + Sync>>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config,
            swarms: HashMap::new(),
            agents: HashMap::new(),
            events: VecDeque::new(),
            hooks: LifecycleHooks::default(),
            next_agent_id: 1,
        }
    }

    /// Set lifecycle hooks
    pub fn with_hooks(mut self, hooks: LifecycleHooks) -> Self {
        self.hooks = hooks;
        self
    }

    // -------------------------------------------------------------------------
    // Swarm Management
    // -------------------------------------------------------------------------

    /// Spawn a new swarm
    pub fn spawn_swarm(&mut self, config: SwarmConfig) -> Result<String, OrchestrationError> {
        // Check limits
        if self.swarms.len() >= self.config.max_swarms {
            return Err(OrchestrationError::TooManySwarms {
                max: self.config.max_swarms,
            });
        }

        let now = Self::now();
        let swarm_id = format!("swarm-{}-{}", config.name, now);

        // Create KREDIT pool if enabled
        let kredit_pool = if config.shared_kredit_pool {
            Some(KreditPool::new(
                config.template.initial_kredit * config.max_agents as u64,
            ))
        } else {
            None
        };

        // Create coordination group if needed
        let coordination_group = if config.max_agents > 1 {
            Some(AgentGroup::with_id(
                format!("{}-coord", swarm_id),
                CoordinationConfig::default(),
            ))
        } else {
            None
        };

        let swarm = Swarm {
            id: swarm_id.clone(),
            config: config.clone(),
            agent_ids: HashSet::new(),
            kredit_pool,
            coordination_group,
            status: SwarmStatus::Creating,
            created_at: now,
            aggregate_health: 1.0,
        };

        self.swarms.insert(swarm_id.clone(), swarm);

        // Spawn initial agents if immediate
        if config.spawn_policy == SpawnPolicy::Immediate {
            for _ in 0..config.min_agents {
                self.spawn_agent_in_swarm(&swarm_id)?;
            }
        }

        // Update status
        if let Some(swarm) = self.swarms.get_mut(&swarm_id) {
            swarm.status = SwarmStatus::Running;
        }

        self.log_event(
            EventType::SwarmCreated,
            Some(&swarm_id),
            None,
            HashMap::new(),
        );

        Ok(swarm_id)
    }

    /// Stop a swarm
    pub fn stop_swarm(&mut self, swarm_id: &str) -> Result<(), OrchestrationError> {
        let swarm =
            self.swarms
                .get_mut(swarm_id)
                .ok_or_else(|| OrchestrationError::SwarmNotFound {
                    swarm_id: swarm_id.to_string(),
                })?;

        swarm.status = SwarmStatus::Stopped;

        // Terminate all agents
        let agent_ids: Vec<_> = swarm.agent_ids.iter().cloned().collect();
        for agent_id in agent_ids {
            self.terminate_agent(&agent_id)?;
        }

        self.log_event(
            EventType::SwarmStopped,
            Some(swarm_id),
            None,
            HashMap::new(),
        );

        Ok(())
    }

    /// Get swarm health
    pub fn swarm_health(&self, swarm_id: &str) -> Option<SwarmHealth> {
        let swarm = self.swarms.get(swarm_id)?;

        let agents: Vec<_> = swarm
            .agent_ids
            .iter()
            .filter_map(|id| self.agents.get(id))
            .collect();

        let total = agents.len();
        let healthy = agents
            .iter()
            .filter(|a| a.status == AgentStatus::Running && a.health > 0.7)
            .count();
        let unhealthy = agents
            .iter()
            .filter(|a| a.status == AgentStatus::Unhealthy)
            .count();

        let avg_health = if total > 0 {
            agents.iter().map(|a| a.health).sum::<f64>() / total as f64
        } else {
            1.0
        };

        let avg_trust = if total > 0 {
            agents.iter().map(|a| a.trust).sum::<f64>() / total as f64
        } else {
            0.0
        };

        Some(SwarmHealth {
            swarm_id: swarm_id.to_string(),
            status: swarm.status,
            total_agents: total,
            healthy_agents: healthy,
            unhealthy_agents: unhealthy,
            average_health: avg_health,
            average_trust: avg_trust,
            kredit_pool_available: swarm.kredit_pool.as_ref().map(|p| p.available),
        })
    }

    /// Scale swarm to target size
    pub fn scale_swarm(&mut self, swarm_id: &str, target: usize) -> Result<(), OrchestrationError> {
        let swarm =
            self.swarms
                .get_mut(swarm_id)
                .ok_or_else(|| OrchestrationError::SwarmNotFound {
                    swarm_id: swarm_id.to_string(),
                })?;

        let current = swarm.agent_ids.len();
        let target = target.clamp(swarm.config.min_agents, swarm.config.max_agents);

        if target > current {
            swarm.status = SwarmStatus::ScalingUp;
            let swarm_id = swarm_id.to_string();
            for _ in current..target {
                self.spawn_agent_in_swarm(&swarm_id)?;
            }
        } else if target < current {
            swarm.status = SwarmStatus::ScalingDown;
            let to_remove: Vec<_> = swarm.agent_ids.iter().skip(target).cloned().collect();
            for agent_id in to_remove {
                self.terminate_agent(&agent_id)?;
            }
        }

        if let Some(swarm) = self.swarms.get_mut(swarm_id) {
            swarm.status = SwarmStatus::Running;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Agent Management
    // -------------------------------------------------------------------------

    /// Spawn a new agent in a swarm
    fn spawn_agent_in_swarm(&mut self, swarm_id: &str) -> Result<String, OrchestrationError> {
        // Check total agent limit
        if self.agents.len() >= self.config.max_total_agents {
            return Err(OrchestrationError::TooManyAgents {
                max: self.config.max_total_agents,
            });
        }

        let swarm = self
            .swarms
            .get(swarm_id)
            .ok_or_else(|| OrchestrationError::SwarmNotFound {
                swarm_id: swarm_id.to_string(),
            })?;

        // Check swarm limit
        if swarm.agent_ids.len() >= swarm.config.max_agents {
            return Err(OrchestrationError::SwarmFull {
                swarm_id: swarm_id.to_string(),
                max: swarm.config.max_agents,
            });
        }

        let template = &swarm.config.template;

        // Pre-spawn hook
        if let Some(ref hook) = self.hooks.pre_spawn {
            if !hook(template) {
                return Err(OrchestrationError::SpawnBlocked {
                    reason: "Pre-spawn hook returned false".into(),
                });
            }
        }

        let now = Self::now();
        let agent_id = format!("{}-agent-{}", swarm_id, self.next_agent_id);
        self.next_agent_id += 1;

        // Calculate initial trust based on propagation mode
        let trust = match swarm.config.trust_propagation {
            TrustPropagation::None => 0.5,
            TrustPropagation::Inherited => template.initial_trust,
            TrustPropagation::Averaged => {
                if swarm.agent_ids.is_empty() {
                    template.initial_trust
                } else {
                    let total: f64 = swarm
                        .agent_ids
                        .iter()
                        .filter_map(|id| self.agents.get(id))
                        .map(|a| a.trust)
                        .sum();
                    total / swarm.agent_ids.len() as f64
                }
            }
            TrustPropagation::Hierarchical => template.initial_trust * 0.9, // Slightly less than template
        };

        let agent = OrchestratedAgent {
            id: agent_id.clone(),
            swarm_id: swarm_id.to_string(),
            status: AgentStatus::Starting,
            trust,
            kredit: if swarm.config.shared_kredit_pool {
                None
            } else {
                Some(template.initial_kredit)
            },
            spawned_at: now,
            last_activity: now,
            restart_count: 0,
            health: 1.0,
            errors: VecDeque::new(),
        };

        // Post-spawn hook
        if let Some(ref hook) = self.hooks.post_spawn {
            hook(&agent);
        }

        self.agents.insert(agent_id.clone(), agent);

        // Add to swarm
        if let Some(swarm) = self.swarms.get_mut(swarm_id) {
            swarm.agent_ids.insert(agent_id.clone());

            // Add to coordination group
            if let Some(ref mut group) = swarm.coordination_group {
                let _ = group.add_member(&agent_id, trust);
            }
        }

        // Update agent status to running
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.status = AgentStatus::Running;
        }

        self.log_event(
            EventType::AgentSpawned,
            Some(swarm_id),
            Some(&agent_id),
            HashMap::new(),
        );

        Ok(agent_id)
    }

    /// Terminate an agent
    pub fn terminate_agent(&mut self, agent_id: &str) -> Result<(), OrchestrationError> {
        let agent = self
            .agents
            .get(agent_id)
            .ok_or_else(|| OrchestrationError::AgentNotFound {
                agent_id: agent_id.to_string(),
            })?;

        // Pre-terminate hook
        if let Some(ref hook) = self.hooks.pre_terminate {
            if !hook(agent) {
                return Err(OrchestrationError::TerminateBlocked {
                    reason: "Pre-terminate hook returned false".into(),
                });
            }
        }

        let swarm_id = agent.swarm_id.clone();

        // Remove from swarm
        if let Some(swarm) = self.swarms.get_mut(&swarm_id) {
            swarm.agent_ids.remove(agent_id);

            // Remove from coordination group
            if let Some(ref mut group) = swarm.coordination_group {
                group.remove_member(agent_id);
            }

            // Release KREDIT back to pool
            if let Some(ref mut pool) = swarm.kredit_pool {
                let reserved = pool.reserved_for(agent_id);
                pool.release(agent_id, reserved);
            }
        }

        // Update status
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.status = AgentStatus::Terminated;
        }

        // Post-terminate hook
        if let Some(ref hook) = self.hooks.post_terminate {
            hook(agent_id);
        }

        self.log_event(
            EventType::AgentTerminated,
            Some(&swarm_id),
            Some(agent_id),
            HashMap::new(),
        );

        Ok(())
    }

    /// Restart an agent
    pub fn restart_agent(&mut self, agent_id: &str) -> Result<(), OrchestrationError> {
        let agent =
            self.agents
                .get_mut(agent_id)
                .ok_or_else(|| OrchestrationError::AgentNotFound {
                    agent_id: agent_id.to_string(),
                })?;

        let swarm_id = agent.swarm_id.clone();

        // Check restart limit
        let max_restarts = self
            .swarms
            .get(&swarm_id)
            .map(|s| s.config.max_restart_attempts)
            .unwrap_or(3);

        if agent.restart_count >= max_restarts {
            return Err(OrchestrationError::MaxRestartsExceeded {
                agent_id: agent_id.to_string(),
                max: max_restarts,
            });
        }

        agent.status = AgentStatus::Restarting;
        agent.restart_count += 1;
        agent.last_activity = Self::now();
        agent.health = 1.0;
        agent.errors.clear();
        agent.status = AgentStatus::Running;

        self.log_event(
            EventType::AgentRestarted,
            Some(&swarm_id),
            Some(agent_id),
            HashMap::new(),
        );

        Ok(())
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&OrchestratedAgent> {
        self.agents.get(agent_id)
    }

    /// Update agent health
    pub fn update_agent_health(&mut self, agent_id: &str, health: f64) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.health = health.clamp(0.0, 1.0);
            agent.last_activity = Self::now();

            if health < 0.3 {
                agent.status = AgentStatus::Unhealthy;
            }
        }
    }

    /// Update agent trust
    pub fn update_agent_trust(&mut self, agent_id: &str, trust: f64) {
        // First, get the swarm_id and update the agent
        let swarm_id = {
            if let Some(agent) = self.agents.get_mut(agent_id) {
                agent.trust = trust.clamp(0.0, 1.0);
                Some(agent.swarm_id.clone())
            } else {
                None
            }
        };

        let Some(swarm_id) = swarm_id else { return };

        // Update in coordination group
        let should_propagate = {
            if let Some(swarm) = self.swarms.get_mut(&swarm_id) {
                if let Some(ref mut group) = swarm.coordination_group {
                    let _ = group.update_trust(agent_id, trust);
                }
                swarm.config.trust_propagation == TrustPropagation::Averaged
            } else {
                false
            }
        };

        // Propagate trust if configured
        if should_propagate {
            self.propagate_trust(&swarm_id);
        }

        self.log_event(
            EventType::TrustUpdated,
            Some(&swarm_id),
            Some(agent_id),
            [("trust".to_string(), trust.to_string())]
                .into_iter()
                .collect(),
        );
    }

    // -------------------------------------------------------------------------
    // Trust Propagation
    // -------------------------------------------------------------------------

    /// Propagate trust across swarm
    fn propagate_trust(&mut self, swarm_id: &str) {
        if !self.config.enable_trust_propagation {
            return;
        }

        let swarm = match self.swarms.get(swarm_id) {
            Some(s) => s,
            None => return,
        };

        if swarm.config.trust_propagation != TrustPropagation::Averaged {
            return;
        }

        // Calculate average trust
        let trusts: Vec<f64> = swarm
            .agent_ids
            .iter()
            .filter_map(|id| self.agents.get(id))
            .map(|a| a.trust)
            .collect();

        if trusts.is_empty() {
            return;
        }

        let avg_trust = trusts.iter().sum::<f64>() / trusts.len() as f64;

        // Apply dampened average to all agents
        let agent_ids: Vec<_> = swarm.agent_ids.iter().cloned().collect();
        for agent_id in agent_ids {
            if let Some(agent) = self.agents.get_mut(&agent_id) {
                // Move 10% toward average
                agent.trust = agent.trust * 0.9 + avg_trust * 0.1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn log_event(
        &mut self,
        event_type: EventType,
        swarm_id: Option<&str>,
        agent_id: Option<&str>,
        data: HashMap<String, String>,
    ) {
        let event = OrchestratorEvent {
            timestamp: Self::now(),
            event_type,
            swarm_id: swarm_id.map(String::from),
            agent_id: agent_id.map(String::from),
            data,
        };

        self.events.push_back(event);

        // Trim event log
        while self.events.len() > self.config.event_log_size {
            self.events.pop_front();
        }
    }

    /// Get recent events
    pub fn recent_events(&self, limit: usize) -> Vec<&OrchestratorEvent> {
        self.events.iter().rev().take(limit).collect()
    }

    /// Get all swarm IDs
    pub fn swarm_ids(&self) -> Vec<&str> {
        self.swarms.keys().map(|s| s.as_str()).collect()
    }

    /// Get swarm
    pub fn get_swarm(&self, swarm_id: &str) -> Option<&Swarm> {
        self.swarms.get(swarm_id)
    }
}

// ============================================================================
// Swarm Health
// ============================================================================

/// Health summary for a swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmHealth {
    /// Swarm ID
    pub swarm_id: String,
    /// Current status
    pub status: SwarmStatus,
    /// Total agents
    pub total_agents: usize,
    /// Healthy agents
    pub healthy_agents: usize,
    /// Unhealthy agents
    pub unhealthy_agents: usize,
    /// Average health score
    pub average_health: f64,
    /// Average trust score
    pub average_trust: f64,
    /// Available KREDIT in pool (if pooled)
    pub kredit_pool_available: Option<u64>,
}

// ============================================================================
// Errors
// ============================================================================

/// Orchestration errors
#[derive(Debug, Clone)]
pub enum OrchestrationError {
    /// Too many swarms.
    TooManySwarms {
        /// Maximum allowed swarms.
        max: usize,
    },
    /// Too many agents.
    TooManyAgents {
        /// Maximum allowed agents.
        max: usize,
    },
    /// Swarm not found.
    SwarmNotFound {
        /// Swarm identifier.
        swarm_id: String,
    },
    /// Agent not found.
    AgentNotFound {
        /// Agent identifier.
        agent_id: String,
    },
    /// Swarm is full.
    SwarmFull {
        /// Swarm identifier.
        swarm_id: String,
        /// Maximum swarm capacity.
        max: usize,
    },
    /// Spawn blocked by hook.
    SpawnBlocked {
        /// Reason for blocking.
        reason: String,
    },
    /// Terminate blocked by hook.
    TerminateBlocked {
        /// Reason for blocking.
        reason: String,
    },
    /// Max restarts exceeded.
    MaxRestartsExceeded {
        /// Agent identifier.
        agent_id: String,
        /// Maximum allowed restarts.
        max: u32,
    },
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestrationError::TooManySwarms { max } => {
                write!(f, "Maximum swarms ({}) exceeded", max)
            }
            OrchestrationError::TooManyAgents { max } => {
                write!(f, "Maximum agents ({}) exceeded", max)
            }
            OrchestrationError::SwarmNotFound { swarm_id } => {
                write!(f, "Swarm not found: {}", swarm_id)
            }
            OrchestrationError::AgentNotFound { agent_id } => {
                write!(f, "Agent not found: {}", agent_id)
            }
            OrchestrationError::SwarmFull { swarm_id, max } => {
                write!(f, "Swarm {} is full (max {})", swarm_id, max)
            }
            OrchestrationError::SpawnBlocked { reason } => write!(f, "Spawn blocked: {}", reason),
            OrchestrationError::TerminateBlocked { reason } => {
                write!(f, "Terminate blocked: {}", reason)
            }
            OrchestrationError::MaxRestartsExceeded { agent_id, max } => {
                write!(f, "Agent {} exceeded max restarts ({})", agent_id, max)
            }
        }
    }
}

impl std::error::Error for OrchestrationError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_orchestrator() {
        let orch = Orchestrator::new(OrchestratorConfig::default());
        assert!(orch.swarms.is_empty());
        assert!(orch.agents.is_empty());
    }

    #[test]
    fn test_spawn_swarm() {
        let mut orch = Orchestrator::new(OrchestratorConfig::default());

        let config = SwarmConfig {
            name: "test".into(),
            template: AgentTemplate::new("worker").with_trust(0.6),
            min_agents: 3,
            max_agents: 10,
            spawn_policy: SpawnPolicy::Immediate,
            ..Default::default()
        };

        let swarm_id = orch.spawn_swarm(config).unwrap();

        let health = orch.swarm_health(&swarm_id).unwrap();
        assert_eq!(health.total_agents, 3);
        assert_eq!(health.status, SwarmStatus::Running);
    }

    #[test]
    fn test_scale_swarm() {
        let mut orch = Orchestrator::new(OrchestratorConfig::default());

        let config = SwarmConfig {
            name: "test".into(),
            template: AgentTemplate::new("worker"),
            min_agents: 2,
            max_agents: 10,
            spawn_policy: SpawnPolicy::Immediate,
            ..Default::default()
        };

        let swarm_id = orch.spawn_swarm(config).unwrap();
        assert_eq!(orch.swarm_health(&swarm_id).unwrap().total_agents, 2);

        orch.scale_swarm(&swarm_id, 5).unwrap();
        assert_eq!(orch.swarm_health(&swarm_id).unwrap().total_agents, 5);

        orch.scale_swarm(&swarm_id, 3).unwrap();
        assert_eq!(orch.swarm_health(&swarm_id).unwrap().total_agents, 3);
    }

    #[test]
    fn test_kredit_pool() {
        let mut pool = KreditPool::new(1000);

        assert!(pool.reserve("agent-1", 100));
        assert_eq!(pool.available, 900);
        assert_eq!(pool.reserved_for("agent-1"), 100);

        pool.release("agent-1", 50);
        assert_eq!(pool.available, 950);
        assert_eq!(pool.reserved_for("agent-1"), 50);
    }

    #[test]
    fn test_agent_template() {
        let template = AgentTemplate::new("worker")
            .with_trust(0.7)
            .with_kredit(5000)
            .with_class(AgentClass::Autonomous)
            .with_capability("read")
            .with_capability("write");

        assert_eq!(template.name, "worker");
        assert!((template.initial_trust - 0.7).abs() < 0.01);
        assert_eq!(template.initial_kredit, 5000);
        assert_eq!(template.capabilities.len(), 2);
    }

    #[test]
    fn test_trust_propagation() {
        let mut orch = Orchestrator::new(OrchestratorConfig::default());

        let config = SwarmConfig {
            name: "test".into(),
            template: AgentTemplate::new("worker").with_trust(0.5),
            min_agents: 3,
            max_agents: 10,
            spawn_policy: SpawnPolicy::Immediate,
            trust_propagation: TrustPropagation::Averaged,
            ..Default::default()
        };

        let swarm_id = orch.spawn_swarm(config).unwrap();

        // Get first agent and update its trust
        let agent_ids: Vec<_> = orch
            .get_swarm(&swarm_id)
            .unwrap()
            .agent_ids
            .iter()
            .cloned()
            .collect();

        orch.update_agent_trust(&agent_ids[0], 0.9);

        // Trust should have propagated somewhat
        let health = orch.swarm_health(&swarm_id).unwrap();
        assert!(health.average_trust > 0.5);
    }
}
