// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! nix-mind-daemon: background daemon for continuous NixOS awareness.
//!
//! Periodically observes the system state, encodes it into the HDC world model,
//! and updates the causal graph with any state transitions. High-surprise events
//! (large prediction errors) are stored in episodic memory for future reference.
//!
//! The daemon writes a `DaemonSnapshot` to disk on every cycle for TUI consumption.
//!
//! Designed to run as a systemd service via the NixOS module.

use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use symthaea_core::hdc::ContinuousHV;
use symthaea_nix::encoding::{NixCodebook, ServiceState, SystemStateEncoder, SystemStateSnapshot};
use symthaea_nix::ipc::{
    AlertEntry, AlertSeverity, AnomalyEntry, CausalEdgeEntry, ConcernEntry, DaemonConfig,
    DaemonSnapshot, default_snapshot_path,
};
use symthaea_nix::mind::active_inference::NixActiveInference;
use symthaea_nix::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use symthaea_nix::mind::episodic_memory::{EpisodeOutcome, NixEpisodicMemory, SystemEpisode};
use symthaea_nix::mind::ollama_bridge::{OllamaBridge, OllamaBridgeConfig};
use symthaea_nix::mind::working_memory::{MemorySource, WorkingMemory};
use symthaea_nix::mind::{JournalAnomalyDetector, NixWorldModel};
use symthaea_nix::observe::SystemObserver;
use symthaea_nix::observe::journal::JournalObserver;
use symthaea_nix::plugin::domain_plugin::NixOsPlugin;
use symthaea_nix::support::health_check::{HealthAssessor, HealthStatus};
use symthaea_nix::support::knowledge::{DynamicKnowledgeArticle, KnowledgeBase, KnowledgeCategory};
use symthaea_nix::support::poml::{PomlContext, PomlProcessor, PomlValue};
use symthaea_nix::support::predictive::{
    AlertThresholds, PredictiveMonitor, SavedPredictiveState, SystemTelemetry,
};
use symthaea_nix::traits::DomainPlugin;

#[cfg(feature = "observability")]
use symthaea_nix::observability::{Metrics, PhaseTimer, init_tracing};

/// Mutable daemon state collected across cycles.
struct DaemonState {
    codebook: NixCodebook,
    world_model: NixWorldModel,
    causal_graph: NixCausalGraph,
    episodic_memory: NixEpisodicMemory,
    working_memory: WorkingMemory,
    anomaly_detector: JournalAnomalyDetector,
    health_assessor: HealthAssessor,
    predictive_monitor: PredictiveMonitor,
    prev_snapshot: Option<SystemStateSnapshot>,
    prev_state_hv: Option<ContinuousHV>,
    observation_count: u64,
    anomaly_count: u64,
    recent_anomalies: Vec<AnomalyEntry>,
    knowledge_base: Option<KnowledgeBase>,
    ollama: Option<OllamaBridge>,
    last_health_status: Option<HealthStatus>,
    last_health_issue_count: usize,
    /// Per-unit Ollama query cooldown: unit → last query time.
    ollama_cooldowns: HashMap<String, Instant>,
    /// Last observed memory usage percentage.
    last_memory_pct: Option<f64>,
    /// Persistent alert state: metric+horizon key → (first_seen, consecutive_cycles, prev_predicted).
    alert_state: HashMap<String, AlertTracking>,
    /// Last watchdog verdict (read from disk, written by `nix-mind watch`).
    watchdog_status: Option<String>,
    /// Cached last-known hardware probe results (fallback when probe fails).
    last_hw_probe: Option<symthaea_nix::observe::hardware::HardwareInfo>,
    /// Whether the daemon is in degraded mode (using cached data).
    degraded: bool,
    /// Active inference engine for formulating maintenance goals.
    active_inference: NixActiveInference,
    /// NixOS domain plugin for enriching anomaly diagnostics.
    nix_plugin: NixOsPlugin,
    /// Count of maintenance plans generated (dry-run).
    maintenance_plan_count: u32,
    /// Cumulative count of persistence write errors (working_memory, predictions, knowledge, causal graph).
    persist_error_count: u64,
}

/// Tracks alert continuity across IPC cycles.
struct AlertTracking {
    first_seen: u64,
    consecutive_cycles: u32,
    prev_predicted_value: f64,
}

impl DaemonState {
    fn new(config: &DaemonConfig) -> Self {
        let mut codebook = NixCodebook::new();
        let mut causal_graph = NixCausalGraph::new(42);
        for (cause, effect, _) in NixOSCausalPatterns::known_patterns() {
            causal_graph.add_structural_edge(cause, effect, 0.5);
        }

        let knowledge_base = if config.enable_knowledge_learning {
            Some(KnowledgeBase::new(&mut codebook))
        } else {
            None
        };

        let ollama = {
            let ollama_config = OllamaBridgeConfig {
                endpoint: config.ollama_endpoint.clone(),
                model: config.ollama_model.clone(),
                timeout: Duration::from_secs(config.ollama_timeout),
                ..OllamaBridgeConfig::default()
            };
            Some(OllamaBridge::new(ollama_config))
        };

        Self {
            codebook,
            world_model: NixWorldModel::default_dim(),
            causal_graph,
            episodic_memory: NixEpisodicMemory::new(),
            working_memory: WorkingMemory::new(),
            anomaly_detector: JournalAnomalyDetector::new(),
            health_assessor: HealthAssessor::default(),
            predictive_monitor: PredictiveMonitor::with_defaults(),
            prev_snapshot: None,
            prev_state_hv: None,
            observation_count: 0,
            anomaly_count: 0,
            recent_anomalies: Vec::new(),
            knowledge_base,
            ollama,
            last_health_status: None,
            last_health_issue_count: 0,
            ollama_cooldowns: HashMap::new(),
            last_memory_pct: None,
            alert_state: HashMap::new(),
            watchdog_status: None,
            last_hw_probe: None,
            degraded: false,
            active_inference: NixActiveInference::new(),
            nix_plugin: NixOsPlugin,
            maintenance_plan_count: 0,
            persist_error_count: 0,
        }
    }

    /// Learn from a resolved anomaly by creating a dynamic knowledge article.
    ///
    /// If Ollama is available, also queries for resolution verification (BL)
    /// to determine if the fix is permanent or temporary.
    fn learn_from_resolution(&mut self, symptom: &str, resolution: &str, commands: Vec<String>) {
        let kb = match self.knowledge_base.as_mut() {
            Some(kb) => kb,
            None => return,
        };

        // Query Ollama for resolution analysis if available (BL)
        let enriched_solution = if let Some(ollama) = self.ollama.as_mut() {
            let prompt = build_resolution_prompt(symptom, resolution, &commands);
            if let Some(response) = ollama.query(&prompt) {
                eprintln!(
                    "nix-mind-daemon: resolution verified via Ollama ({}ms)",
                    response.duration_ms
                );
                format!("{}\n\nAnalysis: {}", resolution, response.text)
            } else {
                resolution.to_string()
            }
        } else {
            resolution.to_string()
        };

        let id = format!("learned_{}", now_secs());
        let article = DynamicKnowledgeArticle {
            id,
            title: format!("Resolved: {}", symptom),
            category: KnowledgeCategory::ServiceIssue,
            symptoms: vec![symptom.to_string()],
            solution: enriched_solution,
            commands,
            learned_at: now_secs() as i64,
            hit_count: 0,
        };
        kb.add_learned_article(article, &mut self.codebook);
    }

    /// Process a system snapshot: encode, detect transitions, learn.
    fn process_snapshot(&mut self, snapshot: SystemStateSnapshot, config: &DaemonConfig) {
        let state_hv = {
            let mut encoder = SystemStateEncoder::new(&mut self.codebook);
            encoder.encode_snapshot(&snapshot)
        };

        self.world_model.observe(state_hv.clone());
        self.observation_count += 1;

        let free_energy = self.world_model.free_energy();

        // Detect state transitions vs previous snapshot
        let mut wm_pushes = 0usize;
        let mut recoveries: Vec<(String, String, String)> = Vec::new();
        if let (Some(prev_snap), Some(prev_hv)) = (&self.prev_snapshot, &self.prev_state_hv) {
            let transitions = detect_transitions(prev_snap, &snapshot);
            if !transitions.is_empty() {
                // Causal learning: observe which changes co-occurred
                let all_keys: Vec<&str> = transitions.iter().map(|t| t.key.as_str()).collect();
                let occurred_keys: Vec<&str> = transitions
                    .iter()
                    .filter(|t| t.occurred)
                    .map(|t| t.key.as_str())
                    .collect();
                self.causal_graph
                    .observe_outcome(&transitions[0].key, &occurred_keys, &all_keys);

                // Add transitions to working memory and collect recoveries
                for transition in &transitions {
                    let label = format!(
                        "{}: {} → {}",
                        transition.key, transition.from, transition.to
                    );
                    self.working_memory.push(
                        state_hv.clone(),
                        MemorySource::SystemObservation,
                        label,
                    );
                    wm_pushes += 1;

                    if transition.is_recovery {
                        recoveries.push((
                            transition.key.clone(),
                            transition.from.clone(),
                            transition.to.clone(),
                        ));
                    }
                }
            }

            // Episodic storage for high-surprise events
            if free_energy > config.surprise_threshold {
                let episode = SystemEpisode {
                    state_before: prev_hv.clone(),
                    action: "system_transition".to_string(),
                    state_after: state_hv.clone(),
                    outcome: EpisodeOutcome::Success,
                    phi_at_encoding: free_energy,
                    prediction_error: free_energy,
                    emotional_valence: 0.0,
                    timestamp: now_secs() as i64,
                };
                self.episodic_memory.record(episode);
            }
        }

        // Graduate evicted items and learn from recoveries after borrows end
        if wm_pushes > 0 {
            self.graduate_evicted(free_energy);
        }
        for (key, from, to) in recoveries {
            self.learn_from_resolution(
                &format!("{} service failure", key),
                &format!(
                    "Service {} recovered automatically ({} → {})",
                    key, from, to
                ),
                vec![format!("systemctl status {}", key)],
            );
        }

        // Run health assessment — cache successful probes, fall back to cached on failure
        let hw = match symthaea_nix::observe::hardware::HardwareObserver::probe() {
            Ok(info) => {
                self.last_hw_probe = Some(info.clone());
                self.degraded = false;
                Some(info)
            }
            Err(_) => {
                if self.last_hw_probe.is_some() {
                    self.degraded = true;
                }
                self.last_hw_probe.clone()
            }
        };
        let (overall, checks) = self.health_assessor.assess_all(&snapshot, hw.as_ref());
        self.last_health_status = Some(overall);
        self.last_health_issue_count = checks
            .iter()
            .filter(|c| c.status != HealthStatus::Healthy)
            .count();
        if overall == HealthStatus::Critical {
            eprintln!(
                "nix-mind-daemon: CRITICAL health detected, cycle {}",
                self.observation_count
            );
        }

        // Feed predictive monitor
        let telemetry = Self::build_telemetry(hw.as_ref(), &snapshot);
        let mem_pct = telemetry.memory_used_pct;
        self.predictive_monitor.ingest(telemetry);
        self.last_memory_pct = if mem_pct > 0.0 { Some(mem_pct) } else { None };

        // Feed the state to the active inference engine
        self.active_inference.observe_state(state_hv.clone());

        self.prev_snapshot = Some(snapshot);
        self.prev_state_hv = Some(state_hv);
    }

    /// Graduate evicted working memory items to episodic memory.
    fn graduate_evicted(&mut self, current_phi: f64) {
        const MIN_STEPS_FOR_GRADUATION: u64 = 3;
        if let Some(evicted) = self.working_memory.take_evicted() {
            if evicted.steps_survived >= MIN_STEPS_FOR_GRADUATION {
                let episode = SystemEpisode {
                    state_before: evicted.content.clone(),
                    action: format!("graduated_wm:{}", evicted.label),
                    state_after: evicted.content,
                    outcome: EpisodeOutcome::Success,
                    phi_at_encoding: current_phi,
                    prediction_error: 1.0 - evicted.activation,
                    emotional_valence: 0.0,
                    timestamp: now_secs() as i64,
                };
                self.episodic_memory.record(episode);
            }
        }
    }

    /// Process journal entries for anomaly detection.
    fn process_journal(&mut self, batch_size: usize) {
        let entries = match JournalObserver::recent_entries(batch_size) {
            Ok(e) => e,
            Err(_) => return,
        };

        let anomalies = self.anomaly_detector.process_entries(&entries);
        for anomaly in &anomalies {
            self.anomaly_count += 1;

            let concern_hv = self.anomaly_detector.encode_entry(&anomaly.entry);

            // Retrieve similar past episodes before moving concern_hv
            let similar_context = if anomaly.anomaly_score > 0.7 {
                self.recall_similar_episodes(&concern_hv)
            } else {
                vec![]
            };

            self.working_memory.push(
                concern_hv,
                MemorySource::SystemObservation,
                format!("anomaly: {}", anomaly.reason),
            );
            self.graduate_evicted(anomaly.anomaly_score as f64);

            // High-score anomalies: ask Ollama for diagnosis and learn from it
            if anomaly.anomaly_score > 0.7 {
                self.query_ollama_for_anomaly(
                    &anomaly.entry.unit,
                    &anomaly.reason,
                    &anomaly.entry.message,
                    &similar_context,
                );
            }

            // Enrich anomaly with NixOS domain diagnosis (AS/AW)
            let diag = self.nix_plugin.diagnose_error(&anomaly.entry.message);

            self.recent_anomalies.push(AnomalyEntry {
                score: anomaly.anomaly_score as f64,
                reason: anomaly.reason.clone(),
                unit: anomaly.entry.unit.clone(),
                error_type: diag.as_ref().map(|d| d.error_type.clone()),
                suggestion: diag.as_ref().and_then(|d| d.suggestion.clone()),
            });
        }

        if self.recent_anomalies.len() > 20 {
            let excess = self.recent_anomalies.len() - 20;
            self.recent_anomalies.drain(..excess);
        }
    }

    /// Retrieve similar past episodes as context strings.
    fn recall_similar_episodes(&self, query_hv: &ContinuousHV) -> Vec<String> {
        self.episodic_memory
            .retrieve_similar(query_hv, 3)
            .into_iter()
            .map(|ep| format!("{} (PE={:.2})", ep.action, ep.prediction_error))
            .collect()
    }

    /// Query Ollama for anomaly diagnosis and learn from the response.
    ///
    /// Rate-limited: at most one query per unit every 5 minutes.
    fn query_ollama_for_anomaly(
        &mut self,
        unit: &str,
        reason: &str,
        message: &str,
        past_context: &[String],
    ) {
        const COOLDOWN: Duration = Duration::from_secs(300); // 5 minutes

        if let Some(last) = self.ollama_cooldowns.get(unit) {
            if last.elapsed() < COOLDOWN {
                return;
            }
        }

        let ollama = match self.ollama.as_mut() {
            Some(o) => o,
            None => return,
        };

        let mut prompt = build_anomaly_prompt(unit, reason, message);
        if !past_context.is_empty() {
            prompt.push_str("\n\nSimilar past events:\n");
            for ctx in past_context {
                prompt.push_str(&format!("- {}\n", ctx));
            }
        }

        if let Some(response) = ollama.query(&prompt) {
            self.learn_from_resolution(
                &format!("{}: {}", unit, reason),
                &response.text,
                vec![format!("journalctl -u {} --since '5 min ago'", unit)],
            );
            self.ollama_cooldowns
                .insert(unit.to_string(), Instant::now());
            eprintln!(
                "nix-mind-daemon: Ollama diagnosed anomaly in {} ({}ms, model: {})",
                unit, response.duration_ms, response.model_used
            );
        }
    }

    /// Read the latest watchdog verdict from disk (written by `nix-mind watch`).
    fn refresh_watchdog_status(&mut self, state_dir: &std::path::Path) {
        let wd_path = state_dir.with_file_name("watchdog_verdict.txt");
        self.watchdog_status = std::fs::read_to_string(&wd_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
    }

    /// Build predictive alerts with deduplication and trend tracking.
    fn build_alerts(&mut self, now: u64) -> Vec<AlertEntry> {
        let mut active_keys = std::collections::HashSet::new();
        let alerts: Vec<AlertEntry> = self
            .predictive_monitor
            .predict_all_horizons()
            .into_iter()
            .filter(|p| p.crosses_threshold || p.predicted_value >= p.threshold * 0.9)
            .map(|p| {
                let key = format!("{}@{}h", p.metric, p.hours_ahead);
                active_keys.insert(key.clone());

                let tracking = self.alert_state.entry(key).or_insert(AlertTracking {
                    first_seen: now,
                    consecutive_cycles: 0,
                    prev_predicted_value: p.predicted_value,
                });
                tracking.consecutive_cycles += 1;
                let prev = tracking.prev_predicted_value;
                tracking.prev_predicted_value = p.predicted_value;

                // Cross-reference recent anomalies to populate journal context (BD)
                let journal_context: Vec<String> = self
                    .recent_anomalies
                    .iter()
                    .filter(|a| anomaly_matches_metric(a, &p.metric))
                    .take(3)
                    .map(|a| {
                        if let Some(ref et) = a.error_type {
                            format!("[{}] {}", et, a.reason)
                        } else {
                            a.reason.clone()
                        }
                    })
                    .collect();

                AlertEntry {
                    metric: p.metric.to_string(),
                    current_value: p.current_value,
                    predicted_value: p.predicted_value,
                    hours_ahead: p.hours_ahead,
                    threshold: p.threshold,
                    confidence: p.confidence as f64,
                    recommended_action: p.recommended_action,
                    severity: if p.crosses_threshold {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    first_seen: tracking.first_seen,
                    last_seen: now,
                    consecutive_cycles: tracking.consecutive_cycles,
                    prev_predicted_value: if tracking.consecutive_cycles > 1 {
                        Some(prev)
                    } else {
                        None
                    },
                    journal_context,
                }
            })
            .collect();

        // Prune alert state for alerts that are no longer active
        self.alert_state.retain(|k, _| active_keys.contains(k));

        // Hard cap to prevent unbounded growth if predictions accumulate
        const MAX_ALERT_ENTRIES: usize = 500;
        if self.alert_state.len() > MAX_ALERT_ENTRIES {
            // Keep entries with highest consecutive_cycles (most established)
            let mut entries: Vec<_> = self.alert_state.drain().collect();
            entries.sort_by(|a, b| b.1.consecutive_cycles.cmp(&a.1.consecutive_cycles));
            entries.truncate(MAX_ALERT_ENTRIES);
            self.alert_state = entries.into_iter().collect();
        }

        alerts
    }

    /// Formulate maintenance goals for persistent high-confidence alerts (dry-run only).
    fn run_active_inference_plans(&mut self, alerts: &[AlertEntry]) {
        for alert in alerts {
            if alert.consecutive_cycles >= 3 && alert.confidence > 0.6 {
                let goal_description = format!(
                    "Maintain {} below {} (currently {} predicted {})",
                    alert.metric, alert.threshold, alert.current_value, alert.predicted_value
                );
                let plan = self.active_inference.process_input(&goal_description);
                if let Some(best) = plan.actions.first() {
                    eprintln!(
                        "nix-mind-daemon: maintenance plan (dry-run): {} → {:?} (EFE={:.3})",
                        alert.metric, best.action, best.expected_free_energy
                    );
                    self.maintenance_plan_count += 1;
                }
            }
        }
    }

    /// Build telemetry from hardware info and system snapshot.
    fn build_telemetry(
        hw: Option<&symthaea_nix::observe::hardware::HardwareInfo>,
        snapshot: &SystemStateSnapshot,
    ) -> SystemTelemetry {
        SystemTelemetry {
            disk_used_pct: hw.map_or(0.0, |h| {
                h.disks.first().map_or(0.0, |d| {
                    if d.total_bytes > 0 {
                        d.used_bytes as f64 / d.total_bytes as f64 * 100.0
                    } else {
                        0.0
                    }
                })
            }),
            memory_used_pct: hw.map_or(0.0, |h| {
                if h.memory_total_mb > 0 {
                    let used = h.memory_total_mb.saturating_sub(h.memory_available_mb);
                    used as f64 / h.memory_total_mb as f64 * 100.0
                } else {
                    0.0
                }
            }),
            store_path_count: snapshot.store_path_count.unwrap_or(0) as u64,
            failed_unit_count: snapshot
                .services
                .iter()
                .filter(|(_, s)| *s == ServiceState::Failed)
                .count() as u32,
            load_average_1m: hw.map_or(0.0, |h| h.load_average[0]),
            swap_used_pct: hw.map_or(0.0, |h| {
                if h.swap_total_mb > 0 {
                    h.swap_used_mb as f64 / h.swap_total_mb as f64 * 100.0
                } else {
                    0.0
                }
            }),
        }
    }

    /// Build a DaemonSnapshot for IPC.
    fn to_ipc_snapshot(&mut self) -> DaemonSnapshot {
        let hierarchy = self.world_model.prediction_hierarchy();
        let hierarchy_errors = hierarchy.errors();

        let concerns: Vec<ConcernEntry> = self
            .working_memory
            .items()
            .iter()
            .map(|item| ConcernEntry {
                label: item.label.clone(),
                activation: item.activation,
                source: format!("{:?}", item.source),
            })
            .collect();

        let now = now_secs();
        let alerts = self.build_alerts(now);

        // Record predictions for accuracy tracking (AP)
        let predictions_for_tracking = self.predictive_monitor.predict_all_horizons();
        self.predictive_monitor
            .record_predictions(&predictions_for_tracking);

        // Active inference: formulate maintenance goals (AO)
        self.run_active_inference_plans(&alerts);

        // Export top-10 strongest causal edges
        let top_causal_edges: Vec<CausalEdgeEntry> = self
            .causal_graph
            .top_edges(10)
            .into_iter()
            .map(|e| CausalEdgeEntry {
                from: e.from,
                to: e.to,
                confidence: e.confidence,
            })
            .collect();

        DaemonSnapshot {
            version: symthaea_nix::ipc::SNAPSHOT_VERSION,
            timestamp: now,
            observation_count: self.observation_count,
            anomaly_count: self.anomaly_count,
            hierarchy_errors,
            free_energy: self.world_model.free_energy(),
            is_surprised: self.world_model.free_energy() > 0.3,
            drift_similarity: self
                .prev_state_hv
                .as_ref()
                .map_or(1.0, |prev| self.world_model.system_state().similarity(prev)),
            causal_edge_count: self.causal_graph.edge_count(),
            episodic_count: self.episodic_memory.len(),
            concerns,
            recent_anomalies: self.recent_anomalies.clone(),
            daemon_running: true,
            daemon_pid: std::process::id(),
            support_status: self.last_health_status.map(|s| format!("{:?}", s)),
            recommendation_count: self.last_health_issue_count,
            alerts,
            top_causal_edges,
            memory_used_percent: self.last_memory_pct,
            watchdog_status: self.watchdog_status.clone(),
            degraded: self.degraded,
            prediction_accuracy: self.predictive_monitor.rolling_mae(),
            maintenance_plan_count: self.maintenance_plan_count,
            load_average_1m: self.last_hw_probe.as_ref().map(|h| h.load_average[0]),
            swap_used_percent: self.last_hw_probe.as_ref().and_then(|h| {
                if h.swap_total_mb > 0 {
                    Some(h.swap_used_mb as f64 / h.swap_total_mb as f64 * 100.0)
                } else {
                    None
                }
            }),
        }
    }
}

/// A detected state transition between two snapshots.
struct StateTransition {
    key: String,
    from: String,
    to: String,
    occurred: bool,
    is_recovery: bool,
}

/// Diff two snapshots to find state transitions.
fn detect_transitions(
    before: &SystemStateSnapshot,
    after: &SystemStateSnapshot,
) -> Vec<StateTransition> {
    let mut transitions = Vec::new();

    let before_services: std::collections::HashMap<&str, &ServiceState> = before
        .services
        .iter()
        .map(|(n, s)| (n.as_str(), s))
        .collect();

    for (name, after_state) in &after.services {
        if let Some(before_state) = before_services.get(name.as_str()) {
            if *before_state != after_state {
                let is_recovery =
                    **before_state == ServiceState::Failed && *after_state != ServiceState::Failed;
                transitions.push(StateTransition {
                    key: name.clone(),
                    from: format!("{:?}", before_state),
                    to: format!("{:?}", after_state),
                    occurred: true,
                    is_recovery,
                });
            }
        } else {
            transitions.push(StateTransition {
                key: name.clone(),
                from: "absent".to_string(),
                to: format!("{:?}", after_state),
                occurred: true,
                is_recovery: false,
            });
        }
    }

    if before.generation != after.generation {
        transitions.push(StateTransition {
            key: "generation".to_string(),
            from: before.generation.map_or("none".into(), |g| g.to_string()),
            to: after.generation.map_or("none".into(), |g| g.to_string()),
            occurred: true,
            is_recovery: false,
        });
    }

    transitions
}

/// Check if an anomaly is relevant to a predictive metric (BD).
///
/// Maps anomaly reasons/units to metric names for journal context correlation.
fn anomaly_matches_metric(anomaly: &AnomalyEntry, metric: &str) -> bool {
    let reason_lower = anomaly.reason.to_lowercase();
    let unit_lower = anomaly.unit.to_lowercase();
    match metric {
        "disk_used_pct" => {
            reason_lower.contains("disk")
                || reason_lower.contains("space")
                || reason_lower.contains("storage")
                || reason_lower.contains("no space")
        }
        "memory_used_pct" => {
            reason_lower.contains("memory")
                || reason_lower.contains("oom")
                || reason_lower.contains("killed process")
        }
        "failed_unit_count" => {
            reason_lower.contains("failed")
                || reason_lower.contains("crash")
                || reason_lower.contains("exit code")
                || unit_lower.contains(".service")
        }
        "store_path_count" => {
            reason_lower.contains("store")
                || reason_lower.contains("nix-build")
                || reason_lower.contains("derivation")
        }
        "load_average_1m" => {
            reason_lower.contains("load")
                || reason_lower.contains("cpu")
                || reason_lower.contains("overload")
        }
        "swap_used_pct" => reason_lower.contains("swap") || reason_lower.contains("paging"),
        _ => false,
    }
}

/// POML template for anomaly diagnosis prompts.
const ANOMALY_DIAGNOSIS_POML: &str = r#"<poml version="2.0">
  <metadata>
    <title>Anomaly Diagnosis</title>
    <model-hints><temperature>0.3</temperature><max-tokens>256</max-tokens></model-hints>
  </metadata>
  <variables>
    <let name="unit">{{ unit }}</let>
    <let name="reason">{{ reason }}</let>
    <let name="message">{{ message }}</let>
  </variables>
  <prompt>
    <system>You are a NixOS systemd diagnostician. Be concise (2-3 sentences max).</system>
    <stepwise-instructions>
      <step id="s1">Identify the root cause of the anomaly in unit '{{ unit }}'.</step>
      <step id="s2">Suggest a concrete fix or investigation command.</step>
    </stepwise-instructions>
    <output-format>Plain text: diagnosis followed by suggested fix.</output-format>
  </prompt>
</poml>"#;

/// POML template for resolution verification prompts.
///
/// After a service recovers, ask the LLM to summarize why the recovery happened
/// and whether the fix is permanent or temporary — feeding the response back
/// into the knowledge base for future reference.
const RESOLUTION_VERIFICATION_POML: &str = r#"<poml version="2.0">
  <metadata>
    <title>Resolution Verification</title>
    <model-hints><temperature>0.2</temperature><max-tokens>192</max-tokens></model-hints>
  </metadata>
  <variables>
    <let name="symptom">{{ symptom }}</let>
    <let name="resolution">{{ resolution }}</let>
    <let name="commands">{{ commands }}</let>
  </variables>
  <prompt>
    <system>You are a NixOS reliability analyst. Be concise (2-3 sentences).</system>
    <stepwise-instructions>
      <step id="s1">Analyze why the symptom '{{ symptom }}' was resolved by '{{ resolution }}'.</step>
      <step id="s2">Determine if the fix is permanent or a workaround that may recur.</step>
      <step id="s3">If temporary, suggest a permanent fix.</step>
    </stepwise-instructions>
    <output-format>Plain text: analysis, permanence verdict, optional permanent fix.</output-format>
  </prompt>
</poml>"#;

/// Build a resolution verification prompt using POML template processing.
fn build_resolution_prompt(symptom: &str, resolution: &str, commands: &[String]) -> String {
    let mut proc = PomlProcessor::new("/dev/null");
    if proc
        .load_template_str("resolution_verification", RESOLUTION_VERIFICATION_POML)
        .is_ok()
    {
        let mut ctx = PomlContext::default();
        ctx.variables
            .insert("symptom".into(), PomlValue::from(symptom));
        ctx.variables
            .insert("resolution".into(), PomlValue::from(resolution));
        ctx.variables
            .insert("commands".into(), PomlValue::from(commands.join(", ")));

        if let Ok(result) = proc.process("resolution_verification", &ctx) {
            return result.prompt;
        }
    }
    format!(
        "A NixOS issue '{}' was resolved by: {}\n\
         Commands used: {}\n\n\
         Was this a permanent fix or a temporary workaround? \
         If temporary, suggest a permanent solution (2-3 sentences).",
        symptom,
        resolution,
        commands.join(", ")
    )
}

/// Build an anomaly diagnosis prompt using POML template processing.
fn build_anomaly_prompt(unit: &str, reason: &str, message: &str) -> String {
    let mut proc = PomlProcessor::new("/dev/null");
    if proc
        .load_template_str("anomaly_diagnosis", ANOMALY_DIAGNOSIS_POML)
        .is_ok()
    {
        let mut ctx = PomlContext::default();
        ctx.variables.insert("unit".into(), PomlValue::from(unit));
        ctx.variables
            .insert("reason".into(), PomlValue::from(reason));
        ctx.variables
            .insert("message".into(), PomlValue::from(message));

        if let Ok(result) = proc.process("anomaly_diagnosis", &ctx) {
            return result.prompt;
        }
    }
    format!(
        "A NixOS systemd unit '{}' has an anomaly.\n\
         Anomaly reason: {}\nLog message: {}\n\n\
         Briefly diagnose the likely cause and suggest a fix (2-3 sentences max).",
        unit, reason, message
    )
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn main() -> ! {
    // Initialize structured JSON logging when observability is enabled.
    #[cfg(feature = "observability")]
    init_tracing();

    let config = DaemonConfig::load_default();
    let mut state = DaemonState::new(&config);
    let snapshot_path = default_snapshot_path();

    // Spawn Prometheus metrics HTTP endpoint in a background thread.
    #[cfg(feature = "observability")]
    {
        let metrics_port = config.metrics_port;
        // Eagerly initialize the global metrics registry (fallible).
        match Metrics::try_global() {
            Ok(_) => {
                std::thread::spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!(
                                "nix-mind-daemon: cannot create tokio runtime for metrics: {e}"
                            );
                            return;
                        }
                    };
                    rt.block_on(async {
                        if let Err(e) =
                            symthaea_nix::observability::serve_metrics(metrics_port).await
                        {
                            eprintln!("nix-mind-daemon: metrics server failed: {e}");
                        }
                    });
                });
                eprintln!(
                    "nix-mind-daemon: Prometheus metrics endpoint on port {}",
                    metrics_port
                );
            }
            Err(e) => {
                eprintln!(
                    "nix-mind-daemon: metrics initialization failed, running without metrics: {e}"
                );
            }
        }
    }

    // Restore persisted working memory
    let wm_path = snapshot_path.with_file_name("working_memory.json");
    if let Ok(json) = std::fs::read_to_string(&wm_path) {
        if let Ok(saved) = serde_json::from_str::<symthaea_nix::mind::SavedWorkingMemory>(&json) {
            let item_count = saved.items.len();
            state.working_memory = WorkingMemory::load(&saved, &mut state.codebook);
            eprintln!(
                "nix-mind-daemon: restored {} working memory items",
                item_count
            );
        }
    }

    // Restore persisted dynamic knowledge articles
    let kb_path = snapshot_path.with_file_name("knowledge_learned.json");
    if let Ok(json) = std::fs::read_to_string(&kb_path) {
        if let Some(kb) = state.knowledge_base.as_mut() {
            let before = kb.dynamic_len();
            kb.load_dynamic(&json, &mut state.codebook);
            let loaded = kb.dynamic_len() - before;
            if loaded > 0 {
                eprintln!(
                    "nix-mind-daemon: restored {} learned knowledge articles",
                    loaded
                );
            }
        }
    }

    // Restore persisted predictive history
    let pred_path = snapshot_path.with_file_name("predictive_history.json");
    if let Ok(json) = std::fs::read_to_string(&pred_path) {
        if let Ok(saved) = serde_json::from_str::<SavedPredictiveState>(&json) {
            let sample_count = saved.samples.len();
            state.predictive_monitor = PredictiveMonitor::load(saved, AlertThresholds::default());
            eprintln!(
                "nix-mind-daemon: restored {} predictive samples",
                sample_count
            );
        }
    }

    // Restore persisted causal graph
    let causal_path = snapshot_path.with_file_name("causal_graph.json");
    if let Ok(loaded) = state.causal_graph.load(&causal_path) {
        if loaded > 0 {
            eprintln!(
                "nix-mind-daemon: restored {} causal edges (total: {})",
                loaded,
                state.causal_graph.edge_count()
            );
        }
    }

    eprintln!(
        "nix-mind-daemon: starting continuous awareness (pid {})",
        std::process::id()
    );
    eprintln!(
        "  snapshot every {}s, poll every {}s, surprise threshold {:.2}",
        config.snapshot_interval, config.poll_interval, config.surprise_threshold
    );
    eprintln!("  IPC path: {}", snapshot_path.display());
    eprintln!(
        "  causal graph bootstrapped with {} edges",
        state.causal_graph.edge_count()
    );
    if let Some(ollama) = state.ollama.as_mut() {
        let available = ollama.check_available();
        eprintln!(
            "  Ollama: {} (endpoint: {}, model: {})",
            if available {
                "available"
            } else {
                "unavailable"
            },
            config.ollama_endpoint,
            config.ollama_model,
        );
    }

    let mut last_snapshot = Instant::now() - Duration::from_secs(config.snapshot_interval);
    let mut cycle = 0u64;

    loop {
        cycle += 1;

        if last_snapshot.elapsed() >= Duration::from_secs(config.snapshot_interval) {
            #[cfg(feature = "observability")]
            let _observe_timer = PhaseTimer::start("observe");

            match SystemObserver::snapshot() {
                Ok(snapshot) => {
                    #[cfg(feature = "observability")]
                    let _process_timer = PhaseTimer::start("process_snapshot");

                    state.process_snapshot(snapshot, &config);

                    #[cfg(feature = "observability")]
                    drop(_process_timer);

                    let fe = state.world_model.free_energy();

                    if fe > config.surprise_threshold {
                        eprintln!(
                            "nix-mind-daemon: surprise detected (FE={:.3}), cycle {}",
                            fe, cycle
                        );
                    }

                    // Update observability gauges after snapshot processing.
                    #[cfg(feature = "observability")]
                    {
                        let m = Metrics::global();
                        m.set_free_energy(fe);
                        m.set_causal_edge_count(state.causal_graph.edge_count() as f64);
                        m.set_episodic_count(state.episodic_memory.len() as f64);
                    }
                }
                Err(e) => {
                    eprintln!("nix-mind-daemon: snapshot failed: {}", e);
                }
            }
            last_snapshot = Instant::now();
        }

        {
            #[cfg(feature = "observability")]
            let _journal_timer = PhaseTimer::start("process_journal");

            #[cfg(feature = "observability")]
            let anomaly_count_before = state.anomaly_count;

            state.process_journal(config.journal_batch_size);

            // Track newly detected anomalies.
            #[cfg(feature = "observability")]
            {
                let new_anomalies = state.anomaly_count - anomaly_count_before;
                if new_anomalies > 0 {
                    let m = Metrics::global();
                    for _ in 0..new_anomalies {
                        m.inc_anomalies();
                    }
                }
            }
        }

        if cycle % config.ipc_write_interval == 0 {
            #[cfg(feature = "observability")]
            let _ipc_timer = PhaseTimer::start("ipc_write");

            state.refresh_watchdog_status(&snapshot_path);
            let ipc_snap = state.to_ipc_snapshot();

            // Update consciousness-level and phi gauges from the IPC snapshot.
            #[cfg(feature = "observability")]
            {
                let m = Metrics::global();
                // consciousness_level: derive from hierarchy errors (lower error = higher consciousness)
                // The snapshot doesn't expose a single "consciousness level" scalar, so we
                // use 1.0 - mean(hierarchy_errors) clamped to [0,1] as a proxy.
                if !ipc_snap.hierarchy_errors.is_empty() {
                    let mean_err: f64 = ipc_snap.hierarchy_errors.iter().sum::<f64>()
                        / ipc_snap.hierarchy_errors.len() as f64;
                    m.set_consciousness_level((1.0 - mean_err).clamp(0.0, 1.0));
                }
                // phi_value: use drift_similarity as a proxy (higher = more integrated)
                m.set_phi_value(ipc_snap.drift_similarity as f64);
                // Refresh edge/episodic counts from the snapshot
                m.set_causal_edge_count(ipc_snap.causal_edge_count as f64);
                m.set_episodic_count(ipc_snap.episodic_count as f64);

                // Increment gate_vetoes_total for any critical alerts (gate veto proxy)
                let critical_count = ipc_snap
                    .alerts
                    .iter()
                    .filter(|a| matches!(a.severity, AlertSeverity::Critical))
                    .count();
                if critical_count > 0 {
                    for _ in 0..critical_count {
                        m.inc_gate_vetoes();
                    }
                }
            }

            if let Err(e) = ipc_snap.write_to(&snapshot_path) {
                eprintln!("nix-mind-daemon: IPC write failed: {}", e);
            }

            let wm_path = snapshot_path.with_file_name("working_memory.json");
            let saved = state.working_memory.save();
            if let Ok(json) = serde_json::to_string_pretty(&saved) {
                if let Err(e) = std::fs::write(&wm_path, json) {
                    state.persist_error_count += 1;
                    eprintln!("nix-mind-daemon: working_memory write failed: {e}");
                }
            }

            let pred_path = snapshot_path.with_file_name("predictive_history.json");
            let pred_saved = state.predictive_monitor.save();
            if let Ok(json) = serde_json::to_string_pretty(&pred_saved) {
                if let Err(e) = std::fs::write(&pred_path, json) {
                    state.persist_error_count += 1;
                    eprintln!("nix-mind-daemon: predictive_history write failed: {e}");
                }
            }

            if let Some(kb) = &state.knowledge_base {
                if kb.dynamic_len() > 0 {
                    let kb_path = snapshot_path.with_file_name("knowledge_learned.json");
                    if let Err(e) = std::fs::write(&kb_path, kb.save_dynamic()) {
                        state.persist_error_count += 1;
                        eprintln!("nix-mind-daemon: knowledge_learned write failed: {e}");
                    }
                }
            }

            // Persist causal graph
            let causal_path = snapshot_path.with_file_name("causal_graph.json");
            if let Err(e) = state.causal_graph.save(&causal_path) {
                state.persist_error_count += 1;
                eprintln!("nix-mind-daemon: causal_graph write failed: {e}");
            }
        }

        // Increment the cycle counter for observability.
        #[cfg(feature = "observability")]
        Metrics::global().inc_consciousness_cycles();

        thread::sleep(Duration::from_secs(config.poll_interval));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DaemonConfig {
        DaemonConfig {
            enable_knowledge_learning: false,
            ..DaemonConfig::default()
        }
    }

    #[test]
    fn test_build_telemetry_from_hardware() {
        use symthaea_nix::observe::hardware::{DiskInfo, HardwareInfo};

        let hw = HardwareInfo {
            cpu_model: "Test".into(),
            cpu_cores: 4,
            memory_total_mb: 16000,
            memory_available_mb: 4000,
            gpus: vec![],
            disks: vec![DiskInfo {
                device: "/dev/sda1".into(),
                mount_point: "/".into(),
                total_bytes: 100_000_000_000,
                used_bytes: 75_000_000_000,
            }],
            load_average: [2.5, 1.5, 1.0],
            swap_total_mb: 4096,
            swap_used_mb: 1024,
        };
        let snapshot = SystemStateSnapshot {
            services: vec![
                ("ok.service".into(), ServiceState::Running),
                ("broken.service".into(), ServiceState::Failed),
            ],
            store_path_count: Some(50_000),
            ..Default::default()
        };

        let telemetry = DaemonState::build_telemetry(Some(&hw), &snapshot);
        assert!((telemetry.disk_used_pct - 75.0).abs() < 0.1);
        assert!((telemetry.memory_used_pct - 75.0).abs() < 0.1);
        assert_eq!(telemetry.store_path_count, 50_000);
        assert_eq!(telemetry.failed_unit_count, 1);
        assert!((telemetry.load_average_1m - 2.5).abs() < 1e-6);
        assert!((telemetry.swap_used_pct - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_build_telemetry_no_hardware() {
        let snapshot = SystemStateSnapshot {
            store_path_count: Some(1000),
            ..Default::default()
        };
        let telemetry = DaemonState::build_telemetry(None, &snapshot);
        assert!((telemetry.disk_used_pct).abs() < 1e-6);
        assert!((telemetry.memory_used_pct).abs() < 1e-6);
        assert_eq!(telemetry.store_path_count, 1000);
    }

    #[test]
    fn test_build_alerts_empty_monitor() {
        let config = test_config();
        let mut state = DaemonState::new(&config);
        let alerts = state.build_alerts(1700000000);
        // With no data ingested, should produce no alerts
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_build_alerts_rising_disk() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        // Feed rising disk data
        for i in 0..20 {
            state.predictive_monitor.ingest(SystemTelemetry {
                disk_used_pct: 70.0 + i as f64,
                memory_used_pct: 40.0,
                store_path_count: 50_000,
                failed_unit_count: 0,
                load_average_1m: 0.5,
                swap_used_pct: 5.0,
            });
        }

        let alerts = state.build_alerts(1700000000);
        // Rising disk from 70→89% should trigger some alerts
        let disk_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.metric == "disk_used_pct")
            .collect();
        assert!(
            !disk_alerts.is_empty(),
            "Rising disk should generate alerts"
        );
        // All alerts should have timestamps set
        for alert in &alerts {
            assert!(alert.first_seen > 0);
            assert!(alert.last_seen >= alert.first_seen);
        }
    }

    #[test]
    fn test_build_alerts_consecutive_tracking() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        for i in 0..20 {
            state.predictive_monitor.ingest(SystemTelemetry {
                disk_used_pct: 70.0 + i as f64,
                memory_used_pct: 40.0,
                store_path_count: 50_000,
                failed_unit_count: 0,
                load_average_1m: 0.5,
                swap_used_pct: 5.0,
            });
        }

        let alerts1 = state.build_alerts(1700000000);
        let alerts2 = state.build_alerts(1700000060);

        // Second call should have higher consecutive_cycles
        for a2 in &alerts2 {
            if let Some(a1) = alerts1
                .iter()
                .find(|a| a.metric == a2.metric && a.hours_ahead == a2.hours_ahead)
            {
                assert!(
                    a2.consecutive_cycles >= a1.consecutive_cycles,
                    "Consecutive cycles should increase"
                );
            }
        }
    }

    #[test]
    fn test_run_active_inference_plans_no_persistent() {
        let config = test_config();
        let mut state = DaemonState::new(&config);
        assert_eq!(state.maintenance_plan_count, 0);

        // Non-persistent alert (1 cycle, below threshold)
        let alerts = vec![AlertEntry {
            metric: "disk_used_pct".into(),
            current_value: 85.0,
            predicted_value: 95.0,
            hours_ahead: 24.0,
            threshold: 90.0,
            confidence: 0.8,
            recommended_action: Some("nix-collect-garbage -d".into()),
            severity: AlertSeverity::Warning,
            first_seen: 1700000000,
            last_seen: 1700000000,
            consecutive_cycles: 1, // below threshold of 3
            prev_predicted_value: None,
            journal_context: vec![],
        }];
        state.run_active_inference_plans(&alerts);
        assert_eq!(
            state.maintenance_plan_count, 0,
            "1-cycle alert should not trigger plan"
        );
    }

    #[test]
    fn test_run_active_inference_plans_persistent() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        let alerts = vec![AlertEntry {
            metric: "disk_used_pct".into(),
            current_value: 85.0,
            predicted_value: 95.0,
            hours_ahead: 24.0,
            threshold: 90.0,
            confidence: 0.8,
            recommended_action: Some("nix-collect-garbage -d".into()),
            severity: AlertSeverity::Critical,
            first_seen: 1700000000,
            last_seen: 1700000300,
            consecutive_cycles: 5, // persistent
            prev_predicted_value: Some(93.0),
            journal_context: vec![],
        }];
        state.run_active_inference_plans(&alerts);
        assert!(
            state.maintenance_plan_count > 0,
            "Persistent alert should trigger maintenance plan"
        );
    }

    #[test]
    fn test_anomaly_matches_metric() {
        let disk_anomaly = AnomalyEntry {
            score: 0.8,
            reason: "No space left on device".into(),
            unit: "nix-daemon.service".into(),
            error_type: Some("disk_full".into()),
            suggestion: Some("Run nix-collect-garbage".into()),
        };
        assert!(anomaly_matches_metric(&disk_anomaly, "disk_used_pct"));
        assert!(!anomaly_matches_metric(&disk_anomaly, "memory_used_pct"));

        let oom_anomaly = AnomalyEntry {
            score: 0.9,
            reason: "OOM killer invoked".into(),
            unit: "nginx.service".into(),
            error_type: None,
            suggestion: None,
        };
        assert!(anomaly_matches_metric(&oom_anomaly, "memory_used_pct"));
        assert!(!anomaly_matches_metric(&oom_anomaly, "disk_used_pct"));

        let service_anomaly = AnomalyEntry {
            score: 0.7,
            reason: "Process crashed with exit code 1".into(),
            unit: "myapp.service".into(),
            error_type: None,
            suggestion: None,
        };
        assert!(anomaly_matches_metric(
            &service_anomaly,
            "failed_unit_count"
        ));
    }

    #[test]
    fn test_journal_context_populated() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        // Add anomalies that should match disk alerts
        state.recent_anomalies.push(AnomalyEntry {
            score: 0.8,
            reason: "No space left on device".into(),
            unit: "nix-daemon.service".into(),
            error_type: Some("disk_full".into()),
            suggestion: Some("Run garbage collection".into()),
        });

        // Feed rising disk data to generate alerts
        for i in 0..20 {
            state.predictive_monitor.ingest(SystemTelemetry {
                disk_used_pct: 70.0 + i as f64,
                memory_used_pct: 40.0,
                store_path_count: 50_000,
                failed_unit_count: 0,
                load_average_1m: 0.5,
                swap_used_pct: 5.0,
            });
        }

        let alerts = state.build_alerts(1700000000);
        let disk_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.metric == "disk_used_pct")
            .collect();

        // Disk alerts should now have journal context from the anomaly
        let has_context = disk_alerts.iter().any(|a| !a.journal_context.is_empty());
        assert!(
            has_context,
            "Disk alerts should have journal context from disk anomaly"
        );
    }

    #[test]
    fn test_degraded_mode_initial_state() {
        let config = test_config();
        let state = DaemonState::new(&config);
        assert!(!state.degraded, "Should not start in degraded mode");
        assert!(state.last_hw_probe.is_none(), "No cached probe initially");
    }

    #[test]
    fn test_degraded_mode_cached_hw_used() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        // Simulate a successful hardware probe
        let hw = symthaea_nix::observe::hardware::HardwareInfo {
            cpu_model: "Test CPU".into(),
            cpu_cores: 4,
            memory_total_mb: 16000,
            memory_available_mb: 8000,
            gpus: vec![],
            disks: vec![],
            load_average: [1.0, 0.8, 0.5],
            swap_total_mb: 4096,
            swap_used_mb: 512,
        };
        state.last_hw_probe = Some(hw.clone());
        state.degraded = false;

        // Simulate probe failure → should use cached data
        state.degraded = true;
        let cached = state.last_hw_probe.clone();
        assert!(cached.is_some(), "Cached hw should be available");
        assert_eq!(cached.unwrap().cpu_cores, 4);
    }

    #[test]
    fn test_degraded_flag_in_ipc_snapshot() {
        let config = test_config();
        let mut state = DaemonState::new(&config);
        state.degraded = true;

        let ipc = state.to_ipc_snapshot();
        assert!(ipc.degraded, "IPC snapshot should reflect degraded state");

        state.degraded = false;
        let ipc = state.to_ipc_snapshot();
        assert!(!ipc.degraded, "IPC snapshot should reflect recovered state");
    }

    #[test]
    fn test_degraded_recovery_clears_flag() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        // Simulate degraded state with cached data
        let hw = symthaea_nix::observe::hardware::HardwareInfo {
            cpu_model: "Test CPU".into(),
            cpu_cores: 8,
            memory_total_mb: 32000,
            memory_available_mb: 16000,
            gpus: vec![],
            disks: vec![],
            load_average: [0.5, 0.3, 0.2],
            swap_total_mb: 8192,
            swap_used_mb: 100,
        };
        state.last_hw_probe = Some(hw);
        state.degraded = true;

        // Simulate successful probe (recovery)
        state.degraded = false;
        assert!(!state.degraded);

        let ipc = state.to_ipc_snapshot();
        assert!(!ipc.degraded);
    }

    #[test]
    fn test_load_and_swap_from_cached_hw_in_snapshot() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        let hw = symthaea_nix::observe::hardware::HardwareInfo {
            cpu_model: "Test".into(),
            cpu_cores: 4,
            memory_total_mb: 16000,
            memory_available_mb: 8000,
            gpus: vec![],
            disks: vec![],
            load_average: [3.5, 2.0, 1.0],
            swap_total_mb: 4096,
            swap_used_mb: 2048,
        };
        state.last_hw_probe = Some(hw);

        let ipc = state.to_ipc_snapshot();
        assert!((ipc.load_average_1m.unwrap() - 3.5).abs() < 1e-6);
        assert!((ipc.swap_used_percent.unwrap() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_build_resolution_prompt_via_poml() {
        let prompt = build_resolution_prompt(
            "nginx.service crashed",
            "Service restarted automatically",
            &["systemctl restart nginx".to_string()],
        );
        assert!(
            prompt.contains("nginx.service crashed"),
            "Prompt should contain the symptom"
        );
        assert!(
            prompt.contains("restart"),
            "Prompt should contain the resolution"
        );
    }

    #[test]
    fn test_build_anomaly_prompt_via_poml() {
        let prompt = build_anomaly_prompt(
            "nix-daemon.service",
            "No space left on device",
            "write error at /nix/store",
        );
        // The POML template substitutes {{ unit }} with the unit name
        assert!(
            prompt.contains("nix-daemon.service"),
            "Prompt should contain unit name. Got: {}",
            prompt
        );
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_learn_from_resolution_without_ollama() {
        let config = DaemonConfig {
            enable_knowledge_learning: true,
            ..DaemonConfig::default()
        };
        let mut state = DaemonState::new(&config);
        // Disable ollama for this test
        state.ollama = None;

        state.learn_from_resolution(
            "test.service failure",
            "Restarted the service",
            vec!["systemctl restart test".into()],
        );

        // Should have learned the article (without Ollama enrichment)
        let kb = state.knowledge_base.as_ref().unwrap();
        assert_eq!(kb.dynamic_len(), 1);
    }

    #[test]
    fn test_persist_error_count_starts_at_zero() {
        let state = DaemonState::new(&test_config());
        assert_eq!(state.persist_error_count, 0);
    }

    #[test]
    fn test_alert_state_capacity_bounded() {
        let config = test_config();
        let mut state = DaemonState::new(&config);

        // Insert many alert tracking entries
        for i in 0..2000 {
            state.alert_state.insert(
                format!("metric_{}@1h", i),
                AlertTracking {
                    first_seen: i as u64,
                    consecutive_cycles: 1,
                    prev_predicted_value: 0.5,
                },
            );
        }
        assert_eq!(state.alert_state.len(), 2000);

        // After build_alerts with no active predictions, all should be pruned
        state.predictive_monitor = PredictiveMonitor::with_defaults();
        let alerts = state.build_alerts(100);
        // All entries pruned since no predictions match
        assert!(
            state.alert_state.is_empty(),
            "Alert state should be pruned when no predictions are active, got {}",
            state.alert_state.len()
        );
        // Should return empty alerts
        assert!(alerts.is_empty());
    }
}
