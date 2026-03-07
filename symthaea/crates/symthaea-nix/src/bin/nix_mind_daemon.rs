//! nix-mind-daemon: background daemon for continuous NixOS awareness.
//!
//! Periodically observes the system state, encodes it into the HDC world model,
//! and updates the causal graph with any state transitions. High-surprise events
//! (large prediction errors) are stored in episodic memory for future reference.
//!
//! The daemon writes a `DaemonSnapshot` to disk on every cycle for TUI consumption.
//!
//! Designed to run as a systemd service via the NixOS module.

use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use symthaea_core::hdc::ContinuousHV;
use symthaea_nix::encoding::{NixCodebook, ServiceState, SystemStateEncoder, SystemStateSnapshot};
use symthaea_nix::ipc::{
    default_snapshot_path, AnomalyEntry, ConcernEntry, DaemonConfig, DaemonSnapshot,
};
use symthaea_nix::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use symthaea_nix::mind::episodic_memory::{EpisodeOutcome, NixEpisodicMemory, SystemEpisode};
use symthaea_nix::mind::ollama_bridge::{OllamaBridge, OllamaBridgeConfig};
use symthaea_nix::mind::working_memory::{MemorySource, WorkingMemory};
use symthaea_nix::mind::{JournalAnomalyDetector, NixWorldModel};
use symthaea_nix::observe::journal::JournalObserver;
use symthaea_nix::observe::SystemObserver;
use symthaea_nix::support::health_check::{HealthAssessor, HealthStatus};
use symthaea_nix::support::knowledge::{DynamicKnowledgeArticle, KnowledgeBase, KnowledgeCategory};
use symthaea_nix::support::poml::{PomlContext, PomlProcessor, PomlValue};
use symthaea_nix::support::predictive::{
    AlertThresholds, PredictiveMonitor, SavedPredictiveState, SystemTelemetry,
};

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
        }
    }

    /// Learn from a resolved anomaly by creating a dynamic knowledge article.
    fn learn_from_resolution(
        &mut self,
        symptom: &str,
        resolution: &str,
        commands: Vec<String>,
    ) {
        let kb = match self.knowledge_base.as_mut() {
            Some(kb) => kb,
            None => return,
        };

        let id = format!("learned_{}", now_secs());
        let article = DynamicKnowledgeArticle {
            id,
            title: format!("Resolved: {}", symptom),
            category: KnowledgeCategory::ServiceIssue,
            symptoms: vec![symptom.to_string()],
            solution: resolution.to_string(),
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
                &format!("Service {} recovered automatically ({} → {})", key, from, to),
                vec![format!("systemctl status {}", key)],
            );
        }

        // Run health assessment
        let hw = symthaea_nix::observe::hardware::HardwareObserver::probe().ok();
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
        let telemetry = SystemTelemetry {
            disk_used_pct: hw.as_ref().map_or(0.0, |h| {
                h.disks.first().map_or(0.0, |d| {
                    if d.total_bytes > 0 {
                        d.used_bytes as f64 / d.total_bytes as f64 * 100.0
                    } else {
                        0.0
                    }
                })
            }),
            memory_used_pct: hw.as_ref().map_or(0.0, |h| {
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
        };
        self.predictive_monitor.ingest(telemetry);

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
                );
            }

            self.recent_anomalies.push(AnomalyEntry {
                score: anomaly.anomaly_score as f64,
                reason: anomaly.reason.clone(),
                unit: anomaly.entry.unit.clone(),
            });
        }

        if self.recent_anomalies.len() > 20 {
            let excess = self.recent_anomalies.len() - 20;
            self.recent_anomalies.drain(..excess);
        }
    }

    /// Query Ollama for anomaly diagnosis and learn from the response.
    fn query_ollama_for_anomaly(&mut self, unit: &str, reason: &str, message: &str) {
        let ollama = match self.ollama.as_mut() {
            Some(o) => o,
            None => return,
        };

        let prompt = build_anomaly_prompt(unit, reason, message);

        if let Some(response) = ollama.query(&prompt) {
            self.learn_from_resolution(
                &format!("{}: {}", unit, reason),
                &response.text,
                vec![format!("journalctl -u {} --since '5 min ago'", unit)],
            );
            eprintln!(
                "nix-mind-daemon: Ollama diagnosed anomaly in {} ({}ms, model: {})",
                unit, response.duration_ms, response.model_used
            );
        }
    }

    /// Build a DaemonSnapshot for IPC.
    fn to_ipc_snapshot(&self) -> DaemonSnapshot {
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

        DaemonSnapshot {
            timestamp: now_secs(),
            observation_count: self.observation_count,
            anomaly_count: self.anomaly_count,
            hierarchy_errors,
            free_energy: self.world_model.free_energy(),
            is_surprised: self.world_model.free_energy() > 0.3,
            drift_similarity: 0.95,
            causal_edge_count: self.causal_graph.edge_count(),
            episodic_count: self.episodic_memory.len(),
            concerns,
            recent_anomalies: self.recent_anomalies.clone(),
            daemon_running: true,
            daemon_pid: std::process::id(),
            support_status: self.last_health_status.map(|s| format!("{:?}", s)),
            recommendation_count: self.last_health_issue_count,
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
                let is_recovery = **before_state == ServiceState::Failed
                    && *after_state != ServiceState::Failed;
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

/// Build an anomaly diagnosis prompt using POML template processing.
fn build_anomaly_prompt(unit: &str, reason: &str, message: &str) -> String {
    let mut proc = PomlProcessor::new("/dev/null");
    if proc
        .load_template_str("anomaly_diagnosis", ANOMALY_DIAGNOSIS_POML)
        .is_ok()
    {
        let mut ctx = PomlContext::default();
        ctx.variables
            .insert("unit".into(), PomlValue::from(unit));
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
    let config = DaemonConfig::load_default();
    let mut state = DaemonState::new(&config);
    let snapshot_path = default_snapshot_path();

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
            if available { "available" } else { "unavailable" },
            config.ollama_endpoint,
            config.ollama_model,
        );
    }

    let mut last_snapshot = Instant::now() - Duration::from_secs(config.snapshot_interval);
    let mut cycle = 0u64;

    loop {
        cycle += 1;

        if last_snapshot.elapsed() >= Duration::from_secs(config.snapshot_interval) {
            match SystemObserver::snapshot() {
                Ok(snapshot) => {
                    state.process_snapshot(snapshot, &config);
                    let fe = state.world_model.free_energy();

                    if fe > config.surprise_threshold {
                        eprintln!(
                            "nix-mind-daemon: surprise detected (FE={:.3}), cycle {}",
                            fe, cycle
                        );
                    }
                }
                Err(e) => {
                    eprintln!("nix-mind-daemon: snapshot failed: {}", e);
                }
            }
            last_snapshot = Instant::now();
        }

        state.process_journal(config.journal_batch_size);

        if cycle % config.ipc_write_interval == 0 {
            let ipc_snap = state.to_ipc_snapshot();
            if let Err(e) = ipc_snap.write_to(&snapshot_path) {
                eprintln!("nix-mind-daemon: IPC write failed: {}", e);
            }

            let wm_path = snapshot_path.with_file_name("working_memory.json");
            let saved = state.working_memory.save();
            if let Ok(json) = serde_json::to_string_pretty(&saved) {
                let _ = std::fs::write(&wm_path, json);
            }

            let pred_path = snapshot_path.with_file_name("predictive_history.json");
            let pred_saved = state.predictive_monitor.save();
            if let Ok(json) = serde_json::to_string_pretty(&pred_saved) {
                let _ = std::fs::write(&pred_path, json);
            }

            if let Some(kb) = &state.knowledge_base {
                if kb.dynamic_len() > 0 {
                    let kb_path = snapshot_path.with_file_name("knowledge_learned.json");
                    let _ = std::fs::write(&kb_path, kb.save_dynamic());
                }
            }
        }

        thread::sleep(Duration::from_secs(config.poll_interval));
    }
}
