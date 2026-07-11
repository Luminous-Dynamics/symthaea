// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca → Mycelix Factcheck Bridge
//!
//! Extracts verifiable claims from Broca-generated text, submits them
//! for verification against the Mycelix knowledge graph, and modulates
//! Broca output based on verdicts.
//!
//! The bridge operates across two cycle phases:
//! - **Phase B (dynamics)**: After Broca generation, extract claims and submit
//! - **Phase OUTPUT**: Apply any pending verdict from previous cycle
//!
//! Feature-gated behind `mycelix` — factcheck only active when the Mycelix
//! conductor bridge is available.
//!
//! Science: Mercier & Sperber (2011) — argumentative theory of reasoning;
//! verification improves epistemic quality of language production.

use std::collections::VecDeque;
use std::sync::{Mutex, mpsc};

// ── Named Constants ──────────────────────────────────────────────────────

/// EMA smoothing factor for factcheck accuracy tracking.
/// Alpha = 0.05 gives ~20-cycle half-life: stable enough to track
/// trends without oscillating on single verdicts.
const FACTCHECK_ACCURACY_EMA_ALPHA: f32 = 0.05;

/// Maximum pending claims in the verification queue.
/// Bounded to prevent unbounded growth if conductor is slow.
const MAX_PENDING_CLAIMS: usize = 32;

/// Minimum claim length (chars) to consider for verification.
/// Very short fragments are unlikely to be verifiable statements.
const MIN_CLAIM_LENGTH: usize = 20;

/// Maximum claims to extract from a single generation.
/// Limits verification load per cycle.
const MAX_CLAIMS_PER_GENERATION: usize = 8;

/// Confidence threshold below which a False verdict suppresses output.
/// Only suppress when the factcheck system is highly confident the claim is false.
const SUPPRESS_CONFIDENCE_THRESHOLD: f32 = 0.85;

/// Cadence penalty for unverified or mixed-verdict claims.
/// Slows generation to allow more careful formulation.
const CADENCE_PENALTY_UNVERIFIED: f32 = 0.15;

/// Cadence penalty for false-verdict claims (stronger).
const CADENCE_PENALTY_FALSE: f32 = 0.30;

/// History ring buffer capacity for verdict records.
const VERDICT_HISTORY_CAP: usize = 64;

// ── Types ────────────────────────────────────────────────────────────────

/// Verdict from the Mycelix factcheck zome.
#[derive(Debug, Clone, PartialEq)]
pub enum FactCheckVerdict {
    /// Claim is supported by knowledge graph evidence.
    True,
    /// Claim is contradicted by knowledge graph evidence.
    False,
    /// Mixed or insufficient evidence.
    Mixed,
    /// No matching claims found in the knowledge graph.
    Unknown,
}

impl FactCheckVerdict {
    /// Whether this verdict represents a positive verification.
    pub fn is_verified(&self) -> bool {
        matches!(self, FactCheckVerdict::True)
    }

    /// Whether this verdict should reduce confidence.
    pub fn is_negative(&self) -> bool {
        matches!(self, FactCheckVerdict::False)
    }
}

/// Result of factchecking a single claim.
#[derive(Debug, Clone)]
pub struct FactCheckResult {
    /// The original claim text.
    pub claim: String,
    /// Verdict from the knowledge graph.
    pub verdict: FactCheckVerdict,
    /// Confidence of the verdict (0.0–1.0).
    pub confidence: f32,
    /// Source claim IDs from the knowledge graph that support the verdict.
    pub source_ids: Vec<String>,
}

/// Modulation signal applied to Broca output after factchecking.
#[derive(Debug, Clone, Default)]
pub struct BrocaModulation {
    /// Confidence multiplier (0.0–1.0): reduces Broca confidence for unverified claims.
    pub confidence_scale: f32,
    /// Whether to suppress output entirely (verdict: False with high confidence).
    pub suppress: bool,
    /// Cadence adjustment: positive = slow down generation for uncertain claims.
    pub cadence_penalty: f32,
    /// Number of claims that were verified this cycle.
    pub claims_verified: u32,
    /// Number of claims that were suppressed this cycle.
    pub claims_suppressed: u32,
}

/// Historical record of a single verdict for telemetry.
#[derive(Debug, Clone)]
pub struct VerdictRecord {
    /// Cycle number when the verdict was received.
    pub cycle: u64,
    /// The claim that was checked.
    pub claim_snippet: String,
    /// Resulting verdict.
    pub verdict: FactCheckVerdict,
    /// Verdict confidence.
    pub confidence: f32,
}

// Re-export the telemetry type from the types module.
pub use super::types::FactcheckTelemetry;

// ── Bridge ───────────────────────────────────────────────────────────────

/// Broca → Mycelix factcheck bridge.
///
/// Extracts verifiable claims from Broca output, queues them for
/// asynchronous verification against the Mycelix knowledge graph,
/// and modulates Broca confidence/cadence based on verdicts.
#[derive(Debug)]
pub struct BrocaFactcheckBridge {
    /// Claims pending verification (FIFO queue).
    pending_claims: VecDeque<String>,
    /// Most recent verdicts received from the conductor.
    pending_verdicts: Vec<FactCheckResult>,
    /// Running accuracy EMA.
    accuracy_ema: f32,
    /// Lifetime counters.
    total_submitted: u64,
    total_verified: u64,
    total_suppressed: u64,
    /// Verdict history ring buffer.
    verdict_history: VecDeque<VerdictRecord>,
    /// Whether the bridge is connected to a live conductor.
    connected: bool,
    /// Receiver for verdicts from async conductor bridge.
    /// Drained each cycle in `process_verdicts()`.
    verdict_rx: Option<Mutex<mpsc::Receiver<FactCheckResult>>>,
    /// Sender for submitting claims to the conductor bridge.
    /// Cloned by the async conductor task.
    claim_tx: Option<Mutex<mpsc::Sender<String>>>,
}

/// Channel handles for connecting the factcheck bridge to an async conductor.
///
/// The conductor bridge task holds:
/// - `claim_rx`: receives claims to verify against the knowledge graph
/// - `verdict_tx`: sends verification results back to the CLS
pub struct FactcheckChannels {
    /// Receiver for claims to verify (held by conductor bridge).
    pub claim_rx: mpsc::Receiver<String>,
    /// Sender for verdicts (held by conductor bridge).
    pub verdict_tx: mpsc::Sender<FactCheckResult>,
}

impl Default for BrocaFactcheckBridge {
    fn default() -> Self {
        Self {
            pending_claims: VecDeque::with_capacity(MAX_PENDING_CLAIMS),
            pending_verdicts: Vec::new(),
            accuracy_ema: 0.5, // Start neutral
            total_submitted: 0,
            total_verified: 0,
            total_suppressed: 0,
            verdict_history: VecDeque::with_capacity(VERDICT_HISTORY_CAP),
            connected: false,
            verdict_rx: None,
            claim_tx: None,
        }
    }
}

impl BrocaFactcheckBridge {
    /// Create a new factcheck bridge.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create mpsc channels for connecting to an async conductor bridge.
    ///
    /// Returns `FactcheckChannels` containing the other ends of the channels.
    /// The conductor bridge task should:
    /// 1. Read claims from `claim_rx`
    /// 2. Query the factcheck zome
    /// 3. Send results through `verdict_tx`
    pub fn create_channels(&mut self) -> FactcheckChannels {
        let (claim_tx, claim_rx) = mpsc::channel::<String>();
        let (verdict_tx, verdict_rx) = mpsc::channel::<FactCheckResult>();
        self.claim_tx = Some(Mutex::new(claim_tx));
        self.verdict_rx = Some(Mutex::new(verdict_rx));
        self.connected = true;
        FactcheckChannels {
            claim_rx,
            verdict_tx,
        }
    }

    /// Mark the bridge as connected to a live conductor.
    pub fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
    }

    /// Whether the bridge has a live conductor connection.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Current accuracy EMA.
    pub fn accuracy_ema(&self) -> f32 {
        self.accuracy_ema
    }

    /// Get current telemetry snapshot.
    pub fn telemetry(&self) -> FactcheckTelemetry {
        FactcheckTelemetry {
            accuracy_ema: self.accuracy_ema,
            total_claims_submitted: self.total_submitted,
            total_claims_verified: self.total_verified,
            total_claims_suppressed: self.total_suppressed,
            claims_this_cycle: 0, // Updated by caller after process_verdicts
            suppressed_this_cycle: false,
            cadence_penalty: 0.0,
            pending_count: self.pending_claims.len(),
        }
    }

    // ── Phase B: Extract & Submit ────────────────────────────────────

    /// Extract verifiable claims from Broca-generated text.
    ///
    /// Uses sentence-boundary heuristics to find declarative statements
    /// that could be fact-checked. Filters out questions, commands,
    /// and very short fragments.
    pub fn extract_claims(text: &str) -> Vec<String> {
        let mut claims = Vec::new();

        // Split on sentence boundaries
        for sentence in text.split_inclusive(&['.', '!'][..]) {
            let trimmed = sentence.trim();

            // Skip too short
            if trimmed.len() < MIN_CLAIM_LENGTH {
                continue;
            }

            // Skip questions
            if trimmed.ends_with('?') {
                continue;
            }

            // Skip imperative/command patterns (starts with verb)
            let lower = trimmed.to_lowercase();
            if lower.starts_with("please ")
                || lower.starts_with("let ")
                || lower.starts_with("try ")
                || lower.starts_with("note ")
                || lower.starts_with("remember ")
            {
                continue;
            }

            // Skip hedged/subjective statements
            if lower.starts_with("i think ")
                || lower.starts_with("i feel ")
                || lower.starts_with("i believe ")
                || lower.starts_with("in my opinion")
                || lower.starts_with("maybe ")
                || lower.starts_with("perhaps ")
            {
                continue;
            }

            // Accept declarative statements
            claims.push(trimmed.to_string());

            if claims.len() >= MAX_CLAIMS_PER_GENERATION {
                break;
            }
        }

        claims
    }

    /// Submit extracted claims for verification.
    ///
    /// Claims are queued internally. In a full conductor integration,
    /// these would be sent asynchronously to the factcheck zome.
    /// Returns the number of claims queued.
    pub fn submit_for_verification(&mut self, claims: &[String]) -> usize {
        let mut queued = 0;
        for claim in claims {
            if self.pending_claims.len() >= MAX_PENDING_CLAIMS {
                // Evict oldest to make room
                self.pending_claims.pop_front();
            }
            self.pending_claims.push_back(claim.clone());
            self.total_submitted += 1;
            queued += 1;

            // Forward to conductor bridge if channel is connected
            if let Some(ref tx) = self.claim_tx {
                if let Ok(guard) = tx.lock() {
                    let _ = guard.send(claim.clone());
                }
            }
        }
        queued
    }

    /// Inject verdicts received from the Mycelix conductor.
    ///
    /// Called by the conductor bridge when factcheck results arrive.
    pub fn receive_verdicts(&mut self, verdicts: Vec<FactCheckResult>) {
        self.pending_verdicts.extend(verdicts);
    }

    // ── Phase OUTPUT: Process Verdicts & Modulate ────────────────────

    /// Process any pending verdicts and compute modulation for Broca.
    ///
    /// Should be called during the output phase, before the language
    /// output is drained into `CycleResult`.
    pub fn process_verdicts(&mut self, cycle: u64) -> BrocaModulation {
        // Drain any verdicts from the conductor channel
        if let Some(ref rx) = self.verdict_rx {
            if let Ok(guard) = rx.lock() {
                while let Ok(result) = guard.try_recv() {
                    self.pending_verdicts.push(result);
                }
            }
        }

        if self.pending_verdicts.is_empty() {
            return BrocaModulation {
                confidence_scale: 1.0,
                ..Default::default()
            };
        }

        let verdicts = std::mem::take(&mut self.pending_verdicts);
        let mut modulation = BrocaModulation {
            confidence_scale: 1.0,
            suppress: false,
            cadence_penalty: 0.0,
            claims_verified: 0,
            claims_suppressed: 0,
        };

        let mut any_false_high_conf = false;
        let mut max_cadence_penalty: f32 = 0.0;

        for result in &verdicts {
            // Update accuracy EMA
            let accuracy_signal = match result.verdict {
                FactCheckVerdict::True => 1.0,
                FactCheckVerdict::False => 0.0,
                FactCheckVerdict::Mixed => 0.5,
                FactCheckVerdict::Unknown => 0.5, // Neutral for unknown
            };
            self.accuracy_ema = self.accuracy_ema * (1.0 - FACTCHECK_ACCURACY_EMA_ALPHA)
                + accuracy_signal * FACTCHECK_ACCURACY_EMA_ALPHA;

            // Record verdict
            match result.verdict {
                FactCheckVerdict::True => {
                    self.total_verified += 1;
                    modulation.claims_verified += 1;
                }
                FactCheckVerdict::False => {
                    if result.confidence >= SUPPRESS_CONFIDENCE_THRESHOLD {
                        any_false_high_conf = true;
                        self.total_suppressed += 1;
                        modulation.claims_suppressed += 1;
                    }
                    max_cadence_penalty = max_cadence_penalty.max(CADENCE_PENALTY_FALSE);
                }
                FactCheckVerdict::Mixed | FactCheckVerdict::Unknown => {
                    max_cadence_penalty = max_cadence_penalty.max(CADENCE_PENALTY_UNVERIFIED);
                }
            }

            // Remove from pending queue if present
            if let Some(pos) = self.pending_claims.iter().position(|c| *c == result.claim) {
                self.pending_claims.remove(pos);
            }

            // Record in history
            if self.verdict_history.len() >= VERDICT_HISTORY_CAP {
                self.verdict_history.pop_front();
            }
            self.verdict_history.push_back(VerdictRecord {
                cycle,
                claim_snippet: if result.claim.len() > 80 {
                    format!("{}...", &result.claim[..77])
                } else {
                    result.claim.clone()
                },
                verdict: result.verdict.clone(),
                confidence: result.confidence,
            });
        }

        // Compute modulation
        if any_false_high_conf {
            modulation.suppress = true;
            modulation.confidence_scale = 0.0;
        } else {
            // Scale confidence based on accuracy trend
            modulation.confidence_scale = self.accuracy_ema.clamp(0.3, 1.0);
        }
        modulation.cadence_penalty = max_cadence_penalty;

        modulation
    }

    /// After Broca generates text, extract claims and submit for verification.
    ///
    /// Convenience method combining extract + submit. Returns the modulation
    /// to apply if there were verdicts from the previous cycle.
    pub fn on_broca_generation(&mut self, text: &str, cycle: u64) -> BrocaModulation {
        // Extract and submit new claims
        let claims = Self::extract_claims(text);
        if !claims.is_empty() {
            self.submit_for_verification(&claims);
        }

        // Process any pending verdicts from the previous cycle
        self.process_verdicts(cycle)
    }

    /// Get the number of pending claims.
    pub fn pending_count(&self) -> usize {
        self.pending_claims.len()
    }

    /// Get recent verdict history.
    pub fn verdict_history(&self) -> &VecDeque<VerdictRecord> {
        &self.verdict_history
    }
}

// ── Conductor Bridge Task ──────────────────────────────────────────────────

/// Async task that bridges factcheck channels to a Holochain conductor.
///
/// Reads claims from `claim_rx`, calls the factcheck zome via the conductor,
/// and sends results back through `verdict_tx`.
///
/// When no conductor is available, returns `FactCheckVerdict::Unknown` with
/// confidence 0.0 so the bridge degrades gracefully.
#[cfg(feature = "mycelix")]
pub struct FactcheckConductorTask;

#[cfg(feature = "mycelix")]
impl FactcheckConductorTask {
    /// Spawn the conductor bridge as a background thread.
    ///
    /// The task runs until the claim channel is closed (bridge dropped).
    /// Each claim is sent to the factcheck zome and the verdict is forwarded
    /// back to the Broca bridge.
    ///
    /// `conductor_url`: WebSocket URL for the Holochain conductor (e.g. "ws://localhost:8888")
    /// `app_id`: The installed app ID (e.g. "mycelix-unified")
    pub fn spawn(
        channels: FactcheckChannels,
        conductor_url: String,
        app_id: String,
    ) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("factcheck-conductor".into())
            .spawn(move || {
                Self::run_blocking(channels, &conductor_url, &app_id);
            })
            .expect("failed to spawn factcheck conductor thread")
    }

    /// Blocking event loop: drain claims, query conductor via WebSocket, send verdicts.
    fn run_blocking(channels: FactcheckChannels, conductor_url: &str, _app_id: &str) {
        use futures_util::{SinkExt, StreamExt};
        use serde_json::json;
        use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                tracing::error!(target: "cognitive_loop::factcheck", "Failed to build tokio runtime: {}", e);
                Self::run_mock_fallback(channels);
                return;
            }
        };

        rt.block_on(async {
            // Connect to Holochain conductor WebSocket
            let mut ws_stream = match connect_async(conductor_url).await {
                Ok((ws, _)) => ws,
                Err(e) => {
                    tracing::error!(target: "cognitive_loop::factcheck", "Failed to connect to factcheck conductor WebSocket at {}: {}", conductor_url, e);
                    // Fall back to offline mock mode
                    Self::run_mock_fallback(channels);
                    return;
                }
            };

            tracing::info!(target: "cognitive_loop::factcheck", "Connected to factcheck conductor at {}", conductor_url);

            let mut request_id = 0u64;

            while let Ok(claim) = channels.claim_rx.recv() {
                request_id += 1;

                // Holochain AppWebsocket call_zome request structure
                let request_payload = json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "call_zome",
                    "params": {
                        "cell_id": null,
                        "role_name": "knowledge",
                        "zome_name": "factcheck",
                        "fn_name": "fact_check",
                        "payload": {
                            "claim": claim
                        },
                        "provenance": "did:mycelix:factcheck",
                    }
                });

                let request_str = serde_json::to_string(&request_payload).unwrap();

                if let Err(e) = ws_stream.send(Message::Text(request_str.into())).await {
                    tracing::error!(target: "cognitive_loop::factcheck", "Failed to send claim to conductor: {}", e);
                    break;
                }

                // Wait for the response
                let mut verdict = FactCheckVerdict::Unknown;
                let mut confidence = 0.0;
                let mut source_ids = Vec::new();

                if let Some(Ok(response_msg)) = ws_stream.next().await {
                    if let Ok(text) = response_msg.into_text() {
                        if let Ok(res_json) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(result) = res_json.get("result") {
                                // Parse verdict
                                if let Some(v_str) = result.get("verdict").and_then(|v| v.as_str()) {
                                    verdict = match v_str {
                                        "True" => FactCheckVerdict::True,
                                        "False" => FactCheckVerdict::False,
                                        "Mixed" => FactCheckVerdict::Mixed,
                                        _ => FactCheckVerdict::Unknown,
                                    };
                                }
                                // Parse confidence
                                if let Some(c_num) = result.get("confidence").and_then(|c| c.as_f64()) {
                                    confidence = c_num as f32;
                                }
                                // Parse source_ids
                                if let Some(ids_array) = result.get("source_ids").and_then(|ids| ids.as_array()) {
                                    source_ids = ids_array.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect();
                                }
                            }
                        }
                    }
                }

                let result = FactCheckResult {
                    claim,
                    verdict,
                    confidence,
                    source_ids,
                };

                if channels.verdict_tx.send(result).is_err() {
                    break;
                }
            }
        });
    }

    /// Offline mock fallback path if conductor connection fails.
    fn run_mock_fallback(channels: FactcheckChannels) {
        while let Ok(claim) = channels.claim_rx.recv() {
            let result = FactCheckResult {
                claim,
                verdict: FactCheckVerdict::Unknown,
                confidence: 0.0,
                source_ids: Vec::new(),
            };

            if channels.verdict_tx.send(result).is_err() {
                break;
            }
        }
    }

    /// Spawn with a custom verdict function (for testing / mock conductors).
    pub fn spawn_with_handler<F>(
        channels: FactcheckChannels,
        handler: F,
    ) -> std::thread::JoinHandle<()>
    where
        F: Fn(&str) -> FactCheckResult + Send + 'static,
    {
        std::thread::Builder::new()
            .name("factcheck-conductor-mock".into())
            .spawn(move || {
                while let Ok(claim) = channels.claim_rx.recv() {
                    let result = handler(&claim);
                    if channels.verdict_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn mock factcheck conductor thread")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_claims_declarative() {
        let text = "The Earth orbits the Sun. Water boils at 100 degrees Celsius at sea level. Is that right?";
        let claims = BrocaFactcheckBridge::extract_claims(text);
        assert_eq!(claims.len(), 2);
        assert!(claims[0].contains("Earth orbits"));
        assert!(claims[1].contains("Water boils"));
    }

    #[test]
    fn test_extract_claims_skips_short() {
        let text = "Hi. The capital of France is Paris, which is a major European city.";
        let claims = BrocaFactcheckBridge::extract_claims(text);
        // "Hi." is too short (< 20 chars), only the second sentence passes
        assert_eq!(claims.len(), 1);
        assert!(claims[0].contains("capital of France"));
    }

    #[test]
    fn test_extract_claims_skips_subjective() {
        let text = "I think cats are great. I believe the moon is made of cheese. The moon orbits the Earth in approximately 27.3 days.";
        let claims = BrocaFactcheckBridge::extract_claims(text);
        assert_eq!(claims.len(), 1);
        assert!(claims[0].contains("moon orbits"));
    }

    #[test]
    fn test_extract_claims_skips_imperatives() {
        let text = "Please note the following. Try to understand this. The Sun is a G-type main-sequence star.";
        let claims = BrocaFactcheckBridge::extract_claims(text);
        assert_eq!(claims.len(), 1);
        assert!(claims[0].contains("Sun is a G-type"));
    }

    #[test]
    fn test_extract_claims_max_limit() {
        // Generate more than MAX_CLAIMS_PER_GENERATION sentences
        let sentences: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "Claim number {} is a declarative statement about something.",
                    i
                )
            })
            .collect();
        let text = sentences.join(" ");
        let claims = BrocaFactcheckBridge::extract_claims(&text);
        assert_eq!(claims.len(), MAX_CLAIMS_PER_GENERATION);
    }

    #[test]
    fn test_submit_and_pending() {
        let mut bridge = BrocaFactcheckBridge::new();
        let claims = vec![
            "Claim A is true.".to_string(),
            "Claim B is true.".to_string(),
        ];
        let queued = bridge.submit_for_verification(&claims);
        assert_eq!(queued, 2);
        assert_eq!(bridge.pending_count(), 2);
        assert_eq!(bridge.total_submitted, 2);
    }

    #[test]
    fn test_submit_evicts_oldest_on_overflow() {
        let mut bridge = BrocaFactcheckBridge::new();
        for i in 0..MAX_PENDING_CLAIMS + 5 {
            bridge.submit_for_verification(&[format!("Claim {} is a test claim for overflow.", i)]);
        }
        assert_eq!(bridge.pending_count(), MAX_PENDING_CLAIMS);
    }

    #[test]
    fn test_process_verdicts_true() {
        let mut bridge = BrocaFactcheckBridge::new();
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "The Earth is round.".to_string(),
            verdict: FactCheckVerdict::True,
            confidence: 0.95,
            source_ids: vec!["wikidata:abc123".to_string()],
        }]);

        let modulation = bridge.process_verdicts(1);
        assert!(!modulation.suppress);
        assert_eq!(modulation.claims_verified, 1);
        assert_eq!(modulation.claims_suppressed, 0);
        assert!(modulation.confidence_scale > 0.0);
        assert_eq!(bridge.total_verified, 1);
    }

    #[test]
    fn test_process_verdicts_false_high_confidence_suppresses() {
        let mut bridge = BrocaFactcheckBridge::new();
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "The moon is made of cheese.".to_string(),
            verdict: FactCheckVerdict::False,
            confidence: 0.95,
            source_ids: vec![],
        }]);

        let modulation = bridge.process_verdicts(1);
        assert!(modulation.suppress);
        assert_eq!(modulation.confidence_scale, 0.0);
        assert_eq!(modulation.claims_suppressed, 1);
        assert_eq!(bridge.total_suppressed, 1);
    }

    #[test]
    fn test_process_verdicts_false_low_confidence_no_suppress() {
        let mut bridge = BrocaFactcheckBridge::new();
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "Something uncertain happened.".to_string(),
            verdict: FactCheckVerdict::False,
            confidence: 0.50, // Below SUPPRESS_CONFIDENCE_THRESHOLD
            source_ids: vec![],
        }]);

        let modulation = bridge.process_verdicts(1);
        assert!(!modulation.suppress);
        assert_eq!(modulation.claims_suppressed, 0);
        assert!(modulation.cadence_penalty > 0.0);
    }

    #[test]
    fn test_process_verdicts_mixed_applies_cadence_penalty() {
        let mut bridge = BrocaFactcheckBridge::new();
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "This is a mixed claim about something.".to_string(),
            verdict: FactCheckVerdict::Mixed,
            confidence: 0.60,
            source_ids: vec![],
        }]);

        let modulation = bridge.process_verdicts(1);
        assert!(!modulation.suppress);
        assert_eq!(modulation.cadence_penalty, CADENCE_PENALTY_UNVERIFIED);
    }

    #[test]
    fn test_accuracy_ema_tracks_verdicts() {
        let mut bridge = BrocaFactcheckBridge::new();
        assert!((bridge.accuracy_ema() - 0.5).abs() < 0.001);

        // Several True verdicts should raise accuracy
        for i in 0..20 {
            bridge.receive_verdicts(vec![FactCheckResult {
                claim: format!("True claim number {} is verified.", i),
                verdict: FactCheckVerdict::True,
                confidence: 0.9,
                source_ids: vec![],
            }]);
            bridge.process_verdicts(i as u64);
        }
        assert!(bridge.accuracy_ema() > 0.7);

        // Several False verdicts should lower it
        for i in 0..20 {
            bridge.receive_verdicts(vec![FactCheckResult {
                claim: format!("False claim number {} is wrong.", i),
                verdict: FactCheckVerdict::False,
                confidence: 0.9,
                source_ids: vec![],
            }]);
            bridge.process_verdicts(20 + i as u64);
        }
        assert!(bridge.accuracy_ema() < 0.5);
    }

    #[test]
    fn test_verdict_history_bounded() {
        let mut bridge = BrocaFactcheckBridge::new();
        for i in 0..(VERDICT_HISTORY_CAP + 10) {
            bridge.receive_verdicts(vec![FactCheckResult {
                claim: format!("Claim {} is a fact about history bounds.", i),
                verdict: FactCheckVerdict::True,
                confidence: 0.9,
                source_ids: vec![],
            }]);
            bridge.process_verdicts(i as u64);
        }
        assert_eq!(bridge.verdict_history().len(), VERDICT_HISTORY_CAP);
    }

    #[test]
    fn test_on_broca_generation_full_flow() {
        let mut bridge = BrocaFactcheckBridge::new();

        // First generation: no verdicts yet, just extracts claims
        let text =
            "The speed of light is approximately 299,792 km/s. Gravity pulls objects toward Earth.";
        let modulation = bridge.on_broca_generation(text, 1);
        assert!(!modulation.suppress);
        assert_eq!(modulation.confidence_scale, 1.0); // No verdicts yet
        assert_eq!(bridge.pending_count(), 2);

        // Inject a verdict for the first claim
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "The speed of light is approximately 299,792 km/s.".to_string(),
            verdict: FactCheckVerdict::True,
            confidence: 0.95,
            source_ids: vec!["wikidata:speed_of_light".to_string()],
        }]);

        // Second generation: processes the verdict
        let text2 = "Water is composed of hydrogen and oxygen atoms.";
        let modulation2 = bridge.on_broca_generation(text2, 2);
        assert_eq!(modulation2.claims_verified, 1);
        assert!(!modulation2.suppress);
        // Pending should now have the claim from generation 2, minus the verified one
        assert_eq!(bridge.pending_count(), 2); // "Gravity pulls..." + "Water is composed..."
    }

    #[test]
    fn test_empty_text_no_claims() {
        let claims = BrocaFactcheckBridge::extract_claims("");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_no_verdicts_no_modulation() {
        let mut bridge = BrocaFactcheckBridge::new();
        let modulation = bridge.process_verdicts(1);
        assert!(!modulation.suppress);
        assert_eq!(modulation.confidence_scale, 1.0);
        assert_eq!(modulation.cadence_penalty, 0.0);
    }

    #[test]
    fn test_telemetry_snapshot() {
        let mut bridge = BrocaFactcheckBridge::new();
        bridge.submit_for_verification(&["Test claim for telemetry snapshot.".to_string()]);
        bridge.receive_verdicts(vec![FactCheckResult {
            claim: "Test claim for telemetry snapshot.".to_string(),
            verdict: FactCheckVerdict::True,
            confidence: 0.9,
            source_ids: vec![],
        }]);
        bridge.process_verdicts(1);

        let telem = bridge.telemetry();
        assert_eq!(telem.total_claims_submitted, 1);
        assert_eq!(telem.total_claims_verified, 1);
        assert_eq!(telem.total_claims_suppressed, 0);
    }

    #[test]
    fn test_channel_based_verdict_flow() {
        let mut bridge = BrocaFactcheckBridge::new();
        let channels = bridge.create_channels();
        assert!(bridge.is_connected());

        // Simulate a conductor: send a verdict through the channel
        channels
            .verdict_tx
            .send(FactCheckResult {
                claim: "Water boils at 100 degrees Celsius at sea level.".to_string(),
                verdict: FactCheckVerdict::True,
                confidence: 0.92,
                source_ids: vec!["chemistry:bp_water".to_string()],
            })
            .unwrap();

        // Bridge should drain the channel and process the verdict
        let modulation = bridge.process_verdicts(1);
        assert_eq!(modulation.claims_verified, 1);
        assert!(!modulation.suppress);
        assert_eq!(bridge.total_verified, 1);
    }

    #[test]
    fn test_channel_based_claim_submission() {
        let mut bridge = BrocaFactcheckBridge::new();
        let channels = bridge.create_channels();

        // Submit a claim — it should arrive on the channel
        bridge.submit_for_verification(&["The Nile is the longest river in Africa.".to_string()]);

        // The conductor side should receive the claim
        let received = channels.claim_rx.try_recv().unwrap();
        assert!(received.contains("Nile"));
    }
}
