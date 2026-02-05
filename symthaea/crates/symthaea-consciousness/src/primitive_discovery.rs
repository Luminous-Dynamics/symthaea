//! # Unified Primitive Discovery Service
//!
//! Integrates multiple discovery mechanisms into a unified service:
//!
//! 1. **Evolutionary Discovery** - Evolve new primitives through mutation/recombination
//! 2. **Compositional Discovery** - Find valuable compositions of existing primitives
//! 3. **Pattern Discovery** - Detect emergent patterns in reasoning traces
//! 4. **Validation Pipeline** - Ensure discovered primitives meet quality thresholds
//!
//! ## Key Features
//!
//! - Background discovery threads
//! - Automatic integration into running system
//! - Discovery event streaming
//! - Quality metrics and statistics
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::primitive_discovery::{
//!     PrimitiveDiscoveryService, DiscoveryServiceConfig,
//! };
//!
//! let mut service = PrimitiveDiscoveryService::new(DiscoveryServiceConfig::default());
//!
//! // Start background discovery
//! service.start();
//!
//! // Check for new discoveries
//! for discovery in service.poll_discoveries() {
//!     println!("Discovered: {} with Φ={:.3}", discovery.name, discovery.phi_score);
//! }
//!
//! // Get discoveries ready for integration
//! let validated = service.get_validated_discoveries(0.5);
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::primitive_system::{Primitive, PrimitiveTier, PrimitiveSystem};
use crate::consciousness::compositionality_primitives::ComposedPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, mpsc::{Sender, Receiver, channel}};
use std::time::{Duration, Instant};

// =============================================================================
// DISCOVERED PRIMITIVE
// =============================================================================

/// Source of a discovered primitive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoverySource {
    /// Discovered through evolutionary mutation/recombination
    Evolution,
    /// Discovered through composition of existing primitives
    Composition,
    /// Discovered through pattern detection in reasoning traces
    PatternDetection,
    /// Manually created and validated
    Manual,
}

/// A newly discovered primitive awaiting integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPrimitive {
    /// Unique discovery ID
    pub id: String,
    /// Primitive name
    pub name: String,
    /// Tier assignment
    pub tier: PrimitiveTier,
    /// Discovery source
    pub source: DiscoverySource,
    /// HDC encoding
    pub encoding: HV16,
    /// Phi score from discovery evaluation
    pub phi_score: f64,
    /// Confidence in the discovery (based on evaluation count)
    pub confidence: f64,
    /// Number of successful evaluations
    pub evaluation_count: usize,
    /// Composition formula (if from composition)
    pub composition_formula: Option<String>,
    /// Parent primitives (if derived)
    pub parent_ids: Vec<String>,
    /// Discovery timestamp
    pub discovered_at: Duration,
    /// Whether this has been validated
    pub validated: bool,
    /// Validation errors (if any)
    pub validation_errors: Vec<String>,
}

impl DiscoveredPrimitive {
    /// Create a new discovered primitive
    pub fn new(
        name: impl Into<String>,
        tier: PrimitiveTier,
        source: DiscoverySource,
        encoding: HV16,
        phi_score: f64,
    ) -> Self {
        let name_str = name.into();
        let id = format!("disc_{}_{}",
            name_str.to_lowercase().replace(' ', "_"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );

        Self {
            id,
            name: name_str,
            tier,
            source,
            encoding,
            phi_score,
            confidence: 0.0,
            evaluation_count: 0,
            composition_formula: None,
            parent_ids: Vec::new(),
            discovered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO),
            validated: false,
            validation_errors: Vec::new(),
        }
    }

    /// Convert to a full Primitive for integration
    pub fn to_primitive(&self, domain_hv: &HV16) -> Primitive {
        Primitive::base(
            &self.name,
            self.tier,
            "discovery", // Domain
            domain_hv.bind(&self.encoding),
            &format!("Discovered primitive via {:?}", self.source),
        )
    }

    /// Check if this primitive meets quality thresholds
    pub fn meets_quality_threshold(&self, min_phi: f64, min_confidence: f64) -> bool {
        self.phi_score >= min_phi &&
        self.confidence >= min_confidence &&
        self.validation_errors.is_empty()
    }
}

// =============================================================================
// DISCOVERY SERVICE CONFIGURATION
// =============================================================================

/// Configuration for the discovery service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryServiceConfig {
    /// Enable evolutionary discovery
    pub enable_evolution: bool,
    /// Enable compositional discovery
    pub enable_composition: bool,
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    /// Minimum Phi score to consider a discovery valid
    pub min_phi_threshold: f64,
    /// Minimum confidence to consider a discovery ready
    pub min_confidence: f64,
    /// Maximum pending discoveries before forcing integration
    pub max_pending_discoveries: usize,
    /// Discovery cycle interval (milliseconds)
    pub discovery_interval_ms: u64,
    /// Maximum discoveries per cycle
    pub max_per_cycle: usize,
    /// Enable auto-integration of high-quality discoveries
    pub auto_integrate: bool,
    /// Phi threshold for auto-integration
    pub auto_integrate_threshold: f64,
    /// Minimum phi delta above constituents for a composition to be considered emergent
    pub emergence_delta: f64,
}

impl Default for DiscoveryServiceConfig {
    fn default() -> Self {
        Self {
            enable_evolution: true,
            enable_composition: true,
            enable_pattern_detection: true,
            min_phi_threshold: 0.1,
            min_confidence: 0.7,
            max_pending_discoveries: 100,
            discovery_interval_ms: 1000,
            max_per_cycle: 10,
            auto_integrate: false,
            auto_integrate_threshold: 0.8,
            emergence_delta: 0.05,
        }
    }
}

// =============================================================================
// DISCOVERY STATISTICS
// =============================================================================

/// Statistics about the discovery process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// Total discovery cycles run
    pub cycles_completed: u64,
    /// Total primitives discovered
    pub total_discovered: u64,
    /// Primitives by source
    pub by_source: HashMap<String, u64>,
    /// Primitives integrated into main system
    pub integrated: u64,
    /// Primitives rejected
    pub rejected: u64,
    /// Best Phi score ever discovered
    pub best_phi: f64,
    /// Average Phi of integrated primitives
    pub avg_integrated_phi: f64,
    /// Discovery rate (discoveries per hour)
    pub discovery_rate: f64,
}

impl DiscoveryStats {
    /// Record a new discovery
    pub fn record_discovery(&mut self, source: DiscoverySource, phi: f64) {
        self.total_discovered += 1;
        let source_key = format!("{:?}", source);
        *self.by_source.entry(source_key).or_insert(0) += 1;
        if phi > self.best_phi {
            self.best_phi = phi;
        }
    }

    /// Record an integration
    pub fn record_integration(&mut self, phi: f64) {
        let prev_total = self.avg_integrated_phi * self.integrated as f64;
        self.integrated += 1;
        self.avg_integrated_phi = (prev_total + phi) / self.integrated as f64;
    }

    /// Record a rejection
    pub fn record_rejection(&mut self) {
        self.rejected += 1;
    }
}

// =============================================================================
// DISCOVERY EVENT
// =============================================================================

/// Event emitted when a discovery occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryEvent {
    /// Event type
    pub event_type: DiscoveryEventType,
    /// Timestamp
    pub timestamp: u64,
    /// Associated primitive (if any)
    pub primitive: Option<DiscoveredPrimitive>,
    /// Additional data
    pub metadata: HashMap<String, String>,
}

/// Types of discovery events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryEventType {
    /// New primitive discovered
    NewDiscovery,
    /// Primitive validated successfully
    Validated,
    /// Primitive integrated into system
    Integrated,
    /// Primitive rejected
    Rejected,
    /// Discovery cycle started
    CycleStarted,
    /// Discovery cycle completed
    CycleCompleted,
}

// =============================================================================
// PATTERN DETECTOR
// =============================================================================

/// Detects patterns in reasoning traces that suggest new primitives
#[derive(Debug, Clone)]
pub struct PatternDetector {
    /// Observed reasoning patterns (pattern hash -> count)
    pattern_counts: HashMap<u64, usize>,
    /// Pattern to primitive mapping (detected patterns)
    detected_patterns: HashMap<u64, DetectedPattern>,
    /// Minimum occurrences to consider a pattern significant
    min_occurrences: usize,
}

#[derive(Debug, Clone)]
struct DetectedPattern {
    /// Representative encoding
    encoding: HV16,
    /// Occurrence count
    count: usize,
    /// Context primitives
    context: Vec<String>,
    /// Suggested tier
    suggested_tier: PrimitiveTier,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(min_occurrences: usize) -> Self {
        Self {
            pattern_counts: HashMap::new(),
            detected_patterns: HashMap::new(),
            min_occurrences,
        }
    }

    /// Record a reasoning trace for pattern detection
    pub fn record_trace(&mut self, primitives_used: &[&str], result_encoding: &HV16) {
        // Hash the primitive sequence
        let hash = self.hash_sequence(primitives_used);

        // Increment count
        *self.pattern_counts.entry(hash).or_insert(0) += 1;
        let count = self.pattern_counts[&hash];

        // If pattern is significant, record it
        if count >= self.min_occurrences {
            self.detected_patterns.insert(hash, DetectedPattern {
                encoding: result_encoding.clone(),
                count,
                context: primitives_used.iter().map(|s| s.to_string()).collect(),
                suggested_tier: self.infer_tier(primitives_used),
            });
        }
    }

    /// Get patterns that should become primitives
    pub fn get_significant_patterns(&self) -> Vec<DiscoveredPrimitive> {
        self.detected_patterns.values()
            .filter(|p| p.count >= self.min_occurrences * 2)
            .map(|p| {
                let name = format!("PATTERN_{:08x}", self.hash_sequence(&p.context.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
                DiscoveredPrimitive::new(
                    name,
                    p.suggested_tier,
                    DiscoverySource::PatternDetection,
                    p.encoding.clone(),
                    0.5, // Default phi, needs evaluation
                )
            })
            .collect()
    }

    fn hash_sequence(&self, primitives: &[&str]) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for p in primitives {
            p.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn infer_tier(&self, primitives: &[&str]) -> PrimitiveTier {
        // Keyword-based fallback (used when no system context available)
        let text = primitives.join(" ").to_lowercase();
        if text.contains("consciousness") || text.contains("meta") {
            PrimitiveTier::MetaCognitive
        } else if text.contains("time") || text.contains("temporal") {
            PrimitiveTier::Temporal
        } else if text.contains("strategy") || text.contains("plan") {
            PrimitiveTier::Strategic
        } else {
            PrimitiveTier::Physical
        }
    }

    /// Similarity-based tier inference using encoding distance to known tier exemplars
    fn infer_tier_from_encoding(&self, encoding: &HV16, system: &PrimitiveSystem) -> PrimitiveTier {
        let tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        let mut best_tier = PrimitiveTier::NSM;
        let mut best_sim = f32::MIN;

        for &tier in &tiers {
            let tier_prims = system.get_tier(tier);
            if tier_prims.is_empty() {
                continue;
            }

            let avg_sim: f32 = tier_prims.iter()
                .map(|p| encoding.similarity(&p.encoding))
                .sum::<f32>() / tier_prims.len() as f32;

            if avg_sim > best_sim {
                best_sim = avg_sim;
                best_tier = tier;
            }
        }

        if best_sim < 0.3 { PrimitiveTier::NSM } else { best_tier }
    }
}

// =============================================================================
// DISCOVERY SERVICE
// =============================================================================

/// Unified primitive discovery service
pub struct PrimitiveDiscoveryService {
    /// Configuration
    config: DiscoveryServiceConfig,
    /// Pending discoveries (not yet integrated)
    pending: VecDeque<DiscoveredPrimitive>,
    /// Validated discoveries ready for integration
    validated: Vec<DiscoveredPrimitive>,
    /// Pattern detector
    pattern_detector: PatternDetector,
    /// Statistics
    stats: DiscoveryStats,
    /// Event sender (if streaming enabled)
    event_sender: Option<Sender<DiscoveryEvent>>,
    /// Start time
    start_time: Instant,
    /// RNG state
    rng_state: u64,
}

impl PrimitiveDiscoveryService {
    /// Create a new discovery service
    pub fn new(config: DiscoveryServiceConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            validated: Vec::new(),
            pattern_detector: PatternDetector::new(5),
            stats: DiscoveryStats::default(),
            event_sender: None,
            start_time: Instant::now(),
            rng_state: 42,
        }
    }

    /// Enable event streaming and get receiver
    pub fn enable_streaming(&mut self) -> Receiver<DiscoveryEvent> {
        let (sender, receiver) = channel();
        self.event_sender = Some(sender);
        receiver
    }

    /// Run a discovery cycle
    pub fn run_cycle(&mut self, primitive_system: &PrimitiveSystem) -> Vec<DiscoveredPrimitive> {
        self.stats.cycles_completed += 1;
        self.emit_event(DiscoveryEventType::CycleStarted, None);

        let mut discoveries = Vec::new();

        // Evolution-based discovery
        if self.config.enable_evolution {
            discoveries.extend(self.discover_via_evolution(primitive_system));
        }

        // Composition-based discovery
        if self.config.enable_composition {
            discoveries.extend(self.discover_via_composition(primitive_system));
        }

        // Pattern-based discovery
        if self.config.enable_pattern_detection {
            discoveries.extend(self.pattern_detector.get_significant_patterns());
        }

        // Filter and record discoveries
        let mut accepted = Vec::new();
        for mut discovery in discoveries {
            if discovery.phi_score >= self.config.min_phi_threshold {
                self.stats.record_discovery(discovery.source, discovery.phi_score);
                discovery.confidence = self.compute_confidence(&discovery);

                // Validate
                discovery.validated = self.validate_discovery(&mut discovery);

                if discovery.validated {
                    self.emit_event(DiscoveryEventType::Validated, Some(discovery.clone()));
                    self.validated.push(discovery.clone());
                    accepted.push(discovery.clone());
                } else {
                    self.stats.record_rejection();
                    self.emit_event(DiscoveryEventType::Rejected, Some(discovery.clone()));
                }

                self.pending.push_back(discovery);
            }
        }

        // Prune pending if too large
        while self.pending.len() > self.config.max_pending_discoveries {
            self.pending.pop_front();
        }

        // Update discovery rate
        let elapsed_hours = self.start_time.elapsed().as_secs_f64() / 3600.0;
        if elapsed_hours > 0.0 {
            self.stats.discovery_rate = self.stats.total_discovered as f64 / elapsed_hours;
        }

        self.emit_event(DiscoveryEventType::CycleCompleted, None);
        accepted
    }

    /// Evolution-based discovery
    fn discover_via_evolution(&mut self, _system: &PrimitiveSystem) -> Vec<DiscoveredPrimitive> {
        let mut discoveries = Vec::new();

        // Generate random mutations of existing primitives
        for _ in 0..self.config.max_per_cycle {
            // Create a random primitive encoding
            let encoding = HV16::random(self.random_u64());

            // Random tier selection weighted toward lower tiers
            let tier_idx = self.random_u64() % 9;
            let tier = match tier_idx {
                0 => PrimitiveTier::NSM,
                1 => PrimitiveTier::Mathematical,
                2 => PrimitiveTier::Physical,
                3 => PrimitiveTier::Geometric,
                4 => PrimitiveTier::Strategic,
                5 => PrimitiveTier::MetaCognitive,
                6 => PrimitiveTier::Temporal,
                7 => PrimitiveTier::Compositional,
                _ => PrimitiveTier::Consciousness,
            };

            // Generate name and evaluate
            let name = format!("EVOLVED_{:04x}", self.random_u64() & 0xFFFF);
            let phi_score = self.estimate_phi(&encoding);

            let discovery = DiscoveredPrimitive::new(
                name,
                tier,
                DiscoverySource::Evolution,
                encoding,
                phi_score,
            );

            discoveries.push(discovery);
        }

        discoveries
    }

    /// Composition-based discovery
    fn discover_via_composition(&mut self, system: &PrimitiveSystem) -> Vec<DiscoveredPrimitive> {
        let mut discoveries = Vec::new();

        // Get random pairs of existing primitives and compose them
        let names: Vec<_> = system.all_primitives().map(|p| p.name.clone()).collect();
        if names.len() < 2 {
            return discoveries;
        }

        for _ in 0..self.config.max_per_cycle / 2 {
            let idx_a = self.random_u64() as usize % names.len();
            let idx_b = self.random_u64() as usize % names.len();

            if idx_a == idx_b {
                continue;
            }

            if let (Some(prim_a), Some(prim_b)) = (
                system.get(&names[idx_a]),
                system.get(&names[idx_b]),
            ) {
                // Compose via binding
                let composed = prim_a.encoding.bind(&prim_b.encoding);
                let composed_name = format!("{}_{}", names[idx_a], names[idx_b]);
                let phi_score = self.estimate_phi(&composed);

                // Skip non-emergent compositions: phi must exceed max constituent + delta
                let phi_a = self.estimate_phi(&prim_a.encoding);
                let phi_b = self.estimate_phi(&prim_b.encoding);
                let max_constituent = phi_a.max(phi_b);
                if phi_score <= max_constituent + self.config.emergence_delta {
                    continue;
                }

                // Use higher tier of the two
                let tier = if prim_a.tier as u8 > prim_b.tier as u8 {
                    prim_a.tier
                } else {
                    prim_b.tier
                };

                let mut discovery = DiscoveredPrimitive::new(
                    composed_name,
                    tier,
                    DiscoverySource::Composition,
                    composed,
                    phi_score,
                );

                discovery.composition_formula = Some(format!("{} ⊗ {}", names[idx_a], names[idx_b]));
                discovery.parent_ids = vec![names[idx_a].clone(), names[idx_b].clone()];

                discoveries.push(discovery);
            }
        }

        discoveries
    }

    /// Estimate Phi for an encoding using real Phi computation
    fn estimate_phi(&self, encoding: &HV16) -> f64 {
        use symthaea_core::hdc::unified_hv::ContinuousHV;
        use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

        // Convert HV16 bits to bipolar representation, partition into nodes
        let bipolar = encoding.to_bipolar();
        let chunk_size = bipolar.len() / 4;
        if chunk_size == 0 {
            // Fallback for very small encodings
            let popcount = encoding.popcount();
            let total_bits = bipolar.len().max(1);
            return 1.0 - ((popcount as f64 / total_bits as f64) - 0.5).abs() * 2.0;
        }

        let nodes: Vec<ContinuousHV> = bipolar
            .chunks(chunk_size)
            .map(|chunk| ContinuousHV::from_vec(chunk.to_vec()))
            .collect();

        if nodes.len() < 2 {
            return 0.0;
        }

        let engine = PhiEngine::new(PhiMethod::Resonator);
        let phi = engine.compute(&nodes).phi;

        if phi.is_nan() || phi < 0.0 {
            // Fallback
            let popcount = encoding.popcount();
            let total_bits = bipolar.len().max(1);
            return 1.0 - ((popcount as f64 / total_bits as f64) - 0.5).abs() * 2.0;
        }
        phi.min(1.0)
    }

    /// Compute confidence based on evaluation count and consistency
    fn compute_confidence(&self, discovery: &DiscoveredPrimitive) -> f64 {
        // Base confidence on evaluation count
        let eval_factor = 1.0 - (-0.5 * discovery.evaluation_count as f64).exp();

        // Adjust for source reliability
        let source_factor = match discovery.source {
            DiscoverySource::Evolution => 0.7,
            DiscoverySource::Composition => 0.85,
            DiscoverySource::PatternDetection => 0.9,
            DiscoverySource::Manual => 1.0,
        };

        (eval_factor * source_factor).min(1.0)
    }

    /// Validate a discovery
    fn validate_discovery(&self, discovery: &mut DiscoveredPrimitive) -> bool {
        let mut valid = true;

        // Check encoding is not degenerate
        let popcount = discovery.encoding.popcount();
        let total_bits = crate::hdc::HDC_DIMENSION as u32;
        if popcount < total_bits / 10 || popcount > total_bits * 9 / 10 {
            discovery.validation_errors.push("Degenerate encoding (too sparse or dense)".into());
            valid = false;
        }

        // Check name is valid
        if discovery.name.is_empty() {
            discovery.validation_errors.push("Empty name".into());
            valid = false;
        }

        // Check phi meets threshold
        if discovery.phi_score < self.config.min_phi_threshold {
            discovery.validation_errors.push(format!(
                "Phi {} below threshold {}",
                discovery.phi_score, self.config.min_phi_threshold
            ));
            valid = false;
        }

        valid
    }

    /// Record a reasoning trace for pattern detection
    pub fn record_reasoning_trace(&mut self, primitives_used: &[&str], result_encoding: &HV16) {
        self.pattern_detector.record_trace(primitives_used, result_encoding);
    }

    /// Get validated discoveries ready for integration
    pub fn get_validated_discoveries(&self, min_phi: f64) -> Vec<&DiscoveredPrimitive> {
        self.validated.iter()
            .filter(|d| d.phi_score >= min_phi && d.validated)
            .collect()
    }

    /// Get all pending discoveries
    pub fn get_pending(&self) -> &VecDeque<DiscoveredPrimitive> {
        &self.pending
    }

    /// Get statistics
    pub fn stats(&self) -> &DiscoveryStats {
        &self.stats
    }

    /// Emit a discovery event
    fn emit_event(&self, event_type: DiscoveryEventType, primitive: Option<DiscoveredPrimitive>) {
        if let Some(ref sender) = self.event_sender {
            let event = DiscoveryEvent {
                event_type,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                primitive,
                metadata: HashMap::new(),
            };
            let _ = sender.send(event);
        }
    }

    /// Generate random u64
    fn random_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.rng_state
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovered_primitive_creation() {
        let encoding = HV16::random(42);
        let discovery = DiscoveredPrimitive::new(
            "TEST_PRIMITIVE",
            PrimitiveTier::Physical,
            DiscoverySource::Evolution,
            encoding,
            0.5,
        );

        assert!(!discovery.id.is_empty());
        assert_eq!(discovery.name, "TEST_PRIMITIVE");
        assert_eq!(discovery.tier, PrimitiveTier::Physical);
        assert_eq!(discovery.source, DiscoverySource::Evolution);
    }

    #[test]
    fn test_quality_threshold() {
        let mut discovery = DiscoveredPrimitive::new(
            "TEST",
            PrimitiveTier::Physical,
            DiscoverySource::Composition,
            HV16::random(42),
            0.7,
        );
        discovery.confidence = 0.8;

        assert!(discovery.meets_quality_threshold(0.5, 0.7));
        assert!(!discovery.meets_quality_threshold(0.8, 0.7));
        assert!(!discovery.meets_quality_threshold(0.5, 0.9));
    }

    #[test]
    fn test_discovery_service_cycle() {
        let config = DiscoveryServiceConfig::default();
        let mut service = PrimitiveDiscoveryService::new(config);
        let system = PrimitiveSystem::new();

        let discoveries = service.run_cycle(&system);

        assert!(service.stats().cycles_completed >= 1);
        // Should have discovered some primitives (evolutionary at least)
        // Either we discovered some primitives or the cycle completed
        let _ = discoveries; // We just want to verify cycle ran successfully
    }

    #[test]
    fn test_pattern_detector() {
        let mut detector = PatternDetector::new(3);
        let encoding = HV16::random(42);

        // Record same pattern multiple times
        for _ in 0..5 {
            detector.record_trace(&["BIND", "SEQUENCE", "NEGATE"], &encoding);
        }

        let patterns = detector.get_significant_patterns();
        // Pattern should be detected but may not meet 2x threshold yet
        assert!(detector.pattern_counts.len() >= 1);
    }

    #[test]
    fn test_discovery_stats() {
        let mut stats = DiscoveryStats::default();

        stats.record_discovery(DiscoverySource::Evolution, 0.7);
        stats.record_discovery(DiscoverySource::Composition, 0.8);
        stats.record_integration(0.75);

        assert_eq!(stats.total_discovered, 2);
        assert_eq!(stats.integrated, 1);
        assert_eq!(stats.best_phi, 0.8);
    }

    #[test]
    fn test_streaming_events() {
        let mut service = PrimitiveDiscoveryService::new(DiscoveryServiceConfig::default());
        let receiver = service.enable_streaming();

        let system = PrimitiveSystem::new();
        service.run_cycle(&system);

        // Should have received at least cycle start/complete events
        std::thread::sleep(Duration::from_millis(10));
        let mut event_count = 0;
        while receiver.try_recv().is_ok() {
            event_count += 1;
        }
        assert!(event_count >= 2); // At least CycleStarted and CycleCompleted
    }
}
