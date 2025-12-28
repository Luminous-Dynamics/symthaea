// Revolutionary Enhancement #2: Causal Pattern Recognition
//
// Library of known causal patterns with template matching.
//
// Key Innovations:
// - Pre-defined motif library for common causal sequences
// - Template matching against real-time event streams
// - Pattern severity and recommended actions
// - User-configurable custom patterns
// - Pattern evolution tracking
//
// Use Cases:
// - Early detection of known failure modes
// - Proactive alerting before problems manifest
// - Learning from historical incidents
// - Sharing knowledge across Symthaea instances

use super::types::Event;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A known causal pattern (motif) in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalMotif {
    /// Unique identifier for this motif
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description of what this pattern indicates
    pub description: String,

    /// Sequence of event types that form this pattern
    /// Example: ["security_check", "phi_measurement", "router_selection"]
    pub sequence: Vec<String>,

    /// Whether order matters (strict sequence vs bag of events)
    pub strict_order: bool,

    /// Minimum confidence to match (0.0-1.0)
    pub min_confidence: f64,

    /// Severity if detected
    pub severity: MotifSeverity,

    /// Recommended actions when pattern is detected
    pub recommendations: Vec<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// How many times this pattern has been observed
    #[serde(default)]
    pub observation_count: usize,

    /// Whether this pattern is user-defined or built-in
    #[serde(default)]
    pub user_defined: bool,
}

/// Severity level of a detected motif
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotifSeverity {
    /// Informational - expected behavior
    Info,

    /// Low concern - monitor but don't alert
    Low,

    /// Medium concern - worth investigating
    Medium,

    /// High concern - likely indicates a problem
    High,

    /// Critical - immediate action required
    Critical,
}

/// Result of matching a pattern against event stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifMatch {
    /// The motif that was matched
    pub motif: CausalMotif,

    /// Confidence of the match (0.0-1.0)
    pub confidence: f64,

    /// Event IDs that matched this pattern
    pub matched_events: Vec<String>,

    /// Timestamp of when pattern was detected
    pub detected_at: chrono::DateTime<chrono::Utc>,

    /// Any deviations from the expected pattern
    pub deviations: Vec<String>,
}

/// Library of known causal patterns
pub struct MotifLibrary {
    /// All known motifs indexed by ID
    motifs: HashMap<String, CausalMotif>,

    /// Statistics about pattern detection
    stats: MotifStats,
}

/// Statistics for motif detection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotifStats {
    pub total_matches: usize,
    pub matches_by_severity: HashMap<String, usize>,
    pub most_common_pattern: Option<String>,
    pub avg_confidence: f64,
}

impl MotifLibrary {
    /// Create a new motif library with built-in patterns
    pub fn new() -> Self {
        let mut library = Self {
            motifs: HashMap::new(),
            stats: MotifStats::default(),
        };

        // Load built-in patterns
        library.load_builtin_patterns();

        library
    }

    /// Load built-in common patterns
    fn load_builtin_patterns(&mut self) {
        // Pattern 1: Normal consciousness pipeline
        self.add_motif(CausalMotif {
            id: "normal_consciousness_flow".to_string(),
            name: "Normal Consciousness Flow".to_string(),
            description: "Expected sequence of consciousness events".to_string(),
            sequence: vec![
                "security_check".to_string(),
                "phi_measurement".to_string(),
                "router_selection".to_string(),
            ],
            strict_order: true,
            min_confidence: 0.7,
            severity: MotifSeverity::Info,
            recommendations: vec![
                "This is expected behavior - no action needed".to_string(),
            ],
            tags: vec!["consciousness".to_string(), "normal".to_string()],
            observation_count: 0,
            user_defined: false,
        });

        // Pattern 2: Degraded consciousness (low phi)
        self.add_motif(CausalMotif {
            id: "degraded_consciousness".to_string(),
            name: "Degraded Consciousness Pattern".to_string(),
            description: "Multiple phi measurements without workspace ignition indicates degraded consciousness".to_string(),
            sequence: vec![
                "phi_measurement".to_string(),
                "phi_measurement".to_string(),
                "phi_measurement".to_string(),
            ],
            strict_order: false,
            min_confidence: 0.8,
            severity: MotifSeverity::High,
            recommendations: vec![
                "Check primitive activation levels".to_string(),
                "Review recent configuration changes".to_string(),
                "Consider increasing coalition size threshold".to_string(),
            ],
            tags: vec!["consciousness".to_string(), "degraded".to_string(), "warning".to_string()],
            observation_count: 0,
            user_defined: false,
        });

        // Pattern 3: Security rejection loop
        self.add_motif(CausalMotif {
            id: "security_rejection_loop".to_string(),
            name: "Security Rejection Loop".to_string(),
            description: "Repeated security rejections indicate misconfiguration or attack".to_string(),
            sequence: vec![
                "security_check".to_string(),
                "error".to_string(),
                "security_check".to_string(),
                "error".to_string(),
            ],
            strict_order: true,
            min_confidence: 0.9,
            severity: MotifSeverity::Critical,
            recommendations: vec![
                "Review security policy configuration".to_string(),
                "Check for potential attack patterns".to_string(),
                "Investigate input sources".to_string(),
            ],
            tags: vec!["security".to_string(), "error".to_string(), "critical".to_string()],
            observation_count: 0,
            user_defined: false,
        });

        // Pattern 4: High cognitive load
        self.add_motif(CausalMotif {
            id: "high_cognitive_load".to_string(),
            name: "High Cognitive Load".to_string(),
            description: "Many primitive activations without workspace ignition suggests cognitive overload".to_string(),
            sequence: vec![
                "primitive_activation".to_string(),
                "primitive_activation".to_string(),
                "primitive_activation".to_string(),
                "primitive_activation".to_string(),
            ],
            strict_order: false,
            min_confidence: 0.75,
            severity: MotifSeverity::Medium,
            recommendations: vec![
                "Consider reducing input complexity".to_string(),
                "Review primitive selection criteria".to_string(),
                "Increase workspace activation threshold".to_string(),
            ],
            tags: vec!["performance".to_string(), "cognitive_load".to_string()],
            observation_count: 0,
            user_defined: false,
        });

        // Pattern 5: Successful learning integration
        self.add_motif(CausalMotif {
            id: "successful_learning".to_string(),
            name: "Successful Learning Integration".to_string(),
            description: "Complete learning cycle from input to response with workspace integration".to_string(),
            sequence: vec![
                "language_step".to_string(),
                "workspace_ignition".to_string(),
                "phi_measurement".to_string(),
                "response_generated".to_string(),
            ],
            strict_order: true,
            min_confidence: 0.8,
            severity: MotifSeverity::Info,
            recommendations: vec![
                "Normal learning flow - continue monitoring".to_string(),
            ],
            tags: vec!["learning".to_string(), "success".to_string()],
            observation_count: 0,
            user_defined: false,
        });
    }

    /// Add a motif to the library
    pub fn add_motif(&mut self, motif: CausalMotif) {
        self.motifs.insert(motif.id.clone(), motif);
    }

    /// Remove a motif from the library
    pub fn remove_motif(&mut self, id: &str) -> Option<CausalMotif> {
        self.motifs.remove(id)
    }

    /// Get a motif by ID
    pub fn get_motif(&self, id: &str) -> Option<&CausalMotif> {
        self.motifs.get(id)
    }

    /// Get all motifs
    pub fn all_motifs(&self) -> Vec<&CausalMotif> {
        self.motifs.values().collect()
    }

    /// Get motifs by tag
    pub fn motifs_by_tag(&self, tag: &str) -> Vec<&CausalMotif> {
        self.motifs.values()
            .filter(|m| m.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Get motifs by severity
    pub fn motifs_by_severity(&self, severity: MotifSeverity) -> Vec<&CausalMotif> {
        self.motifs.values()
            .filter(|m| m.severity == severity)
            .collect()
    }

    /// Match a sequence of events against known patterns
    pub fn match_sequence(&mut self, events: &[(String, Event)]) -> Vec<MotifMatch> {
        let mut matches = Vec::new();
        let mut matched_motif_ids = Vec::new();

        // First pass: find all matches (immutable borrow)
        for motif in self.motifs.values() {
            if let Some(m) = self.try_match_motif(motif, events) {
                // Update statistics
                self.stats.total_matches += 1;
                *self.stats.matches_by_severity
                    .entry(format!("{:?}", motif.severity))
                    .or_insert(0) += 1;

                matched_motif_ids.push(motif.id.clone());
                matches.push(m);
            }
        }

        // Second pass: update observation counts (mutable borrow)
        for motif_id in matched_motif_ids {
            if let Some(motif) = self.motifs.get_mut(&motif_id) {
                motif.observation_count += 1;
            }
        }

        // Update most common pattern
        if let Some((id, _)) = self.motifs.iter()
            .max_by_key(|(_, m)| m.observation_count) {
            self.stats.most_common_pattern = Some(id.clone());
        }

        // Update average confidence
        if !matches.is_empty() {
            let total_confidence: f64 = matches.iter().map(|m| m.confidence).sum();
            self.stats.avg_confidence = total_confidence / matches.len() as f64;
        }

        matches
    }

    /// Try to match a single motif against event sequence
    fn try_match_motif(&self, motif: &CausalMotif, events: &[(String, Event)]) -> Option<MotifMatch> {
        if events.is_empty() || motif.sequence.is_empty() {
            return None;
        }

        let event_types: Vec<String> = events.iter()
            .map(|(_, e)| e.event_type.clone())
            .collect();

        if motif.strict_order {
            self.match_strict_sequence(motif, events, &event_types)
        } else {
            self.match_flexible_sequence(motif, events, &event_types)
        }
    }

    /// Match with strict ordering (subsequence match)
    fn match_strict_sequence(
        &self,
        motif: &CausalMotif,
        events: &[(String, Event)],
        event_types: &[String],
    ) -> Option<MotifMatch> {
        // Find if motif sequence appears as subsequence
        let pattern_len = motif.sequence.len();

        // Guard: If pattern is longer than events, no match possible
        if pattern_len > event_types.len() {
            return None;
        }

        for i in 0..=event_types.len() - pattern_len {
            let window = &event_types[i..i + pattern_len];

            // Check if window matches pattern
            let matches = window.iter()
                .zip(motif.sequence.iter())
                .filter(|(a, b)| a == b)
                .count();

            let confidence = matches as f64 / pattern_len as f64;

            if confidence >= motif.min_confidence {
                let matched_events: Vec<String> = events[i..i + pattern_len]
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect();

                return Some(MotifMatch {
                    motif: motif.clone(),
                    confidence,
                    matched_events,
                    detected_at: chrono::Utc::now(),
                    deviations: vec![],  // TODO: Track deviations
                });
            }
        }

        None
    }

    /// Match with flexible ordering (bag of events)
    fn match_flexible_sequence(
        &self,
        motif: &CausalMotif,
        events: &[(String, Event)],
        event_types: &[String],
    ) -> Option<MotifMatch> {
        // Count occurrences of each type in both sequences
        let mut pattern_counts: HashMap<&str, usize> = HashMap::new();
        for t in &motif.sequence {
            *pattern_counts.entry(t.as_str()).or_insert(0) += 1;
        }

        let mut event_counts: HashMap<&str, usize> = HashMap::new();
        for t in event_types {
            *event_counts.entry(t.as_str()).or_insert(0) += 1;
        }

        // Calculate how many pattern elements are satisfied
        let mut satisfied = 0;
        let mut total = 0;
        let mut matched_indices = Vec::new();

        for (pattern_type, required_count) in pattern_counts.iter() {
            let actual_count = event_counts.get(pattern_type).copied().unwrap_or(0);
            let satisfied_count = (*required_count).min(actual_count);
            satisfied += satisfied_count;
            total += required_count;

            // Find indices of matched events
            if satisfied_count > 0 {
                for (i, (_, event)) in events.iter().enumerate() {
                    if event.event_type == *pattern_type && matched_indices.len() < satisfied_count {
                        matched_indices.push(i);
                    }
                }
            }
        }

        let confidence = satisfied as f64 / total as f64;

        if confidence >= motif.min_confidence {
            let matched_events: Vec<String> = matched_indices.iter()
                .map(|&i| events[i].0.clone())
                .collect();

            Some(MotifMatch {
                motif: motif.clone(),
                confidence,
                matched_events,
                detected_at: chrono::Utc::now(),
                deviations: vec![],  // TODO: Track deviations
            })
        } else {
            None
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &MotifStats {
        &self.stats
    }

    /// Export library to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.motifs)
    }

    /// Import library from JSON
    pub fn import_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let motifs: HashMap<String, CausalMotif> = serde_json::from_str(json)?;
        for (id, motif) in motifs {
            self.motifs.insert(id, motif);
        }
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_events(types: &[&str]) -> Vec<(String, Event)> {
        types.iter().enumerate()
            .map(|(i, t)| {
                let id = format!("event_{}", i);
                let event = Event {
                    timestamp: chrono::Utc::now(),
                    event_type: t.to_string(),
                    data: json!({}),
                };
                (id, event)
            })
            .collect()
    }

    #[test]
    fn test_motif_library_creation() {
        let library = MotifLibrary::new();

        // Should have built-in patterns
        assert!(library.all_motifs().len() >= 5);

        // Check specific patterns exist
        assert!(library.get_motif("normal_consciousness_flow").is_some());
        assert!(library.get_motif("degraded_consciousness").is_some());
        assert!(library.get_motif("security_rejection_loop").is_some());
    }

    #[test]
    fn test_strict_sequence_match() {
        let mut library = MotifLibrary::new();

        // Create events matching normal consciousness flow
        let events = create_test_events(&[
            "security_check",
            "phi_measurement",
            "router_selection",
        ]);

        let matches = library.match_sequence(&events);

        // Should match normal_consciousness_flow pattern
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|m| m.motif.id == "normal_consciousness_flow"));
        assert!(matches[0].confidence >= 0.7);
    }

    #[test]
    fn test_flexible_sequence_match() {
        let mut library = MotifLibrary::new();

        // Create events matching degraded consciousness (3 phi measurements, order doesn't matter)
        let events = create_test_events(&[
            "phi_measurement",
            "router_selection",  // Extra event
            "phi_measurement",
            "phi_measurement",
        ]);

        let matches = library.match_sequence(&events);

        // Should match degraded_consciousness pattern
        assert!(matches.iter().any(|m| m.motif.id == "degraded_consciousness"));
    }

    #[test]
    fn test_custom_motif() {
        let mut library = MotifLibrary::new();

        // Add custom user-defined pattern
        let custom = CausalMotif {
            id: "custom_pattern".to_string(),
            name: "Custom Test Pattern".to_string(),
            description: "User-defined test pattern".to_string(),
            sequence: vec!["event_a".to_string(), "event_b".to_string()],
            strict_order: true,
            min_confidence: 0.8,
            severity: MotifSeverity::Low,
            recommendations: vec!["Test recommendation".to_string()],
            tags: vec!["test".to_string()],
            observation_count: 0,
            user_defined: true,
        };

        library.add_motif(custom);

        // Verify it was added
        assert!(library.get_motif("custom_pattern").is_some());
        assert!(library.get_motif("custom_pattern").unwrap().user_defined);
    }

    #[test]
    fn test_motif_by_severity() {
        let library = MotifLibrary::new();

        let critical = library.motifs_by_severity(MotifSeverity::Critical);
        assert!(!critical.is_empty());

        let info = library.motifs_by_severity(MotifSeverity::Info);
        assert!(!info.is_empty());
    }

    #[test]
    fn test_motif_by_tag() {
        let library = MotifLibrary::new();

        let consciousness_patterns = library.motifs_by_tag("consciousness");
        assert!(!consciousness_patterns.is_empty());

        let security_patterns = library.motifs_by_tag("security");
        assert!(!security_patterns.is_empty());
    }

    #[test]
    fn test_statistics_tracking() {
        let mut library = MotifLibrary::new();

        let events = create_test_events(&[
            "security_check",
            "phi_measurement",
            "router_selection",
        ]);

        // First match
        let matches1 = library.match_sequence(&events);
        assert_eq!(library.stats().total_matches, matches1.len());

        // Second match
        let matches2 = library.match_sequence(&events);
        assert_eq!(library.stats().total_matches, matches1.len() + matches2.len());
    }
}
