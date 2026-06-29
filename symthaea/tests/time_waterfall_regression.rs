use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SourceMode {
    ScriptedDemo,
    LiveSimulation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DashboardEvent {
    pub event_id: String,
    pub event_type: String,
    pub label: String,
    pub description: String,
    pub severity: String,
    pub absolute_frame_index: u64,
    pub history_offset: u64,
    pub causal_role: String,
    pub metric_impacts: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegimeInterval {
    pub r#type: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimeWaterfallFrame {
    pub frame_index: u64,
    pub sim_time_s: f64,
    pub source_mode: SourceMode,
    pub metrics: [f64; 7],
    pub confidence: f64,
    pub anomaly_bits: u8,
    pub anomaly_flags: Vec<String>,
    pub is_chronicle: bool,
    pub events: Vec<DashboardEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportSummary {
    pub event_count: usize,
    pub peak_fep_prediction_error: f64,
    pub min_phi_integration: f64,
    pub peak_anomaly_score: f64,
    pub peak_memory_pressure: f64,
    pub peak_mip_instability: f64,
    pub perturbation_frame: Option<u64>,
    pub recovery_frame: Option<u64>,
    pub first_chronicle_marked_frame: Option<u64>,
    pub durability_commit_frame: Option<u64>,
    pub chronicle_marked_frame_count: usize,
    pub durable_record_event_count: usize,
    pub frames_to_recovery: Option<u64>,
    pub seconds_to_recovery: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvidenceBundle {
    pub schema_version: String,
    pub exported_at: String,
    pub source_mode: SourceMode,
    pub scenario: String,
    pub frame_order: String,
    pub history_len: usize,
    pub metric_names: Vec<String>,
    pub anomaly_bit_legend: std::collections::HashMap<String, String>,
    pub frames: Vec<TimeWaterfallFrame>,
    pub intervals: Vec<RegimeInterval>,
    pub summary: ExportSummary,
}

fn build_test_demo_bundle() -> EvidenceBundle {
    let mut frames = Vec::with_capacity(64);
    let base_tick = 1280u64; // absolute start offset

    // Phase 1: Stable baseline (20 frames: 1280–1299)
    for i in 0..20u64 {
        let abs_idx = base_tick + i;
        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: i as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [3.2, 0.15, 0.6, 0.85, 0.05, 0.8, 0.1],
            confidence: 0.85,
            anomaly_bits: 0,
            anomaly_flags: Vec::new(),
            is_chronicle: false,
            events: Vec::new(),
        });
    }

    // Phase 2: Ingress Contradiction (10 frames: 1300–1309)
    for i in 0..10u64 {
        let abs_idx = base_tick + 20 + i;
        let mut anomaly_flags = Vec::new();
        anomaly_flags.push("prediction_contradiction".to_string());

        let mut events = Vec::new();
        if i == 0 {
            events.push(DashboardEvent {
                event_id: "evt_001300_ingress_contradiction".to_string(),
                event_type: "ingress_contradiction".to_string(),
                label: "P: Ingress Contradiction".to_string(),
                description: " aquifer intrusion mismatch.".to_string(),
                severity: "high".to_string(),
                absolute_frame_index: abs_idx,
                history_offset: 20,
                causal_role: "perturbation".to_string(),
                metric_impacts: vec![
                    ("fep_prediction_error".to_string(), 0.35),
                    ("phi_integration".to_string(), -0.2),
                ],
            });
        }

        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: (20 + i) as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [1.2, 2.85, 0.9, 0.65, 0.55, 1.8, 0.7],
            confidence: 0.65,
            anomaly_bits: 0b0010001,
            anomaly_flags,
            is_chronicle: i == 0, // first chronicle marked frame (1300)
            events,
        });
    }

    // Phase 3: False-green diagnostic (5 frames: 1310–1314)
    for i in 0..5u64 {
        let abs_idx = base_tick + 30 + i;
        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: (30 + i) as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [1.0, 0.001, 0.5, 0.999, 0.35, 0.001, 0.001],
            confidence: 0.70,
            anomaly_bits: 0b1000000,
            anomaly_flags: vec!["false_green_diagnostic".to_string()],
            is_chronicle: false,
            events: Vec::new(),
        });
    }

    // Phase 4: Recovery mode (5 frames: 1315–1319)
    for i in 0..5u64 {
        let abs_idx = base_tick + 35 + i;
        let mut events = Vec::new();
        if i == 0 {
            events.push(DashboardEvent {
                event_id: "evt_001315_recovery_begins".to_string(),
                event_type: "recovery_begins".to_string(),
                label: "REC: Recovery begins".to_string(),
                description: "Recovery start.".to_string(),
                severity: "medium".to_string(),
                absolute_frame_index: abs_idx,
                history_offset: 35,
                causal_role: "mitigation".to_string(),
                metric_impacts: vec![
                    ("workspace_activation".to_string(), 0.22),
                    ("fep_prediction_error".to_string(), -0.45),
                ],
            });
        }
        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: (35 + i) as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [2.0, 0.5, 0.8, 0.75, 0.25, 1.0, 0.3],
            confidence: 0.75,
            anomaly_bits: 0,
            anomaly_flags: vec!["recovery_mode".to_string()],
            is_chronicle: false,
            events,
        });
    }

    // Phase 5: Transition/Stable (14 frames: 1320–1333)
    for i in 0..14u64 {
        let abs_idx = base_tick + 40 + i;
        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: (40 + i) as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [3.0, 0.15, 0.6, 0.85, 0.05, 0.8, 0.1],
            confidence: 0.85,
            anomaly_bits: 0,
            anomaly_flags: Vec::new(),
            is_chronicle: false,
            events: Vec::new(),
        });
    }

    // Phase 6: Durable commit (10 frames: 1334–1343)
    for i in 0..10u64 {
        let abs_idx = base_tick + 54 + i;
        let mut events = Vec::new();
        if i == 0 {
            events.push(DashboardEvent {
                event_id: "evt_001334_durable_record".to_string(),
                event_type: "chronicle_event".to_string(),
                label: "CHR: Chronicle event".to_string(),
                description: "Durability commit.".to_string(),
                severity: "low".to_string(),
                absolute_frame_index: abs_idx,
                history_offset: 54,
                causal_role: "durable_record".to_string(),
                metric_impacts: vec![("phi_integration".to_string(), 0.1)],
            });
        }
        frames.push(TimeWaterfallFrame {
            frame_index: abs_idx,
            sim_time_s: (54 + i) as f64 * 0.1,
            source_mode: SourceMode::ScriptedDemo,
            metrics: [3.2, 0.15, 0.6, 0.88, 0.05, 0.75, 0.08],
            confidence: 0.95,
            anomaly_bits: 0,
            anomaly_flags: Vec::new(),
            is_chronicle: false,
            events,
        });
    }

    // Compute intervals using the exact updated logic
    let mut intervals = Vec::new();

    // 1. Prediction Contradiction
    let mut cont_start = None;
    for f in &frames {
        let has_contr = f
            .anomaly_flags
            .iter()
            .any(|s| s == "prediction_contradiction");
        if has_contr && cont_start.is_none() {
            cont_start = Some(f.frame_index);
        } else if !has_contr && cont_start.is_some() {
            let start = cont_start.take().unwrap();
            intervals.push(RegimeInterval {
                r#type: "prediction_contradiction".to_string(),
                start_frame: start,
                end_frame: f.frame_index - 1,
                duration_s: (f.frame_index - start) as f64 * 0.1,
            });
        }
    }
    if let Some(start) = cont_start {
        let last_f = frames.last().map(|f| f.frame_index).unwrap_or(start);
        intervals.push(RegimeInterval {
            r#type: "prediction_contradiction".to_string(),
            start_frame: start,
            end_frame: last_f,
            duration_s: (last_f + 1 - start) as f64 * 0.1,
        });
    }

    // 2. False-Green
    let mut fg_start = None;
    for f in &frames {
        let has_fg = f
            .anomaly_flags
            .iter()
            .any(|s| s == "false_green_diagnostic");
        if has_fg && fg_start.is_none() {
            fg_start = Some(f.frame_index);
        } else if !has_fg && fg_start.is_some() {
            let start = fg_start.take().unwrap();
            intervals.push(RegimeInterval {
                r#type: "false_green_diagnostic".to_string(),
                start_frame: start,
                end_frame: f.frame_index - 1,
                duration_s: (f.frame_index - start) as f64 * 0.1,
            });
        }
    }
    if let Some(start) = fg_start {
        let last_f = frames.last().map(|f| f.frame_index).unwrap_or(start);
        intervals.push(RegimeInterval {
            r#type: "false_green_diagnostic".to_string(),
            start_frame: start,
            end_frame: last_f,
            duration_s: (last_f + 1 - start) as f64 * 0.1,
        });
    }

    // 3. Recovery
    let mut rec_start = None;
    let mut in_event_recovery = false;
    for f in &frames {
        let has_rec_flag = f.anomaly_flags.iter().any(|s| s == "recovery_mode");
        let has_rec_event = f.events.iter().any(|e| e.event_type == "recovery_begins");
        if has_rec_event {
            in_event_recovery = true;
        }
        let is_stable = f.anomaly_flags.is_empty() && f.metrics[4] < 0.22;
        if is_stable {
            in_event_recovery = false;
        }

        let is_recovering = has_rec_flag || in_event_recovery;

        if is_recovering && rec_start.is_none() {
            rec_start = Some(f.frame_index);
        } else if !is_recovering && rec_start.is_some() {
            let start = rec_start.take().unwrap();
            intervals.push(RegimeInterval {
                r#type: "recovery_mode".to_string(),
                start_frame: start,
                end_frame: f.frame_index - 1,
                duration_s: (f.frame_index - start) as f64 * 0.1,
            });
        }
    }
    if let Some(start) = rec_start {
        let last_f = frames.last().map(|f| f.frame_index).unwrap_or(start);
        intervals.push(RegimeInterval {
            r#type: "recovery_mode".to_string(),
            start_frame: start,
            end_frame: last_f,
            duration_s: (last_f + 1 - start) as f64 * 0.1,
        });
    }

    // 4. Durability commit. Chronicle marking is earlier evidence selection;
    // durable storage starts only when the durable record event arrives.
    let mut dur_start = None;
    for f in &frames {
        let is_dur = f
            .events
            .iter()
            .any(|e| e.causal_role == "durable_record" || e.event_type == "chronicle_event");
        if is_dur && dur_start.is_none() {
            dur_start = Some(f.frame_index);
        }
    }
    if let Some(start) = dur_start {
        let last_f = frames.last().map(|f| f.frame_index).unwrap_or(start);
        intervals.push(RegimeInterval {
            r#type: "durability_commit".to_string(),
            start_frame: start,
            end_frame: last_f,
            duration_s: (last_f + 1 - start) as f64 * 0.1,
        });
    }

    let summary = ExportSummary {
        event_count: 3,
        peak_fep_prediction_error: 2.85,
        min_phi_integration: 1.0,
        peak_anomaly_score: 0.55,
        peak_memory_pressure: 1.8,
        peak_mip_instability: 0.7,
        perturbation_frame: Some(1300),
        recovery_frame: Some(1315),
        first_chronicle_marked_frame: Some(1300),
        durability_commit_frame: Some(1334),
        chronicle_marked_frame_count: 10,
        durable_record_event_count: 1,
        frames_to_recovery: Some(15),
        seconds_to_recovery: Some(1.5),
    };

    EvidenceBundle {
        schema_version: "time_waterfall_export_v0.2".to_string(),
        exported_at: "2026-06-27T12:00:00+02:00".to_string(),
        source_mode: SourceMode::ScriptedDemo,
        scenario: "IngressContradictionDemo".to_string(),
        frame_order: "chronological".to_string(),
        history_len: 64,
        metric_names: vec![
            "phi".to_string(),
            "fep_prediction_error".to_string(),
            "workspace_activation".to_string(),
            "hot_confidence".to_string(),
            "anomaly_score".to_string(),
            "memory_pressure".to_string(),
            "mip_instability".to_string(),
        ],
        anomaly_bit_legend: std::collections::HashMap::new(),
        frames,
        intervals,
        summary,
    }
}

#[test]
fn test_schema_regression() {
    let fixture_path = "tests/fixtures/time_waterfall_ingress_contradiction_v0_2.json";
    let bundle = build_test_demo_bundle();
    let json = serde_json::to_string_pretty(&bundle).unwrap();
    let expected = fs::read_to_string(fixture_path).unwrap();
    let actual_json: serde_json::Value = serde_json::from_str(&json).unwrap();
    let expected_json: serde_json::Value = serde_json::from_str(&expected).unwrap();
    assert_eq!(actual_json, expected_json);

    // Verify frames are chronological
    for i in 1..bundle.frames.len() {
        assert!(bundle.frames[i].frame_index > bundle.frames[i - 1].frame_index);
    }

    // Verify event details
    let mut found_event = false;
    for f in &bundle.frames {
        for e in &f.events {
            assert!(e.absolute_frame_index > 0);
            assert!(e.history_offset >= 20); // starts from contradiction phase
            found_event = true;
        }
    }
    assert!(found_event);

    // Verify intervals correctly detected
    let contr_interval = bundle
        .intervals
        .iter()
        .find(|i| i.r#type == "prediction_contradiction")
        .unwrap();
    assert_eq!(contr_interval.start_frame, 1300);
    assert_eq!(contr_interval.end_frame, 1309);

    let false_green_interval = bundle
        .intervals
        .iter()
        .find(|i| i.r#type == "false_green_diagnostic")
        .unwrap();
    assert_eq!(false_green_interval.start_frame, 1310);
    assert_eq!(false_green_interval.end_frame, 1314);

    let recovery_interval = bundle
        .intervals
        .iter()
        .find(|i| i.r#type == "recovery_mode")
        .unwrap();
    assert_eq!(recovery_interval.start_frame, 1315);
    assert_eq!(recovery_interval.end_frame, 1319);

    let durability_interval = bundle
        .intervals
        .iter()
        .find(|i| i.r#type == "durability_commit")
        .unwrap();
    assert_eq!(durability_interval.start_frame, 1334);
    assert_eq!(durability_interval.end_frame, 1343);

    // Verify durability commit frame is not confused with chronicle marked frame
    assert_eq!(bundle.summary.first_chronicle_marked_frame, Some(1300));
    assert_eq!(bundle.summary.durability_commit_frame, Some(1334));
    assert_ne!(
        bundle.summary.first_chronicle_marked_frame,
        bundle.summary.durability_commit_frame
    );

    // Verify summary recovery time is correct
    assert_eq!(bundle.summary.seconds_to_recovery, Some(1.5));
}
