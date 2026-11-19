use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const REQUIRED_RESOURCE_DIMENSIONS: &[&str] = &[
    "propulsion_energy",
    "hotel_auxiliary_energy",
    "fuel_inventory",
    "battery_usable_energy",
    "potable_water",
    "food_nutrient_groups",
    "grey_black_water_capacity",
    "consumables",
    "critical_spares",
    "emergency_reserve",
];

const REQUIRED_OWNER_ROLES: &[&str] = &[
    "sailing_route_candidate",
    "tack_gybe_transition",
    "sail_capability_graph",
    "operational_design_domain",
    "mission_energy_budget",
    "resource_service_schedule",
    "resource_autonomy_vector",
    "service_continuity",
];

#[derive(Clone, Debug)]
struct Finding {
    code: String,
    severity: String,
    path: String,
    message: String,
}

#[derive(Clone, Debug)]
struct AuditReport {
    case_id: String,
    disposition: String,
    findings: Vec<Finding>,
}

impl AuditReport {
    fn has_code(&self, code: &str) -> bool {
        self.findings.iter().any(|finding| finding.code == code)
    }

    fn to_value(&self) -> Value {
        json!({
            "audit_version": "0.1.0",
            "case_id": self.case_id,
            "disposition": self.disposition,
            "findings": self.findings.iter().map(|finding| json!({
                "code": finding.code,
                "severity": finding.severity,
                "path": finding.path,
                "message": finding.message,
            })).collect::<Vec<_>>(),
            "claim_ceiling": "contract_and_manifest_admissibility_only",
            "digest_note": "A syntactically valid digest is an identity pointer, not proof of authenticity or physical truth."
        })
    }
}

fn add(
    findings: &mut Vec<Finding>,
    code: &str,
    severity: &str,
    path: &str,
    message: &str,
) {
    findings.push(Finding {
        code: code.to_owned(),
        severity: severity.to_owned(),
        path: path.to_owned(),
        message: message.to_owned(),
    });
}

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |current, key| current.get(*key))
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path).and_then(Value::as_str)
}

fn parse_time(value: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(value).ok()
}

fn parse_time_at(
    value: &Value,
    path: &[&str],
    label: &str,
    findings: &mut Vec<Finding>,
) -> Option<DateTime<FixedOffset>> {
    let path_label = path.join(".");
    match string_at(value, path) {
        Some(raw) => match parse_time(raw) {
            Some(parsed) => Some(parsed),
            None => {
                add(
                    findings,
                    "INVALID_TIMESTAMP",
                    "ERROR",
                    &path_label,
                    &format!("{label} must be RFC 3339 with an explicit UTC offset."),
                );
                None
            }
        },
        None => {
            add(
                findings,
                "MISSING_TIMESTAMP",
                "ERROR",
                &path_label,
                &format!("{label} is required."),
            );
            None
        }
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn check_digest(
    value: &Value,
    key: &str,
    location: &str,
    findings: &mut Vec<Finding>,
) {
    match value.get(key).and_then(Value::as_str) {
        Some(digest) if valid_digest(digest) => {}
        Some(_) => add(
            findings,
            "INVALID_DIGEST",
            "ERROR",
            &format!("{location}.{key}"),
            "Expected exactly 64 hexadecimal SHA-256 characters.",
        ),
        None => add(
            findings,
            "MISSING_DIGEST",
            "ERROR",
            &format!("{location}.{key}"),
            "A source/configuration digest is required; missing identity is not inferred.",
        ),
    }
}

fn check_owner_reference(
    reference: &Value,
    location: &str,
    synthetic: bool,
    latest_decision: Option<&DateTime<FixedOffset>>,
    findings: &mut Vec<Finding>,
) {
    const ALLOWED_REPOSITORIES: &[&str] = &[
        "Luminous-Dynamics/symthaea",
        "Luminous-Dynamics/symtropy",
        "Luminous-Dynamics/sol-atlas-leptos",
        "Luminous-Dynamics/mycelix",
        "external",
    ];

    match string_at(reference, &["owner_repo"]) {
        Some(repository) if ALLOWED_REPOSITORIES.contains(&repository) => {}
        Some(_) => add(
            findings,
            "UNKNOWN_OWNER_REPOSITORY",
            "ERROR",
            &format!("{location}.owner_repo"),
            "Repository owner is not in the declared cross-stack owner set.",
        ),
        None => add(
            findings,
            "MISSING_OWNER_REPOSITORY",
            "ERROR",
            &format!("{location}.owner_repo"),
            "Every owner reference must name its source repository.",
        ),
    }

    check_digest(reference, "artifact_sha256", location, findings);

    for key in ["artifact_id", "revision", "claim_limit"] {
        if string_at(reference, &[key]).is_none_or(str::is_empty) {
            add(
                findings,
                "MISSING_OWNER_REFERENCE_FIELD",
                "ERROR",
                &format!("{location}.{key}"),
                "Owner references require stable identity, revision and a non-empty claim limit.",
            );
        }
    }

    let disposition = string_at(reference, &["disposition"]).unwrap_or("unknown");
    let evidence_class = string_at(reference, &["evidence_class"]).unwrap_or("unknown");

    if synthetic {
        if disposition != "synthetic_only" || evidence_class != "synthetic" {
            add(
                findings,
                "SYNTHETIC_PROVENANCE_MISMATCH",
                "ERROR",
                location,
                "A synthetic case may contain only synthetic_only / synthetic references.",
            );
        }
    } else {
        match disposition {
            "supported" => {}
            "partial" | "unknown" | "incomplete" | "synthetic_only" => add(
                findings,
                "OWNER_EVIDENCE_INCOMPLETE",
                "INCOMPLETE",
                location,
                "Owner evidence is not fully supported for a non-synthetic comparison.",
            ),
            "rejected" | "expired" | "revoked" => add(
                findings,
                "OWNER_EVIDENCE_BLOCKED",
                "BLOCKER",
                location,
                "Rejected, expired or revoked evidence cannot support the case.",
            ),
            _ => add(
                findings,
                "UNKNOWN_OWNER_DISPOSITION",
                "ERROR",
                location,
                "Owner evidence disposition is not recognized.",
            ),
        }

        match evidence_class {
            "measured" | "validated_model" | "published" | "derived" => {}
            "unvalidated_model" | "vendor_claim" | "synthetic" | "assumed" | "unknown" => add(
                findings,
                "OWNER_EVIDENCE_CLASS_INSUFFICIENT",
                "INCOMPLETE",
                location,
                "This evidence class cannot independently support a non-synthetic comparison.",
            ),
            _ => add(
                findings,
                "UNKNOWN_EVIDENCE_CLASS",
                "ERROR",
                location,
                "Evidence class is not recognized.",
            ),
        }
    }

    if let Some(valid_until) = reference.get("valid_until").and_then(Value::as_str) {
        match parse_time(valid_until) {
            Some(expiry) => {
                if latest_decision.is_some_and(|decision| expiry < *decision) {
                    add(
                        findings,
                        "OWNER_EVIDENCE_EXPIRED_AT_DECISION",
                        "BLOCKER",
                        &format!("{location}.valid_until"),
                        "The evidence validity interval ends before the latest decision time.",
                    );
                }
            }
            None => add(
                findings,
                "INVALID_TIMESTAMP",
                "ERROR",
                &format!("{location}.valid_until"),
                "valid_until must be RFC 3339 with an explicit UTC offset.",
            ),
        }
    }
}

fn collect_owner_references<'a>(
    value: &'a Value,
    location: &str,
    output: &mut Vec<(&'a Value, String)>,
) {
    match value {
        Value::Object(object) => {
            if object.contains_key("owner_repo")
                && object.contains_key("artifact_sha256")
                && object.contains_key("artifact_id")
            {
                output.push((value, location.to_owned()));
            } else {
                for (key, child) in object {
                    collect_owner_references(child, &format!("{location}.{key}"), output);
                }
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                collect_owner_references(child, &format!("{location}[{index}]"), output);
            }
        }
        _ => {}
    }
}

fn check_mission_window(value: &Value, findings: &mut Vec<Finding>) {
    let start = parse_time_at(
        value,
        &["mission", "arrival_window", "start"],
        "mission arrival-window start",
        findings,
    );
    let end = parse_time_at(
        value,
        &["mission", "arrival_window", "end"],
        "mission arrival-window end",
        findings,
    );
    if let (Some(start), Some(end)) = (start, end) {
        if start > end {
            add(
                findings,
                "INVALID_MISSION_WINDOW",
                "ERROR",
                "mission.arrival_window",
                "Arrival-window start must not be after its end.",
            );
        }
    }
    if string_at(value, &["mission", "arrival_window", "clock_id"])
        .is_none_or(str::is_empty)
    {
        add(
            findings,
            "MISSING_CLOCK_ID",
            "ERROR",
            "mission.arrival_window.clock_id",
            "Mission time intervals require an explicit clock identity.",
        );
    }
}

fn check_decision_snapshots(
    value: &Value,
    family: &str,
    findings: &mut Vec<Finding>,
) -> Option<DateTime<FixedOffset>> {
    let snapshots = match at(value, &["environment", "decision_snapshots"])
        .and_then(Value::as_array)
    {
        Some(snapshots) if !snapshots.is_empty() => snapshots,
        _ => {
            add(
                findings,
                "MISSING_DECISION_SNAPSHOTS",
                "INCOMPLETE",
                "environment.decision_snapshots",
                "At least one decision-time forecast snapshot is required.",
            );
            return None;
        }
    };

    let mut latest: Option<DateTime<FixedOffset>> = None;
    let mut previous: Option<DateTime<FixedOffset>> = None;

    for (index, snapshot) in snapshots.iter().enumerate() {
        let prefix = format!("environment.decision_snapshots[{index}]");
        let decision = parse_time_at(
            snapshot,
            &["decision_time"],
            "decision time",
            findings,
        );
        let issued = parse_time_at(
            snapshot,
            &["source_issue_time"],
            "forecast issue time",
            findings,
        );
        let available = parse_time_at(
            snapshot,
            &["available_at"],
            "forecast availability time",
            findings,
        );

        if let (Some(issued), Some(available)) = (issued.as_ref(), available.as_ref()) {
            if issued > available {
                add(
                    findings,
                    "FORECAST_AVAILABLE_BEFORE_ISSUE",
                    "ERROR",
                    &prefix,
                    "Forecast availability cannot precede its issue time.",
                );
            }
        }

        if let (Some(decision), Some(available)) = (decision.as_ref(), available.as_ref()) {
            if available > decision && family != "perfect_foresight_oracle" {
                add(
                    findings,
                    "FUTURE_INFORMATION_LEAKAGE",
                    "BLOCKER",
                    &prefix,
                    "A deployable policy cannot consume a forecast before its availability time.",
                );
            }
            if previous.as_ref().is_some_and(|prior| decision < prior) {
                add(
                    findings,
                    "DECISION_TIMELINE_NOT_MONOTONIC",
                    "ERROR",
                    &prefix,
                    "Decision snapshots must be in nondecreasing time order.",
                );
            }
            previous = Some(decision.clone());
            if latest.as_ref().is_none_or(|known| decision > known) {
                latest = Some(decision.clone());
            }
        }

        let valid_start = parse_time_at(
            snapshot,
            &["valid_window", "start"],
            "forecast validity start",
            findings,
        );
        let valid_end = parse_time_at(
            snapshot,
            &["valid_window", "end"],
            "forecast validity end",
            findings,
        );
        if let (Some(start), Some(end)) = (valid_start, valid_end) {
            if start > end {
                add(
                    findings,
                    "INVALID_FORECAST_VALIDITY_WINDOW",
                    "ERROR",
                    &prefix,
                    "Forecast validity start must not be after its end.",
                );
            }
        }
        if string_at(snapshot, &["valid_window", "clock_id"]).is_none_or(str::is_empty) {
            add(
                findings,
                "MISSING_CLOCK_ID",
                "ERROR",
                &format!("{prefix}.valid_window.clock_id"),
                "Forecast validity intervals require an explicit clock identity.",
            );
        }
    }

    latest
}

fn check_resource_dimensions(value: &Value, findings: &mut Vec<Finding>) {
    let dimensions = match at(value, &["required_resource_dimensions"]).and_then(Value::as_array)
    {
        Some(dimensions) => dimensions,
        None => {
            add(
                findings,
                "MISSING_RESOURCE_DIMENSIONS",
                "INCOMPLETE",
                "required_resource_dimensions",
                "A complete resource-autonomy roster is required.",
            );
            return;
        }
    };

    let mut seen = BTreeSet::new();
    for dimension in dimensions {
        let Some(dimension) = dimension.as_str() else {
            add(
                findings,
                "INVALID_RESOURCE_DIMENSION",
                "ERROR",
                "required_resource_dimensions",
                "Every resource dimension must be a string.",
            );
            continue;
        };
        if !seen.insert(dimension.to_owned()) {
            add(
                findings,
                "DUPLICATE_RESOURCE_DIMENSION",
                "ERROR",
                "required_resource_dimensions",
                "Resource dimensions must be unique.",
            );
        }
    }

    for required in REQUIRED_RESOURCE_DIMENSIONS {
        if !seen.contains(*required) {
            add(
                findings,
                "RESOURCE_DIMENSION_OMITTED",
                "INCOMPLETE",
                "required_resource_dimensions",
                &format!("Required liveaboard resource dimension '{required}' is absent."),
            );
        }
    }
}

fn check_required_owner_roles(value: &Value, findings: &mut Vec<Finding>) {
    let receipts = match at(value, &["owner_receipts"]).and_then(Value::as_array) {
        Some(receipts) => receipts,
        None => {
            add(
                findings,
                "MISSING_OWNER_RECEIPTS",
                "INCOMPLETE",
                "owner_receipts",
                "Owner-native evidence references are required.",
            );
            return;
        }
    };

    let present: BTreeSet<String> = receipts
        .iter()
        .filter_map(|receipt| string_at(receipt, &["role"]).map(str::to_owned))
        .collect();

    for required in REQUIRED_OWNER_ROLES {
        if !present.contains(*required) {
            add(
                findings,
                "REQUIRED_OWNER_ROLE_MISSING",
                "INCOMPLETE",
                "owner_receipts",
                &format!("Required owner role '{required}' is missing."),
            );
        }
    }
}

fn check_hard_gates(value: &Value, synthetic: bool, findings: &mut Vec<Finding>) {
    let required_ids = match at(value, &["hard_gate_ids"]).and_then(Value::as_array) {
        Some(ids) if !ids.is_empty() => ids,
        _ => {
            add(
                findings,
                "MISSING_HARD_GATE_IDS",
                "ERROR",
                "hard_gate_ids",
                "A non-empty declared hard-gate roster is mandatory.",
            );
            return;
        }
    };

    let mut required = BTreeSet::new();
    for id in required_ids {
        let Some(id) = id.as_str() else {
            add(
                findings,
                "INVALID_HARD_GATE_ID",
                "ERROR",
                "hard_gate_ids",
                "Hard-gate identifiers must be strings.",
            );
            continue;
        };
        if !required.insert(id.to_owned()) {
            add(
                findings,
                "DUPLICATE_HARD_GATE_ID",
                "ERROR",
                "hard_gate_ids",
                "Hard-gate identifiers must be unique.",
            );
        }
    }

    let evaluations = match at(value, &["hard_gate_evaluations"]).and_then(Value::as_array) {
        Some(evaluations) if !evaluations.is_empty() => evaluations,
        _ => {
            add(
                findings,
                "MISSING_HARD_GATE_EVALUATIONS",
                "BLOCKER",
                "hard_gate_evaluations",
                "Every declared hard gate requires an explicit owner-backed evaluation.",
            );
            return;
        }
    };

    let available_roles: BTreeSet<String> = at(value, &["owner_receipts"])
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|receipt| string_at(receipt, &["role"]).map(str::to_owned))
        .collect();

    let mut evaluated = BTreeSet::new();
    for (index, evaluation) in evaluations.iter().enumerate() {
        let path = format!("hard_gate_evaluations[{index}]");
        let Some(id) = string_at(evaluation, &["gate_id"]) else {
            add(
                findings,
                "INVALID_HARD_GATE_EVALUATION",
                "ERROR",
                &path,
                "Hard-gate evaluation has no gate_id.",
            );
            continue;
        };

        if !evaluated.insert(id.to_owned()) {
            add(
                findings,
                "DUPLICATE_HARD_GATE_EVALUATION",
                "ERROR",
                &path,
                "A hard gate may be evaluated only once in a case manifest.",
            );
        }
        if !required.contains(id) {
            add(
                findings,
                "UNDECLARED_HARD_GATE_EVALUATION",
                "ERROR",
                &path,
                "An evaluation references a gate not present in hard_gate_ids.",
            );
        }

        if let Some(roles) = evaluation
            .get("owner_receipt_roles")
            .and_then(Value::as_array)
        {
            for role in roles.iter().filter_map(Value::as_str) {
                if !available_roles.contains(role) {
                    add(
                        findings,
                        "HARD_GATE_OWNER_ROLE_MISSING",
                        "BLOCKER",
                        &path,
                        &format!("Required gate evidence role '{role}' is not present."),
                    );
                }
            }
        } else {
            add(
                findings,
                "HARD_GATE_OWNER_ROLE_MISSING",
                "BLOCKER",
                &path,
                "Hard-gate evaluation must name its owner evidence roles.",
            );
        }

        match string_at(evaluation, &["status"]).unwrap_or("unknown") {
            "pass" => {}
            "unknown" if synthetic => add(
                findings,
                "SYNTHETIC_GATE_UNRESOLVED",
                "INFO",
                &path,
                "Synthetic fixtures intentionally do not qualify an operational hard gate.",
            ),
            "fail" | "blocked" => add(
                findings,
                "HARD_GATE_FAILED",
                "BLOCKER",
                &path,
                "A failed or blocked hard gate excludes the candidate; other objectives cannot compensate.",
            ),
            "unknown" => add(
                findings,
                "HARD_GATE_UNRESOLVED",
                "BLOCKER",
                &path,
                "Unknown hard-gate status cannot be treated as a pass.",
            ),
            _ => add(
                findings,
                "INVALID_HARD_GATE_STATUS",
                "ERROR",
                &path,
                "Hard-gate status must be pass, fail, unknown or blocked.",
            ),
        }

        let evidence_ids = evaluation
            .get("evidence_ids")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if evidence_ids == 0 {
            add(
                findings,
                "HARD_GATE_EVIDENCE_MISSING",
                "BLOCKER",
                &path,
                "Every hard-gate disposition requires at least one evidence identifier.",
            );
        }
    }

    for missing in required.difference(&evaluated) {
        add(
            findings,
            "HARD_GATE_EVALUATION_MISSING",
            "BLOCKER",
            "hard_gate_evaluations",
            &format!("Declared hard gate '{missing}' has no evaluation."),
        );
    }
}

fn audit_case(value: &Value) -> AuditReport {
    let case_id = string_at(value, &["case_id"]).unwrap_or("<missing>").to_owned();
    let mut findings = Vec::new();

    if !value.is_object() {
        add(
            &mut findings,
            "CASE_NOT_OBJECT",
            "ERROR",
            "$",
            "Case manifest root must be a JSON object.",
        );
        return AuditReport {
            case_id,
            disposition: "INVALID".to_owned(),
            findings,
        };
    }

    let record_status = string_at(value, &["record_status"]).unwrap_or("unknown");
    let synthetic = record_status == "synthetic";

    if ![
        "synthetic",
        "shadow_replay",
        "measured_campaign",
        "published_reproduction",
    ]
    .contains(&record_status)
    {
        add(
            &mut findings,
            "INVALID_RECORD_STATUS",
            "ERROR",
            "record_status",
            "record_status must identify synthetic, shadow_replay, measured_campaign or published_reproduction.",
        );
    }

    if string_at(value, &["schema_version"]) != Some("0.1.0") {
        add(
            &mut findings,
            "UNSUPPORTED_SCHEMA_VERSION",
            "ERROR",
            "schema_version",
            "This audit implements only the declared 0.1.0 case contract.",
        );
    }

    if string_at(value, &["claim_ceiling"]) != Some("cross_owner_composition_only") {
        add(
            &mut findings,
            "CLAIM_CEILING_MISMATCH",
            "ERROR",
            "claim_ceiling",
            "The P0 composition audit may not claim a physical, safety or regulatory qualification.",
        );
    }

    for (path, key) in [
        (&["comparison"][..], "service_fingerprint_sha256"),
        (&["craft", "craft_identity"][..], "configuration_sha256"),
        (&["mission"][..], "mission_profile_sha256"),
    ] {
        if let Some(parent) = at(value, path) {
            check_digest(parent, key, &path.join("."), &mut findings);
        } else {
            add(
                &mut findings,
                "MISSING_CASE_IDENTITY",
                "ERROR",
                &path.join("."),
                "Required identity object is missing.",
            );
        }
    }

    if string_at(value, &["craft", "craft_identity", "id"]).is_none_or(str::is_empty)
        || string_at(value, &["craft", "craft_identity", "revision"]).is_none_or(str::is_empty)
    {
        add(
            &mut findings,
            "MISSING_CRAFT_IDENTITY",
            "ERROR",
            "craft.craft_identity",
            "Craft identity and revision are required.",
        );
    }

    if at(value, &["mission", "occupants"])
        .and_then(Value::as_u64)
        .is_none_or(|occupants| occupants == 0)
    {
        add(
            &mut findings,
            "INVALID_OCCUPANCY",
            "ERROR",
            "mission.occupants",
            "Occupancy must be a positive integer.",
        );
    }

    let family = string_at(value, &["policy_arm", "family"]).unwrap_or("unknown");
    const POLICY_FAMILIES: &[&str] = &[
        "fixed_conservative",
        "polar_vmg",
        "transition_aware_vmg",
        "weather_routing",
        "reserve_aware_weather_routing",
        "integrated_advisory",
        "perfect_foresight_oracle",
    ];
    if !POLICY_FAMILIES.contains(&family) {
        add(
            &mut findings,
            "INVALID_POLICY_FAMILY",
            "ERROR",
            "policy_arm.family",
            "Policy family is not part of the versioned benchmark arm set.",
        );
    }
    let information_policy =
        string_at(value, &["policy_arm", "information_policy"]).unwrap_or("unknown");

    let comfort_requirement =
        string_at(value, &["mission", "comfort_requirement"]).unwrap_or("unknown");
    let comfort_fallback =
        string_at(value, &["policy_arm", "comfort_fallback_status"]).unwrap_or("unknown");
    match (comfort_requirement, comfort_fallback) {
        ("hard", "none") | ("soft", "none") | ("not_applicable", "none") => {}
        ("hard", "no_feasible_route")
        | ("hard", "automatically_relaxed")
        | ("hard", "operator_authorized_relaxation") => add(
            &mut findings,
            "HARD_COMFORT_CONSTRAINT_NOT_MET",
            "BLOCKER",
            "policy_arm.comfort_fallback_status",
            "A route that does not satisfy the declared hard comfort envelope is not feasible; it cannot be promoted after automatic or operator relaxation.",
        ),
        ("soft", "automatically_relaxed") | ("soft", "operator_authorized_relaxation") => add(
            &mut findings,
            "SOFT_COMFORT_PREFERENCE_UNMET",
            "INCOMPLETE",
            "policy_arm.comfort_fallback_status",
            "The soft comfort preference was relaxed; retain the result as incomplete for comfort-sensitive comparison.",
        ),
        ("soft", "no_feasible_route") => add(
            &mut findings,
            "NO_FEASIBLE_COMFORT_ROUTE",
            "BLOCKER",
            "policy_arm.comfort_fallback_status",
            "No route satisfying the declared comfort preference was returned.",
        ),
        ("not_applicable", _) => add(
            &mut findings,
            "COMFORT_FALLBACK_WITHOUT_REQUIREMENT",
            "ERROR",
            "policy_arm.comfort_fallback_status",
            "A comfort fallback is invalid when comfort is not an applicable mission requirement.",
        ),
        _ => add(
            &mut findings,
            "INVALID_COMFORT_POLICY",
            "ERROR",
            "mission.comfort_requirement",
            "Comfort requirement or fallback status is not recognized.",
        ),
    }

    if at(value, &["policy_arm", "compute_budget_ms"])
        .and_then(Value::as_u64)
        .is_none_or(|budget| budget == 0)
    {
        add(
            &mut findings,
            "INVALID_COMPUTE_BUDGET",
            "ERROR",
            "policy_arm.compute_budget_ms",
            "The comparison compute budget must be a positive integer.",
        );
    }

    if family == "perfect_foresight_oracle" {
        if information_policy != "realized_environment_evaluator_only" {
            add(
                &mut findings,
                "PERFECT_FORESIGHT_POLICY_MISLABELED",
                "ERROR",
                "policy_arm.information_policy",
                "Perfect foresight must be explicitly evaluator-only and non-deployable.",
            );
        }
    } else if information_policy != "as_of_decision_time" {
        add(
            &mut findings,
            "DEPLOYABLE_POLICY_INFORMATION_MISMATCH",
            "ERROR",
            "policy_arm.information_policy",
            "A deployable arm must use only information available at each decision time.",
        );
    }

    check_mission_window(value, &mut findings);
    let latest_decision = check_decision_snapshots(value, family, &mut findings);
    if family != "perfect_foresight_oracle" {
        let realized_digest = string_at(
            value,
            &["environment", "realized_environment_ref", "artifact_sha256"],
        );
        if let (Some(realized_digest), Some(snapshots)) = (
            realized_digest,
            at(value, &["environment", "decision_snapshots"]).and_then(Value::as_array),
        ) {
            for (index, snapshot) in snapshots.iter().enumerate() {
                if string_at(snapshot, &["forecast_ref", "artifact_sha256"])
                    == Some(realized_digest)
                {
                    add(
                        &mut findings,
                        "REALIZED_ENVIRONMENT_LEAKAGE",
                        "BLOCKER",
                        &format!("environment.decision_snapshots[{index}].forecast_ref"),
                        "The evaluator-only realized environment cannot be supplied as a deployable forecast input.",
                    );
                }
            }
        }
    }
    check_resource_dimensions(value, &mut findings);
    check_required_owner_roles(value, &mut findings);
    check_hard_gates(value, synthetic, &mut findings);

    let mut owner_references = Vec::new();
    collect_owner_references(value, "$", &mut owner_references);
    if owner_references.is_empty() {
        add(
            &mut findings,
            "NO_OWNER_REFERENCES",
            "INCOMPLETE",
            "owner_receipts",
            "No owner-native source references were found.",
        );
    }
    for (reference, location) in owner_references {
        check_owner_reference(
            reference,
            &location,
            synthetic,
            latest_decision.as_ref(),
            &mut findings,
        );
    }

    if at(value, &["comparison", "matching_deviations"])
        .and_then(Value::as_array)
        .is_some_and(|deviations| !deviations.is_empty())
    {
        add(
            &mut findings,
            "MATCHING_DEVIATIONS_PRESENT",
            "INCOMPLETE",
            "comparison.matching_deviations",
            "A case with declared service-matching deviations is not eligible for an apples-to-apples comparison.",
        );
    }

    let disposition = if findings.iter().any(|finding| finding.severity == "ERROR") {
        "INVALID"
    } else if findings.iter().any(|finding| finding.severity == "BLOCKER") {
        "BLOCKED"
    } else if findings
        .iter()
        .any(|finding| finding.severity == "INCOMPLETE")
    {
        "INCOMPLETE"
    } else if synthetic {
        "SYNTHETIC_ONLY"
    } else {
        "ELIGIBLE_FOR_COMPARISON"
    };

    AuditReport {
        case_id,
        disposition: disposition.to_owned(),
        findings,
    }
}

fn shared_receipt_map(value: &Value) -> BTreeMap<String, Vec<String>> {
    let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let receipts = at(value, &["owner_receipts"])
        .and_then(Value::as_array)
        .into_iter()
        .flatten();

    for receipt in receipts {
        let role = string_at(receipt, &["role"]).unwrap_or("unknown");
        if role == "sailing_route_candidate" {
            continue;
        }
        if let Some(digest) = string_at(receipt, &["reference", "artifact_sha256"]) {
            result.entry(role.to_owned()).or_default().push(digest.to_ascii_lowercase());
        } else {
            result.entry(role.to_owned()).or_default().push("<missing>".to_owned());
        }
    }

    for digests in result.values_mut() {
        digests.sort();
    }
    result
}

fn compare_pair(baseline: &Value, candidate: &Value) -> Value {
    let baseline_audit = audit_case(baseline);
    let candidate_audit = audit_case(candidate);
    let mut mismatches = Vec::new();

    let required_equal_paths: &[&[&str]] = &[
        &["record_status"],
        &["comparison", "comparison_group_id"],
        &["comparison", "service_fingerprint_sha256"],
        &["craft", "craft_identity", "id"],
        &["craft", "craft_identity", "revision"],
        &["craft", "craft_identity", "configuration_sha256"],
        &["craft", "loading_condition_id"],
        &["mission", "mission_identity", "id"],
        &["mission", "mission_identity", "revision"],
        &["mission", "mission_profile_sha256"],
        &["mission", "occupants"],
        &["mission", "target_duration_h"],
        &["mission", "comfort_requirement"],
        &["mission", "arrival_window"],
        &["mission", "route_or_waypoint_ref", "artifact_sha256"],
        &["mission", "service_level_ref", "artifact_sha256"],
        &["mission", "reserve_policy_ref", "artifact_sha256"],
        &["environment", "decision_snapshots"],
        &["environment", "realized_environment_ref", "artifact_sha256"],
        &["required_resource_dimensions"],
        &["hard_gate_ids"],
    ];

    for path in required_equal_paths {
        if at(baseline, path) != at(candidate, path) {
            mismatches.push(path.join("."));
        }
    }

    let baseline_policy = string_at(baseline, &["policy_arm", "information_policy"]);
    let candidate_policy = string_at(candidate, &["policy_arm", "information_policy"]);
    let baseline_family = string_at(baseline, &["policy_arm", "family"]);
    let candidate_family = string_at(candidate, &["policy_arm", "family"]);

    let both_oracle = baseline_family == Some("perfect_foresight_oracle")
        && candidate_family == Some("perfect_foresight_oracle");
    let mixed_oracle = (baseline_family == Some("perfect_foresight_oracle"))
        != (candidate_family == Some("perfect_foresight_oracle"));

    if mixed_oracle
        || baseline_policy != candidate_policy
        || (!both_oracle && baseline_policy != Some("as_of_decision_time"))
    {
        mismatches.push("policy_arm.information_policy".to_owned());
    }

    if baseline.get("comparison").and_then(|v| v.get("matching_deviations"))
        != candidate.get("comparison").and_then(|v| v.get("matching_deviations"))
        || at(baseline, &["comparison", "matching_deviations"])
            .and_then(Value::as_array)
            .is_some_and(|deviations| !deviations.is_empty())
        || at(candidate, &["comparison", "matching_deviations"])
            .and_then(Value::as_array)
            .is_some_and(|deviations| !deviations.is_empty())
    {
        mismatches.push("comparison.matching_deviations".to_owned());
    }

    if shared_receipt_map(baseline) != shared_receipt_map(candidate) {
        mismatches.push("owner_receipts.shared_inputs".to_owned());
    }

    mismatches.sort();
    mismatches.dedup();

    let disposition = if !mismatches.is_empty() {
        "INCOMPARABLE"
    } else if baseline_audit.disposition == "SYNTHETIC_ONLY"
        && candidate_audit.disposition == "SYNTHETIC_ONLY"
    {
        "MATCHED_SYNTHETIC_CASES_ONLY"
    } else if baseline_audit.disposition == "ELIGIBLE_FOR_COMPARISON"
        && candidate_audit.disposition == "ELIGIBLE_FOR_COMPARISON"
    {
        if both_oracle {
            "MATCHED_PERFECT_FORESIGHT_BOUND_ONLY"
        } else {
            "MATCHED_FOR_COMPARISON"
        }
    } else {
        "MATCHED_BUT_NOT_ADMISSIBLE"
    };

    json!({
        "audit_version": "0.1.0",
        "baseline_case_id": string_at(baseline, &["case_id"]),
        "candidate_case_id": string_at(candidate, &["case_id"]),
        "disposition": disposition,
        "mismatched_fields": mismatches,
        "baseline_case_disposition": baseline_audit.disposition,
        "candidate_case_disposition": candidate_audit.disposition,
        "finding_codes": {
            "baseline": baseline_audit.findings.iter().map(|f| f.code.clone()).collect::<Vec<_>>(),
            "candidate": candidate_audit.findings.iter().map(|f| f.code.clone()).collect::<Vec<_>>()
        },
        "claim_ceiling": "matched_manifest_comparability_only",
        "note": "This oracle establishes manifest comparability, not physical performance, route safety, or superiority."
    })
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&bytes).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn print_json(value: &Value) -> Result<(), String> {
    let output = serde_json::to_string_pretty(value)
        .map_err(|error| format!("could not serialize audit report: {error}"))?;
    println!("{output}");
    Ok(())
}

fn usage() {
    eprintln!(
        "Usage:\n  marine-sail-liveaboard-audit audit-case <case.json>\n  marine-sail-liveaboard-audit compare-pair <baseline.json> <candidate.json>"
    );
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    match arguments.get(1).map(String::as_str) {
        Some("audit-case") if arguments.len() == 3 => {
            let case = read_json(&arguments[2])?;
            print_json(&audit_case(&case).to_value())
        }
        Some("compare-pair") if arguments.len() == 4 => {
            let baseline = read_json(&arguments[2])?;
            let candidate = read_json(&arguments[3])?;
            print_json(&compare_pair(&baseline, &candidate))
        }
        _ => {
            usage();
            Err("invalid command line".to_owned())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-sail-liveaboard-case-0001.json"
        ))
        .expect("synthetic fixture must parse")
    }

    fn set_all_evidence_supported(value: &mut Value) {
        value["record_status"] = json!("shadow_replay");

        fn visit(node: &mut Value) {
            match node {
                Value::Object(object) => {
                    if object.contains_key("artifact_sha256") && object.contains_key("owner_repo") {
                        object.insert("disposition".to_owned(), json!("supported"));
                        object.insert("evidence_class".to_owned(), json!("measured"));
                    } else {
                        for child in object.values_mut() {
                            visit(child);
                        }
                    }
                }
                Value::Array(items) => {
                    for child in items {
                        visit(child);
                    }
                }
                _ => {}
            }
        }

        visit(value);
        if let Some(gates) = value["hard_gate_evaluations"].as_array_mut() {
            for gate in gates {
                gate["status"] = json!("pass");
            }
        }
    }

    #[test]
    fn synthetic_fixture_is_never_promoted_to_physical_evidence() {
        let report = audit_case(&fixture());
        assert_eq!(report.disposition, "SYNTHETIC_ONLY");
        assert!(report.has_code("SYNTHETIC_GATE_UNRESOLVED"));
    }

    #[test]
    fn forecast_unavailable_at_decision_is_rejected() {
        let mut value = fixture();
        value["environment"]["decision_snapshots"][0]["available_at"] =
            json!("2026-11-19T08:05:00Z");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("FUTURE_INFORMATION_LEAKAGE"));
    }

    #[test]
    fn forecast_issue_after_availability_is_invalid() {
        let mut value = fixture();
        value["environment"]["decision_snapshots"][0]["source_issue_time"] =
            json!("2026-11-19T06:10:00Z");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.has_code("FORECAST_AVAILABLE_BEFORE_ISSUE"));
    }

    #[test]
    fn revoked_owner_receipt_blocks_non_synthetic_case() {
        let mut value = fixture();
        set_all_evidence_supported(&mut value);
        value["owner_receipts"][2]["reference"]["disposition"] = json!("revoked");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("OWNER_EVIDENCE_BLOCKED"));
    }

    #[test]
    fn omitted_autonomy_dimension_prevents_comparison() {
        let mut value = fixture();
        value["required_resource_dimensions"]
            .as_array_mut()
            .unwrap()
            .retain(|dimension| dimension.as_str() != Some("potable_water"));
        let report = audit_case(&value);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.has_code("RESOURCE_DIMENSION_OMITTED"));
    }

    #[test]
    fn unknown_hard_gate_blocks_non_synthetic_case() {
        let mut value = fixture();
        set_all_evidence_supported(&mut value);
        value["hard_gate_evaluations"][0]["status"] = json!("unknown");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("HARD_GATE_UNRESOLVED"));
    }

    #[test]
    fn realized_environment_cannot_be_used_as_a_deployable_forecast() {
        let mut value = fixture();
        let realized_digest = value["environment"]["realized_environment_ref"]["artifact_sha256"]
            .clone();
        value["environment"]["decision_snapshots"][0]["forecast_ref"]["artifact_sha256"] =
            realized_digest;
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("REALIZED_ENVIRONMENT_LEAKAGE"));
    }

    #[test]
    fn unsupported_schema_version_is_invalid() {
        let mut value = fixture();
        value["schema_version"] = json!("0.2.0");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.has_code("UNSUPPORTED_SCHEMA_VERSION"));
    }

    #[test]
    fn automatic_comfort_relaxation_cannot_pass_a_hard_requirement() {
        let mut value = fixture();
        value["policy_arm"]["comfort_fallback_status"] = json!("automatically_relaxed");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("HARD_COMFORT_CONSTRAINT_NOT_MET"));
    }

    #[test]
    fn soft_comfort_relaxation_remains_incomplete() {
        let mut value = fixture();
        value["mission"]["comfort_requirement"] = json!("soft");
        value["policy_arm"]["comfort_fallback_status"] = json!("automatically_relaxed");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.has_code("SOFT_COMFORT_PREFERENCE_UNMET"));
    }

    #[test]
    fn malformed_digest_is_invalid() {
        let mut value = fixture();
        value["craft"]["craft_identity"]["configuration_sha256"] = json!("not-a-digest");
        let report = audit_case(&value);
        assert_eq!(report.disposition, "INVALID");
        assert!(report.has_code("INVALID_DIGEST"));
    }

    #[test]
    fn matching_synthetic_pair_is_not_a_performance_claim() {
        let baseline = fixture();
        let mut candidate = fixture();
        candidate["case_id"] = json!("SYNTHETIC-BASELINE-CASE-0002");
        candidate["comparison"]["candidate_id"] = json!("SYNTHETIC-POLAR-VMG-01");
        candidate["policy_arm"]["policy_id"] = json!("SYNTHETIC-S1-POLAR");
        candidate["policy_arm"]["family"] = json!("polar_vmg");
        let report = compare_pair(&baseline, &candidate);
        assert_eq!(report["disposition"], "MATCHED_SYNTHETIC_CASES_ONLY");
    }

    #[test]
    fn mismatched_occupancy_makes_pair_incomparable() {
        let baseline = fixture();
        let mut candidate = fixture();
        candidate["mission"]["occupants"] = json!(3);
        let report = compare_pair(&baseline, &candidate);
        assert_eq!(report["disposition"], "INCOMPARABLE");
        assert!(report["mismatched_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "mission.occupants"));
    }

    #[test]
    fn perfect_foresight_is_not_information_matched_to_deployable_policy() {
        let baseline = fixture();
        let mut candidate = fixture();
        candidate["policy_arm"]["family"] = json!("perfect_foresight_oracle");
        candidate["policy_arm"]["information_policy"] =
            json!("realized_environment_evaluator_only");
        let report = compare_pair(&baseline, &candidate);
        assert_eq!(report["disposition"], "INCOMPARABLE");
    }

    #[test]
    fn all_declared_gate_evaluations_must_exist() {
        let mut value = fixture();
        value["hard_gate_evaluations"]
            .as_array_mut()
            .unwrap()
            .pop();
        let report = audit_case(&value);
        assert_eq!(report.disposition, "BLOCKED");
        assert!(report.has_code("HARD_GATE_EVALUATION_MISSING"));
    }
}
