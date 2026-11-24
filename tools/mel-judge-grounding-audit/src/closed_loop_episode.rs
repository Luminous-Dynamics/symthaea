//! Structural replay/admission reducer for one MEL closed-loop episode.
//!
//! This module validates cross-layer chronology, exact join keys, conservative
//! capture accounting, action/disposition compatibility and detector eligibility.
//! It does NOT authenticate signatures, open receipt artifacts, replay external
//! verifiers, decode audio, or qualify detector accuracy. Its outcomes are therefore
//! explicitly "reported" and "structurally eligible", never ground-truth claims.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

const SCHEMA: &str = "mel-closed-loop-episode-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Episode {
    schema: String,
    episode_id: String,
    session: Session,
    execution_identity: ExecutionIdentity,
    forecast: Forecast,
    disposition: Disposition,
    decision: Decision,
    effect: Option<Effect>,
    capture: Capture,
    observation: Observation,
    provenance: Provenance,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Session {
    session_id: String,
    turn_id: String,
    input_sequence: u64,
    target_window: TimeWindow,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeWindow {
    clock_domain: String,
    start_ns: u64,
    end_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionIdentity {
    kind: String,
    policy_id: String,
    policy_version: String,
    backend_id: String,
    identity_sha256: String,
    model_artifact_sha256: Option<String>,
    configuration_sha256: String,
    tokenizer_sha256: Option<String>,
    adapter_sha256: String,
    runtime_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Forecast {
    forecast_id: String,
    revision: u64,
    committed_sequence: u64,
    input_through_sequence: u64,
    target_window: TimeWindow,
    expires_at_sequence: u64,
    uncertainty: String,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Disposition {
    action: String,
    forecast_id: String,
    forecast_revision: u64,
    sequence: u64,
    actor_identity: String,
    scope_sha256: String,
    superseding_forecast_id: Option<String>,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    action: String,
    sequence: u64,
    observed_through_input_sequence: u64,
    forecast_id: Option<String>,
    forecast_revision: Option<u64>,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Effect {
    reported_stage: String,
    forecast_id: Option<String>,
    forecast_revision: Option<u64>,
    decision_sequence: u64,
    producer_identity: String,
    output_artifact_sha256: Option<String>,
    capture_id: Option<String>,
    veto_race: Option<VetoRaceEvidence>,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VetoRaceEvidence {
    authorization_id: String,
    admission_stage: String,
    admission_event_sequence: u64,
    veto_event_sequence: u64,
    reported_stage_event_sequence: u64,
    prior_output_epoch: u64,
    veto_output_epoch: u64,
    callback_disposition: String,
    callback_event_sequence: Option<u64>,
    callback_observed_epoch: Option<u64>,
    callback_admitted_frames: u64,
    stale_callback_discarded_frames: u64,
    invalidated_application_frames: u64,
    discarded_unconsumed_notes: u64,
    post_veto_new_admissions: u64,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    source_kind: String,
    capture_id: String,
    payload_sha256: String,
    recorder_identity: String,
    target_window: TimeWindow,
    reported_closure: String,
    sequence_accounting: SequenceAccounting,
    frame_accounting: FrameAccounting,
    heartbeat_accounting: HeartbeatAccounting,
    terminal_watermark: u64,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SequenceAccounting {
    expected: u64,
    observed: u64,
    missing: u64,
    duplicate: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameAccounting {
    expected: u64,
    observed: u64,
    dropped: u64,
    xrun_count: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeartbeatAccounting {
    expected: u64,
    observed: u64,
    missing: u64,
    duplicate: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    detector_result: String,
    capture_id: String,
    capture_payload_sha256: String,
    observer_identity: String,
    detector_build_sha256: String,
    detector_config_sha256: String,
    qualification_status: String,
    qualification_receipt: Option<ReceiptRef>,
    event_manifest_sha256: Option<String>,
    receipt: ReceiptRef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptRef {
    receipt_sha256: String,
    verifier_profile_id: String,
    verification_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    episode_verifier_profile: String,
    manifest_version: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ActionDisposition {
    AuthorizedForBoundForecast,
    AcceptedButNoOutputDecision,
    Vetoed,
    Ignored,
    BranchedWithoutActiveMutation,
    AwaitingSuccessorForecast,
    NeedsClarification,
    Abstained,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ObservationOutcome {
    DetectorReportedPresent,
    DetectorReportedAbsent,
    Ambiguous,
    NotEstimable,
    Unobserved,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VetoRaceClassification {
    QueuedOutputWithoutCallbackObservation,
    ApplicationOutputStaleDiscardedBeforeCallback,
    CallbackAdmittedBeforeVetoResidualPossible,
    PostVetoPlaybackPredictionFromPreVetoCallbackAdmission,
    PostVetoLoopbackObservedFromPreVetoCallbackAdmission,
}

#[derive(Debug, Clone, Serialize)]
pub struct EpisodeReport {
    pub schema: String,
    pub episode_id: String,
    pub session_id: String,
    pub turn_id: String,
    pub forecast_id: String,
    pub forecast_revision: u64,
    pub action_disposition: ActionDisposition,
    pub decision_action: String,
    pub effect_stage_reported: Option<String>,
    pub effect_capture_binding: bool,
    pub veto_race_classification: Option<VetoRaceClassification>,
    pub veto_callback_admitted_frames: Option<u64>,
    pub veto_stale_callback_discarded_frames: Option<u64>,
    pub veto_invalidated_application_frames: Option<u64>,
    pub veto_discarded_unconsumed_notes: Option<u64>,
    pub veto_post_boundary_new_admissions: Option<u64>,
    pub capture_source_kind: String,
    pub capture_sequence_accounting_consistent: bool,
    pub capture_frame_accounting_applicable: bool,
    pub capture_frame_accounting_consistent: bool,
    pub capture_heartbeat_accounting_applicable: bool,
    pub capture_heartbeat_accounting_consistent: bool,
    pub capture_accounting_complete: bool,
    pub observation_structurally_eligible: bool,
    pub observation_outcome: ObservationOutcome,
    pub independent_identity_separation_structural: bool,
    pub upstream_receipts_authenticated: bool,
    pub upstream_verifiers_replayed: bool,
    pub findings: Vec<String>,
    pub claim_ceiling: Vec<String>,
}

/// Parse and reduce one episode. This only validates the manifest and structural
/// joins. Receipt references remain untrusted until the owning verifiers have been
/// independently replayed by the caller/integration layer.
pub fn parse_and_reduce(input: &str) -> Result<EpisodeReport, String> {
    let episode: Episode = serde_json::from_str(input)
        .map_err(|error| format!("invalid closed-loop episode JSON: {error}"))?;
    reduce_episode(&episode)
}

fn reduce_episode(episode: &Episode) -> Result<EpisodeReport, String> {
    let mut findings = Vec::new();

    if episode.schema != SCHEMA {
        return Err(format!("unsupported episode schema: {}", episode.schema));
    }
    require_text(&episode.episode_id, "episode_id")?;
    require_text(&episode.session.session_id, "session.session_id")?;
    require_text(&episode.session.turn_id, "session.turn_id")?;
    require_text(
        &episode.execution_identity.policy_id,
        "execution_identity.policy_id",
    )?;
    require_text(
        &episode.execution_identity.policy_version,
        "execution_identity.policy_version",
    )?;
    require_text(
        &episode.execution_identity.backend_id,
        "execution_identity.backend_id",
    )?;
    require_text(
        &episode.provenance.episode_verifier_profile,
        "provenance.episode_verifier_profile",
    )?;
    if episode.provenance.manifest_version != 1 {
        return Err("unsupported episode manifest_version".into());
    }

    validate_identity(&episode.execution_identity)?;
    validate_window(&episode.session.target_window, "session.target_window")?;
    validate_window(&episode.forecast.target_window, "forecast.target_window")?;
    validate_window(&episode.capture.target_window, "capture.target_window")?;
    validate_receipt(&episode.forecast.receipt, "forecast.receipt")?;
    validate_receipt(&episode.disposition.receipt, "disposition.receipt")?;
    validate_receipt(&episode.decision.receipt, "decision.receipt")?;
    validate_receipt(&episode.capture.receipt, "capture.receipt")?;
    validate_receipt(&episode.observation.receipt, "observation.receipt")?;
    require_text(
        &episode.disposition.actor_identity,
        "disposition.actor_identity",
    )?;
    validate_digest(
        &episode.disposition.scope_sha256,
        "disposition.scope_sha256",
    )?;
    validate_digest(&episode.capture.payload_sha256, "capture.payload_sha256")?;
    validate_digest(
        &episode.observation.capture_payload_sha256,
        "observation.capture_payload_sha256",
    )?;
    validate_digest(
        &episode.observation.detector_build_sha256,
        "observation.detector_build_sha256",
    )?;
    validate_digest(
        &episode.observation.detector_config_sha256,
        "observation.detector_config_sha256",
    )?;
    if let Some(digest) = &episode.observation.event_manifest_sha256 {
        validate_digest(digest, "observation.event_manifest_sha256")?;
    }
    if let Some(receipt) = &episode.observation.qualification_receipt {
        validate_receipt(receipt, "observation.qualification_receipt")?;
    }

    if !same_window(
        &episode.session.target_window,
        &episode.forecast.target_window,
    ) {
        return Err("forecast target window does not equal the session target window".into());
    }
    if !same_window(
        &episode.session.target_window,
        &episode.capture.target_window,
    ) {
        findings
            .push("capture target window does not exactly cover the session target window".into());
    }
    if !matches!(
        episode.forecast.uncertainty.as_str(),
        "calibrated" | "uncalibrated" | "unavailable"
    ) {
        return Err("unknown forecast uncertainty state".into());
    }
    if episode.forecast.input_through_sequence != episode.session.input_sequence {
        return Err(
            "forecast input_through_sequence does not match the frozen session input sequence"
                .into(),
        );
    }
    if episode.forecast.committed_sequence <= episode.forecast.input_through_sequence {
        return Err("forecast was not committed after the input sequence it claims to use".into());
    }
    if episode.disposition.forecast_id != episode.forecast.forecast_id
        || episode.disposition.forecast_revision != episode.forecast.revision
    {
        return Err(
            "musician disposition is not bound to the exact forecast ID and revision".into(),
        );
    }
    if episode.disposition.sequence <= episode.forecast.committed_sequence {
        return Err("musician disposition precedes or equals forecast commitment sequence".into());
    }
    if episode.disposition.sequence > episode.forecast.expires_at_sequence {
        return Err("musician disposition references an expired forecast".into());
    }
    if episode.decision.sequence <= episode.disposition.sequence {
        return Err("turn-taking decision does not follow the musician disposition".into());
    }
    if episode.decision.observed_through_input_sequence != episode.session.input_sequence {
        return Err(
            "turn-taking decision input watermark differs from the frozen session input".into(),
        );
    }
    if episode.decision.forecast_id.as_deref() != Some(episode.forecast.forecast_id.as_str())
        || episode.decision.forecast_revision != Some(episode.forecast.revision)
    {
        return Err("turn-taking decision does not bind the exact forecast ID and revision".into());
    }

    if episode.decision.sequence > episode.forecast.expires_at_sequence {
        return Err("turn-taking decision occurred after forecast expiry".into());
    }
    let veto_race_classification = validate_veto_race_for_episode(episode)?;
    let action_disposition = validate_action_semantics(episode)?;
    let effect_stage_reported = episode
        .effect
        .as_ref()
        .map(|effect| effect.reported_stage.clone());
    let veto_race = episode
        .effect
        .as_ref()
        .and_then(|effect| effect.veto_race.as_ref());

    if let Some(effect) = &episode.effect {
        validate_receipt(&effect.receipt, "effect.receipt")?;
        if effect.reported_stage == "outcome_unknown" {
            findings.push("effect outcome is unknown; reconcile the exact submission before retrying or mutating again".into());
        }
        require_text(&effect.producer_identity, "effect.producer_identity")?;
        if let Some(digest) = &effect.output_artifact_sha256 {
            validate_digest(digest, "effect.output_artifact_sha256")?;
        }
        if effect.decision_sequence != episode.decision.sequence {
            return Err(
                "effect decision_sequence does not bind the exact turn-taking decision".into(),
            );
        }
        if effect.forecast_id.as_deref() != Some(episode.forecast.forecast_id.as_str())
            || effect.forecast_revision != Some(episode.forecast.revision)
        {
            return Err("effect does not bind the exact forecast ID and revision".into());
        }
        validate_effect_stage(&effect.reported_stage)?;
        if is_output_stage(&effect.reported_stage) && effect.output_artifact_sha256.is_none() {
            return Err("output effect stage is missing its candidate artifact digest".into());
        }
    }

    let effect_capture_binding = match &episode.effect {
        Some(effect) => {
            if let Some(capture_id) = &effect.capture_id {
                if capture_id != &episode.capture.capture_id {
                    return Err("effect capture_id does not match the independent capture".into());
                }
                true
            } else {
                if effect.reported_stage == "loopback_observed" {
                    return Err("loopback_observed effect is missing its capture_id binding".into());
                }
                false
            }
        }
        None => false,
    };

    if !matches!(
        episode.capture.source_kind.as_str(),
        "rendered_audio_loopback" | "midi_output_monitor"
    ) {
        return Err("unknown capture source_kind".into());
    }
    if !matches!(
        episode.capture.reported_closure.as_str(),
        "eligible" | "ineligible" | "unknown"
    ) {
        return Err("unknown capture reported_closure state".into());
    }
    if !matches!(
        episode.observation.qualification_status.as_str(),
        "qualified" | "unqualified" | "unknown"
    ) {
        return Err("unknown detector qualification state".into());
    }
    let capture_sequence_accounting_consistent = episode
        .capture
        .sequence_accounting
        .observed
        .checked_add(episode.capture.sequence_accounting.missing)
        == Some(episode.capture.sequence_accounting.expected)
        && episode.capture.sequence_accounting.duplicate == 0;
    let capture_frame_accounting_applicable =
        episode.capture.source_kind == "rendered_audio_loopback";
    let capture_frame_accounting_consistent = episode
        .capture
        .frame_accounting
        .observed
        .checked_add(episode.capture.frame_accounting.dropped)
        == Some(episode.capture.frame_accounting.expected)
        && episode.capture.frame_accounting.dropped == 0
        && episode.capture.frame_accounting.xrun_count == 0;
    let capture_heartbeat_accounting_applicable =
        episode.capture.source_kind == "midi_output_monitor";
    let capture_heartbeat_accounting_consistent = episode
        .capture
        .heartbeat_accounting
        .observed
        .checked_add(episode.capture.heartbeat_accounting.missing)
        == Some(episode.capture.heartbeat_accounting.expected)
        && episode.capture.heartbeat_accounting.duplicate == 0;
    let source_accounting_complete = match episode.capture.source_kind.as_str() {
        "rendered_audio_loopback" => {
            episode.capture.frame_accounting.expected > 0
                && episode.capture.frame_accounting.observed
                    == episode.capture.frame_accounting.expected
                && capture_frame_accounting_consistent
                && episode.capture.heartbeat_accounting.expected == 0
                && episode.capture.heartbeat_accounting.observed == 0
                && episode.capture.heartbeat_accounting.missing == 0
                && episode.capture.heartbeat_accounting.duplicate == 0
        }
        "midi_output_monitor" => {
            episode.capture.frame_accounting.expected == 0
                && episode.capture.frame_accounting.observed == 0
                && episode.capture.frame_accounting.dropped == 0
                && episode.capture.frame_accounting.xrun_count == 0
                && episode.capture.heartbeat_accounting.expected > 0
                && episode.capture.heartbeat_accounting.observed
                    == episode.capture.heartbeat_accounting.expected
                && capture_heartbeat_accounting_consistent
        }
        _ => false,
    };
    let capture_accounting_complete = episode.capture.reported_closure == "eligible"
        && same_window(
            &episode.session.target_window,
            &episode.capture.target_window,
        )
        && episode.capture.sequence_accounting.expected > 0
        && episode.capture.sequence_accounting.observed
            == episode.capture.sequence_accounting.expected
        && capture_sequence_accounting_consistent
        && source_accounting_complete
        && episode.capture.terminal_watermark > 0;

    if !capture_sequence_accounting_consistent {
        findings.push("capture sequence accounting is inconsistent or contains duplicates".into());
    }
    if capture_frame_accounting_applicable && !capture_frame_accounting_consistent {
        findings.push(
            "audio capture frame accounting is incomplete, dropped, or contains XRUNs".into(),
        );
    }
    if capture_heartbeat_accounting_applicable && !capture_heartbeat_accounting_consistent {
        findings
            .push("MIDI monitor heartbeat accounting is incomplete or contains duplicates".into());
    }
    if !capture_accounting_complete {
        findings.push("capture is not structurally eligible for negative-event scoring".into());
    }

    if episode.observation.capture_id != episode.capture.capture_id {
        return Err("event observation references a different capture_id".into());
    }
    if episode.observation.capture_payload_sha256 != episode.capture.payload_sha256 {
        return Err(
            "event observation payload digest does not match the exact capture payload".into(),
        );
    }

    require_text(
        &episode.capture.recorder_identity,
        "capture.recorder_identity",
    )?;
    require_text(
        &episode.observation.observer_identity,
        "observation.observer_identity",
    )?;
    let mut identities = BTreeSet::new();
    identities.insert(episode.capture.recorder_identity.as_str());
    identities.insert(episode.observation.observer_identity.as_str());
    let mut independent_identity_separation_structural = identities.len() == 2;
    if let Some(effect) = &episode.effect {
        identities.insert(effect.producer_identity.as_str());
        independent_identity_separation_structural &= identities.len() == 3;
    }
    if !independent_identity_separation_structural {
        findings.push("producer, capture recorder and event observer identities are not structurally distinct".into());
    }

    let detector_qualified_shape = episode.observation.qualification_status == "qualified"
        && episode.observation.qualification_receipt.is_some()
        && episode.observation.event_manifest_sha256.is_some();

    let observation_structurally_eligible = capture_accounting_complete
        && independent_identity_separation_structural
        && detector_qualified_shape;

    if !detector_qualified_shape {
        findings
            .push("detector qualification or event-manifest binding is missing/unqualified".into());
    }

    let observation_outcome = match episode.observation.detector_result.as_str() {
        "present" if observation_structurally_eligible => {
            ObservationOutcome::DetectorReportedPresent
        }
        "absent" if observation_structurally_eligible => ObservationOutcome::DetectorReportedAbsent,
        "ambiguous" if observation_structurally_eligible => ObservationOutcome::Ambiguous,
        "not_estimable" if observation_structurally_eligible => ObservationOutcome::NotEstimable,
        "present" | "absent" | "ambiguous" | "not_estimable" | "unobserved" => {
            ObservationOutcome::Unobserved
        }
        other => return Err(format!("unknown detector_result: {other}")),
    };

    if matches!(
        episode.observation.detector_result.as_str(),
        "present" | "absent"
    ) && observation_outcome == ObservationOutcome::Unobserved
    {
        findings.push("detector present/absent claim was downgraded to unobserved because eligibility did not close".into());
    }

    Ok(EpisodeReport {
        schema: SCHEMA.into(),
        episode_id: episode.episode_id.clone(),
        session_id: episode.session.session_id.clone(),
        turn_id: episode.session.turn_id.clone(),
        forecast_id: episode.forecast.forecast_id.clone(),
        forecast_revision: episode.forecast.revision,
        action_disposition,
        decision_action: episode.decision.action.clone(),
        effect_stage_reported,
        effect_capture_binding,
        veto_race_classification,
        veto_callback_admitted_frames: veto_race.map(|race| race.callback_admitted_frames),
        veto_stale_callback_discarded_frames: veto_race.map(|race| race.stale_callback_discarded_frames),
        veto_invalidated_application_frames: veto_race.map(|race| race.invalidated_application_frames),
        veto_discarded_unconsumed_notes: veto_race.map(|race| race.discarded_unconsumed_notes),
        veto_post_boundary_new_admissions: veto_race.map(|race| race.post_veto_new_admissions),
        capture_source_kind: episode.capture.source_kind.clone(),
        capture_sequence_accounting_consistent,
        capture_frame_accounting_applicable,
        capture_frame_accounting_consistent,
        capture_heartbeat_accounting_applicable,
        capture_heartbeat_accounting_consistent,
        capture_accounting_complete,
        observation_structurally_eligible,
        observation_outcome,
        independent_identity_separation_structural,
        upstream_receipts_authenticated: false,
        upstream_verifiers_replayed: false,
        findings,
        claim_ceiling: vec![
            "synthetic or supplied receipt references are not authenticated by this reducer".into(),
            "upstream verifier outputs are not replayed by this reducer; an integration caller must replay them independently".into(),
            "reported closure plus arithmetic consistency is structural eligibility, not proof of physical capture completeness".into(),
            "detector-reported absence is not ground-truth absence; held-out detector sensitivity and false-negative qualification remain separate".into(),
            "identity strings are compared structurally; authenticated principal/role separation is not established here".into(),
            "late-veto admission/callback/epoch counts are untrusted structural projections; the owning action-effect verifier and veto-race verifier are not replayed here".into(),
            "a pre-veto callback admission may remain in flight after veto; epoch invalidation is not proof of physical silence or DAC cancellation".into(),
            "no waveform decoding, acoustic latency, hardware, musician preference, causal collaboration or SOTA result is established".into(),
        ],
    })
}

fn validate_action_semantics(episode: &Episode) -> Result<ActionDisposition, String> {
    let action = episode.disposition.action.as_str();
    let decision = episode.decision.action.as_str();
    let effect_is_output = episode
        .effect
        .as_ref()
        .is_some_and(|effect| is_output_stage(&effect.reported_stage));

    match action {
        "accept" => {
            if !matches!(decision, "RESPOND" | "CONTINUE" | "WAIT" | "YIELD") {
                return Err(format!("unsupported turn decision for accept: {decision}"));
            }
            if matches!(decision, "WAIT" | "YIELD") && effect_is_output {
                return Err(
                    "output effect is incompatible with WAIT/YIELD following accept".into(),
                );
            }
            if matches!(decision, "RESPOND" | "CONTINUE") {
                Ok(ActionDisposition::AuthorizedForBoundForecast)
            } else {
                Ok(ActionDisposition::AcceptedButNoOutputDecision)
            }
        }
        "veto" => {
            if decision != "YIELD" {
                return Err(
                    "veto must dominate with YIELD; later RESPOND/CONTINUE is unauthorized".into(),
                );
            }
            Ok(ActionDisposition::Vetoed)
        }
        "ignore" => {
            if !matches!(decision, "WAIT" | "YIELD") || effect_is_output {
                return Err("ignore cannot authorize RESPOND/CONTINUE or an output effect".into());
            }
            Ok(ActionDisposition::Ignored)
        }
        "branch" => {
            if !matches!(decision, "WAIT" | "YIELD") || effect_is_output {
                return Err("branch cannot mutate the active forecast without a separate successor transaction".into());
            }
            Ok(ActionDisposition::BranchedWithoutActiveMutation)
        }
        "steer" => {
            if episode
                .disposition
                .superseding_forecast_id
                .as_deref()
                .is_none_or(str::is_empty)
            {
                return Err("steer requires an explicit successor forecast ID".into());
            }
            if !matches!(decision, "WAIT" | "YIELD") || effect_is_output {
                return Err(
                    "steer cannot execute the superseded forecast; wait for successor forecast"
                        .into(),
                );
            }
            Ok(ActionDisposition::AwaitingSuccessorForecast)
        }
        "clarify" => {
            if !matches!(decision, "WAIT" | "YIELD") || effect_is_output {
                return Err("clarify cannot authorize output before ambiguity is resolved".into());
            }
            Ok(ActionDisposition::NeedsClarification)
        }
        "abstain" => {
            if !matches!(decision, "WAIT" | "YIELD") || effect_is_output {
                return Err("abstain cannot authorize output".into());
            }
            Ok(ActionDisposition::Abstained)
        }
        other => Err(format!("unknown musician disposition action: {other}")),
    }
}

fn validate_veto_race_for_episode(
    episode: &Episode,
) -> Result<Option<VetoRaceClassification>, String> {
    let race = episode
        .effect
        .as_ref()
        .and_then(|effect| effect.veto_race.as_ref());
    if episode.disposition.action != "veto" {
        if race.is_some() {
            return Err("veto_race evidence is only valid for a veto disposition".into());
        }
        return Ok(None);
    }

    let Some(effect) = episode.effect.as_ref() else {
        return Ok(None);
    };
    if !is_output_stage(&effect.reported_stage) {
        if race.is_some() {
            return Err("veto_race evidence is inconsistent with a non-output effect stage".into());
        }
        return Ok(None);
    }
    let race = race.ok_or_else(|| {
        "veto with a reported output effect requires stage-aware veto_race evidence".to_string()
    })?;
    validate_veto_race(effect, race)?;

    let classification = match race.callback_disposition.as_str() {
        "not_observed" => VetoRaceClassification::QueuedOutputWithoutCallbackObservation,
        "stale_discarded_after_veto" => {
            VetoRaceClassification::ApplicationOutputStaleDiscardedBeforeCallback
        }
        "admitted_before_veto" => match effect.reported_stage.as_str() {
            "loopback_observed" => {
                VetoRaceClassification::PostVetoLoopbackObservedFromPreVetoCallbackAdmission
            }
            "cpal_predicted" => {
                VetoRaceClassification::PostVetoPlaybackPredictionFromPreVetoCallbackAdmission
            }
            _ => VetoRaceClassification::CallbackAdmittedBeforeVetoResidualPossible,
        },
        _ => return Err("unknown callback disposition in veto_race evidence".into()),
    };
    Ok(Some(classification))
}

fn validate_veto_race(effect: &Effect, race: &VetoRaceEvidence) -> Result<(), String> {
    require_text(&race.authorization_id, "effect.veto_race.authorization_id")?;
    validate_receipt(&race.receipt, "effect.veto_race.receipt")?;

    if !matches!(
        race.admission_stage.as_str(),
        "queue_applied" | "application_enqueued"
    ) {
        return Err("veto_race admission_stage must be queue_applied or application_enqueued".into());
    }
    if effect_stage_rank(&effect.reported_stage).is_none()
        || effect_stage_rank(&race.admission_stage).is_none()
        || effect_stage_rank(&effect.reported_stage) < effect_stage_rank(&race.admission_stage)
    {
        return Err("veto_race reported stage precedes or contradicts its admission stage".into());
    }
    if race.admission_event_sequence == 0
        || race.admission_event_sequence >= race.veto_event_sequence
        || race.reported_stage_event_sequence == 0
    {
        return Err("veto_race event sequence is invalid or admission did not precede veto".into());
    }
    if race.reported_stage_event_sequence < race.admission_event_sequence {
        return Err("veto_race reported stage precedes output admission".into());
    }
    if race.veto_output_epoch != race.prior_output_epoch.checked_add(1).ok_or_else(|| {
        "veto_race prior output epoch is exhausted".to_string()
    })? {
        return Err("veto_race veto must advance the prior output epoch exactly once".into());
    }
    if race.post_veto_new_admissions != 0 {
        return Err("veto_race reports new output admission after the veto boundary".into());
    }

    match race.callback_disposition.as_str() {
        "not_observed" => {
            if race.callback_event_sequence.is_some()
                || race.callback_observed_epoch.is_some()
                || race.callback_admitted_frames != 0
                || race.stale_callback_discarded_frames != 0
            {
                return Err("unobserved callback disposition must not carry callback event, epoch or frame counts".into());
            }
            if !matches!(
                effect.reported_stage.as_str(),
                "queue_applied" | "application_enqueued"
            ) || race.reported_stage_event_sequence >= race.veto_event_sequence
            {
                return Err("unobserved callback is incompatible with the reported effect stage or event order".into());
            }
        }
        "admitted_before_veto" => {
            let callback_sequence = race.callback_event_sequence.ok_or_else(|| {
                "pre-veto callback admission is missing its event sequence".to_string()
            })?;
            let callback_epoch = race.callback_observed_epoch.ok_or_else(|| {
                "pre-veto callback admission is missing its observed output epoch".to_string()
            })?;
            if callback_sequence <= race.admission_event_sequence
                || callback_sequence >= race.veto_event_sequence
                || callback_epoch != race.prior_output_epoch
                || race.callback_admitted_frames == 0
                || race.stale_callback_discarded_frames != 0
            {
                return Err("callback admission is not proven to precede veto under the prior epoch".into());
            }
            if !matches!(
                effect.reported_stage.as_str(),
                "callback_observed" | "cpal_predicted" | "loopback_observed"
            ) {
                return Err("pre-veto callback admission is inconsistent with the reported effect stage".into());
            }
            if effect.reported_stage == "callback_observed"
                && race.reported_stage_event_sequence != callback_sequence
            {
                return Err("callback_observed stage sequence does not match the callback admission event".into());
            }
            if matches!(
                effect.reported_stage.as_str(),
                "cpal_predicted" | "loopback_observed"
            ) && race.reported_stage_event_sequence <= race.veto_event_sequence
            {
                return Err("post-veto playback/loopback observation must follow the veto event".into());
            }
        }
        "stale_discarded_after_veto" => {
            let callback_sequence = race.callback_event_sequence.ok_or_else(|| {
                "stale callback discard is missing its event sequence".to_string()
            })?;
            let callback_epoch = race.callback_observed_epoch.ok_or_else(|| {
                "stale callback discard is missing its observed output epoch".to_string()
            })?;
            if callback_sequence <= race.veto_event_sequence
                || callback_epoch < race.veto_output_epoch
                || race.callback_admitted_frames != 0
                || race.stale_callback_discarded_frames == 0
                || effect.reported_stage != "application_enqueued"
                || race.reported_stage_event_sequence >= race.veto_event_sequence
            {
                return Err("stale callback discard is inconsistent with the post-veto epoch fence".into());
            }
        }
        other => return Err(format!("unknown callback disposition in veto_race evidence: {other}")),
    }
    Ok(())
}

fn effect_stage_rank(stage: &str) -> Option<u8> {
    match stage {
        "queue_applied" => Some(1),
        "application_enqueued" => Some(2),
        "callback_observed" => Some(3),
        "cpal_predicted" => Some(4),
        "loopback_observed" => Some(5),
        _ => None,
    }
}

fn validate_effect_stage(stage: &str) -> Result<(), String> {
    match stage {
        "not_requested"
        | "authorized"
        | "queue_applied"
        | "application_enqueued"
        | "callback_observed"
        | "cpal_predicted"
        | "loopback_observed"
        | "not_applied"
        | "outcome_unknown" => Ok(()),
        other => Err(format!("unknown effect stage: {other}")),
    }
}

fn is_output_stage(stage: &str) -> bool {
    matches!(
        stage,
        "queue_applied"
            | "application_enqueued"
            | "callback_observed"
            | "cpal_predicted"
            | "loopback_observed"
    )
}

fn validate_identity(identity: &ExecutionIdentity) -> Result<(), String> {
    validate_digest(
        &identity.identity_sha256,
        "execution_identity.identity_sha256",
    )?;
    validate_digest(
        &identity.configuration_sha256,
        "execution_identity.configuration_sha256",
    )?;
    validate_digest(
        &identity.adapter_sha256,
        "execution_identity.adapter_sha256",
    )?;
    validate_digest(
        &identity.runtime_sha256,
        "execution_identity.runtime_sha256",
    )?;
    if let Some(digest) = &identity.model_artifact_sha256 {
        validate_digest(digest, "execution_identity.model_artifact_sha256")?;
    }
    if let Some(digest) = &identity.tokenizer_sha256 {
        validate_digest(digest, "execution_identity.tokenizer_sha256")?;
    }
    match identity.kind.as_str() {
        "synthetic_rule_fixture" => {}
        "production_model" if identity.model_artifact_sha256.is_some() => {}
        "production_model" => {
            return Err("production execution identity requires a model artifact digest".into());
        }
        other => return Err(format!("unknown execution identity kind: {other}")),
    }
    Ok(())
}

fn validate_window(window: &TimeWindow, label: &str) -> Result<(), String> {
    require_text(&window.clock_domain, &format!("{label}.clock_domain"))?;
    if window.end_ns <= window.start_ns {
        return Err(format!("{label} must have a positive half-open interval"));
    }
    Ok(())
}

fn same_window(left: &TimeWindow, right: &TimeWindow) -> bool {
    left.clock_domain == right.clock_domain
        && left.start_ns == right.start_ns
        && left.end_ns == right.end_ns
}

fn validate_receipt(receipt: &ReceiptRef, label: &str) -> Result<(), String> {
    validate_digest(&receipt.receipt_sha256, &format!("{label}.receipt_sha256"))?;
    validate_digest(
        &receipt.verification_receipt_sha256,
        &format!("{label}.verification_receipt_sha256"),
    )?;
    require_text(
        &receipt.verifier_profile_id,
        &format!("{label}.verifier_profile_id"),
    )
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be a lowercase 64-character SHA-256 digest"
        ));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRESENT: &str = include_str!(
        "../../../docs/research/fixtures/mel-closed-loop-episode-present.synthetic.json"
    );
    const ABSENT: &str = include_str!(
        "../../../docs/research/fixtures/mel-closed-loop-episode-absent.synthetic.json"
    );
    const UNOBSERVED: &str = include_str!(
        "../../../docs/research/fixtures/mel-closed-loop-episode-unobserved.synthetic.json"
    );
    const MIDI_ABSENT: &str = include_str!(
        "../../../docs/research/fixtures/mel-closed-loop-episode-midi-absent.synthetic.json"
    );

    #[test]
    fn complete_present_window_is_reported_present_not_ground_truth() {
        let report = parse_and_reduce(PRESENT).expect("valid fixture");
        assert_eq!(
            report.observation_outcome,
            ObservationOutcome::DetectorReportedPresent
        );
        assert!(report.capture_accounting_complete);
        assert!(report.observation_structurally_eligible);
        assert!(!report.upstream_receipts_authenticated);
    }

    #[test]
    fn complete_empty_window_can_be_absent_without_producer_effect_record() {
        let report = parse_and_reduce(ABSENT).expect("valid complete-empty fixture");
        assert_eq!(
            report.observation_outcome,
            ObservationOutcome::DetectorReportedAbsent
        );
        assert!(report.capture_accounting_complete);
        assert!(report.effect_stage_reported.is_none());
        assert_eq!(report.action_disposition, ActionDisposition::Ignored);
    }

    #[test]
    fn complete_empty_midi_monitor_window_is_absent_without_audio_frames_or_producer() {
        let report = parse_and_reduce(MIDI_ABSENT).expect("valid MIDI monitor fixture");
        assert_eq!(
            report.observation_outcome,
            ObservationOutcome::DetectorReportedAbsent
        );
        assert_eq!(report.capture_source_kind, "midi_output_monitor");
        assert!(report.capture_accounting_complete);
        assert!(!report.capture_frame_accounting_applicable);
        assert!(report.capture_heartbeat_accounting_applicable);
        assert!(report.capture_heartbeat_accounting_consistent);
        assert!(report.effect_stage_reported.is_none());
    }

    #[test]
    fn midi_heartbeat_gap_downgrades_absence_to_unobserved() {
        let mut value: serde_json::Value = serde_json::from_str(MIDI_ABSENT).expect("fixture JSON");
        value["capture"]["heartbeat_accounting"]["observed"] = serde_json::json!(3);
        value["capture"]["heartbeat_accounting"]["missing"] = serde_json::json!(1);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("heartbeat gap must downgrade observation");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
        assert!(!report.capture_accounting_complete);
    }

    #[test]
    fn audio_capture_cannot_claim_unexplained_midi_heartbeats() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["capture"]["heartbeat_accounting"]["expected"] = serde_json::json!(1);
        value["capture"]["heartbeat_accounting"]["observed"] = serde_json::json!(1);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("cross-modality accounting should downgrade");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
        assert!(!report.capture_accounting_complete);
    }

    #[test]
    fn incomplete_capture_fixture_is_unobserved() {
        let report = parse_and_reduce(UNOBSERVED).expect("valid incomplete fixture");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
        assert!(!report.capture_accounting_complete);
        assert!(!report.observation_structurally_eligible);
    }

    #[test]
    fn incomplete_capture_downgrades_absence_to_unobserved() {
        let mut value: serde_json::Value = serde_json::from_str(ABSENT).expect("fixture JSON");
        value["capture"]["reported_closure"] = serde_json::json!("ineligible");
        value["capture"]["sequence_accounting"]["observed"] = serde_json::json!(98);
        value["capture"]["sequence_accounting"]["missing"] = serde_json::json!(2);
        value["capture"]["frame_accounting"]["observed"] = serde_json::json!(47520);
        value["capture"]["frame_accounting"]["dropped"] = serde_json::json!(480);
        value["capture"]["frame_accounting"]["xrun_count"] = serde_json::json!(1);
        value["observation"]["detector_result"] = serde_json::json!("absent");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("should remain a report, not an absence");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
        assert!(!report.observation_structurally_eligible);
    }

    #[test]
    fn payload_substitution_is_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["observation"]["capture_payload_sha256"] =
            serde_json::json!("abababababababababababababababababababababababababababababababab");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn recorder_and_event_observer_must_be_distinct() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        let recorder_identity = value["capture"]["recorder_identity"].clone();
        value["observation"]["observer_identity"] = recorder_identity;
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("mismatch should downgrade to unobserved");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
        assert!(!report.independent_identity_separation_structural);
    }

    #[test]
    fn veto_cannot_be_followed_by_respond() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["disposition"]["action"] = serde_json::json!("veto");
        value["decision"]["action"] = serde_json::json!("RESPOND");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn stale_forecast_cannot_authorize_output() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["forecast"]["expires_at_sequence"] = serde_json::json!(42);
        value["disposition"]["sequence"] = serde_json::json!(42);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn steer_cannot_execute_the_superseded_forecast() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["disposition"]["action"] = serde_json::json!("steer");
        value["disposition"]["superseding_forecast_id"] = serde_json::json!("forecast-next");
        value["decision"]["action"] = serde_json::json!("WAIT");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn production_policy_requires_real_artifact_digest() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["execution_identity"]["kind"] = serde_json::json!("production_model");
        value["execution_identity"]["model_artifact_sha256"] = serde_json::Value::Null;
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn effect_must_bind_the_same_capture_when_claiming_loopback_observation() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["effect"]["reported_stage"] = serde_json::json!("loopback_observed");
        value["effect"]["capture_id"] = serde_json::Value::Null;
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn forecast_target_clock_must_match_capture_clock() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["capture"]["target_window"]["clock_domain"] = serde_json::json!("other-clock");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("clock mismatch must not become eligible");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
    }

    fn attach_valid_pre_veto_loopback_race(value: &mut serde_json::Value) {
        value["disposition"]["action"] = serde_json::json!("veto");
        value["decision"]["action"] = serde_json::json!("YIELD");
        value["effect"]["reported_stage"] = serde_json::json!("loopback_observed");
        value["effect"]["veto_race"] = serde_json::json!({
            "authorization_id": "authorization-fixture",
            "admission_stage": "application_enqueued",
            "admission_event_sequence": 44,
            "veto_event_sequence": 46,
            "reported_stage_event_sequence": 48,
            "prior_output_epoch": 7,
            "veto_output_epoch": 8,
            "callback_disposition": "admitted_before_veto",
            "callback_event_sequence": 45,
            "callback_observed_epoch": 7,
            "callback_admitted_frames": 240,
            "stale_callback_discarded_frames": 0,
            "invalidated_application_frames": 120,
            "discarded_unconsumed_notes": 3,
            "post_veto_new_admissions": 0,
            "receipt": {
                "receipt_sha256": "2121212121212121212121212121212121212121212121212121212121212121",
                "verifier_profile_id": "action-effect-veto-race-v1",
                "verification_receipt_sha256": "2222222222222222222222222222222222222222222222222222222222222222"
            }
        });
    }

    #[test]
    fn pre_veto_callback_can_be_observed_after_veto_without_false_unauthorized_admission() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        attach_valid_pre_veto_loopback_race(&mut value);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("pre-veto admission remains a valid residual");
        assert_eq!(
            report.veto_race_classification,
            Some(VetoRaceClassification::PostVetoLoopbackObservedFromPreVetoCallbackAdmission)
        );
        assert_eq!(report.veto_callback_admitted_frames, Some(240));
        assert_eq!(report.veto_post_boundary_new_admissions, Some(0));
        assert_eq!(report.action_disposition, ActionDisposition::Vetoed);
    }

    #[test]
    fn veto_output_without_owner_race_evidence_is_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["disposition"]["action"] = serde_json::json!("veto");
        value["decision"]["action"] = serde_json::json!("YIELD");
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn veto_race_rejects_callback_admission_after_veto() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        attach_valid_pre_veto_loopback_race(&mut value);
        value["effect"]["veto_race"]["callback_event_sequence"] = serde_json::json!(47);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn veto_race_rejects_epoch_regression_or_multiple_advancement() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        attach_valid_pre_veto_loopback_race(&mut value);
        value["effect"]["veto_race"]["veto_output_epoch"] = serde_json::json!(9);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn veto_race_rejects_any_new_admission_after_the_boundary() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        attach_valid_pre_veto_loopback_race(&mut value);
        value["effect"]["veto_race"]["post_veto_new_admissions"] = serde_json::json!(1);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        assert!(parse_and_reduce(&raw).is_err());
    }

    #[test]
    fn veto_race_distinguishes_stale_discard_from_pre_veto_callback_admission() {
        let mut value: serde_json::Value = serde_json::from_str(PRESENT).expect("fixture JSON");
        value["disposition"]["action"] = serde_json::json!("veto");
        value["decision"]["action"] = serde_json::json!("YIELD");
        value["effect"]["reported_stage"] = serde_json::json!("application_enqueued");
        value["effect"]["capture_id"] = serde_json::Value::Null;
        value["effect"]["veto_race"] = serde_json::json!({
            "authorization_id": "authorization-fixture",
            "admission_stage": "application_enqueued",
            "admission_event_sequence": 44,
            "veto_event_sequence": 46,
            "reported_stage_event_sequence": 44,
            "prior_output_epoch": 7,
            "veto_output_epoch": 8,
            "callback_disposition": "stale_discarded_after_veto",
            "callback_event_sequence": 47,
            "callback_observed_epoch": 8,
            "callback_admitted_frames": 0,
            "stale_callback_discarded_frames": 240,
            "invalidated_application_frames": 0,
            "discarded_unconsumed_notes": 3,
            "post_veto_new_admissions": 0,
            "receipt": {
                "receipt_sha256": "2323232323232323232323232323232323232323232323232323232323232323",
                "verifier_profile_id": "action-effect-veto-race-v1",
                "verification_receipt_sha256": "2424242424242424242424242424242424242424242424242424242424242424"
            }
        });
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report = parse_and_reduce(&raw).expect("old-epoch frames are reported as stale");
        assert_eq!(
            report.veto_race_classification,
            Some(VetoRaceClassification::ApplicationOutputStaleDiscardedBeforeCallback)
        );
        assert_eq!(report.veto_stale_callback_discarded_frames, Some(240));
        assert_eq!(report.veto_callback_admitted_frames, Some(0));
    }

    #[test]
    fn duplicate_and_missing_capture_sequences_never_qualify_absence() {
        let mut value: serde_json::Value = serde_json::from_str(ABSENT).expect("fixture JSON");
        value["capture"]["sequence_accounting"]["duplicate"] = serde_json::json!(1);
        let raw = serde_json::to_string(&value).expect("serialize fixture");
        let report =
            parse_and_reduce(&raw).expect("duplicate sequence is a downgraded observation");
        assert_eq!(report.observation_outcome, ObservationOutcome::Unobserved);
    }
}
