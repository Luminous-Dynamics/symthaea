// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local participant-facing runner for one blinded cognition-study block.

use axum::extract::{Path as AxPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_muse::evidence_digest::sha256_hex;
use symthaea_muse::study_runner::{
    StudyRunnerPackage, StudySessionEvent, StudySessionLog, append_session_event, new_session_log,
    runner_package_commitment, validate_session_log,
};

struct RunnerState {
    package: StudyRunnerPackage,
    artifact_root: PathBuf,
    log_path: PathBuf,
    log: Mutex<StudySessionLog>,
}

#[derive(Debug, Deserialize)]
struct EventRequest {
    client_elapsed_ms: u64,
    event: StudySessionEvent,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    event_count: usize,
    finalized: bool,
    issues: Vec<symthaea_muse::study_runner::StudyRunnerIssue>,
    log_sha256: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let package_path = args.next().ok_or("missing PACKAGE.json")?;
    let artifact_root = args.next().ok_or("missing ARTIFACT_ROOT")?;
    let evidence_dir = args.next().ok_or("missing EVIDENCE_DIR")?;
    let bind = args.next().unwrap_or_else(|| "127.0.0.1:8420".into());
    if args.next().is_some() {
        return Err(
            "usage: cognitive_study_runner PACKAGE.json ARTIFACT_ROOT EVIDENCE_DIR [BIND]".into(),
        );
    }

    let runner_package: StudyRunnerPackage = read_json(&package_path)?;
    if runner_package_commitment(&runner_package)? != runner_package.package_sha256 {
        return Err("runner package commitment mismatch".into());
    }
    let artifact_root = PathBuf::from(artifact_root);
    for presentation in &runner_package.presentations {
        let relative = Path::new(&presentation.audio_relative_path);
        if !safe_relative_path(relative) {
            return Err(format!("unsafe audio path: {}", presentation.audio_relative_path).into());
        }
        let bytes = fs::read(artifact_root.join(relative))?;
        if sha256_hex(&bytes) != presentation.audio_sha256 {
            return Err(format!("audio digest mismatch: {}", presentation.presentation_id).into());
        }
    }
    let evidence_dir = PathBuf::from(evidence_dir);
    fs::create_dir_all(&evidence_dir)?;
    let log_path = evidence_dir.join(format!("{}.session.json", runner_package.block_id));
    let log = if log_path.exists() {
        let log: StudySessionLog = read_json(&log_path)?;
        let issues = validate_session_log(&runner_package, &log, false);
        if !issues.is_empty() {
            return Err(format!("existing session log is invalid: {issues:#?}").into());
        }
        log
    } else {
        let log = new_session_log(&runner_package);
        write_json_atomic(&log_path, &log)?;
        log
    };

    let state = Arc::new(RunnerState {
        package: runner_package,
        artifact_root,
        log_path,
        log: Mutex::new(log),
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/api/package", get(package))
        .route("/api/status", get(status))
        .route("/api/event", post(record_event))
        .route("/audio/{presentation_id}", get(audio))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("cognitive study runner: http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn package(State(state): State<Arc<RunnerState>>) -> Json<StudyRunnerPackage> {
    Json(state.package.clone())
}

async fn status(State(state): State<Arc<RunnerState>>) -> Json<StatusResponse> {
    let log = state.log.lock().expect("runner log mutex poisoned");
    let issues = validate_session_log(&state.package, &log, false);
    Json(StatusResponse {
        event_count: log.events.len(),
        finalized: log
            .events
            .iter()
            .any(|entry| matches!(&entry.event, StudySessionEvent::BlockFinalized)),
        issues,
        log_sha256: log.log_sha256.clone(),
    })
}

async fn record_event(
    State(state): State<Arc<RunnerState>>,
    Json(request): Json<EventRequest>,
) -> Result<Json<StudySessionLog>, (StatusCode, String)> {
    let server_received_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(internal_error)?
        .as_millis() as u64;
    let mut log = state.log.lock().map_err(internal_error)?;
    append_session_event(
        &state.package,
        &mut log,
        server_received_unix_ms,
        request.client_elapsed_ms,
        request.event,
    )
    .map_err(|issues| (StatusCode::BAD_REQUEST, format!("{issues:#?}")))?;
    write_json_atomic(&state.log_path, &*log).map_err(internal_error)?;
    Ok(Json(log.clone()))
}

async fn audio(
    State(state): State<Arc<RunnerState>>,
    AxPath(presentation_id): AxPath<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let presentation = state
        .package
        .presentations
        .iter()
        .find(|candidate| candidate.presentation_id == presentation_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown presentation".into()))?;
    let relative = Path::new(&presentation.audio_relative_path);
    if !safe_relative_path(relative) {
        return Err((StatusCode::BAD_REQUEST, "unsafe artifact path".into()));
    }
    let bytes = fs::read(state.artifact_root.join(relative)).map_err(internal_error)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let temp = path.with_extension("json.tmp");
    {
        let mut writer = BufWriter::new(File::create(&temp)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(temp, path)?;
    Ok(())
}

fn internal_error(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded music study</title><style>
body{font:16px system-ui;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.5}button,input{font:inherit}.card{border:1px solid #bbb;border-radius:10px;padding:18px;margin:16px 0}.hidden{display:none}audio{width:100%}.row{display:flex;gap:12px;flex-wrap:wrap}.error{color:#9b1c1c;white-space:pre-wrap}label{display:block;margin:12px 0}button{padding:10px 16px}.doc{white-space:pre-wrap;background:#f6f6f6;padding:14px;border-radius:8px;max-height:45vh;overflow:auto}
</style></head><body><h1>Blinded music study</h1><div id="app">Loading…</div><div id="error" class="error"></div>
<script>
const started=performance.now();let pkg,index=0,replays={},responses={};const app=document.querySelector('#app'),err=document.querySelector('#error');
const elapsed=()=>Math.round(performance.now()-started);async function send(event){err.textContent='';const r=await fetch('/api/event',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({client_elapsed_ms:elapsed(),event})});if(!r.ok){throw new Error(await r.text())}return r.json()}
async function boot(){pkg=await fetch('/api/package').then(r=>r.json());renderConsent()}
function renderConsent(){app.innerHTML='<div class=card><h2>Consent</h2><pre id=consentDoc class=doc></pre><label><input id=consentCheck type=checkbox> I have read the consent document and agree to participate.</label><button id=consent disabled>Continue to instructions</button></div>';document.querySelector('#consentDoc').textContent=pkg.protocol.consent_document_text;const check=document.querySelector('#consentCheck'),button=document.querySelector('#consent');check.onchange=()=>button.disabled=!check.checked;button.onclick=async()=>{try{await send({ConsentAccepted:{consent_document_sha256:pkg.protocol.consent_document_sha256}});renderInstructions()}catch(e){err.textContent=e}}}
function renderInstructions(){app.innerHTML='<div class=card><h2>Study instructions</h2><pre id=instructionsDoc class=doc></pre><label><input id=instructionsCheck type=checkbox> I have read and understand these instructions.</label><button id=instructions disabled>Begin listening</button></div>';document.querySelector('#instructionsDoc').textContent=pkg.protocol.instructions_text;const check=document.querySelector('#instructionsCheck'),button=document.querySelector('#instructions');check.onchange=()=>button.disabled=!check.checked;button.onclick=async()=>{try{await send({InstructionsAcknowledged:{instructions_sha256:pkg.protocol.instructions_sha256}});renderPresentation()}catch(e){err.textContent=e}}}
function renderPresentation(){if(index>=pkg.presentations.length){return renderRanking()}const p=pkg.presentations[index];const replay=replays[p.presentation_id]||0;app.innerHTML=`<div class=card><h2>Example ${index+1} of ${pkg.presentations.length}</h2><p>Code: <strong>${p.anonymous_code}</strong></p><audio id=audio controls preload=auto src="/audio/${encodeURIComponent(p.presentation_id)}"></audio><form id=form class=hidden><label><input name=recognized type=checkbox> I recognized the opening material when it returned.</label><label>Development instability (0–100)<input name=instability type=range min=0 max=100 value=50></label><label>How earned did the recapitulation feel? (0–100)<input name=earned type=range min=0 max=100 value=50></label>${pkg.protocol.require_attention_check?`<label>${pkg.protocol.attention_check_prompt}<select name=attention required><option value="">Choose…</option>${pkg.protocol.attention_check_options.map((option,i)=>`<option value="${i}">${option}</option>`).join('')}</select></label>`:''}<button>Record response</button></form></div>`;const a=document.querySelector('#audio'),f=document.querySelector('#form');let playStart=0,activeReplay=replay;a.onplay=async()=>{activeReplay=replays[p.presentation_id]||0;playStart=performance.now();try{await send({PlaybackStarted:{presentation_id:p.presentation_id,replay_index:activeReplay}});replays[p.presentation_id]=activeReplay+1}catch(e){a.pause();err.textContent=e}};a.onended=async()=>{const listened=Math.round(performance.now()-playStart);try{await send({PlaybackCompleted:{presentation_id:p.presentation_id,replay_index:activeReplay,listened_ms:listened,media_duration_ms:p.duration_ms}});f.classList.remove('hidden')}catch(e){err.textContent=e}};f.onsubmit=async ev=>{ev.preventDefault();const d=new FormData(f);try{await send({ResponseRecorded:{presentation_id:p.presentation_id,return_recognized:d.has('recognized'),development_instability:Number(d.get('instability'))/100,earned_recapitulation:Number(d.get('earned'))/100,attention_check_response:pkg.protocol.require_attention_check?Number(d.get('attention')):null,elapsed_ms:elapsed()}});responses[p.presentation_id]=true;index++;renderPresentation()}catch(e){err.textContent=e}}}
function renderRanking(){const opts=pkg.presentations.map(p=>`<option value="${p.presentation_id}">${p.anonymous_code}</option>`).join('');app.innerHTML=`<div class=card><h2>Rank the four examples</h2><p>Select each code exactly once, best to worst.</p><form id=rank>${[1,2,3,4].map(n=>`<label>Rank ${n}<select name=r${n}>${opts}</select></label>`).join('')}<button>Submit and finalize</button></form></div>`;document.querySelector('#rank').onsubmit=async ev=>{ev.preventDefault();const d=new FormData(ev.target),ids=[1,2,3,4].map(n=>d.get('r'+n));if(new Set(ids).size!==4){err.textContent='Each example must appear exactly once.';return}try{await send({RankingsSubmitted:{presentation_ids_best_to_worst:ids}});await send('BlockFinalized');app.innerHTML='<div class=card><h2>Complete</h2><p>Your sealed response block has been recorded. Thank you.</p></div>'}catch(e){err.textContent=e}}}
boot().catch(e=>err.textContent=e);
</script></body></html>"#;
