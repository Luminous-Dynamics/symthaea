use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

const EXECUTION_DOMAIN: &[u8] = b"symthaea.spine.000b.execution-receipt.v1\0";
const INTEGRATION_DOMAIN: &[u8] = b"symthaea.spine.000b.cycle-integration.v1\0";
const APPLICATION_DOMAIN: &[u8] = b"symthaea.spine.000b.state-application.v1\0";
const CYCLE_DOMAIN: &[u8] = b"symthaea.spine.000b.cycle-evidence.v1\0";
const GENESIS_DOMAIN: &[u8] = b"symthaea.spine.000b.evidence-genesis.v1\0";
const CHAIN_DOMAIN: &[u8] = b"symthaea.spine.000b.evidence-chain-link.v1\0";

#[derive(Clone, Copy)]
struct ProposalBits {
    confidence_delta_bits: u64,
    lr_modulation_bits: u64,
    exploration_delta_bits: u64,
    arousal_delta_bits: u32,
    valence_delta_bits: u32,
    flags: u32,
}

#[derive(Clone, Copy)]
struct IntegratedBits {
    proposal: ProposalBits,
    n_contributors: u32,
}

#[derive(Clone)]
struct Execution<'a> {
    cycle_number: u64,
    subsystem_identity: &'a str,
    source_path: &'a str,
    schedule_interval: u32,
    urgency: u8,
    eligible_to_run: bool,
    execution_outcome: u8,
    emitted: bool,
    admitted: bool,
    proposal: Option<ProposalBits>,
    overlap_refs: Vec<&'a str>,
}

#[derive(Clone)]
struct Influence<'a> {
    subsystem_identity: &'a str,
    integrated_without_subject: IntegratedBits,
    changed_channel_mask: u8,
    uniquely_contributed_flags: u32,
    integration_changed: bool,
}

#[derive(Clone)]
struct Integration<'a> {
    cycle_number: u64,
    admitted_count: u32,
    integrated_all: IntegratedBits,
    subjects: Vec<Influence<'a>>,
}

#[derive(Clone)]
enum CanonicalValue {
    F64Bits(u64),
    F32Bits(u32),
    U64(u64),
    U32(u32),
    Bool(bool),
    Digest32([u8; 32]),
}

#[derive(Clone)]
struct Application<'a> {
    cycle_number: u64,
    application_index: u32,
    destination_id: &'a str,
    source_tag: u8,
    source_flag: u32,
    applied: bool,
    state_change_status: u8,
    before: Option<CanonicalValue>,
    after: Option<CanonicalValue>,
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.is_ascii()
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-')
        })
}

fn valid_ref(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value.is_ascii()
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'_' | b'.' | b'/' | b':' | b'@' | b'+' | b'-')
        })
}

fn valid_path(value: &str) -> bool {
    if value.is_empty()
        || value.len() > 512
        || !value.is_ascii()
        || value.starts_with('/')
        || value.contains('\\')
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b'-'))
    {
        return false;
    }
    value
        .split('/')
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn push_string(out: &mut Vec<u8>, value: &str, kind: &str) -> Result<()> {
    let valid = match kind {
        "identity" | "destination" => valid_identity(value),
        "path" => valid_path(value),
        "ref" => valid_ref(value),
        _ => false,
    };
    if !valid {
        bail!("invalid canonical {kind}: {value:?}");
    }
    let len = u16::try_from(value.len()).context("canonical string length overflow")?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn proposal_bytes(value: ProposalBits, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.confidence_delta_bits.to_le_bytes());
    out.extend_from_slice(&value.lr_modulation_bits.to_le_bytes());
    out.extend_from_slice(&value.exploration_delta_bits.to_le_bytes());
    out.extend_from_slice(&value.arousal_delta_bits.to_le_bytes());
    out.extend_from_slice(&value.valence_delta_bits.to_le_bytes());
    out.extend_from_slice(&value.flags.to_le_bytes());
}

fn integrated_bytes(value: IntegratedBits, out: &mut Vec<u8>) {
    proposal_bytes(value.proposal, out);
    out.extend_from_slice(&value.n_contributors.to_le_bytes());
}

fn validate_execution(value: &Execution<'_>) -> Result<()> {
    if value.urgency > 2 {
        bail!("unknown urgency tag");
    }
    if value.execution_outcome > 5 {
        bail!("unknown execution outcome tag");
    }
    match value.execution_outcome {
        0 | 1 | 4 | 5 => {
            if value.emitted || value.admitted || value.proposal.is_some() {
                bail!("skipped/panic/failed execution truth table violated");
            }
        }
        2 => {
            if !value.emitted || value.admitted || value.proposal.is_none() {
                bail!("neutral execution truth table violated");
            }
        }
        3 => {
            if !value.emitted || value.proposal.is_none() {
                bail!("non-neutral execution truth table violated");
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn encode_execution(value: &Execution<'_>) -> Result<Vec<u8>> {
    validate_execution(value)?;
    let mut refs = value.overlap_refs.clone();
    refs.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    refs.dedup();
    if refs.len() != value.overlap_refs.len() {
        bail!("duplicate overlap ref");
    }
    let ref_count = u16::try_from(refs.len()).context("overlap ref count overflow")?;

    let mut out = Vec::new();
    out.extend_from_slice(&value.cycle_number.to_le_bytes());
    push_string(&mut out, value.subsystem_identity, "identity")?;
    push_string(&mut out, value.source_path, "path")?;
    out.extend_from_slice(&value.schedule_interval.to_le_bytes());
    out.push(value.urgency);
    push_bool(&mut out, value.eligible_to_run);
    out.push(value.execution_outcome);
    push_bool(&mut out, value.emitted);
    push_bool(&mut out, value.admitted);
    match value.proposal {
        None => out.push(0),
        Some(proposal) => {
            out.push(1);
            proposal_bytes(proposal, &mut out);
        }
    }
    out.extend_from_slice(&ref_count.to_le_bytes());
    for reference in refs {
        push_string(&mut out, reference, "ref")?;
    }
    Ok(out)
}

fn encode_integration(value: &Integration<'_>) -> Result<Vec<u8>> {
    let mut subjects = value.subjects.clone();
    subjects.sort_unstable_by(|a, b| a.subsystem_identity.as_bytes().cmp(b.subsystem_identity.as_bytes()));
    for pair in subjects.windows(2) {
        if pair[0].subsystem_identity == pair[1].subsystem_identity {
            bail!("duplicate integration subject identity");
        }
    }
    if value.admitted_count as usize != subjects.len() {
        bail!("admitted_count must equal subject_count");
    }

    let mut out = Vec::new();
    out.extend_from_slice(&value.cycle_number.to_le_bytes());
    out.extend_from_slice(&value.admitted_count.to_le_bytes());
    integrated_bytes(value.integrated_all, &mut out);
    out.extend_from_slice(&u32::try_from(subjects.len())?.to_le_bytes());
    for subject in subjects {
        if subject.changed_channel_mask & 0b1110_0000 != 0 {
            bail!("changed-channel reserved bits set");
        }
        if subject.integration_changed
            != (subject.changed_channel_mask != 0 || subject.uniquely_contributed_flags != 0)
        {
            bail!("integration_changed inconsistent with influence fields");
        }
        push_string(&mut out, subject.subsystem_identity, "identity")?;
        integrated_bytes(subject.integrated_without_subject, &mut out);
        out.push(subject.changed_channel_mask);
        out.extend_from_slice(&subject.uniquely_contributed_flags.to_le_bytes());
        push_bool(&mut out, subject.integration_changed);
    }
    Ok(out)
}

fn encode_value(value: &CanonicalValue, out: &mut Vec<u8>) {
    match value {
        CanonicalValue::F64Bits(bits) => {
            out.push(1);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        CanonicalValue::F32Bits(bits) => {
            out.push(2);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        CanonicalValue::U64(value) => {
            out.push(3);
            out.extend_from_slice(&value.to_le_bytes());
        }
        CanonicalValue::U32(value) => {
            out.push(4);
            out.extend_from_slice(&value.to_le_bytes());
        }
        CanonicalValue::Bool(value) => {
            out.push(5);
            push_bool(out, *value);
        }
        CanonicalValue::Digest32(value) => {
            out.push(6);
            out.extend_from_slice(value);
        }
    }
}

fn encode_optional_value(value: &Option<CanonicalValue>, out: &mut Vec<u8>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            encode_value(value, out);
        }
    }
}

fn value_kind(value: &CanonicalValue) -> u8 {
    match value {
        CanonicalValue::F64Bits(_) => 1,
        CanonicalValue::F32Bits(_) => 2,
        CanonicalValue::U64(_) => 3,
        CanonicalValue::U32(_) => 4,
        CanonicalValue::Bool(_) => 5,
        CanonicalValue::Digest32(_) => 6,
    }
}

fn encode_application(value: &Application<'_>) -> Result<Vec<u8>> {
    if value.source_tag > 5 {
        bail!("unknown application source tag");
    }
    if value.source_tag != 5 && value.source_flag != 0 {
        bail!("scalar application must have source_flag=0");
    }
    if value.state_change_status > 2 {
        bail!("unknown state-change tag");
    }
    if value.state_change_status <= 1 {
        match (&value.before, &value.after) {
            (Some(before), Some(after)) if value_kind(before) == value_kind(after) => {}
            _ => bail!("observed state change requires compatible before/after"),
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&value.cycle_number.to_le_bytes());
    out.extend_from_slice(&value.application_index.to_le_bytes());
    push_string(&mut out, value.destination_id, "destination")?;
    out.push(value.source_tag);
    out.extend_from_slice(&value.source_flag.to_le_bytes());
    push_bool(&mut out, value.applied);
    out.push(value.state_change_status);
    encode_optional_value(&value.before, &mut out);
    encode_optional_value(&value.after, &mut out);
    Ok(out)
}

fn sha256(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn execution_digest(value: &Execution<'_>) -> Result<[u8; 32]> {
    Ok(sha256(EXECUTION_DOMAIN, &encode_execution(value)?))
}

fn integration_digest(value: &Integration<'_>) -> Result<[u8; 32]> {
    Ok(sha256(INTEGRATION_DOMAIN, &encode_integration(value)?))
}

fn application_digest(value: &Application<'_>) -> Result<[u8; 32]> {
    Ok(sha256(APPLICATION_DOMAIN, &encode_application(value)?))
}

fn encode_cycle(
    cycle_number: u64,
    executions: &[Execution<'_>],
    integration: &Integration<'_>,
    applications: &[Application<'_>],
) -> Result<Vec<u8>> {
    if integration.cycle_number != cycle_number {
        bail!("integration/envelope cycle mismatch");
    }
    let mut executions = executions.to_vec();
    executions.sort_unstable_by(|a, b| a.subsystem_identity.as_bytes().cmp(b.subsystem_identity.as_bytes()));
    for pair in executions.windows(2) {
        if pair[0].subsystem_identity == pair[1].subsystem_identity {
            bail!("duplicate execution identity");
        }
    }
    if executions.iter().any(|r| r.cycle_number != cycle_number) {
        bail!("execution/envelope cycle mismatch");
    }

    let mut applications = applications.to_vec();
    applications.sort_unstable_by_key(|r| r.application_index);
    if applications.iter().any(|r| r.cycle_number != cycle_number) {
        bail!("application/envelope cycle mismatch");
    }
    for (index, application) in applications.iter().enumerate() {
        if application.application_index != u32::try_from(index)? {
            bail!("application indices must be contiguous from zero");
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&cycle_number.to_le_bytes());
    out.extend_from_slice(&u32::try_from(executions.len())?.to_le_bytes());
    for execution in &executions {
        push_string(&mut out, execution.subsystem_identity, "identity")?;
        out.extend_from_slice(&execution_digest(execution)?);
    }
    out.extend_from_slice(&integration_digest(integration)?);
    out.extend_from_slice(&u32::try_from(applications.len())?.to_le_bytes());
    for application in &applications {
        out.extend_from_slice(&application.application_index.to_le_bytes());
        out.extend_from_slice(&application_digest(application)?);
    }
    Ok(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode_32(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        bail!("digest hex must be 64 characters");
    }
    let mut out = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk)?;
        out[index] = u8::from_str_radix(text, 16)?;
    }
    Ok(out)
}

fn expected<'a>(vectors: &'a Value, key: &str) -> Result<&'a str> {
    vectors
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing vector field {key}"))
}

fn assert_hex(vectors: &Value, key: &str, bytes: &[u8]) -> Result<()> {
    let actual = hex_encode(bytes);
    let wanted = expected(vectors, key)?;
    if actual != wanted {
        bail!("{key} mismatch\nexpected={wanted}\nactual={actual}");
    }
    Ok(())
}

fn sample() -> (Execution<'static>, Execution<'static>, Integration<'static>, Application<'static>, Application<'static>) {
    let neutral = ProposalBits {
        confidence_delta_bits: 0.0f64.to_bits(),
        lr_modulation_bits: 1.0f64.to_bits(),
        exploration_delta_bits: 0.0f64.to_bits(),
        arousal_delta_bits: 0.0f32.to_bits(),
        valence_delta_bits: 0.0f32.to_bits(),
        flags: 0,
    };
    let proposal = ProposalBits {
        confidence_delta_bits: 0.25f64.to_bits(),
        flags: 1,
        ..neutral
    };
    let execution_a = Execution {
        cycle_number: 7,
        subsystem_identity: "manager_z",
        source_path: "src/cognitive_loop/managers/z.rs",
        schedule_interval: 7,
        urgency: 1,
        eligible_to_run: true,
        execution_outcome: 3,
        emitted: true,
        admitted: true,
        proposal: Some(proposal),
        overlap_refs: vec!["spine000c:manager_z"],
    };
    let execution_b = Execution {
        cycle_number: 7,
        subsystem_identity: "manager_a",
        source_path: "src/cognitive_loop/managers/a.rs",
        schedule_interval: 11,
        urgency: 1,
        eligible_to_run: true,
        execution_outcome: 1,
        emitted: false,
        admitted: false,
        proposal: None,
        overlap_refs: vec![],
    };
    let integration = Integration {
        cycle_number: 7,
        admitted_count: 1,
        integrated_all: IntegratedBits {
            proposal,
            n_contributors: 1,
        },
        subjects: vec![Influence {
            subsystem_identity: "manager_z",
            integrated_without_subject: IntegratedBits {
                proposal: neutral,
                n_contributors: 0,
            },
            changed_channel_mask: 1,
            uniquely_contributed_flags: 1,
            integration_changed: true,
        }],
    };
    let application_0 = Application {
        cycle_number: 7,
        application_index: 0,
        destination_id: "prediction_confidence",
        source_tag: 0,
        source_flag: 0,
        applied: true,
        state_change_status: 1,
        before: Some(CanonicalValue::F64Bits(0.5f64.to_bits())),
        after: Some(CanonicalValue::F64Bits(0.75f64.to_bits())),
    };
    let application_1 = Application {
        cycle_number: 7,
        application_index: 1,
        destination_id: "episodic_memory.consolidate_recent",
        source_tag: 5,
        source_flag: 2,
        applied: true,
        state_change_status: 2,
        before: None,
        after: None,
    };
    (execution_a, execution_b, integration, application_0, application_1)
}

fn main() -> Result<()> {
    let fixture = env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/spine_000b_receipt_commitment_v1_vectors.json".into());
    let vectors: Value = serde_json::from_str(
        &fs::read_to_string(Path::new(&fixture)).with_context(|| format!("read {fixture}"))?,
    )?;
    if vectors.get("authority").and_then(Value::as_str) != Some("measurement-only") {
        bail!("vector authority boundary changed");
    }

    let (ea, eb, integration, app0, app1) = sample();
    let ea_bytes = encode_execution(&ea)?;
    let eb_bytes = encode_execution(&eb)?;
    let integration_bytes = encode_integration(&integration)?;
    let app0_bytes = encode_application(&app0)?;
    let app1_bytes = encode_application(&app1)?;
    let cycle_bytes = encode_cycle(7, &[ea.clone(), eb.clone()], &integration, &[app0.clone(), app1.clone()])?;

    assert_hex(&vectors, "execution_a_bytes_hex", &ea_bytes)?;
    assert_hex(&vectors, "execution_b_bytes_hex", &eb_bytes)?;
    assert_hex(&vectors, "integration_bytes_hex", &integration_bytes)?;
    assert_hex(&vectors, "application_0_bytes_hex", &app0_bytes)?;
    assert_hex(&vectors, "application_1_bytes_hex", &app1_bytes)?;
    assert_hex(&vectors, "cycle_bytes_hex", &cycle_bytes)?;

    let ea_digest = sha256(EXECUTION_DOMAIN, &ea_bytes);
    let eb_digest = sha256(EXECUTION_DOMAIN, &eb_bytes);
    let integration_hash = sha256(INTEGRATION_DOMAIN, &integration_bytes);
    let app0_digest = sha256(APPLICATION_DOMAIN, &app0_bytes);
    let app1_digest = sha256(APPLICATION_DOMAIN, &app1_bytes);
    let cycle_digest = sha256(CYCLE_DOMAIN, &cycle_bytes);

    for (key, digest) in [
        ("execution_a_sha256", ea_digest),
        ("execution_b_sha256", eb_digest),
        ("integration_sha256", integration_hash),
        ("application_0_sha256", app0_digest),
        ("application_1_sha256", app1_digest),
        ("cycle_sha256", cycle_digest),
    ] {
        assert_hex(&vectors, key, &digest)?;
    }

    let subject_digest = hex_decode_32(expected(&vectors, "subject_manifest_sha256")?)?;
    let root0 = sha256(GENESIS_DOMAIN, &subject_digest);
    assert_hex(&vectors, "genesis_root_sha256", &root0)?;
    let mut link = Vec::with_capacity(64);
    link.extend_from_slice(&root0);
    link.extend_from_slice(&cycle_digest);
    let root1 = sha256(CHAIN_DOMAIN, &link);
    assert_hex(&vectors, "chain_root_after_cycle_sha256", &root1)?;

    // Input execution ordering is not semantic: canonical identity sort must agree.
    let reversed = encode_cycle(7, &[eb.clone(), ea.clone()], &integration, &[app0.clone(), app1.clone()])?;
    if reversed != cycle_bytes {
        bail!("execution input order changed canonical cycle bytes");
    }

    // Changing semantic application indices must change the cycle identity.
    let mut swapped0 = app0.clone();
    let mut swapped1 = app1.clone();
    swapped0.application_index = 1;
    swapped1.application_index = 0;
    let changed = encode_cycle(7, &[ea.clone(), eb.clone()], &integration, &[swapped0, swapped1])?;
    if changed == cycle_bytes {
        bail!("application index change failed to change canonical cycle bytes");
    }

    // Reject noncanonical paths and unknown outcome tags.
    let mut bad_path = ea.clone();
    bad_path.source_path = "../escape.rs";
    if encode_execution(&bad_path).is_ok() {
        bail!("noncanonical path was accepted");
    }
    let mut bad_outcome = ea;
    bad_outcome.execution_outcome = 255;
    if encode_execution(&bad_outcome).is_ok() {
        bail!("unknown outcome tag was accepted");
    }

    // Exercise all CanonicalValue wire tags independently of the golden sample.
    let values = [
        CanonicalValue::F64Bits(1),
        CanonicalValue::F32Bits(2),
        CanonicalValue::U64(3),
        CanonicalValue::U32(4),
        CanonicalValue::Bool(true),
        CanonicalValue::Digest32([6u8; 32]),
    ];
    let mut tag_bytes = Vec::new();
    for value in &values {
        encode_value(value, &mut tag_bytes);
    }
    if tag_bytes.is_empty() {
        bail!("canonical value tags were not encoded");
    }

    println!("SPINE-000B Rust receipt commitment vectors: PASS");
    println!("authority=measurement-only");
    println!("runtime_evidence_claimed=false");
    println!("causal_load_claimed=false");
    Ok(())
}
