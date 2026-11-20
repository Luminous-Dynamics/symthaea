use muse_evidence_bundle::{verify_evidence_bundle_v1, EvidenceInputsV1, BundleStatusV1};
use serde_json::to_vec_pretty;
use std::{env, fs, path::{Path, PathBuf}, process};

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("read {}: {error}", path.display())),
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    let usage = "muse-evidence-bundle verify MANIFEST.json EVIDENCE_DIR\n\
        EVIDENCE_DIR may contain case.json, source-audio.wav, gain-envelope-contract.json,\n\
        symbolic-result.json, render-receipt.json, render-bundle/,\n\
        acoustic-result.json, score-audio-result.json, and human-acceptance.json.";
    if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
        println!("{usage}");
        return Ok(0);
    }
    if args.len() != 4 || args[1] != "verify" {
        println!("{usage}");
        return Ok(2);
    }
    let manifest_path = PathBuf::from(&args[2]);
    let evidence_dir = PathBuf::from(&args[3]);
    let manifest = fs::read(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let case = read_optional(&evidence_dir.join("case.json"))?;
    let source_audio = read_optional(&evidence_dir.join("source-audio.wav"))?;
    let gain_contract = read_optional(&evidence_dir.join("gain-envelope-contract.json"))?;
    let symbolic = read_optional(&evidence_dir.join("symbolic-result.json"))?;
    let render = read_optional(&evidence_dir.join("render-receipt.json"))?;
    let acoustic = read_optional(&evidence_dir.join("acoustic-result.json"))?;
    let score_audio = read_optional(&evidence_dir.join("score-audio-result.json"))?;
    let human = read_optional(&evidence_dir.join("human-acceptance.json"))?;
    let render_root = evidence_dir.join("render-bundle");
    let render_bundle_root = render_root.is_dir().then_some(render_root.as_path());

    let inputs = EvidenceInputsV1 {
        case_json: case.as_deref(),
        source_audio_wav: source_audio.as_deref(),
        gain_envelope_contract_json: gain_contract.as_deref(),
        symbolic_result_json: symbolic.as_deref(),
        render_receipt_json: render.as_deref(),
        render_bundle_root,
        acoustic_result_json: acoustic.as_deref(),
        score_audio_result_json: score_audio.as_deref(),
        human_acceptance_json: human.as_deref(),
    };
    let report = verify_evidence_bundle_v1(&manifest, inputs)?;
    let rendered = to_vec_pretty(&report)
        .map_err(|error| format!("serialize verification report: {error}"))?;
    println!("{}", String::from_utf8(rendered).map_err(|error| format!("report is not UTF-8: {error}"))?);
    match report.verified_status {
        BundleStatusV1::Pass => Ok(0),
        BundleStatusV1::Fail => Ok(2),
        BundleStatusV1::Abstain => Ok(3),
    }
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("muse-evidence-bundle: {error}");
            process::exit(4);
        }
    }
}
