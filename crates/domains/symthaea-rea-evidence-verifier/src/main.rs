use std::path::PathBuf;

use symthaea_rea_evidence_verifier::verify_bundle;

fn usage() -> &'static str {
    "Usage: symthaea-rea-evidence-verifier --artifact-dir DIR --source-root DIR --expected-head 40_HEX_SHA [--expected-receipt-sha256 SHA256]\n\
     \n\
     DIR is a downloaded Unblob provider-evidence artifact. SOURCE_ROOT is the checked-out\n\
     benchmarks/rea-determinism/v1 directory. EXPECTED_HEAD must come from an independently\n\
     trusted PR/release reference, not from receipt.json. An independent receipt digest enables\n\
     receipt-digest-anchored mode; without it, the result is integrity-consistency-only."
}

fn main() {
    if let Err(message) = run() {
        eprintln!("verification failed: {message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut artifact = None;
    let mut source = None;
    let mut head = None;
    let mut receipt_sha = None;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| usage().to_owned())?;
        match arg.as_str() {
            "--artifact-dir" => artifact = Some(PathBuf::from(value)),
            "--source-root" => source = Some(PathBuf::from(value)),
            "--expected-head" => head = Some(value),
            "--expected-receipt-sha256" => receipt_sha = Some(value),
            _ => return Err(usage().to_owned()),
        }
    }
    let artifact = artifact.ok_or_else(|| usage().to_owned())?;
    let source = source.ok_or_else(|| usage().to_owned())?;
    let head = head.ok_or_else(|| usage().to_owned())?;
    let verified = verify_bundle(&artifact, &source, &head, receipt_sha.as_deref())
        .map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&verified).map_err(|e| e.to_string())?);
    Ok(())
}
