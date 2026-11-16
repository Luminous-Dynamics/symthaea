use std::{env, fs, path::PathBuf, process};

use symthaea_marine_geometry::cad_exchange::{
    verify_translation_receipt, CadTranslationReceipt,
};

const USAGE: &str = concat!(
    "Usage: marine-cad-receipt-verify <receipt.json>\n",
    "Verifies receipt integrity, AP242 edition, independent backend and required evidence."
);

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let input = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(format!("exactly one receipt input is accepted\n{USAGE}"));
    }
    let input = input.ok_or_else(|| USAGE.to_string())?;
    let bytes = fs::read(&input)
        .map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let receipt: CadTranslationReceipt = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid CAD translation receipt: {error}"))?;

    match verify_translation_receipt(&receipt) {
        Ok(()) => {
            println!(
                "qualified-within-profile: candidate={} receipt={}",
                receipt.payload.candidate_id, receipt.receipt_digest_blake3
            );
            Ok(0)
        }
        Err(violations) => {
            println!(
                "not-qualified: candidate={} violations={}",
                receipt.payload.candidate_id,
                serde_json::to_string(&violations)
                    .map_err(|error| format!("cannot serialize violations: {error}"))?
            );
            Ok(1)
        }
    }
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}
