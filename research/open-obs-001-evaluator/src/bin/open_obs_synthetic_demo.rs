use open_obs_evaluator::synthetic::{
    generate_manifest_with_baseline, Baseline, PROFILE_ID,
};
use std::fs;
use std::process;

fn usage() -> &'static str {
    "usage: open_obs_synthetic_demo [--baseline matched-track|peak-threshold|always-reject|always-candidate] [--output PATH]"
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut baseline = Baseline::MatchedTrack;
    let mut output_path: Option<String> = None;
    let mut stdout_requested = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--baseline" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--baseline requires a value".to_owned())?;
                baseline = Baseline::parse(value)
                    .ok_or_else(|| format!("unsupported baseline {value:?}"))?;
            }
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a path".to_owned())?;
                if output_path.replace(value.clone()).is_some() {
                    return Err("--output may be supplied only once".to_owned());
                }
            }
            "--stdout" => {
                stdout_requested = true;
            }
            value => return Err(format!("unexpected argument {value:?}; {usage()}")),
        }
        index += 1;
    }

    if stdout_requested && output_path.is_some() {
        return Err("--stdout cannot be combined with --output".to_owned());
    }

    let bytes = generate_manifest_with_baseline(baseline)?;
    if let Some(path) = output_path {
        fs::write(&path, &bytes)
            .map_err(|error| format!("cannot write {path:?}: {error}"))?;
        eprintln!(
            "profile={} baseline={} rows={} bytes={} destination={}",
            PROFILE_ID,
            baseline.id(),
            bytes.iter().filter(|byte| **byte == b'\n').count().saturating_sub(1),
            bytes.len(),
            path
        );
    } else {
        use std::io::Write;
        std::io::stdout()
            .write_all(&bytes)
            .map_err(|error| format!("cannot write synthetic manifest: {error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        eprintln!("{}", usage());
        process::exit(2);
    }
}
