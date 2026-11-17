use std::{env, fs, path::PathBuf, process};

use serde::Serialize;
use symthaea_marine_geometry::AdmissionProfile;
use symthaea_marine_geometry::full_envelope::FullEnvelopeSpec;
use symthaea_marine_geometry::full_envelope_mesh::{
    FullEnvelopeMeshArtifact, FullEnvelopeMeshIssue, generate_full_envelope_mesh,
    verify_full_envelope_mesh_artifact,
};

const USAGE: &str = concat!(
    "Usage:\n",
    "  marine-full-envelope-mesh <spec.json> <admission-profile.json> [--output <artifact.json>]\n",
    "  marine-full-envelope-mesh --verify <spec.json> <admission-profile.json> <artifact.json>\n",
    "Generates or replays a closed external hull mesh with semantic surface regions.\n",
    "It does not calculate vessel stability, hydrodynamics, manufacturing feasibility, or approval.\n",
    "Exit codes: 0 = generated/verified; 1 = unsupported/incomplete; 2 = rejected or CLI/I/O/JSON error."
);

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
enum CliReport {
    Generated {
        artifact: FullEnvelopeMeshArtifact,
    },
    Verified {
        manifest_digest_blake3: String,
        mesh_digest_blake3: String,
    },
    Rejected {
        issues: Vec<FullEnvelopeMeshIssue>,
    },
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, label: &str) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid {label}: {error}"))
}

fn write_report(output: Option<PathBuf>, report: &CliReport) -> Result<(), String> {
    let serialized = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("cannot serialize full-envelope mesh report: {error}"))?;
    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }
    Ok(())
}

fn issue_exit_code(issues: &[FullEnvelopeMeshIssue]) -> i32 {
    if issues.iter().any(|item| {
        item.level == symthaea_marine_geometry::full_envelope_mesh::FullEnvelopeMeshIssueLevel::Rejected
    }) {
        2
    } else {
        1
    }
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(USAGE.to_owned());
    }
    if args[0] == "-h" || args[0] == "--help" {
        println!("{USAGE}");
        return Ok(0);
    }

    if args[0] == "--verify" {
        if args.len() != 4 {
            return Err(format!(
                "verification requires spec, profile and artifact paths\n{USAGE}"
            ));
        }
        let artifact_path = PathBuf::from(args.pop().unwrap());
        let profile_path = PathBuf::from(args.pop().unwrap());
        let spec_path = PathBuf::from(args.pop().unwrap());
        args.clear();

        let specification: FullEnvelopeSpec = read_json(&spec_path, "full-envelope specification")?;
        let admission_profile: AdmissionProfile =
            read_json(&profile_path, "G0/G1 admission profile")?;
        let artifact: FullEnvelopeMeshArtifact =
            read_json(&artifact_path, "full-envelope mesh artifact")?;
        match verify_full_envelope_mesh_artifact(&specification, &admission_profile, &artifact) {
            Ok(()) => {
                write_report(
                    None,
                    &CliReport::Verified {
                        manifest_digest_blake3: artifact.manifest_digest_blake3,
                        mesh_digest_blake3: artifact.mesh_digest_blake3,
                    },
                )?;
                Ok(0)
            }
            Err(issues) => {
                let code = issue_exit_code(&issues);
                write_report(None, &CliReport::Rejected { issues })?;
                Ok(code)
            }
        }
    } else {
        let mut output = None;
        let mut positional = Vec::new();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--output" => {
                    let path = args
                        .get(index + 1)
                        .ok_or_else(|| "--output requires a path".to_string())?;
                    output = Some(PathBuf::from(path));
                    index += 2;
                }
                option if option.starts_with('-') => {
                    return Err(format!("unknown option: {option}\n{USAGE}"));
                }
                value => {
                    positional.push(value.to_owned());
                    index += 1;
                }
            }
        }
        if positional.len() != 2 {
            return Err(format!(
                "generation requires spec and admission-profile paths\n{USAGE}"
            ));
        }
        let specification: FullEnvelopeSpec = read_json(
            &PathBuf::from(&positional[0]),
            "full-envelope specification",
        )?;
        let admission_profile: AdmissionProfile =
            read_json(&PathBuf::from(&positional[1]), "G0/G1 admission profile")?;
        match generate_full_envelope_mesh(&specification, &admission_profile) {
            Ok(artifact) => {
                write_report(output, &CliReport::Generated { artifact })?;
                Ok(0)
            }
            Err(issues) => {
                let code = issue_exit_code(&issues);
                write_report(output, &CliReport::Rejected { issues })?;
                Ok(code)
            }
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
