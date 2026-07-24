use clap::{Parser, Subcommand};
use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use symthaea_communication::benchmark::{
    BenchmarkReport, EvaluationPlan, ReleaseGate, compare_reports,
};
use symthaea_communication::human::{LocalJsonlProvider, WorkerPolicy};
use symthaea_communication::pilot::{
    PilotOutcome, PilotSample, evaluate_sample, rescore, summarize,
};
use symthaea_communication::provider::{ProviderManifest, SupportRegistry};
use symthaea_communication::run::{verify_run_bundle, write_run_bundle};

#[derive(Parser)]
#[command(
    name = "symthaea-comm-eval",
    about = "Evidence-gated communication benchmark tooling"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute the deterministic BLAKE3 hash used by artifact manifests.
    HashArtifact { path: PathBuf },
    /// Validate that a benchmark plan pins all required inputs.
    ValidatePlan { plan: PathBuf },
    /// Validate a provider manifest, including artifact and license declarations.
    ValidateProvider {
        manifest: PathBuf,
        /// Local model file or directory whose contents must match the manifest hash.
        #[arg(long)]
        artifact: Option<PathBuf>,
    },
    /// Apply release gates to a benchmark report.
    Gate {
        report: PathBuf,
        #[arg(long = "require-scope")]
        required_scopes: Vec<String>,
        #[arg(long = "require-metric")]
        required_metrics: Vec<String>,
        #[arg(long)]
        plan: Option<PathBuf>,
        #[arg(long)]
        provider_manifest: Option<PathBuf>,
    },
    /// Generate support claims exclusively from a passing report.
    Registry {
        report: PathBuf,
        output: PathBuf,
        #[arg(long = "require-scope")]
        required_scopes: Vec<String>,
        #[arg(long = "require-metric")]
        required_metrics: Vec<String>,
        #[arg(long)]
        plan: Option<PathBuf>,
        #[arg(long)]
        provider_manifest: Option<PathBuf>,
    },
    /// Fail when candidate metrics regress beyond the allowed relative fraction.
    Compare {
        baseline: PathBuf,
        candidate: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        maximum_relative_regression: f64,
    },
    /// Write an immutable, checksummed benchmark run directory.
    Bundle {
        plan: PathBuf,
        provider_manifest: PathBuf,
        report: PathBuf,
        registry: PathBuf,
        environment: PathBuf,
        per_sample: PathBuf,
        output: PathBuf,
    },
    /// Verify every file and the checksum root of an immutable run directory.
    VerifyRun { directory: PathBuf },
    /// Run WAV/JSONL samples through an allowlisted local worker.
    Pilot {
        samples: PathBuf,
        worker: PathBuf,
        provider_id: String,
        output: PathBuf,
        #[arg(long = "worker-arg")]
        worker_args: Vec<String>,
    },
    /// Aggregate per-sample pilot JSONL without manufacturing release evidence.
    SummarizePilot { outcomes: PathBuf, output: PathBuf },
    /// Recompute WER/CER for existing pilot outcomes (e.g. after a metrics
    /// fix) without re-invoking the provider.
    RescorePilot {
        samples: PathBuf,
        outcomes: PathBuf,
        output: PathBuf,
    },
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    BufReader::new(file)
        .lines()
        .filter_map(|line| match line {
            Ok(line) if !line.trim().is_empty() => Some(Ok(line)),
            Ok(_) => None,
            Err(error) => Some(Err(error.to_string())),
        })
        .map(|line| line.and_then(|line| serde_json::from_str(&line).map_err(|e| e.to_string())))
        .collect()
}

fn write_jsonl<T: serde::Serialize>(path: &Path, values: &[T]) -> Result<(), String> {
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    for value in values {
        serde_json::to_writer(&mut output, value).map_err(|error| error.to_string())?;
        output.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    output.sync_all().map_err(|error| error.to_string())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn gate(required_scopes: Vec<String>, required_metrics: Vec<String>) -> ReleaseGate {
    ReleaseGate {
        required_scopes: required_scopes.into_iter().collect::<BTreeSet<_>>(),
        required_metrics: required_metrics.into_iter().collect::<BTreeSet<_>>(),
        ..ReleaseGate::default()
    }
}

fn configured_gate(
    required_scopes: Vec<String>,
    required_metrics: Vec<String>,
    plan_path: Option<PathBuf>,
    provider_path: Option<PathBuf>,
) -> Result<ReleaseGate, String> {
    let mut result = if let Some(path) = plan_path {
        let plan: EvaluationPlan = read_json(&path)?;
        plan.validate()?;
        plan.release_gate()
    } else {
        gate(required_scopes.clone(), required_metrics.clone())
    };
    result.required_scopes.extend(required_scopes);
    result.required_metrics.extend(required_metrics);
    if let Some(path) = provider_path {
        let provider: ProviderManifest = read_json(&path)?;
        provider.validate()?;
        result.expected_provider = Some(provider.id);
        result.expected_model_hash = Some(provider.artifact_hash);
    }
    Ok(result)
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::HashArtifact { path } => {
            let hash = if path.is_dir() {
                symthaea_communication::artifact::hash_tree(&path)
            } else {
                symthaea_communication::artifact::hash_file(&path)
            }
            .map_err(|error| error.to_string())?;
            println!("{hash}");
        }
        Command::ValidatePlan { plan } => {
            let plan: EvaluationPlan = read_json(&plan)?;
            plan.validate()?;
            println!("evaluation plan {} is structurally valid", plan.id);
        }
        Command::ValidateProvider { manifest, artifact } => {
            let manifest: ProviderManifest = read_json(&manifest)?;
            manifest.validate()?;
            if let Some(path) = artifact {
                manifest.verify_components(&path)?;
            }
            println!(
                "provider {}@{} is structurally valid",
                manifest.id, manifest.version
            );
        }
        Command::Gate {
            report,
            required_scopes,
            required_metrics,
            plan,
            provider_manifest,
        } => {
            let report: BenchmarkReport = read_json(&report)?;
            configured_gate(required_scopes, required_metrics, plan, provider_manifest)?
                .evaluate(&report)
                .map_err(|failures| format!("release gate failed: {failures:?}"))?;
            println!("benchmark {} passes its release gate", report.benchmark_id);
        }
        Command::Registry {
            report,
            output,
            required_scopes,
            required_metrics,
            plan,
            provider_manifest,
        } => {
            let report: BenchmarkReport = read_json(&report)?;
            let registry = SupportRegistry::from_passing_report(
                &report,
                &configured_gate(required_scopes, required_metrics, plan, provider_manifest)?,
            )
            .map_err(|failures| format!("release gate failed: {failures:?}"))?;
            let bytes = serde_json::to_vec_pretty(&registry).map_err(|error| error.to_string())?;
            fs::write(&output, bytes).map_err(|error| format!("{}: {error}", output.display()))?;
            println!(
                "wrote {} evidence-backed support claims",
                registry.claims.len()
            );
        }
        Command::Compare {
            baseline,
            candidate,
            maximum_relative_regression,
        } => {
            if !(0.0..1.0).contains(&maximum_relative_regression) {
                return Err("maximum relative regression must be in [0, 1)".into());
            }
            let baseline: BenchmarkReport = read_json(&baseline)?;
            let candidate: BenchmarkReport = read_json(&candidate)?;
            let regressions = compare_reports(&baseline, &candidate, maximum_relative_regression);
            if !regressions.is_empty() {
                return Err(format!("metric regressions: {regressions:?}"));
            }
            println!("candidate has no matched metric regression above the configured limit");
        }
        Command::Bundle {
            plan,
            provider_manifest,
            report,
            registry,
            environment,
            per_sample,
            output,
        } => {
            let plan: EvaluationPlan = read_json(&plan)?;
            plan.validate()?;
            let provider: ProviderManifest = read_json(&provider_manifest)?;
            provider.validate()?;
            let report: BenchmarkReport = read_json(&report)?;
            let mut gate = plan.release_gate();
            gate.expected_provider = Some(provider.id.clone());
            gate.expected_model_hash = Some(provider.artifact_hash.clone());
            gate.evaluate(&report)
                .map_err(|failures| format!("release gate failed: {failures:?}"))?;
            let registry: SupportRegistry = read_json(&registry)?;
            let expected_registry = SupportRegistry::from_passing_report(&report, &gate)
                .map_err(|failures| format!("release gate failed: {failures:?}"))?;
            if registry != expected_registry {
                return Err("support registry does not match the passing report".into());
            }
            let environment: std::collections::BTreeMap<String, String> = read_json(&environment)?;
            let per_sample = fs::read(&per_sample).map_err(|error| error.to_string())?;
            let checksums = write_run_bundle(
                &output,
                &plan,
                &provider,
                &report,
                &registry,
                &environment,
                &per_sample,
            )?;
            println!("wrote immutable run bundle {}", checksums.bundle_hash);
        }
        Command::VerifyRun { directory } => {
            let checksums = verify_run_bundle(&directory)?;
            println!("verified run bundle {}", checksums.bundle_hash);
        }
        Command::Pilot {
            samples,
            worker,
            provider_id,
            output,
            worker_args,
        } => {
            let policy = WorkerPolicy::allow_one(&worker)?;
            let mut provider =
                LocalJsonlProvider::spawn(provider_id, &worker, &worker_args, policy)?;
            let input = fs::File::open(&samples).map_err(|error| error.to_string())?;
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)
                .map_err(|error| error.to_string())?;
            let mut count = 0_u64;
            for line in BufReader::new(input).lines() {
                let line = line.map_err(|error| error.to_string())?;
                if line.trim().is_empty() {
                    continue;
                }
                let sample: PilotSample =
                    serde_json::from_str(&line).map_err(|error| error.to_string())?;
                let outcome = evaluate_sample(&mut provider, &sample);
                serde_json::to_writer(&mut output, &outcome).map_err(|error| error.to_string())?;
                output.write_all(b"\n").map_err(|error| error.to_string())?;
                count += 1;
            }
            output.sync_all().map_err(|error| error.to_string())?;
            println!("wrote {count} per-sample pilot outcomes");
        }
        Command::SummarizePilot { outcomes, output } => {
            let input = fs::File::open(&outcomes).map_err(|error| error.to_string())?;
            let values: Vec<PilotOutcome> = BufReader::new(input)
                .lines()
                .filter_map(|line| match line {
                    Ok(line) if !line.trim().is_empty() => Some(Ok(line)),
                    Ok(_) => None,
                    Err(error) => Some(Err(error.to_string())),
                })
                .map(|line| {
                    line.and_then(|line| {
                        serde_json::from_str(&line).map_err(|error| error.to_string())
                    })
                })
                .collect::<Result<_, _>>()?;
            let bytes = serde_json::to_vec_pretty(&summarize(&values))
                .map_err(|error| error.to_string())?;
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)
                .and_then(|mut file| {
                    file.write_all(&bytes)?;
                    file.sync_all()
                })
                .map_err(|error| error.to_string())?;
            println!("wrote pilot summary for {} outcomes", values.len());
        }
        Command::RescorePilot {
            samples,
            outcomes,
            output,
        } => {
            let samples: Vec<PilotSample> = read_jsonl(&samples)?;
            let outcomes: Vec<PilotOutcome> = read_jsonl(&outcomes)?;
            let rescored = rescore(&samples, &outcomes);
            write_jsonl(&output, &rescored)?;
            println!("rescored {} outcomes", rescored.len());
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
