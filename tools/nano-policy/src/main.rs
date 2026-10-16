use nano_policy_prototype::{parse_tsv, select, Policy, Selection, MAX_INPUT_BYTES};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const USAGE: &str = "nano-policy-prototype --input candidates.tsv --policy random|maxmin --budget N [--seed N]

Input TSV:
  candidate_id<TAB>feature_1<TAB>feature_2...
  one candidate per row; every feature must be a finite, pre-outcome numeric descriptor.

Policies:
  random  seeded sampling without replacement
  maxmin  greedy farthest-point selection in standardized feature space

The first selected candidate is shared between policies for a given seed. Output is TSV with
one row for every input candidate, including exact-feature duplicates and unselected rows.
Distances are normalized descriptor-space diagnostics, not physical distances or quality scores.
Do not include target labels, measured spectra, energies, hidden split membership, or future outcomes
in the candidate feature columns.";

fn main() {
    if let Err(error) = run() {
        eprintln!("nano-policy-prototype: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return if args.is_empty() {
            Err("missing arguments; see --help".to_string())
        } else {
            Ok(())
        };
    }

    let mut input_path: Option<String> = None;
    let mut policy_value: Option<String> = None;
    let mut budget_value: Option<String> = None;
    let mut seed_value: Option<String> = None;
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("missing value after {flag}"))?
            .clone();
        index += 1;

        match flag {
            "--input" => {
                if input_path.replace(value).is_some() {
                    return Err("--input was supplied more than once".to_string());
                }
            }
            "--policy" => {
                if policy_value.replace(value).is_some() {
                    return Err("--policy was supplied more than once".to_string());
                }
            }
            "--budget" => {
                if budget_value.replace(value).is_some() {
                    return Err("--budget was supplied more than once".to_string());
                }
            }
            "--seed" => {
                if seed_value.replace(value).is_some() {
                    return Err("--seed was supplied more than once".to_string());
                }
            }
            _ => return Err(format!("unknown argument {flag:?}; see --help")),
        }
    }

    let input_path = input_path.ok_or_else(|| "--input is required".to_string())?;
    let policy = Policy::parse(
        &policy_value.ok_or_else(|| "--policy is required".to_string())?,
    )?;
    let budget = budget_value
        .ok_or_else(|| "--budget is required".to_string())?
        .parse::<usize>()
        .map_err(|_| "--budget must be a positive integer".to_string())?;
    let seed = seed_value
        .unwrap_or_else(|| "0".to_string())
        .parse::<u64>()
        .map_err(|_| "--seed must be an unsigned 64-bit integer".to_string())?;

    let path = Path::new(&input_path);
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot inspect input file: {error}"))?;
    if !metadata.is_file() {
        return Err("input path must be a regular file".to_string());
    }
    if metadata.len() > MAX_INPUT_BYTES as u64 {
        return Err(format!("input file exceeds the {MAX_INPUT_BYTES}-byte limit"));
    }
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read input as UTF-8 text: {error}"))?;
    let table = parse_tsv(&source)?;
    let report = select(&table, policy, budget, seed)?;

    let mut selected_by_index: Vec<Option<Selection>> = vec![None; table.candidates.len()];
    for selection in report.selected.iter().copied() {
        selected_by_index[selection.candidate_index] = Some(selection);
    }

    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    writeln!(output, "# nano-policy-prototype-v1").map_err(io_error)?;
    writeln!(output, "# policy={}", report.policy.as_str()).map_err(io_error)?;
    writeln!(output, "# seed={}", report.seed).map_err(io_error)?;
    writeln!(output, "# budget={}", report.budget).map_err(io_error)?;
    writeln!(output, "# candidate_count={}", table.candidates.len()).map_err(io_error)?;
    writeln!(
        output,
        "# unique_feature_vectors={}",
        report.eligible_count(table.candidates.len())
    )
    .map_err(io_error)?;
    writeln!(output, "# feature_names={}", table.feature_names.join(",")).map_err(io_error)?;
    writeln!(
        output,
        "candidate_id\tinput_index\tstatus\tselection_rank\tdistance_to_prior_selected\tduplicate_of"
    )
    .map_err(io_error)?;

    for (candidate_index, candidate) in table.candidates.iter().enumerate() {
        let duplicate_id = report.duplicate_of[candidate_index]
            .map(|first_index| table.candidates[first_index].id.as_str())
            .unwrap_or("");
        let (status, rank, distance) = if let Some(selection) = selected_by_index[candidate_index] {
            let rank = selection.rank.to_string();
            let distance = selection
                .distance_to_prior_selected
                .map(|value| format!("{value:.12}"))
                .unwrap_or_else(|| "-".to_string());
            ("selected", rank, distance)
        } else if report.duplicate_of[candidate_index].is_some() {
            ("duplicate_features", String::new(), String::new())
        } else {
            ("not_selected", String::new(), String::new())
        };

        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}",
            candidate.id, candidate_index, status, rank, distance, duplicate_id
        )
        .map_err(io_error)?;
    }
    output.flush().map_err(io_error)?;
    Ok(())
}

fn io_error(error: io::Error) -> String {
    format!("output error: {error}")
}
