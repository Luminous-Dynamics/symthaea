//! Validator for the pinned, upstream HiddenBench 65-task corpus.
//!
//! This binary validates task structure and corpus identity. It does not run
//! model agents or expose evaluator answers to any agent-facing component.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;

const EXPECTED_TASKS: usize = 65;
const EXPECTED_GROUP_SIZES: [(usize, usize); 2] = [(3, 7), (4, 58)];

#[derive(Debug, Deserialize)]
struct Task {
    id: Value,
    name: String,
    description: String,
    shared_information: Vec<String>,
    hidden_information: Vec<String>,
    possible_answers: Vec<String>,
    correct_answer: String,
}

#[derive(Debug, Deserialize)]
struct TaskSet(Vec<Task>);

#[derive(Debug, Serialize)]
struct CorpusReport {
    protocol_id: &'static str,
    source: &'static str,
    task_count: usize,
    group_size_histogram: BTreeMap<usize, usize>,
    shared_fact_count: usize,
    hidden_fact_count: usize,
    answer_option_count: usize,
    structural_status: &'static str,
    claim_ceiling: &'static str,
}

fn id_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn validate_tasks(tasks: &[Task]) -> Result<CorpusReport, Vec<String>> {
    let mut errors = Vec::new();
    let mut ids = HashSet::new();
    let mut names = HashSet::new();
    let mut group_size_histogram = BTreeMap::new();
    let mut shared_fact_count = 0usize;
    let mut hidden_fact_count = 0usize;
    let mut answer_option_count = 0usize;

    for (index, task) in tasks.iter().enumerate() {
        let prefix = format!("task[{}]", index);

        match id_string(&task.id) {
            Some(id) if !ids.insert(id.clone()) => {
                errors.push(format!("{prefix}: duplicate task id {id}"));
            }
            Some(_) => {}
            None => errors.push(format!("{prefix}: id must be a non-empty string or number")),
        }

        if task.name.trim().is_empty() {
            errors.push(format!("{prefix}: name is empty"));
        } else if !names.insert(task.name.clone()) {
            errors.push(format!("{prefix}: duplicate task name {}", task.name));
        }
        if task.description.trim().is_empty() {
            errors.push(format!("{prefix}: description is empty"));
        }

        if task.shared_information.is_empty()
            || task.shared_information.iter().any(|fact| fact.trim().is_empty())
        {
            errors.push(format!("{prefix}: shared_information must contain non-empty facts"));
        }
        if task.hidden_information.is_empty()
            || task.hidden_information.iter().any(|fact| fact.trim().is_empty())
        {
            errors.push(format!("{prefix}: hidden_information must contain non-empty facts"));
        }
        if task.possible_answers.len() < 2
            || task.possible_answers.iter().any(|answer| answer.trim().is_empty())
        {
            errors.push(format!("{prefix}: possible_answers must contain at least two non-empty options"));
        }

        let mut answer_set = HashSet::new();
        for answer in &task.possible_answers {
            if !answer_set.insert(answer) {
                errors.push(format!("{prefix}: duplicate possible answer {answer}"));
            }
        }
        if !task.possible_answers.iter().any(|answer| answer == &task.correct_answer) {
            errors.push(format!("{prefix}: correct_answer is not an exact possible_answers member"));
        }

        *group_size_histogram.entry(task.hidden_information.len()).or_insert(0) += 1;
        shared_fact_count += task.shared_information.len();
        hidden_fact_count += task.hidden_information.len();
        answer_option_count += task.possible_answers.len();
    }

    if tasks.len() != EXPECTED_TASKS {
        errors.push(format!(
            "expected {EXPECTED_TASKS} tasks, found {}",
            tasks.len()
        ));
    }

    for (group_size, expected_count) in EXPECTED_GROUP_SIZES {
        let actual_count = group_size_histogram.get(&group_size).copied().unwrap_or(0);
        if actual_count != expected_count {
            errors.push(format!(
                "expected {expected_count} tasks with {group_size} hidden facts / agents, found {actual_count}"
            ));
        }
    }

    for group_size in group_size_histogram.keys() {
        if !EXPECTED_GROUP_SIZES.iter().any(|(expected, _)| expected == group_size) {
            errors.push(format!("unexpected hidden-information group size: {group_size}"));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(CorpusReport {
        protocol_id: "IG-HIDDEN-INFO-001",
        source: "Yassellee/HiddenBench_ICML@3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1:data/benchmark.json",
        task_count: tasks.len(),
        group_size_histogram,
        shared_fact_count,
        hidden_fact_count,
        answer_option_count,
        structural_status: "PASS_STRUCTURAL_CORPUS",
        claim_ceiling: "Corpus structure only; no agent run, accuracy, causal effect, or SOTA claim.",
    })
}

fn load_and_validate(path: &str) -> Result<CorpusReport, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read benchmark {path}: {error}"))?;
    let tasks: Vec<Task> = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid HiddenBench task JSON: {error}"))?;
    validate_tasks(&tasks).map_err(|errors| errors.join("\n"))
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: validate_hiddenbench <benchmark.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one benchmark path");
        std::process::exit(2);
    }

    match load_and_validate(&path) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("serializing a corpus report should not fail")
            );
        }
        Err(error) => {
            eprintln!("HiddenBench corpus validation failed:\n{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_CORPUS: &str = include_str!("../../../external/hiddenbench/benchmark.json");

    #[test]
    fn pinned_official_corpus_passes_structural_validation() {
        let tasks: Vec<Task> = serde_json::from_str(PINNED_CORPUS).expect("pinned task file parses");
        let report = validate_tasks(&tasks).expect("pinned task corpus passes");
        assert_eq!(report.task_count, 65);
        assert_eq!(report.group_size_histogram.get(&3), Some(&7));
        assert_eq!(report.group_size_histogram.get(&4), Some(&58));
        assert_eq!(report.structural_status, "PASS_STRUCTURAL_CORPUS");
    }

    #[test]
    fn duplicate_task_ids_are_rejected() {
        let mut tasks: Vec<Task> = serde_json::from_str(PINNED_CORPUS).expect("pinned task file parses");
        tasks[1].id = tasks[0].id.clone();
        let errors = validate_tasks(&tasks).expect_err("duplicate id must fail");
        assert!(errors.iter().any(|error| error.contains("duplicate task id")));
    }

    #[test]
    fn correct_answer_must_match_option_exactly() {
        let mut tasks: Vec<Task> = serde_json::from_str(PINNED_CORPUS).expect("pinned task file parses");
        tasks[0].correct_answer = "not an option".to_string();
        let errors = validate_tasks(&tasks).expect_err("unknown correct answer must fail");
        assert!(errors.iter().any(|error| error.contains("correct_answer is not an exact")));
    }

    #[test]
    fn changed_group_size_histogram_is_rejected() {
        let mut tasks: Vec<Task> = serde_json::from_str(PINNED_CORPUS).expect("pinned task file parses");
        tasks[0].hidden_information.pop();
        let errors = validate_tasks(&tasks).expect_err("altered group profile must fail");
        assert!(errors.iter().any(|error| error.contains("expected 58 tasks with 4 hidden facts")));
    }
}
