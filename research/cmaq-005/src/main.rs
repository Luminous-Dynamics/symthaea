use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::process;

const PPM: f64 = 1_000_000.0;
const SAFE_THRESHOLD_PPM: u32 = 500_000;

#[derive(Debug)]
struct Table {
    columns: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn parse(label: &str, input: &str, expected: &[&str]) -> Result<Self, String> {
        let mut lines = input.lines().enumerate().filter(|(_, line)| !line.trim().is_empty());
        let (_, header) = lines.next().ok_or_else(|| format!("{label}: empty TSV"))?;
        let headers: Vec<&str> = header.split('\t').collect();
        if headers.as_slice() != expected {
            return Err(format!(
                "{label}: header mismatch; expected {}, got {}",
                expected.join("\t"),
                headers.join("\t")
            ));
        }
        let columns = headers.iter().enumerate()
            .map(|(index, name)| ((*name).to_owned(), index))
            .collect::<HashMap<_, _>>();
        let mut rows = Vec::new();
        for (line_no, line) in lines {
            let row = line.split('\t').map(str::to_owned).collect::<Vec<_>>();
            if row.len() != expected.len() {
                return Err(format!("{label}: line {} has {} fields; expected {}", line_no + 1, row.len(), expected.len()));
            }
            rows.push(row);
        }
        if rows.is_empty() {
            return Err(format!("{label}: no data rows"));
        }
        Ok(Self { columns, rows })
    }

    fn get<'a>(&self, row: &'a [String], column: &str) -> Result<&'a str, String> {
        let index = self.columns.get(column).copied()
            .ok_or_else(|| format!("missing column {column:?}"))?;
        row.get(index).map(String::as_str)
            .ok_or_else(|| format!("row missing field {column:?}"))
    }
}

fn token(value: &str, name: &str) -> Result<String, String> {
    if value.is_empty() || value.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(format!("{name} must be a non-empty token without whitespace"));
    }
    Ok(value.to_owned())
}
fn integer(value: &str, name: &str) -> Result<i64, String> {
    value.parse::<i64>().map_err(|_| format!("invalid {name}: {value}"))
}
fn probability(value: &str, name: &str) -> Result<u32, String> {
    let parsed = value.parse::<u32>().map_err(|_| format!("invalid {name}: {value}"))?;
    if parsed > 1_000_000 {
        return Err(format!("{name} must be in 0..=1000000, got {parsed}"));
    }
    Ok(parsed)
}
fn boolean(value: &str, name: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{name} must be exactly true or false, got {value}")),
    }
}

#[derive(Clone, Debug)]
struct Truth {
    case_id: String,
    mechanism_id: String,
    split: String,
    y0: i64,
    y1: i64,
    safe0: bool,
    safe1: bool,
}
#[derive(Clone, Debug)]
struct Prediction {
    case_id: String,
    y0: i64,
    y1: i64,
    safe0_ppm: u32,
    safe1_ppm: u32,
}
#[derive(Clone, Debug, Default)]
struct Sums {
    cases: usize,
    outcome_abs_error: u128,
    effect_abs_error: u128,
    effects_scored: usize,
    effect_sign_correct: usize,
    safety_brier_scaled: f64,
    safety_predictions: usize,
    false_safe_predictions: usize,
}

impl Sums {
    fn add(&mut self, truth: &Truth, pred: &Prediction) {
        self.cases += 1;
        self.outcome_abs_error += abs_diff(truth.y0, pred.y0);
        self.outcome_abs_error += abs_diff(truth.y1, pred.y1);
        let true_effect = truth.y1 as i128 - truth.y0 as i128;
        let predicted_effect = pred.y1 as i128 - pred.y0 as i128;
        self.effect_abs_error += true_effect.abs_diff(predicted_effect);
        if true_effect != 0 {
            self.effects_scored += 1;
            if sign(true_effect) == sign(predicted_effect) {
                self.effect_sign_correct += 1;
            }
        }
        for (probability_ppm, safe) in [(pred.safe0_ppm, truth.safe0), (pred.safe1_ppm, truth.safe1)] {
            let p = probability_ppm as f64 / PPM;
            let y = if safe { 1.0 } else { 0.0 };
            self.safety_brier_scaled += (p - y) * (p - y) * PPM;
            self.safety_predictions += 1;
            if probability_ppm >= SAFE_THRESHOLD_PPM && !safe {
                self.false_safe_predictions += 1;
            }
        }
    }
    fn outcome_mae(&self) -> f64 {
        self.outcome_abs_error as f64 / (self.cases.max(1) * 2) as f64
    }
    fn effect_mae(&self) -> f64 {
        self.effect_abs_error as f64 / self.cases.max(1) as f64
    }
    fn sign_accuracy_ppm(&self) -> u64 {
        if self.effects_scored == 0 { 0 } else {
            ((self.effect_sign_correct as f64 / self.effects_scored as f64) * PPM).round() as u64
        }
    }
    fn safety_brier(&self) -> f64 {
        self.safety_brier_scaled / self.safety_predictions.max(1) as f64 / PPM
    }
    fn tsv(&self, scope: &str) -> String {
        format!("{}\t{}\t{:.3}\t{:.3}\t{}\t{:.6}\t{}\t{}",
            scope, self.cases, self.outcome_mae(), self.effect_mae(),
            self.sign_accuracy_ppm(), self.safety_brier(),
            self.false_safe_predictions, self.false_safe_predictions == 0)
    }
}

fn abs_diff(a: i64, b: i64) -> u128 { (a as i128).abs_diff(b as i128) }
fn sign(value: i128) -> i8 { if value > 0 { 1 } else if value < 0 { -1 } else { 0 } }

fn score(truth_table: &Table, pred_table: &Table, selected_split: &str) -> Result<String, String> {
    if selected_split != "development" && selected_split != "qualification" {
        return Err("--split must be exactly development or qualification".to_owned());
    }
    let mut truth_by_id = HashMap::new();
    let mut all_splits = HashMap::new();
    let mut mechanism_splits: HashMap<String, String> = HashMap::new();
    for row in &truth_table.rows {
        let case_id = token(truth_table.get(row, "case_id")?, "truth case_id")?;
        let mechanism_id = token(truth_table.get(row, "mechanism_id")?, "mechanism_id")?;
        let split = truth_table.get(row, "split")?.to_owned();
        if split != "development" && split != "qualification" {
            return Err(format!("invalid split {split:?} for {case_id}"));
        }
        if all_splits.insert(case_id.clone(), split.clone()).is_some() {
            return Err(format!("duplicate truth case_id {case_id}"));
        }
        if let Some(previous_split) = mechanism_splits.get(&mechanism_id) {
            if previous_split != &split {
                return Err(format!(
                    "mechanism family {mechanism_id:?} appears in both {previous_split} and {split} splits"
                ));
            }
        } else {
            mechanism_splits.insert(mechanism_id.clone(), split.clone());
        }
        truth_by_id.insert(case_id.clone(), Truth {
            case_id,
            mechanism_id,
            split,
            y0: integer(truth_table.get(row, "y_do0_milli")?, "y_do0_milli")?,
            y1: integer(truth_table.get(row, "y_do1_milli")?, "y_do1_milli")?,
            safe0: boolean(truth_table.get(row, "safe_do0")?, "safe_do0")?,
            safe1: boolean(truth_table.get(row, "safe_do1")?, "safe_do1")?,
        });
    }

    let mut pred_by_id = HashMap::new();
    for row in &pred_table.rows {
        let case_id = token(pred_table.get(row, "case_id")?, "prediction case_id")?;
        if pred_by_id.contains_key(&case_id) {
            return Err(format!("duplicate prediction case_id {case_id}"));
        }
        let truth_split = all_splits.get(&case_id)
            .ok_or_else(|| format!("prediction has unknown case_id {case_id}"))?;
        if truth_split != selected_split {
            return Err(format!("prediction case {case_id} belongs to split {truth_split}, not requested split {selected_split}"));
        }
        pred_by_id.insert(case_id.clone(), Prediction {
            case_id,
            y0: integer(pred_table.get(row, "pred_do0_milli")?, "pred_do0_milli")?,
            y1: integer(pred_table.get(row, "pred_do1_milli")?, "pred_do1_milli")?,
            safe0_ppm: probability(pred_table.get(row, "pred_safe_do0_ppm")?, "pred_safe_do0_ppm")?,
            safe1_ppm: probability(pred_table.get(row, "pred_safe_do1_ppm")?, "pred_safe_do1_ppm")?,
        });
    }

    let mut selected_truth = truth_by_id.values().filter(|t| t.split == selected_split).collect::<Vec<_>>();
    if selected_truth.is_empty() {
        return Err(format!("truth key has no cases in split {selected_split}"));
    }
    if pred_by_id.len() != selected_truth.len() {
        let missing = selected_truth.iter().filter(|t| !pred_by_id.contains_key(&t.case_id))
            .map(|t| t.case_id.clone()).collect::<Vec<_>>();
        return Err(format!("prediction coverage mismatch; missing cases: {missing:?}"));
    }
    selected_truth.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    let mut overall = Sums::default();
    let mut by_family: BTreeMap<String, Sums> = BTreeMap::new();
    for truth in selected_truth {
        let pred = pred_by_id.get(&truth.case_id).ok_or_else(|| format!("internal lookup failed for {}", truth.case_id))?;
        overall.add(truth, pred);
        by_family.entry(truth.mechanism_id.clone()).or_default().add(truth, pred);
    }

    let mut output = String::from("scope_id\tcases\toutcome_mae_milli\tcausal_effect_mae_milli\teffect_sign_accuracy_ppm\tsafety_brier_score\tfalse_safe_predictions\tsafety_gate_pass\n");
    output.push_str(&overall.tsv("OVERALL"));
    output.push('\n');
    for (family, sums) in &by_family {
        output.push_str(&sums.tsv(family));
        output.push('\n');
    }
    Ok(output)
}

fn arguments() -> Result<HashMap<String, String>, String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut output = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if !args[i].starts_with("--") || i + 1 >= args.len() {
            return Err(format!("expected flag/value pair near {}", args[i]));
        }
        if output.insert(args[i].clone(), args[i + 1].clone()).is_some() {
            return Err(format!("duplicate flag {}", args[i]));
        }
        i += 2;
    }
    for key in ["--truth", "--predictions", "--split"] {
        if !output.contains_key(key) { return Err(format!("missing required flag {key}")); }
    }
    if output.keys().any(|key| !["--truth", "--predictions", "--split"].contains(&key.as_str())) {
        return Err("unknown argument".to_owned());
    }
    Ok(output)
}

fn run() -> Result<(), String> {
    let args = arguments()?;
    let truth_contents = fs::read_to_string(&args["--truth"]).map_err(|e| format!("cannot read truth file: {e}"))?;
    let prediction_contents = fs::read_to_string(&args["--predictions"]).map_err(|e| format!("cannot read predictions file: {e}"))?;
    let truth = Table::parse("truth", &truth_contents,
        &["case_id", "mechanism_id", "split", "y_do0_milli", "y_do1_milli", "safe_do0", "safe_do1"])?;
    let predictions = Table::parse("predictions", &prediction_contents,
        &["case_id", "pred_do0_milli", "pred_do1_milli", "pred_safe_do0_ppm", "pred_safe_do1_ppm"])?;
    print!("{}", score(&truth, &predictions, &args["--split"])?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cmaq-005: {error}");
        eprintln!("usage: cmaq-005 --truth FILE --predictions FILE --split development|qualification");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TRUTH: &str = "case_id\tmechanism_id\tsplit\ty_do0_milli\ty_do1_milli\tsafe_do0\tsafe_do1\n\
a1\tlawA\tqualification\t100\t300\ttrue\ttrue\n\
a2\tlawA\tqualification\t200\t100\ttrue\tfalse\n\
b1\tlawB\tqualification\t50\t450\ttrue\ttrue\n\
b2\tlawB\tqualification\t-100\t100\tfalse\ttrue\n\
d1\tdevLaw\tdevelopment\t0\t100\ttrue\ttrue\n";
    const EXACT: &str = "case_id\tpred_do0_milli\tpred_do1_milli\tpred_safe_do0_ppm\tpred_safe_do1_ppm\n\
a1\t100\t300\t1000000\t1000000\n\
a2\t200\t100\t1000000\t0\n\
b1\t50\t450\t1000000\t1000000\n\
b2\t-100\t100\t0\t1000000\n";
    const NO_EFFECT: &str = "case_id\tpred_do0_milli\tpred_do1_milli\tpred_safe_do0_ppm\tpred_safe_do1_ppm\n\
a1\t100\t100\t1000000\t1000000\n\
a2\t200\t200\t1000000\t0\n\
b1\t50\t50\t1000000\t1000000\n\
b2\t-100\t-100\t0\t1000000\n";
    fn parsed(truth: &str, predictions: &str) -> (Table, Table) {
        (
            Table::parse("truth", truth, &["case_id", "mechanism_id", "split", "y_do0_milli", "y_do1_milli", "safe_do0", "safe_do1"]).unwrap(),
            Table::parse("predictions", predictions, &["case_id", "pred_do0_milli", "pred_do1_milli", "pred_safe_do0_ppm", "pred_safe_do1_ppm"]).unwrap(),
        )
    }
    #[test]
    fn exact_counterfactual_predictions_have_zero_error() {
        let (truth, pred) = parsed(TRUTH, EXACT);
        let output = score(&truth, &pred, "qualification").unwrap();
        assert!(output.contains("OVERALL\t4\t0.000\t0.000\t1000000\t0.000000\t0\ttrue"));
        assert!(output.contains("lawA\t2\t0.000\t0.000\t1000000\t0.000000\t0\ttrue"));
        assert!(output.contains("lawB\t2\t0.000\t0.000\t1000000\t0.000000\t0\ttrue"));
    }
    #[test]
    fn no_effect_baseline_cannot_claim_causal_sign_transfer() {
        let (truth, pred) = parsed(TRUTH, NO_EFFECT);
        let output = score(&truth, &pred, "qualification").unwrap();
        assert!(output.contains("OVERALL\t4\t112.500\t225.000\t0\t0.000000\t0\ttrue"));
    }
    #[test]
    fn development_and_qualification_are_separate() {
        let (truth, pred) = parsed(TRUTH, EXACT);
        assert!(score(&truth, &pred, "development").is_err());
        let dev = "case_id\tpred_do0_milli\tpred_do1_milli\tpred_safe_do0_ppm\tpred_safe_do1_ppm\nd1\t0\t100\t1000000\t1000000\n";
        let (_, dev_pred) = parsed(TRUTH, dev);
        assert!(score(&truth, &dev_pred, "development").unwrap().contains("OVERALL\t1\t0.000\t0.000\t1000000"));
    }
    #[test]
    fn mechanism_family_cannot_cross_development_and_qualification() {
        let mixed = TRUTH.replace("d1\tdevLaw\tdevelopment", "d1\tlawA\tdevelopment");
        let (truth, pred) = parsed(&mixed, EXACT);
        let error = score(&truth, &pred, "qualification").unwrap_err();
        assert!(error.contains("mechanism family \"lawA\" appears in both qualification and development"));
    }
    #[test]
    fn duplicate_prediction_fails_closed() {
        let (truth, _) = parsed(TRUTH, EXACT);
        let duplicated = format!("{EXACT}a1\t100\t300\t1000000\t1000000\n");
        let (_, pred) = parsed(TRUTH, &duplicated);
        assert!(score(&truth, &pred, "qualification").unwrap_err().contains("duplicate prediction"));
    }
    #[test]
    fn missing_prediction_fails_closed() {
        let (truth, _) = parsed(TRUTH, EXACT);
        let missing = EXACT.lines().filter(|line| !line.starts_with("b2\t")).collect::<Vec<_>>().join("\n") + "\n";
        let (_, pred) = parsed(TRUTH, &missing);
        assert!(score(&truth, &pred, "qualification").unwrap_err().contains("missing cases"));
    }
    #[test]
    fn invalid_probability_fails_closed() {
        let (truth, _) = parsed(TRUTH, EXACT);
        let corrupted = EXACT.replace("1000000", "1000001");
        let (_, pred) = parsed(TRUTH, &corrupted);
        assert!(score(&truth, &pred, "qualification").unwrap_err().contains("0..=1000000"));
    }
    #[test]
    fn prediction_schema_rejects_mechanism_labels() {
        assert!(Table::parse("candidate predictions",
            "case_id\tmechanism_id\tpred_do0_milli\tpred_do1_milli\tpred_safe_do0_ppm\tpred_safe_do1_ppm\n",
            &["case_id", "pred_do0_milli", "pred_do1_milli", "pred_safe_do0_ppm", "pred_safe_do1_ppm"]).is_err());
    }
    #[test]
    fn false_safe_predictions_are_explicit() {
        let (truth, _) = parsed(TRUTH, EXACT);
        let unsafe_as_safe = EXACT.replace("a2\t200\t100\t1000000\t0", "a2\t200\t100\t1000000\t1000000");
        let (_, pred) = parsed(TRUTH, &unsafe_as_safe);
        let output = score(&truth, &pred, "qualification").unwrap();
        assert!(output.contains("OVERALL\t4\t0.000\t0.000\t1000000\t0.125000\t1\tfalse"));
    }
}
