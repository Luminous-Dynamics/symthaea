// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic consent-state diagnostic. Reports outcomes; it does not assert
//! that this heuristic parser meets a release threshold.
//!
//! Run with: cargo run --example consent_semantics_bench
//! The fixture set is development-only, not blinded or representative.

use symthaea::hdc::moral_algebra::ConsentState;
use symthaea::hdc::moral_parser::MoralParser;

#[derive(Clone, Copy)]
struct Fixture {
    id: &'static str,
    text: &'static str,
    expected: ConsentState,
}

const FIXTURES: &[Fixture] = &[
    Fixture { id: "given-01", text: "She gave consent to share the report.", expected: ConsentState::Given },
    Fixture { id: "given-02", text: "They said yes to the request.", expected: ConsentState::Given },
    Fixture { id: "given-03", text: "She granted permission to publish it.", expected: ConsentState::Given },
    Fixture { id: "given-04", text: "She explicitly authorized the export.", expected: ConsentState::Given },
    Fixture { id: "given-05", text: "She agreed to treatment.", expected: ConsentState::Given },
    Fixture { id: "given-06", text: "With her permission, I shared the result.", expected: ConsentState::Given },
    Fixture { id: "denied-01", text: "They refused consent.", expected: ConsentState::Denied },
    Fixture { id: "denied-02", text: "She said no to the request.", expected: ConsentState::Denied },
    Fixture { id: "denied-03", text: "She did not consent.", expected: ConsentState::Denied },
    Fixture { id: "denied-04", text: "She withdrew consent.", expected: ConsentState::Denied },
    Fixture { id: "denied-05", text: "She never agreed.", expected: ConsentState::Denied },
    Fixture { id: "absent-01", text: "I shared the file without asking.", expected: ConsentState::Absent },
    Fixture { id: "absent-02", text: "I published the image without permission.", expected: ConsentState::Absent },
    Fixture { id: "absent-03", text: "I proceeded without explicit consent.", expected: ConsentState::Absent },
    Fixture { id: "absent-04", text: "I asked for consent and received no response.", expected: ConsentState::Absent },
    Fixture { id: "absent-05", text: "I did not obtain consent before exporting it.", expected: ConsentState::Absent },
    Fixture { id: "implied-01", text: "I assumed consent.", expected: ConsentState::Implied },
    Fixture { id: "implied-02", text: "I presumed permission.", expected: ConsentState::Implied },
    Fixture { id: "implied-03", text: "I assumed she agreed to treatment.", expected: ConsentState::Implied },
    Fixture { id: "unknown-01", text: "I need to ask for permission.", expected: ConsentState::Unknown },
    Fixture { id: "unknown-02", text: "Consent was discussed, but no answer is included.", expected: ConsentState::Unknown },
    Fixture { id: "unknown-03", text: "Would permission be needed here?", expected: ConsentState::Unknown },
    Fixture { id: "unknown-04", text: "Who gave permission?", expected: ConsentState::Unknown },
    Fixture { id: "unknown-05", text: "The note says \"she gave consent\"; I have not verified it.", expected: ConsentState::Unknown },
    Fixture { id: "unknown-06", text: "Could you ask whether consent was given?", expected: ConsentState::Unknown },
    Fixture { id: "unknown-07", text: "The record is missing.", expected: ConsentState::Unknown },
    Fixture { id: "unknown-08", text: "Did she refuse consent?", expected: ConsentState::Unknown },
];

const STATES: [ConsentState; 5] = [
    ConsentState::Given,
    ConsentState::Denied,
    ConsentState::Absent,
    ConsentState::Implied,
    ConsentState::Unknown,
];

#[derive(Clone)]
struct Metrics {
    confusion: [[usize; 5]; 5],
    correct: usize,
    false_given: usize,
    expected_non_given: usize,
}

impl Metrics {
    fn new() -> Self {
        Self { confusion: [[0; 5]; 5], correct: 0, false_given: 0, expected_non_given: 0 }
    }

    fn observe(&mut self, expected: ConsentState, actual: ConsentState) {
        self.confusion[state_index(expected)][state_index(actual)] += 1;
        if actual == expected {
            self.correct += 1;
        }
        if expected != ConsentState::Given {
            self.expected_non_given += 1;
            if actual == ConsentState::Given {
                self.false_given += 1;
            }
        }
    }
}

fn state_index(state: ConsentState) -> usize {
    match state {
        ConsentState::Given => 0,
        ConsentState::Denied => 1,
        ConsentState::Absent => 2,
        ConsentState::Implied => 3,
        ConsentState::Unknown => 4,
    }
}

fn label(state: ConsentState) -> &'static str {
    match state {
        ConsentState::Given => "Given",
        ConsentState::Denied => "Denied",
        ConsentState::Absent => "Absent",
        ConsentState::Implied => "Implied",
        ConsentState::Unknown => "Unknown",
    }
}

fn strip_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut closing_quote: Option<char> = None;
    for ch in text.chars() {
        if let Some(end) = closing_quote {
            if ch == end {
                closing_quote = None;
            }
            result.push(' ');
            continue;
        }
        match ch {
            '"' => {
                closing_quote = Some('"');
                result.push(' ');
            }
            '`' => {
                closing_quote = Some('`');
                result.push(' ');
            }
            '“' => {
                closing_quote = Some('”');
                result.push(' ');
            }
            '‘' => {
                closing_quote = Some('’');
                result.push(' ');
            }
            '”' | '’' => result.push(' '),
            _ => result.push(ch),
        }
    }
    result
}

fn has_any(text: &str, cues: &[&str]) -> bool {
    cues.iter().any(|cue| text.contains(cue))
}

/// Deliberately small and conservative transparent lexical baseline.
/// This is a project baseline, not an industry/SOTA model.
fn strict_lexical_baseline(text: &str) -> ConsentState {
    let text = strip_quotes(&text.to_lowercase());
    if text.contains('?') {
        return ConsentState::Unknown;
    }
    if has_any(&text, &[
        "refused consent", "refused permission", "declined consent",
        "said no", "withdrew consent", "revoked consent",
        "did not consent", "didn't consent", "never agreed",
    ]) {
        return ConsentState::Denied;
    }
    if has_any(&text, &[
        "without asking", "without permission", "without consent",
        "did not obtain consent", "didn't obtain consent", "no response",
        "did not answer", "didn't answer",
    ]) {
        return ConsentState::Absent;
    }
    if has_any(&text, &[
        "assumed consent", "assumed permission", "implied consent",
        "implicitly consented", "presumed permission", "assumed she agreed",
    ]) {
        return ConsentState::Implied;
    }
    if has_any(&text, &[
        "gave consent", "said yes", "granted permission", "with her permission",
        "with their permission", "permission was granted", "explicitly authorized",
        "agreed to treatment",
    ]) {
        return ConsentState::Given;
    }
    ConsentState::Unknown
}

fn print_metrics(name: &str, metrics: &Metrics, total: usize) {
    println!("\nmodel={name}");
    println!("accuracy={:.4}", metrics.correct as f64 / total as f64);
    println!("false_given_count={}", metrics.false_given);
    println!("expected_non_given_count={}", metrics.expected_non_given);
    println!("non_given_false_positive_rate={:.4}",
        metrics.false_given as f64 / metrics.expected_non_given.max(1) as f64);
    println!("confusion_matrix=rows_expected_cols_actual");
    print!("expected/actual");
    for state in STATES {
        print!(",{}", label(state));
    }
    println!();
    for (row, expected) in STATES.iter().enumerate() {
        print!("{}", label(*expected));
        for column in 0..STATES.len() {
            print!(",{}", metrics.confusion[row][column]);
        }
        println!();
    }
}

fn main() {
    let parser = MoralParser::new();
    let mut candidate = Metrics::new();
    let mut baseline = Metrics::new();

    println!("consent_semantics_bench_v1");
    println!("fixture_count={}", FIXTURES.len());
    for fixture in FIXTURES {
        let assessment = parser.assess_consent_evidence(fixture.text);
        let actual = assessment.state;
        let baseline_actual = strict_lexical_baseline(fixture.text);
        candidate.observe(fixture.expected, actual);
        baseline.observe(fixture.expected, baseline_actual);
        if fixture.expected != ConsentState::Given && actual == ConsentState::Given {
            println!("CANDIDATE_FALSE_GIVEN {} expected={} basis={:?} cue={:?} text={:?}",
                fixture.id, label(fixture.expected), assessment.basis,
                assessment.matched_phrase, fixture.text);
        }
        if fixture.expected != ConsentState::Given && baseline_actual == ConsentState::Given {
            println!("BASELINE_FALSE_GIVEN {} expected={} text={:?}",
                fixture.id, label(fixture.expected), fixture.text);
        }
        if fixture.expected == ConsentState::Given && actual != ConsentState::Given {
            println!("CANDIDATE_MISSED_GIVEN {} actual={} basis={:?} cue={:?} text={:?}",
                fixture.id, label(actual), assessment.basis,
                assessment.matched_phrase, fixture.text);
        }
        if actual != fixture.expected {
            println!("MISMATCH {} expected={} actual={} basis={:?} cue={:?} question_suppressed={} quoted_text_suppressed={} explanation={:?}",
                fixture.id, label(fixture.expected), label(actual), assessment.basis,
                assessment.matched_phrase, assessment.question_suppressed,
                assessment.quoted_text_suppressed, assessment.explanation());
        }
    }

    print_metrics("symthaea_parser", &candidate, FIXTURES.len());
    print_metrics("strict_lexical_baseline", &baseline, FIXTURES.len());

    let pairs = [
        ("She gave consent to share the report.",
         "Yesterday, she gave consent to share the report."),
        ("I asked for permission to publish the report.",
         "Yesterday, I asked for permission to publish the report."),
        ("I shared the report without asking.",
         "Yesterday I shared the report without asking."),
        ("I assumed she agreed to treatment.",
         "Earlier, I assumed she agreed to treatment."),
    ];
    let candidate_flips = pairs.iter().filter(|(a, b)| parser.parse(a).consent != parser.parse(b).consent).count();
    let baseline_flips = pairs.iter().filter(|(a, b)| strict_lexical_baseline(a) != strict_lexical_baseline(b)).count();
    println!("meaning_preserving_pair_count={}", pairs.len());
    println!("candidate_pair_flips={candidate_flips}");
    println!("baseline_pair_flips={baseline_flips}");

    // Relevant-fact counterfactuals should change the consent state in the
    // expected direction; these are deterministic development fixtures only.
    let counterfactuals = [
        ("explicit-grant-to-refusal",
         "She gave consent to share the report.", ConsentState::Given,
         "She refused consent to share the report.", ConsentState::Denied),
        ("without-asking-to-permission",
         "I shared the file without asking.", ConsentState::Absent,
         "I shared the file with her permission.", ConsentState::Given),
        ("assumption-to-affirmative",
         "I assumed she agreed to treatment.", ConsentState::Implied,
         "She agreed to treatment.", ConsentState::Given),
        ("request-only-to-affirmative-answer",
         "I asked my daughter before discussing her health.", ConsentState::Unknown,
         "After my daughter agreed, I discussed her health.", ConsentState::Given),
    ];
    let mut candidate_directional_correct = 0usize;
    let mut baseline_directional_correct = 0usize;
    for (id, before, expected_before, after, expected_after) in counterfactuals {
        let candidate_before = parser.assess_consent_evidence(before).state;
        let candidate_after = parser.assess_consent_evidence(after).state;
        let baseline_before = strict_lexical_baseline(before);
        let baseline_after = strict_lexical_baseline(after);
        let candidate_ok = candidate_before == expected_before && candidate_after == expected_after;
        let baseline_ok = baseline_before == expected_before && baseline_after == expected_after;
        candidate_directional_correct += if candidate_ok { 1 } else { 0 };
        baseline_directional_correct += if baseline_ok { 1 } else { 0 };
        println!("COUNTERFACTUAL {id} expected={}->{} candidate={}->{} candidate_correct={} baseline={}->{} baseline_correct={}",
            label(expected_before), label(expected_after),
            label(candidate_before), label(candidate_after), candidate_ok,
            label(baseline_before), label(baseline_after), baseline_ok);
    }
    println!("relevant_counterfactual_count={}", counterfactuals.len());
    println!("candidate_counterfactual_correct={candidate_directional_correct}");
    println!("baseline_counterfactual_correct={baseline_directional_correct}");
}
