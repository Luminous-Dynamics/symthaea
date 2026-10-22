// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic consent-state diagnostic. Reports outcomes; it does not assert
//! that this heuristic parser meets a release threshold.
//!
//! Run with: cargo run --example consent_semantics_bench
//! The fixture set is a development diagnostic, not a sealed evaluation set.

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
];

const STATES: [ConsentState; 5] = [
    ConsentState::Given,
    ConsentState::Denied,
    ConsentState::Absent,
    ConsentState::Implied,
    ConsentState::Unknown,
];

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

fn main() {
    let parser = MoralParser::new();
    let mut confusion = [[0usize; 5]; 5];
    let mut correct = 0usize;
    let mut false_given = 0usize;
    let mut expected_non_given = 0usize;

    println!("consent_semantics_bench_v1");
    println!("fixture_count={}", FIXTURES.len());
    for fixture in FIXTURES {
        let actual = parser.parse(fixture.text).consent;
        confusion[state_index(fixture.expected)][state_index(actual)] += 1;
        if actual == fixture.expected {
            correct += 1;
        }
        if fixture.expected != ConsentState::Given {
            expected_non_given += 1;
            if actual == ConsentState::Given {
                false_given += 1;
                println!("FALSE_GIVEN {} expected={} actual={} text={:?}",
                    fixture.id, label(fixture.expected), label(actual), fixture.text);
            }
        } else if actual != fixture.expected {
            println!("MISSED_GIVEN {} expected={} actual={} text={:?}",
                fixture.id, label(fixture.expected), label(actual), fixture.text);
        }
    }

    println!("accuracy={:.4}", correct as f64 / FIXTURES.len() as f64);
    println!("false_given_count={}", false_given);
    println!("non_given_false_positive_rate={:.4}",
        false_given as f64 / expected_non_given.max(1) as f64);
    println!("confusion_matrix=rows_expected_cols_actual");
    print!("expected/actual");
    for state in STATES {
        print!(",{}", label(state));
    }
    println!();
    for (row, expected) in STATES.iter().enumerate() {
        print!("{}", label(*expected));
        for column in 0..STATES.len() {
            print!(",{}", confusion[row][column]);
        }
        println!();
    }
}
