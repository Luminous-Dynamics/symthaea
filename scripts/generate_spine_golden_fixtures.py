#!/usr/bin/env python3
"""Generate SPINE-000B P1R golden fixtures (Issue #3036).

The fixture matrix is intentionally explicit. Coverage tags are consumed by the
fail-closed P1R verifier so aggregate 0/1/2/N labels cannot hide a missing
single-channel or interaction class.
"""

import json
from pathlib import Path

from spine_000b_influence_oracle import Proposal, influence_report, integrate

FIXTURE_PATH = (
    Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "spine_000b_golden_fixtures.json"
)


def case(name, description, cycle_number, coverage_tags, proposals):
    return {
        "name": name,
        "description": description,
        "cycle_number": cycle_number,
        "coverage_tags": coverage_tags,
        "proposals": proposals,
    }


def main():
    cases = [
        case(
            "case_00_empty",
            "0 contributors (empty collector)",
            1,
            ["zero.empty"],
            [],
        ),
        case(
            "case_01_single_confidence",
            "1 contributor (confidence delta only)",
            2,
            ["one.confidence_only"],
            [{"subsystem_name": "sub_a", "confidence_delta": 0.25}],
        ),
        case(
            "case_02_single_lr",
            "1 contributor (LR modulation only)",
            3,
            ["one.lr_only"],
            [{"subsystem_name": "sub_a", "lr_modulation": 1.5}],
        ),
        case(
            "case_03_single_exploration",
            "1 contributor (exploration delta only)",
            4,
            ["one.exploration_only"],
            [{"subsystem_name": "sub_a", "exploration_delta": 0.375}],
        ),
        case(
            "case_04_single_arousal",
            "1 contributor (arousal f32 only)",
            5,
            ["one.arousal_only"],
            [{"subsystem_name": "sub_a", "arousal_delta": 0.25}],
        ),
        case(
            "case_05_single_valence",
            "1 contributor (valence f32 only)",
            6,
            ["one.valence_only"],
            [{"subsystem_name": "sub_a", "valence_delta": -0.125}],
        ),
        case(
            "case_06_single_flag",
            "1 contributor (flag only)",
            7,
            ["one.flag_only"],
            [{"subsystem_name": "sub_a", "flags": 4}],
        ),
        case(
            "case_07_single_mixed_scalar_flag",
            "1 contributor (mixed scalar plus flags)",
            8,
            ["one.mixed_scalar_flag"],
            [
                {
                    "subsystem_name": "sub_a",
                    "confidence_delta": 0.125,
                    "lr_modulation": 1.1,
                    "flags": 3,
                }
            ],
        ),
        case(
            "case_08_two_identical_scalar",
            "2 contributors (identical confidence deltas)",
            9,
            ["two.identical_scalar"],
            [
                {"subsystem_name": "sub_a", "confidence_delta": 0.25},
                {"subsystem_name": "sub_b", "confidence_delta": 0.25},
            ],
        ),
        case(
            "case_09_two_unequal_scalar",
            "2 contributors (unequal confidence deltas)",
            10,
            ["two.unequal_scalar"],
            [
                {"subsystem_name": "sub_a", "confidence_delta": 0.25},
                {"subsystem_name": "sub_b", "confidence_delta": -0.25},
            ],
        ),
        case(
            "case_10_two_shared_flag",
            "2 contributors (same shared flag)",
            11,
            ["two.shared_flag"],
            [
                {"subsystem_name": "sub_a", "flags": 4},
                {"subsystem_name": "sub_b", "flags": 4},
            ],
        ),
        case(
            "case_11_two_unique_flag",
            "2 contributors (one unique flag plus independent scalar contributor)",
            12,
            ["two.unique_flag"],
            [
                {"subsystem_name": "sub_a", "flags": 1},
                {"subsystem_name": "sub_b", "confidence_delta": 0.1},
            ],
        ),
        case(
            "case_12_two_mixed_flags",
            "2 contributors (overlapping and unique flags)",
            13,
            ["two.mixed_shared_unique_flags"],
            [
                {"subsystem_name": "sub_a", "flags": 5},
                {"subsystem_name": "sub_b", "flags": 4},
            ],
        ),
        case(
            "case_13_two_lr_geometric_mean",
            "2 contributors (LR geometric mean)",
            14,
            ["two.lr_geometric_mean"],
            [
                {"subsystem_name": "sub_a", "lr_modulation": 1.2},
                {"subsystem_name": "sub_b", "lr_modulation": 0.8},
            ],
        ),
        case(
            "case_14_two_mixed_f64_f32",
            "2 contributors (mixed f64 and f32 channels)",
            15,
            ["two.mixed_f64_f32"],
            [
                {
                    "subsystem_name": "sub_a",
                    "confidence_delta": 0.125,
                    "arousal_delta": 0.25,
                },
                {
                    "subsystem_name": "sub_b",
                    "exploration_delta": -0.5,
                    "valence_delta": -0.125,
                },
            ],
        ),
        case(
            "case_15_n_contributors_multi",
            "4 contributors (mixed all channels and overlapping flags)",
            16,
            ["n.mixed_all_channels"],
            [
                {
                    "subsystem_name": "manager_alpha",
                    "confidence_delta": 0.1,
                    "lr_modulation": 1.2,
                    "arousal_delta": 0.05,
                    "flags": 5,
                },
                {
                    "subsystem_name": "manager_beta",
                    "confidence_delta": 0.2,
                    "lr_modulation": 0.8,
                    "valence_delta": -0.1,
                    "flags": 4,
                },
                {
                    "subsystem_name": "manager_gamma",
                    "confidence_delta": 0.1,
                    "lr_modulation": 1.2,
                    "exploration_delta": 0.15,
                    "flags": 2,
                },
                {
                    "subsystem_name": "manager_delta",
                    "confidence_delta": -0.4,
                    "lr_modulation": 1.0,
                    "arousal_delta": -0.025,
                    "valence_delta": 0.05,
                    "flags": 1,
                },
            ],
        ),
    ]

    fixture_data = []
    for c in cases:
        proposals = [Proposal.from_json(p) for p in c["proposals"]]
        report = influence_report(c["cycle_number"], proposals)
        integrated_all = integrate(proposals)
        fixture_data.append(
            {
                "name": c["name"],
                "description": c["description"],
                "cycle_number": c["cycle_number"],
                "coverage_tags": c["coverage_tags"],
                "input_proposals": c["proposals"],
                "expected_integrated": {
                    "confidence_delta": integrated_all.confidence_delta,
                    "lr_modulation": integrated_all.lr_modulation,
                    "exploration_delta": integrated_all.exploration_delta,
                    "arousal_delta": integrated_all.arousal_delta,
                    "valence_delta": integrated_all.valence_delta,
                    "flags": integrated_all.flags,
                    "n_contributors": integrated_all.n_contributors,
                    "exact_bits": integrated_all.exact_bits(),
                },
                "expected_report": report,
            }
        )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(fixture_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(fixture_data)} golden fixtures at {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
