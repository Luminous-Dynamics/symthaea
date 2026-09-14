#!/usr/bin/env python3
"""Generate SPINE-000B golden fixtures for Rust ↔ Python oracle equivalence verification (Issue #3036).
"""

import json
from pathlib import Path
from spine_000b_influence_oracle import Proposal, influence_report, integrate

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "spine_000b_golden_fixtures.json"

def main():
    cases = [
        {
            "name": "case_00_empty",
            "description": "0 contributors (empty collector)",
            "cycle_number": 1,
            "proposals": [],
        },
        {
            "name": "case_01_single_confidence",
            "description": "1 contributor (confidence delta only)",
            "cycle_number": 2,
            "proposals": [
                {"subsystem_name": "sub_a", "confidence_delta": 0.25}
            ],
        },
        {
            "name": "case_02_single_lr",
            "description": "1 contributor (LR modulation only)",
            "cycle_number": 3,
            "proposals": [
                {"subsystem_name": "sub_a", "lr_modulation": 1.5}
            ],
        },
        {
            "name": "case_03_single_flag",
            "description": "1 contributor (flag only)",
            "cycle_number": 4,
            "proposals": [
                {"subsystem_name": "sub_a", "flags": 4}
            ],
        },
        {
            "name": "case_04_two_equal_confidence",
            "description": "2 contributors (equal confidence delta)",
            "cycle_number": 5,
            "proposals": [
                {"subsystem_name": "sub_a", "confidence_delta": 0.25},
                {"subsystem_name": "sub_b", "confidence_delta": 0.25},
            ],
        },
        {
            "name": "case_05_two_unequal_confidence",
            "description": "2 contributors (unequal confidence delta)",
            "cycle_number": 6,
            "proposals": [
                {"subsystem_name": "sub_a", "confidence_delta": 0.25},
                {"subsystem_name": "sub_b", "confidence_delta": -0.25},
            ],
        },
        {
            "name": "case_06_two_mixed_flags",
            "description": "2 contributors (overlapping and unique flags)",
            "cycle_number": 7,
            "proposals": [
                {"subsystem_name": "sub_a", "flags": 5}, # 0x1 | 0x4
                {"subsystem_name": "sub_b", "flags": 4}, # 0x4
            ],
        },
        {
            "name": "case_07_n_contributors_multi",
            "description": "4 contributors (multi-subsystem mixed signals)",
            "cycle_number": 8,
            "proposals": [
                {"subsystem_name": "manager_alpha", "confidence_delta": 0.1, "lr_modulation": 1.2, "arousal_delta": 0.05, "flags": 1},
                {"subsystem_name": "manager_beta", "confidence_delta": 0.2, "lr_modulation": 0.8, "valence_delta": -0.1, "flags": 2},
                {"subsystem_name": "manager_gamma", "confidence_delta": 0.1, "lr_modulation": 1.2, "exploration_delta": 0.15, "flags": 4},
                {"subsystem_name": "manager_delta", "confidence_delta": -0.4, "lr_modulation": 1.0, "flags": 0},
            ],
        },
    ]

    fixture_data = []
    for c in cases:
        proposals = [Proposal.from_json(p) for p in c["proposals"]]
        report = influence_report(c["cycle_number"], proposals)

        # Also store integrated_all struct details directly
        integrated_all = integrate(proposals)
        entry = {
            "name": c["name"],
            "description": c["description"],
            "cycle_number": c["cycle_number"],
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
        fixture_data.append(entry)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(fixture_data)} golden fixtures at {FIXTURE_PATH}")

if __name__ == "__main__":
    main()
