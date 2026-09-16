#!/usr/bin/env python3
from __future__ import annotations
import pathlib, subprocess, unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts' / 'qualify-assure-linux-ima-rust196-v3.sh'

class QualificationV3RecipeTests(unittest.TestCase):
    def test_bash_syntax(self) -> None:
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)

    def test_required_fail_closed_controls_are_present(self) -> None:
        text = SCRIPT.read_text(encoding='utf-8')
        required = [
            'git show "$PRODUCT_HEAD:$path"',
            'EXPECTED_PRODUCT_BLOB',
            'EXPECTED_PRODUCT_SHA256',
            'cargo fetch --locked',
            'export CARGO_NET_OFFLINE=true',
            'cargo metadata --locked --offline',
            'cargo check --locked --offline',
            'checked_in_golden_vector_replays_to_frozen_sha256_pcr',
            'cargo test --locked --offline',
            '--all-targets -- -D warnings',
            'qualification_result=PASS',
            'authoritative_cargo_network=OFFLINE',
        ]
        for needle in required:
            self.assertIn(needle, text)

    def test_pass_is_emitted_only_after_all_authoritative_gates(self) -> None:
        text = SCRIPT.read_text(encoding='utf-8')
        pass_index = text.index('qualification_result=PASS')
        for gate in [
            'stage offline_check',
            'stage offline_golden_replay',
            'stage offline_full_tests',
            'stage offline_strict_clippy_all_targets',
        ]:
            self.assertLess(text.index(gate), pass_index)

if __name__ == '__main__':
    unittest.main(verbosity=2)
