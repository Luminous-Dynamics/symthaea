#!/usr/bin/env python3
from __future__ import annotations

import unittest

import workbench_root_nar_membership as nar


class NarTargetSpellingContracts(unittest.TestCase):
    def test_duplicate_separator_rejected(self):
        with self.assertRaises(nar.NarError):
            nar.canonical_target("bin//wb_command")

    def test_normalized_dot_component_rejected(self):
        with self.assertRaises(nar.NarError):
            nar.canonical_target("bin/./wb_command")

    def test_trailing_separator_rejected(self):
        with self.assertRaises(nar.NarError):
            nar.canonical_target("bin/wb_command/")


if __name__ == "__main__":
    unittest.main(verbosity=2)
