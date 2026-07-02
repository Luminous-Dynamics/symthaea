# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
from zerotrustml.hybrid_utils import combine_detection_votes


def test_hybrid_score_overrides_base_flag():
    assert combine_detection_votes(False, 0.7, 0.5) is True


def test_hybrid_score_respects_threshold():
    assert combine_detection_votes(False, 0.2, 0.5) is False
    assert combine_detection_votes(True, 0.2, 0.5) is True


def test_no_hybrid_score_returns_base_flag():
    assert combine_detection_votes(False, None, 0.4) is False
