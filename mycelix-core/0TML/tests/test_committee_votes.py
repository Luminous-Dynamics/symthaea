#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Quick tests for committee voting helper."""

from zerotrustml.experimental.edge_validation import CommitteeVote, aggregate_committee_votes


def test_aggregate_committee_votes_majority_accepts():
    votes = [
        CommitteeVote("validator-1", "proof", 0.8, True, "", None),
        CommitteeVote("validator-2", "proof", 0.7, True, "", None),
        CommitteeVote("validator-3", "proof", 0.3, False, "", None),
    ]
    score, accepted = aggregate_committee_votes(votes)
    assert accepted is True
    assert 0.7 <= score <= 0.8


def test_aggregate_committee_votes_rejects():
    votes = [
        CommitteeVote("validator-1", "proof", 0.2, False, "", None),
        CommitteeVote("validator-2", "proof", 0.1, False, "", None),
        CommitteeVote("validator-3", "proof", 0.9, True, "", None),
    ]
    score, accepted = aggregate_committee_votes(votes)
    assert accepted is False
    assert score == 0.9
