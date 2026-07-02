# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import numpy as np
import pytest

import zerotrustml.core.phase10_coordinator as coordinator_module
from zerotrustml.core.phase10_coordinator import Phase10Coordinator, Phase10Config


def test_krum_aggregation_uses_shared_module(monkeypatch):
    called = {}

    def fake_aggregate_gradients(gradients, reputations, algorithm, num_byzantine):
        called["args"] = (gradients, reputations, algorithm, num_byzantine)
        return gradients[0]

    monkeypatch.setattr(coordinator_module, "aggregate_gradients", fake_aggregate_gradients)

    config = Phase10Config(
        postgres_enabled=False,
        holochain_enabled=False,
        localfile_enabled=False,
        aggregation_algorithm="krum",
    )
    coordinator = Phase10Coordinator(config)

    aggregated, byzantine = coordinator._krum_aggregation([[0.0, 1.0], [1.0, 2.0]])

    assert "args" in called
    assert called["args"][2] == "krum"
    assert isinstance(aggregated, np.ndarray)
    assert isinstance(byzantine, int)
