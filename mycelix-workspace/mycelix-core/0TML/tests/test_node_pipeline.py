# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import pytest

from zerotrustml.core.node import Node, NodeConfig


@pytest.mark.asyncio
async def test_node_training_updates_accuracy():
    config = NodeConfig(
        node_id="hospital-1",
        data_path="/tmp",
        aggregation="krum",
        batch_size=16,
        learning_rate=0.01,
    )

    node = Node(config)
    await node.start()

    results = await node.train(rounds=2)

    assert node.rounds_completed == 2
    assert len(results) == 2
    # Accuracy should be a valid percentage in [0, 1]
    assert 0.0 <= node.accuracy <= 1.0
    assert all("accuracy" in round_info for round_info in results)
