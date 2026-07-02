import numpy as np

from zerotrustml.aggregation import aggregate_gradients
from zerotrustml.core.phase10_coordinator import Phase10Config, Phase10Coordinator


def test_hierarchical_krum_matches_krum_for_small_n():
    gradients = [
        np.array([0.1, 0.2]),
        np.array([0.12, 0.18]),
        np.array([0.11, 0.19]),
        np.array([0.09, 0.21]),
    ]
    reputations = [1.0] * len(gradients)

    direct = aggregate_gradients(
        gradients,
        reputations,
        algorithm="krum",
        num_byzantine=1,
    )

    hierarchical = aggregate_gradients(
        gradients,
        reputations,
        algorithm="hierarchical_krum",
        branching_factor=10,
        num_byzantine=1,
    )

    assert direct.shape == hierarchical.shape
    assert np.allclose(direct, hierarchical)


def test_hierarchical_krum_handles_many_gradients():
    honest_gradients = [np.array([0.1, 0.2])] * 90
    byzantine_gradients = [np.array([5.0, -3.0])] * 10
    all_gradients = honest_gradients + byzantine_gradients
    reputations = [1.0] * len(all_gradients)

    aggregated = aggregate_gradients(
        all_gradients,
        reputations,
        algorithm="hierarchical_krum",
        branching_factor=10,
    )

    assert aggregated.shape == (2,)
    assert 0.09 <= aggregated[0] <= 0.11
    assert 0.19 <= aggregated[1] <= 0.21


def test_hierarchical_krum_in_phase10_coordinator():
    config = Phase10Config(
        postgres_enabled=False,
        holochain_enabled=False,
        localfile_enabled=False,
        aggregation_algorithm="hierarchical_krum",
    )
    coordinator = Phase10Coordinator(config)

    gradients = [
        [0.1, 0.2],
        [0.12, 0.18],
        [0.11, 0.19],
        [0.09, 0.21],
    ]

    aggregated, byzantine_count = coordinator._krum_aggregation(gradients)

    assert isinstance(byzantine_count, int)
    assert aggregated.shape == (2,)
    assert 0.09 <= aggregated[0] <= 0.12
    assert 0.18 <= aggregated[1] <= 0.21

