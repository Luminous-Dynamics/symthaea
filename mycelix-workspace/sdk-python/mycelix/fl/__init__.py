"""
Federated Learning Module

Byzantine-resistant distributed machine learning with MATL integration.

Aggregation methods:
    - FedAvg: Standard federated averaging
    - TrimmedMean: Excludes outliers before averaging
    - CoordinateMedian: Robust to up to 50% Byzantine
    - Krum: Selects gradient closest to neighbors
    - TrustWeighted: MATL-integrated trust weighting

Example:
    >>> from mycelix import fl
    >>> import numpy as np
    >>>
    >>> coordinator = fl.FLCoordinator(min_participants=3)
    >>> coordinator.register_participant('node_1')
    >>> coordinator.register_participant('node_2')
    >>> coordinator.register_participant('node_3')
    >>>
    >>> coordinator.start_round()
    >>> for node in ['node_1', 'node_2', 'node_3']:
    ...     coordinator.submit_update(fl.GradientUpdate(
    ...         participant_id=node,
    ...         model_version=1,
    ...         gradients=np.random.randn(10),
    ...         batch_size=32,
    ...         loss=0.5,
    ...     ))
    >>> coordinator.aggregate_round()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import numpy as np

from mycelix import matl


# ============================================================================
# Enums
# ============================================================================


class AggregationMethod(str, Enum):
    """Aggregation methods for FL."""

    FEDAVG = "fedavg"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"
    TRUST_WEIGHTED = "trust_weighted"


# ============================================================================
# Types
# ============================================================================


@dataclass
class GradientUpdate:
    """Gradient update from a participant."""

    participant_id: str
    model_version: int
    gradients: np.ndarray
    batch_size: int
    loss: float
    accuracy: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregatedGradient:
    """Result of gradient aggregation."""

    gradients: np.ndarray
    participant_count: int
    excluded_count: int
    aggregation_method: AggregationMethod
    timestamp: float = field(default_factory=time.time)


@dataclass
class Participant:
    """FL participant state."""

    id: str
    reputation: matl.ReputationScore
    pogq: Optional[matl.ProofOfGradientQuality] = None
    last_contribution: Optional[GradientUpdate] = None
    rounds_participated: int = 0


@dataclass
class FLRound:
    """State of an FL round."""

    round_id: int
    model_version: int
    participants: dict[str, Participant]
    updates: list[GradientUpdate]
    status: str  # 'collecting', 'aggregating', 'completed'
    start_time: float
    end_time: Optional[float] = None
    aggregated_result: Optional[AggregatedGradient] = None
    excluded_ids: list[str] = field(default_factory=list)


@dataclass
class FLConfig:
    """FL coordinator configuration."""

    min_participants: int = 3
    max_participants: int = 100
    round_timeout: float = 60.0  # seconds
    byzantine_tolerance: float = 0.34  # Validated maximum (was 0.33)
    aggregation_method: AggregationMethod = AggregationMethod.TRUST_WEIGHTED
    trust_threshold: float = 0.5


# ============================================================================
# Aggregation Algorithms
# ============================================================================


def fed_avg(updates: list[GradientUpdate]) -> np.ndarray:
    """
    Federated Averaging - weighted average by batch size.

    Args:
        updates: List of gradient updates

    Returns:
        Aggregated gradients
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    total_samples = sum(u.batch_size for u in updates)
    gradient_size = len(updates[0].gradients)
    result = np.zeros(gradient_size)

    for update in updates:
        weight = update.batch_size / total_samples
        result += update.gradients * weight

    return result


def trimmed_mean(
    updates: list[GradientUpdate], trim_percentage: float = 0.1
) -> np.ndarray:
    """
    Trimmed Mean - excludes outliers before averaging.

    Args:
        updates: List of gradient updates
        trim_percentage: Fraction to trim from each end

    Returns:
        Aggregated gradients
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    gradient_size = len(updates[0].gradients)
    result = np.zeros(gradient_size)
    trim_count = int(len(updates) * trim_percentage)

    for i in range(gradient_size):
        values = sorted([u.gradients[i] for u in updates])
        trimmed = values[trim_count : len(values) - trim_count]
        if trimmed:
            result[i] = sum(trimmed) / len(trimmed)

    return result


def coordinate_median(updates: list[GradientUpdate]) -> np.ndarray:
    """
    Coordinate-wise Median - robust to up to 50% Byzantine.

    Args:
        updates: List of gradient updates

    Returns:
        Aggregated gradients
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    gradient_size = len(updates[0].gradients)
    result = np.zeros(gradient_size)

    for i in range(gradient_size):
        values = [u.gradients[i] for u in updates]
        result[i] = np.median(values)

    return result


def krum(updates: list[GradientUpdate], num_select: int = 1) -> np.ndarray:
    """
    Krum - selects gradient closest to neighbors.

    Args:
        updates: List of gradient updates
        num_select: Number of updates to select

    Returns:
        Aggregated gradients
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    n = len(updates)
    num_neighbors = n - 2

    # Calculate pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.linalg.norm(
                    updates[i].gradients - updates[j].gradients
                )

    # Calculate Krum scores
    scores = []
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        score = sum(sorted_distances[1 : num_neighbors + 1])  # Skip self (0)
        scores.append((i, score))

    # Select updates with lowest scores
    scores.sort(key=lambda x: x[1])
    selected = [updates[scores[i][0]] for i in range(min(num_select, len(scores)))]

    return fed_avg(selected)


def trust_weighted_aggregation(
    updates: list[GradientUpdate],
    participants: dict[str, Participant],
    trust_threshold: float = 0.5,
) -> AggregatedGradient:
    """
    Trust-Weighted Aggregation with MATL integration.

    Args:
        updates: List of gradient updates
        participants: Participant registry
        trust_threshold: Minimum trust to include

    Returns:
        AggregatedGradient with trust weighting
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    gradient_size = len(updates[0].gradients)
    result = np.zeros(gradient_size)
    total_weight = 0.0
    excluded_count = 0
    included_count = 0

    # Calculate trust weights
    weights: dict[str, float] = {}

    for update in updates:
        participant = participants.get(update.participant_id)
        if not participant:
            excluded_count += 1
            continue

        rep_value = matl.reputation_value(participant.reputation)
        if rep_value < trust_threshold:
            excluded_count += 1
            continue

        # Calculate weight from reputation and PoGQ
        weight = rep_value
        if participant.pogq:
            composite = matl.calculate_composite(participant.pogq, participant.reputation)
            weight = composite.final_score

        weight *= update.batch_size
        weights[update.participant_id] = weight
        total_weight += weight
        included_count += 1

    # Aggregate with trust weights
    if total_weight > 0:
        for update in updates:
            weight = weights.get(update.participant_id)
            if weight is not None:
                normalized_weight = weight / total_weight
                result += update.gradients * normalized_weight

    return AggregatedGradient(
        gradients=result,
        participant_count=included_count,
        excluded_count=excluded_count,
        aggregation_method=AggregationMethod.TRUST_WEIGHTED,
    )


# ============================================================================
# FLCoordinator
# ============================================================================


class FLCoordinator:
    """
    Federated Learning Coordinator.

    Manages FL rounds and participant interactions.
    """

    def __init__(
        self,
        min_participants: int = 3,
        max_participants: int = 100,
        trust_threshold: float = 0.5,
        aggregation_method: AggregationMethod = AggregationMethod.TRUST_WEIGHTED,
    ) -> None:
        self.config = FLConfig(
            min_participants=min_participants,
            max_participants=max_participants,
            trust_threshold=trust_threshold,
            aggregation_method=aggregation_method,
        )
        self._participants: dict[str, Participant] = {}
        self._current_round: Optional[FLRound] = None
        self._round_history: list[FLRound] = []
        self._model_version = 0

    def register_participant(self, participant_id: str) -> Participant:
        """Register a new participant."""
        participant = Participant(
            id=participant_id,
            reputation=matl.create_reputation(participant_id),
        )
        self._participants[participant_id] = participant
        return participant

    def start_round(self) -> FLRound:
        """Start a new FL round."""
        if self._current_round and self._current_round.status != "completed":
            raise RuntimeError("Previous round not completed")

        self._model_version += 1
        self._current_round = FLRound(
            round_id=len(self._round_history) + 1,
            model_version=self._model_version,
            participants=dict(self._participants),
            updates=[],
            status="collecting",
            start_time=time.time(),
        )
        return self._current_round

    def submit_update(self, update: GradientUpdate) -> bool:
        """Submit a gradient update."""
        if not self._current_round or self._current_round.status != "collecting":
            return False

        if update.model_version != self._current_round.model_version:
            return False

        participant = self._participants.get(update.participant_id)
        if not participant:
            return False

        rep_value = matl.reputation_value(participant.reputation)
        if rep_value < self.config.trust_threshold:
            return False

        # Create PoGQ for this update
        quality = self._calculate_update_quality(update)
        participant.pogq = matl.create_pogq(quality, 0.8, 0.1)
        participant.last_contribution = update

        self._current_round.updates.append(update)

        if len(self._current_round.updates) >= self.config.max_participants:
            return self.aggregate_round()

        return True

    def aggregate_round(self) -> bool:
        """Aggregate the current round."""
        if not self._current_round or self._current_round.status != "collecting":
            return False

        if len(self._current_round.updates) < self.config.min_participants:
            return False

        self._current_round.status = "aggregating"

        # Pre-filter Byzantine updates using norm-based z-score detection
        filtered = detect_and_filter_byzantine(
            self._current_round.updates,
            self.config.byzantine_tolerance,
        )
        excluded_count = len(self._current_round.updates) - len(filtered)

        if len(filtered) < self.config.min_participants:
            self._current_round.status = "collecting"
            return False

        # Track which participant IDs survived filtering
        filtered_ids = {u.participant_id for u in filtered}
        self._current_round.excluded_ids = [
            u.participant_id
            for u in self._current_round.updates
            if u.participant_id not in filtered_ids
        ]

        # Perform aggregation on filtered updates
        method = self.config.aggregation_method
        if method == AggregationMethod.FEDAVG:
            gradients = fed_avg(filtered)
            result = AggregatedGradient(
                gradients=gradients,
                participant_count=len(filtered),
                excluded_count=excluded_count,
                aggregation_method=method,
            )
        elif method == AggregationMethod.TRIMMED_MEAN:
            trim_pct = max(0.1, self.config.byzantine_tolerance)
            gradients = trimmed_mean(filtered, trim_pct)
            result = AggregatedGradient(
                gradients=gradients,
                participant_count=len(filtered),
                excluded_count=excluded_count,
                aggregation_method=method,
            )
        elif method == AggregationMethod.MEDIAN:
            gradients = coordinate_median(filtered)
            result = AggregatedGradient(
                gradients=gradients,
                participant_count=len(filtered),
                excluded_count=excluded_count,
                aggregation_method=method,
            )
        elif method == AggregationMethod.KRUM:
            gradients = krum(filtered)
            result = AggregatedGradient(
                gradients=gradients,
                participant_count=len(filtered),
                excluded_count=excluded_count,
                aggregation_method=method,
            )
        else:  # TRUST_WEIGHTED
            result = trust_weighted_aggregation(
                filtered,
                self._participants,
                self.config.trust_threshold,
            )
            result.excluded_count += excluded_count

        # Store the aggregated result
        self._current_round.aggregated_result = result

        # Update reputations (only for participants included in aggregation)
        self._update_reputations(filtered_ids)

        self._current_round.status = "completed"
        self._current_round.end_time = time.time()
        self._round_history.append(self._current_round)

        return True

    def get_round_stats(self) -> dict:
        """Get statistics about FL rounds."""
        total_participation = sum(len(r.updates) for r in self._round_history)
        avg_participation = (
            total_participation / len(self._round_history)
            if self._round_history
            else 0
        )

        return {
            "total_rounds": len(self._round_history),
            "current_round": self._current_round,
            "participant_count": len(self._participants),
            "average_participation": avg_participation,
        }

    def _calculate_update_quality(self, update: GradientUpdate) -> float:
        """Calculate quality score for an update."""
        loss_quality = np.exp(-update.loss)
        grad_mag = np.linalg.norm(update.gradients)
        mag_quality = 1.0 / (1.0 + np.log1p(grad_mag))
        return (loss_quality + mag_quality) / 2

    def _update_reputations(self, included_ids: set[str]) -> None:
        """Update participant reputations after round.

        Only participants whose gradients were included in aggregation
        receive positive reputation updates. Excluded (Byzantine) participants
        are not penalized here — that's handled by the MATL decay system.
        """
        if not self._current_round:
            return

        for update in self._current_round.updates:
            if update.participant_id not in included_ids:
                continue
            participant = self._participants.get(update.participant_id)
            if participant:
                participant.reputation = matl.record_positive(participant.reputation)
                participant.rounds_participated += 1


# ============================================================================
# Byzantine Detection
# ============================================================================


def detect_byzantine(
    updates: list[GradientUpdate], z_threshold: float = 3.0
) -> list[int]:
    """
    Detect Byzantine updates using gradient norm z-score analysis.

    Args:
        updates: List of gradient updates
        z_threshold: Z-score threshold for outlier detection

    Returns:
        List of indices flagged as Byzantine
    """
    if len(updates) < 3:
        return []

    norms = np.array([np.linalg.norm(u.gradients) for u in updates])
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    if std_norm < 1e-10:
        return []

    flagged = []
    for i, norm in enumerate(norms):
        z_score = abs(norm - mean_norm) / std_norm
        if z_score > z_threshold:
            flagged.append(i)

    return flagged


def detect_and_filter_byzantine(
    updates: list[GradientUpdate],
    byzantine_tolerance: float,
    z_threshold: float = 3.0,
) -> list[GradientUpdate]:
    """
    Detect and remove Byzantine updates, respecting the tolerance limit.

    Args:
        updates: List of gradient updates
        byzantine_tolerance: Maximum fraction of updates to exclude (0.34)
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Filtered list of gradient updates
    """
    if len(updates) < 3:
        return updates

    flagged = detect_byzantine(updates, z_threshold)

    if not flagged:
        return updates

    # Sort flagged indices by severity (highest norm deviation first)
    norms = np.array([np.linalg.norm(u.gradients) for u in updates])
    mean_norm = np.mean(norms)
    flagged.sort(key=lambda i: abs(norms[i] - mean_norm), reverse=True)

    # Don't exclude more than byzantine_tolerance fraction
    max_exclude = int(len(updates) * byzantine_tolerance)
    flagged = flagged[:max_exclude]

    flagged_set = set(flagged)
    return [u for i, u in enumerate(updates) if i not in flagged_set]


# ============================================================================
# Utilities
# ============================================================================


def serialize_gradients(gradients: np.ndarray) -> bytes:
    """Serialize gradients for transmission."""
    return gradients.astype(np.float64).tobytes()


def deserialize_gradients(data: bytes) -> np.ndarray:
    """Deserialize gradients from transmission."""
    return np.frombuffer(data, dtype=np.float64)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "AggregationMethod",
    # Types
    "GradientUpdate",
    "AggregatedGradient",
    "Participant",
    "FLRound",
    "FLConfig",
    # Aggregation functions
    "fed_avg",
    "trimmed_mean",
    "coordinate_median",
    "krum",
    "trust_weighted_aggregation",
    # Coordinator
    "FLCoordinator",
    # Byzantine detection
    "detect_byzantine",
    "detect_and_filter_byzantine",
    # Utilities
    "serialize_gradients",
    "deserialize_gradients",
]
