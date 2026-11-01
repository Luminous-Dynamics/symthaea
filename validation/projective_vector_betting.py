"""Projective vector betting under a declared conditional mgf envelope.

Research validation helper. The code checks matrix shape/symmetry/PSD, but
cannot prove that a supplied matrix dominates the unknown conditional mgf.
That coverage property is an explicit statistical assumption/certificate.
"""
from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Iterable, Sequence


def _finite_vector(values: Iterable[float], name: str) -> list[float]:
    result = [float(x) for x in values]
    if any(not isfinite(x) for x in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def assert_symmetric_psd(matrix: Sequence[Sequence[float]], tol: float = 1e-10) -> None:
    """Numerically screen symmetry/PSD using scale-normalized LDL^T.

    This is a floating-point validation helper, not an exact PSD certificate.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    raw = [[float(x) for x in row] for row in matrix]
    if any(not isfinite(x) for row in raw for x in row):
        raise ValueError("matrix entries must be finite")
    scale = max((abs(x) for row in raw for x in row), default=0.0)
    if scale == 0.0:
        return
    # Normalize so the same tolerance works for tiny and very large matrices.
    a = [[x / scale for x in row] for row in raw]
    for i in range(n):
        for j in range(i):
            if abs(a[i][j] - a[j][i]) > tol:
                raise ValueError("matrix must be symmetric")
    lower = [[0.0] * n for _ in range(n)]
    diagonal = [0.0] * n
    for j in range(n):
        pivot = a[j][j] - sum(
            lower[j][k] * lower[j][k] * diagonal[k] for k in range(j)
        )
        if pivot < -tol:
            raise ValueError("matrix is not positive semidefinite")
        if pivot <= tol:
            diagonal[j] = 0.0
            for i in range(j + 1, n):
                residual = a[i][j] - sum(
                    lower[i][k] * lower[j][k] * diagonal[k] for k in range(j)
                )
                if abs(residual) > tol:
                    raise ValueError("zero PSD pivot has nonzero residual")
                lower[i][j] = 0.0
        else:
            diagonal[j] = pivot
            for i in range(j + 1, n):
                residual = a[i][j] - sum(
                    lower[i][k] * lower[j][k] * diagonal[k] for k in range(j)
                )
                lower[i][j] = residual / pivot


def equicorrelation_covariance(dimension: int, rho: float) -> list[list[float]]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not (-1.0 / max(1, dimension - 1) <= rho <= 1.0):
        raise ValueError("rho is outside the equicorrelation PSD range")
    return [
        [1.0 if i == j else float(rho) for j in range(dimension)]
        for i in range(dimension)
    ]


def equicorrelation_envelope(
    dimension: int, rho_min: float, rho_max: float, reference_rho: float
) -> list[list[float]]:
    """Return V = Sigma(reference) + delta I dominating every Sigma(rho).

    For dimension d, the eigenvalues of Sigma(rho)-Sigma(r0) are
    (rho-r0)(d-1) on span(1) and -(rho-r0) on its orthogonal complement.
    Hence delta=max((d-1)(rho_max-r0), r0-rho_min, 0).
    """
    if rho_min > reference_rho or reference_rho > rho_max:
        raise ValueError("reference_rho must lie inside [rho_min, rho_max]")
    base = equicorrelation_covariance(dimension, reference_rho)
    delta = max(
        0.0,
        (dimension - 1) * (rho_max - reference_rho),
        reference_rho - rho_min,
    )
    result = [
        [base[i][j] + (delta if i == j else 0.0) for j in range(dimension)]
        for i in range(dimension)
    ]
    assert_symmetric_psd(result)
    return result


def projective_equicorrelation_envelope(
    dimension: int, rho_min: float = 0.2, rho_max: float = 0.8
) -> list[list[float]]:
    """A single kernel whose finite principal submatrices dominate all rho.

    V_d = (1-rho_min) I_d + rho_max 11^T. For every rho in
    [rho_min,rho_max],

      V_d - Sigma_d(rho)
        = (rho-rho_min) I_d + (rho_max-rho) 11^T,

    which is PSD because both coefficients are nonnegative. Unlike the
    tighter dimension-specific envelope above, this family is projectively
    consistent: every principal submatrix is the same formula at the smaller
    dimension. This form requires 0 <= rho_min <= rho_max <= 1.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not (0.0 <= rho_min <= rho_max <= 1.0):
        raise ValueError("projective envelope requires 0 <= rho_min <= rho_max <= 1")
    diagonal = 1.0 - rho_min + rho_max
    result = [
        [diagonal if i == j else rho_max for j in range(dimension)]
        for i in range(dimension)
    ]
    assert_symmetric_psd(result)
    return result


def matrix_dominates(upper: Sequence[Sequence[float]], lower: Sequence[Sequence[float]]) -> bool:
    if len(upper) != len(lower):
        return False
    difference = [
        [float(upper[i][j]) - float(lower[i][j]) for j in range(len(upper))]
        for i in range(len(upper))
    ]
    try:
        assert_symmetric_psd(difference, tol=1e-8)
    except ValueError:
        return False
    return True


def coordinate_local_bets(
    histories: Sequence[Sequence[float]],
    threshold: float = 1.5,
    cap: float = 1.5,
) -> list[float]:
    """Choose nonnegative bets from each coordinate's own pre-update history.

    Caller must invoke this before exposing the current-round scores. Empty
    histories produce zero bets. The threshold/cap change power, not validity.
    """
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    bets: list[float] = []
    for history in histories:
        past = _finite_vector(history, "history")
        if not past:
            bets.append(0.0)
            continue
        mean = sum(past) / len(past)
        bet = max(0.0, mean - threshold / sqrt(len(past)))
        bets.append(min(cap, bet))
    return bets


def vector_log_factor(
    scores: Sequence[float],
    bets: Sequence[float],
    active: Sequence[bool],
    mgf_proxy: Sequence[Sequence[float]],
) -> float:
    """Compute lambda_A^T Z_A - 1/2 lambda_A^T V_A lambda_A.

    V_A is the principal submatrix for active intersection coordinates.
    An empty/dormant intersection contributes log-factor zero. The caller must
    establish the conditional mgf domination assumption for mgf_proxy.
    """
    z = _finite_vector(scores, "scores")
    lam = _finite_vector(bets, "bets")
    if len(z) != len(lam) or len(z) != len(active):
        raise ValueError("scores, bets and active must have equal lengths")
    d = len(z)
    if len(mgf_proxy) != d or any(len(row) != d for row in mgf_proxy):
        raise ValueError("mgf_proxy dimension mismatch")
    if any(x < 0 for x in lam):
        raise ValueError("bets must be nonnegative")
    assert_symmetric_psd(mgf_proxy)
    ids = [i for i, on in enumerate(active) if on]
    if not ids:
        return 0.0
    linear = sum(lam[i] * z[i] for i in ids)
    quadratic = sum(
        lam[i] * float(mgf_proxy[i][j]) * lam[j]
        for i in ids for j in ids
    )
    return linear - 0.5 * quadratic


def gaussian_log_factor_expectation(
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    bets: Sequence[float],
    active: Sequence[bool],
    mgf_proxy: Sequence[Sequence[float]],
) -> float:
    """Analytic log E[factor] for a Gaussian score vector."""
    mu = _finite_vector(mean, "mean")
    if not matrix_dominates(mgf_proxy, covariance):
        raise ValueError("mgf_proxy does not dominate covariance")
    ids = [i for i, on in enumerate(active) if on]
    if not ids:
        return 0.0
    linear = sum(float(bets[i]) * mu[i] for i in ids)
    variance_true = sum(
        float(bets[i]) * float(covariance[i][j]) * float(bets[j])
        for i in ids for j in ids
    )
    variance_proxy = sum(
        float(bets[i]) * float(mgf_proxy[i][j]) * float(bets[j])
        for i in ids for j in ids
    )
    return linear + 0.5 * (variance_true - variance_proxy)


def fixed_log_mixture(log_capitals: Sequence[float], prior: Sequence[float]) -> float:
    """Stable fixed-prior mixture of valid e-process capitals (not their max)."""
    if len(log_capitals) == 0 or len(log_capitals) != len(prior):
        raise ValueError("log_capitals and prior must have equal nonzero length")
    weights = _finite_vector(prior, "prior")
    if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("prior weights must be nonnegative and sum to one")
    active_terms = [
        log(w) + float(x)
        for x, w in zip(log_capitals, weights)
        if w > 0
    ]
    if not active_terms:
        raise ValueError("at least one prior weight must be positive")
    top = max(active_terms)
    return top + log(sum(exp(x - top) for x in active_terms))


class CoordinateHistory:
    """Minimal state helper: read bets before appending the current score vector."""
    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._history: list[list[float]] = [[] for _ in range(dimension)]

    @property
    def dimension(self) -> int:
        return len(self._history)

    def bets(self, threshold: float = 1.5, cap: float = 1.5) -> list[float]:
        return coordinate_local_bets(self._history, threshold, cap)

    def observe(self, scores: Sequence[float], active: Sequence[bool]) -> None:
        z = _finite_vector(scores, "scores")
        if len(z) != self.dimension or len(active) != self.dimension:
            raise ValueError("observation dimension mismatch")
        for i, on in enumerate(active):
            if on:
                self._history[i].append(z[i])

    def snapshot(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(x) for x in self._history)


class CovariancePreconditionedBet:
    """Predictable projected-gradient strategy for one fixed intersection.

    Call bets(active) before revealing each round's scores, then call
    observe(scores) exactly once. The update changes only the next round's
    bet. Active coordinates must form a nested sequence. Scores for inactive
    coordinates are ignored, which is important for projectivity.

    The update is a power heuristic only. Validity follows from the ordinary
    predictable nonnegative-bet e-factor and the caller's conditional mgf
    envelope, not from convergence of the gradient update.
    """

    def __init__(
        self,
        mgf_proxy: Sequence[Sequence[float]],
        step_size: float = 0.05,
        cap: float = 1.5,
    ):
        assert_symmetric_psd(mgf_proxy)
        self._proxy = [[float(x) for x in row] for row in mgf_proxy]
        self._dimension = len(self._proxy)
        self._diagonal = [self._proxy[i][i] for i in range(self._dimension)]
        if any(x <= 0.0 for x in self._diagonal):
            raise ValueError("mgf_proxy diagonal entries must be strictly positive")
        if not isfinite(step_size) or step_size < 0.0:
            raise ValueError("step_size must be finite and nonnegative")
        if not isfinite(cap) or cap < 0.0:
            raise ValueError("cap must be finite and nonnegative")
        self.step_size = float(step_size)
        self.cap = float(cap)
        self._bets = [0.0] * self._dimension
        self._seen_active = [False] * self._dimension
        self._pending_active: list[bool] | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def bets(self, active: Sequence[bool]) -> list[float]:
        """Freeze and return the pre-update bet vector for the current round."""
        if self._pending_active is not None:
            raise RuntimeError("observe the current round before requesting new bets")
        mask = [bool(x) for x in active]
        if len(mask) != self._dimension:
            raise ValueError("active mask dimension mismatch")
        if any(seen and not on for seen, on in zip(self._seen_active, mask)):
            raise ValueError("active sets must be nested; retraction is not supported")
        self._pending_active = mask
        return [self._bets[i] if mask[i] else 0.0 for i in range(self._dimension)]

    def observe(self, scores: Sequence[float]) -> None:
        """Update next-round bets using only the just-observed active scores."""
        if self._pending_active is None:
            raise RuntimeError("call bets(active) before observing scores")
        z = _finite_vector(scores, "scores")
        if len(z) != self._dimension:
            raise ValueError("score dimension mismatch")
        active_ids = [i for i, on in enumerate(self._pending_active) if on]
        old = list(self._bets)
        updated = list(self._bets)
        for i in active_ids:
            covariance_drift = sum(
                self._proxy[i][j] * old[j] for j in active_ids
            )
            gradient = z[i] - covariance_drift
            candidate = old[i] + self.step_size * gradient / self._diagonal[i]
            if not isfinite(candidate):
                raise ValueError("bet update overflowed; refusing non-finite state")
            updated[i] = min(self.cap, max(0.0, candidate))
        self._bets = updated
        self._seen_active = list(self._pending_active)
        self._pending_active = None

    def snapshot(self) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Return immutable bet/activation state for audit and prefix tests."""
        if self._pending_active is not None:
            raise RuntimeError("cannot snapshot while a round awaits observation")
        return tuple(self._bets), tuple(self._seen_active)
