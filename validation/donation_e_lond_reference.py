"""Exact reference and indexed donation e-LOND for single-pass e-values.

This module implements the one-arrival-per-time procedure of Xu, Fischer
and Ramdas (UAI 2026 / arXiv:2603.24792). It is not a repeated-evidence
dynamic-closure procedure: once E_i arrives, it remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, Sequence


Number = Fraction | int | str


def as_fraction(value: Number) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("use exact Fraction, int or rational string; floats are forbidden")
    return value if isinstance(value, Fraction) else Fraction(value)


def _validate_parameters(
    delta: Number, gamma: Sequence[Number], e_values: Sequence[Number]
) -> tuple[Fraction, list[Fraction], list[Fraction]]:
    level = as_fraction(delta)
    weights = [as_fraction(x) for x in gamma]
    evidence = [as_fraction(x) for x in e_values]
    if not 0 < level <= 1:
        raise ValueError("delta must be in (0,1]")
    if len(weights) != len(evidence):
        raise ValueError("gamma and e_values must have equal lengths")
    if any(x < 0 for x in weights):
        raise ValueError("gamma weights must be nonnegative")
    if sum(weights, Fraction(0)) > 1:
        raise ValueError("gamma weights must sum to at most one")
    if any(x < 0 for x in evidence):
        raise ValueError("e-values must be nonnegative")
    return level, weights, evidence


def _min_one(value: Fraction) -> Fraction:
    return min(value, Fraction(1))


def donation_wealth_reference(
    previous_e_values: Sequence[Number],
    previous_gamma: Sequence[Number],
    rejected: Iterable[int],
    delta: Number,
) -> Fraction:
    """O(t) exact reference for Wbar_t in the donation e-LOND paper."""
    level = as_fraction(delta)
    e_values = [as_fraction(x) for x in previous_e_values]
    gamma = [as_fraction(x) for x in previous_gamma]
    if len(e_values) != len(gamma):
        raise ValueError("previous e-values and gamma must have equal lengths")
    if any(x < 0 for x in e_values) or any(x < 0 for x in gamma):
        raise ValueError("e-values and gamma weights must be nonnegative")
    rejected_ids = set(rejected)
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 1 or i > len(e_values) for i in rejected_ids):
        raise ValueError("rejected IDs must be 1-based indices in the previous stream")
    r = len(rejected_ids)
    q = Fraction(1, 1) / (level * (r + 1))
    wealth = Fraction(0)
    for index, (e_value, weight) in enumerate(zip(e_values, gamma), start=1):
        if index in rejected_ids:
            if weight == 0:
                raise ValueError("a zero-weight hypothesis cannot be in the rejection set")
            wealth += min(weight, weight * e_value - q)
        else:
            wealth += weight * _min_one(e_value)
    return wealth


def donation_threshold_reference(
    previous_e_values: Sequence[Number],
    previous_gamma: Sequence[Number],
    rejected: Iterable[int],
    next_gamma: Number,
    delta: Number,
) -> dict:
    """Return exact e-value rejection threshold and wealth by O(t) scan."""
    level = as_fraction(delta)
    weight = as_fraction(next_gamma)
    if weight < 0:
        raise ValueError("next gamma weight must be nonnegative")
    rejected_ids = set(rejected)
    r = len(rejected_ids)
    wealth = donation_wealth_reference(
        previous_e_values, previous_gamma, rejected_ids, level
    )
    capped = min(level * (r + 1) * wealth, Fraction(1))
    denominator = Fraction(1) - capped
    if weight == 0:
        threshold = None
        test_level = Fraction(0)
    elif denominator == 0:
        threshold = Fraction(0)
        test_level = None
    else:
        test_level = level * weight * (r + 1) / denominator
        threshold = Fraction(1, 1) / test_level
    baseline_threshold = None if weight == 0 else Fraction(1, 1) / (
        level * weight * (r + 1)
    )
    return {
        "wealth": wealth,
        "rejection_count": r,
        "capped_wealth": capped,
        "denominator": denominator,
        "test_level": test_level,
        "e_value_threshold": threshold,
        "baseline_e_lond_threshold": baseline_threshold,
        "threshold_scope": (
            "zero_weight_never_reject"
            if weight == 0
            else "zero_threshold_accepts_any_nonnegative_e_value"
            if threshold == 0
            else "finite_exact_threshold"
        ),
    }


@dataclass(slots=True)
class _Node:
    key: Fraction
    bucket_count: int = 0
    bucket_gamma: Fraction = Fraction(0)
    bucket_gamma_e: Fraction = Fraction(0)
    height: int = 1
    subtree_count: int = 0
    subtree_gamma: Fraction = Fraction(0)
    subtree_gamma_e: Fraction = Fraction(0)
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    def __post_init__(self) -> None:
        self.subtree_count = self.bucket_count
        self.subtree_gamma = self.bucket_gamma
        self.subtree_gamma_e = self.bucket_gamma_e


def _height(node: Optional[_Node]) -> int:
    return node.height if node else 0


def _count(node: Optional[_Node]) -> int:
    return node.subtree_count if node else 0


def _gamma(node: Optional[_Node]) -> Fraction:
    return node.subtree_gamma if node else Fraction(0)


def _gamma_e(node: Optional[_Node]) -> Fraction:
    return node.subtree_gamma_e if node else Fraction(0)


def _refresh(node: _Node) -> _Node:
    node.height = 1 + max(_height(node.left), _height(node.right))
    node.subtree_count = node.bucket_count + _count(node.left) + _count(node.right)
    node.subtree_gamma = node.bucket_gamma + _gamma(node.left) + _gamma(node.right)
    node.subtree_gamma_e = node.bucket_gamma_e + _gamma_e(node.left) + _gamma_e(node.right)
    return node


def _rotate_right(root: _Node) -> _Node:
    pivot = root.left
    if pivot is None:
        raise AssertionError("right rotation requires a left child")
    root.left = pivot.right
    pivot.right = _refresh(root)
    return _refresh(pivot)


def _rotate_left(root: _Node) -> _Node:
    pivot = root.right
    if pivot is None:
        raise AssertionError("left rotation requires a right child")
    root.right = pivot.left
    pivot.left = _refresh(root)
    return _refresh(pivot)


def _rebalance(node: _Node) -> _Node:
    _refresh(node)
    skew = _height(node.left) - _height(node.right)
    if skew > 1:
        assert node.left is not None
        if _height(node.left.left) < _height(node.left.right):
            node.left = _rotate_left(node.left)
        return _rotate_right(node)
    if skew < -1:
        assert node.right is not None
        if _height(node.right.right) < _height(node.right.left):
            node.right = _rotate_right(node.right)
        return _rotate_left(node)
    return node


def _insert(
    node: Optional[_Node], key: Fraction, gamma: Fraction, gamma_e: Fraction
) -> _Node:
    if node is None:
        return _Node(
            key=key,
            bucket_count=1,
            bucket_gamma=gamma,
            bucket_gamma_e=gamma_e,
        )
    if key < node.key:
        node.left = _insert(node.left, key, gamma, gamma_e)
    elif key > node.key:
        node.right = _insert(node.right, key, gamma, gamma_e)
    else:
        node.bucket_count += 1
        node.bucket_gamma += gamma
        node.bucket_gamma_e += gamma_e
        return _refresh(node)
    return _rebalance(node)


def _prefix_le(node: Optional[_Node], threshold: Fraction) -> tuple[int, Fraction, Fraction]:
    """Return count, sum gamma, sum gamma*E for keys <= threshold."""
    if node is None:
        return 0, Fraction(0), Fraction(0)
    if threshold < node.key:
        return _prefix_le(node.left, threshold)
    left_count, left_gamma, left_gamma_e = (
        _count(node.left), _gamma(node.left), _gamma_e(node.left)
    )
    right_count, right_gamma, right_gamma_e = _prefix_le(node.right, threshold)
    return (
        left_count + node.bucket_count + right_count,
        left_gamma + node.bucket_gamma + right_gamma,
        left_gamma_e + node.bucket_gamma_e + right_gamma_e,
    )


class DonationELONDIndex:
    """O(log t) donation e-LOND threshold index with an independent scan oracle.

    The order-statistics AVL stores only rejected hypotheses, keyed by
    x_i=gamma_i*(E_i-1). Nonrejected capped wealth is maintained as a scalar.
    Rejection sets are monotone, matching online e-LOND.
    """

    def __init__(self, delta: Number):
        self.delta = as_fraction(delta)
        if not 0 < self.delta <= 1:
            raise ValueError("delta must be in (0,1]")
        self.gamma: list[Fraction] = []
        self.e_values: list[Fraction] = []
        self.rejected: list[bool] = []
        self.nonrejected_wealth = Fraction(0)
        self.gamma_mass = Fraction(0)
        self.root: Optional[_Node] = None

    @property
    def rejection_count(self) -> int:
        return _count(self.root)

    @property
    def selected_key_count(self) -> int:
        return _count(self.root)

    def threshold(self, next_gamma: Number) -> dict:
        """Compute next threshold in O(log |R|) by prefix count and sums."""
        weight = as_fraction(next_gamma)
        if weight < 0:
            raise ValueError("next gamma weight must be nonnegative")
        r = self.rejection_count
        q = Fraction(1, 1) / (self.delta * (r + 1))
        prefix_count, prefix_gamma, prefix_gamma_e = _prefix_le(self.root, q)
        total_gamma = _gamma(self.root)
        rejected_wealth = (
            prefix_gamma_e - q * prefix_count + total_gamma - prefix_gamma
        )
        wealth = self.nonrejected_wealth + rejected_wealth
        capped = min(self.delta * (r + 1) * wealth, Fraction(1))
        denominator = Fraction(1) - capped
        if weight == 0:
            test_level = Fraction(0)
            threshold = None
        elif denominator == 0:
            test_level = None
            threshold = Fraction(0)
        else:
            test_level = self.delta * weight * (r + 1) / denominator
            threshold = Fraction(1, 1) / test_level
        baseline = None if weight == 0 else Fraction(1, 1) / (
            self.delta * weight * (r + 1)
        )
        return {
            "wealth": wealth,
            "rejection_count": r,
            "capped_wealth": capped,
            "denominator": denominator,
            "test_level": test_level,
            "e_value_threshold": threshold,
            "baseline_e_lond_threshold": baseline,
            "prefix_count": prefix_count,
            "prefix_gamma": prefix_gamma,
            "prefix_gamma_e": prefix_gamma_e,
            "threshold_key": q,
        }

    def add(self, e_value: Number, gamma: Number, rejected: bool) -> dict:
        """Append one hypothesis using its precomputed arrival decision."""
        e = as_fraction(e_value)
        weight = as_fraction(gamma)
        if e < 0 or weight < 0:
            raise ValueError("e-values and gamma weights must be nonnegative")
        if not isinstance(rejected, bool):
            raise TypeError("rejected must be a bool")
        if self.gamma_mass + weight > 1:
            raise ValueError("cumulative gamma weights must sum to at most one")
        index = len(self.e_values) + 1
        if weight == 0 and rejected:
            raise ValueError("zero-weight hypothesis cannot be rejected")
        self.e_values.append(e)
        self.gamma.append(weight)
        self.rejected.append(rejected)
        self.gamma_mass += weight
        if rejected:
            key = weight * (e - 1)
            self.root = _insert(self.root, key, weight, weight * e)
        else:
            self.nonrejected_wealth += weight * _min_one(e)
        return {
            "hypothesis_id": index,
            "rejected": bool(rejected),
            "rejection_count": self.rejection_count,
        }

    def step(self, e_value: Number, gamma: Number) -> dict:
        """Compute and apply one donation e-LOND online decision."""
        before = self.threshold(gamma)
        threshold = before["e_value_threshold"]
        e = as_fraction(e_value)
        reject = threshold is not None and e >= threshold
        appended = self.add(e, gamma, reject)
        return {
            **before,
            "e_value": e,
            "rejected": reject,
            **appended,
        }

    def verify(self) -> bool:
        """Independent O(t) recomputation of wealth and threshold."""
        if len(self.e_values) != len(self.gamma) or len(self.gamma) != len(self.rejected):
            return False
        if self.gamma_mass != sum(self.gamma, Fraction(0)):
            return False
        if not self.e_values:
            return self.nonrejected_wealth == 0 and self.root is None
        # Rebuild exact expected wealth from the complete prefix.
        r = sum(self.rejected)
        q = Fraction(1, 1) / (self.delta * (r + 1))
        naive = donation_wealth_reference(
            self.e_values,
            self.gamma,
            [i + 1 for i, flag in enumerate(self.rejected) if flag],
            self.delta,
        )
        return self.current_wealth() == naive

    def current_wealth(self) -> Fraction:
        """Compute current-time Wbar using the next-round q convention."""
        r = self.rejection_count
        q = Fraction(1, 1) / (self.delta * (r + 1))
        prefix_count, prefix_gamma, prefix_gamma_e = _prefix_le(self.root, q)
        rejected_wealth = (
            prefix_gamma_e - q * prefix_count + _gamma(self.root) - prefix_gamma
        )
        return self.nonrejected_wealth + rejected_wealth

    def assert_valid_avl(self) -> None:
        """Check AVL order, height, count and both subtree weighted sums."""
        def visit(node: Optional[_Node], low=None, high=None):
            if node is None:
                return 0, 0, Fraction(0), Fraction(0)
            if (low is not None and not low < node.key) or (
                high is not None and not node.key < high
            ):
                raise AssertionError("AVL ordering invariant failed")
            lh, lc, lg, lge = visit(node.left, low, node.key)
            rh, rc, rg, rge = visit(node.right, node.key, high)
            expected_height = 1 + max(lh, rh)
            expected_count = node.bucket_count + lc + rc
            expected_gamma = node.bucket_gamma + lg + rg
            expected_gamma_e = node.bucket_gamma_e + lge + rge
            if abs(lh - rh) > 1:
                raise AssertionError("AVL balance invariant failed")
            if node.height != expected_height or node.subtree_count != expected_count:
                raise AssertionError("AVL height/count augmentation failed")
            if node.subtree_gamma != expected_gamma or node.subtree_gamma_e != expected_gamma_e:
                raise AssertionError("AVL weighted-sum augmentation failed")
            return expected_height, expected_count, expected_gamma, expected_gamma_e
        visit(self.root)


def run_donation_e_lond(
    e_values: Sequence[Number],
    gamma: Sequence[Number],
    delta: Number,
) -> dict:
    """Run exact donation e-LOND and e-LOND for a fixed single-pass stream."""
    level, weights, evidence = _validate_parameters(delta, gamma, e_values)
    index = DonationELONDIndex(level)
    donation_rows, baseline_rows = [], []
    baseline_rejected: list[int] = []
    for t, (e, weight) in enumerate(zip(evidence, weights), start=1):
        donation = index.step(e, weight)
        baseline_threshold = (
            None
            if weight == 0
            else Fraction(1, 1) / (level * weight * (len(baseline_rejected) + 1))
        )
        baseline_reject = (
            baseline_threshold is not None and e >= baseline_threshold
        )
        if baseline_reject:
            baseline_rejected.append(t)
        baseline_rows.append({
            "time": t,
            "e_value": e,
            "gamma": weight,
            "threshold": baseline_threshold,
            "rejected": baseline_reject,
            "rejection_count": len(baseline_rejected),
        })
        donation_rows.append(donation)
    # Run the independent O(t) scan once after the O(t log t) online pass.
    # Verifying every prefix here would accidentally make the wrapper O(t^2).
    if not index.verify():
        raise AssertionError("donation index disagrees with final O(t) scan")
    index.assert_valid_avl()
    return {
        "delta": level,
        "count": len(evidence),
        "donation": donation_rows,
        "e_lond": baseline_rows,
        "donation_rejections": tuple(
            i + 1 for i, rejected in enumerate(index.rejected) if rejected
        ),
        "e_lond_rejections": tuple(baseline_rejected),
        "scope": (
            "single-pass online e-value stream; published donation e-LOND "
            "SupFDR theorem assumes one e-value per arriving hypothesis"
        ),
    }


if __name__ == "__main__":
    raise SystemExit("Run test_donation_e_lond_reference.py.")
