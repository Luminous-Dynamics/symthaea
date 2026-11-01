"""Exact dynamic index for the raw affine projective e-closure separator.

Research/validation code only. Fraction inputs avoid floating-point ambiguity.
A production float path needs outward-rounded intervals or certified precision.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional

Number = Fraction | int

def as_fraction(value: Number) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)

@dataclass(slots=True)
class _Node:
    key: Fraction
    multiplicity: int = 1
    height: int = 1
    size: int = 1
    subtree_sum: Fraction = Fraction(0)
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    def __post_init__(self) -> None:
        self.subtree_sum = self.key

def _height(node: Optional[_Node]) -> int:
    return node.height if node else 0

def _size(node: Optional[_Node]) -> int:
    return node.size if node else 0

def _sum(node: Optional[_Node]) -> Fraction:
    return node.subtree_sum if node else Fraction(0)

def _refresh(node: _Node) -> _Node:
    node.height = 1 + max(_height(node.left), _height(node.right))
    node.size = node.multiplicity + _size(node.left) + _size(node.right)
    node.subtree_sum = node.key * node.multiplicity + _sum(node.left) + _sum(node.right)
    return node

def _rotate_right(root: _Node) -> _Node:
    pivot = root.left
    if pivot is None:
        raise AssertionError("right rotation requires left child")
    root.left = pivot.right
    pivot.right = _refresh(root)
    return _refresh(pivot)

def _rotate_left(root: _Node) -> _Node:
    pivot = root.right
    if pivot is None:
        raise AssertionError("left rotation requires right child")
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

def _insert(node: Optional[_Node], key: Fraction) -> _Node:
    if node is None:
        return _Node(key)
    if key < node.key:
        node.left = _insert(node.left, key)
    elif key > node.key:
        node.right = _insert(node.right, key)
    else:
        node.multiplicity += 1
        return _refresh(node)
    return _rebalance(node)

def _pop_min(node: _Node) -> tuple[Optional[_Node], _Node]:
    if node.left is None:
        remainder = node.right
        node.right = None
        return remainder, node
    node.left, minimum = _pop_min(node.left)
    return _rebalance(node), minimum

def _delete(node: Optional[_Node], key: Fraction) -> Optional[_Node]:
    if node is None:
        raise KeyError(f"key not present: {key}")
    if key < node.key:
        node.left = _delete(node.left, key)
    elif key > node.key:
        node.right = _delete(node.right, key)
    else:
        if node.multiplicity > 1:
            node.multiplicity -= 1
            return _refresh(node)
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        right_without_min, successor = _pop_min(node.right)
        successor.left, successor.right = node.left, right_without_min
        return _rebalance(successor)
    return _rebalance(node)

def _prefix_lt(node: Optional[_Node], key: Fraction) -> tuple[int, Fraction]:
    """Return exact count and sum of stored keys strictly less than key."""
    if node is None:
        return 0, Fraction(0)
    if key <= node.key:
        return _prefix_lt(node.left, key)
    right_count, right_sum = _prefix_lt(node.right, key)
    return (
        _size(node.left) + node.multiplicity + right_count,
        _sum(node.left) + node.key * node.multiplicity + right_sum,
    )

def raw_separator_score(b_values: Iterable[Number], selected: Iterable[bool]) -> Fraction:
    """O(n) reference: sum_i max(0, 1{i in R}/|R| - b_i)."""
    values, mask = [as_fraction(x) for x in b_values], list(selected)
    if len(values) != len(mask):
        raise ValueError("b_values and selected must have equal length")
    r = sum(mask)
    if not r:
        return Fraction(0)
    return sum(
        (max(Fraction(0), (Fraction(1, r) if mask[i] else 0) - values[i])
         for i in range(len(values))),
        Fraction(0),
    )

class ProjectiveClosureIndex:
    """Worst-case O(log n) sparse updates, toggles and score queries.

    The global baseline C = sum(max(0, -b_i)) is maintained separately.
    The AVL tree contains selected b_i values and stores subtree count/sum.
    Inactive hypotheses must be represented by b_i = 0 until activation.
    """
    def __init__(self, b_values: Iterable[Number], selected: Optional[Iterable[bool]] = None):
        self.b = [as_fraction(x) for x in b_values]
        self.selected = list(selected) if selected is not None else [False] * len(self.b)
        if len(self.b) != len(self.selected):
            raise ValueError("b_values and selected must have equal length")
        self.root: Optional[_Node] = None
        self.baseline = sum((max(Fraction(0), -x) for x in self.b), Fraction(0))
        for value, is_selected in zip(self.b, self.selected):
            if is_selected:
                self.root = _insert(self.root, value)

    @property
    def rejection_count(self) -> int:
        # Root augmentation is cached, so this stays O(1); do not scan the mask.
        return _size(self.root)

    def append(self, b_value: Number, selected: bool = False) -> int:
        """Append an active hypothesis and return its zero-based index."""
        value = as_fraction(b_value)
        self.b.append(value)
        self.selected.append(bool(selected))
        self.baseline += max(Fraction(0), -value)
        if selected:
            self.root = _insert(self.root, value)
        return len(self.b) - 1

    def update_evidence(self, index: int, new_b_value: Number) -> None:
        """Replace b_i after one hypothesis's evidence changes."""
        if not 0 <= index < len(self.b):
            raise IndexError(index)
        new_value, old_value = as_fraction(new_b_value), self.b[index]
        if new_value == old_value:
            return
        self.baseline += max(Fraction(0), -new_value) - max(Fraction(0), -old_value)
        if self.selected[index]:
            self.root = _delete(self.root, old_value)
            self.root = _insert(self.root, new_value)
        self.b[index] = new_value

    def set_selected(self, index: int, selected: bool) -> None:
        """Toggle one hypothesis in or out of the rejection set."""
        if not 0 <= index < len(self.b):
            raise IndexError(index)
        selected = bool(selected)
        if selected == self.selected[index]:
            return
        if selected:
            self.root = _insert(self.root, self.b[index])
        else:
            self.root = _delete(self.root, self.b[index])
        self.selected[index] = selected

    def score(self) -> Fraction:
        """Return Q(R) using two O(log n) prefix queries."""
        r = self.rejection_count
        if not r:
            return Fraction(0)
        threshold = Fraction(1, r)
        count_lt, sum_lt = _prefix_lt(self.root, threshold)
        _, sum_negative = _prefix_lt(self.root, Fraction(0))
        return self.baseline + Fraction(count_lt, r) - (sum_lt - sum_negative)

    def verify(self) -> bool:
        """Independent O(n) differential check; not used by score()."""
        return self.score() == raw_separator_score(self.b, self.selected)

    def assert_valid_avl(self) -> None:
        """Check ordering, AVL height/balance, subtree counts and sums."""
        def visit(node: Optional[_Node], low=None, high=None):
            if node is None:
                return 0, 0, Fraction(0)
            if (low is not None and not low < node.key) or (high is not None and not node.key < high):
                raise AssertionError("AVL key-order invariant failed")
            lh, ls, lsum = visit(node.left, low, node.key)
            rh, rs, rsum = visit(node.right, node.key, high)
            expected_height = 1 + max(lh, rh)
            expected_size = node.multiplicity + ls + rs
            expected_sum = node.key * node.multiplicity + lsum + rsum
            if abs(lh - rh) > 1:
                raise AssertionError("AVL balance invariant failed")
            if node.height != expected_height or node.size != expected_size:
                raise AssertionError("AVL height/size augmentation failed")
            if node.subtree_sum != expected_sum:
                raise AssertionError("AVL sum augmentation failed")
            return expected_height, expected_size, expected_sum
        visit(self.root, None, None)
