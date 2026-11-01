"""Merkle-authenticated receipts for exact projective closure scores.

The verifier proves a score relative to a trusted snapshot commitment. It does
not prove the source evidence is true or that the source roster is complete.
The manifest root must be independently validated and anchored by the caller.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional

EMPTY_HASH = hashlib.sha256(b"projective-closure-empty-node-v1").hexdigest()
EMPTY_SELECTION_DIGEST = hashlib.sha256(b"projective-closure-empty-selection-v1").hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fraction(value: Fraction | int | str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("use exact Fraction, int or rational string; floats are forbidden")
    result = value if isinstance(value, Fraction) else Fraction(value)
    return result


def _frac_obj(value: Fraction) -> list[str]:
    return [str(value.numerator), str(value.denominator)]


def _frac_from_obj(value: Any) -> Fraction:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("rational must be [numerator, denominator]")
    num, den = value
    if not isinstance(num, str) or not isinstance(den, str):
        raise ValueError("rational numerator and denominator must be strings")
    if not num.lstrip("-").isdigit() or not den.isdigit() or den == "0":
        raise ValueError("invalid rational encoding")
    result = Fraction(int(num), int(den))
    if _frac_obj(result) != value:
        raise ValueError("rational encoding is not canonical")
    return result


@dataclass(frozen=True)
class Aggregate:
    size: int
    selected_count: int
    selected_sum: Fraction
    baseline: Fraction
    selected_digest: str


@dataclass(frozen=True)
class _Node:
    key: Fraction
    hypothesis_id: str
    selected: bool
    left: Optional["_Node"]
    right: Optional["_Node"]
    aggregate: Aggregate
    node_hash: str


def _empty_aggregate() -> Aggregate:
    return Aggregate(0, 0, Fraction(0), Fraction(0), EMPTY_SELECTION_DIGEST)


def _agg_obj(agg: Aggregate) -> dict[str, Any]:
    return {
        "size": agg.size,
        "selected_count": agg.selected_count,
        "selected_sum": _frac_obj(agg.selected_sum),
        "baseline": _frac_obj(agg.baseline),
        "selected_digest": agg.selected_digest,
    }


def _agg_from_obj(obj: Any) -> Aggregate:
    if not isinstance(obj, dict):
        raise ValueError("aggregate must be an object")
    size, selected_count = obj.get("size"), obj.get("selected_count")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or not 0 <= selected_count <= size
    ):
        raise ValueError("invalid aggregate cardinality")
    selected_digest = obj.get("selected_digest")
    if (
        not isinstance(selected_digest, str)
        or len(selected_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in selected_digest)
    ):
        raise ValueError("invalid selected digest")
    return Aggregate(
        size=size,
        selected_count=selected_count,
        selected_sum=_frac_from_obj(obj.get("selected_sum")),
        baseline=_frac_from_obj(obj.get("baseline")),
        selected_digest=selected_digest,
    )


def _ref(node: Optional[_Node]) -> dict[str, Any]:
    if node is None:
        return {"hash": EMPTY_HASH, "aggregate": _agg_obj(_empty_aggregate())}
    return {"hash": node.node_hash, "aggregate": _agg_obj(node.aggregate)}


def _combine_aggregate(
    left: Aggregate,
    key: Fraction,
    hypothesis_id: str,
    selected: bool,
    right: Aggregate,
) -> Aggregate:
    selected_here = 1 if selected else 0
    selected_sum_here = key if selected else Fraction(0)
    id_bytes = hypothesis_id.encode("utf-8")
    entry_selection = (
        b"\x01" + len(id_bytes).to_bytes(4, "big") + id_bytes
        if selected
        else b"\x00"
    )
    digest_payload = (
        b"projective-closure-selection-node-v1"
        + bytes.fromhex(left.selected_digest)
        + entry_selection
        + bytes.fromhex(right.selected_digest)
    )
    selected_digest = hashlib.sha256(digest_payload).hexdigest()
    return Aggregate(
        size=left.size + 1 + right.size,
        selected_count=left.selected_count + selected_here + right.selected_count,
        selected_sum=left.selected_sum + selected_sum_here + right.selected_sum,
        baseline=left.baseline + max(Fraction(0), -key) + right.baseline,
        selected_digest=selected_digest,
    )


def _node_hash(
    key: Fraction,
    hypothesis_id: str,
    selected: bool,
    left_ref: Mapping[str, Any],
    right_ref: Mapping[str, Any],
    aggregate: Aggregate,
) -> str:
    return _hash(
        {
            "version": 1,
            "key": _frac_obj(key),
            "hypothesis_id": hypothesis_id,
            "selected": selected,
            "left": dict(left_ref),
            "right": dict(right_ref),
            "aggregate": _agg_obj(aggregate),
        }
    )


def _build_balanced(
    entries: list[tuple[Fraction, str, bool]],
    start: int = 0,
    end: Optional[int] = None,
) -> Optional[_Node]:
    if end is None:
        end = len(entries)
    if start >= end:
        return None
    mid = (start + end) // 2
    key, hypothesis_id, selected = entries[mid]
    left = _build_balanced(entries, start, mid)
    right = _build_balanced(entries, mid + 1, end)
    left_agg = left.aggregate if left else _empty_aggregate()
    right_agg = right.aggregate if right else _empty_aggregate()
    aggregate = _combine_aggregate(
        left_agg, key, hypothesis_id, selected, right_agg
    )
    left_ref, right_ref = _ref(left), _ref(right)
    node_hash = _node_hash(
        key, hypothesis_id, selected, left_ref, right_ref, aggregate
    )
    return _Node(
        key, hypothesis_id, selected, left, right, aggregate, node_hash
    )


@dataclass(frozen=True)
class Snapshot:
    tree_root: Optional[_Node]
    manifest_root: str
    sequence: int
    previous_snapshot_hash: Optional[str]
    configuration: Mapping[str, Any]
    snapshot_hash: str

    @property
    def tree_root_hash(self) -> str:
        return self.tree_root.node_hash if self.tree_root else EMPTY_HASH

    @property
    def aggregate(self) -> Aggregate:
        return self.tree_root.aggregate if self.tree_root else _empty_aggregate()


def build_snapshot(
    entries: Iterable[Mapping[str, Any]],
    *,
    manifest_root: str,
    sequence: int,
    previous_snapshot_hash: Optional[str] = None,
    configuration: Optional[Mapping[str, Any]] = None,
) -> Snapshot:
    """Build one deterministic balanced committed snapshot.

    Entry b_i values must already be computed from the declared alpha, weights
    and e-values. The caller must independently validate manifest_root.
    """
    if not isinstance(manifest_root, str) or len(manifest_root) != 64:
        raise ValueError("manifest_root must be a 64-character SHA-256 digest")
    if any(ch not in "0123456789abcdef" for ch in manifest_root):
        raise ValueError("manifest_root must be lowercase hexadecimal")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a nonnegative integer")
    if previous_snapshot_hash is not None and (
        len(previous_snapshot_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in previous_snapshot_hash)
    ):
        raise ValueError("previous_snapshot_hash must be lowercase SHA-256 or None")
    config = dict(configuration or {})
    _canonical(config)  # fail if config is not canonically JSON serializable
    prepared: list[tuple[Fraction, str, bool]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        hypothesis_id = entry.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("hypothesis_id must be a non-empty string")
        if hypothesis_id in seen_ids:
            raise ValueError("hypothesis_id values must be unique")
        seen_ids.add(hypothesis_id)
        selected = entry.get("selected")
        if not isinstance(selected, bool):
            raise ValueError("selected must be a bool")
        prepared.append((_fraction(entry.get("b")), hypothesis_id, selected))
    prepared.sort(key=lambda item: (item[0], item[1]))
    root = _build_balanced(prepared)
    tree_root_hash = root.node_hash if root else EMPTY_HASH
    header = {
        "version": 1,
        "tree_root_hash": tree_root_hash,
        "manifest_root": manifest_root,
        "sequence": sequence,
        "previous_snapshot_hash": previous_snapshot_hash,
        "configuration": config,
    }
    snapshot_hash = _hash(header)
    return Snapshot(
        root,
        manifest_root,
        sequence,
        previous_snapshot_hash,
        config,
        snapshot_hash,
    )


def _make_prefix_proof(root: Optional[_Node], threshold: Fraction) -> dict[str, Any]:
    path: list[dict[str, Any]] = []
    node = root
    while node is not None:
        direction = "right" if node.key < threshold else "left"
        path.append(
            {
                "key": _frac_obj(node.key),
                "hypothesis_id": node.hypothesis_id,
                "selected": node.selected,
                "left": _ref(node.left),
                "right": _ref(node.right),
                "follow": direction,
            }
        )
        node = node.right if direction == "right" else node.left
    return {
        "tree_root_hash": root.node_hash if root else EMPTY_HASH,
        "root_aggregate": _agg_obj(root.aggregate if root else _empty_aggregate()),
        "threshold": _frac_obj(threshold),
        "path": path,
    }


@dataclass(frozen=True)
class PrefixAggregate:
    selected_count: int
    selected_sum: Fraction


def _decode_ref(value: Any) -> tuple[str, Aggregate]:
    if not isinstance(value, dict):
        raise ValueError("child reference must be an object")
    digest = value.get("hash")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ValueError("invalid child digest")
    return digest, _agg_from_obj(value.get("aggregate"))


def verify_prefix_proof(
    proof: Mapping[str, Any],
    *,
    expected_tree_root_hash: str,
    expected_threshold: Fraction,
) -> tuple[Aggregate, PrefixAggregate]:
    """Verify a strict-< threshold proof from a trusted tree root."""
    if proof.get("tree_root_hash") != expected_tree_root_hash:
        raise ValueError("prefix proof is anchored to the wrong tree root")
    threshold = _frac_from_obj(proof.get("threshold"))
    if threshold != expected_threshold:
        raise ValueError("prefix proof threshold mismatch")
    path = proof.get("path")
    if not isinstance(path, list):
        raise ValueError("prefix proof path must be a list")

    child_hash = EMPTY_HASH
    child_aggregate = _empty_aggregate()
    child_prefix = PrefixAggregate(0, Fraction(0))

    for step in reversed(path):
        if not isinstance(step, dict):
            raise ValueError("path step must be an object")
        key = _frac_from_obj(step.get("key"))
        hypothesis_id = step.get("hypothesis_id")
        selected = step.get("selected")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("invalid hypothesis id in proof")
        if not isinstance(selected, bool):
            raise ValueError("selected flag in proof must be bool")
        direction = "right" if key < threshold else "left"
        if step.get("follow") != direction:
            raise ValueError("proof follows a non-canonical search direction")

        left_hash, left_agg = _decode_ref(step.get("left"))
        right_hash, right_agg = _decode_ref(step.get("right"))
        if direction == "left":
            if left_hash != child_hash or left_agg != child_aggregate:
                raise ValueError("left child proof does not match its parent commitment")
            prefix = child_prefix
        else:
            if right_hash != child_hash or right_agg != child_aggregate:
                raise ValueError("right child proof does not match its parent commitment")
            prefix = PrefixAggregate(
                left_agg.selected_count
                + (1 if selected else 0)
                + child_prefix.selected_count,
                left_agg.selected_sum
                + (key if selected else Fraction(0))
                + child_prefix.selected_sum,
            )

        aggregate = _combine_aggregate(
            left_agg, key, hypothesis_id, selected, right_agg
        )
        node_hash = _node_hash(
            key,
            hypothesis_id,
            selected,
            {"hash": left_hash, "aggregate": _agg_obj(left_agg)},
            {"hash": right_hash, "aggregate": _agg_obj(right_agg)},
            aggregate,
        )
        child_hash, child_aggregate, child_prefix = (
            node_hash,
            aggregate,
            prefix,
        )

    root_aggregate = _agg_from_obj(proof.get("root_aggregate"))
    if child_hash != expected_tree_root_hash:
        raise ValueError("recomputed path does not reach trusted tree root")
    if child_aggregate != root_aggregate:
        raise ValueError("root aggregate does not match authenticated path")
    return child_aggregate, child_prefix


def make_score_receipt(
    snapshot: Snapshot,
    *,
    alpha: Fraction | int | str,
    weight_digest: str,
    evidence_version: str,
) -> dict[str, Any]:
    """Create authenticated Q(R) receipt for nonempty selected set R."""
    alpha_value = _fraction(alpha)
    if alpha_value <= 0:
        raise ValueError("alpha must be positive")
    for name, value in (("weight_digest", weight_digest), ("evidence_version", evidence_version)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    config_alpha = snapshot.configuration.get("alpha")
    config_weight_digest = snapshot.configuration.get("weight_digest")
    config_evidence_version = snapshot.configuration.get("evidence_version")
    if config_alpha is None or _fraction(config_alpha) != alpha_value:
        raise ValueError("alpha is not bound to the committed snapshot configuration")
    if config_weight_digest != weight_digest:
        raise ValueError("weight digest is not bound to the committed snapshot configuration")
    if config_evidence_version != evidence_version:
        raise ValueError("evidence version is not bound to the committed snapshot configuration")
    agg = snapshot.aggregate
    r = agg.selected_count
    if r == 0:
        raise ValueError("score receipt requires a nonempty rejection set")
    threshold = Fraction(1, r)
    p_threshold = _make_prefix_proof(snapshot.tree_root, threshold)
    p_zero = _make_prefix_proof(snapshot.tree_root, Fraction(0))
    # The exact score is redundantly computed here; verifier reconstructs it.
    prefix_count = verify_prefix_proof(
        p_threshold,
        expected_tree_root_hash=snapshot.tree_root_hash,
        expected_threshold=threshold,
    )[1].selected_count
    prefix_sum = verify_prefix_proof(
        p_threshold,
        expected_tree_root_hash=snapshot.tree_root_hash,
        expected_threshold=threshold,
    )[1].selected_sum
    negative_sum = verify_prefix_proof(
        p_zero,
        expected_tree_root_hash=snapshot.tree_root_hash,
        expected_threshold=Fraction(0),
    )[1].selected_sum
    score = agg.baseline + Fraction(prefix_count, r) - (prefix_sum - negative_sum)

    return {
        "version": 1,
        "snapshot": {
            "snapshot_hash": snapshot.snapshot_hash,
            "tree_root_hash": snapshot.tree_root_hash,
            "manifest_root": snapshot.manifest_root,
            "sequence": snapshot.sequence,
            "previous_snapshot_hash": snapshot.previous_snapshot_hash,
            "configuration": dict(snapshot.configuration),
        },
        "alpha": _frac_obj(alpha_value),
        "weight_digest": weight_digest,
        "evidence_version": evidence_version,
        "rejection_count": r,
        "selected_digest": agg.selected_digest,
        "threshold": _frac_obj(threshold),
        "root_aggregate": _agg_obj(agg),
        "prefix_lt_threshold": p_threshold,
        "prefix_lt_zero": p_zero,
        "closure_score": _frac_obj(score),
    }


def verify_score_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_snapshot_hash: str,
    expected_manifest_root: str,
    expected_weight_digest: Optional[str] = None,
    expected_evidence_version: Optional[str] = None,
    expected_alpha: Optional[Fraction] = None,
    expected_selected_digest: Optional[str] = None,
) -> Fraction:
    """Verify receipt against caller-pinned snapshot/configuration commitments."""
    if receipt.get("version") != 1:
        raise ValueError("unsupported receipt version")
    snapshot = receipt.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot header missing")
    header = {
        "version": 1,
        "tree_root_hash": snapshot.get("tree_root_hash"),
        "manifest_root": snapshot.get("manifest_root"),
        "sequence": snapshot.get("sequence"),
        "previous_snapshot_hash": snapshot.get("previous_snapshot_hash"),
        "configuration": snapshot.get("configuration"),
    }
    if _hash(header) != expected_snapshot_hash:
        raise ValueError("snapshot header does not match caller-pinned commitment")
    if snapshot.get("snapshot_hash") != expected_snapshot_hash:
        raise ValueError("receipt snapshot hash mismatch")
    if snapshot.get("manifest_root") != expected_manifest_root:
        raise ValueError("receipt manifest root mismatch")

    weight_digest = receipt.get("weight_digest")
    evidence_version = receipt.get("evidence_version")
    if expected_weight_digest is not None and weight_digest != expected_weight_digest:
        raise ValueError("weight digest mismatch")
    if expected_evidence_version is not None and evidence_version != expected_evidence_version:
        raise ValueError("evidence version mismatch")
    alpha = _frac_from_obj(receipt.get("alpha"))
    if alpha <= 0 or (expected_alpha is not None and alpha != expected_alpha):
        raise ValueError("alpha mismatch or invalid alpha")
    configuration = snapshot.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("snapshot configuration missing")
    try:
        committed_alpha = _fraction(configuration.get("alpha"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid committed alpha") from exc
    if committed_alpha != alpha:
        raise ValueError("receipt alpha is not bound to snapshot configuration")
    if configuration.get("weight_digest") != weight_digest:
        raise ValueError("receipt weight digest is not bound to snapshot configuration")
    if configuration.get("evidence_version") != evidence_version:
        raise ValueError("receipt evidence version is not bound to snapshot configuration")

    r = receipt.get("rejection_count")
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError("rejection_count must be positive")
    root_aggregate = _agg_from_obj(receipt.get("root_aggregate"))
    if root_aggregate.selected_count != r:
        raise ValueError("rejection count does not match authenticated root")
    selected_digest = receipt.get("selected_digest")
    if selected_digest != root_aggregate.selected_digest:
        raise ValueError("selected set digest does not match authenticated root")
    if expected_selected_digest is not None and selected_digest != expected_selected_digest:
        raise ValueError("selected set digest does not match requested rejection set")

    threshold = Fraction(1, r)
    if _frac_from_obj(receipt.get("threshold")) != threshold:
        raise ValueError("receipt threshold is not 1/r")
    tree_root_hash = snapshot.get("tree_root_hash")
    if not isinstance(tree_root_hash, str):
        raise ValueError("tree root hash missing")

    root1, prefix1 = verify_prefix_proof(
        receipt.get("prefix_lt_threshold", {}),
        expected_tree_root_hash=tree_root_hash,
        expected_threshold=threshold,
    )
    root0, prefix0 = verify_prefix_proof(
        receipt.get("prefix_lt_zero", {}),
        expected_tree_root_hash=tree_root_hash,
        expected_threshold=Fraction(0),
    )
    if root1 != root_aggregate or root0 != root_aggregate:
        raise ValueError("prefix proofs do not share the committed root aggregate")

    score = (
        root_aggregate.baseline
        + Fraction(prefix1.selected_count, r)
        - (prefix1.selected_sum - prefix0.selected_sum)
    )
    if _frac_from_obj(receipt.get("closure_score")) != score:
        raise ValueError("claimed closure score does not match authenticated prefixes")
    return score
