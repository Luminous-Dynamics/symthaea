#!/usr/bin/env python3
"""Reject panic-based control flow from hardened non-test Rust paths."""
from pathlib import Path
import sys

from tree_sitter import Language, Parser
import tree_sitter_rust

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
parser = Parser(Language(tree_sitter_rust.language()))
errors: list[str] = []


def node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def preceding_attributes(item, source: bytes) -> list[str]:
    parent = item.parent
    if parent is None:
        return []
    named = parent.named_children
    try:
        index = named.index(item)
    except ValueError:
        return []
    attributes: list[str] = []
    for sibling in reversed(named[:index]):
        if sibling.type == "attribute_item":
            attributes.append(node_text(source, sibling))
            continue
        if sibling.type in {"line_comment", "block_comment"}:
            continue
        break
    return attributes


def item_is_exempt(item, source: bytes) -> bool:
    attributes = preceding_attributes(item, source)
    for attribute in attributes:
        compact = "".join(attribute.split())
        if "cfg(test)" in compact or "cfg(any(test," in compact:
            return True
        if "legacy-fail-stop-apis" in attribute:
            return True
    return False


def node_is_exempt(node, source: bytes) -> bool:
    current = node
    while current is not None:
        if current.type in {
            "function_item",
            "mod_item",
            "impl_item",
            "trait_item",
            "const_item",
            "static_item",
        } and item_is_exempt(current, source):
            return True
        current = current.parent
    return False


def test_only_module_files() -> set[Path]:
    path = SRC / "lib.rs"
    source = path.read_bytes()
    tree = parser.parse(source)
    result: set[Path] = set()
    for node in walk(tree.root_node):
        if node.type != "mod_item" or not item_is_exempt(node, source):
            continue
        name = node.child_by_field_name("name")
        if name is None:
            continue
        module = node_text(source, name)
        result.add(SRC / f"{module}.rs")
        result.add(SRC / module / "mod.rs")
    return result


test_only_files = test_only_module_files()
for path in sorted(SRC.rglob("*.rs")):
    if path in test_only_files:
        continue
    source = path.read_bytes()
    tree = parser.parse(source)
    for node in walk(tree.root_node):
        if node_is_exempt(node, source):
            continue
        violation: str | None = None
        if node.type == "macro_invocation":
            invocation = node_text(source, node).lstrip()
            for macro in ("panic!", "unreachable!", "todo!", "unimplemented!"):
                if invocation.startswith(macro):
                    violation = macro[:-1]
                    break
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            if function is not None and function.type == "field_expression":
                field = function.child_by_field_name("field")
                if field is not None:
                    method = node_text(source, field)
                    if method in {"unwrap", "expect"}:
                        violation = method
        if violation is not None:
            row, column = node.start_point
            errors.append(
                f"{path.relative_to(ROOT)}:{row + 1}:{column + 1}: "
                f"hardened runtime path uses {violation}"
            )

if errors:
    print("panic-free runtime validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("validated panic-free hardened runtime paths")
