#!/usr/bin/env python3
"""Parse every Rust module and reject duplicate public API identities."""
from pathlib import Path
import sys

from tree_sitter import Language, Parser
import tree_sitter_rust

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
parser = Parser(Language(tree_sitter_rust.language()))
errors: list[str] = []


def text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


for path in sorted(SRC.rglob("*.rs")):
    source = path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node
    if root.has_error:
        for node in walk(root):
            if node.type == "ERROR" or node.is_missing:
                row, column = node.start_point
                errors.append(
                    f"{path.relative_to(ROOT)}:{row + 1}:{column + 1}: Rust parse error"
                )

    for node in walk(root):
        if node.type == "function_item":
            name_node = node.child_by_field_name("name")
            params = node.child_by_field_name("parameters")
            if name_node is None or params is None:
                continue
            seen: dict[str, int] = {}
            for child in params.named_children:
                if child.type == "self_parameter":
                    continue
                pattern = child.child_by_field_name("pattern")
                if pattern is None or pattern.type != "identifier":
                    continue
                name = text(source, pattern)
                row = pattern.start_point[0] + 1
                if name in seen:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{row}: function {text(source, name_node)} "
                        f"repeats parameter {name!r} (first at line {seen[name]})"
                    )
                else:
                    seen[name] = row

        elif node.type == "struct_item":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None or body is None:
                continue
            seen: dict[str, int] = {}
            for child in body.named_children:
                if child.type != "field_declaration":
                    continue
                name_node_field = child.child_by_field_name("name")
                if name_node_field is None:
                    continue
                name = text(source, name_node_field)
                row = name_node_field.start_point[0] + 1
                if name in seen:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{row}: struct {text(source, name_node)} "
                        f"repeats field {name!r} (first at line {seen[name]})"
                    )
                else:
                    seen[name] = row

        elif node.type == "enum_item":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None or body is None:
                continue
            seen: dict[str, int] = {}
            for child in body.named_children:
                if child.type != "enum_variant":
                    continue
                variant = child.child_by_field_name("name")
                if variant is None:
                    continue
                name = text(source, variant)
                row = variant.start_point[0] + 1
                if name in seen:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{row}: enum {text(source, name_node)} "
                        f"repeats variant {name!r} (first at line {seen[name]})"
                    )
                else:
                    seen[name] = row

        elif node.type == "mod_item" and node.child_by_field_name("body") is None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            name = text(source, name_node)
            candidates = [path.parent / f"{name}.rs", path.parent / name / "mod.rs"]
            if not any(candidate.exists() for candidate in candidates):
                row = name_node.start_point[0] + 1
                errors.append(
                    f"{path.relative_to(ROOT)}:{row}: module {name!r} has no source file"
                )

if errors:
    print("Rust API-shape validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("validated Rust syntax, module resolution, and duplicate API identities")
