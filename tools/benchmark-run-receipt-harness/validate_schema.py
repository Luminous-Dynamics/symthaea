#!/usr/bin/env python3
"""CI-only Draft 2020-12 validator; production semantics are checked in Rust."""
import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def reject_external_refs(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                if not isinstance(child, str) or not child.startswith("#/"):
                    raise ValueError(f"external schema reference is forbidden at {path}.{key}")
            reject_external_refs(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_external_refs(child, f"{path}[{index}]")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    schema_path = Path(args.schema)
    instance_path = Path(args.instance)
    schema = load_json(schema_path)
    instance = load_json(instance_path)

    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("schema must declare JSON Schema Draft 2020-12")

    reject_external_refs(schema)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if errors:
        for error in errors:
            location = "/".join(str(part) for part in error.absolute_path)
            print(f"{location or '$'}: {error.message}")
        raise SystemExit(1)

    print(json.dumps({
        "schema": "luminous.benchmark-receipt-schema-audit.v1",
        "result": "valid",
        "schema_sha256": __import__("hashlib").sha256(schema_path.read_bytes()).hexdigest(),
        "instance_sha256": __import__("hashlib").sha256(instance_path.read_bytes()).hexdigest(),
        "dialect": "2020-12",
        "external_refs_allowed": False,
        "synthetic": instance.get("benchmark", {}).get("access_class") == "synthetic",
        "run_id": instance.get("run_id"),
    }, indent=2))


if __name__ == "__main__":
    main()
