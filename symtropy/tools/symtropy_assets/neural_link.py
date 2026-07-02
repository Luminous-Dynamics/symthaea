import json
import os
import urllib.error
import urllib.request

import yaml

from registry_manager import _connect

OLLAMA_URL = os.environ.get("SYMTROPY_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("SYMTROPY_NEURAL_LINK_MODEL", "gemma4:e2b")

# Must match crates/bridges/symtropy-foundry/src/orchestrator.rs's Blueprint /
# BiomeBlueprint / SamplingRule structs exactly — that's the runtime consumer.
BLUEPRINT_SCHEMA_DESCRIPTION = """{
  "world_name": string,
  "biomes": [
    {
      "name": string,
      "density": number (0.0-1.0),
      "sampling_rules": [
        {"role": string, "count_min": integer, "count_max": integer, "phi_threshold": number (0.0-1.0)}
      ]
    }
  ]
}"""


def _known_roles():
    """Best-effort: distinct asset behavior roles already in the registry, so
    the model prefers roles that actually have assets to sample. Returns []
    if the registry is empty or unreachable — sampling_rules referencing an
    unknown role just yields zero assets at runtime, it isn't fatal."""
    try:
        conn = _connect()
        try:
            cur = conn.execute("SELECT DISTINCT role FROM behaviors ORDER BY role")
            return [row[0] for row in cur.fetchall() if row[0]]
        finally:
            conn.close()
    except Exception:
        return []


def _build_prompt(prompt, roles):
    if roles:
        role_hint = (
            f"Asset roles already registered and sample-able: {', '.join(roles)}. "
            "Prefer reusing these when they fit the theme; otherwise invent a "
            "new short snake_case role name."
        )
    else:
        role_hint = (
            "No asset roles are registered yet — invent short snake_case "
            "role names appropriate to the theme."
        )

    return (
        f"Generate a Symtropy world blueprint for this creative prompt:\n"
        f'"{prompt}"\n\n'
        f"{role_hint}\n\n"
        "Respond with ONLY a JSON object matching exactly this schema "
        f"(no markdown, no commentary):\n{BLUEPRINT_SCHEMA_DESCRIPTION}\n\n"
        "Include 1-4 biomes, each with 1-5 sampling_rules. "
        "count_max must be >= count_min."
    )


def _call_ollama(prompt_text, model):
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_text,
            "format": "json",
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["response"]


def _validate_blueprint(data):
    """Raises ValueError on structural mismatch; coerces numeric types the
    model may have emitted as strings."""
    if not isinstance(data, dict):
        raise ValueError("blueprint is not a JSON object")
    if not isinstance(data.get("world_name"), str) or not data["world_name"]:
        raise ValueError("missing/invalid world_name")

    biomes = data.get("biomes")
    if not isinstance(biomes, list) or not biomes:
        raise ValueError("missing/empty biomes list")

    for biome in biomes:
        if not isinstance(biome.get("name"), str) or not biome["name"]:
            raise ValueError("biome missing name")
        biome["density"] = max(0.0, min(1.0, float(biome.get("density", 0.5))))

        rules = biome.get("sampling_rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"biome '{biome['name']}' missing sampling_rules")

        for rule in rules:
            if not isinstance(rule.get("role"), str) or not rule["role"]:
                raise ValueError("sampling_rule missing role")
            rule["count_min"] = max(0, int(rule.get("count_min", 1)))
            rule["count_max"] = max(rule["count_min"], int(rule.get("count_max", rule["count_min"])))
            if rule.get("phi_threshold") is not None:
                rule["phi_threshold"] = max(0.0, min(1.0, float(rule["phi_threshold"])))

    return data


def _fallback_blueprint():
    """Deterministic template used only when Ollama is unreachable or every
    generation attempt produced an invalid blueprint — never silent, always
    logged by the caller."""
    return {
        "world_name": "AI_Generated_Reality",
        "biomes": [
            {
                "name": "generated_biome",
                "density": 0.5,
                "sampling_rules": [
                    {"role": "network_node", "count_min": 5, "count_max": 10, "phi_threshold": 0.1}
                ],
            }
        ],
    }


def generate_world_blueprint(prompt: str, output_path: str, model: str = None):
    """
    Calls a local Ollama model to synthesize a Symtropy world blueprint
    matching symtropy-foundry's Blueprint schema, then writes it as YAML.

    Falls back to a deterministic template (with a clearly logged reason) if
    Ollama is unreachable or the model fails to produce a valid blueprint
    after one retry.
    """
    model = model or DEFAULT_MODEL
    print(f"Neural-Link: Synthesizing blueprint for prompt: '{prompt}' (model={model})")

    roles = _known_roles()
    prompt_text = _build_prompt(prompt, roles)

    blueprint = None
    for attempt in range(1, 3):
        try:
            raw = _call_ollama(prompt_text, model)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Neural-Link: Ollama unreachable at {OLLAMA_URL} ({e}); using fallback template.")
            break

        try:
            blueprint = _validate_blueprint(json.loads(raw))
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 1:
                print(f"Neural-Link: attempt {attempt} produced an invalid blueprint ({e}); retrying once.")
            else:
                print(f"Neural-Link: retry also produced an invalid blueprint ({e}); using fallback template.")

    used_fallback = blueprint is None
    if used_fallback:
        blueprint = _fallback_blueprint()

    with open(output_path, "w") as f:
        yaml.dump(blueprint, f)

    status = "fallback template" if used_fallback else f"model {model}"
    print(f"Neural-Link: Blueprint generated at {output_path} (source: {status})")
    return not used_fallback
