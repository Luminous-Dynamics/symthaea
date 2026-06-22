MANIFEST_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "type": {"type": "string", "enum": ["model", "material", "texture", "audio", "image", "font", "shader", "scene", "other"]},
        "source": {
            "type": "object",
            "properties": {
                "source_name": {"type": "string"},
                "source_url": {"type": "string"},
                "creator": {"type": "string"},
                "acquired_at": {"type": "string"}
            },
            "required": ["source_url"]
        },
        "license": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "url": {"type": "string"},
                "attribution_required": {"type": "boolean"},
                "commercial_allowed": {"type": "boolean"},
                "derivative_allowed": {"type": "boolean"}
            },
            "required": ["id"]
        },
        "ai": {
            "type": "object",
            "properties": {
                "provenance_state": {"type": "string", "enum": ["not_ai", "ai_assisted_disclosed", "ai_generated_disclosed", "unknown_ai_provenance"]}
            },
            "required": ["provenance_state"]
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "path": {"type": "string"},
                    "sha256": {"type": ["string", "null"]}
                },
                "required": ["role", "path"]
            }
        },
        "style": {
            "type": "object",
            "properties": {
                "biome": {"type": "string"},
                "palette_id": {"type": "string"},
                "material_family": {"type": "string"},
                "emissive_role": {"type": "string"}
            }
        }
    },
    "required": ["id", "source", "license", "ai"]
}
