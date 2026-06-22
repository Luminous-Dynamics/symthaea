import os
import re
import yaml
from abc import ABC, abstractmethod


KENNEY_ASSET_BASE_URL = "https://kenney.nl/assets"
KENNEY_SUPPORT_URL = "https://kenney.nl/support"
CC0_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

class SourceAdapter(ABC):
    def __init__(self, raw_vault_path):
        self.raw_vault_path = raw_vault_path

    @abstractmethod
    def fetch_manifest(self, identifier):
        pass

    def save_snapshot(self, identifier, metadata):
        os.makedirs(os.path.join(self.raw_vault_path, identifier), exist_ok=True)
        with open(os.path.join(self.raw_vault_path, identifier, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

class KenneyAdapter(SourceAdapter):
    def fetch_manifest(self, identifier):
        slug = self._normalize_identifier(identifier)
        return {
            "id": f"kenney.{slug}",
            "title": slug.replace("-", " ").title(),
            "type": "other",
            "source": {
                "source_name": "Kenney",
                "source_url": f"{KENNEY_ASSET_BASE_URL}/{slug}",
                "creator": "Kenney",
                "acquisition_method": "public_asset_page_manifest",
                "license_basis_url": KENNEY_SUPPORT_URL,
            },
            "license": {
                "id": "CC0-1.0",
                "url": CC0_LICENSE_URL,
                "attribution_required": False,
                "commercial_allowed": True,
                "derivative_allowed": True,
            },
            "ai": {"provenance_state": "not_ai"},
        }

    def _normalize_identifier(self, identifier):
        slug = str(identifier).strip().lower()
        slug = re.sub(r"[^a-z0-9-]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        if not slug:
            raise ValueError("Kenney asset identifier must contain at least one letter or number")
        return slug
