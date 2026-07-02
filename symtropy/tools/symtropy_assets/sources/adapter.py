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


POLYHAVEN_ASSET_PAGE_BASE_URL = "https://polyhaven.com/a"
POLYHAVEN_API_BASE_URL = "https://api.polyhaven.com"

class PolyHavenAdapter(SourceAdapter):
    """Every Poly Haven asset (models, textures, HDRIs) is CC0-1.0 site-wide.

    Unlike Kenney's slugs, Poly Haven asset IDs are exact-case and used
    directly in API/download URLs (e.g. "ClassicNightstand_01") — do NOT
    lowercase-normalize them the way KenneyAdapter does, or the source_url
    and any later API lookups by this ID will silently 404.
    """

    VALID_TYPES = ("model", "texture", "image")  # image covers HDRIs

    def fetch_manifest(self, identifier, asset_type="model"):
        asset_id = self._validate_identifier(identifier)
        if asset_type not in self.VALID_TYPES:
            raise ValueError(f"asset_type must be one of {self.VALID_TYPES}, got {asset_type!r}")

        return {
            "id": f"polyhaven.{asset_id}",
            "title": re.sub(r"[_-]", " ", asset_id).strip(),
            "type": "model" if asset_type == "model" else "texture",
            "source": {
                "source_name": "Poly Haven",
                "source_url": f"{POLYHAVEN_ASSET_PAGE_BASE_URL}/{asset_id}",
                "creator": "Poly Haven",
                "acquisition_method": "public_api_manifest",
                "license_basis_url": f"{POLYHAVEN_API_BASE_URL}/files/{asset_id}",
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

    def _validate_identifier(self, identifier):
        asset_id = str(identifier).strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", asset_id):
            raise ValueError(
                f"Poly Haven asset identifier {identifier!r} must be alphanumeric/underscore "
                "and preserve its original case (e.g. 'ClassicNightstand_01') — "
                "it's used verbatim in API and download URLs."
            )
        return asset_id


QUATERNIUS_PACK_PAGE_BASE_URL = "https://quaternius.com/packs"

class QuaterniusAdapter(SourceAdapter):
    """Quaternius packs (stylized nature/sci-fi/robot 3D model kits) are CC0.

    Confirmed by fetching a real pack page (Downtown City MegaKit): "Free to
    use in personal, educational and commercial projects", CC0, both the
    Standard and Source download variants.

    Identifiers are pack names with everything lowercased and concatenated —
    no hyphens or underscores (e.g. "downtowncitymegakit",
    "universalanimationlibrary2"), unlike Kenney's hyphenated slugs. Passing
    a human-readable name here will normalize it the same way so
    "Downtown City MegaKit" and "downtowncitymegakit" both work.
    """

    def fetch_manifest(self, identifier):
        pack_id = self._normalize_identifier(identifier)
        return {
            "id": f"quaternius.{pack_id}",
            "title": self._title_from_identifier(identifier, pack_id),
            "type": "model",
            "source": {
                "source_name": "Quaternius",
                "source_url": f"{QUATERNIUS_PACK_PAGE_BASE_URL}/{pack_id}.html",
                "creator": "Quaternius",
                "acquisition_method": "public_asset_page_manifest",
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
        pack_id = re.sub(r"[^a-z0-9]+", "", str(identifier).strip().lower())
        if not pack_id:
            raise ValueError("Quaternius pack identifier must contain at least one letter or number")
        return pack_id

    def _title_from_identifier(self, original, pack_id):
        # If the caller already passed a readable name (has spaces/mixed
        # case), keep it as the title; otherwise fall back to the bare
        # concatenated id, which isn't very readable but is all we have.
        original = str(original).strip()
        return original if re.search(r"[ A-Z]", original) else pack_id


AMBIENTCG_ASSET_PAGE_BASE_URL = "https://ambientcg.com/a"
AMBIENTCG_API_BASE_URL = "https://ambientcg.com/api/v2"

class AmbientCGAdapter(SourceAdapter):
    """ambientCG PBR materials/textures are CC0-1.0 site-wide.

    Confirmed by fetching a real asset page (Tiles141): "All assets are
    released under the Creative Commons CC0 license, making them free to
    use without attribution - even in commercial circumstances."

    Like Poly Haven, ambientCG asset IDs are exact-case
    (e.g. "Tiles141", "Metal032", not "tiles141") and used verbatim in the
    asset page URL and the public API (api.polyhaven.com-equivalent:
    ambientcg.com/api/v2) — do not lowercase-normalize.
    """

    def fetch_manifest(self, identifier):
        asset_id = self._validate_identifier(identifier)
        return {
            "id": f"ambientcg.{asset_id}",
            "title": re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", asset_id).strip(),
            "type": "material",
            "source": {
                "source_name": "ambientCG",
                "source_url": f"{AMBIENTCG_ASSET_PAGE_BASE_URL}/{asset_id}",
                "creator": "ambientCG",
                "acquisition_method": "public_api_manifest",
                "license_basis_url": f"{AMBIENTCG_API_BASE_URL}/full_json?type=Material&assetId={asset_id}",
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

    def _validate_identifier(self, identifier):
        asset_id = str(identifier).strip()
        if not re.fullmatch(r"[A-Za-z0-9]+", asset_id):
            raise ValueError(
                f"ambientCG asset identifier {identifier!r} must be alphanumeric "
                "and preserve its original case (e.g. 'Tiles141') — "
                "it's used verbatim in the asset page URL and API."
            )
        return asset_id
