#!/usr/bin/env python3
"""Independent static contract checks for the exact-head NeuroBridge local capsule."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROFILE = "symthaea-neurobridge-static-contracts-v0.1"
ROOT = Path(__file__).resolve().parents[2]


class ContractError(RuntimeError):
    pass


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def json_doc(path: str):
    return json.loads(text(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def contains(path: str, needle: str) -> None:
    require(needle in text(path), f"{path}: missing contract text: {needle!r}")


def no_match(path: str, pattern: str, message: str) -> None:
    require(re.search(pattern, text(path), flags=re.MULTILINE) is None, message)


def quarantine_contract() -> None:
    mod = "crates/domains/symthaea-psych-bench/src/benchmarks/neural_validation/mod.rs"
    doc = "docs/neuroscience/NEURAL_BENCHMARK_QUALIFICATION_V1.md"
    mapping = "docs/neuroscience/FSAVERAGE5_GLASSER_TRANSFORM_QUALIFICATION_V1.md"
    contains(mod, "pub(crate) mod cortical_similarity;")
    no_match(mod, r"^pub use cortical_similarity::", "legacy neural benchmarks are publicly re-exported")
    contains(mod, "pub const QUALIFIED_EXTERNAL_BENCHMARKS: &[&str] = &[];")
    contains(mod, "NeuralValidationQualification::Quarantined")
    for benchmark in (
        "CorticalSimilarity",
        "TemporalDynamics",
        "BidirectionalValidation",
        "SubstrateComparison",
        "ParcellationRobustness",
        "EvidenceUpgrade",
        "EegValidation",
        "HybridSubstrate",
    ):
        contains(mod, f'"{benchmark}"')
        contains(doc, f"`{benchmark}`")
    for invariant in (
        "NBQ-002 — No implicit fallback",
        "NBQ-004 — Transform lineage complete",
        "NBQ-007 — Evidence-use admission",
        "NBQ-009 — Claim bounded by evidence",
    ):
        contains(doc, invariant)
    for invariant in (
        "total vertices: `20_484`",
        "indices `0..10_242`: left hemisphere",
        "indices `10_242..20_484`: right hemisphere",
        "FMQ-003 — Semantic label resolution",
        "FMQ-006 — Complete parcel coverage",
        "FMQ-010 — Independent derivation check",
        "FMQ-012 — Lossiness explicit",
        "FMQ-014 — No authority upgrade",
    ):
        contains(mapping, invariant)


def atlas_contract() -> None:
    area = json_doc("data/neuroscience/hcp_mmp1_area_order_v1.json")
    areas = area["areas"]
    require(len(areas) == 180 and len(set(areas)) == 180, "area namespace must contain 180 unique names")
    require(areas[:6] == ["V1", "MST", "V6", "V2", "V3", "V4"], "area namespace prefix drift")
    require(areas[-2:] == ["a32pr", "p24"], "area namespace suffix drift")
    require(area["source"]["commit"] == "e3a33a5a50d4ca86ab8fbaa0407d6c3296fcab12", "area source commit drift")
    require(area["source"]["blob_sha"] == "78a240b52845dd01c8676ecfabef59e2a0526a85", "area source blob drift")

    compiler_doc = "docs/neuroscience/FSAVERAGE5_GLASSER_MAP_COMPILER_V1.md"
    contains(compiler_doc, "no real atlas-derived mapping is qualified yet")
    contains(compiler_doc, "semantic labels, not raw `.annot` integers")
    contains(compiler_doc, "14 synthetic contract tests")

    cross = "scripts/compare_fsaverage5_glasser_maps.py"
    cross_doc = "docs/neuroscience/FSAVERAGE5_GLASSER_CROSSCHECK_V1.md"
    contains(cross, 'independence_established": False')
    contains(cross, "requires_external_provenance_review")
    contains(cross_doc, "no built-in permissive percentage threshold")
    contains(cross_doc, "no real independent-lineage comparison has qualified yet")
    contains(cross_doc, "13 synthetic contract tests")


def extractor_contract() -> None:
    mills = json_doc("data/neuroscience/hcpmmp1_mills_figshare_fsaverage_v2.json")
    require(mills["schema"] == "symthaea-hcpmmp1-fsaverage-lineage-v1", "Mills schema drift")
    require(mills["lineage_id"] == "mills-figshare-hcpmmp1-fsaverage-v2", "Mills lineage drift")
    require(mills["files"]["left"]["url"] == "https://ndownloader.figshare.com/files/5528816", "Mills left URL drift")
    require(mills["files"]["left"]["md5"] == "46a102b59b2fb1bb4bd62d51bf02e975", "Mills left MD5 drift")
    require(mills["files"]["right"]["url"] == "https://ndownloader.figshare.com/files/5528819", "Mills right URL drift")
    require(mills["files"]["right"]["md5"] == "75e96b331940227bbcb07c1c791c2463", "Mills right MD5 drift")
    require(mills["files"]["left"]["expected_vertices"] == 163842, "Mills left vertex count drift")
    require(mills["files"]["right"]["expected_vertices"] == 163842, "Mills right vertex count drift")
    require(mills["acknowledgement_required"] is True, "Mills acknowledgement boundary changed")
    contains("docs/neuroscience/FSAVERAGE_HCPMMP1_SEMANTIC_EXTRACTOR_V1.md", "does not auto-download")
    contains("docs/neuroscience/FSAVERAGE_HCPMMP1_SEMANTIC_EXTRACTOR_V1.md", "`???`")
    contains("docs/neuroscience/FSAVERAGE_HCPMMP1_SEMANTIC_EXTRACTOR_V1.md", "does not satisfy FMQ-010")


def lineage_b_contract() -> None:
    method = json_doc("data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json")
    require(method["schema"] == "symthaea-hcpmmp1-neuromaps-method-v1", "Lineage-B schema drift")
    require(method["target_vertices_per_hemisphere"] == 10242, "Lineage-B target geometry drift")
    require(
        method["source_atlas"]
        == {
            "automatic_acquisition_permitted": False,
            "hemisphere_pair_required": True,
            "left_file_id": "npz0",
            "left_filename": "Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii",
            "provider": "BALSA/Human Connectome Project",
            "right_file_id": "pkN9",
            "right_filename": "Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii",
            "scene_id": "WN56",
            "source_bytes_status": "operator_pinned_required",
            "study_id": "RVVG",
        },
        "WN56/RVVG source pair drift",
    )
    expected_roles = {
        "hcp_left_dlabel",
        "hcp_right_dlabel",
        "fslr32k_left_medialwall_roi",
        "fslr32k_left_sphere_to_fsaverage",
        "fslr32k_left_vaavg",
        "fslr32k_right_medialwall_roi",
        "fslr32k_right_sphere_to_fsaverage",
        "fslr32k_right_vaavg",
        "fsaverage10k_left_medialwall_roi",
        "fsaverage10k_left_sphere",
        "fsaverage10k_left_vaavg",
        "fsaverage10k_right_medialwall_roi",
        "fsaverage10k_right_sphere",
        "fsaverage10k_right_vaavg",
    }
    require(set(method["required_inputs"]) == expected_roles, "Lineage-B exact input-role set drift")
    provenance = method["method_provenance"]
    require(provenance["repository"] == "netneurolab/neuromaps", "neuromaps repository drift")
    require(provenance["commit"] == "ffcc2e0f657943ce00a1b6a968396f32250e495c", "neuromaps commit drift")
    require(provenance["transforms_blob_sha"] == "cddd03f2f2f6da94119732d57b4e4d0f1f1563bd", "neuromaps transforms blob drift")
    require(provenance["registry_blob_sha"] == "f56eebc42375c11b18d4a2fca6c9ea151e1b50af", "neuromaps registry blob drift")
    require(provenance["atlas_fetcher_blob_sha"] == "82259057dbeb847795b2b461699300e0c51f3b55", "neuromaps atlas fetcher blob drift")
    require(provenance["label_resample_method"] == "ADAP_BARY_AREA", "resample method drift")
    require(provenance["area_correction"] == "average_vertex_area_metrics", "area correction drift")
    require(provenance["target_mask_profile"] == "symthaea-positive-label-mask-v1", "target mask drift")
    require(provenance["license"] == "CC-BY-NC-SA-4.0", "neuromaps license metadata drift")
    require(provenance["citation_doi"] == "10.1038/s41592-022-01625-w", "neuromaps citation drift")
    require(method["template_bundles"]["fsLR32k"]["md5"] == "7932b4418f63d28935b5adf67150b16f", "fsLR32k bundle drift")
    require(method["template_bundles"]["fsaverage10k"]["md5"] == "c61384c271ee2e6b5449222281137414", "fsaverage10k bundle drift")

    independence = method["independence_contract"]
    require(independence["same_atlas_root_required"] is True, "same-atlas-root boundary drift")
    require(independence["execution_independence_requires_external_proof"] is True, "execution-independence boundary drift")
    require(independence["transform_method_distinct_from_mills"] is True, "transform-method distinction drift")
    require(independence["transform_implementation_family_independent"] is False, "implementation-family independence overclaim")
    require(independence["semantic_normalizer_independent"] is False, "semantic-normalizer independence overclaim")
    require(independence["independence_established_by_this_manifest"] is False, "manifest minted independence")
    require(independence["external_provenance_review_required"] is True, "external provenance review no longer required")

    for path in (
        "scripts/hcpmmp_neuromaps_common.py",
        "scripts/hcpmmp_neuromaps_gifti.py",
        "scripts/derive_hcpmmp1_neuromaps_lineage_b.py",
    ):
        no_match(path, r"(^|[^A-Za-z0-9_])(urllib|requests|curl|wget)([^A-Za-z0-9_]|$)", f"{path}: network acquisition surface detected")
    contains("scripts/hcpmmp_neuromaps_common.py", "must be distinct")
    no_match("scripts/derive_hcpmmp1_neuromaps_lineage_b.py", r"independence_established[\"']?\s*[:=]\s*true", "Lineage-B source mints independence")


def generator_custody_snapshot_contract() -> None:
    derive = "scripts/derive_hcpmmp1_neuromaps_lineage_b.py"
    contains(derive, "GENERATOR_FILE_KEYS={'common','gifti','derive'}")
    contains(derive, "generator_implementation_digest")
    contains(derive, "generator implementation changed during derivation")
    contains(derive, "verify_inputs(run)")
    contains("docs/neuroscience/HCPMMP1_NEUROMAPS_GENERATOR_PROVENANCE_V1.md", "exact Symthaea code bytes")
    contains("docs/neuroscience/HCPMMP1_NEUROMAPS_GENERATOR_PROVENANCE_V1.md", "does not itself prove that the code is scientifically valid")

    for needle in ("publish-lock", "os.O_EXCL", "0o600", "0o700", "os.rename", "BUNDLE_RECEIPT_PROFILE"):
        contains(derive, needle)
    no_match(derive, r"print\(json\.dumps\(evidence", "CLI prints the full evidence record")
    contains("docs/neuroscience/HCPMMP1_NEUROMAPS_LINEAGE_B_BUNDLE_CUSTODY_V1.md", "never deliberately truncates or replaces")
    contains("docs/neuroscience/HCPMMP1_NEUROMAPS_LINEAGE_B_BUNDLE_CUSTODY_V1.md", "does not establish")

    snapshot = "scripts/hcpmmp_neuromaps_execution_snapshot.py"
    for needle in (
        "REQUIRED_INPUT_ROLES",
        "_open_regular_source",
        "stat.S_ISREG",
        "os.O_EXCL",
        "0o400",
        "_cleanup_failed_snapshot",
        "destination_fd: int | None = None",
        "os.close(source_fd)",
    ):
        contains(snapshot, needle)
    no_match(snapshot, r"(^|[^A-Za-z0-9_])(subprocess|requests|urllib|socket|aiohttp)([^A-Za-z0-9_]|$)", "snapshot primitive gained execution/network authority")
    contains("docs/neuroscience/HCPMMP1_LINEAGE_B_SCIENTIFIC_INPUT_SNAPSHOT_V1.md", "not yet wired into `derive()`")


def main() -> int:
    try:
        quarantine_contract()
        atlas_contract()
        extractor_contract()
        lineage_b_contract()
        generator_custody_snapshot_contract()
    except (ContractError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"profile": PROFILE, "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
