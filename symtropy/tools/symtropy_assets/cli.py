import argparse
import sys
import os
import sqlite3
import paths
import json
import yaml
from license_gate import validate_manifest
from observer import screen_asset_thumbnail
from neural_link import generate_world_blueprint
from registry_manager import ingest_manifest, get_asset, update_asset_audit_status, register_file, get_assets_needing_conversion
from exporters.export_pack import generate_export_reports
from review_manager import get_pending_review, review_asset
from style_gate import validate_style
from status_manager import get_registry_status
from migration_manager import run_migrations

def run_conversion(asset_id, filepath):
    script_path = os.path.join(os.path.dirname(__file__), 'converters', 'normalize_glb_basic.py')
    import subprocess
    cmd = ["blender", "--background", "--python", script_path, "--", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        base_path_no_ext = os.path.splitext(filepath)[0]
        report_path = f"{base_path_no_ext}_tech_report.json"
        normalized_path = f"{base_path_no_ext}_normalized.glb"
        thumb_path = f"{base_path_no_ext}_thumb.png"

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            # AI Screening
            is_approved, reason = screen_asset_thumbnail(thumb_path, report)
            if not is_approved:
                update_asset_audit_status(asset_id, tech_status="QUARANTINE_VISION")
                return False, f"Asset {asset_id} failed AI screening: {reason}"

            # Update technical status
            update_asset_audit_status(asset_id, tech_status=report["recommended_status"])

            # Register physical properties if available
            if "physical_properties" in report and report["physical_properties"]:
                # Take the first mesh's properties as the object reference
                first_mesh = list(report["physical_properties"].values())[0]
                props = first_mesh
                conn = sqlite3.connect(paths.get_registry_path())
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE assets SET mass_kg = ?, com_x = ?, com_y = ?, com_z = ?,
                                     inertia_ixx = ?, inertia_iyy = ?, inertia_izz = ?
                    WHERE id = ?
                """, (props["mass"], props["com"][0], props["com"][1], props["com"][2],
                      props["inertia"][0], props["inertia"][4], props["inertia"][8], asset_id))
                conn.commit()
                conn.close()

            # Register the optimized file
            if os.path.exists(normalized_path):

                asset_root = paths.get_asset_root()
                rel_normalized = os.path.relpath(normalized_path, asset_root)
                register_file(asset_id, "optimized", rel_normalized)
                return True, f"Asset {asset_id} normalized, audited (Status: {report['recommended_status']}), and registered as optimized."
            else:
                return False, f"Asset {asset_id} audited but normalized file not found at {normalized_path}."
        else:
            return False, "Conversion succeeded but report not found."
    else:
        return False, f"Conversion failed: {result.stderr}"

def main():
    parser = argparse.ArgumentParser(description="Symtropy Asset Foundry CLI")
    parser.add_argument("--asset-root", help="Path to symtropy-assets directory")
    parser.add_argument("--db", help="Path to assets.sqlite registry")

    subparsers = parser.add_subparsers(dest="command")

    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate source manifests")
    validate_parser.add_argument("manifest", help="Path to manifest file")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and register assets")
    ingest_parser.add_argument("manifest", help="Path to manifest file")

    # Audit Style
    audit_style_parser = subparsers.add_parser("audit-style", help="Perform style audit")
    audit_style_parser.add_argument("manifest", help="Path to manifest file")

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Normalize and audit asset")
    convert_parser.add_argument("filepath", help="Path to raw asset file")
    convert_parser.add_argument("--asset-id", required=True, help="Asset ID for registry update")

    # Audit All
    subparsers.add_parser("audit-all", help="Normalize and audit all pending assets")

    # Gallery
    gallery_parser = subparsers.add_parser("gallery", help="Generate review gallery")
    gallery_parser.add_argument("--output", default="gallery.md", help="Output file")

    # Neural Link
    neural_parser = subparsers.add_parser("neural-link", help="Generate world blueprint from prompt")
    neural_parser.add_argument("prompt", help="Creative world prompt")
    neural_parser.add_argument("--output", default="world_blueprint.yaml", help="Blueprint output file")

    # Registry
    registry_parser = subparsers.add_parser("registry", help="Registry management")
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command")
    registry_subparsers.add_parser("status", help="Show registry status")
    registry_subparsers.add_parser("migrate", help="Run database migrations")

    # Export
    export_parser = subparsers.add_parser("export", help="Export Bevy-ready pack")
    export_parser.add_argument("--pack", required=True, help="Pack ID")
    export_parser.add_argument("--target", required=True, help="Target directory")
    export_parser.add_argument("--biome", help="Filter by biome")

    # Review
    review_parser = subparsers.add_parser("review", help="Manual asset review")
    review_parser.add_argument("--list", action="store_true", help="List pending reviews")
    review_parser.add_argument("--approve", metavar="ASSET_ID", help="Approve asset")
    review_parser.add_argument("--reviewer", required=True, help="Reviewer name")
    review_parser.add_argument("--notes", help="Review notes")

    args = parser.parse_args()

    # Apply path overrides
    if args.asset_root:
        paths.set_asset_root(os.path.abspath(args.asset_root))
    if args.db:
        paths.set_registry_path(os.path.abspath(args.db))

    if args.command == "validate":
        status, message = validate_manifest(args.manifest)
        print(f"Status: {status} - {message}")
    elif args.command == "ingest":
        status, message = validate_manifest(args.manifest)
        if status in ["APPROVED_CC0", "APPROVED_ATTRIBUTION_REQUIRED", "QUARANTINE_REVIEW"]:
            asset_id = ingest_manifest(args.manifest, status)
            print(f"Ingested {asset_id} with status {status}")
        else:
            print(f"Ingestion failed: {message}")
    elif args.command == "audit-style":
        with open(args.manifest, 'r') as f:
            manifest = yaml.safe_load(f)
        report = validate_style(manifest)
        update_asset_audit_status(manifest.get("id"), style_status=report["style_status"])
        print(json.dumps(report, indent=2))
    elif args.command == "convert":
        success, message = run_conversion(args.asset_id, args.filepath)
        print(message)
    elif args.command == "audit-all":
        unconverted = get_assets_needing_conversion()
        print(f"Found {len(unconverted)} assets needing conversion.")
        asset_root = paths.get_asset_root()
        for asset_id, rel_path in unconverted:
            abs_path = os.path.join(asset_root, rel_path)
            print(f"Converting {asset_id} ({abs_path})...")
            success, message = run_conversion(asset_id, abs_path)
            print(message)
    elif args.command == "gallery":
        conn = sqlite3.connect(paths.get_registry_path())
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, technical_status FROM assets WHERE status = 'PENDING'")
        pending = cursor.fetchall()

        with open(args.output, "w") as f:
            f.write("# Foundry Review Gallery\n\n")
            for aid, title, tech in pending:
                f.write(f"## {title} ({aid})\n")
                f.write(f"Technical Status: {tech}\n\n")
                # Look for thumb
                thumb_path = os.path.join("raw_vault", aid.split('.')[-1] + "_thumb.png") # Simplified
                f.write(f"![Thumbnail]({thumb_path})\n\n")
                f.write("---\n\n")
        print(f"Generated gallery at {args.output}")
    elif args.command == "neural-link":
        generate_world_blueprint(args.prompt, args.output)
    elif args.command == "registry":
        if args.registry_command == "status":
            print(json.dumps(get_registry_status(), indent=2))
        elif args.registry_command == "migrate":
            run_migrations()
    elif args.command == "export":
        msg = generate_export_reports(args.pack, args.target, biome_filter=args.biome)
        print(msg)
    elif args.command == "review":
        if args.list:
            print(get_pending_review())
        elif args.approve:
            print(review_asset(args.approve, "APPROVED_CC0", args.reviewer, args.notes))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
