#!/usr/bin/env python3
import os
import re

LUMINOUS_ROOT = "/srv/luminous-dynamics"

# Map of crate names to their absolute paths
crate_map = {}

def build_crate_map():
    print("Building crate map...")
    for root, dirs, files in os.walk(LUMINOUS_ROOT):
        if "Cargo.toml" in files:
            try:
                with open(os.path.join(root, "Cargo.toml"), 'r') as f:
                    content = f.read()
                    match = re.search(r'^name\s*=\s*"(.*?)"', content, re.MULTILINE)
                    if match:
                        crate_name = match.group(1)
                        crate_map[crate_name] = os.path.abspath(root)
            except Exception:
                continue
    print(f"Mapped {len(crate_map)} crates.")

def fix_manifests():
    print("Repairing manifest paths...")
    for root, dirs, files in os.walk(LUMINOUS_ROOT):
        if "Cargo.toml" in files:
            manifest_path = os.path.join(root, "Cargo.toml")
            if ".git" in manifest_path or "target" in manifest_path:
                continue
                
            try:
                with open(manifest_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                changed = False
                for line in lines:
                    # Look for path = "..." pattern
                    match = re.search(r'(.*?path\s*=\s*")(.*?)(".*)', line)
                    if match:
                        prefix, rel_path, suffix = match.groups()
                        
                        # Only fix if it looks like a local crate dependency (starts with . or is just a name)
                        if rel_path.startswith(".") or "/" in rel_path:
                            abs_target = os.path.abspath(os.path.join(root, rel_path))
                            
                            # Check if it's broken
                            if not os.path.exists(abs_target):
                                # It's broken. Try to find the crate by name if it's a symthaea or symtropy crate
                                target_name = os.path.basename(rel_path.rstrip("/"))
                                if target_name in crate_map:
                                    new_abs_target = crate_map[target_name]
                                    new_rel_path = os.path.relpath(new_abs_target, root)
                                    line = f'{prefix}{new_rel_path}{suffix}\n'
                                    changed = True
                                    print(f"  Fixed {target_name} in {os.path.relpath(manifest_path, LUMINOUS_ROOT)}")
                    new_lines.append(line)
                
                if changed:
                    with open(manifest_path, 'w') as f:
                        f.writelines(new_lines)
            except Exception as e:
                print(f"Error processing {manifest_path}: {e}")

if __name__ == "__main__":
    build_crate_map()
    fix_manifests()
    print("Repair complete.")
