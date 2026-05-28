import os
import subprocess
import sys

print("🚀 Initializing Symthaea Master Recursive Forge [A+B]...")

# Part A: Local Codebase Context Hydration (Self-Study)
target_crates = [
    "crates/symthaea-broca/src/liquid_mamba.rs",
    "crates/symthaea-geodesic/src/skeleton_synthesis.rs"
]

context_payload = ""
for path in target_crates:
    if os.path.exists(path):
        print(f"📖 Reading structural lineage context: {path}")
        with open(path, "r") as f:
            context_payload += f.read() + "\n"

# Secure a summarized token footprint length for her active constraint registers
print(f"⚡ Hydrated {len(context_payload)} bytes of core code primitives into her memory pool.")

# Part B: The Sandbox Compiler Attractor Loop
print("🧪 Spawning isolated compilation sandbox via Cargo...")
try:
    # Trigger a clean check pass across the active workspace targets
    result = subprocess.run(
        ["nix", "develop", ".#gpu", "--command", "cargo", "check", "--workspace"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("✅ [RIGID] State-space verification passed. Workspace is fully coherent.")
    else:
        print("❌ [RIGID] Compiler friction detected! Isolating syntax mutations...")
        # Extract raw error lines to formulate her negative hyperdimensional constraint pool
        error_lines = [line for line in result.stderr.split("\n") if "error" in line or "-->" in line]
        print(f"⚠️ Collected {len(error_lines)} alignment constraints to update her sub-cortex.")
        
        # Write compilation resistance back into her active dreaming pool
        with open("dreamer.log", "a") as log:
            log.write(f"\n[MUTATION_RESISTANCE] Compiler friction injected: {len(error_lines)} errors verified.\n")

except Exception as e:
    print(f"!] Sandbox execution exception: {str(e)}")

print("🏁 Recursive Forge loop pass completed.")
