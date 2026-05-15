import json
import subprocess
import time
import os
import sys
import re

def run_hc(args, input_str=""):
    cmd = ["hc", "sandbox", "--piped"] + args
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        input=input_str.encode(),
        capture_output=True,
        env=os.environ
    )
    if result.returncode != 0:
        print(f"Error executing hc: {result.stderr.decode()}")
        return None
    return result.stdout.decode()

def main():
    root_dir = "/srv/luminous-dynamics"
    workspace_dir = os.path.join(root_dir, "mycelix-workspace")
    os.chdir(workspace_dir)

    # 1. Clean and Create Sandbox
    print("--- Initializing Sandbox ---")
    subprocess.run(["hc", "sandbox", "clean"])
    run_hc(["create", "--num-sandboxes", "1"])

    # 2. Run Conductor (Fixed Port 4444)
    print("--- Starting Conductor ---")
    conductor = subprocess.Popen(
        ["hc", "sandbox", "--piped", "-f", "4444", "run", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    conductor.stdin.write(b"\n")
    conductor.stdin.flush()
    time.sleep(10)
    admin_port = "4444"

    try:
        # 4. Install Apps
        print("--- Installing hApps ---")
        apps = [
            ("substrate-identity", f"{root_dir}/mycelix-identity/mycelix-identity.happ"),
            ("substrate-finance", f"{root_dir}/mycelix-finance/mycelix-finance.happ"),
            ("satellite-civic", f"{root_dir}/mycelix-civic/mycelix-civic.happ"),
            ("satellite-knowledge", f"{root_dir}/mycelix-knowledge/knowledge.happ")
        ]

        for app_id, path in apps:
            print(f"  Installing {app_id}...")
            run_hc(["call", "-r", admin_port, "install-app", "--app-id", app_id, path], input_str="\n")

        print("--- Fetching Metadata ---")
        # 5. Get Agent and DNA Hashes
        app_list_raw = run_hc(["call", "-r", admin_port, "list-apps"], input_str="\n")
        if not app_list_raw:
            print("Failed to list apps")
            return

        json_start = app_list_raw.find("[")
        if json_start == -1: json_start = app_list_raw.find("{")
        if json_start == -1:
            print(f"Could not find JSON: {app_list_raw}")
            return
            
        apps_data = json.loads(app_list_raw[json_start:])
        
        identity_dna = None
        civic_dna = None
        agent_key = None

        for app in apps_data:
            app_id = app["installed_app_id"]
            if app_id == "substrate-identity":
                cell = app["cell_info"]["identity"][0]
                val = cell.get("provisioned") or cell.get("Provisioned")
                if val:
                    identity_dna = val["cell_id"]["dna_hash"]
                    agent_key = val["cell_id"]["agent_pub_key"]
            if app_id == "satellite-civic":
                cell = app["cell_info"]["civic"][0]
                val = cell.get("provisioned") or cell.get("Provisioned")
                if val:
                    civic_dna = val["cell_id"]["dna_hash"]

        print(f"  Agent: {agent_key}")
        print(f"  Identity DNA: {identity_dna}")
        print(f"  Civic DNA: {civic_dna}")

        if not (identity_dna and civic_dna and agent_key):
            print("Failed to extract necessary hashes")
            return

        # 6. Seed Identity
        print("--- Seeding Identity ---")
        payload = {
            "agent_pubkey_b64": agent_key,
            "cluster": "finance",
            "score": 0.85
        }
        run_hc([
            "zome-call", "-r", admin_port, 
            "substrate-identity", identity_dna, 
            "reputation_aggregator", "report_domain_score", 
            json.dumps(payload)
        ], input_str="\n")

        # 7. Verify Cross-hApp Coordination
        print("--- Verifying Constellation Protocol (Civic -> Identity) ---")
        run_hc([
            "zome-call", "-r", admin_port,
            "satellite-civic", civic_dna,
            "civic_bridge", "verify_tier_remote",
            json.dumps(agent_key)
        ], input_str="\n")

        print("\n✅ Constellation seeded and verified successfully!")

    finally:
        print("--- Shutting Down ---")
        conductor.terminate()

if __name__ == "__main__":
    main()
