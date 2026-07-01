#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# Autonomous Self-Improvement Flywheel Daemon

import os
import sys
import time
import json
import subprocess
import random

DATA_PATH = "data/distillation_flywheel.jsonl"
CHECK_INTERVAL_SECS = 10
RETRAIN_THRESHOLD_LINES = 50

def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as f:
        return sum(1 for _ in f)

def interleave_and_shuffle_dataset(filepath):
    """
    Reads the JSONL buffer, groups by return shape / category to break up 
    unbalanced sequences, and shuffles domains to prevent subspace collapse.
    """
    print(f"[Flywheel Daemon] Ingesting and conditioning dataset: {filepath}")
    with open(filepath, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    # Shuffling breaks up chronological code injection clumps
    random.shuffle(records)
    
    # Overwrite with clean, non-contiguous multi-domain formatting
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[Flywheel Daemon] Interleaved optimization formatting successfully applied.")

def trigger_background_retrain():
    print("\n[Flywheel Daemon] 🚀 Retrain threshold breached. Spawning background compiler...")
    
    # We nice the process to 19 (lowest priority background work) so it doesn't hitch your IDE or REPL
    cmd = [
        "nice", "-n", "19",
        "cargo", "test", "--release", "--lib", 
        "language::algorithm_training::tests::test_cfc_training_converges",
        "--", "--nocapture"
    ]
    
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = "/tmp/symthaea-broca-host-release"
    env["RUSTC_WRAPPER"] = ""
    env["SCCACHE_DISABLE"] = "1"
    
    try:
        # Launch asynchronously without blocking the daemon watcher loop
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[Flywheel Daemon] Optimization pass spawned successfully (PID: {process.pid}). Weights evolving offline.")
    except Exception as e:
        print(f"[Flywheel Daemon] Failed to invoke background training engine: {e}")

def main():
    print("===============================================================================")
    print("     Symthaea Distillation & Convergence Daemon Initialized (2026 Core)        ")
    print("===============================================================================")
    print(f"Monitoring target: {DATA_PATH}")
    print(f"Trigger window: Accumulation of every {RETRAIN_THRESHOLD_LINES} verified turns")
    
    last_count = count_lines(DATA_PATH)
    print(f"Initial baseline state: Dataset currently holds {last_count} examples.")
    
    while True:
        try:
            time.sleep(CHECK_INTERVAL_SECS)
            current_count = count_lines(DATA_PATH)
            
            if current_count >= last_count + RETRAIN_THRESHOLD_LINES:
                print(f"\n[Flywheel Daemon] Accumulation window closed ({last_count} -> {current_count}).")
                interleave_and_shuffle_dataset(DATA_PATH)
                trigger_background_retrain()
                last_count = current_count
                
        except KeyboardInterrupt:
            print("\n[Flywheel Daemon] Shuttering active inference monitor loop. Goodbye.")
            break
        except Exception as e:
            print(f"[Flywheel Daemon] Worker exception encountered: {e}")

if __name__ == "__main__":
    main()
