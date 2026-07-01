#!/usr/bin/env bash
# ==============================================================================
# SYMTHAEA REAL-TIME CONTINUUM HARNESS
# Enforces strict CPU Affinity and FIFO Scheduling for zero-jitter execution.
# ==============================================================================

# Ensure the release binary is compiled before we attempt to elevate it
echo "⚙️  Verifying release compilation..."
cargo build --release --features full

if [ "$EUID" -ne 0 ]; then
  echo "⚠️  WARNING: Real-time scheduler elevation requires root (sudo) privileges."
  echo "   Falling back to standard Completely Fair Scheduler (CFS)..."
  echo "----------------------------------------------------------------------"
  ./target/release/symthaea
  exit 0
fi

echo "🚀 IGNITION: Initializing Symthaea on isolated CPU cores with FIFO Priority..."
echo "   - Binding process to CPU Cores: 2, 3"
echo "   - Elevating thread policy to SCHED_FIFO (Priority 90)"
echo "----------------------------------------------------------------------"

# Execute the binary using taskset (affinity) and chrt (real-time scheduling)
taskset -c 2,3 chrt -f 90 ./target/release/symthaea
