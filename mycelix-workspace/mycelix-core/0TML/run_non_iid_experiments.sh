#!/usr/bin/env bash
# Launch all Non-IID experiments for Day 5
# Usage: ./run_non_iid_experiments.sh [experiment_type]
#   experiment_type: "all" (default), "dirichlet_0.1", "dirichlet_0.5", "pathological"

set -e

echo "🔬 Non-IID Experiment Launcher"
echo "==============================="
echo ""

EXPERIMENT_TYPE="${1:-all}"

# Function to launch experiment in background
launch_experiment() {
    local config=$1
    local name=$2
    local log=$3

    echo "🚀 Launching: $name"
    echo "📝 Config: $config"
    echo "📊 Log: $log"

    nohup nix develop --command python -u experiments/runner.py \
        --config "$config" &> "$log" 2>&1 &

    local pid=$!
    echo "✅ Started with PID: $pid"
    echo ""

    # Wait a moment and check if it started successfully
    sleep 2
    if ps -p $pid > /dev/null; then
        echo "✅ $name running successfully"
    else
        echo "❌ $name failed to start - check $log"
        return 1
    fi
    echo ""
}

case "$EXPERIMENT_TYPE" in
    "dirichlet_0.1")
        echo "📊 Running: Dirichlet α=0.1 (Highly Non-IID)"
        launch_experiment \
            "experiments/configs/mnist_non_iid_dirichlet_0.1.yaml" \
            "Dirichlet α=0.1" \
            "/tmp/mnist_non_iid_0.1.log"
        ;;

    "dirichlet_0.5")
        echo "📊 Running: Dirichlet α=0.5 (Moderately Non-IID)"
        launch_experiment \
            "experiments/configs/mnist_non_iid_dirichlet_0.5.yaml" \
            "Dirichlet α=0.5" \
            "/tmp/mnist_non_iid_0.5.log"
        ;;

    "pathological")
        echo "📊 Running: Pathological (Extreme Non-IID)"
        launch_experiment \
            "experiments/configs/mnist_non_iid_pathological.yaml" \
            "Pathological" \
            "/tmp/mnist_non_iid_pathological.log"
        ;;

    "all")
        echo "📊 Running: ALL Non-IID Experiments"
        echo ""

        # Launch all three experiments
        launch_experiment \
            "experiments/configs/mnist_non_iid_dirichlet_0.1.yaml" \
            "Dirichlet α=0.1" \
            "/tmp/mnist_non_iid_0.1.log"

        launch_experiment \
            "experiments/configs/mnist_non_iid_dirichlet_0.5.yaml" \
            "Dirichlet α=0.5" \
            "/tmp/mnist_non_iid_0.5.log"

        launch_experiment \
            "experiments/configs/mnist_non_iid_pathological.yaml" \
            "Pathological" \
            "/tmp/mnist_non_iid_pathological.log"
        ;;

    *)
        echo "❌ Unknown experiment type: $EXPERIMENT_TYPE"
        echo ""
        echo "Usage: $0 [experiment_type]"
        echo "  experiment_type: all, dirichlet_0.1, dirichlet_0.5, pathological"
        exit 1
        ;;
esac

echo "================================"
echo "✅ All experiments launched!"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f /tmp/mnist_non_iid_0.1.log"
echo "   tail -f /tmp/mnist_non_iid_0.5.log"
echo "   tail -f /tmp/mnist_non_iid_pathological.log"
echo ""
echo "⏱️  Expected completion time:"
echo "   - Dirichlet α=0.1: ~60 min per baseline (150 rounds)"
echo "   - Dirichlet α=0.5: ~60 min per baseline (150 rounds)"
echo "   - Pathological: ~80 min per baseline (200 rounds)"
echo ""
echo "🎯 Total estimated time: ~10-12 hours for all experiments"
