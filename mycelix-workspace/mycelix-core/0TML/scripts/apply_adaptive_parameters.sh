#!/bin/bash
# Apply Adaptive Byzantine Detection Parameters
#
# Usage:
#   source scripts/apply_adaptive_parameters.sh <dataset_name>
#
# Datasets: cifar10, emnist_balanced, breast_cancer
#
# This script loads dataset-specific parameters from .env.adaptive
# based on research findings that Byzantine detection parameters
# must be adapted to dataset characteristics.

set -e

DATASET="${1:-}"

if [ -z "$DATASET" ]; then
    echo "❌ ERROR: Dataset name required"
    echo "Usage: source $0 <dataset_name>"
    echo "Datasets: cifar10, emnist_balanced, breast_cancer"
    return 1 2>/dev/null || exit 1
fi

# Load adaptive parameter configurations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTIVE_ENV="$SCRIPT_DIR/../.env.adaptive"

if [ ! -f "$ADAPTIVE_ENV" ]; then
    echo "❌ ERROR: Adaptive parameters file not found: $ADAPTIVE_ENV"
    return 1 2>/dev/null || exit 1
fi

# Source the adaptive parameters file
source "$ADAPTIVE_ENV"

# Normalize dataset name
DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]' | tr '-' '_')

# Apply dataset-specific parameters
case "$DATASET" in
    cifar10|CIFAR10)
        export BEHAVIOR_RECOVERY_THRESHOLD="$CIFAR10_BEHAVIOR_RECOVERY_THRESHOLD"
        export BEHAVIOR_RECOVERY_BONUS="$CIFAR10_BEHAVIOR_RECOVERY_BONUS"
        export LABEL_SKEW_COS_MIN="$CIFAR10_LABEL_SKEW_COS_MIN"
        export LABEL_SKEW_COS_MAX="$CIFAR10_LABEL_SKEW_COS_MAX"
        export COMMITTEE_REJECT_FLOOR="$CIFAR10_COMMITTEE_REJECT_FLOOR"
        export REPUTATION_FLOOR="$CIFAR10_REPUTATION_FLOOR"
        echo "✅ Applied CIFAR-10 parameters"
        ;;

    emnist_balanced|EMNIST_BALANCED)
        export BEHAVIOR_RECOVERY_THRESHOLD="$EMNIST_BALANCED_BEHAVIOR_RECOVERY_THRESHOLD"
        export BEHAVIOR_RECOVERY_BONUS="$EMNIST_BALANCED_BEHAVIOR_RECOVERY_BONUS"
        export LABEL_SKEW_COS_MIN="$EMNIST_BALANCED_LABEL_SKEW_COS_MIN"
        export LABEL_SKEW_COS_MAX="$EMNIST_BALANCED_LABEL_SKEW_COS_MAX"
        export COMMITTEE_REJECT_FLOOR="$EMNIST_BALANCED_COMMITTEE_REJECT_FLOOR"
        export REPUTATION_FLOOR="$EMNIST_BALANCED_REPUTATION_FLOOR"
        echo "✅ Applied EMNIST Balanced parameters"
        ;;

    breast_cancer|BREAST_CANCER)
        export BEHAVIOR_RECOVERY_THRESHOLD="$BREAST_CANCER_BEHAVIOR_RECOVERY_THRESHOLD"
        export BEHAVIOR_RECOVERY_BONUS="$BREAST_CANCER_BEHAVIOR_RECOVERY_BONUS"
        export LABEL_SKEW_COS_MIN="$BREAST_CANCER_LABEL_SKEW_COS_MIN"
        export LABEL_SKEW_COS_MAX="$BREAST_CANCER_LABEL_SKEW_COS_MAX"
        export COMMITTEE_REJECT_FLOOR="$BREAST_CANCER_COMMITTEE_REJECT_FLOOR"
        export REPUTATION_FLOOR="$BREAST_CANCER_REPUTATION_FLOOR"
        echo "✅ Applied Breast Cancer parameters"
        ;;

    *)
        echo "⚠️  WARNING: Unknown dataset '$DATASET', using default parameters"
        export BEHAVIOR_RECOVERY_THRESHOLD="$DEFAULT_BEHAVIOR_RECOVERY_THRESHOLD"
        export BEHAVIOR_RECOVERY_BONUS="$DEFAULT_BEHAVIOR_RECOVERY_BONUS"
        export LABEL_SKEW_COS_MIN="$DEFAULT_LABEL_SKEW_COS_MIN"
        export LABEL_SKEW_COS_MAX="$DEFAULT_LABEL_SKEW_COS_MAX"
        export COMMITTEE_REJECT_FLOOR="$DEFAULT_COMMITTEE_REJECT_FLOOR"
        export REPUTATION_FLOOR="$DEFAULT_REPUTATION_FLOOR"
        echo "✅ Applied default parameters"
        ;;
esac

# Display applied parameters
echo ""
echo "Active Parameters for $DATASET:"
echo "  BEHAVIOR_RECOVERY_THRESHOLD: $BEHAVIOR_RECOVERY_THRESHOLD"
echo "  BEHAVIOR_RECOVERY_BONUS: $BEHAVIOR_RECOVERY_BONUS"
echo "  LABEL_SKEW_COS_MIN: $LABEL_SKEW_COS_MIN"
echo "  LABEL_SKEW_COS_MAX: $LABEL_SKEW_COS_MAX"
echo "  COMMITTEE_REJECT_FLOOR: $COMMITTEE_REJECT_FLOOR"
echo "  REPUTATION_FLOOR: $REPUTATION_FLOOR"
echo ""
