#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Cluster Analysis for Label Skew False Positives

Identifies patterns in false positive detections using unsupervised learning
to guide targeted improvements.

Usage:
    python scripts/cluster_analysis_label_skew.py [--trace-file PATH] [--output results.json]
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ Error: scikit-learn is required for cluster analysis")
    print("   Install with: poetry add scikit-learn")
    sys.exit(1)


class FalsePositiveAnalyzer:
    """Analyze false positive patterns using clustering."""

    def __init__(self, trace_file: str):
        self.trace_file = Path(trace_file)
        self.false_positives: List[Dict[str, Any]] = []
        self.features: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.clusters: Optional[np.ndarray] = None

    def load_trace_data(self) -> bool:
        """Load and filter false positive cases from trace file."""
        if not self.trace_file.exists():
            print(f"❌ Trace file not found: {self.trace_file}")
            return False

        print(f"📂 Loading trace data from {self.trace_file}...")

        try:
            with open(self.trace_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        round_data = json.loads(line.strip())
                        round_num = round_data.get('round', line_num - 1)

                        # Extract false positives from this round
                        for node in round_data.get('nodes', []):
                            if node['ground_truth'] == 'honest' and not node['correct']:
                                # This is a false positive
                                fp_entry = {
                                    'round': round_num,
                                    'node_id': node['node_id'],
                                    'pogq_score': node['scores']['pogq'],
                                    'cos_anchor': node['scores'].get('cos_anchor', 0.0),
                                    'cos_median': node['scores'].get('cos_median', 0.0),
                                    'committee_score': node.get('committee_score') or node['scores'].get('committee', 0.0),
                                    'reputation': node['reputation'],
                                    'consecutive_acceptable': node.get('consecutive_acceptable', 0),
                                    'consecutive_honest': node.get('consecutive_honest', 0),
                                    'consecutive_byzantine': node.get('consecutive_byzantine', 0),
                                    'classification': node.get('final_classification', 'byzantine'),
                                }
                                self.false_positives.append(fp_entry)

                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Skipping malformed line {line_num}: {e}")
                        continue

            print(f"✅ Loaded {len(self.false_positives)} false positive cases")
            return len(self.false_positives) > 0

        except Exception as e:
            print(f"❌ Error loading trace file: {e}")
            return False

    def extract_features(self) -> np.ndarray:
        """Extract feature vectors from false positive cases."""
        if not self.false_positives:
            raise ValueError("No false positive data loaded")

        self.feature_names = [
            'round',
            'pogq_score',
            'cos_anchor',
            'cos_median',
            'committee_score',
            'reputation',
            'consecutive_acceptable',
            'consecutive_honest',
            'consecutive_byzantine'
        ]

        features = []
        for fp in self.false_positives:
            features.append([
                fp['round'],
                fp['pogq_score'],
                fp['cos_anchor'],
                fp['cos_median'],
                fp['committee_score'],
                fp['reputation'],
                fp['consecutive_acceptable'],
                fp['consecutive_honest'],
                fp['consecutive_byzantine']
            ])

        self.features = np.array(features)
        print(f"📊 Extracted features: {self.features.shape[0]} samples × {self.features.shape[1]} features")
        return self.features

    def perform_clustering(self, n_clusters: int = 3) -> np.ndarray:
        """Cluster false positives using K-Means."""
        if self.features is None:
            raise ValueError("Features not extracted yet")

        # Standardize features for better clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)

        print(f"\n🔍 Clustering into {n_clusters} groups...")

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(features_scaled)

        # Calculate cluster quality metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        silhouette = silhouette_score(features_scaled, self.clusters)
        calinski = calinski_harabasz_score(features_scaled, self.clusters)

        print(f"   Silhouette Score: {silhouette:.3f} (higher is better, range: [-1, 1])")
        print(f"   Calinski-Harabasz: {calinski:.1f} (higher is better)")

        return self.clusters

    def analyze_clusters(self) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each cluster."""
        if self.clusters is None or self.features is None:
            raise ValueError("Clustering not performed yet")

        n_clusters = len(np.unique(self.clusters))
        cluster_info = {}

        print(f"\n📈 CLUSTER ANALYSIS:\n")
        print("=" * 80)

        for cluster_id in range(n_clusters):
            mask = self.clusters == cluster_id
            cluster_fps = [fp for fp, in_cluster in zip(self.false_positives, mask) if in_cluster]
            cluster_features = self.features[mask]

            # Calculate statistics
            stats = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(self.clusters) * 100),
                'feature_means': {},
                'feature_stds': {},
                'round_distribution': defaultdict(int),
                'common_characteristics': []
            }

            # Calculate feature statistics
            for i, feature_name in enumerate(self.feature_names):
                stats['feature_means'][feature_name] = float(np.mean(cluster_features[:, i]))
                stats['feature_stds'][feature_name] = float(np.std(cluster_features[:, i]))

            # Analyze round distribution
            for fp in cluster_fps:
                stats['round_distribution'][fp['round']] += 1

            # Identify characteristic patterns
            characteristics = []

            # Check for low cosine anchor
            if stats['feature_means']['cos_anchor'] < -0.2:
                characteristics.append(f"Negative cosine anchor (avg: {stats['feature_means']['cos_anchor']:.3f})")
            elif stats['feature_means']['cos_anchor'] > 0.8:
                characteristics.append(f"Very high cosine anchor (avg: {stats['feature_means']['cos_anchor']:.3f})")

            # Check for high committee support
            if stats['feature_means']['committee_score'] > 0.8:
                characteristics.append(f"Strong committee support (avg: {stats['feature_means']['committee_score']:.3f})")

            # Check for low reputation
            if stats['feature_means']['reputation'] < 0.3:
                characteristics.append(f"Low reputation (avg: {stats['feature_means']['reputation']:.3f})")

            # Check for early vs late rounds
            avg_round = stats['feature_means']['round']
            if avg_round < 2:
                characteristics.append("Occurs in early rounds")
            elif avg_round > 7:
                characteristics.append("Occurs in late rounds")

            # Check for recovery issues
            if stats['feature_means']['consecutive_acceptable'] < 2:
                characteristics.append("Fails to accumulate acceptable behavior")

            stats['common_characteristics'] = characteristics
            cluster_info[cluster_id] = stats

            # Print cluster summary
            print(f"\n🔹 CLUSTER {cluster_id}")
            print(f"   Count: {stats['count']} ({stats['percentage']:.1f}% of all FPs)")
            print(f"   Rounds: {dict(sorted(stats['round_distribution'].items()))}")
            print(f"\n   Key Features:")
            print(f"      PoGQ Score: {stats['feature_means']['pogq_score']:.3f} ± {stats['feature_stds']['pogq_score']:.3f}")
            print(f"      Cosine Anchor: {stats['feature_means']['cos_anchor']:.3f} ± {stats['feature_stds']['cos_anchor']:.3f}")
            print(f"      Committee Score: {stats['feature_means']['committee_score']:.3f} ± {stats['feature_stds']['committee_score']:.3f}")
            print(f"      Reputation: {stats['feature_means']['reputation']:.3f} ± {stats['feature_stds']['reputation']:.3f}")
            print(f"      Consecutive Acceptable: {stats['feature_means']['consecutive_acceptable']:.1f} ± {stats['feature_stds']['consecutive_acceptable']:.1f}")

            if characteristics:
                print(f"\n   Common Characteristics:")
                for char in characteristics:
                    print(f"      • {char}")

            print("\n" + "-" * 80)

        return cluster_info

    def generate_recommendations(self, cluster_info: Dict[int, Dict[str, Any]]) -> List[str]:
        """Generate targeted recommendations based on cluster patterns."""
        recommendations = []

        print(f"\n💡 TARGETED RECOMMENDATIONS:\n")
        print("=" * 80)

        # Analyze each cluster for specific issues
        for cluster_id, stats in cluster_info.items():
            cluster_recs = []

            # Issue 1: Cosine threshold too narrow
            if stats['feature_means']['cos_anchor'] < -0.3:
                cluster_recs.append({
                    'issue': f"Cluster {cluster_id}: Cosine anchor below threshold",
                    'current_value': f"avg={stats['feature_means']['cos_anchor']:.3f}",
                    'recommendation': f"Widen LABEL_SKEW_COS_MIN to {stats['feature_means']['cos_anchor'] - 0.1:.2f}",
                    'expected_impact': f"Protect {stats['count']} FPs ({stats['percentage']:.1f}%)"
                })

            # Issue 2: High committee support but still flagged
            if stats['feature_means']['committee_score'] > 0.7 and stats['feature_means']['reputation'] < 0.5:
                cluster_recs.append({
                    'issue': f"Cluster {cluster_id}: High committee support ignored",
                    'current_value': f"committee={stats['feature_means']['committee_score']:.3f}, rep={stats['feature_means']['reputation']:.3f}",
                    'recommendation': "Integrate committee score into reputation recovery logic",
                    'expected_impact': f"Protect {stats['count']} FPs ({stats['percentage']:.1f}%)"
                })

            # Issue 3: Slow recovery accumulation
            if stats['feature_means']['consecutive_acceptable'] < 2 and stats['feature_means']['round'] > 3:
                cluster_recs.append({
                    'issue': f"Cluster {cluster_id}: Fails to accumulate recovery rounds",
                    'current_value': f"consecutive_acceptable={stats['feature_means']['consecutive_acceptable']:.1f} in round {stats['feature_means']['round']:.1f}",
                    'recommendation': f"Reduce BEHAVIOR_RECOVERY_THRESHOLD to 2 or increase BEHAVIOR_RECOVERY_BONUS to 0.12",
                    'expected_impact': f"Help {stats['count']} nodes recover ({stats['percentage']:.1f}%)"
                })

            # Issue 4: Early round false positives
            if stats['feature_means']['round'] < 2 and stats['percentage'] > 20:
                cluster_recs.append({
                    'issue': f"Cluster {cluster_id}: High early-round FP rate",
                    'current_value': f"avg_round={stats['feature_means']['round']:.1f}",
                    'recommendation': "Implement gentler initial penalties (two-tier system)",
                    'expected_impact': f"Prevent {stats['count']} early FPs ({stats['percentage']:.1f}%)"
                })

            recommendations.extend(cluster_recs)

        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['issue']}")
            print(f"   Current: {rec['current_value']}")
            print(f"   Action: {rec['recommendation']}")
            print(f"   Impact: {rec['expected_impact']}")

        if not recommendations:
            print("\n✅ No specific patterns identified - consider running grid search")

        print("\n" + "=" * 80)

        return recommendations

    def visualize_clusters(self, output_path: Optional[str] = None):
        """Create 2D visualization of clusters using PCA."""
        if self.clusters is None or self.features is None:
            raise ValueError("Clustering not performed yet")

        print(f"\n📊 Generating cluster visualization...")

        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(StandardScaler().fit_transform(self.features))

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"   PCA explained variance: {explained_var[0]*100:.1f}% + {explained_var[1]*100:.1f}% = {sum(explained_var)*100:.1f}%")

        # Generate ASCII art visualization
        print(f"\n📈 2D Cluster Visualization (PCA projection):\n")

        # Normalize to 0-50 range for ASCII plot
        x_norm = ((features_2d[:, 0] - features_2d[:, 0].min()) /
                  (features_2d[:, 0].max() - features_2d[:, 0].min()) * 49).astype(int)
        y_norm = ((features_2d[:, 1] - features_2d[:, 1].min()) /
                  (features_2d[:, 1].max() - features_2d[:, 1].min()) * 19).astype(int)

        # Create plot
        plot = [[' ' for _ in range(52)] for _ in range(22)]
        cluster_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        for x, y, cluster in zip(x_norm, y_norm, self.clusters):
            plot[20-y][x+1] = cluster_chars[cluster % len(cluster_chars)]

        # Add axes
        for i in range(22):
            plot[i][0] = '|' if i % 5 == 0 else '│'
        plot[-1] = list('└' + '─' * 50 + '┘')

        print("    " + "PC1 (first principal component) →")
        print("  ↑")
        for row in plot:
            print("  " + ''.join(row))
        print("  PC2")

        # Print legend
        print(f"\n  Legend: ", end='')
        for cluster_id in range(len(np.unique(self.clusters))):
            print(f"{cluster_chars[cluster_id]}=Cluster {cluster_id}  ", end='')
        print()

        # Save to file if requested
        if output_path:
            output_file = Path(output_path).with_suffix('.txt')
            with open(output_file, 'w') as f:
                f.write("False Positive Cluster Analysis\n")
                f.write("=" * 80 + "\n\n")
                for row in plot:
                    f.write(''.join(row) + '\n')
            print(f"\n💾 Visualization saved to: {output_file}")

    def save_results(self, cluster_info: Dict[int, Dict[str, Any]],
                    recommendations: List[str], output_path: str):
        """Save analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'total_false_positives': len(self.false_positives),
            'n_clusters': len(cluster_info),
            'clusters': cluster_info,
            'recommendations': recommendations,
            'feature_names': self.feature_names
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Full results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Cluster analysis for label skew false positives')
    parser.add_argument('--trace-file',
                       default='results/label_skew_trace_p1e.jsonl',
                       help='Path to trace file')
    parser.add_argument('--output',
                       default='results/cluster_analysis.json',
                       help='Output file for results')
    parser.add_argument('--n-clusters',
                       type=int,
                       default=3,
                       help='Number of clusters (default: 3)')

    args = parser.parse_args()

    # Run analysis
    analyzer = FalsePositiveAnalyzer(args.trace_file)

    if not analyzer.load_trace_data():
        print("❌ Failed to load trace data")
        return 1

    analyzer.extract_features()
    analyzer.perform_clustering(n_clusters=args.n_clusters)
    cluster_info = analyzer.analyze_clusters()
    recommendations = analyzer.generate_recommendations(cluster_info)
    analyzer.visualize_clusters(output_path=args.output)
    analyzer.save_results(cluster_info, recommendations, args.output)

    print(f"\n✅ Cluster analysis complete!")
    print(f"\n🎯 SUMMARY:")
    print(f"   • Analyzed {len(analyzer.false_positives)} false positive cases")
    print(f"   • Identified {len(cluster_info)} distinct patterns")
    print(f"   • Generated {len(recommendations)} targeted recommendations")
    print(f"\n💡 Next steps:")
    print(f"   1. Review cluster patterns above")
    print(f"   2. Implement targeted fixes for each cluster")
    print(f"   3. Run grid search to optimize parameters")
    print(f"   4. Re-run analysis to verify improvement")

    return 0


if __name__ == '__main__':
    sys.exit(main())
