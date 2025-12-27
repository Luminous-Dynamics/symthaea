#!/usr/bin/env python3
"""
PyPhi Validation Results Analysis
Week 4 Day 5 - Statistical Analysis & Visualization

Analyzes pyphi_validation_results.csv to compute:
- Correlation metrics (Pearson r, Spearman ρ)
- Error metrics (RMSE, MAE, max/min errors)
- Topology-specific analysis
- Size-specific trends
- Publication-ready visualizations

Usage:
    python scripts/analyze_pyphi_results.py pyphi_validation_results.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(csv_path):
    """Load validation results from CSV"""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} validation results from {csv_path}")
    print(f"   Topologies: {df['topology'].nunique()}")
    print(f"   Sizes: {sorted(df['n'].unique())}")
    print(f"   Seeds: {sorted(df['seed'].unique())}")
    return df


def compute_statistics(df):
    """Compute comprehensive statistical metrics"""
    stats_dict = {
        'total_comparisons': len(df),
        'mean_phi_hdc': df['phi_hdc'].mean(),
        'mean_phi_exact': df['phi_exact'].mean(),
        'std_phi_hdc': df['phi_hdc'].std(),
        'std_phi_exact': df['phi_exact'].std(),
    }

    # Error metrics
    stats_dict['mean_error'] = df['error'].mean()
    stats_dict['std_error'] = df['error'].std()
    stats_dict['rmse'] = np.sqrt((df['error'] ** 2).mean())
    stats_dict['mae'] = df['error'].abs().mean()
    stats_dict['max_error'] = df['error'].max()
    stats_dict['min_error'] = df['error'].min()
    stats_dict['median_error'] = df['error'].median()

    # Relative error metrics
    stats_dict['mean_relative_error'] = df['relative_error'].mean()
    stats_dict['median_relative_error'] = df['relative_error'].median()

    # Correlation metrics
    pearson_r, pearson_p = stats.pearsonr(df['phi_hdc'], df['phi_exact'])
    spearman_rho, spearman_p = stats.spearmanr(df['phi_hdc'], df['phi_exact'])

    stats_dict['pearson_r'] = pearson_r
    stats_dict['pearson_p'] = pearson_p
    stats_dict['spearman_rho'] = spearman_rho
    stats_dict['spearman_p'] = spearman_p

    # R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['phi_exact'], df['phi_hdc']
    )
    stats_dict['r_squared'] = r_value ** 2
    stats_dict['regression_slope'] = slope
    stats_dict['regression_intercept'] = intercept
    stats_dict['regression_stderr'] = std_err

    return stats_dict


def print_statistics(stats_dict):
    """Print statistical summary in a nice format"""
    print("\n" + "="*60)
    print("📊 STATISTICAL SUMMARY")
    print("="*60)

    print(f"\n🔢 Sample Size:")
    print(f"   Total Comparisons:     {stats_dict['total_comparisons']}")

    print(f"\n📈 Central Tendency:")
    print(f"   Mean Φ_HDC:            {stats_dict['mean_phi_hdc']:.6f}")
    print(f"   Mean Φ_exact:          {stats_dict['mean_phi_exact']:.6f}")
    print(f"   Std Φ_HDC:             {stats_dict['std_phi_hdc']:.6f}")
    print(f"   Std Φ_exact:           {stats_dict['std_phi_exact']:.6f}")

    print(f"\n❌ Error Metrics:")
    print(f"   Mean Error:            {stats_dict['mean_error']:.6f}")
    print(f"   Median Error:          {stats_dict['median_error']:.6f}")
    print(f"   Std Error:             {stats_dict['std_error']:.6f}")
    print(f"   RMSE:                  {stats_dict['rmse']:.6f}")
    print(f"   MAE:                   {stats_dict['mae']:.6f}")
    print(f"   Max Error:             {stats_dict['max_error']:.6f}")
    print(f"   Min Error:             {stats_dict['min_error']:.6f}")

    print(f"\n📊 Relative Error:")
    print(f"   Mean Relative Error:   {stats_dict['mean_relative_error']:.2%}")
    print(f"   Median Relative Error: {stats_dict['median_relative_error']:.2%}")

    print(f"\n🔗 Correlation Metrics:")
    print(f"   Pearson r:             {stats_dict['pearson_r']:.6f} (p={stats_dict['pearson_p']:.2e})")
    print(f"   Spearman ρ:            {stats_dict['spearman_rho']:.6f} (p={stats_dict['spearman_p']:.2e})")
    print(f"   R²:                    {stats_dict['r_squared']:.6f}")

    print(f"\n📐 Linear Regression (Φ_exact → Φ_HDC):")
    print(f"   Slope:                 {stats_dict['regression_slope']:.6f}")
    print(f"   Intercept:             {stats_dict['regression_intercept']:.6f}")
    print(f"   Std Error:             {stats_dict['regression_stderr']:.6f}")
    print(f"   Equation:              Φ_HDC = {stats_dict['regression_slope']:.4f} * Φ_exact + {stats_dict['regression_intercept']:.4f}")


def evaluate_success_criteria(stats_dict):
    """Evaluate against Week 4 success criteria"""
    print("\n" + "="*60)
    print("✅ SUCCESS CRITERIA EVALUATION")
    print("="*60)

    # Minimum (Acceptable)
    print("\n📌 Minimum (Acceptable):")
    print(f"   r > 0.6:           {'✅' if stats_dict['pearson_r'] > 0.6 else '❌'} (r={stats_dict['pearson_r']:.3f})")

    # Target (Expected)
    print("\n🎯 Target (Expected):")
    print(f"   r > 0.8:           {'✅' if stats_dict['pearson_r'] > 0.8 else '❌'} (r={stats_dict['pearson_r']:.3f})")
    print(f"   RMSE < 0.15:       {'✅' if stats_dict['rmse'] < 0.15 else '❌'} ({stats_dict['rmse']:.3f})")
    print(f"   MAE < 0.10:        {'✅' if stats_dict['mae'] < 0.10 else '❌'} ({stats_dict['mae']:.3f})")

    # Stretch (Ideal)
    print("\n🌟 Stretch (Ideal):")
    print(f"   r > 0.9:           {'✅' if stats_dict['pearson_r'] > 0.9 else '❌'} (r={stats_dict['pearson_r']:.3f})")
    print(f"   RMSE < 0.10:       {'✅' if stats_dict['rmse'] < 0.10 else '❌'} ({stats_dict['rmse']:.3f})")
    print(f"   MAE < 0.05:        {'✅' if stats_dict['mae'] < 0.05 else '❌'} ({stats_dict['mae']:.3f})")

    # Overall conclusion
    print("\n🎯 Overall Assessment:")
    if stats_dict['pearson_r'] > 0.9 and stats_dict['rmse'] < 0.10:
        print("   ⭐ EXCELLENT! Ready for publication.")
        print("   Strong approximation with low error.")
    elif stats_dict['pearson_r'] > 0.8 and stats_dict['rmse'] < 0.15:
        print("   ✅ GOOD! Publication-ready with calibration.")
        print("   Approximation captures topology ordering well.")
    elif stats_dict['pearson_r'] > 0.7:
        print("   📊 ACCEPTABLE. Useful for ranking.")
        print("   Consider calibration for quantitative use.")
    else:
        print("   ⚠️  WEAK. Refinement needed.")
        print("   Review approximation methodology.")


def topology_analysis(df):
    """Analyze results by topology type"""
    print("\n" + "="*60)
    print("🔍 TOPOLOGY-SPECIFIC ANALYSIS")
    print("="*60 + "\n")

    topology_stats = []

    for topology in sorted(df['topology'].unique()):
        topo_df = df[df['topology'] == topology]

        stats = {
            'Topology': topology,
            'N': len(topo_df),
            'Φ_HDC': topo_df['phi_hdc'].mean(),
            'Φ_exact': topo_df['phi_exact'].mean(),
            'Error': topo_df['error'].mean(),
            'Rel.Err': topo_df['relative_error'].mean(),
            'RMSE': np.sqrt((topo_df['error'] ** 2).mean()),
        }
        topology_stats.append(stats)

        print(f"📐 {topology:12} | N={stats['N']:3} | "
              f"Φ_HDC={stats['Φ_HDC']:.4f} | Φ_exact={stats['Φ_exact']:.4f} | "
              f"Error={stats['Error']:.4f} ({stats['Rel.Err']:.1%}) | "
              f"RMSE={stats['RMSE']:.4f}")

    return pd.DataFrame(topology_stats)


def size_analysis(df):
    """Analyze results by topology size"""
    print("\n" + "="*60)
    print("📏 SIZE-SPECIFIC ANALYSIS")
    print("="*60 + "\n")

    size_stats = []

    for size in sorted(df['n'].unique()):
        size_df = df[df['n'] == size]

        stats = {
            'n': size,
            'N': len(size_df),
            'Φ_HDC': size_df['phi_hdc'].mean(),
            'Φ_exact': size_df['phi_exact'].mean(),
            'Error': size_df['error'].mean(),
            'Rel.Err': size_df['relative_error'].mean(),
            'RMSE': np.sqrt((size_df['error'] ** 2).mean()),
            'Duration_s': size_df['duration_ms'].mean() / 1000,
        }
        size_stats.append(stats)

        print(f"🔢 n={size} | N={stats['N']:3} | "
              f"Φ_HDC={stats['Φ_HDC']:.4f} | Φ_exact={stats['Φ_exact']:.4f} | "
              f"Error={stats['Error']:.4f} ({stats['Rel.Err']:.1%}) | "
              f"RMSE={stats['RMSE']:.4f} | "
              f"Duration={stats['Duration_s']:.1f}s")

    return pd.DataFrame(size_stats)


def create_visualizations(df, stats_dict, output_dir='pyphi_validation_plots'):
    """Create publication-quality visualizations"""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n📊 Creating visualizations in {output_dir}/...")

    # 1. Main scatter plot with regression
    plt.figure(figsize=(10, 10))
    plt.scatter(df['phi_exact'], df['phi_hdc'], alpha=0.6, s=50)

    # Add regression line
    x = np.linspace(df['phi_exact'].min(), df['phi_exact'].max(), 100)
    y = stats_dict['regression_slope'] * x + stats_dict['regression_intercept']
    plt.plot(x, y, 'r-', linewidth=2,
             label=f'y = {stats_dict["regression_slope"]:.3f}x + {stats_dict["regression_intercept"]:.3f}')

    # Add identity line
    plt.plot([df['phi_exact'].min(), df['phi_exact'].max()],
             [df['phi_exact'].min(), df['phi_exact'].max()],
             'k--', linewidth=1, alpha=0.5, label='Perfect Agreement')

    plt.xlabel('Φ_exact (PyPhi IIT 3.0)', fontsize=14)
    plt.ylabel('Φ_HDC (HDC Approximation)', fontsize=14)
    plt.title(f'Φ_HDC vs Φ_exact Validation\n' +
              f'r={stats_dict["pearson_r"]:.3f}, RMSE={stats_dict["rmse"]:.3f}',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_main.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir}/scatter_main.png")
    plt.close()

    # 2. Scatter plot colored by topology
    plt.figure(figsize=(12, 10))
    for topology in df['topology'].unique():
        topo_df = df[df['topology'] == topology]
        plt.scatter(topo_df['phi_exact'], topo_df['phi_hdc'],
                   label=topology, alpha=0.7, s=60)

    plt.plot([df['phi_exact'].min(), df['phi_exact'].max()],
             [df['phi_exact'].min(), df['phi_exact'].max()],
             'k--', linewidth=1, alpha=0.5)

    plt.xlabel('Φ_exact (PyPhi)', fontsize=14)
    plt.ylabel('Φ_HDC', fontsize=14)
    plt.title('Φ_HDC vs Φ_exact by Topology', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_by_topology.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir}/scatter_by_topology.png")
    plt.close()

    # 3. Error distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Absolute error histogram
    axes[0, 0].hist(df['error'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['error'].mean(), color='r', linestyle='--',
                       label=f'Mean={df["error"].mean():.4f}')
    axes[0, 0].set_xlabel('Absolute Error |Φ_HDC - Φ_exact|')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Relative error histogram
    axes[0, 1].hist(df['relative_error'] * 100, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['relative_error'].mean() * 100, color='r', linestyle='--',
                       label=f'Mean={df["relative_error"].mean():.1%}')
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Relative Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error by topology
    topology_errors = [df[df['topology'] == t]['error'] for t in sorted(df['topology'].unique())]
    axes[1, 0].boxplot(topology_errors, labels=sorted(df['topology'].unique()))
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error by Topology')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Error by size
    size_errors = [df[df['n'] == n]['error'] for n in sorted(df['n'].unique())]
    axes[1, 1].boxplot(size_errors, labels=[f'n={n}' for n in sorted(df['n'].unique())])
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error by Topology Size')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir}/error_analysis.png")
    plt.close()

    # 4. Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = df['phi_hdc'] - df['phi_exact']
    plt.scatter(df['phi_exact'], residuals, alpha=0.6, s=50)
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Φ_exact', fontsize=14)
    plt.ylabel('Residuals (Φ_HDC - Φ_exact)', fontsize=14)
    plt.title('Residual Plot', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir}/residuals.png")
    plt.close()

    # 5. Topology ranking comparison
    topology_hdc = df.groupby('topology')['phi_hdc'].mean().sort_values(ascending=False)
    topology_exact = df.groupby('topology')['phi_exact'].mean().sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(range(len(topology_hdc)), topology_hdc.values)
    ax1.set_yticks(range(len(topology_hdc)))
    ax1.set_yticklabels(topology_hdc.index)
    ax1.set_xlabel('Mean Φ_HDC')
    ax1.set_title('Topology Ranking (HDC)')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.barh(range(len(topology_exact)), topology_exact.values)
    ax2.set_yticks(range(len(topology_exact)))
    ax2.set_yticklabels(topology_exact.index)
    ax2.set_xlabel('Mean Φ_exact')
    ax2.set_title('Topology Ranking (PyPhi)')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/topology_ranking.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {output_dir}/topology_ranking.png")
    plt.close()

    print(f"\n✅ All visualizations created in {output_dir}/")


def main():
    """Main analysis pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_pyphi_results.py <csv_file>")
        print("Example: python analyze_pyphi_results.py pyphi_validation_results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    print("="*60)
    print("🔬 PyPhi Validation Results Analysis")
    print("="*60)

    # Load data
    df = load_results(csv_path)

    # Compute statistics
    stats_dict = compute_statistics(df)
    print_statistics(stats_dict)

    # Evaluate success criteria
    evaluate_success_criteria(stats_dict)

    # Topology analysis
    topology_df = topology_analysis(df)

    # Size analysis
    size_df = size_analysis(df)

    # Create visualizations
    create_visualizations(df, stats_dict)

    # Save summary statistics
    output_file = 'pyphi_validation_summary.txt'
    with open(output_file, 'w') as f:
        f.write("PyPhi Validation Results Summary\n")
        f.write("="*60 + "\n\n")
        for key, value in stats_dict.items():
            f.write(f"{key:25} = {value}\n")
    print(f"\n✅ Summary statistics saved to {output_file}")

    # Save topology stats
    topology_df.to_csv('pyphi_validation_topology_stats.csv', index=False)
    print(f"✅ Topology statistics saved to pyphi_validation_topology_stats.csv")

    # Save size stats
    size_df.to_csv('pyphi_validation_size_stats.csv', index=False)
    print(f"✅ Size statistics saved to pyphi_validation_size_stats.csv")

    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
