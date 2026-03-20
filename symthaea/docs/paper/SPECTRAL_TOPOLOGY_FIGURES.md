# Figure Descriptions: Spectral Topology Manuscript

**For**: "Asymptotic Convergence of Algebraic Connectivity in Network Topologies: Why Three Dimensions Suffice"
**Date**: March 18, 2026

---

## Figure 1: Asymptotic Convergence of Spectral Connectivity with Dimension

**Title**: lambda-2_HDC converges asymptotically for k-regular hypercubes

**Type**: Scatter plot with fitted curve

**Axes**:
- X-axis: Hypercube dimension k (integer, range 1--7, with extrapolation shown dashed to k = 10)
- Y-axis: lambda-2_HDC (range 0.48--1.02, with axis break between 0.52 and 0.98 to accommodate the 1D degenerate point)

**Data points**:
- Black circles: Mean lambda-2_HDC for each dimension (n = 10 per point)
- Gray error bars: +/- 1 SD (most are smaller than the marker size)
- The 1D point (lambda-2_HDC = 1.0000) is plotted but marked with an open circle and labeled "degenerate (N = 2)"

**Fitted curve**:
- Solid orange line: lambda-2(k) = 0.4998 - 0.0522 exp(-0.89k), fitted to k = 2--7 data
- Dashed orange line: Extrapolation to k = 8--10
- Horizontal dashed gray line: Asymptote at lambda-2_max = 0.4998

**Annotations**:
- Text box: "R-squared = 0.998" and fitted parameter values
- Arrow pointing to k = 3 data point, labeled "3D: 99.2% of asymptote"
- Arrow pointing to k = 4 data point, labeled "4D: 99.6%"
- Shaded region (light blue): 95% bootstrap confidence band for the fitted curve

**Inset** (optional):
- Zoomed view of k = 3--7 range (y-axis 0.495--0.500) to show the convergence detail that is compressed in the main plot

**Dimensions**: Single column (86 mm wide) or double column (178 mm) depending on journal layout. Minimum 300 DPI for raster elements; vector preferred.

**Software**: matplotlib with publication style (serif font, 8--10 pt labels, no grid)

**Color**: Accessible palette. Orange curve against white background. Black data points. Gray for error bars and reference lines.

---

## Figure 2: Spectral Connectivity Rankings Across 19 Network Topologies

**Title**: lambda-2_HDC varies substantially across network topologies

**Type**: Horizontal bar chart (lollipop or Cleveland dot plot preferred for PRE style)

**Axes**:
- X-axis: lambda-2_HDC (range 0.48--0.50)
- Y-axis: Topology name (19 entries, sorted by descending lambda-2_HDC)

**Data**:
- Each topology shown as a dot (mean) with horizontal error bar (+/- 1 SD)
- Color-coded by category:
  - Blue: Classical architectures (8 topologies)
  - Red: Hypercubes (3 topologies)
  - Green: Exotic topologies (5 topologies)
  - Purple: Manifolds (2 topologies)
  - Orange: Non-orientable baseline (1 topology)

**Annotations**:
- Vertical dashed line at asymptotic maximum (lambda-2_max = 0.4998)
- Bracket showing "Top cluster: span = 0.0022" for ranks 1--3
- Label on complete graph bar: "lowest: 0.4834"
- Category legend in upper-left corner

**Dimensions**: Single column (86 mm) or 1.5 columns (130 mm). The 19 topology labels require sufficient vertical space.

**Notes**: Use abbreviated topology names if space is tight (e.g., "HC-4D" for Hypercube 4D, "Mob-2D" for Mobius Strip 2D). Include degree k in parentheses after each name.

---

## Figure 3: Category Comparison of Spectral Connectivity

**Title**: Hypercubes dominate spectral connectivity across topology categories

**Type**: Box-and-whisker plot (or violin plot with overlaid data points)

**Axes**:
- X-axis: Topology category (7 categories, ordered by median lambda-2_HDC)
- Y-axis: lambda-2_HDC (range 0.48--0.50)

**Categories** (left to right, by descending median):
1. Hypercubes (n = 3 topologies, 30 measurements)
2. Classical (n = 8, 80 measurements)
3. Tier 1 Exotic (n = 2, 20 measurements)
4. Manifolds (n = 2, 20 measurements)
5. Tier 2 Exotic (n = 2, 20 measurements)
6. Tier 3 Exotic (n = 1, 10 measurements)
7. Non-Orientable 1D (n = 1, 10 measurements)

**Box elements**:
- Box: IQR (25th--75th percentile)
- Horizontal line: Median
- Whiskers: 1.5 x IQR
- Individual data points (jittered) overlaid as small circles

**Annotations**:
- Significance brackets between Hypercubes and each other category (Tukey HSD p-values)
- ANOVA result: "F(6,12) = 48.3, p < 0.0001"
- Note the large within-category variance for Classical (heterogeneous architectures)

**Dimensions**: Single column (86 mm) or double column (178 mm).

**Color**: Each category in a distinct color from an accessible palette (e.g., ColorBrewer Set2 or Okabe-Ito).

---

## Figure 4: Non-Orientability Effects on Spectral Connectivity

**Title**: Spectral effects of topological twists depend on dimension matching

**Type**: Grouped bar chart or paired dot plot

**Layout**: Three pairs of comparisons arranged left to right:

**Panel A** (or left group): 1D non-orientable vs. 1D orientable baseline
- Mobius Strip 1D (lambda-2_HDC = 0.4875) vs. Ring (lambda-2_HDC = 0.4954)
- Delta = -0.0079 (degradation from twist)
- Label: "1D twist in 1D: DEGRADED"

**Panel B** (or center group): 2D non-orientable vs. 2D orientable baseline
- Mobius Strip 2D (lambda-2_HDC = 0.4943) vs. Torus (lambda-2_HDC = 0.4940)
- Delta = +0.0003 (negligible difference)
- Label: "2D twist in 2D: MATCHED"

**Panel C** (or right group): Double twist vs. single twist in 2D
- Klein Bottle 2D (lambda-2_HDC = 0.4901) vs. Mobius Strip 2D (lambda-2_HDC = 0.4943)
- Delta = -0.0042 (degradation from excess twist)
- Label: "Double twist in 2D: DEGRADED"

**Data representation**:
- Bars or dots with error bars (+/- 1 SD)
- Connecting lines between matched pairs with delta labeled
- Significance asterisks (p-values from t-tests)

**Annotations**:
- Summary text: "Dimension-matched twists preserve spectral connectivity; mismatched or redundant twists degrade it"
- Cohen's d values for each comparison

**Dimensions**: Double column (178 mm) for the three-panel layout, or single column with panels stacked vertically.

**Color**: Orientable baselines in blue; non-orientable topologies in orange/red. Consistent with Figure 2 color scheme.

---

## General Figure Specifications

**Format**: Vector PDF for line art; 300+ DPI PNG as fallback. Physical Review E prefers EPS or PDF.

**Font**: Times New Roman or Computer Modern, 8 pt minimum for axis labels, 10 pt for panel labels (a, b, c).

**Line width**: Minimum 0.5 pt for axes and tick marks; 1.0--1.5 pt for data curves.

**Color**: All figures must be interpretable in grayscale (use pattern/marker variation as secondary encoding). Color versions for online publication.

**Panel labels**: Lowercase letters (a), (b), (c) in bold, positioned at upper-left of each panel.

**Error bars**: Represent +/- 1 standard deviation unless otherwise noted. State in each figure caption.

**Reproducibility**: All figures can be regenerated from raw data using the Python script `generate_figures.py` in the code repository.
