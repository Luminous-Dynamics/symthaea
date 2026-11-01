* Stata 19 adapter for the deterministic clustered-IV CSV.
* Usage:
*   stata -b do stata19_weakrobust.do generated.csv [output-prefix]
*
* This is the one-endogenous-regressor clustered design. The cluster variable
* is encoded from the stable string cluster_id in the generated CSV.

args csv_path output_prefix
if "`csv_path'" == "" local csv_path "connectome-cluster-iv-dgp-example.csv"
if "`output_prefix'" == "" local output_prefix "stata19"

import delimited using "`csv_path'", clear varnames(1) stringcols(1 2)
encode cluster_id, generate(cluster_numeric)
unab instruments : z*

* Cluster-robust 2SLS followed by Stata 19 weak-instrument-robust AR/CLR.
ivregress 2sls outcome (treatment = `instruments'), vce(cluster cluster_numeric)
estimates store iv_cluster
estat weakrobust, ci
estimates save "`output_prefix'-cluster.ster", replace

* Robust-variance comparator. Keep it separate from the clustered result.
ivregress 2sls outcome (treatment = `instruments'), vce(robust)
estimates store iv_robust
estat weakrobust, ci
estimates save "`output_prefix'-robust.ster", replace
