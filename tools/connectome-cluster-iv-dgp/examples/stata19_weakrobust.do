* Stata 19 adapter for the deterministic clustered-IV CSV.
* Usage:
*   stata -b do stata19_weakrobust.do generated.csv [output-prefix]
*
* This is the one-endogenous-regressor clustered design. The cluster variable
* is encoded from the stable string cluster_id in the generated CSV. The input
* is estimator-visible CSV only; never import the separate oracle JSON.

version 19.0
set more off

args csv_path output_prefix
if "`csv_path'" == "" local csv_path "connectome-cluster-iv-dgp-example.csv"
if "`output_prefix'" == "" local output_prefix "stata19"

capture log close _all
log using "`output_prefix'-weakrobust.log", text replace

import delimited using "`csv_path'", clear varnames(1) stringcols(1 2)
encode cluster_id, generate(cluster_numeric)
unab instruments : z*

* Cluster-robust 2SLS followed by Stata 19 weak-instrument-robust AR/CLR.
ivregress 2sls outcome (treatment = `instruments'), vce(cluster cluster_numeric)
estimates store iv_cluster
estat firststage
estat weakrobust, ci ar
return list
estat weakrobust, ci clr reps(25000) rseed(2026)
return list
estimates save "`output_prefix'-cluster.ster", replace

* Robust-variance comparator. Keep it separate from the clustered result.
ivregress 2sls outcome (treatment = `instruments'), vce(robust)
estimates store iv_robust
estat firststage
estat weakrobust, ci ar
return list
estat weakrobust, ci clr reps(25000) rseed(2026)
return list
estimates save "`output_prefix'-robust.ster", replace

log close
