#!/usr/bin/env Rscript
# Cross-language comparator adapter for generated clustered-IV CSV data.
# Usage: Rscript clusteriv_020.R generated.csv [output-prefix]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("usage: Rscript clusteriv_020.R generated.csv [output-prefix]")
}
csv_path <- args[[1L]]
output_prefix <- if (length(args) >= 2L) args[[2L]] else "clusteriv-020"

if (!requireNamespace("clusterIV", quietly = TRUE)) {
  stop("install clusterIV 0.2.0 from CRAN before running this adapter")
}
package_version <- as.character(utils::packageVersion("clusterIV"))
if (!identical(package_version, "0.2.0")) {
  stop(sprintf("expected clusterIV 0.2.0; found %s", package_version))
}

# Preserve the runtime, package and platform context for exact reproduction.
capture.output(sessionInfo(), file = paste0(output_prefix, "-sessionInfo.txt"))

data <- utils::read.csv(csv_path, check.names = FALSE)
cluster_count <- length(unique(data$cluster_id))
# The cross-fit estimator regresses on instruments leaving out two entire
# clusters. A two-cluster parity fixture cannot supply a training sample.
if (cluster_count < 3L) {
  stop(sprintf(
    "need at least 3 clusters for leave-two-cluster-out crossfit; found %d. Use this tiny file for CSV parity only.",
    cluster_count
  ))
}
z_names <- grep("^z[0-9]+$", names(data), value = TRUE)
if (length(z_names) < 1L) stop("no z0, z1, ... instrument columns found")
if (!all(c("outcome", "treatment", "cluster_id") %in% names(data))) {
  stop("CSV lacks outcome, treatment, or cluster_id")
}
z <- as.matrix(data[, z_names, drop = FALSE])

# This adapter receives only estimator-visible CSV columns; it never reads
# the separately sealed .oracle.json file. Keep the complete panel and every
# confidence-set object. The plain and leave-two-cluster-out cross-fit
# variants are both registered baselines.
for (variance_method in c("plain", "crossfit")) {
  for (calibration_method in c("chisq", "normal")) {
    fit <- clusterIV::iv_infer(
      y = data$outcome,
      x = data$treatment,
      z = z,
      cluster = data$cluster_id,
      beta0 = 0,
      level = 0.95,
      calibration = calibration_method,
      variance = variance_method
    )
    output_path <- sprintf(
      "%s-%s-%s.txt",
      output_prefix,
      variance_method,
      calibration_method
    )
    capture.output(print(fit), file = output_path)
    saveRDS(fit, file = paste0(output_path, ".rds"))
    message("wrote ", output_path, " and full fitted object ", output_path, ".rds")
  }
}
