#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Fail-closed quality gate for one source-bound ann_benchmark CSV.
# This scores the proposed low-noise gate only; hard-neighbor routing remains
# unqualified until a separate unsafe-event/risk contract is evaluated.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ann-benchmark.csv> [more-run-csvs...]" >&2
  exit 2
fi

FAILED=0
for REPORT in "$@"; do
  if [[ ! -f "$REPORT" ]]; then
    echo "FAIL $REPORT: file does not exist" >&2
    FAILED=1
    continue
  fi

  RECEIPT="${REPORT%.csv}.receipt.txt"
  if [[ ! -f "$RECEIPT" ]]; then
    echo "FAIL $REPORT: missing sidecar receipt $RECEIPT" >&2
    FAILED=1
    continue
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "FAIL $REPORT: sha256sum is required to verify the output receipt" >&2
    FAILED=1
    continue
  fi
  if ! grep -qx 'exit_status=0' "$RECEIPT"; then
    echo "FAIL $REPORT: sidecar receipt does not record exit_status=0" >&2
    FAILED=1
    continue
  fi
  EXPECTED_SHA="$(awk -F= '$1 == "output_sha256" { print $2; exit }' "$RECEIPT")"
  ACTUAL_SHA="$(sha256sum "$REPORT" | awk '{ print $1 }')"
  RECEIPT_SOURCE_SHA="$(awk -F= '$1 == "source_sha" { print $2; exit }' "$RECEIPT")"
  CSV_SOURCE_SHA="$(awk -F= '/^# source_sha=/ { sub(/^# source_sha=/, ""); print; exit }' "$REPORT")"
  STDERR_PATH="$(awk -F= '$1 == "stderr_log" { sub(/^[^=]*=/, ""); print; exit }' "$RECEIPT")"
  EXPECTED_STDERR_SHA="$(awk -F= '$1 == "stderr_sha256" { print $2; exit }' "$RECEIPT")"
  if [[ -z "$EXPECTED_SHA" || "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
    echo "FAIL $REPORT: output SHA-256 does not match sidecar receipt" >&2
    FAILED=1
    continue
  fi
  if [[ -z "$RECEIPT_SOURCE_SHA" || "$RECEIPT_SOURCE_SHA" != "$CSV_SOURCE_SHA" ]]; then
    echo "FAIL $REPORT: receipt source SHA does not match CSV metadata" >&2
    FAILED=1
    continue
  fi
  if [[ -z "$STDERR_PATH" || ! -f "$STDERR_PATH" || -z "$EXPECTED_STDERR_SHA" ]]; then
    echo "FAIL $REPORT: stderr log or its SHA-256 is missing from receipt" >&2
    FAILED=1
    continue
  fi
  ACTUAL_STDERR_SHA="$(sha256sum "$STDERR_PATH" | awk '{ print $1 }')"
  if [[ "$EXPECTED_STDERR_SHA" != "$ACTUAL_STDERR_SHA" ]]; then
    echo "FAIL $REPORT: stderr SHA-256 does not match sidecar receipt" >&2
    FAILED=1
    continue
  fi

  echo "=== $REPORT ==="
  if awk '
    function fail(message) {
      print "FAIL: " message
      failed = 1
    }
    function require_metric(key, arm, description,    full) {
      full = "arm_" arm "_" key
      if (!(full in values)) {
        fail("missing metric " full)
        return ""
      }
      return values[full]
    }
    function check_floor(value, floor, name,    x) {
      if (value == "") return
      x = value + 0
      if (x != x || x < floor) fail(name " = " value ", required >= " floor)
      else print "PASS: " name " = " value " (>= " floor ")"
    }
    function check_ceiling(value, ceiling, name,    x) {
      if (value == "") return
      x = value + 0
      if (x != x || x > ceiling) fail(name " = " value ", required <= " ceiling)
      else print "PASS: " name " = " value " (<= " ceiling ")"
    }
    function check_strict_floor(value, floor, name,    x) {
      if (value == "") return
      x = value + 0
      if (x != x || x <= floor) fail(name " = " value ", required > " floor)
      else print "PASS: " name " = " value " (> " floor ")"
    }
    BEGIN { FS = ","; OFS = "," }

    /^# / {
      line = substr($0, 3)
      pos = index(line, "=")
      if (pos > 0) {
        key = substr(line, 1, pos - 1)
        value = substr(line, pos + 1)
        values[key] = value
        if (key == "lsh_configs") {
          expected_count = split(value, expected_arms, ";")
          for (i = 1; i <= expected_count; i++) {
            if (expected_arms[i] == "") fail("empty LSH arm in metadata")
            expected_arm[expected_arms[i]] = 1
          }
        }
        if (key == "unique_queries") expected_queries = value + 0
        if (key == "rounds") expected_rounds = value + 0
      }
      next
    }

    /^query_index,round,kind,group_id,/ {
      header_seen = 1
      if (index($0, "group_id") == 0) fail("CSV header lacks group_id")
      next
    }

    /^[0-9]+,[0-9]+,(near_neighbor|hard_neighbor|novel),/ {
      fields = split($0, row, ",")
      if (fields != 17) {
        fail("raw CSV row has " fields " columns, expected 17")
        next
      }
      query = row[1]
      round = row[2] + 0
      kind = row[3]
      arm = row[5] "x" row[6]
      if (!(arm in expected_arm)) fail("raw row uses undeclared LSH arm " arm)
      if (round < 0 || (expected_rounds > 0 && round >= expected_rounds))
        fail("round index outside declared range: " round)
      row_key = arm SUBSEP query SUBSEP round
      if (row_key in seen_row) fail("duplicate query/round row for arm " arm " query " query " round " round)
      seen_row[row_key] = 1
      rows[arm]++
      query_key = arm SUBSEP query
      if (!(query_key in seen_query)) {
        seen_query[query_key] = kind
        distinct_queries[arm]++
      } else if (seen_query[query_key] != kind) {
        fail("query kind changes across rounds for arm " arm " query " query)
      }
      if (round == 0) class_count[arm SUBSEP kind]++
      next
    }

    END {
      if (values["protocol"] != "BROCA-ANN-RETRIEVAL-V1") fail("wrong or missing protocol ID")
      if (length(values["source_sha"]) != 40 || values["source_sha"] !~ /^[0-9a-fA-F]+$/)
        fail("missing/invalid 40-hex source SHA")
      if (!header_seen) fail("missing raw CSV header with group_id")
      if (values["bootstrap_unit"] != "source_cluster_for_near_neighbor; singleton_query_for_novel")
        fail("missing/unsupported group-bootstrap unit metadata")
      if (expected_count < 1) fail("missing LSH configuration list")
      if (expected_queries < 100) fail("unique query count " expected_queries ", required >= 100")
      if (expected_rounds < 1) fail("missing/invalid timing round count")

      for (i = 1; i <= expected_count; i++) {
        arm = expected_arms[i]
        if (!(arm in expected_arm)) continue
        if (distinct_queries[arm] != expected_queries)
          fail("arm " arm " has " distinct_queries[arm] " distinct queries, expected " expected_queries)
        if (rows[arm] != expected_queries * expected_rounds)
          fail("arm " arm " has " rows[arm] " raw rows, expected " expected_queries * expected_rounds)
        for (j = 1; j <= 3; j++) {
          kinds[j] = (j == 1 ? "near_neighbor" : (j == 2 ? "hard_neighbor" : "novel"))
          metric_key = "arm_" arm "_" kinds[j] "_query_count"
          expected_n = values[metric_key] + 0
          if (!(metric_key in values)) fail("missing stratum summary " metric_key)
          if (expected_n < 1) fail("empty required query stratum " metric_key)
          if (class_count[arm SUBSEP kinds[j]] != expected_n)
            fail("raw round-0 count for " metric_key " differs from summary")
        }

        near_n = require_metric("near_neighbor_query_count", arm, "near-neighbor count") + 0
        if (near_n < 100) fail("arm " arm " has fewer than 100 low-noise queries")
        for (j = 1; j <= 3; j++) {
          keys[j] = (j == 1 ? "near_neighbor_bootstrap_group_count" : (j == 2 ? "hard_neighbor_bootstrap_group_count" : "novel_bootstrap_group_count"))
          full = "arm_" arm "_" keys[j]
          if (!(full in values)) fail("missing bootstrap group count " full)
        }
        group_n = require_metric("near_neighbor_bootstrap_group_count", arm, "near-neighbor independent groups") + 0
        if (group_n < 10) fail("arm " arm " has fewer than 10 near-neighbor groups; interval is not qualified")
        lower = require_metric("near_neighbor_median_speedup_group_bootstrap_95ci_low", arm, "median speedup group-bootstrap lower bound")
        if (lower != "") check_strict_floor(lower, 1.0, "arm " arm " near-neighbor group-bootstrap lower 95% bound")

        check_floor(require_metric("near_neighbor_mean_recall_at_k", arm, "mean strict Recall@K"), 0.95, "arm " arm " near-neighbor mean strict Recall@K")
        check_floor(require_metric("near_neighbor_worst_recall_at_k", arm, "worst strict Recall@K"), 0.80, "arm " arm " near-neighbor worst strict Recall@K")
        check_floor(require_metric("near_neighbor_mean_tie_aware_recall_at_k", arm, "mean tie-aware Recall@K"), 0.98, "arm " arm " near-neighbor mean tie-aware Recall@K")
        check_ceiling(require_metric("near_neighbor_mean_candidate_fraction", arm, "mean candidate fraction"), 0.50, "arm " arm " near-neighbor mean candidate fraction")
        check_ceiling(require_metric("near_neighbor_exhaustive_query_fraction", arm, "exhaustive query fraction"), 0.50, "arm " arm " near-neighbor exhaustive query fraction")
        check_ceiling(require_metric("near_neighbor_fallback_query_fraction", arm, "fallback query fraction"), 0.05, "arm " arm " near-neighbor exact-fallback fraction")
        check_floor(require_metric("near_neighbor_ratio_exact_p95_over_approximate_p95", arm, "p95 latency ratio"), 2.0, "arm " arm " near-neighbor p95 speedup")
        print "NOTICE: arm " arm " hard-neighbor stratum remains NOT QUALIFIED for adaptive routing; it needs a separately preregistered unsafe event and risk gate."
      }
      if (failed) exit 1
      print "PASS: report structure and all proposed low-noise gates passed. This does not qualify hard-neighbor/OOD routing or prove end-to-end memory utility."
    }
  ' "$REPORT"; then
    :
  else
    FAILED=1
  fi
done

exit "$FAILED"
