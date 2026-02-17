#!/usr/bin/env bash

echo "======================================================================="
echo "FAILURE ANALYSIS: Why did P1c+P1d make things worse?"
echo "======================================================================="
echo

echo "1. COSINE SIMILARITY DISTRIBUTION (Round 0 - with robust anchor):"
jq -c 'select(.round == 0) | .nodes[] | select(.ground_truth == "honest") | {node: .node_id, cos: (.scores.cos_anchor | tonumber | . * 1000 | floor / 1000), class: (if .correct then "✓" else "✗ FP" end)}' results/label_skew_trace_fixed.jsonl

echo
echo "2. THRESHOLD VALUES:"
jq 'select(.round == 0) | .thresholds | {cos_min: .label_skew_cos_min, cos_max: .label_skew_cos_max}' results/label_skew_trace_fixed.jsonl

echo
echo "3. HOW MANY HONEST NODES ARE OUTSIDE [0.3, 0.8]?"
echo "Round | Total Honest | cos < 0.3 | 0.3-0.8 (safe) | cos > 0.8 | FP Rate"
for round in {0..9}; do
  total=$(jq --arg r "$round" 'select(.round == ($r | tonumber)) | [.nodes[] | select(.ground_truth == "honest")] | length' results/label_skew_trace_fixed.jsonl)
  below=$(jq --arg r "$round" 'select(.round == ($r | tonumber)) | [.nodes[] | select(.ground_truth == "honest" and (.scores.cos_anchor | tonumber) < 0.3)] | length' results/label_skew_trace_fixed.jsonl)
  safe=$(jq --arg r "$round" 'select(.round == ($r | tonumber)) | [.nodes[] | select(.ground_truth == "honest" and (.scores.cos_anchor | tonumber) >= 0.3 and (.scores.cos_anchor | tonumber) <= 0.8)] | length' results/label_skew_trace_fixed.jsonl)
  above=$(jq --arg r "$round" 'select(.round == ($r | tonumber)) | [.nodes[] | select(.ground_truth == "honest" and (.scores.cos_anchor | tonumber) > 0.8)] | length' results/label_skew_trace_fixed.jsonl)
  fp_rate=$(jq --arg r "$round" 'select(.round == ($r | tonumber)) | .stats.false_positive_rate' results/label_skew_trace_fixed.jsonl)
  printf "%5d | %12d | %9d | %14d | %9d | %d%%\n" "$round" "$total" "$below" "$safe" "$above" "$(echo "$fp_rate * 100" | bc | cut -d. -f1)"
done

echo
echo "4. ROOT CAUSE: What's different from baseline?"
echo "Comparing anchor selection:"
echo "BASELINE (first honest):"
jq 'select(.round == 0) | .anchor_info.node_id' results/label_skew_trace.jsonl

echo "WITH FIX (reputation-weighted centroid):"
echo "  - Using top-5 honest nodes weighted by reputation"
echo "  - In round 0, all honest nodes have rep=1.0"
echo "  - So it's averaging gradients of nodes 0,1,2,3,8 (top 5)"

echo
echo "5. HYPOTHESIS: The centroid anchor is LESS representative!"
echo "  - Single node gradient aligns well with nodes similar to it"
echo "  - Centroid averages out diversity, creating artificial 'middle' point"
echo "  - Nodes dissimilar to centroid get flagged"

