#!/usr/bin/env bash
# Analysis script for label skew trace data

TRACE_FILE="results/label_skew_trace.jsonl"

echo "======================================================================="
echo "LABEL SKEW FAILURE ANALYSIS"
echo "======================================================================="
echo

# 1. Overall statistics per round
echo "1. FALSE POSITIVE RATE PER ROUND:"
echo "Round | FP Rate | Detection Rate | Avg Honest Rep | Anchor Node"
echo "------|---------|---------------|----------------|------------"
jq -r '.round as $r | .stats.false_positive_rate as $fp | .stats.detection_rate as $dr | (.nodes | map(select(.ground_truth == "honest")) | map(.reputation) | add / length) as $avg_rep | .anchor_info.node_id as $anchor | "\($r) | \($fp*100 | floor)% | \($dr*100 | floor)% | \($avg_rep | tonumber | . * 100 | floor / 100) | Node \($anchor)"' $TRACE_FILE

echo
echo "2. FALSE POSITIVE NODES PER ROUND (showing first 5):"
for round in {0..4}; do
  echo "Round $round:"
  jq --arg r "$round" 'select(.round == ($r | tonumber)) | .nodes[] | select(.error_type == "false_positive") | "  Node \(.node_id): cos_anchor=\(.scores.cos_anchor | tonumber | . * 1000 | floor / 1000), PoGQ=\(.scores.pogq | tonumber | . * 1000 | floor / 1000), rep=\(.reputation)"' $TRACE_FILE | head -5
done

echo
echo "3. ANCHOR NODE ANALYSIS:"
echo "Round | Anchor Node ID | Anchor Gradient Norm | Anchor is Byzantine?"
echo "------|----------------|---------------------|--------------------"
jq -r '.round as $r | .anchor_info.node_id as $anchor | .anchor_info.norm as $norm | (.nodes[] | select(.node_id == $anchor) | .ground_truth) as $truth | "\($r) | \($anchor) | \($norm | tonumber | . * 1000 | floor / 1000) | \($truth)"' $TRACE_FILE

echo
echo "4. COSINE SIMILARITY DISTRIBUTION (Round 0):"
echo "Honest nodes' cosine similarity with anchor:"
jq 'select(.round == 0) | .nodes[] | select(.ground_truth == "honest") | "  Node \(.node_id): \(.scores.cos_anchor | tonumber | . * 1000 | floor / 1000) - \(if .correct then "✓ Correct" else "✗ FALSE POSITIVE" end)"' $TRACE_FILE

echo
echo "5. ROOT CAUSE EVIDENCE:"
echo
echo "a) Anchor Selection (always Node 0):"
jq -r '.round as $r | .anchor_info.node_id' $TRACE_FILE | sort | uniq -c

echo
echo "b) Honest nodes with negative cosine (should be protected by guard):"
jq 'select(.round < 3) | .round as $r | .nodes[] | select(.ground_truth == "honest" and .error_type == "false_positive" and (.scores.cos_anchor | tonumber) < 0) | "  Round \($r), Node \(.node_id): cos=\(.scores.cos_anchor | tonumber | . * 1000 | floor / 1000)"' $TRACE_FILE | head -10

echo
echo "c) Committee scores for false positives (Round 0):"
jq 'select(.round == 0) | .nodes[] | select(.error_type == "false_positive") | "  Node \(.node_id): committee=\(.scores.pogq), cos_anchor=\(.scores.cos_anchor | tonumber | . * 1000 | floor / 1000)"' $TRACE_FILE

echo
echo "======================================================================="
echo "KEY FINDINGS:"
echo "======================================================================="
echo "1. Anchor is ALWAYS Node 0 (first honest node by index)"
echo "2. Honest nodes with diverse gradients flagged as Byzantine"
echo "3. Cosine guard at -0.8 is protecting orthogonal gradients inappropriately"
echo "4. Reputation collapse amplifies initial misclassifications"
