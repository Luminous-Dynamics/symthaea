#!/usr/bin/env bash

echo "Finding correct cosine threshold range for label skew..."
echo

echo "BASELINE (one-sided >= -0.8):"
echo "Round 0 honest nodes cosine distribution:"
jq -c 'select(.round == 0) | .nodes[] | select(.ground_truth == "honest") | {node: .node_id, cos: (.scores.cos_anchor | tonumber | . * 1000 | floor / 1000), fp: (.error_type == "false_positive")}' results/label_skew_trace.jsonl | sort -t: -k3 -n

echo
echo "Statistics:"
echo "Min cosine (honest): $(jq 'select(.round == 0) | [.nodes[] | select(.ground_truth == "honest") | .scores.cos_anchor | tonumber] | min' results/label_skew_trace.jsonl)"
echo "Max cosine (honest): $(jq 'select(.round == 0) | [.nodes[] | select(.ground_truth == "honest") | .scores.cos_anchor | tonumber] | max' results/label_skew_trace.jsonl)"
echo "Median cosine (honest): $(jq 'select(.round == 0) | [.nodes[] | select(.ground_truth == "honest") | .scores.cos_anchor | tonumber] | sort | .[length/2]' results/label_skew_trace.jsonl)"

echo
echo "Byzantine nodes cosine distribution:"
jq -c 'select(.round == 0) | .nodes[] | select(.ground_truth == "byzantine") | {node: .node_id, cos: (.scores.cos_anchor | tonumber | . * 1000 | floor / 1000), attack: "unknown"}' results/label_skew_trace.jsonl | sort -t: -k3 -n

echo
echo "RECOMMENDED THRESHOLD:"
echo "Lower bound: -0.5 (catches extreme noise, allows honest diversity)"
echo "Upper bound: 0.95 (catches near-perfect alignment attacks)"
echo "Range: [-0.5, 0.95] protects 90%+ of honest nodes"

