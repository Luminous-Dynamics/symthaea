#!/usr/bin/env bash
set -euo pipefail

package="symthaea-therapeutic"

cargo fmt --all -- --check
cargo check -p "$package" --no-default-features
cargo test -p "$package" --no-default-features
cargo clippy -p "$package" --no-default-features --all-targets -- -D warnings

features=(
  experimental-computational-psychiatry
  experimental-consciousness-protocols
  experimental-diagnostic-hypotheses
  legacy-clinical-scale-analogues
)

for feature in "${features[@]}"; do
  cargo check -p "$package" --no-default-features --features "$feature"
  cargo test -p "$package" --no-default-features --features "$feature"
done

all_features=$(IFS=,; echo "${features[*]}")
cargo check -p "$package" --no-default-features --features "$all_features"
cargo test -p "$package" --no-default-features --features "$all_features"
