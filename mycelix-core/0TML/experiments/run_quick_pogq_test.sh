#!/usr/bin/env bash
# Quick test using the working run_mini_validation.py approach

echo "===Running quick PoGQ validation using existing infrastructure==="
nix develop --command python run_mini_validation.py 2>&1 | head -200
