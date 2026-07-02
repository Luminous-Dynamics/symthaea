"""
Small demo showing how to call the Rust HyperFeel core from Python.

Run after building the Rust binary:
  cd Mycelix-Core/0TML/mycelix-hyperfeel
  cargo build --release
and ensure `hyperfeel_cli` is on PATH, or adjust HyperFeelConfig.binary.
"""

from __future__ import annotations

import numpy as np

from zerotrustml.hyperfeel_bridge import (
    encode_gradient,
    aggregate_hypergradients,
    HyperFeelConfig,
    HyperFeelUnavailable,
)


def main():
    try:
        cfg = HyperFeelConfig()  # Uses PATH by default
        dummy_grad1 = np.random.randn(1000).astype(np.float32)
        dummy_grad2 = np.random.randn(1000).astype(np.float32)

        hg1 = encode_gradient(dummy_grad1, config=cfg)
        hg2 = encode_gradient(dummy_grad2, config=cfg)

        aggregated = aggregate_hypergradients([hg1, hg2], config=cfg)

        print("Encoded hypergradient 1: vector length =", len(hg1["vector"]["data"]))
        print("Encoded hypergradient 2: vector length =", len(hg2["vector"]["data"]))
        print("Aggregated hypergradient: vector length =", len(aggregated["vector"]["data"]))
        print("Aggregated timestamp:", aggregated["timestamp"])
    except HyperFeelUnavailable as exc:
        print("HyperFeel CLI is unavailable:", exc)


if __name__ == "__main__":
    main()

