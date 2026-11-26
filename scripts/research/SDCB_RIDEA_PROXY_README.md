# SDCB R-IDeA proxy training reference

Standalone linear-feature proxy trainer for the upstream R-IDeA MSE + hinge-margin + L2 loss.

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_ridea_proxy.rs -o /tmp/sdcb-ridea-proxy
/tmp/sdcb-ridea-proxy
```

## Objective

mean((g(x)-y)^2) + alpha*mean(relu(tau - abs(f_hat(x)-g(x)))) + l2*mean(weights^2).

The current predictor's values are frozen. Proxy weights start from caller-supplied coefficients and are fitted with bounded full-batch gradient descent. Callers may supply polynomial, HDC-derived or other explicit features, but must validate the representation separately.

## Relationship to the upstream code

This implements the loss structure seen in the pinned upstream R-IDeA helper, but with a linear feature map and a transparent full-batch optimizer. It does not reproduce the original model architecture, Pyro optimizer setup, candidate-specific proxy pipeline, or experiments.

## Output

Proxy coefficients, total loss, MSE, margin penalty, L2 penalty, and disagreement summaries. Feed candidate proxy predictions into `sdcb_ridea.rs` only after checking held-out fit/calibration.

13 test functions are authored; Rust compilation/execution has not been performed in this environment.