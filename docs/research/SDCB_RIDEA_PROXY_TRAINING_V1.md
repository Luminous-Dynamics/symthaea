# R-IDeA proxy-training reference v1

**Status:** bounded, dependency-free training-objective reference
**Implementation:** [scripts/research/sdcb_ridea_proxy.rs](../../scripts/research/sdcb_ridea_proxy.rs)
**Schema:** [SDCB_RIDEA_PROXY_TRAINING_V1.schema.json](SDCB_RIDEA_PROXY_TRAINING_V1.schema.json)
**Example:** [SDCB_RIDEA_PROXY_TRAINING_V1.example.json](SDCB_RIDEA_PROXY_TRAINING_V1.example.json)
**Expected result:** [SDCB_RIDEA_PROXY_TRAINING_V1.expected.json](SDCB_RIDEA_PROXY_TRAINING_V1.expected.json)

## 1. Why this module exists

R-IDeA's DeA term depends on a proxy predictor g. A score-only reference that accepts g(x) as an input does not test whether the proxy-training objective is implemented correctly. The authors' released helper fits g by balancing observed-data fit against a minimum disagreement margin from the frozen current predictor f_hat.

This module adds a transparent linear-feature reference for that objective. It is intentionally not the authors' complete Pyro/architecture-specific training code. The caller chooses a feature map, supplies the current model's predictions and initial coefficients, and receives fitted proxy coefficients plus loss decomposition and disagreement diagnostics.

## 2. Training objective

Given feature rows phi(x_i), observed targets y_i and frozen current predictions f_hat_i, fit proxy weights w:

\[ g_w(x_i)=w^T\phi(x_i) \]

\[ L(w)=\frac{1}{n}\sum_i(g_w(x_i)-y_i)^2+\alpha\frac{1}{n}\sum_i\max(0,\tau-|\hat f_i-g_w(x_i)|)+\lambda\frac{1}{d}\|w\|_2^2. \]

The terms match the released R-IDeA helper's code structure:

- `mse = mean((g_vals - y) ** 2)`;
- `margin = mean(relu(tau - abs(mu_f - g_vals)))`;
- `reg = l2 * mean(theta_g ** 2)`;
- `loss = mse + alpha * margin + reg`.

The frozen current predictions are not optimized. The proxy begins at caller-supplied coefficients, normally copied from the current model, as in the released helper's initialization from the current parameter location.

## 3. Gradient

Away from the hinge kink, the per-example margin subgradient with respect to g is:

\[ \frac{\partial}{\partial g}\max(0,\tau-|\hat f-g|)=\mathbf{1}_{|\hat f-g|<\tau}\operatorname{sign}(\hat f-g). \]

The implementation uses full-batch gradient descent and reports objective components before any acquisition score is calculated. The proxy's held-out quality and calibration still require separate evaluation.

## 4. Frozen fixture

One feature x=1, target y=0.1, current prediction f_hat=0, initial proxy coefficient w=0.05, tau=0.5, alpha=2, l2=0, learning rate 0.02, 100 steps.

The independent gradient calculation gives w=0.5225432 and disagreement |f_hat-g|=0.5225432. The proxy is pulled just beyond the requested margin. This demonstrates the training objective's effect; it does not show that the learned proxy approximates the best-in-class predictor.

## 5. Boundaries

- This is a linear-feature proxy trainer. It is not the original Pyro model, a neural proxy, or the entire upstream R-IDeA reproduction.
- Feature engineering, model capacity, optimizer, stopping and proxy validation affect results. No universal setting is implied.
- The module does not calculate EIG, MMD, R-I, DeA or EVSI. It produces proxy weights and prediction diagnostics for a caller to connect with `sdcb_ridea.rs`.
- A high disagreement margin is not automatically good. The proxy must still fit data and be checked on held-out observations.
- No hidden ground-truth function is accessed.

## 6. Validation status

13 Rust unit tests are authored and not compiled/executed in this environment. They cover hand-calculated prediction, MSE descent, hinge-margin effect, L2 shrinkage, finite-difference gradient agreement, invalid shapes/non-finite inputs and search limits. The expected result file is an independent calculation, not a Rust execution.