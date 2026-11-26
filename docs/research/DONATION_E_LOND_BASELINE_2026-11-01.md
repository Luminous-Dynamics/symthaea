# Exact donation e-LOND baseline and O(log t) verification

**Status:** research reference implementation and benchmark adapter. The algorithm is the published donation e-LOND procedure for a one-pass stream with one e-value per arriving hypothesis. It is not a repeated-evidence dynamic e-closure algorithm.

## 1. Published guarantee and contract

Xu, Fischer and Ramdas (UAI 2026), *Improving online FDR procedures via online analogs of e-closure and compound e-values*, construct online compound e-values by weighted donations. Donation e-LOND:
- controls SupFDR at level delta under arbitrary dependence of the input e-values;
- strictly improves standard e-LOND;
- computes each online decision in O(log t) using an augmented binary search tree.

The method assumes an infinite sequence H_1,H_2,... where each hypothesis arrives once with a single e-value E_i; past E_i are fixed after arrival. The arbitrary-dependence guarantee is across the e-values. This is a serious industry comparator for single-pass online testing, but it is **not automatically valid** when old hypotheses continue receiving new evidence and E_i becomes E_i,t. Updating past e-values in the donation formula is a new statistical problem requiring a process-level proof.

External references:
- UAI 2026 proceedings: https://proceedings.mlr.press/v337/xu26a.html
- Full preprint: https://arxiv.org/abs/2603.24792

## 2. Exact formula

Fix delta in (0,1] and a nonnegative summable sequence gamma_i with sum_i gamma_i <= 1. Let R_(t-1) be the previous rejection set and r=|R_(t-1)|.

Define the donation wealth

\`\`\`
Wbar_t =
  sum_{i in R_(t-1)} gamma_i *
      min(E_i - 1/(delta*gamma_i*(r+1)), 1)
  + sum_{i not in R_(t-1)} gamma_i * min(E_i, 1).
\`\`\`

The donation e-LOND test level is

\`\`\`
alpha_t = delta*gamma_t*(r+1)
          / (1 - min(delta*(r+1)*Wbar_t, 1)).
\`\`\`

The hypothesis is rejected when E_t >= 1/alpha_t. If gamma_t=0, its threshold is treated as +infinity (it cannot be rejected). If the denominator is zero, alpha_t is infinite and the e-value threshold is zero; every nonnegative E_t is accepted.

Standard e-LOND has test level delta*gamma_t*(r+1), i.e. e-value threshold 1/(delta*gamma_t*(r+1)). The donation wealth can only reduce the e-value threshold.

## 3. Exact O(log t) data structure

For each previously rejected i, define

\`\`\`
x_i = gamma_i * (E_i - 1)
q_t = 1/(delta*(r+1)).
\`\`\`

The rejected-coordinate term can be rewritten as

\`\`\`
min(gamma_i, gamma_i*E_i - q_t)
  = gamma_i*E_i - q_t, if x_i <= q_t
  = gamma_i,            if x_i > q_t.
\`\`\`

Therefore,

\`\`\`
Wbar_t = W_nonrejected
       + prefix_gamma_e(q_t)
       - q_t * prefix_count(q_t)
       + total_rejected_gamma
       - prefix_gamma(q_t),
\`\`\`

where the prefixes are over the rejected indices with x_i <= q_t.

The AVL tree key is exact rational x_i. Equal keys are stored in one bucket with bucket count, sum(gamma_i), and sum(gamma_i*E_i). Each node also stores subtree count and both weighted sums. Prefix query is O(log r). The nonrejected wealth

\`\`\`
W_nonrejected = sum_{i not in R} gamma_i * min(E_i,1)
\`\`\`

is a scalar accumulator, updated in O(1) for each new non-rejected arrival. A rejected arrival inserts one key in O(log r). Hence threshold query and a new arrival are O(log t), with O(t) storage.

A separate exact O(t) reference formula and state verifier are retained. The wrapper runs the independent scan once at the end; running the O(t) scan after every arrival would turn an otherwise O(t log t) online simulation into O(t^2).

## 4. Implementation

Files:
- validation/donation_e_lond_reference.py
- validation/test_donation_e_lond_reference.py
- validation/benchmark_donation_e_lond_reference.py
- validation/test_benchmark_donation_e_lond_reference.py

The code is Fraction-only. It rejects floating point in exact evidence, checks delta and the cumulative gamma budget, supports zero-weight arrivals, aggregates duplicate AVL keys, and uses inclusive prefix semantics x_i <= q_t exactly as in the paper.

The exact scan verifier recomputes Wbar_t from the full history. The AVL structural verifier checks sorted order, balance, cached height, subtree count, sum gamma and sum gamma*E.

## 5. Reproducible benchmark

Run:

\`\`\`bash
python validation/benchmark_donation_e_lond_reference.py \\
  --sizes 1000 10000 50000 --repeats 3 \\
  --output donation-e-lond-results.json
\`\`\`

The benchmark uses exact Fraction e-values, telescoping gamma_i=1/(i(i+1)), delta=1/10, a fixed seed and three measurements:
- total indexed one-pass decision run (AVL updates + O(log t) thresholds);
- one final O(t) reference threshold scan;
- one final independent O(t) state verifier.

It reports AVL height and exact threshold parity. This is a Python algorithmic benchmark, not Rust production latency. The single synthetic input generator is not a statistical validity proof.

## 6. Test matrix

- exact tie x_i=q_t must use the inclusive prefix branch;
- duplicate x_i values must aggregate correctly;
- zero gamma never rejects;
- zero denominator returns zero e-value threshold;
- randomized exact streams compare every indexed threshold and wealth against the O(t) formula;
- e-LOND rejection sets must be contained in donation e-LOND rejection sets;
- cumulative gamma > 1, float evidence, negative e-values and invalid zero-weight rejection are rejected;
- benchmark result must match the independent reference.

Tests are authored but have not been executed from a connected repository checkout in this environment. No CI pass is claimed.

## 7. What this baseline does not prove for dynamic evidence

The published donation proof defines a single E_i per arriving hypothesis. The donation sequence is chosen from that fixed stream and yields online compound e-values controlling SupFDR. In our dynamic setting, an old hypothesis can acquire additional evidence after later hypotheses arrive. Simply replacing E_i with the current M_i,t in Wbar_t does not follow from the published theorem: it changes both the donation amounts and the time path of compound e-values.

A worthwhile new theorem would have to establish a projective **process-valued** donation construction:
- each coordinate starts at neutral one while dormant;
- old evidence updates are predictable in a common global filtration;
- donor/recipient transfers preserve a countable true-null supermartingale or valid time-uniform compound certificate;
- the active-set restrictions have exact historical prefix equality;
- the same procedure supports the required stopped-FDR or SupFDR adjuster without leaking wealth from false coordinates into true-null certificates.

Until that theorem exists, benchmark donation e-LOND only in the single-pass lane and retain #9695 as the repeated-evidence arbitrary-dependence baseline.
