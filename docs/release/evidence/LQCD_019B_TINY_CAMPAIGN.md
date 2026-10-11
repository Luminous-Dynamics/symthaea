# LQCD-019B — independent tiny multi-observable pure-SU(3) campaign

## Authority scope

This is an **independently executed standard-library Python campaign-development subject**. It imports no Symthaea/Rust code. It is designed to challenge the HB+1OR transition and the multi-observable qualification logic before larger production ensembles are attempted.

It is **not** a physical lattice-QCD result, a reference-benchmark reproduction, a physical scale setting, a topology-sector result, or a glueball prediction.

## Frozen subject

- lattice: `2 x 2 x 2 x 2`
- beta: `5.7`
- action/transition semantics: pure-SU(3) Wilson, Cabibbo-Marinari heat-bath + one overrelaxation sweep per cycle
- force backend: direct six-staple force, with an independent five-probe parity check before production
- starts: one cold identity chain + one independently disordered chain
- production streams: distinct ChaCha8 stream coordinates
- burn-in: `60` cycles
- retained stride: `2` cycles
- retained samples: `24` per chain / `48` total
- topology/energy diagnostic flow: analytic Wilson-action Lie-Euler, `dt=0.001`, four steps to `t=0.004`; this reuses the independently qualified LQCD-017D clover and direct-staple semantics. It is not the production RK3 flow subject.

Exact executed local subject SHA-256:

`17da8cce4cba5948ed73b9020dbeb229f8c69e056a331e80f787bf366dd77b04`

Frozen canonical result SHA-256 asserted by the subject:

`d2c2f400678fc5961ceced3c03f71c5a4dc70afb05e63451961a93100ec8cf8b`

Frozen retained-history CSV SHA-256:

`5c70d027dc2b77758474de081c5aaed9f506caabd1695a32a049dd6d05ce8996`

Frozen stdout SHA-256:

`e8d2b81c03e2d7ab957f1a71f4447e71c563c929b7a524e3669bf7f4de3210bf`

The subject verifies the published ChaCha8 zero-key block and requires direct-staple / five-probe subgroup-force parity before running. The observed maximum force disagreement is `8.881784197001252e-16`.

## Frozen diagnostics

| observable | cold mean | disordered mean | max rank/folded R-hat | ESS cold | ESS disordered |
|---|---:|---:|---:|---:|---:|
| plaquette | 0.5956255854 | 0.5921309445 | 1.0326583917 | 24.0 | 17.2477 |
| Polyakov Re | -0.1813283054 | -0.0240673134 | 1.3990380189 | 6.1081 | 3.7497 |
| Polyakov Im | -0.3833747943 | 0.1946920704 | 1.6307253245 | 5.6052 | 5.5395 |
| |Polyakov| | 0.4789190296 | 0.4066933283 | 1.0157574815 | 5.7657 | 14.1202 |
| center-aligned Polyakov Re | 0.4763483735 | 0.3997578564 | 1.0130246007 | 5.7759 | 13.4746 |
| temporal-spatial 1x1 Wilson mean | 0.5938639243 | 0.5829398978 | 1.0042778076 | 24.0 | 18.8186 |
| flowed Q at t=0.004 | -0.0046949107 | -0.0010069907 | 1.0485935373 | 12.4499 | 24.0 |
| flowed clover E at t=0.004 | 1.4655632679 | 1.4833113142 | 0.9969762776 | 10.0630 | 10.4097 |

No universal R-hat/ESS acceptance threshold is encoded by this evidence subject.

## Center-symmetry result

Treating the raw real or imaginary component of the pure-SU(3) Polyakov loop as an ordinary scalar convergence observable is misleading because those components retain `Z3` center orientation.

The retained center-sector histories are:

- cold chain: sector counts `{0: 2, 1: 0, 2: 22}`, with `1` observed retained-sample sector transition;
- disordered chain: sector counts `{0: 8, 1: 13, 2: 3}`, with `2` observed retained-sample sector transitions.

This explains the strong disagreement in raw `Re P` / `Im P` while the center-invariant `|P|` and center-aligned real observable are much more compatible.

The campaign therefore identifies **center-sector occupancy/mixing as a distinct slow categorical mode** that must be recorded separately from center-invariant Polyakov observables. A future production qualification should not reject an otherwise stationary ensemble merely because two chains occupy different symmetry-related sectors, nor should it ignore poor sector mobility when sector sampling matters to the scientific question.

## Topology boundary

The `t=0.004` clover-Q histories are retained as continuous slow-mode diagnostics. On this tiny/shallow-flow subject all values remain close to zero; no claim is made that nearest-integer classification represents physical topological sectors. The campaign does not establish topological tunnelling or susceptibility.

## Files

- `scripts/lqcd-019b-tiny-campaign.py` — exact deterministic subject;
- `docs/release/evidence/LQCD_019B_TINY_CAMPAIGN_HISTORY.csv` — all 48 retained multi-observable histories, including E(t) and Q(t) at t=0.001..0.004;
- `docs/release/evidence/LQCD_019B_TINY_CAMPAIGN_RESULT.txt` — frozen diagnostics stdout.

## Scientific conclusion

This campaign is useful precisely because it refuses a one-number notion of equilibration. Plaquette and local Wilson-loop histories can look healthy while symmetry-sector dynamics tell a different story. The result motivates a dedicated center-sector diagnostic in the production qualification stack and a larger-volume/longer-chain campaign before external benchmark reproduction is attempted.
