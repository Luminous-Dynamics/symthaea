# LQCD-020P frozen regression targets

Independent authority: LQCD-020O / PR #2439.

For the six-vector synthetic fixture:

- free four-parameter model: `sigma = 0.1847670004239007`, `chi2/dof = 0.13566139440388655`;
- fixed-`e=pi/12`, free-`l` model: `sigma = 0.17693500020385283`, `chi2/dof = 0.6665391335487374`;
- fixed-`e=pi/12`, `l=0` model: `sigma = 0.17607757425274784`, `chi2/dof = 1.1223735883704522`.

The production Rust tests must reproduce these values within their declared tolerances. These are qualification values only, not physical SU(3) scale-setting results.
