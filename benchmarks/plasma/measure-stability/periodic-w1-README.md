# Periodic particle-measure Wasserstein stability fixtures v0.1

This fixture defines geodesic W1 on a one-dimensional torus, distinct from interval W1.

For normalized circle coordinates x in [0,1), the circular transport formula is

`W1_circle(mu,nu) = min_c integral_0^1 |F_mu(x)-F_nu(x)-c| dx`.

For discrete measures, the CDF difference is piecewise constant. The minimizing offset c is a length-weighted median of its interval values. The result is multiplied by the physical period length.

Known-answer controls:
- x=0 and x=1 are the same torus point (distance zero);
- moving a unit mass from 0.95 to 0.05 follows the shortest seam-crossing arc (distance 0.1, not 0.9);
- a common periodic translation preserves the distance between two measures;
- physical distance scales with the declared period length.

The Rust implementation is `WeightedEmpiricalMeasure1D::wasserstein1_periodic`. It requires identical declared physical intervals and interprets them as a torus only because the caller explicitly selected the periodic metric. It does not change the interval W1 method, whose endpoints remain distinct.

The method currently sorts the combined wrapped support events, so its complexity is O((n+m) log(n+m)); the interval W1 sweep is O(n+m). Neither metric establishes a W1-to-Coulomb-field stability theorem.
