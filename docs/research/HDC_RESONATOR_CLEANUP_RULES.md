# HDC Resonator Cleanup Rules

This harness compares four cleanup nonlinearities in the same coupled two-factor factorization task: sign, softmax, ReLU, and polynomial.

The comparison follows the rule definitions described by Yeung, Poduval, and Imani (2026): sign uses the similarity response; softmax uses temperature-scaled weights; ReLU uses normalized positive similarities; polynomial uses normalized positive similarities raised to degree p. Their reported bipolar main settings use sign projection for sign and softmax, while ReLU and polynomial retain real-valued reconstructed states; their validation selects softmax inverse temperature beta=20 and polynomial degree p=2. This harness therefore uses temperature 0.05 (approximately beta=20), polynomial degree 2, and the rule-specific projection choices rather than tuning each rule independently on the qualification matrix. The paper also emphasizes separating correct convergence, spurious convergence, and non-convergence because identical accuracy can hide different failure modes. citeturn1search0turn1search12

## Matrix

- dimensions: 2,048 and 8,192
- codebook sizes: 4 and 8
- shared-component probabilities: 0.0, 0.50, 0.75
- query noise: 0.0 and 0.20
- cleanup rules: sign, softmax, ReLU, polynomial
- solver seeds: 0x51, 0xA7, 0xD3
- maximum iterations: 16

The matrix contains 96 cells and 2,304 resonator trials. The same deterministic codebook and corrupted query are used for the exhaustive n² reference and each cleanup rule.

## Interpretation boundary

This is a controlled algorithm-family comparison, not a universal ranking. Rule behavior remains stratified by representation geometry, dimension, noise, convergence mode, and projection choice. The existing temperature and geometry sweeps remain separate controls.

Production behavior is preserved: ResonatorConfig still defaults to the pre-existing softmax cleanup and does not enable the research sign projection. The new cleanup-rule fields are opt-in configuration surfaces.

## External context

The 2026 cleanup-rule study explicitly frames resonator networks as a family indexed by the cleanup nonlinearity and reports that the nonlinear choice changes capacity and dominant failure modes. citeturn1search0 The earlier resonator literature treats factorization as a coupled search problem whose operational capacity depends on dimensionality and codebook structure. citeturn0search5
