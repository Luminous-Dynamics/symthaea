# Family 269 Laughlin gap and disorder-stability crosswalk

This is a preliminary, source-pinned paper-to-Lean scope audit for OpenAI Math's spherical fermionic Laughlin (V_1) gap and bounded scalar-disorder stability work.

The machine-readable crosswalk is `FAMILY_269_LAUGHLIN_GAP_DISORDER_CROSSWALK_V0_1.json`.

## Critical scope distinction

The selected Family 269 Lean scope formalizes the unperturbed Fock-space inequality, the spherical Laughlin gap statements and the planar endpoint. The scope note explicitly places the October paper's stability under projected bounded scalar potentials and uniqueness of the perturbed ground state outside the selected Lean formalization.

This is a scope boundary, not a claim that the paper's perturbative theorem is false.

## Distinct constants

- (1/25): sharp finite-spherical comparator inequality in the Fock/LaughlinGap formal path.
- (1/100): separately formalized weaker finite-spherical statement in `OAI.Analysis.Laughlin.Main`.
- (gamma_*=4616733319001/10^{14}>1/25): asymptotic Fock-space inequality coefficient for sufficiently large flux.

Do not conflate these with the paper's existential perturbed gap (Delta_*) or disorder threshold (lambda_*).

## Validation limits

Exact source files were inspected at the pinned commit. No clean Lean replay, complete axiom/dependency audit, independent proof review or finite-size numerical reproduction has been performed by this audit. Comparator statement files intentionally contain `sorry`; proof solutions are separate modules.
