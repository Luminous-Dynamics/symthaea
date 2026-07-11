# symthaea-astronomy

Observational astronomy for Symthaea — blackbody, orbital, distance, and
relativistic relations. Complements `mycelix-space` (orbital-mechanics
propagation) with the observational layer.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `wien_peak_wavelength_nm` — blackbody peak (Sun 5778 K → ~502 nm).
- `orbital_period_years` / `semi_major_axis_au` — Kepler's third law (solar units).
- `distance_modulus` / `absolute_magnitude` — the distance–magnitude relation.
- `schwarzschild_radius` — event horizon (Sun → ~2953 m).

```bash
cargo test -p symthaea-astronomy
```
