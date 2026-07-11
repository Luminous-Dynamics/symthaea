# symthaea-physiology

Human physiology for Symthaea — nutrition/metabolism and pharmacokinetics.
Complements the clinical crates with quantitative energy-balance and
drug-kinetics.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `nutrition` — BMI, BMR (Mifflin–St Jeor), TDEE, macronutrient energy.
- `pharmacokinetics` — half-life ↔ rate constant, one-compartment concentration
  decay, clearance.

```bash
cargo test -p symthaea-physiology
```
