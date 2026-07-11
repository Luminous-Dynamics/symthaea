# symthaea-acoustics

Physical acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
combination, and the Doppler effect. Complements `symthaea-dsp` (digital signals)
with the physical-sound layer.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `speed_of_sound_air(temp_celsius)` — ≈343.2 m/s at 20 °C.
- `wavelength` / `frequency` — `λ = c/f`.
- `combine_decibels` — incoherent SPL sum (two 60 dB → 63 dB).
- `doppler_frequency` — observed pitch under relative motion.

```bash
cargo test -p symthaea-acoustics
```
