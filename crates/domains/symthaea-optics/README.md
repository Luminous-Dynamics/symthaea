# symthaea-optics

Geometric optics for Symthaea, completing the applied classical-physics layer
(the core crates have Maxwell/FDTD, not ray optics). Angles in radians.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- Thin-lens/mirror imaging (`image_distance`, `magnification`).
- Snell's law refraction + total internal reflection (`refraction_angle`,
  `critical_angle`).
- Diffraction gratings (`grating_angle`).

```bash
cargo test -p symthaea-optics
```
