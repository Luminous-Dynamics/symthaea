# symthaea-dsp

Digital signal processing for Symthaea — spectra, convolution, filtering,
sampling theory. Connects to the audio work (Broca, muse).

Pure `std`, zero deps, no `symthaea-core` link. Checked vs known transforms.

- `dft::{dft, magnitude}` — DFT and magnitude spectrum.
- `signal::{convolve, moving_average, nyquist_frequency, will_alias}`.

```bash
cargo test -p symthaea-dsp
```
