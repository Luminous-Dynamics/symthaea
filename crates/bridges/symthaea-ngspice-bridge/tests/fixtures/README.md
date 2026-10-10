# ngspice fixtures

## RC transient reference

`rc_step_reference.cir` is a self-contained ideal RC step circuit:
- source: 0 V to 1 V pulse with a 1 ns rise/fall;
- resistor: 1 kΩ;
- capacitor: 1 µF, initially 0 V;
- transient interval: 0–2 ms, maximum step 0.1 ms.

For the ideal first-order step, `tau = R*C = 1 ms`, and
`v_out(t) = 1 - exp(-t/tau)` V after the edge (neglecting the finite 1 ns
rise). Thus the analytical reference is approximately 0.63212056 V at 1 ms
and 0.86466472 V at 2 ms. Because ngspice may choose additional adaptive
points, compare by nearest time sample or interpolate under an explicitly
declared tolerance; do not require exactly three output points.

To manually exercise the netlist with ngspice 47 or another explicitly
recorded supported version, run it from a fresh, empty working directory:

```sh
ngspice -n -b -o run.log /path/to/rc_step_reference.cir
```

The `-n` option disables user `.spiceinit` loading; `-b` selects batch
mode and `-o` captures batch logs. The netlist writes ASCII vector data to
`rc_step_actual.raw` in the working directory. Preserve the executable
version, netlist bytes, environment/working-directory manifest, rawfile, and
log as separate artifacts in any automated run.

**Evidence boundary:** the checked-in `rc_step_ascii.raw` is a deterministic
analytical golden fixture for parser tests, not a rawfile claimed to have been
produced by a real ngspice invocation. The `.cir` file is a reproducible
solver-input fixture; it has not been executed as part of this change.
