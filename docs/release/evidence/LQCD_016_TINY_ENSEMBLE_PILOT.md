# LQCD-016 tiny pure-SU(3) ensemble pilot

## Executed subject

A standard-library Python implementation of the frozen ChaCha8 stream semantics, embedded-SU(2) proposal law, local Wilson-action delta, Metropolis sweep, disordered initialization and measurement schedule was executed independently before commit.

Normative checked-in subject:

`scripts/lqcd-tiny-ensemble-oracle.py --self-test`

SHA-256 of the exact locally executed pre-commit script:

`a7b1cb1a2051c59a63b1c883c2b825b3d1b09543f10f3bbf83e086507f059dbd`

The oracle imports no Symthaea or Rust code. It first checks the published ChaCha8 all-zero-key reference vector before running the lattice pilot.

## Fixed pilot geometry

- lattice: `2 x 2 x 1 x 2`
- Wilson beta: `5.7`
- seed: 32 bytes of `0x5a`
- campaign ensemble slot: `17`
- cold transition replica: `0`
- disordered transition replica: `1`
- cold transition stream: `0x0100001100000000`
- disordered transition stream: `0x0100001100010000`
- disordered initialization: two rounds of embedded-SU(2) rotations with maximum angle pi on a separate initialization stream
- disordered initialization is **not** represented as Haar-random SU(3)

The cold and disordered chains use distinct transition streams. They are therefore appropriate inputs to downstream independent-chain diagnostics in a way that common-random-number coupling would not be.

## Executed output

```text
ok
short: angle=0.17999999999999999 burn=20 stride=2 n=10 cold_stream=0x0100001100000000 disordered_stream=0x0100001100010000
short: cold_initial=1 disordered_initial=-0.0082855725487685754
short: cold_mean=0.73369130681533901 disordered_mean=0.26322102718280421 difference=0.47047027963253479
short: cold_acceptance=0.87812500000000004 disordered_acceptance=0.90625 split_rhat=11.586746995780688
longer: angle=0.5 burn=200 stride=10 n=20 cold_stream=0x0100001100000000 disordered_stream=0x0100001100010000
longer: cold_initial=1 disordered_initial=-0.0082855725487685754
longer: cold_mean=0.60236547258277762 disordered_mean=0.63922315348440306 difference=-0.036857680901625445
longer: cold_acceptance=0.69062500000000004 disordered_acceptance=0.69999999999999996 split_rhat=1.2271025266120743
```

## Interpretation

The short pilot is a deliberate negative control: with proposal width `0.18` and only 20 burn-in sweeps, the cold and disordered starts remain grossly separated. Classical split-R-hat is about `11.59`. The run therefore does **not** establish equilibration.

A wider proposal (`0.5`) and a much longer 200-sweep burn-in materially improve mixing on this tiny fixture, but split-R-hat remains about `1.227`. This pilot therefore still does **not** establish convergence. The correct result of the experiment is that additional tuning/longer runs and stronger diagnostics are required.

The acceptance-rate change (`~0.89` in the short narrow-proposal pilot versus `~0.70` in the longer wider-proposal pilot) also demonstrates that proposal width is a scientifically material algorithm parameter. This motivates LQCD-016A's separate pre-production tuning phase rather than silent or in-production adaptation.

## Non-claims

These are tiny qualification lattices and a simple random-walk Metropolis algorithm. The recorded means are not continuum results, not precision pure-SU(3) reference values, and not glueball predictions. Split-R-hat here is the classical split diagnostic, not rank-normalized/folded R-hat. No universal convergence threshold is encoded.

The pilot establishes useful failure evidence: the infrastructure can expose obviously inadequate burn-in and incomplete cross-start mixing instead of declaring success from a plausible-looking plaquette trace.
