# symthaea-domain-awareness-model-assurance

Small explicit adapter between `symthaea-model-assurance` and the domain-awareness
Operational Design Domain (ODD).

It maps every generic assurance state explicitly:

```text
Aligned    -> Aligned
Restricted -> Restricted
Unsafe     -> Unsafe
Incomplete -> Incomplete
```

The adapter also binds a deterministic report receipt into `OperationalConditions`
evidence references. It does not alter visibility, navigation, sensor, communications,
or operator evidence, and it does not grant physical authority.

This separation keeps `symthaea-model-assurance` reusable outside domain-awareness
while allowing model divergence to participate in the existing degraded-mode ODD
assessment.

## Verification

```bash
cargo test -p symthaea-domain-awareness-model-assurance
```
