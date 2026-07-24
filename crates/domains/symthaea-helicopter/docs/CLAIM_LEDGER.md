# Symthaea Helicopter Claim Ledger

The crate separates research, simulator, HIL, ground-test, flight-test, and
regulatory claims. Evidence from a lower level does not automatically satisfy a
higher level.

The default ledger defines four bounded claims:

- deterministic reduced-order simulation, capped at software-in-the-loop;
- the physical control boundary, capped at hardware-in-the-loop;
- a traceable model of a named research airframe, capped at ground test;
- airworthiness or regulated approval, requiring actual flight-test,
  independent safety review, and regulatory approval evidence.

`Supported` means the declared requirements for the requested level are present
and verified. `Incomplete` means evidence is missing. `Refused` means the
requested level exceeds the claim's declared ceiling. Nothing in this crate by
itself establishes airworthiness, permission to fly, or regulatory approval.
