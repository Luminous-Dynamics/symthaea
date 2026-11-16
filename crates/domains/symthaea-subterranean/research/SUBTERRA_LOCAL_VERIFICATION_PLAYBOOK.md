# Local verification playbook — Symthaea Subterra research

The four Subterra Cargo packages are nested research workspaces and intentionally do not join the production Symthaea workspace. A root-level cargo test --workspace therefore does not test them.

Run the explicit verification script from any directory inside the Symthaea checkout:

~~~bash
bash crates/domains/symthaea-subterranean/research/verify-subterra-reference-packages.sh
~~~

The script checks tool availability, then runs formatting, tests and strict Clippy individually for:
- subterra-voi-kernel
- subterra-survival-planner
- subterra-robust-voi
- subterra-evidence-adapter

The robust EVSI package depends on the one-step EVSI package, which is ordered first.

## Evidence discipline

The script exits non-zero on the first missing tool or failed command. It does not turn skipped checks into passes. A successful run qualifies only the exact checked-out source and host environment. It does not establish calibration, evidence authenticity, robot-in-the-loop behavior, terrestrial analogue performance, flight readiness or planetary capability.
