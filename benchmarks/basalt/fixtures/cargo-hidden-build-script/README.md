# BASALT Cargo build-script hidden-input fixture

This fixture is intentionally incorrect Cargo dependency metadata for a controlled challenge.

- build.rs reads build.rs-input.txt.
- build.rs declares only rerun-if-changed=build.rs.
- build.rs-input.txt is Git-ignored and untracked.
- A warm Cargo target directory is expected to keep the old generated output after the input changes.
- A separate fresh target directory is expected to execute build.rs again and emit the new value.

The fixture has no external dependencies and is invoked with --offline --locked. It is not a production build template.

The runner records Cargo process IDs and output digests. It does not capture the build-script child PID or independently observe file access at the kernel layer. An independent observer lane remains future work.
