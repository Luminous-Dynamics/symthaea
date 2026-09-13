# Qualification Attempt V3 — evidence boundaries

This branch-only V3 attempt layer is generic infrastructure for qualification history.

It establishes only:

- exact source/profile/recipe/input-closure/environment attempt theorem identity;
- distinct retry registration identity via declared attempt sequence;
- provider attempt provenance separate from semantic work identity;
- pre-start infrastructure/cancellation/unknown dispositions distinct from post-start recipe failure;
- immutable terminal observation identity;
- append-only-compatible history construction that preserves failed attempts after later PASS attempts;
- partial profile history without interpreting unattempted recipes as failure.

It does **not** establish:

- externally anchored chronology of attempt sequence values;
- successful qualification of an entire profile;
- proof of profile cross-cutting rules;
- current admission or merge authority;
- provider authenticity or detached attestation;
- scientific validity.

A later positive portable receipt should require exact successful coverage of every `required_recipe_id`, separately bind evidence for every `required_cross_cutting_rule`, and preserve the complete attempt history rather than rewriting it to green.
