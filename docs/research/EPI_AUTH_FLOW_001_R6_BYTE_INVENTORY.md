# EPI-AUTH-FLOW-001 R6 — qualified byte-bound authority-flow inventory

## Purpose

R6 converts the successfully qualified R5R5-R2 discovery and exact production-consumer witness measurements into a persistent byte-bound inventory of the exact production source subject.

R6 is a fresh direct child of unchanged `main`; it is not stacked on either the measurement product or the never-merge qualifier.

## Qualified derivation

```text
source/main
adb69f11fa8068b019cc5bb598d0c7726a197fc9
tree 35a6c5fdba319556af9bb487838734f67c8ac0d6
        |
        +-- R5R5-R2 measurement product
            9d7f3cebc90ed69553d0487bd9f312c2259f5fdf
            tree b801476b789ea3e5c534b351a1bd4c96fbb9aeca
                    |
                    +-- exact qualifier
                        9c4501835500b4c90c4f975b41c63a885d5f972b
                        tree 3d5c2f8f1a1fc1f47975aeb97a72fc7e3154d8db
```

Qualified workflow evidence:

```text
run      35616089981
job      106387046876
artifact 10684849865
artifact sha256 e2389e8dc1700898e5d956055cc7862a14f7f723398027491c2c9cce6e64415a
```

Qualified stdout files:

```text
discovery sha256 07cc24071bd84cb277dd91a668f1da96b29e5c4c37322b83b10b3ff195870504
consumer  sha256 26ed61bb6f4cd49ac1673b3a96e95285772417586728eb657f244e64f36fdea2
```

They established, under their declared measurement-only claim ceilings:

```text
PASS_DISCOVERY
PASS_CONSUMER_WITNESS
```

## Persistent reconstruction

The Actions artifact is retained only temporarily. R6 therefore does not depend on keeping the ZIP forever.

Instead, `scripts/inventory_epistemic_authority_flow.py`:

1. retrieves the exact two R5 measurement programs from the immutable product commit;
2. replays them against the unchanged production source tree;
3. requires their stdout SHA-256 values to equal the exact outputs from the successful qualifier artifact;
4. parses those replayed outputs as the canonical inventory input;
5. byte-binds every discovered source file to the immutable source commit.

This preserves both **program identity** and **observed qualified-output identity** without creating a second hand-edited path manifest.

## Inventory semantics

The qualified discovery contains exactly:

```text
17 surfaces
193 surface memberships
108 unique production-file paths
```

For every unique path R6 binds:

```text
repository-relative path
Git blob object id
byte length
SHA-256
```

Each surface retains the exact R5R5 qualified path-set digest:

```text
sha256(utf8("\n".join(sorted_paths)))
```

R6 additionally emits a domain-separated surface-content digest over path + blob + byte length + SHA-256 and a separate domain-separated union digest.

## Conservative lexical semantics

The R5 discovery is intentionally broad:

```text
file appears in lexical surface
!= every matching occurrence is runtime code
```

A production Rust file may appear because of source inside an embedded `#[cfg(test)]` module. Exact production consumer chains therefore remain separate from broad file-level lexical membership.

## Qualified production-consumer witnesses

R6 requires exactly:

```text
coding_confidence_to_generic_llm_context
formal_verification_is_separate_optional_property
generic_llm_context_is_internal_not_admission
generic_llm_context_to_system_prompt
verified_generation_overbroad_guarantee_vocabulary
verified_generation_property_split
verified_generation_real_execution_fail_closed
```

These witnesses establish exact source-level chains under the R5R5-R2 measurement contract; they do not approve the semantics carried by those chains.

## Source-subject transfer rule

The qualifier proved the R5R5-R2 product is a direct child of the exact source subject and changes only:

```text
docs/research/EPI_AUTH_FLOW_001_R5R5_R2_DISCOVERY.md
scripts/audit_epistemic_expression_consumers.py
scripts/discover_epistemic_authority_flow.py
```

The qualifier itself changes only:

```text
.github/workflows/qual-epi-auth-flow-001-r5r5-r2.yml
```

No production Rust file was modified in either delta. R6 therefore binds the inventory to:

```text
adb69f11fa8068b019cc5bb598d0c7726a197fc9
```

An R6 qualification is not portable to a newer source tree. Any source-tree change requires fresh discovery/inventory revalidation before downstream repair evidence transfers.

## Exact R6 qualification

The R6 qualifier should:

1. prove `main -> R6 product -> qualifier` exact ancestry;
2. prove R6 changes only this research document and inventory program;
3. ensure the frozen R5 product and R5 qualifier commits are available;
4. detach the exact R6 product;
5. syntax-check the inventory program;
6. execute it twice and require byte-identical output;
7. require the exact source/product/qualifier/run/job/artifact identities and qualified stdout hashes;
8. require all 17 exact qualified path-set digests;
9. require all seven production-consumer witness groups;
10. require `UNION count=108 memberships=193`;
11. preserve the complete report and report SHA-256.

Only then may the exact R6 subject claim:

```text
PASS_INVENTORY
```

## Claim ceiling

`PASS_INVENTORY` means only that the exact qualified discovery surfaces and consumer witnesses were faithfully reconstructed and byte-bound to the exact source subject.

It does not establish:

```text
semantic correctness of an authority mapping
truth of a claim
cryptographic verification
authentication
reproducibility
replication
calibrated probability/certainty
causal validity
formal correctness beyond an exact proved property
scientific truth
source/evidence admission
expression authority
permission to suppress hedging
permission to bypass a gate
external action authority
```

The next producer and consumer repairs remain proposition-specific and separately qualified.
