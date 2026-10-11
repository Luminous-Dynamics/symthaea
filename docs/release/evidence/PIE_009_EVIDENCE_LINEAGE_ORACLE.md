# PIE-009B campaign evidence-lineage oracle

## Purpose

Freeze implementation-independent semantics for pessimistic / nominal / optimistic PIE campaigns so scenario sweeps vary values **inside one frozen evidence bundle** rather than cherry-picking different source versions.

The oracle is `scripts/pie-evidence-lineage-oracle.py` and imports no Symthaea code.

## Executed evidence

The final candidate self-test was executed locally on 2026-09-12 and returned `ok` before the checked-in reference was created.

The fixtures prove:

1. pessimistic, nominal, and optimistic scenarios preserve the same record IDs, source families, versions, process IDs, and parameter IDs;
2. each scenario selects only the corresponding low / nominal / high value from the frozen record envelope;
3. duplicate active sources for the same process/parameter slot fail closed;
4. multiple active versions from one evidence family fail closed;
5. a superseded evidence record may not remain active in a frozen campaign bundle;
6. a new dataset or source version creates a new bundle/root rather than mutating prior campaign evidence in place;
7. invalid low/nominal/high envelopes fail closed.

## Intended campaign rule

A scenario family is derived from a single frozen evidence bundle:

`bundle -> pessimistic`

`bundle -> nominal`

`bundle -> optimistic`

A new source/version creates `bundle'`, not a silent rewrite of `bundle`.

This preserves reproducibility and prevents an optimistic run from receiving newer/favorable evidence that the pessimistic run did not use.

## Limitations

This oracle does not judge source quality or resolve conflicting independent scientific sources. Future PIE evidence graphs may represent competing sources explicitly, but a campaign must still declare exactly which frozen evidence bundle it selected.

The oracle also does not prove Moon/Mars process performance, economics, plant feasibility, or hardware authority.

Tracks PIE-009 #1641, spatial evidence #1702, Phase-0 gates #1647, and master #1604.
