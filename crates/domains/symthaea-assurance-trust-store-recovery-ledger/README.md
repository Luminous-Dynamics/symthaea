# symthaea-assurance-trust-store-recovery-ledger

Tamper-evident acceptance ledger for completed trust-store recoveries.

A recovery authorization is intentionally single-use at the acceptance layer. Once a valid replacement has been accepted, the same authorization, old checkpoint, recovery commit, replacement counter epoch, or first replacement checkpoint cannot be accepted again to create a competing branch.

The ledger records:

- logical store identity,
- authorization id,
- recovery commit id,
- exact previous checkpoint digest,
- replacement store/counter epoch,
- exact first replacement checkpoint digest,
- continuity qualification artifact reference/digest,
- acceptance time,
- predecessor acceptance-record digest.

The ledger is evidence only. It performs no hardware I/O, cannot discharge the wider safety case, and never grants physical authority.
