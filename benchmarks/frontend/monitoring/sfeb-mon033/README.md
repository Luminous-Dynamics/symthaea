# MON-033 — PostgreSQL Offset-Boundary Restart Challenge

MON-033 turns Debezium's PostgreSQL DBZ-2889 restart fix into an observable recovery experiment.

Targets:
- lsn_events_processed=1 on a counted row operation.
- messageType=MESSAGE after a transactional logical message.

Use the read-only observer to capture the exact source offset. An offset target that is not reached is indeterminate; do not weaken the predicate or edit connector offsets.

## Use

\`\`\`sh
bash scripts/wait-offset-target.sh mon025-pg-source lsn_events_processed_1 120 1 out/mon033-lsn-one
bash scripts/wait-offset-target.sh mon025-pg-source transactional_message 120 1 out/mon033-message

# Only after a target proof exists, in a disposable lab:
export MON025_ALLOW_OFFSET_RESTART=YES
bash scripts/restart-after-offset-target.sh out/mon033-message
\`\`\`

After restart, run SQL truth, CDC event-set and MON-032 transaction closure checks. Restart success alone is not qualification. The offset observer uses GET only; it never PATCHes or DELETEs source offsets.

Run bash scripts/test-offset-target.sh for local known-answer parser checks. No live service was started during authoring.
