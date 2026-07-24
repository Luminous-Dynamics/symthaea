#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
service = (root / 'src/service.rs').read_text()
coord = (root / 'src/coordination.rs').read_text()
lib = (root / 'src/lib.rs').read_text()

required_service = [
    'pub enum DurableTransitionKind',
    'pub struct DurableReconciliationRequired',
    'pub enum DurableReconciliationOutcome',
    'pending_reconciliation: Option<DurableReconciliationRequired>',
    'fn halt_for_uncertain_commit(',
    'pub fn inspect_pending_reconciliation(',
    'DurableTransitionKind::Observation',
    'DurableTransitionKind::RuntimeEvent',
    'DurableTransitionKind::ActionPreparation',
    'DurableTransitionKind::ActionCompletion',
]
missing = [item for item in required_service if item not in service]
if missing:
    print('missing durable uncertainty contracts:', *missing, sep='\n- ')
    sys.exit(1)

if 'self.halt(ServiceHaltReason::PersistenceOutcomeUnknown' in service:
    print('outcome-ambiguous persistence still halts without a reconciliation witness')
    sys.exit(1)
if 'self.halt(ServiceHaltReason::CompletionPersistenceOutcomeUnknown' in service:
    print('ambiguous completion still halts without a reconciliation witness')
    sys.exit(1)

for item in ['pub fn candidate_revision(', 'pub fn inspect_durable_head(']:
    if item not in coord:
        print(f'missing coordinated-writer witness API: {item}')
        sys.exit(1)

for item in ['DurableReconciliationOutcome', 'DurableReconciliationRequired', 'DurableTransitionKind']:
    if item not in lib:
        print(f'missing public reconciliation export: {item}')
        sys.exit(1)

print('durable uncertainty contracts: ok')
