package assurance

import "testing"

func oar(id, ref, phase, result string) OARRecord {
	return OARRecord{ReceiptID: id, DecisionRef: ref, AgentID: "agent-a", RunID: "run-a", ActionType: "refund", ActionTarget: "stripe", Phase: OARPhase(phase), Result: result, Verification: Verified}
}

func TestOARAllowedExecutionNeedsReadbackConfirmation(t *testing.T) {
	root := oar("decision", "", "decision", "allowed")
	if got := ProjectOARLifecycle([]OARRecord{root}); got.Projection != OARNonterminal { t.Fatalf("allowed decision: %+v", got) }
	exec := oar("execution", "decision", "execution", "executed")
	if got := ProjectOARLifecycle([]OARRecord{root, exec}); got.Projection != OARExecutedUnconfirmed { t.Fatalf("execution must not be confirmation: %+v", got) }
	conf := oar("confirmation", "execution", "confirmation", "confirmed")
	if got := ProjectOARLifecycle([]OARRecord{root, exec, conf}); got.Projection != OARConfirmed { t.Fatalf("full lifecycle: %+v", got) }
}

func TestOARHeldApprovedExecutionConfirmed(t *testing.T) {
	held := oar("held", "", "decision", "held")
	if got := ProjectOARLifecycle([]OARRecord{held}); got.Projection != OARNonterminal { t.Fatalf("held without approval: %+v", got) }
	approved := oar("approved", "held", "decision", "approved")
	exec := oar("execution", "approved", "execution", "executed")
	conf := oar("confirmation", "execution", "confirmation", "confirmed")
	got := ProjectOARLifecycle([]OARRecord{held, approved, exec, conf})
	if got.Projection != OARConfirmed || got.ApprovalID != "approved" { t.Fatalf("approved lifecycle: %+v", got) }
}

func TestOARRefusalAndNotConfirmedAreDistinct(t *testing.T) {
	denied := oar("denied", "", "decision", "denied")
	if got := ProjectOARLifecycle([]OARRecord{denied}); got.Projection != OARRefused { t.Fatalf("denied: %+v", got) }
	held := oar("held", "", "decision", "held")
	rejected := oar("rejected", "held", "decision", "rejected")
	if got := ProjectOARLifecycle([]OARRecord{held, rejected}); got.Projection != OARRefused { t.Fatalf("rejected approval: %+v", got) }
	allowed := oar("allowed", "", "decision", "allowed")
	exec := oar("execution", "allowed", "execution", "executed")
	notConfirmed := oar("confirmation", "execution", "confirmation", "not_confirmed")
	if got := ProjectOARLifecycle([]OARRecord{allowed, exec, notConfirmed}); got.Projection != OARNotConfirmed { t.Fatalf("not confirmed: %+v", got) }
}

func TestOARRejectsBrokenCausalAndIdentityEdges(t *testing.T) {
	root := oar("decision", "", "decision", "denied")
	exec := oar("execution", "decision", "execution", "executed")
	if got := ProjectOARLifecycle([]OARRecord{root, exec}); got.Projection != OARContradicted { t.Fatalf("execution after denial: %+v", got) }
	root = oar("decision", "", "decision", "allowed")
	exec = oar("execution", "decision", "execution", "executed")
	exec.RunID = "another-run"
	if got := ProjectOARLifecycle([]OARRecord{root, exec}); got.Projection != OARContradicted { t.Fatalf("cross-run reference: %+v", got) }
	root = oar("decision", "", "decision", "allowed")
	exec = oar("execution", "decision", "execution", "executed")
	conf := oar("confirmation", "some-other-execution", "confirmation", "confirmed")
	if got := ProjectOARLifecycle([]OARRecord{root, exec, conf}); got.Projection != OARContradicted { t.Fatalf("wrong exact execution reference: %+v", got) }
}

func TestOARVerifierMissingOrInvalidNeverPromotes(t *testing.T) {
	root := oar("decision", "", "decision", "allowed")
	root.Verification = Missing
	if got := ProjectOARLifecycle([]OARRecord{root}); got.Projection != OARIndeterminate { t.Fatalf("missing verifier: %+v", got) }
	root.Verification = Invalid
	if got := ProjectOARLifecycle([]OARRecord{root}); got.Projection != OARContradicted { t.Fatalf("invalid verifier: %+v", got) }
}

func TestOARDuplicateRecordsAndMultipleExecutionsAreContradicted(t *testing.T) {
	root := oar("decision", "", "decision", "allowed")
	dup := root
	if got := ProjectOARLifecycle([]OARRecord{root, dup}); got.Projection != OARContradicted { t.Fatalf("duplicate id: %+v", got) }
	first := oar("execution-1", "decision", "execution", "executed")
	second := oar("execution-2", "decision", "execution", "failed")
	if got := ProjectOARLifecycle([]OARRecord{root, first, second}); got.Projection != OARContradicted { t.Fatalf("multiple executions: %+v", got) }
}

func TestOARFailedExecutionIsNotReportedAsExecuted(t *testing.T) {
	root := oar("decision", "", "decision", "allowed")
	failed := oar("execution", "decision", "execution", "failed")
	if got := ProjectOARLifecycle([]OARRecord{root, failed}); got.Projection != OARFailedExecution { t.Fatalf("failed execution must remain distinct from executed-unconfirmed: %+v", got) }
}
