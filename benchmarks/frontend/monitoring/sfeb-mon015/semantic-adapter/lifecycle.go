package assurance

// OAR 0.3 lifecycle projection over normalized records. The caller must first
// run the maintained OAR verifier; this code never validates OAR signatures,
// seals, chains, checkpoints, disclosures, approval proofs or authority proofs.

type OARPhase string

const (
	OARDecision     OARPhase = "decision"
	OARExecution    OARPhase = "execution"
	OARConfirmation OARPhase = "confirmation"
)

type OARRecord struct {
	ReceiptID    string
	DecisionRef  string
	AgentID      string
	RunID        string
	ActionType   string
	ActionTarget string
	Phase        OARPhase
	Result       string
	Verification Verification
}

type OARProjection string

const (
	OARConfirmed           OARProjection = "confirmed"
	OARNotConfirmed        OARProjection = "not_confirmed"
	OARRefused             OARProjection = "refused"
	OARFailedExecution     OARProjection = "failed_execution"
	OARExecutedUnconfirmed OARProjection = "executed_unconfirmed"
	OARNonterminal         OARProjection = "nonterminal"
	OARIndeterminate       OARProjection = "indeterminate"
	OARContradicted        OARProjection = "contradicted"
)

type OARReport struct {
	Projection     OARProjection
	RootDecisionID string
	ApprovalID     string
	ExecutionID    string
	ConfirmationID string
	Findings       []string
}

// ProjectOARLifecycle projects one action lifecycle. It accepts a root decision
// (allowed/held/denied), optional approval decision (approved/rejected/expired),
// execution (executed/failed/refused), and optional confirmation. executed is
// never a terminal confirmed effect without confirmation/confirmed.
func ProjectOARLifecycle(records []OARRecord) OARReport {
	out := OARReport{Projection: OARIndeterminate}
	if len(records) == 0 {
		out.Findings = append(out.Findings, "empty_lifecycle")
		return out
	}
	byID := make(map[string]OARRecord, len(records))
	var root, approval, execution, confirmation *OARRecord
	for i := range records {
		r := records[i]
		if r.Verification == Invalid {
			return oarFinding(out, OARContradicted, "upstream_oar_verifier_failed")
		}
		if r.Verification != Verified {
			return oarFinding(out, OARIndeterminate, "upstream_oar_verification_missing")
		}
		if r.ReceiptID == "" || r.AgentID == "" || r.RunID == "" || r.ActionType == "" || r.ActionTarget == "" {
			return oarFinding(out, OARIndeterminate, "required_lifecycle_identity_missing")
		}
		if _, exists := byID[r.ReceiptID]; exists {
			return oarFinding(out, OARContradicted, "duplicate_receipt_id")
		}
		byID[r.ReceiptID] = r
		switch r.Phase {
		case OARDecision:
			switch r.Result {
			case "allowed", "held", "denied":
				if r.DecisionRef != "" || root != nil {
					return oarFinding(out, OARContradicted, "multiple_or_nonroot_base_decisions")
				}
				cp := r
				root = &cp
			case "approved", "rejected", "expired":
				if approval != nil {
					return oarFinding(out, OARContradicted, "multiple_approval_decisions")
				}
				cp := r
				approval = &cp
			default:
				return oarFinding(out, OARContradicted, "unknown_decision_result")
			}
		case OARExecution:
			if execution != nil {
				return oarFinding(out, OARContradicted, "multiple_executions_for_action")
			}
			if r.Result != "executed" && r.Result != "failed" && r.Result != "refused" {
				return oarFinding(out, OARContradicted, "invalid_execution_result")
			}
			cp := r
			execution = &cp
		case OARConfirmation:
			if confirmation != nil {
				return oarFinding(out, OARContradicted, "multiple_confirmations_for_execution")
			}
			if r.Result != "confirmed" && r.Result != "not_confirmed" {
				return oarFinding(out, OARContradicted, "invalid_confirmation_result")
			}
			cp := r
			confirmation = &cp
		default:
			return oarFinding(out, OARContradicted, "unknown_phase")
		}
	}
	if root == nil {
		return oarFinding(out, OARIndeterminate, "root_decision_missing")
	}
	out.RootDecisionID = root.ReceiptID

	// All records in one action must preserve the exact action identity.
	for _, r := range records {
		if r.ReceiptID == root.ReceiptID {
			continue
		}
		if !sameOARScope(*root, r) {
			return oarFinding(out, OARContradicted, "action_identity_changed_within_lifecycle")
		}
		parent, ok := byID[r.DecisionRef]
		if r.DecisionRef == "" || !ok {
			return oarFinding(out, OARContradicted, "decision_ref_missing_or_unresolvable")
		}
		if !sameOARScope(parent, r) {
			return oarFinding(out, OARContradicted, "decision_ref_crosses_action_scope")
		}
		switch r.Phase {
		case OARDecision:
			if approval == nil || parent.Phase != OARDecision || parent.Result != "held" {
				return oarFinding(out, OARContradicted, "approval_does_not_follow_held_decision")
			}
		case OARExecution:
			if parent.Phase != OARDecision || (parent.Result != "allowed" && parent.Result != "approved") {
				return oarFinding(out, OARContradicted, "execution_does_not_follow_allowed_or_approved_decision")
			}
		case OARConfirmation:
			if parent.Phase != OARExecution {
				return oarFinding(out, OARContradicted, "confirmation_does_not_follow_execution")
			}
		}
	}

	if approval != nil {
		out.ApprovalID = approval.ReceiptID
		if approval.DecisionRef != root.ReceiptID {
			return oarFinding(out, OARContradicted, "approval_does_not_reference_root_held_decision")
		}
		if root.Result != "held" {
			return oarFinding(out, OARContradicted, "approval_present_without_held_root")
		}
	}
	if root.Result == "denied" {
		if execution != nil || confirmation != nil || approval != nil {
			return oarFinding(out, OARContradicted, "execution_or_approval_follows_denied_root")
		}
		out.Projection = OARRefused
		return out
	}
	if root.Result == "held" && approval == nil {
		if execution != nil || confirmation != nil {
			return oarFinding(out, OARContradicted, "execution_without_approval_of_held_action")
		}
		out.Projection = OARNonterminal
		return out
	}
	if approval != nil && (approval.Result == "rejected" || approval.Result == "expired") {
		if execution != nil || confirmation != nil {
			return oarFinding(out, OARContradicted, "execution_follows_rejected_or_expired_approval")
		}
		out.Projection = OARRefused
		return out
	}
	if root.Result == "held" && (approval == nil || approval.Result != "approved") {
		out.Projection = OARNonterminal
		return out
	}
	if execution == nil {
		out.Projection = OARNonterminal
		return out
	}
	out.ExecutionID = execution.ReceiptID
	if execution.DecisionRef != root.ReceiptID && (approval == nil || execution.DecisionRef != approval.ReceiptID) {
		return oarFinding(out, OARContradicted, "execution_not_bound_to_root_or_approval")
	}
	if execution.Result == "refused" {
		if confirmation != nil {
			return oarFinding(out, OARContradicted, "confirmation_conflicts_with_refused_execution")
		}
		out.Projection = OARRefused
		return out
	}
	if confirmation == nil {
		if execution.Result == "failed" {
			out.Projection = OARFailedExecution
		} else {
			out.Projection = OARExecutedUnconfirmed
		}
		return out
	}
	out.ConfirmationID = confirmation.ReceiptID
	if confirmation.DecisionRef != execution.ReceiptID {
		return oarFinding(out, OARContradicted, "confirmation_not_bound_to_exact_execution")
	}
	if confirmation.Result == "confirmed" {
		out.Projection = OARConfirmed
	} else {
		out.Projection = OARNotConfirmed
	}
	return out
}

func sameOARScope(a, b OARRecord) bool {
	return a.AgentID == b.AgentID && a.RunID == b.RunID && a.ActionType == b.ActionType && a.ActionTarget == b.ActionTarget
}

func oarFinding(r OARReport, p OARProjection, f string) OARReport {
	r.Projection = p
	r.Findings = append(r.Findings, f)
	return r
}
