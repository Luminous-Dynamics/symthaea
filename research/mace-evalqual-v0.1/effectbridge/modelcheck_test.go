package effectbridge

import (
	"fmt"
	"testing"
)

type efxModelTerminal uint8

const (
	efxModelEmpty efxModelTerminal = iota
	efxModelEffect
	efxModelFence
)

type efxModelTx uint8

const (
	efxModelNoTx efxModelTx = iota
	efxModelEffectTx
	efxModelFenceTx
)

const (
	efxModelIntentA uint8 = 1 << iota
	efxModelIntentB
)

type efxProtocolState struct {
	terminal        efxModelTerminal
	terminalIntent  uint8
	receiptVisible  bool
	businessVisible bool
	pending         uint8
	tx              efxModelTx
	txIntent        uint8
	txReceipt       bool
	txBusiness      bool
}

type efxProtocolEdge struct {
	action string
	state  efxProtocolState
}

func efxProtocolInvariant(s efxProtocolState) error {
	switch s.terminal {
	case efxModelEmpty:
		if s.terminalIntent != 0 || s.receiptVisible || s.businessVisible {
			return fmt.Errorf("empty destination exposes terminal state: %+v", s)
		}
	case efxModelEffect:
		if s.terminalIntent == 0 || !s.receiptVisible || !s.businessVisible {
			return fmt.Errorf("effect is not atomically bound to receipt and business row: %+v", s)
		}
	case efxModelFence:
		if s.terminalIntent == 0 || !s.receiptVisible || s.businessVisible {
			return fmt.Errorf("fence must expose receipt without business effect: %+v", s)
		}
	default:
		return fmt.Errorf("invalid terminal state: %+v", s)
	}

	if s.tx == efxModelNoTx {
		if s.txIntent != 0 || s.txReceipt || s.txBusiness {
			return fmt.Errorf("transaction residue without an active transaction: %+v", s)
		}
	} else {
		if s.terminal != efxModelEmpty || s.txIntent == 0 {
			return fmt.Errorf("transaction overlaps committed state or lacks intent: %+v", s)
		}
		if s.tx == efxModelFenceTx && s.txBusiness {
			return fmt.Errorf("fence transaction staged a business effect: %+v", s)
		}
		if s.tx != efxModelEffectTx && s.tx != efxModelFenceTx {
			return fmt.Errorf("invalid transaction kind: %+v", s)
		}
	}
	if s.pending & ^(efxModelIntentA|efxModelIntentB) != 0 {
		return fmt.Errorf("unknown pending-intent bits: %+v", s)
	}
	return nil
}

func efxProtocolReconcile(s efxProtocolState) string {
	switch s.terminal {
	case efxModelEmpty:
		return "not_found_unfenced" // indeterminate, never proof of non-execution
	case efxModelEffect:
		return "found"
	case efxModelFence:
		return "not_executed_fenced"
	default:
		return "invalid"
	}
}

func efxProtocolSuccessors(s efxProtocolState) []efxProtocolEdge {
	var next []efxProtocolEdge
	add := func(name string, candidate efxProtocolState) {
		if candidate != s {
			next = append(next, efxProtocolEdge{action: name, state: candidate})
		}
	}

	// Requests can be sent before a destination decision and can remain delayed
	// across a service crash/restart. A bit represents one pending intent class.
	for _, intent := range []uint8{efxModelIntentA, efxModelIntentB} {
		if s.pending&intent == 0 {
			candidate := s
			candidate.pending |= intent
			add(fmt.Sprintf("send(%d)", intent), candidate)
		}

		// Delivery serializes with an active destination transaction. Once a
		// terminal decision exists, a late packet is consumed/rejected and cannot
		// replace that decision.
		if s.pending&intent != 0 && s.tx == efxModelNoTx {
			candidate := s
			candidate.pending &^= intent
			if candidate.terminal == efxModelEmpty {
				candidate.tx = efxModelEffectTx
				candidate.txIntent = intent
			}
			add(fmt.Sprintf("deliver(%d)", intent), candidate)
		}

		// A fence obtains the same serialized destination transaction boundary
		// as effect delivery. A different intent on an occupied key is a conflict.
		if s.tx == efxModelNoTx && s.terminal == efxModelEmpty {
			candidate := s
			candidate.tx = efxModelFenceTx
			candidate.txIntent = intent
			add(fmt.Sprintf("begin_fence(%d)", intent), candidate)
		}
	}

	if s.tx != efxModelNoTx {
		if !s.txReceipt {
			candidate := s
			candidate.txReceipt = true
			add("stage_receipt", candidate)
		}
		if s.tx == efxModelEffectTx && !s.txBusiness {
			candidate := s
			candidate.txBusiness = true
			add("stage_business_row", candidate)
		}
		if s.tx == efxModelEffectTx && s.txReceipt && s.txBusiness {
			candidate := s
			candidate.terminal = efxModelEffect
			candidate.terminalIntent = s.txIntent
			candidate.receiptVisible = true
			candidate.businessVisible = true
			candidate.tx = efxModelNoTx
			candidate.txIntent = 0
			candidate.txReceipt = false
			candidate.txBusiness = false
			add("commit_effect_atomically", candidate)
		}
		if s.tx == efxModelFenceTx && s.txReceipt {
			candidate := s
			candidate.terminal = efxModelFence
			candidate.terminalIntent = s.txIntent
			candidate.receiptVisible = true
			candidate.businessVisible = false
			candidate.tx = efxModelNoTx
			candidate.txIntent = 0
			candidate.txReceipt = false
			candidate.txBusiness = false
			add("commit_fence", candidate)
		}

		// A failed write or process crash rolls back all staged transaction data.
		candidate := s
		candidate.tx = efxModelNoTx
		candidate.txIntent = 0
		candidate.txReceipt = false
		candidate.txBusiness = false
		add("rollback_or_crash_before_commit", candidate)
	}
	return next
}

func TestEffectFenceFiniteStateModelHasNoSafetyCounterexample(t *testing.T) {
	initial := efxProtocolState{}
	if err := efxProtocolInvariant(initial); err != nil {
		t.Fatal(err)
	}

	queue := []efxProtocolState{initial}
	depth := map[efxProtocolState]int{initial: 0}
	parents := make(map[efxProtocolState]efxProtocolState)
	actions := make(map[efxProtocolState]string)
	transitions := 0
	maxDepth := 0
	for len(queue) > 0 {
		state := queue[0]
		queue = queue[1:]
		if err := efxProtocolInvariant(state); err != nil {
			trace := []string{}
			cursor := state
			for cursor != initial {
				trace = append(trace, actions[cursor])
				cursor = parents[cursor]
			}
			for left, right := 0, len(trace)-1; left < right; left, right = left+1, right-1 {
				trace[left], trace[right] = trace[right], trace[left]
			}
			t.Fatalf("safety invariant failed: %v; trace=%v", err, trace)
		}

		if got := efxProtocolReconcile(state); got == "not_found_unfenced" && state.terminal != efxModelEmpty {
			t.Fatalf("reconciliation reported unfenced absence after a terminal commit: %+v", state)
		}
		if state.terminal == efxModelFence && state.businessVisible {
			t.Fatalf("fenced operation exposed a business mutation: %+v", state)
		}
		if state.terminal == efxModelEffect && (!state.receiptVisible || !state.businessVisible) {
			t.Fatalf("effect is visible without its atomic business and receipt rows: %+v", state)
		}

		for _, edge := range efxProtocolSuccessors(state) {
			transitions++
			if err := efxProtocolInvariant(edge.state); err != nil {
				t.Fatalf("transition %s violated invariant: %v", edge.action, err)
			}
			if _, seen := depth[edge.state]; seen {
				continue
			}
			depth[edge.state] = depth[state] + 1
			if depth[edge.state] > maxDepth {
				maxDepth = depth[edge.state]
			}
			parents[edge.state] = state
			actions[edge.state] = edge.action
			queue = append(queue, edge.state)
		}
	}

	var sawEffect, sawFence, sawLateAfterFence, sawRolledBackPending bool
	for state := range depth {
		if state.terminal == efxModelEffect {
			sawEffect = true
		}
		if state.terminal == efxModelFence {
			sawFence = true
			if state.pending != 0 {
				sawLateAfterFence = true
			}
		}
		if state.terminal == efxModelEmpty && state.tx == efxModelNoTx && state.pending != 0 {
			sawRolledBackPending = true
		}
	}
	if !sawEffect || !sawFence || !sawLateAfterFence || !sawRolledBackPending {
		t.Fatalf("state exploration missed required cases: effect=%v fence=%v late=%v rollback=%v", sawEffect, sawFence, sawLateAfterFence, sawRolledBackPending)
	}
	t.Logf("explored %d unique states, %d transitions, max shortest trace depth %d", len(depth), transitions, maxDepth)
}
