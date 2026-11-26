// Package oarinterop projects Open Action Record outcomes into the
// three relying-party outcome classes. It is not an OAR bundle verifier.
package oarinterop

type Class string

const (
	Committed  Class = "committed"
	Refused    Class = "refused"
	Unresolved Class = "unresolved"
)

// OutcomeClass maps the OAR outcome vocabulary to its relying-party class.
// Unknown and non-terminal outcomes are unresolved by default. Only the
// read-only confirmation outcome "confirmed" maps to committed.
func OutcomeClass(outcome string) Class {
	switch outcome {
	case "confirmed":
		return Committed
	case "denied", "rejected", "expired", "refused":
		return Refused
	default:
		return Unresolved
	}
}
