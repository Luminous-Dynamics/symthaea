import unittest

from two_phase_source_escrow_check import (
    build_trace,
    negative_controls,
    verify_event_log,
)


class TwoPhaseSourceEscrowTests(unittest.TestCase):
    def test_independent_replay_on_small_trace(self):
        events, activation = build_trace(11)
        self.assertTrue(verify_event_log(events, activation))

    def test_assignment_is_strictly_after_reservation_and_precedes_credit(self):
        events, activation = build_trace(12)
        reserved = {}
        assigned = {}
        for event in events:
            if event["kind"] == "UnassignedReserved":
                reserved[tuple(event["token_id"])] = event["round"]
            elif event["kind"] == "Assigned":
                token_id = tuple(event["token_id"])
                assignment_id = tuple(event["assignment_id"])
                self.assertGreater(event["round"], reserved[token_id])
                assigned[assignment_id] = (
                    event["round"], event["destination"]
                )
            elif event["kind"] == "Credited":
                assignment_id = tuple(event["assignment_id"])
                assignment_round, destination = assigned[assignment_id]
                self.assertGreaterEqual(event["round"], assignment_round)
                self.assertGreaterEqual(event["round"], activation[destination])

    def test_negative_controls_reject_mutations(self):
        events, activation = build_trace(13)
        controls = negative_controls(events, activation)
        self.assertTrue(controls)
        self.assertTrue(all(controls.values()))


if __name__ == "__main__":
    unittest.main()
