import unittest
from fractions import Fraction as Q
import audit_power as h

class ExtendedAuditPower(unittest.TestCase):
    def test_extreme_alarm_counts_move_posterior_in_expected_directions(self):
        sens,fpr=h.AUDITS['high']; prior=Q(1,2)
        self.assertGreater(h.posterior(2,2,sens,fpr,prior),Q(95,100))
        self.assertLess(h.posterior(2,0,sens,fpr,prior),Q(5,100))

if __name__=='__main__': unittest.main(verbosity=2)
