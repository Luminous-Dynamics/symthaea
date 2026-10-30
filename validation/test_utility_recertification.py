import itertools, unittest
import numpy as np
from scipy.optimize import linprog
from utility_recertification import (
    capacity_utilities, charges, distribution, gcm, rc_tails,
    sample_count, solve_utility_recertification, verify_all_subsets
)


def exhaustive_lp(e,u,alpha):
    order=np.argsort(-e,kind="stable");s=np.asarray(e)[order];v=np.asarray(u)[order];k=len(s)
    coeff=np.empty(k);coeff[0]=v[0]
    for j in range(1,k):
        r=j+1;coeff[j]=r*v[j]-(r-2)*v[j-1]
    A=[];b=[]
    for j in range(k-1):
        row=np.zeros(k);row[j+1]=1;row[j]=-1;A.append(row);b.append(0.)
    for j in range(k):
        row=np.zeros(k);row[j]=-1
        if j+1<k:row[j+1]=2
        if j+2<k:row[j+2]=-1
        A.append(row);b.append(0.)
    for t in range(1,k+1):
        for T in itertools.combinations(range(k),t):
            row=np.zeros(k)
            for j in T:
                row[j]+=1
                if j+1<k:row[j+1]-=1
            A.append(row);b.append(alpha*np.mean(s[list(T)]))
    r=linprog(-coeff,A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=[(0,None)]*k,method="highs",
        options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
    assert r.success,r.message
    S=np.r_[r.x,0.];rank=np.arange(1,k+1,dtype=float)
    q=rank*S[:-1]-(rank-1)*S[1:]
    return float(np.dot(v,q))


class UtilityRecertificationTests(unittest.TestCase):
    def test_uniform_utility_recovers_RC(self):
        rng=np.random.default_rng(1)
        for k in range(2,21):
            for _ in range(10):
                e=np.exp(rng.normal(0,1.7,k))
                got=solve_utility_recertification(e,np.ones(k),.1)
                np.testing.assert_allclose(got["tails"],rc_tails(e,.1),atol=1e-9,rtol=1e-9)

    def test_capacity_K_recovers_RC(self):
        rng=np.random.default_rng(2)
        for k in range(2,15):
            e=np.exp(rng.normal(0,1.5,k))
            got=solve_utility_recertification(e,capacity_utilities(e,k),.1)
            np.testing.assert_allclose(got["tails"],rc_tails(e,.1),atol=1e-8)

    def test_all_subset_constraints_randomized(self):
        rng=np.random.default_rng(3)
        for k in range(2,11):
            for _ in range(40):
                e=np.exp(rng.normal(0,1.5,k));u=rng.lognormal(0,1.2,k)
                got=solve_utility_recertification(e,u,.1)
                check=verify_all_subsets(e,.1,got["tails"])
                self.assertTrue(check["valid"],check)
                self.assertLessEqual(got["max_subset_violation"],1e-8)

    def test_cutting_plane_matches_full_exhaustive_LP(self):
        rng=np.random.default_rng(4)
        for k in range(2,9):
            for _ in range(8):
                e=np.exp(rng.normal(0,1.2,k));u=rng.lognormal(0,.8,k)
                got=solve_utility_recertification(e,u,.1)
                full=exhaustive_lp(e,u,.1)
                self.assertAlmostEqual(got["objective_value"],full,places=7)

    def test_non_suffix_cut_added(self):
        e=np.array([.27616548,.49790624,.05057260,9.36088684])
        u=np.array([.39201160,60.18331104,.18420038,23.76504101])
        got=solve_utility_recertification(e,u,.1)
        self.assertTrue(verify_all_subsets(e,.1,got["tails"])["valid"])
        self.assertGreater(got["number_of_cuts"],len(e))

    def test_distribution_valid(self):
        e=np.array([10.,4.,1.,.2]);u=np.array([4.,3.,2.,1.])
        got=solve_utility_recertification(e,u,.1)
        p=distribution(got["tails"])
        self.assertAlmostEqual(float(p.sum()),1.,places=8)
        self.assertTrue(np.all(p>=-1e-8))
        self.assertAlmostEqual(float(charges(got["tails"]).sum()),
                               float(got["cumulative_charges"][0]),places=8)

    def test_capacity_weights_top_rank(self):
        e=np.array([1.,100.,3.,2.])
        u=capacity_utilities(e,2)
        np.testing.assert_array_equal(u,np.array([0.,1.,1.,0.]))
        got=solve_utility_recertification(e,u,.1)
        self.assertAlmostEqual(float(got["sorted_utilities"][:2].sum()),2.)

    def test_shared_uniform_count_sampling(self):
        q=np.array([.8,.6,.2])
        self.assertEqual(sample_count(q,.1),3)
        self.assertEqual(sample_count(q,.5),2)
        self.assertEqual(sample_count(q,.9),0)

    def test_gcm_convex_minorant(self):
        rng=np.random.default_rng(5)
        for n in range(2,30):
            for _ in range(20):
                y=rng.random(n);g=gcm(y)
                self.assertTrue(np.all(g<=y+1e-10))
                self.assertTrue(np.all(np.diff(g,n=2)>=-1e-10))


if __name__=="__main__": unittest.main(verbosity=2)
