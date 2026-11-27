# Two-quadrature oscillatory unstable-mode oracle v0.1

This manufactured reference represents a complex mode with real state (x=[\Re a,\Im a]^T):

`x[k+1] = exp(gamma*dt) * R(2*pi*f*dt) * x[k] + B*u[k]`.

A diagnostic measures one real projection (c^T x_k) with Gaussian noise and arrives after a declared integer delay. A command injects a two-component actuator vector at its declared sparse update steps and delivery delay.

The finite-horizon observability Fisher matrix is
`Wo = sum_k (A^k)^T c*c^T A^k / sigma^2`
over samples available by the horizon. The finite-horizon controllability Gramian is
`Wc = sum_j A^q B B^T (A^q)^T`
for each delivered command, where q is the number of remaining evolution steps after application.

Qualification uses the **minimum eigenvalue** of each 2x2 matrix. This detects when only one quadrature is observable or influenceable. Trace / maximum eigenvalue alone could look healthy while one phase direction remains effectively absent.

The known-answer fixture has five available sensor samples and five delivered commands:
- (W_o=[275.1525629919475,0.6727614529200991;0.6727614529200991,278.6730137656183]), minimum eigenvalue 275.0283782893929.
- (W_c=[0.7374311161436637,-0.004965766841578345;-0.004965766841578345,0.7360497761471079]), minimum eigenvalue 0.7317268780176508.
- Effective diagnostic and actuator bandwidths: 25 Hz and 20 Hz.
- Loop latency 65 ms; growth e-fold fraction 0.013.

A separate aliasing negative control sets the oscillation to 50 Hz while sensor and actuator schedules repeat every 20 ms. The mode phase advances by a complete cycle between samples/commands; both Gramian minimum eigenvalues collapse toward zero even though sample/command counts remain nonzero. This is precisely why mode-phase diversity matters.

This is a one-mode linear analytic reference. It is not a Vlasov solver, multi-mode stability certificate, or closed-loop controller result.