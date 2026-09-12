#!/usr/bin/env python3
"""Independent direct-staple parity check against the LQCD-017D finite-difference flow oracle."""

import importlib.util
from pathlib import Path

REFERENCE = Path(__file__).with_name("lqcd-topology-flow-oracle.py")
spec = importlib.util.spec_from_file_location("lqcd_017d", REFERENCE)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def link_staple(field, dims, site, mu):
    staple = ref.zero3()
    x_plus_mu = ref.shift(site, mu, 1, dims)
    for nu in range(4):
        if nu == mu:
            continue
        x_plus_nu = ref.shift(site, nu, 1, dims)
        forward = ref.mul(
            ref.mul(
                ref.link(field, dims, x_plus_mu, nu),
                ref.dagger(ref.link(field, dims, x_plus_nu, mu)),
            ),
            ref.dagger(ref.link(field, dims, site, nu)),
        )
        staple = ref.add(staple, forward)

        x_minus_nu = ref.shift(site, nu, -1, dims)
        x_minus_nu_plus_mu = ref.shift(x_minus_nu, mu, 1, dims)
        backward = ref.mul(
            ref.mul(
                ref.dagger(ref.link(field, dims, x_minus_nu_plus_mu, nu)),
                ref.dagger(ref.link(field, dims, x_minus_nu, mu)),
            ),
            ref.link(field, dims, x_minus_nu, nu),
        )
        staple = ref.add(staple, backward)
    return staple


def analytic_gradient(field, dims):
    result = {}
    for site in ref.sites(dims):
        for mu in range(4):
            x = ref.mul(ref.link(field, dims, site, mu), link_staple(field, dims, site, mu))
            result[(site, mu)] = [
                ref.trace(ref.mul(generator, x)).imag / 3.0
                for generator in ref.GELL_MANN
            ]
    return result


def apply_gradient(field, dims, gradient, dt):
    updated = ref.field_copy(field)
    for (site, mu), components in gradient.items():
        hermitian = ref.zero3()
        for coefficient, generator in zip(components, ref.GELL_MANN):
            hermitian = ref.add(hermitian, ref.scale(coefficient, generator))
        rotation = ref.matrix_exp(ref.scale(-1.0j * dt, hermitian))
        ref.set_link(updated, dims, site, mu, ref.mul(rotation, ref.link(field, dims, site, mu)))
    return updated


def self_test():
    field = ref.deterministic_fixture(ref.DIMS)
    finite = ref.finite_difference_gradient(field, ref.DIMS, ref.GRAD_EPS)
    analytic = analytic_gradient(field, ref.DIMS)
    max_gradient_error = max(
        abs(a - b)
        for key in finite
        for a, b in zip(finite[key], analytic[key])
    )
    if max_gradient_error > 1.1e-9:
        raise AssertionError(("gradient parity", max_gradient_error))

    reference_flow = ref.finite_difference_wilson_flow_step(
        field, ref.DIMS, ref.FLOW_DT, ref.GRAD_EPS
    )
    analytic_flow = apply_gradient(field, ref.DIMS, analytic, ref.FLOW_DT)
    max_link_error = ref.max_field_error(reference_flow, analytic_flow)
    if max_link_error > 1.2e-12:
        raise AssertionError(("flow parity", max_link_error))

    analytic_action = ref.wilson_action(analytic_flow, ref.DIMS)
    reference_action = ref.wilson_action(reference_flow, ref.DIMS)
    analytic_q = ref.clover_topological_charge(analytic_flow, ref.DIMS)
    reference_q = ref.clover_topological_charge(reference_flow, ref.DIMS)

    print("ok")
    print(f"max_gradient_abs_error={max_gradient_error:.17g}")
    print(f"max_flow_link_abs_error={max_link_error:.17g}")
    print(f"analytic_action={analytic_action:.17g}")
    print(f"reference_action={reference_action:.17g}")
    print(f"analytic_q={analytic_q:.17g}")
    print(f"reference_q={reference_q:.17g}")


if __name__ == "__main__":
    self_test()
