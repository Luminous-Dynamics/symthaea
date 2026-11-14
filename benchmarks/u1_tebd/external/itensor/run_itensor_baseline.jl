#!/usr/bin/env julia
# Matched fixed-charge U(1) TEBD baseline using ITensorMPS.
#
# For N<=12, report an independent dense discrete-circuit oracle. Continuous
# dense evolution is restricted to N<=10 to bound the full Hamiltonian memory.

using ITensors
using ITensorMPS
using JSON3
using LinearAlgebra
using Statistics

function term_parameters(term)
    hopping = ComplexF64(Float64(term.hopping.re), Float64(term.hopping.im))
    return (
        bond=Int(term.bond_index),
        mu_left=Float64(term.mu_left),
        mu_right=Float64(term.mu_right),
        interaction=Float64(term.density_interaction),
        hopping=hopping,
    )
end

function local_hamiltonian(term)
    h = zeros(ComplexF64, 4, 4)
    h[2, 2] = term.mu_right
    h[3, 3] = term.mu_left
    h[4, 4] = term.mu_left + term.mu_right + term.interaction
    h[2, 3] = term.hopping
    h[3, 2] = conj(term.hopping)
    return h
end

local_gate(term, tau) = exp(-im * tau * local_hamiltonian(term))

function make_itensor_gate(term, sites, tau)
    left = sites[term.bond]
    right = sites[term.bond + 1]
    id_left = op("Id", left)
    id_right = op("Id", right)
    n_left = 0.5 * id_left - op("Sz", left)
    n_right = 0.5 * id_right - op("Sz", right)
    h = term.mu_left * n_left * id_right +
        term.mu_right * id_left * n_right +
        term.interaction * n_left * n_right +
        term.hopping * op("S+", left) * op("S-", right) +
        conj(term.hopping) * op("S-", left) * op("S+", right)
    return exp(-im * tau * h)
end

function ordered_terms(case)
    terms = Dict{Int,Any}()
    for raw in case.bonds
        term = term_parameters(raw)
        haskey(terms, term.bond) && error("duplicate Hamiltonian bond $(term.bond)")
        terms[term.bond] = term
    end
    return terms
end

function strang_evolve(psi, sites, case, cutoff, maxdim)
    terms = ordered_terms(case)
    gate_count = 0
    for _ in 1:Int(case.steps)
        for (parity, scale) in ((0, 0.5), (1, 1.0), (0, 0.5))
            gates = ITensor[]
            for bond in sort(collect(keys(terms)))
                if bond % 2 == parity
                    push!(gates, make_itensor_gate(terms[bond], sites, Float64(case.time_step) * scale))
                end
            end
            if !isempty(gates)
                # QN-aware apply performs SVD compression with native cutoff/maxdim.
                # No explicit normalize! follows this call.
                psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
                gate_count += length(gates)
            end
        end
    end
    return psi, gate_count
end

function dense_apply_two_site!(state, gate, nsites, bond)
    dim = 1 << nsites
    out = zeros(ComplexF64, dim)
    left_shift = nsites - 1 - bond
    right_shift = nsites - 2 - bond
    for basis in 0:(dim - 1)
        left_bit = (basis >> left_shift) & 1
        right_bit = (basis >> right_shift) & 1
        local_in = 2 * left_bit + right_bit
        base = basis & ~(1 << left_shift) & ~(1 << right_shift)
        amplitude = state[basis + 1]
        for local_out in 0:3
            out_left = (local_out >> 1) & 1
            out_right = local_out & 1
            target = base | (out_left << left_shift) | (out_right << right_shift)
            out[target + 1] += gate[local_out + 1, local_in + 1] * amplitude
        end
    end
    copyto!(state, out)
    return state
end

function dense_discrete_oracle(case, initial)
    n = Int(case.site_count)
    terms = ordered_terms(case)
    state = copy(initial)
    for _ in 1:Int(case.steps)
        for (parity, scale) in ((0, 0.5), (1, 1.0), (0, 0.5))
            for bond in sort(collect(keys(terms)))
                if bond % 2 == parity
                    dense_apply_two_site!(state, local_gate(terms[bond], Float64(case.time_step) * scale), n, bond)
                end
            end
        end
    end
    return state
end

function continuous_oracle(case, initial)
    n = Int(case.site_count)
    dim = 1 << n
    identity = Matrix{ComplexF64}(I, 2, 2)
    full_h = zeros(ComplexF64, dim, dim)
    terms = ordered_terms(case)
    for (bond, term) in terms
        left = Matrix{ComplexF64}(I, 1, 1)
        right = Matrix{ComplexF64}(I, 1, 1)
        for site in 1:n
            if site < bond
                left = kron(left, identity)
            elseif site > bond + 1
                right = kron(right, identity)
            end
        end
        full_h .+= kron(kron(left, local_hamiltonian(term)), right)
    end
    total_time = Float64(case.time_step) * Int(case.steps)
    return exp(-im * total_time * full_h) * initial
end

function dense_from_mps(psi, sites)
    contracted = prod(psi)
    tensor = Array(contracted, sites...)
    # Julia stores the first dimension fastest. Reverse axes to make site 1
    # the most-significant bit, matching the benchmark's shared convention.
    return vec(permutedims(tensor, reverse(1:length(sites))))
end

function state_metrics(actual, reference)
    l2 = norm(actual - reference)
    n_actual = real(dot(actual, actual))
    n_ref = real(dot(reference, reference))
    fidelity = abs2(dot(reference, actual)) / max(n_actual * n_ref, eps(Float64))
    return Dict(
        "state_l2_error" => Float64(l2),
        "fidelity" => Float64(fidelity),
        "actual_norm_squared" => Float64(n_actual),
        "reference_norm_squared" => Float64(n_ref),
    )
end

function charge_metrics(state, target_charge)
    leakage = 0.0
    mean_charge = 0.0
    variance = 0.0
    for (index, amplitude) in enumerate(state)
        charge = count_ones(UInt(index - 1))
        probability = abs2(amplitude)
        mean_charge += charge * probability
        charge != target_charge && (leakage += probability)
    end
    for (index, amplitude) in enumerate(state)
        charge = count_ones(UInt(index - 1))
        variance += (charge - mean_charge)^2 * abs2(amplitude)
    end
    return Dict(
        "total_charge_target" => target_charge,
        "total_charge_expectation" => Float64(mean_charge),
        "total_charge_variance" => Float64(variance),
        "charge_leakage_probability" => Float64(leakage),
    )
end

function initial_mps(case, sites)
    bits = Int.(collect(case.initial_bits))
    length(bits) == length(sites) || error("initial bit count mismatch")
    return MPS(sites, j -> bits[j] == 1 ? "Dn" : "Up")
end

function initial_dense(case)
    n = Int(case.site_count)
    state = zeros(ComplexF64, 1 << n)
    index = 0
    for bit in Int.(collect(case.initial_bits))
        index = (index << 1) | bit
    end
    state[index + 1] = 1.0
    return state
end

function run_once(case, cutoff, maxdim)
    n = Int(case.site_count)
    sites = siteinds("S=1/2", n; conserve_qns=true)
    psi = initial_mps(case, sites)
    norm_in = Float64(real(inner(psi, psi)))
    gate_count = 0
    elapsed = @elapsed begin
        psi, gate_count = strang_evolve(psi, sites, case, cutoff, maxdim)
    end
    norm_out = Float64(real(inner(psi, psi)))
    max_bond = Int(maxlinkdim(psi))
    charge_expectation = Float64(sum(0.5 .- expect(psi, "Sz")))

    dense_metrics = nothing
    continuous_metrics = nothing
    charge = nothing
    if n <= 12 && Bool(case.oracle.discrete_dense)
        actual = dense_from_mps(psi, sites)
        initial = initial_dense(case)
        discrete = dense_discrete_oracle(case, initial)
        dense_metrics = state_metrics(actual, discrete)
        charge = charge_metrics(actual, Int(case.total_charge))
        if n <= 10 && Bool(case.oracle.continuous_dense)
            continuous_metrics = state_metrics(actual, continuous_oracle(case, initial))
        end
    else
        gate_count = 3 * Int(case.steps) * length(case.bonds)
    end

    return Dict(
        "implementation" => "ITensorMPS",
        "itensormps_version" => "0.4.1",
        "itensors_version" => string(pkgversion(ITensors)),
        "julia_version" => string(VERSION),
        "case_id" => String(case.id),
        "site_count" => n,
        "total_charge" => Int(case.total_charge),
        "time_step" => Float64(case.time_step),
        "steps" => Int(case.steps),
        "strang_order" => ["even_dt_over_2", "odd_dt", "even_dt_over_2"],
        "gate_count" => gate_count,
        "cutoff" => cutoff,
        "cutoff_semantics" => "ITensorMPS native SVD truncation-error threshold",
        "maxdim" => maxdim,
        "renormalization" => "disabled_no_normalize_call",
        "runtime_seconds" => elapsed,
        "gate_throughput_per_second" => elapsed > 0 ? gate_count / elapsed : nothing,
        "input_norm_squared" => norm_in,
        "output_norm_squared" => norm_out,
        "norm_drift" => abs(norm_out - norm_in) / max(abs(norm_in), eps(Float64)),
        "observed_max_bond" => max_bond,
        "structural_u1_conservation" => true,
        "charge_expectation_from_sz" => charge_expectation,
        "dense_discrete_oracle" => dense_metrics,
        "continuous_time_oracle" => continuous_metrics,
        "charge_metrics" => charge,
    )
end

function main()
    fixture_path = joinpath(@__DIR__, "..", "..", "fixtures", "u1_tebd_cases_v1.json")
    case_id = "u1-two-site-hopping-known-answer"
    track = "fixed_bond"
    cutoff = nothing
    maxdim = nothing
    repeats = 20
    warmups = 5
    output_path = nothing

    for arg in ARGS
        if startswith(arg, "--fixture=")
            fixture_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--case=")
            case_id = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--track=")
            track = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--cutoff=")
            cutoff = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--maxdim=")
            maxdim = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--repeats=")
            repeats = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--warmups=")
            warmups = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output=")
            output_path = split(arg, "=", limit=2)[2]
        else
            error("unknown argument: $arg")
        end
    end

    fixture = JSON3.read(read(fixture_path, String))
    cases = filter(item -> String(item.id) == case_id, fixture.cases)
    length(cases) == 1 || error("fixture case not found: $case_id")
    case = only(cases)

    if track == "fixed_bond"
        cutoff = cutoff === nothing ? 0.0 : cutoff
        maxdim = maxdim === nothing ? Int(case.max_bond_dimension) : maxdim
    elseif track == "matched_error"
        cutoff = cutoff === nothing ? 1e-8 : cutoff
        maxdim = maxdim === nothing ? 256 : maxdim
    else
        error("track must be fixed_bond or matched_error")
    end

    for _ in 1:max(0, warmups)
        run_once(case, cutoff, maxdim)
    end
    records = [run_once(case, cutoff, maxdim) for _ in 1:max(1, repeats)]
    times = [record["runtime_seconds"] for record in records]
    summary = Dict(
        "schema" => "luminous.u1_tebd_benchmark_result.v1",
        "fixture_schema" => String(fixture.schema),
        "track" => track,
        "case_id" => case_id,
        "repeats" => length(records),
        "warmups" => warmups,
        "runtime_median_seconds" => median(times),
        "runtime_p95_seconds" => quantile(times, 0.95),
        "host" => Dict(
            "julia" => string(VERSION),
            "kernel" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => string(Sys.CPU_NAME),
            "threads" => Threads.nthreads(),
        ),
        "records" => records,
    )
    output = JSON3.write(summary)
    if output_path !== nothing
        mkpath(dirname(output_path))
        write(output_path, output * "\n")
    end
    println(output)
end

main()
