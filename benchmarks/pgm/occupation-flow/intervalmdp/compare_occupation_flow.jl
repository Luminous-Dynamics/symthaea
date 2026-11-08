using IntervalMDP

# The exact reference uses arbitrary-precision rationals end to end.
const Q = Rational{BigInt}
q(n::Integer, d::Integer = 1) = BigInt(n) // BigInt(d)

# IntervalMDP stores target states in rows and source-state ambiguity sets
# in columns. The clock-expanded state layout is:
#   t=0: states 1,2
#   t=1: states 3,4
#   t=2: states 5,6
#   t=3: terminal states 7,8
const N_STATES = 8
lower = fill(q(0), N_STATES, N_STATES)
upper = fill(q(0), N_STATES, N_STATES)

function set_interval!(source, destination, lo, hi)
    lower[destination, source] = q(lo[1], lo[2])
    upper[destination, source] = q(hi[1], hi[2])
end

# Stage 0.
set_interval!(1, 3, (2, 5), (4, 5))
set_interval!(1, 4, (1, 5), (3, 5))
set_interval!(2, 3, (3, 10), (7, 10))
set_interval!(2, 4, (3, 10), (7, 10))

# Stage 1.
set_interval!(3, 5, (1, 2), (9, 10))
set_interval!(3, 6, (1, 10), (1, 2))
set_interval!(4, 5, (1, 5), (3, 5))
set_interval!(4, 6, (2, 5), (4, 5))

# Stage 2: deterministic transition into the zero-reward terminal layer.
set_interval!(5, 7, (1, 1), (1, 1))
set_interval!(6, 8, (1, 1), (1, 1))

# Terminal states are absorbing.
set_interval!(7, 7, (1, 1), (1, 1))
set_interval!(8, 8, (1, 1), (1, 1))

ambiguity = IntervalAmbiguitySets(; lower, upper)
model = IntervalMarkovChain(ambiguity, [1])

# Rewards at t=0, t=1, t=2, t=3 respectively. The terminal reward is zero.
const ACTION_A = Q[q(0), q(0), q(0), q(1), q(0), q(1), q(0), q(0)]
const ACTION_B = Q[q(0), q(0), q(1), q(0), q(0), q(0), q(0), q(0)]

# Regret is optimized on one shared transition process:
#   regret(A) = U_B - U_A
#   regret(B) = U_A - U_B
# IntervalMDP supports signed finite-horizon rewards, so no artificial
# path-dependent or arbitrary reward shift is needed.
const REGRET_A = ACTION_B .- ACTION_A
const REGRET_B = ACTION_A .- ACTION_B

function solve_reward(scores::Vector{Q}, mode, bellman_algorithm)
    property = FiniteTimeReward(scores, q(1), 3)
    specification = Specification(property, mode)
    problem = VerificationProblem(model, specification)
    solution = solve(problem, RobustValueIteration(bellman_algorithm))
    return (
        value = value_function(solution)[1],
        iterations = num_iterations(solution),
        residual = residual(solution)[1],
    )
end

# Pessimistic = minimum over admissible transition rows.
# Optimistic = maximum over admissible transition rows.
const QUERIES = [
    ("action_a_min", ACTION_A, Pessimistic, q(9, 25)),
    ("action_a_max", ACTION_A, Optimistic, q(32, 25)),
    ("action_b_min", ACTION_B, Pessimistic, q(2, 5)),
    ("action_b_max", ACTION_B, Optimistic, q(4, 5)),
    ("regret_a_max", REGRET_A, Optimistic, q(11, 25)),
    ("regret_b_max", REGRET_B, Optimistic, q(22, 25)),
]

# Compare two independent IntervalMDP Bellman algorithms:
# - OMaximization: specialised exact interval-row optimiser.
# - VertexEnumeration: explicit exact local-polytope vertex reference.
const ALGORITHMS = [
    ("omax", OMaximization()),
    ("vertex_enumeration", VertexEnumeration()),
]

println("IntervalMDP version: ", Base.pkgversion(IntervalMDP))
println("Arithmetic: Rational{BigInt}")
println("States: ", N_STATES, "; finite reward horizon: 3")
println()

for (algorithm_name, algorithm) in ALGORITHMS
    println("algorithm=", algorithm_name)
    for (query_name, scores, mode, expected) in QUERIES
        result = solve_reward(scores, mode, algorithm)
        @assert result.value == expected (
            "$algorithm_name / $query_name: got $(result.value), expected $expected"
        )
        println(
            query_name,
            " value=", result.value,
            " iterations=", result.iterations,
            " residual=", result.residual,
        )
    end
end

println()
println("PASS: exact rational values match all six preregistered endpoints.")
